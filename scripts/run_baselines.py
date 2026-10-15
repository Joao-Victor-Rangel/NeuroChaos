"""Baselines FC/anatomia → SC no MICA-MICs, validação cruzada por sujeito (plano v2 §3).

  python scripts/run_baselines.py --parc schaefer200 aparc

Modelos: B0 média do grupo; B1 distância geodésica; B2 linear multimodal (com e sem FC);
B3 ridge entre sujeitos sobre a FC; controle: B2 com a FC de OUTRO sujeito.
Saídas em results/baselines/<parc>/.
"""

import argparse
import csv
import json

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import binomtest, wilcoxon

from neurochaos import mica, sc_metrics, sc_models
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable

PRIMARY = ["pearson", "pearson_desvio", "auc_top20", "pearson_forca_nodal"]


def load_all(parc):
    data = {}
    for s, d in mica.load_matrices(parc).items():
        data[s] = {"y": sc_models.target(d["sc"]), "fc": sc_models.upper(d["fc"]),
                   "gd": sc_models.upper(d["gd"]), "mpc": sc_models.upper(d["mpc"]),
                   "inter": sc_models.interhemispheric(d["labels"]), "labels": d["labels"]}
    return data


read_folds = mica.read_folds


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty_like(adj)
    out[o] = adj
    return out


def boot_ci(x, rng, n=5000):
    x = np.asarray(x, float)
    bs = rng.choice(x, size=(n, len(x))).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run(parc, rng):
    data = load_all(parc)
    folds = read_folds()
    subs = sorted(data)
    n_nodes = len(data[subs[0]]["labels"])
    makers = [sc_models.GroupMean, sc_models.Distance, lambda: sc_models.LinearMultimodal(False),
              sc_models.LinearMultimodal, sc_models.KernelRidgeFC]
    names = [m().name for m in makers] + ["CTRL_B2_FC_trocada"]
    preds = {n: {} for n in names}
    mus = {}
    lam = []
    for k in sorted(set(folds.values())):
        test = [s for s in subs if folds[s] == k]
        train = [data[s] for s in subs if folds[s] != k]
        mu = np.mean([t["y"] for t in train], axis=0)
        perm = rng.permutation(len(test))
        while np.any(perm == np.arange(len(test))):  # derangement: ninguém recebe a própria FC
            perm = rng.permutation(len(test))
        for mk in makers:
            m = mk().fit(train)
            if hasattr(m, "lam"):
                lam.append(m.lam)
            for s in test:
                preds[m.name][s] = m.predict(data[s])
            if m.name == "B2_linear_multimodal":
                for i, s in enumerate(test):
                    swapped = dict(data[s], fc=data[test[perm[i]]]["fc"])
                    preds["CTRL_B2_FC_trocada"][s] = m.predict(swapped)
        for s in test:
            mus[s] = mu

    rows = []
    for n in names:
        for s in subs:
            r = {"model": n, "subject": s, "fold": folds[s]}
            r.update(sc_metrics.edge_metrics(preds[n][s], data[s]["y"], mus[s], n_nodes))
            rows.append(r)
    ident = {n: sc_metrics.identification(preds[n], {s: data[s]["y"] for s in subs}, mus) for n in names}

    summary = []
    for n in names:
        rr = [r for r in rows if r["model"] == n]
        rec = {"model": n}
        for mt in PRIMARY + ["spearman", "mse"]:
            v = [r[mt] for r in rr]
            rec[mt] = float(np.nanmean(v))
            if mt in PRIMARY and np.all(np.isfinite(v)):
                rec[f"{mt}_ci"] = boot_ci(v, rng)
        rec["ident_accuracy"] = ident[n]["accuracy"]
        rec["ident_chance"] = ident[n]["chance"]
        acc = ident[n]["accuracy"]
        rec["ident_p_binomial"] = (float(binomtest(int(round(acc * len(subs))), len(subs), 1 / len(subs),
                                                   alternative="greater").pvalue) if np.isfinite(acc) else None)
        rec["diff_identifiability"] = ident[n]["differential_identifiability"]
        summary.append(rec)

    # Comparações pareadas por sujeito (Wilcoxon) contra B0 e de B2 contra o controle; Holm.
    tests = []
    by = {(r["model"], r["subject"]): r for r in rows}
    pairs = [(n, "B0_media_grupo") for n in names if n != "B0_media_grupo"]
    pairs += [("B2_linear_multimodal", "CTRL_B2_FC_trocada"), ("B2_linear_multimodal", "B2_sem_FC")]
    for a_, b_ in pairs:
        for mt in PRIMARY:
            x = np.array([by[(a_, s)][mt] for s in subs])
            y = np.array([by[(b_, s)][mt] for s in subs])
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 5 or np.allclose(x[ok], y[ok]):
                continue
            p = wilcoxon(x[ok], y[ok]).pvalue
            tests.append({"a": a_, "b": b_, "metric": mt, "mean_diff": float(np.mean(x[ok] - y[ok])), "p": float(p)})
    if tests:
        for t, ph in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = float(ph)

    out = resolve("results/baselines") / parc
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(to_jsonable({
        "parcellation": parc, "n_nodes": n_nodes, "n_subjects": len(subs), "summary": summary,
        "tests": tests, "ridge_lambdas_chosen": lam,
        "target": "log1p(SC/mean(SC)) no triângulo superior"}), indent=1), encoding="utf-8")
    np.savez_compressed(out / "identification.npz", **{n: ident[n]["matrix"] for n in names},
                        subjects=np.array(subs))
    return summary, tests, ident, out


def figure(parc, summary, ident, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [s["model"] for s in summary]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, mt, title in zip(axes[:3], ["pearson", "pearson_desvio", "auc_top20"],
                             ["r aresta (SC total)", "r do desvio individual", "AUC arestas fortes (top 20%)"]):
        vals = [s[mt] for s in summary]
        cis = [s.get(f"{mt}_ci", (v, v)) for s, v in zip(summary, vals)]
        err = np.array([[v - c[0], c[1] - v] for v, c in zip(vals, cis)]).T
        ax.barh(range(len(names)), vals, xerr=err, color="#4C72B0")
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [""] * len(names), fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.invert_yaxis()
    best = max((s for s in summary if s["model"] != "CTRL_B2_FC_trocada"),
               key=lambda s: s["ident_accuracy"] if np.isfinite(s["ident_accuracy"]) else -1)["model"]
    im = axes[3].imshow(ident[best]["matrix"], cmap="magma")
    axes[3].set_title(f"Identificação ({best}): acc={ident[best]['accuracy']:.2f} (acaso {ident[best]['chance']:.2f})",
                      fontsize=8)
    axes[3].set_xlabel("SC real (sujeito)")
    axes[3].set_ylabel("SC prevista (sujeito)")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    fig.suptitle(f"Baselines FC/anatomia → SC — MICA-MICs, {parc}, 5 folds por sujeito (n=50)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "baselines.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", nargs="+", default=["schaefer200", "aparc"])
    a = ap.parse_args()
    rng = np.random.default_rng(20261014)
    for parc in a.parc:
        summary, tests, ident, out = run(parc, rng)
        figure(parc, summary, ident, out)
        print(f"\n=== {parc} ===")
        print(f"{'modelo':24s} {'r':>6s} {'r_desvio':>9s} {'AUC_top20':>9s} {'r_força':>8s} {'ident':>6s} {'p_ident':>8s} {'Idiff':>7s}")
        for s in summary:
            print(f"{s['model']:24s} {s['pearson']:6.3f} {s['pearson_desvio']:9.3f} {s['auc_top20']:9.3f} "
                  f"{s['pearson_forca_nodal']:8.3f} {s['ident_accuracy']:6.2f} {s['ident_p_binomial'] or float('nan'):8.2g} "
                  f"{s['diff_identifiability']:7.4f}")
        sig = [t for t in tests if t.get("p_holm", 1) < 0.05]
        print(f"testes pareados: {len(tests)}; significativos após Holm: {len(sig)}")
        for t in sig:
            print(f"  {t['a']} vs {t['b']} [{t['metric']}]: Δ={t['mean_diff']:+.4f}, p_holm={t['p_holm']:.2g}")
        print("saídas em", out)


if __name__ == "__main__":
    main()