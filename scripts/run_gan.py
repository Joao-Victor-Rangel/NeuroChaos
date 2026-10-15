"""GAN condicional e ablação U-Net (só L1) no MICA-MICs, com os MESMOS folds e métricas dos baselines.

  python scripts/run_gan.py --parc schaefer200 --seeds 0 1 2 --epochs 300

Saídas em results/gan/<parc>/: por sujeito, resumo, comparação com B2 e histórico de treino.
"""

import argparse
import csv
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

from neurochaos import gan, mica, sc_metrics, sc_models
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable

PRIMARY = ["pearson", "pearson_desvio", "auc_top20", "pearson_forca_nodal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", default="schaefer200")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=5, help="sujeitos de validação tirados do treino")
    ap.add_argument("--lam-l1", type=float, default=100.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--residual-b2", action="store_true",
                    help="GAN aprende só o resíduo sobre a previsão do B2 (que entra como canal extra)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print("dispositivo:", gan.device(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    mats = mica.load_matrices(a.parc)
    for s, d in mats.items():
        d["y"] = sc_models.target(d["sc"])
    folds = mica.read_folds()
    subs = sorted(mats)
    n = len(mats[subs[0]]["labels"])
    tag = "_resB2" if a.residual_b2 else ""
    out = resolve("results/gan") / (a.parc + tag)
    out.mkdir(parents=True, exist_ok=True)
    variants = {"GAN_pix2pix" + tag: 1.0, "UNet_so_L1" + tag: 0.0}
    labels = mats[subs[0]]["labels"]
    feat = {s: {"y": d["y"], "fc": sc_models.upper(d["fc"]), "gd": sc_models.upper(d["gd"]),
                "mpc": sc_models.upper(d["mpc"]), "inter": sc_models.interhemispheric(labels)}
            for s, d in mats.items()}
    b2_preds = {}

    preds = {(v, sd): {} for v in variants for sd in a.seeds}
    mus, hist = {}, []
    t0 = time.time()
    for k in sorted(set(folds.values())):
        test = [s for s in subs if folds[s] == k]
        train_all = [s for s in subs if folds[s] != k]
        rng = np.random.default_rng(1000 + k)
        val = list(rng.choice(train_all, size=a.n_val, replace=False))
        train = [s for s in train_all if s not in val]
        mu = np.mean([mats[s]["y"] for s in train], axis=0)  # prior só dos sujeitos de treino
        fold_base = None
        if a.residual_b2:  # B2 ajustado SÓ nos sujeitos de treino do fold (validação e teste ficam fora)
            b2 = sc_models.LinearMultimodal().fit([feat[s] for s in train])
            fold_base = {s: b2.predict(feat[s]) for s in train + val + test}
            for s in test:
                b2_preds[s] = fold_base[s]
        Xtr, Ytr, _ = gan.build_tensors(mats, train, mu, n, fold_base)
        Xva, Yva, Ova = gan.build_tensors(mats, val, mu, n, fold_base)
        Xte, _, _ = gan.build_tensors(mats, test, mu, n, fold_base)
        for s in test:
            mus[s] = mu
        iu = np.triu_indices(n, 1)
        for v, lam_adv in variants.items():
            for sd in a.seeds:
                print(f"fold {k} | {v} | semente {sd}  ({time.time() - t0:.0f}s)", flush=True)
                G, info = gan.train(Xtr, Ytr, Xva, Yva, lam_adv=lam_adv, lam_l1=a.lam_l1, epochs=a.epochs,
                                    seed=sd, log=print if sd == a.seeds[0] else None,
                                    Ova=Ova if a.residual_b2 else None, zero_init=a.residual_b2)
                hist.append({"fold": k, "variant": v, "seed": sd, **info})
                dev = gan.predict(G, Xte)
                for s, dmat in zip(test, dev):
                    preds[(v, sd)][s] = (fold_base[s] if a.residual_b2 else mu) + dmat[iu]

    # Ensemble de sementes (média das previsões) e avaliação.
    rows, summary, ident_all = [], [], {}
    truth = {s: mats[s]["y"] for s in subs}
    entries = [(f"{v}_ens", v, a.seeds) for v in variants] + \
              [(f"{v}_s{sd}", v, [sd]) for v in variants for sd in a.seeds]
    if a.residual_b2:  # o próprio B2 deste script (ajustado em 35 sujeitos): referência pareada exata
        entries.insert(0, ("B2_base_35sujeitos", None, None))
    for tag, v, keys in entries:
        if True:
            p = dict(b2_preds) if v is None else \
                {s: np.mean([preds[(v, sd)][s] for sd in keys], axis=0) for s in subs}
            rr = []
            for s in subs:
                r = {"model": tag, "subject": s, "fold": folds[s]}
                r.update(sc_metrics.edge_metrics(p[s], truth[s], mus[s], n))
                rr.append(r)
            idn = sc_metrics.identification(p, truth, mus)
            ident_all[tag] = idn["matrix"]
            rec = {"model": tag}
            for mt in PRIMARY:
                rec[mt] = float(np.nanmean([r[mt] for r in rr]))
            rec["ident_accuracy"] = idn["accuracy"]
            rec["ident_p_binomial"] = float(binomtest(int(round(idn["accuracy"] * len(subs))), len(subs),
                                                      1 / len(subs), alternative="greater").pvalue)
            rec["diff_identifiability"] = idn["differential_identifiability"]
            summary.append(rec)
            rows.extend(rr)

    # Comparação pareada com o melhor baseline (B2) — mesmos sujeitos, mesmos folds.
    base = {}
    with open(resolve("results/baselines") / a.parc / "per_subject.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == "B2_linear_multimodal":
                base[r["subject"]] = r
    tests = []
    for v in variants:
        for mt in ("pearson_desvio", "pearson"):
            x = np.array([next(r[mt] for r in rows if r["model"] == f"{v}_ens" and r["subject"] == s) for s in subs])
            if a.residual_b2:
                y = np.array([next(r[mt] for r in rows if r["model"] == "B2_base_35sujeitos" and r["subject"] == s)
                              for s in subs])
            else:
                y = np.array([float(base[s][mt]) for s in subs])
            tests.append({"a": f"{v}_ens", "b": "B2_base_35sujeitos" if a.residual_b2 else "B2_linear_multimodal",
                          "metric": mt,
                          "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                          "p_wilcoxon": float(wilcoxon(x, y).pvalue)})

    with open(out / "per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(to_jsonable({
        "parcellation": a.parc, "device": str(gan.device()), "torch": torch.__version__, "seeds": a.seeds, "epochs": a.epochs, "lam_l1": a.lam_l1,
        "summary": summary, "vs_B2": tests, "training": hist,
        "elapsed_s": time.time() - t0}), indent=1), encoding="utf-8")
    np.savez_compressed(out / "identification.npz", **ident_all)

    b2 = json.loads((resolve("results/baselines") / a.parc / "summary.json").read_text(encoding="utf-8"))
    b2s = next(s for s in b2["summary"] if s["model"] == "B2_linear_multimodal")
    print(f"\n{'modelo':22s} {'r':>6s} {'r_desvio':>9s} {'AUC_top20':>9s} {'ident':>6s} {'p_ident':>8s}")
    print(f"{'B2 (referência)':22s} {b2s['pearson']:6.3f} {b2s['pearson_desvio']:9.3f} {b2s['auc_top20']:9.3f} "
          f"{b2s['ident_accuracy']:6.2f} {b2s['ident_p_binomial']:8.2g}")
    for s in summary:
        print(f"{s['model']:22s} {s['pearson']:6.3f} {s['pearson_desvio']:9.3f} {s['auc_top20']:9.3f} "
              f"{s['ident_accuracy']:6.2f} {s['ident_p_binomial']:8.2g}")
    for t in tests:
        print(f"{t['a']} vs {t['b']} [{t['metric']}]: {t['mean_a']:.4f} vs {t['mean_b']:.4f}, p={t['p_wilcoxon']:.2g}")
    print(f"tempo total: {time.time() - t0:.0f}s; saídas em {out}")


if __name__ == "__main__":
    main()