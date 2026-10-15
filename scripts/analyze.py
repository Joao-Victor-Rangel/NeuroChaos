"""Agrega uma grade: tabela por execução, resumo por condição, contrastes entre topologias e figuras.

  python scripts/analyze.py --grid configs/grid_debug.yaml

Estatística (guia §8.4): média por condição com IC95% bootstrap sobre sementes; contrastes
distance − shuffled e distance − homogeneous por (K, Aθ) com teste de permutação e correção
de Holm sobre TODAS as métricas primárias × contrastes (definidas a priori no YAML).
"""

import argparse
import warnings
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from neurochaos.config import load_config, resolve
from neurochaos.grid import load_grid

COND = ["family", "topology", "K", "theta_nA", "gamma_nA"]


def flatten(m):
    a = m["activity"]
    r = {k: a[k] for k in ("rate_E_mean", "rate_I_mean", "rate_E_max", "frac_silent_E", "cv_isi_E")}
    for b in ("theta", "gamma"):
        pairs = list(m["plv"][b].values())
        r[f"plv_{b}_mean"] = np.mean([p["plv"] for p in pairs])
        r[f"plv_{b}_z"] = np.nanmean([p.get("z", np.nan) for p in pairs])
        r[f"plv_{b}_fracsig"] = np.mean([p.get("p", 1) < 0.05 for p in pairs])
        ent = list(m["entrain"][b].values())
        r[f"entrain_{b}_mean"] = np.mean([e["plv"] for e in ent])
        r[f"psd_relpower_{b}"] = np.mean([v[f"relpower_{b}"] for v in m["psd"].values()])
    pac = list(m["pac"].values())
    r["pac_mi_mean"] = np.nanmean([p["mi"] for p in pac])
    r["pac_z_mean"] = np.nanmean([p.get("z", np.nan) for p in pac])
    r["pac_fracsig"] = np.mean([p.get("p", 1) < 0.05 for p in pac])
    r["psd_peak_hz"] = np.mean([v["peak_hz"] for v in m["psd"].values()])
    ch = m.get("chaos")
    if ch:
        r["lambda_linear"] = ch["lambda_linear_per_s"] if ch["lambda_linear_per_s"] is not None else np.nan
        r["p_diverge_all"] = ch["p_diverge_all"]
        r["p_diverge_max_eps"] = ch["p_diverge_max_eps"]
        r["determinism_max_abs_mV"] = ch["determinism_max_abs_mV"]
    return r


def load_rows(root):
    rows = []
    for mf in sorted(root.glob("*/*/metrics.json")):
        d = mf.parent
        ov = json.loads((d / "overrides.json").read_text(encoding="utf-8"))
        m = json.loads(mf.read_text(encoding="utf-8"))["metrics"]
        row = {"family": d.parent.name, "run": d.name,
               "topology": ov.get("connectome.topology", "distance"),
               "K": float(ov.get("coupling.K", 1.0)),
               "theta_nA": float(ov.get("stimulus.theta.amp_nA", 0.0)),
               "gamma_nA": float(ov.get("stimulus.gamma.amp_nA", 0.0)),
               "seed": int(ov.get("seeds.conn", -1))}
        row.update(flatten(m))
        # Nulos por deslocamento circular só valem se a fase NÃO é estritamente periódica:
        # com theta (ou gamma) injetado, deslocar preserva o travamento/MI → z e p sem sentido.
        if row["theta_nA"] > 0:
            for k in ("pac_z_mean", "pac_fracsig", "plv_theta_z", "plv_theta_fracsig"):
                row[k] = np.nan
        if row["gamma_nA"] > 0:
            for k in ("plv_gamma_z", "plv_gamma_fracsig"):
                row[k] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def boot_ci(x, rng, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perm_test(a, b, rng, n=5000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, np.nan
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        cnt += abs(pool[: len(a)].mean() - pool[len(a):].mean()) >= abs(obs) - 1e-15
    ba = rng.choice(a, size=(2000, len(a))).mean(1) - rng.choice(b, size=(2000, len(b))).mean(1)
    return float(obs), float((cnt + 1) / (n + 1)), float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def summarize(df, metrics, rng):
    recs = []
    for key, g in df.groupby(COND):
        rec = dict(zip(COND, key))
        rec["n"] = len(g)
        for mt in metrics:
            if mt not in g:
                continue
            lo, hi = boot_ci(g[mt], rng)
            rec[f"{mt}_mean"] = float(np.nanmean(g[mt]))
            rec[f"{mt}_ci_lo"] = lo
            rec[f"{mt}_ci_hi"] = hi
        recs.append(rec)
    return pd.DataFrame(recs)


def contrasts(df, metrics, rng):
    main = df[df.family == "main"]
    recs = []
    for (K, th), g in main.groupby(["K", "theta_nA"]):
        ref = g[g.topology == "distance"]
        for other in ("shuffled", "homogeneous"):
            o = g[g.topology == other]
            if not len(o) or not len(ref):
                continue
            for mt in metrics:
                d, p, lo, hi = perm_test(ref[mt], o[mt], rng)
                recs.append({"K": K, "theta_nA": th, "contrast": f"distance-{other}", "metric": mt,
                             "diff": d, "ci_lo": lo, "ci_hi": hi, "p_perm": p})
    c = pd.DataFrame(recs)
    if len(c):
        c["p_holm"] = holm(c["p_perm"].values)
        c["significant_holm_0.05"] = c["p_holm"] < 0.05
    return c


def figures(df, summ, metrics, root, figdir, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    main = summ[summ.family == "main"]
    thetas = sorted(main.theta_nA.unique())
    colors = {"distance": "#1f77b4", "shuffled": "#d62728", "homogeneous": "#7f7f7f"}
    fig, axes = plt.subplots(len(metrics), len(thetas), figsize=(3.1 * len(thetas), 2.3 * len(metrics)),
                             sharex=True, squeeze=False)
    for i, mt in enumerate(metrics):
        for j, th in enumerate(thetas):
            ax = axes[i, j]
            for topo, gg in main[main.theta_nA == th].groupby("topology"):
                gg = gg.sort_values("K")
                if f"{mt}_mean" not in gg:
                    continue
                ax.plot(gg.K, gg[f"{mt}_mean"], "o-", ms=3, color=colors.get(topo), label=topo)
                ax.fill_between(gg.K, gg[f"{mt}_ci_lo"], gg[f"{mt}_ci_hi"], color=colors.get(topo), alpha=0.18)
            if i == 0:
                ax.set_title(f"Aθ = {th:g} nA", fontsize=9)
            if j == 0:
                ax.set_ylabel(mt, fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel("K (× K_base)")
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "metrics_vs_K.png", dpi=130)
    plt.close(fig)

    # Diagrama de bifurcação: amostras estroboscópicas (fase 0 de theta) de V médio da região 0.
    lab0 = labels[0]
    fig, axes = plt.subplots(1, len(thetas), figsize=(3.1 * len(thetas), 2.8), sharey=True, squeeze=False)
    for j, th in enumerate(thetas):
        ax = axes[0, j]
        sel = df[(df.family == "main") & (df.topology == "distance") & (df.theta_nA == th)]
        for _, r in sel.iterrows():
            f = root / r.family / r.run / "data.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            key = f"strobe_{lab0}"
            if key in z:
                y = z[key]
                ax.plot(np.full(len(y), r.K) + (r.seed - 5.5) * 0.008, y, ",", color="k", alpha=0.6)
        ax.set_title(f"distance, Aθ = {th:g} nA", fontsize=9)
        ax.set_xlabel("K (× K_base)")
    axes[0, 0].set_ylabel(f"V̄ {lab0} na fase θ = 0 (mV)")
    fig.tight_layout()
    fig.savefig(figdir / "bifurcation_strobe_vs_K.png", dpi=150)
    plt.close(fig)

    # Controles de gamma: PAC MI médio em função de (Aθ, Aγ), topologia distance, K = 1.
    sub = summ[(summ.topology == "distance") & (summ.K == 1.0)]
    if "pac_mi_mean_mean" in sub and sub.gamma_nA.nunique() > 1:
        piv = sub.pivot_table(index="gamma_nA", columns="theta_nA", values="pac_mi_mean_mean")
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(piv.shape[1]), [f"{v:g}" for v in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{v:g}" for v in piv.index])
        ax.set_xlabel("Aθ (nA)")
        ax.set_ylabel("Aγ (nA)")
        ax.set_title("PAC MI (média) — distance, K=1", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(figdir / "controls_pac_theta_gamma.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    a = ap.parse_args()
    grid = load_grid(a.grid)
    cfg = load_config(grid["base_config"])
    metrics = cfg["analysis"]["primary_metrics"]
    root = resolve("results") / grid["name"]
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # médias de condições sem dados válidos

    df = load_rows(root)
    if df.empty:
        raise SystemExit(f"nenhum metrics.json em {root}")
    df.to_csv(out / "runs.csv", index=False)
    all_metrics = [c for c in df.columns if c not in COND + ["run", "seed"]]
    summ = summarize(df, all_metrics, rng)
    summ.to_csv(out / "summary_by_condition.csv", index=False)
    con = contrasts(df, metrics, rng)
    con.to_csv(out / "contrasts_topology.csv", index=False)

    qc = {
        "n_runs": int(len(df)),
        "n_expected": int(sum(1 for _ in Path(root).glob("*/*/overrides.json"))),
        "determinism_all_zero": bool((df.get("determinism_max_abs_mV", pd.Series([0])) == 0).all()),
        "primary_metrics": metrics,
        "n_tests_holm": int(len(con)),
        "n_significant_holm": int(con["significant_holm_0.05"].sum()) if len(con) else 0,
    }
    if len(con):
        sig = con[con["significant_holm_0.05"]].sort_values("p_holm")
        qc["significant"] = sig[["K", "theta_nA", "contrast", "metric", "diff", "ci_lo", "ci_hi", "p_holm"]] \
            .to_dict(orient="records")
    (out / "qc_summary.json").write_text(json.dumps(qc, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k != "significant"}, indent=1))
    if qc.get("significant"):
        print("\nContrastes significativos (Holm):")
        print(pd.DataFrame(qc["significant"]).to_string(index=False))

    labels = [r["label"] for r in cfg["geometry"]["regions"]]
    figures(df, summ, metrics, root, out / "figures", labels)
    print("saídas em", out)


if __name__ == "__main__":
    main()