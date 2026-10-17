"""Adendo 6: gêmeo biofísico. A EC individual (gêmeo do adendo 4, ajustada na 1ª metade do BOLD) vira a fiação da
rede cortical NEURON/NetPyNE (aparc, 70 regiões). A FC lenta simulada é comparada com a FC da 2ª metade.

  python scripts/run_gemeo_neuron.py --workers 4
"""

import argparse
import csv
import hashlib
import json
import shutil

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import binomtest, wilcoxon

from neurochaos import ec, hopf, mica
from neurochaos.config import load_config, resolve
from neurochaos.jsonutil import to_jsonable
from neurochaos.runner import run_jobs
from run_cortical_subjects import corr_upper, holm, sim_fcs

PARC = "aparc"
TR = 0.6
CONDS = ("gemeo_proprio", "gemeo_outro", "gemeo_grupo", "sc_grupo", "gemeo_embaralhado")


def write_atlas(dst, W, src):
    dst.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst / "weights.csv", W, delimiter=",", fmt="%.6g")
    for f in ("distances_mm.csv", "region_labels.tsv"):
        shutil.copy(src / f, dst / f)


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cortical_atlas.yaml")
    ap.add_argument("--duration", type=float, default=20000.0)
    ap.add_argument("--transient", type=float, default=2000.0)
    ap.add_argument("--K", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/gemeo_neuron")
    ap.add_argument("--n-subjects", type=int, default=0, help="0 = todos (>0 só para teste rápido)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    subs = mica.subjects()
    folds = mica.read_folds()
    croot = resolve("data/connectomes/gemeo_aparc")
    grp_dir = resolve("data/connectomes/mica_aparc/group")
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) gêmeos lineares (só a metade A) e exportação para o formato atlas
    sc = {s: mica.load(s, PARC, "sc")[0] for s in subs}
    labels = mica.load_subject(subs[0], PARC, kinds=("fc",))["labels"]
    iu = np.triu_indices(len(labels), 1)
    EC, FB, FmA = {}, {}, {}
    for s in subs:
        ts, _ = mica.load(s, PARC, "timeseries")
        h = len(ts) // 2
        grp = np.mean([sc[t] for t in subs if folds[t] != folds[s]], axis=0)
        np.fill_diagonal(grp, 0.0)
        fcA = hopf.fc(hopf.bandpass(ts[:h], TR))
        freqs = hopf.intrinsic_freqs(ts[:h], TR)
        EC[s], _ = ec.fit_ec(0.2 * grp / grp.max(), fcA, freqs, (grp > 0).astype(float))
        FmA[s] = ec.model_fc(EC[s], freqs)
        FB[s] = hopf.fc(hopf.bandpass(ts[h:], TR))
        write_atlas(croot / s, EC[s], grp_dir)
    for k in sorted(set(folds.values())):
        write_atlas(croot / f"grupo_fold{k}", np.mean([EC[t] for t in subs if folds[t] != k], 0), grp_dir)
    perm = derangement(len(subs), 20261017)
    other = {s: subs[perm[i]] for i, s in enumerate(subs)}
    print("gêmeos exportados em", croot, flush=True)

    # 2) simulações NEURON
    rel = lambda p: str(p.relative_to(resolve(".")))
    # emenda 1: total de conexões longas igual ao da SC do grupo em todas as condições
    from neurochaos import connectome as cn
    _, Wg, _ = cn.load_atlas("data/connectomes/mica_aparc/group")
    off = ~np.eye(len(Wg), dtype=bool)
    Pg = cn.to_probabilities(np.where(off, np.log1p(Wg / Wg[off].mean()), 0.0), cfg["connectome"]["p0"], cfg["connectome"]["pmax"])
    total_P = float(Pg.sum())
    print(f"total de probabilidade de conexão (SC grupo) = {total_P:.2f}", flush=True)
    common = {"connectome.match_total_P": total_P, "simulation.duration_ms": a.duration, "simulation.transient_ms": a.transient,
              "simulation.record_all_v": False, "simulation.save_traces": False,
              "stimulus.theta.amp_nA": 0.0, "coupling.K": a.K, "seeds.conn": 1, "seeds.stim": 1001, "seeds.loc": 2001}
    jobs = []
    for s in (subs[: a.n_subjects] if a.n_subjects else subs):
        conds = {"gemeo_proprio": {"connectome.topology": "atlas", "connectome.atlas_dir": rel(croot / s)},
                 "gemeo_outro": {"connectome.topology": "atlas", "connectome.atlas_dir": rel(croot / other[s])},
                 "gemeo_grupo": {"connectome.topology": "atlas", "connectome.atlas_dir": rel(croot / f"grupo_fold{folds[s]}")},
                 "sc_grupo": {"connectome.topology": "atlas", "connectome.atlas_dir": "data/connectomes/mica_aparc/group"},
                 "gemeo_embaralhado": {"connectome.topology": "shuffled", "connectome.source": "atlas",
                                       "connectome.atlas_dir": rel(croot / s)}}
        for c, ov in conds.items():
            jobs.append((out / f"{s}_{c}", dict(common, **ov)))
    print(f"{len(jobs)} simulações NEURON de {a.duration / 1000:g} s", flush=True)
    st = run_jobs(a.config, jobs, n_workers=a.workers)
    bad = [str(d) for d, s_, _ in st if s_ == "error"]
    if bad:
        print("FALHAS:", bad, flush=True)

    # 3) análise
    fs = float(cfg["simulation"]["signal_fs_hz"])
    rows, simF = [], {}
    for s in subs:
        for c in CONDS:
            d = out / f"{s}_{c}"
            if not (d / "metrics.json").exists():
                continue
            z = np.load(d / "data.npz", allow_pickle=True)
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["metrics"]
            F = sim_fcs(z, labels, fs)["fc_lenta"]
            if c == "gemeo_proprio":
                simF[s] = F
            rows.append({"subject": s, "cond": c, "r_sim_x_fcB": corr_upper(F, FB[s]),
                         "rate_E": m["activity"]["rate_E_mean"], "rate_I": m["activity"]["rate_I_mean"],
                         "plv_gamma": float(np.mean([p["plv"] for p in m["plv"]["gamma"].values()])),
                         "pac_mi": float(np.nanmean([p["mi"] for p in m["pac"].values()]))})
    by = {(r["subject"], r["cond"]): r for r in rows}
    ok = [s for s in subs if all((s, c) in by for c in CONDS)]

    def wil(b):
        x = np.array([by[(s, "gemeo_proprio")]["r_sim_x_fcB"] for s in ok])
        y = np.array([by[(s, b)]["r_sim_x_fcB"] for s in ok])
        m_ = np.isfinite(x) & np.isfinite(y)
        return {"media_proprio": float(np.nanmean(x)), "media_" + b: float(np.nanmean(y)),
                "n_proprio_maior": int(np.sum(x[m_] > y[m_])), "n": int(m_.sum()), "p": float(wilcoxon(x[m_], y[m_]).pvalue)}

    def ident(A, B):
        A = A - np.nanmean(A, 0)
        B = B - B.mean(0)
        A = np.nan_to_num(A)
        C = np.corrcoef(A, B)[: len(A), len(A):]
        return np.argmax(C, 1) == np.arange(len(A))

    S = np.array([simF[s][iu] for s in ok])
    B = np.array([np.arctanh(np.clip(FB[s][iu], -0.999, 0.999)) for s in ok])
    L = np.array([np.arctanh(np.clip(FmA[s][iu], -0.999, 0.999)) for s in ok])
    hitN, hitL = ident(S, B), ident(L, B)
    tests = {"N1_proprio_vs_outro": wil("gemeo_outro"), "N2_proprio_vs_sc_grupo": wil("sc_grupo"),
             "N3_identificacao": {"acc": float(hitN.mean()), "hits": hitN.tolist(),
                                  "p": float(binomtest(int(hitN.sum()), len(ok), 1 / len(ok), alternative="greater").pvalue)}}
    for k, p in zip(tests, holm([tests[k]["p"] for k in tests])):
        tests[k]["p_holm"] = float(p)
    sec = {"proprio_vs_gemeo_grupo": wil("gemeo_grupo"), "proprio_vs_embaralhado": wil("gemeo_embaralhado"),
           "ident_gemeo_linear_aparc": float(hitL.mean()),
           "retencao_neuron_sobre_linear": float(hitN.mean() / max(hitL.mean(), 1e-9)),
           "por_condicao": {c: {mt: float(np.nanmean([by[(s, c)][mt] for s in ok]))
                                for mt in ("r_sim_x_fcB", "rate_E", "rate_I", "plv_gamma", "pac_mi")} for c in CONDS}}
    with open(out / "per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {"n": len(ok), "testes": tests, "secundarios": sec, "others": other,
           "preregistro_sha256": hashlib.sha256(resolve("docs/preregistro_adendo6_gemeo_neuron.md").read_bytes()).hexdigest()}
    (out / "summary.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    for k, v in tests.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "hits"})
    for k, v in sec["por_condicao"].items():
        print(f"  {k:18s}", {kk: round(vv, 4) for kk, vv in v.items()})
    print("  ident gêmeo linear (aparc):", sec["ident_gemeo_linear_aparc"], "retenção:", round(sec["retencao_neuron_sobre_linear"], 2))


if __name__ == "__main__":
    main()