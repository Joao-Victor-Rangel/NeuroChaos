"""Rede cortical NEURON/NetPyNE (RS+FS, canais NMODL) sobre conectomas reais do MICA-MICs.

Para cada sujeito s, quatro conectomas na MESMA rede (mesmas sementes):
  propria (SC de s) · outro_sujeito (SC de outra pessoa) · grupo (SC média) · embaralhada (SC de s permutada)
A FC simulada (correlação entre regiões das flutuações lentas da taxa E, e do envelope gama) é comparada
com a FC do BOLD real de s. Pergunta: a SC individual faz a rede reproduzir melhor a fisiologia de s?

  python scripts/run_cortical_subjects.py --n-subjects 10 --duration 20000 --workers 4
"""

import argparse
import csv
import json

import _bootstrap  # noqa: F401
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import wilcoxon

from neurochaos import mica
from neurochaos.config import load_config, resolve
from neurochaos.jsonutil import to_jsonable
from neurochaos.runner import run_jobs

CONDS = ("propria", "outro_sujeito", "grupo", "embaralhada")


def sim_fcs(z, labels, fs):
    """FC simulada por dois métodos a partir das taxas E por região (Hz/célula, 1 kHz)."""
    R = np.array([z[f"rate_{lab}_E"] for lab in labels]).T  # T x N
    sos_lo = butter(2, 2.0, btype="low", fs=fs, output="sos")
    slow = sosfiltfilt(sos_lo, R - R.mean(0), axis=0)
    sos_g = butter(4, [30.0, 80.0], btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos_g, R - R.mean(0), axis=0), axis=0))
    env = sosfiltfilt(sos_lo, env - env.mean(0), axis=0)
    e = int(0.5 * fs)
    out = {}
    for name, X in (("fc_lenta", slow[e:-e]), ("fc_aec_gama", env[e:-e])):
        ok = X.std(0) > 1e-9
        C = np.full((len(labels), len(labels)), np.nan)
        C[np.ix_(ok, ok)] = np.corrcoef(X[:, ok].T)
        out[name] = C
    return out


def corr_upper(A, B):
    iu = np.triu_indices(len(A), 1)
    a, b = A[iu], B[iu]
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty_like(adj)
    out[o] = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cortical_atlas.yaml")
    ap.add_argument("--n-subjects", type=int, default=10)
    ap.add_argument("--duration", type=float, default=20000.0)
    ap.add_argument("--transient", type=float, default=2000.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--theta-nA", type=float, default=0.0)
    ap.add_argument("--K", type=float, default=2.0,
                    help="acoplamento global em múltiplos de K_base; 2 = maior valor ainda no regime válido")
    a = ap.parse_args()

    cfg = load_config(a.config)
    parc = "aparc"
    subs_all = mica.subjects()
    rng = np.random.default_rng(20261014)
    subs = sorted(rng.choice(subs_all, size=a.n_subjects, replace=False).tolist())
    others = {s: subs_all[(subs_all.index(s) + 1 + int(rng.integers(len(subs_all) - 1))) % len(subs_all)]
              for s in subs}
    croot = "data/connectomes/mica_aparc"
    engine = "_core" if cfg["simulation"].get("coreneuron", False) else ""  # não misturar motores/ruído
    out = resolve("results/cortical_subjects") / f"K{a.K:g}_th{a.theta_nA:g}{engine}"
    common = {"simulation.duration_ms": a.duration, "simulation.transient_ms": a.transient,
              "simulation.record_all_v": False, "simulation.save_traces": False,
              "stimulus.theta.amp_nA": a.theta_nA, "coupling.K": a.K,
              # mesmas sementes para todas as condições de um sujeito: só o conectoma muda
              "seeds.conn": 1, "seeds.stim": 1001, "seeds.loc": 2001}
    jobs = []
    for s in subs:
        conds = {
            "propria": {"connectome.topology": "atlas", "connectome.atlas_dir": f"{croot}/{s}"},
            "outro_sujeito": {"connectome.topology": "atlas", "connectome.atlas_dir": f"{croot}/{others[s]}"},
            "grupo": {"connectome.topology": "atlas", "connectome.atlas_dir": f"{croot}/group"},
            "embaralhada": {"connectome.topology": "shuffled", "connectome.source": "atlas",
                            "connectome.atlas_dir": f"{croot}/{s}"},
        }
        for c, ov in conds.items():
            jobs.append((out / f"{s}_{c}", dict(common, **ov)))
    print(f"{len(jobs)} simulações ({a.n_subjects} sujeitos × {len(CONDS)} conectomas), {a.duration / 1000:g} s cada")
    run_jobs(a.config, jobs, n_workers=a.workers)

    fs = float(cfg["simulation"]["signal_fs_hz"])
    rows = []
    for s in subs:
        fc_emp = mica.load_subject(s, parc, kinds=("fc",))["fc"]
        labels = mica.load_subject(s, parc, kinds=("fc",))["labels"]
        for c in CONDS:
            d = out / f"{s}_{c}"
            z = np.load(d / "data.npz", allow_pickle=True)
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["metrics"]
            fcs = sim_fcs(z, labels, fs)
            r = {"subject": s, "cond": c, "other": others[s] if c == "outro_sujeito" else "",
                 "rate_E": m["activity"]["rate_E_mean"], "rate_I": m["activity"]["rate_I_mean"],
                 "cv_isi_E": m["activity"]["cv_isi_E"]}
            for name, F in fcs.items():
                r[f"{name}_x_bold"] = corr_upper(F, fc_emp)
            for b in ("theta", "gamma"):
                r[f"plv_{b}"] = float(np.mean([p["plv"] for p in m["plv"][b].values()]))
            r["pac_mi"] = float(np.nanmean([p["mi"] for p in m["pac"].values()]))
            rows.append(r)

    metrics = ["fc_lenta_x_bold", "fc_aec_gama_x_bold", "plv_gamma", "plv_theta", "pac_mi", "rate_E"]
    summary = {c: {mt: float(np.nanmean([r[mt] for r in rows if r["cond"] == c])) for mt in metrics} for c in CONDS}
    tests = []
    by = {(r["subject"], r["cond"]): r for r in rows}
    for b in ("outro_sujeito", "grupo", "embaralhada"):
        for mt in metrics:
            x = np.array([by[(s, "propria")][mt] for s in subs])
            y = np.array([by[(s, b)][mt] for s in subs])
            ok = np.isfinite(x) & np.isfinite(y)
            p = float(wilcoxon(x[ok], y[ok]).pvalue) if ok.sum() >= 5 and not np.allclose(x[ok], y[ok]) else float("nan")
            tests.append({"a": "propria", "b": b, "metric": mt, "mean_diff": float(np.nanmean(x - y)),
                          "n_a_maior": int(np.sum(x[ok] > y[ok])), "n": int(ok.sum()), "p": p})
    ps = [t["p"] for t in tests if np.isfinite(t["p"])]
    adj = iter(holm(ps))
    for t in tests:
        t["p_holm"] = float(next(adj)) if np.isfinite(t["p"]) else float("nan")

    with open(out / "per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(to_jsonable({
        "subjects": subs, "others": others, "duration_ms": a.duration, "theta_nA": a.theta_nA, "K": a.K,
        "summary": summary, "tests": tests}), indent=1), encoding="utf-8")
    print(f"\n{'conectoma':14s} " + " ".join(f"{m_:>18s}" for m_ in metrics))
    for c in CONDS:
        print(f"{c:14s} " + " ".join(f"{summary[c][m_]:18.4f}" for m_ in metrics))
    print("\npropria vs ... (Wilcoxon pareado, Holm):")
    for t in tests:
        print(f"  {t['b']:14s} {t['metric']:20s} Δ={t['mean_diff']:+.4f}  {t['n_a_maior']}/{t['n']}  p_holm={t['p_holm']:.2g}")
    print("saídas em", out)


if __name__ == "__main__":
    main()