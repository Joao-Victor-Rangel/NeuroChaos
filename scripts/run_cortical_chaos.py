"""Adendo E (exploratório): caos/sensibilidade a perturbações na rede cortical (NEURON/CoreNEURON) com conectoma
real. 10 sujeitos × {SC própria, SC própria embaralhada}; trajetórias gêmeas (λ linear e divergência finita).

  python scripts/run_cortical_chaos.py --workers 6
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import wilcoxon

from neurochaos import mica
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable
from neurochaos.runner import run_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cortical_atlas.yaml")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    subs_all = mica.subjects()
    rng = np.random.default_rng(20261014)  # mesma seleção dos 10 sujeitos da 1ª rodada cortical
    subs = sorted(rng.choice(subs_all, size=10, replace=False).tolist())
    croot = "data/connectomes/mica_aparc"
    out = resolve("results/escada/caos_cortical")
    common = {"simulation.duration_ms": 6000, "simulation.transient_ms": 2000, "simulation.record_all_v": True,
              "simulation.save_traces": False, "simulation.coreneuron": False, "coupling.K": 2.0, "seeds.conn": 1, "seeds.stim": 1001, "seeds.loc": 2001,
              "chaos.enabled": True, "chaos.targets_per_region": 1, "chaos.t_perturb_ms": [3000.0, 4000.0, 5000.0],
              "chaos.eps_nA": [1.0e-6, 1.0e-4, 1.0e-2], "chaos.horizon_ms": 1000.0}
    jobs = []
    for s in subs:
        jobs.append((out / f"{s}_propria", dict(common, **{"connectome.topology": "atlas",
                                                          "connectome.atlas_dir": f"{croot}/{s}"})))
        jobs.append((out / f"{s}_embaralhada", dict(common, **{"connectome.topology": "shuffled",
                                                              "connectome.source": "atlas",
                                                              "connectome.atlas_dir": f"{croot}/{s}"})))
    run_jobs(a.config, jobs, n_workers=a.workers, no_arrays=True)

    rows = []
    for s in subs:
        for c in ("propria", "embaralhada"):
            ch = json.loads((out / f"{s}_{c}" / "metrics.json").read_text(encoding="utf-8"))["metrics"]["chaos"]
            r = {"subject": s, "cond": c, "lambda_linear": ch["lambda_linear_per_s"],
                 "determinismo_mV": ch["determinism_max_abs_mV"], "p_div_all": ch["p_diverge_all"]}
            r.update({f"p_div_eps{k}": v for k, v in ch["p_diverge_by_eps"].items()})
            rows.append(r)
    res = {"subjects": subs, "rows": rows, "tests": {}}
    keys = [k for k in rows[0] if k not in ("subject", "cond", "determinismo_mV")]
    for k in keys:
        x = np.array([next(r[k] for r in rows if r["subject"] == s and r["cond"] == "propria") for s in subs], float)
        y = np.array([next(r[k] for r in rows if r["subject"] == s and r["cond"] == "embaralhada") for s in subs], float)
        ok = np.isfinite(x) & np.isfinite(y)
        p = float(wilcoxon(x[ok], y[ok]).pvalue) if ok.sum() >= 5 and not np.allclose(x[ok], y[ok]) else float("nan")
        res["tests"][k] = {"propria": float(np.nanmean(x)), "embaralhada": float(np.nanmean(y)), "p": p}
        print(f"{k:16s} própria={np.nanmean(x):+.3f}  embaralhada={np.nanmean(y):+.3f}  p={p:.3g}")
    print("determinismo (máx |ΔV| na repetição):", max(r["determinismo_mV"] for r in rows), "mV")
    (out / "summary.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()