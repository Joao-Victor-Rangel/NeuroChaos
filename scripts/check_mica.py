"""Integridade do MICA-MICs extraído + divisão em folds por sujeito (guia §3 e plano v2 §3.1).

Verifica para cada sujeito × parcelação × modalidade: arquivo presente, dimensões, NaN/Inf,
faixas (FC em [-1, 1], SC ≥ 0) e — teste de ORDEM dos nós — que a FC recalculada a partir das
séries BOLD coincide com a FC fornecida.

  python scripts/check_mica.py --parc schaefer200 aparc --folds 5
"""

import argparse
import csv
import json

import _bootstrap  # noqa: F401
import numpy as np

from neurochaos import mica
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", nargs="+", default=["schaefer200", "aparc"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261014)
    a = ap.parse_args()
    subs = mica.subjects()
    rep = {"n_subjects": len(subs), "subjects": subs, "parcellations": {}}
    problems = []
    for parc in a.parc:
        info = {}
        n_ts = []
        fc_err = []
        density = []
        for s in subs:
            try:
                d = mica.load_subject(s, parc)
                ts, lab_ts = mica.load(s, parc, "timeseries")
            except Exception as e:  # noqa: BLE001
                problems.append(f"{s}/{parc}: {e}")
                continue
            n = len(d["labels"])
            for k in ("sc", "edgeLength", "fc", "gd", "mpc"):
                M = d[k]
                if not np.all(np.isfinite(M)):
                    problems.append(f"{s}/{parc}/{k}: NaN/Inf")
                if not np.allclose(M, M.T):
                    problems.append(f"{s}/{parc}/{k}: não simétrica")
            if d["fc"].min() < -1 - 1e-6 or d["fc"].max() > 1 + 1e-6:
                problems.append(f"{s}/{parc}: FC fora de [-1,1]")
            if d["sc"].min() < 0:
                problems.append(f"{s}/{parc}: SC negativa")
            if lab_ts != d["labels"]:
                problems.append(f"{s}/{parc}: rótulos das séries ≠ matrizes")
            fc_re = np.corrcoef(ts.T)
            iu = np.triu_indices(n, 1)
            fc_err.append(float(np.max(np.abs(fc_re[iu] - d["fc"][iu]))))
            density.append(float(np.mean(d["sc"][iu] > 0)))
            n_ts.append(ts.shape[0])
        info["n_nodes"] = n
        info["timepoints"] = {"min": int(min(n_ts)), "max": int(max(n_ts))}
        info["fc_recomputed_max_abs_err"] = float(max(fc_err))
        info["sc_density"] = {"mean": float(np.mean(density)), "min": float(min(density)),
                              "max": float(max(density))}
        rep["parcellations"][parc] = info
        print(f"{parc}: {n} nós | séries {info['timepoints']} | "
              f"erro máx FC recalculada = {info['fc_recomputed_max_abs_err']:.2e} | "
              f"densidade SC = {info['sc_density']['mean']:.3f} "
              f"[{info['sc_density']['min']:.3f}, {info['sc_density']['max']:.3f}]")
    rep["problems"] = problems
    print(f"{len(subs)} sujeitos; {len(problems)} problemas")
    for p in problems[:20]:
        print("  ", p)

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(subs))
    man = resolve("data/manifests")
    with open(man / "subjects_split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "subject_id", "fold", "seed"])
        for k, i in enumerate(order):
            w.writerow(["mica_mics", subs[i], k % a.folds, a.seed])
    (man / "mica_integrity.json").write_text(json.dumps(to_jsonable(rep), indent=1), encoding="utf-8")
    print("escrito:", man / "subjects_split.csv", "e", man / "mica_integrity.json")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()