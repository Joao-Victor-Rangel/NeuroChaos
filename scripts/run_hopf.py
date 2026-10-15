"""Simulação fisiológica: qual SC reproduz melhor o BOLD de cada pessoa? (plano v2 §3.4)

Por sujeito: G ajustado na 1ª metade do BOLD, avaliado na 2ª (nunca vista). SC comparadas:
  real (difusão do próprio sujeito) · B2 (prevista por FC+anatomia, fora do fold) ·
  grupo (média do fold de treino) · outro_sujeito (SC real de outra pessoa) ·
  embaralhada (SC real com arestas permutadas: mesma distribuição de pesos, topologia destruída).
Teto: correlação entre a FC da 1ª e a da 2ª metade do próprio sujeito.

  python scripts/run_hopf.py --parc schaefer200
"""

import argparse
import csv
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import wilcoxon

from neurochaos import hopf, mica, sc_models
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable

TR = 0.6  # s (Royer et al., 2022)


def from_target(v, n):
    M = np.zeros((n, n))
    M[np.triu_indices(n, 1)] = np.expm1(np.maximum(v, 0.0))
    return M + M.T


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty_like(adj)
    out[o] = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", default="schaefer200")
    ap.add_argument("--g-grid", type=float, nargs="+", default=list(np.round(np.arange(0.0, 3.01, 0.25), 2)))
    ap.add_argument("--n-rep", type=int, default=1)
    ap.add_argument("--sim-tr", type=int, default=3000,
                    help="duração da simulação em TRs; a FC do MODELO é estimada numa série longa (30 min)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sc-transform", choices=["log", "raw"], default="raw")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()
    t0 = time.time()

    mats = mica.load_matrices(a.parc)
    folds = mica.read_folds()
    subs = sorted(mats)[: a.limit] if a.limit else sorted(mats)
    allsubs = sorted(mats)
    n = len(mats[allsubs[0]]["labels"])
    labels = mats[allsubs[0]]["labels"]
    feat = {s: {"y": sc_models.target(d["sc"]), "fc": sc_models.upper(d["fc"]), "gd": sc_models.upper(d["gd"]),
                "mpc": sc_models.upper(d["mpc"]), "inter": sc_models.interhemispheric(labels)}
            for s, d in mats.items()}

    # SC prevista pelo B2 e média do grupo, ambas fora do fold.
    # O B2 do sujeito avaliado recebe a FC só da 1ª metade do BOLD (a 2ª é a de teste): sem vazamento.
    b2_models, sc_grp = {}, {}
    for k in sorted(set(folds.values())):
        train = [feat[s] for s in allsubs if folds[s] != k]
        m = sc_models.LinearMultimodal().fit(train)
        for s in allsubs:
            if folds[s] == k:
                b2_models[s] = m
                sc_grp[s] = from_target(m.mu, n)
    rng = np.random.default_rng(20261014)
    other = {s: allsubs[(i + 1 + rng.integers(len(allsubs) - 1)) % len(allsubs)] for i, s in enumerate(allsubs)}
    iu = np.triu_indices(n, 1)

    rows = []
    for s in subs:
        ts, _ = mica.load(s, a.parc, "timeseries")
        h = len(ts) // 2
        A, B = hopf.bandpass(ts[:h], TR), hopf.bandpass(ts[h:], TR)
        fcA, fcB = hopf.fc(A), hopf.fc(B)
        omega = hopf.intrinsic_freqs(ts[:h], TR)  # só da metade de ajuste
        sc_real = mats[s]["sc"]
        perm = rng.permutation(len(iu[0]))
        sh = np.zeros((n, n))
        sh[iu] = sc_real[iu][perm]
        sc_b2 = from_target(b2_models[s].predict(dict(feat[s], fc=fcA[iu])), n)
        candidates = {"real": sc_real, "B2": sc_b2, "grupo": sc_grp[s],
                      "outro_sujeito": mats[other[s]]["sc"], "embaralhada": sh + sh.T}
        row = {"subject": s, "teto_fcA_fcB": hopf.fit_score(fcA, fcB), "n_tr": len(ts),
               "direto_SC_FC_B": hopf.fit_score(np.log1p(sc_real / sc_real[iu].mean()), fcB)}
        for name, sc in candidates.items():
            C = hopf.coupling(sc, a.sc_transform)
            fcs = hopf.sim_fc_grid(C, omega, a.g_grid, a.sim_tr, TR, n_rep=a.n_rep, seed=int(s[-3:]))
            fitA = [hopf.fit_score(f, fcA) if np.all(np.isfinite(f)) else -np.inf for f in fcs]  # G instável → descartado
            g = int(np.argmax(fitA))
            row[f"{name}_G"] = a.g_grid[g]
            row[f"{name}_ajuste_A"] = fitA[g]
            row[f"{name}_teste_B"] = hopf.fit_score(fcs[g], fcB)
        rows.append(row)
        print(f"{s}: teto={row['teto_fcA_fcB']:.3f} | " +
              " ".join(f"{k}={row[f'{k}_teste_B']:.3f}(G={row[f'{k}_G']:g})" for k in candidates) +
              f"  [{time.time() - t0:.0f}s]", flush=True)

    names = ["real", "B2", "grupo", "outro_sujeito", "embaralhada"]
    summary = {nm: {"teste_B_media": float(np.mean([r[f"{nm}_teste_B"] for r in rows])),
                    "G_mediano": float(np.median([r[f"{nm}_G"] for r in rows]))} for nm in names}
    summary["teto"] = float(np.mean([r["teto_fcA_fcB"] for r in rows]))
    summary["direto_SC_FC"] = float(np.mean([r["direto_SC_FC_B"] for r in rows]))
    tests = []
    for x, y in [("real", "outro_sujeito"), ("real", "grupo"), ("real", "embaralhada"), ("real", "B2"),
                 ("B2", "grupo"), ("B2", "outro_sujeito"), ("grupo", "embaralhada")]:
        dx = np.array([r[f"{x}_teste_B"] for r in rows])
        dy = np.array([r[f"{y}_teste_B"] for r in rows])
        tests.append({"a": x, "b": y, "mean_diff": float(np.mean(dx - dy)),
                      "n_a_melhor": int(np.sum(dx > dy)), "p": float(wilcoxon(dx, dy).pvalue)})
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = float(ph)

    out = resolve("results/hopf") / (a.parc + a.out_tag)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(to_jsonable({
        "parcellation": a.parc, "sc_transform": a.sc_transform, "sim_tr": a.sim_tr, "TR_s": TR, "G_grid": a.g_grid, "n_rep": a.n_rep, "n_subjects": len(rows),
        "summary": summary, "tests": tests, "elapsed_s": time.time() - t0}), indent=1), encoding="utf-8")

    print(f"\nteto (FC metade A × metade B): {summary['teto']:.3f} | correlação direta log SC × FC, sem modelo: "
          f"{summary['direto_SC_FC']:.3f}")
    for nm in names:
        print(f"{nm:14s} FC simulada × FC empírica (metade B): {summary[nm]['teste_B_media']:.3f}  "
              f"G mediano={summary[nm]['G_mediano']:g}")
    for t in tests:
        print(f"{t['a']} vs {t['b']}: Δ={t['mean_diff']:+.4f}, {t['n_a_melhor']}/{len(rows)} sujeitos, "
              f"p_holm={t['p_holm']:.2g}")
    print(f"tempo: {time.time() - t0:.0f}s; saídas em {out}")


if __name__ == "__main__":
    main()