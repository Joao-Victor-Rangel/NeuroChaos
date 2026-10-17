"""Adendo 5: lesão virtual no gêmeo digital. Para cada gêmeo (adendo 4, variante "grupo"), zera a linha/coluna da
EC de cada região e mede o impacto à distância (média de |ΔFC| nas arestas que não tocam a região).
Testa se o vetor de impacto é individual e reprodutível entre metades (L1) e se vai além da força do nó (L2).

  python scripts/run_lesao.py
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import binomtest, wilcoxon

from neurochaos import ec, hopf, mica
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable

TR = 0.6


def impacts(EC, freqs):
    n = len(EC)
    F = ec.model_fc(EC, freqs)
    out = np.zeros(n)
    for k in range(n):
        L = EC.copy()
        L[k, :] = 0.0
        L[:, k] = 0.0
        D = np.abs(ec.model_fc(L, freqs) - F)
        keep = np.ones(n, bool)
        keep[k] = False
        sub = D[np.ix_(keep, keep)]
        out[k] = sub[np.triu_indices(n - 1, 1)].mean()
    return out


def subject_job(job):
    s, parc, C0, mask, eps, iters = job
    ts, _ = mica.load(s, parc, "timeseries")
    h = len(ts) // 2
    r = {}
    for half, seg in (("A", ts[:h]), ("B", ts[h:])):
        fc = hopf.fc(hopf.bandpass(seg, TR))
        freqs = hopf.intrinsic_freqs(seg, TR)
        EC, _ = ec.fit_ec(C0, fc, freqs, mask, n_iter=iters, eps=eps)
        r[half] = {"EC": EC, "freqs": np.asarray(freqs), "I": impacts(EC, freqs), "forca": EC.sum(1)}
    return s, r


def group_job(job):
    k, EC, freqs = job
    return k, impacts(EC, freqs)


def ident(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    C = np.corrcoef(A, B)[: len(A), len(A):]
    return np.argmax(C, axis=1) == np.arange(len(A))


def resid(I, x):
    X = np.c_[np.ones_like(x), x]
    return I - X @ np.linalg.lstsq(X, I, rcond=None)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", default="schaefer200")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    eps, iters = 0.01, 100
    t0 = time.time()
    out = resolve("results/lesao_virtual")
    out.mkdir(parents=True, exist_ok=True)
    subs = mica.subjects()
    folds = mica.read_folds()
    sc = {s: mica.load(s, a.parc, "sc")[0] for s in subs}
    labels = mica.load(subs[0], a.parc, "sc")[1]
    jobs = []
    for s in subs:
        grp = np.mean([sc[t] for t in subs if folds[t] != folds[s]], axis=0)
        np.fill_diagonal(grp, 0.0)
        jobs.append((s, a.parc, 0.2 * grp / grp.max(), (grp > 0).astype(float), eps, iters))
    R = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for s, r in ex.map(subject_job, jobs):
            R[s] = r
            print(f"{s}: impacto médio A={r['A']['I'].mean():.4f} B={r['B']['I'].mean():.4f}  [{time.time() - t0:.0f}s]",
                  flush=True)
        gjobs = []
        for k in sorted(set(folds.values())):
            tr = [s for s in subs if folds[s] != k]
            gjobs.append((k, np.mean([R[s]["A"]["EC"] for s in tr], 0), np.mean([R[s]["A"]["freqs"] for s in tr], 0)))
        Ig = dict(ex.map(group_job, gjobs))

    IA = np.array([R[s]["A"]["I"] for s in subs])
    IB = np.array([R[s]["B"]["I"] for s in subs])
    SA = np.array([R[s]["A"]["forca"] for s in subs])
    SB = np.array([R[s]["B"]["forca"] for s in subs])
    RA = np.array([resid(IA[i], SA[i]) for i in range(len(subs))])
    RB = np.array([resid(IB[i], SB[i]) for i in range(len(subs))])
    tests = {}
    for name, (X, Y) in {"L1_impacto": (IA, IB), "L2_residuo_forca": (RA, RB)}.items():
        hit = ident(X, Y)
        tests[name] = {"acc": float(hit.mean()), "p_binom": float(binomtest(int(hit.sum()), len(hit), 1 / len(hit),
                                                                              alternative="greater").pvalue),
                       "hits": hit.tolist()}
    ps = sorted(tests, key=lambda k: tests[k]["p_binom"])
    prev = 0.0
    for i, k in enumerate(ps):  # Holm (2 testes)
        prev = max(prev, min(1.0, tests[k]["p_binom"] * (len(ps) - i)))
        tests[k]["p_holm"] = prev
    r_ind = np.array([np.corrcoef(IA[i], IB[i])[0, 1] for i in range(len(subs))])
    r_grp = np.array([np.corrcoef(Ig[folds[s]], IB[i])[0, 1] for i, s in enumerate(subs)])
    r_between = np.mean([np.corrcoef(IA[i], IB[j])[0, 1] for i in range(len(subs)) for j in range(len(subs)) if i != j])
    order = np.argsort(-IA.mean(0))[:10]
    res = {"testes": tests,
           "descritivos": {"r_impacto_forca_medio": float(np.mean([np.corrcoef(IA[i], SA[i])[0, 1] for i in range(len(subs))])),
                           "r_IA_IB_mesma_pessoa": float(r_ind.mean()), "r_IA_IB_entre_pessoas": float(r_between),
                           "r_grupo_IB": float(r_grp.mean()), "n_individual_maior_que_grupo": int(np.sum(r_ind > r_grp)),
                           "p_wilcoxon_individual_vs_grupo": float(wilcoxon(r_ind, r_grp).pvalue),
                           "top10_impacto": [(labels[k], float(IA.mean(0)[k])) for k in order]},
           "preregistro_sha256": hashlib.sha256(resolve("docs/preregistro_adendo5_lesao_virtual.md").read_bytes()).hexdigest(),
           "subjects": subs, "elapsed_s": time.time() - t0}
    np.savez_compressed(out / f"{a.parc}.npz", subs=np.array(subs), IA=IA, IB=IB, SA=SA, SB=SB,
                        Ig=np.array([Ig[k] for k in sorted(Ig)]))
    (out / f"{a.parc}.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    for k, v in tests.items():
        print(f"{k:18s} acc={v['acc']:.0%} p={v['p_binom']:.2g} p_holm={v['p_holm']:.2g}")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["descritivos"].items() if k != "top10_impacto"})
    print("top10:", [l for l, _ in res["descritivos"]["top10_impacto"]])
    print(f"tempo {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()