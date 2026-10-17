"""Adendo 4: gêmeo digital funcional. Ajusta a EC (Hopf linearizado) em cada metade do BOLD e testa se o gêmeo
reconhece a pessoa (P1: FC do modelo A × FC empírica B) e se seus parâmetros são estáveis (P2: EC_A × EC_B).

  python scripts/run_gemeo.py --variant grupo                     # primária
  python scripts/run_gemeo.py --variant proprio                   # S1
  python scripts/run_gemeo.py --variant grupo --eps 0.002 --iters 500   # S2
  python scripts/run_gemeo.py --variant grupo --parc schaefer400        # S3
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import binomtest

from neurochaos import ec, hopf, mica
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable

TR = 0.6


def z(M, iu):
    return np.arctanh(np.clip(M[iu], -0.999, 0.999))


def ec_vec(C, iu):
    v = np.maximum(C[iu], 0.0)
    return np.log1p(v / v.mean())


def fit_subject(job):
    s, parc, C0, mask, eps, iters = job
    ts, _ = mica.load(s, parc, "timeseries")
    h = len(ts) // 2
    out = {}
    for half, seg in (("A", ts[:h]), ("B", ts[h:])):
        fc = hopf.fc(hopf.bandpass(seg, TR))
        freqs = hopf.intrinsic_freqs(seg, TR)
        EC, hist = ec.fit_ec(C0, fc, freqs, mask, n_iter=iters, eps=eps, log_every=iters)
        out[half] = {"fc": fc, "EC": EC, "F": ec.model_fc(EC, freqs), "F0": ec.model_fc(C0, freqs),
                     "iter": hist[-1][1], "r_ajuste": hist[-1][2]}
    return s, out


def ident(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    C = np.corrcoef(A, B)[: len(A), len(A):]
    hit = np.argmax(C, axis=1) == np.arange(len(A))
    return hit, float(np.mean(np.diag(C)) - np.mean(C[~np.eye(len(A), dtype=bool)]))


def binom(hit):
    return float(binomtest(int(hit.sum()), len(hit), 1 / len(hit), alternative="greater").pvalue)


def mcnemar(h1, h2):
    b, c = int(np.sum(h1 & ~h2)), int(np.sum(~h1 & h2))
    return {"so_1": b, "so_2": c, "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["grupo", "proprio"], default="grupo")
    ap.add_argument("--parc", default="schaefer200")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    tag = f"{a.parc}_{a.variant}_eps{a.eps:g}_it{a.iters}"
    out = resolve("results/gemeo")
    out.mkdir(parents=True, exist_ok=True)
    subs = mica.subjects()
    folds = mica.read_folds()
    sc = {s: mica.load(s, a.parc, "sc")[0] for s in subs}
    n = len(sc[subs[0]])
    iu = np.triu_indices(n, 1)

    jobs = []
    for s in subs:
        if a.variant == "grupo":
            grp = np.mean([sc[t] for t in subs if folds[t] != folds[s]], axis=0)
        else:
            grp = sc[s].copy()
        np.fill_diagonal(grp, 0.0)
        jobs.append((s, a.parc, 0.2 * grp / grp.max(), (grp > 0).astype(float), a.eps, a.iters))
    res_s = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for s, r in ex.map(fit_subject, jobs):
            res_s[s] = r
            print(f"{s}: ajuste A r={r['A']['r_ajuste']:.3f} (it {r['A']['iter']}), "
                  f"B r={r['B']['r_ajuste']:.3f} (it {r['B']['iter']})  [{time.time() - t0:.0f}s]", flush=True)

    g = lambda k, h: np.array([(z(res_s[s][h][k], iu) if k != "EC" else ec_vec(res_s[s][h][k], iu)) for s in subs])
    FA, FB, F0A, ECA, ECB, FmA = g("fc", "A"), g("fc", "B"), g("F0", "A"), g("EC", "A"), g("EC", "B"), g("F", "A")
    tests = {}
    for name, (X, Y) in {"P1_gemeoA_x_fcB": (FmA, FB), "P2_ecA_x_ecB": (ECA, ECB),
                         "teto_fcA_x_fcB": (FA, FB), "C1_semajuste_x_fcB": (F0A, FB)}.items():
        hit, idiff = ident(X, Y)
        tests[name] = {"acc": float(hit.mean()), "p_binom": binom(hit), "idiff": idiff, "hits": hit.tolist()}
    ps = sorted([("P1_gemeoA_x_fcB", tests["P1_gemeoA_x_fcB"]["p_binom"]), ("P2_ecA_x_ecB", tests["P2_ecA_x_ecB"]["p_binom"])],
                key=lambda x: x[1])
    for i, (k, p) in enumerate(ps):  # Holm (2 testes)
        tests[k]["p_holm"] = float(min(1.0, max(p * (2 - i), tests[ps[i - 1][0]]["p_holm"] if i else 0)))
    h1, ht = np.array(tests["P1_gemeoA_x_fcB"]["hits"]), np.array(tests["teto_fcA_x_fcB"]["hits"])
    res = {"tag": tag, "args": vars(a), "testes": tests,
           "mcnemar_P1_vs_teto": mcnemar(h1, ht),
           "retencao_P1_sobre_teto": float(h1.mean() / max(ht.mean(), 1e-9)),
           "r_medio": {"gemeoA_x_fcA": float(np.mean([np.corrcoef(FmA[i], FA[i])[0, 1] for i in range(len(subs))])),
                       "gemeoA_x_fcB": float(np.mean([np.corrcoef(FmA[i], FB[i])[0, 1] for i in range(len(subs))])),
                       "fcA_x_fcB": float(np.mean([np.corrcoef(FA[i], FB[i])[0, 1] for i in range(len(subs))])),
                       "ecA_x_ecB": float(np.mean([np.corrcoef(ECA[i], ECB[i])[0, 1] for i in range(len(subs))]))},
           "iter_media": {h: float(np.mean([res_s[s][h]["iter"] for s in subs])) for h in "AB"},
           "preregistro_sha256": hashlib.sha256(resolve("docs/preregistro_adendo4_gemeo.md").read_bytes()).hexdigest(),
           "subjects": subs, "elapsed_s": time.time() - t0}
    np.savez_compressed(out / f"{tag}.npz", subs=np.array(subs), FA=FA, FB=FB, FmA=FmA, ECA=ECA, ECB=ECB, F0A=F0A)
    (out / f"{tag}.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")
    print(f"\n[{tag}]")
    for k, v in tests.items():
        print(f"  {k:22s} acc={v['acc']:.0%}  p={v['p_binom']:.2g}" + (f"  p_holm={v['p_holm']:.2g}" if "p_holm" in v else ""))
    print("  McNemar P1 vs teto:", res["mcnemar_P1_vs_teto"], " retenção:", round(res["retencao_P1_sobre_teto"], 2))
    print("  r médio:", {k: round(v, 3) for k, v in res["r_medio"].items()}, " iter:", res["iter_media"])
    print(f"  tempo {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()