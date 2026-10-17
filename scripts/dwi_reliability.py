"""Confiabilidade da SC reconstruída no piloto: quanto ela muda só pelo acaso da tractografia?

Para cada sujeito com sc_seed1.csv e sc_seed2.csv (mesmos dados, sementes diferentes):
- r dentro do sujeito (semente 1 × semente 2) vs r entre sujeitos;
- com ≥ 3 sujeitos: desvios em relação à média do grupo, identificação semente 1 → semente 2 e ICC por aresta.

  python scripts/dwi_reliability.py results/dwi_pilot_previa
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


def load(p):
    M = np.loadtxt(p, delimiter=",")
    if M.shape[0] == 401:  # rótulo 0 (fundo) presente em algumas versões
        M = M[1:, 1:]
    return M


def vec(M):
    iu = np.triu_indices(len(M), 1)
    v = M[iu]
    return np.log1p(v / v[v > 0].mean())


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/dwi_pilot")
    subs = sorted(d.name for d in root.iterdir() if (d / "sc_seed1.csv").exists() and (d / "sc_seed2.csv").exists())
    if not subs:
        print("nenhum sujeito completo em", root)
        return
    S1 = {s: vec(load(root / s / "sc_seed1.csv")) for s in subs}
    S2 = {s: vec(load(root / s / "sc_seed2.csv")) for s in subs}
    res = {"subjects": subs, "n_nos": int(load(root / subs[0] / "sc_seed1.csv").shape[0])}
    res["densidade"] = {s: float(np.mean(S1[s] > 0)) for s in subs}
    res["r_dentro"] = {s: float(np.corrcoef(S1[s], S2[s])[0, 1]) for s in subs}
    if len(subs) >= 2:
        res["r_entre"] = {f"{a}|{b}": float(np.corrcoef(S1[a], S1[b])[0, 1]) for a, b in combinations(subs, 2)}
    print(f"sujeitos: {subs} | nós: {res['n_nos']}")
    for s in subs:
        print(f"  {s}: densidade {res['densidade'][s]:.3f} | r semente1×semente2 = {res['r_dentro'][s]:.4f}")
    if "r_entre" in res:
        print(f"  r entre sujeitos (média) = {np.mean(list(res['r_entre'].values())):.4f}")
    if len(subs) >= 3:
        mu = np.mean([S1[s] for s in subs] + [S2[s] for s in subs], axis=0)
        D1 = np.array([S1[s] - mu for s in subs])
        D2 = np.array([S2[s] - mu for s in subs])
        C = np.corrcoef(D1, D2)[: len(subs), len(subs):]
        res["desvio_r_dentro"] = float(np.mean(np.diag(C)))
        res["desvio_r_entre"] = float(np.mean(C[~np.eye(len(subs), dtype=bool)]))
        res["identificacao"] = float(np.mean(np.argmax(C, axis=1) == np.arange(len(subs))))
        # ICC(3,1) por aresta: variância entre sujeitos vs ruído entre sementes
        X = np.stack([np.array([S1[s] for s in subs]), np.array([S2[s] for s in subs])], axis=2)  # sub × aresta × 2
        ms_sub = 2 * X.mean(2).var(0, ddof=1)
        ms_err = ((X - X.mean(2, keepdims=True)) ** 2).sum((0, 2)) / (len(subs) - 1)
        icc = (ms_sub - ms_err) / (ms_sub + ms_err + 1e-12)
        res["icc_mediana"] = float(np.nanmedian(icc))
        res["icc_frac_acima_0.5"] = float(np.nanmean(icc > 0.5))
        print(f"  desvio individual: r dentro = {res['desvio_r_dentro']:.3f} vs entre = {res['desvio_r_entre']:.3f} | "
              f"identificação {res['identificacao']:.0%} (acaso {1 / len(subs):.0%}) | ICC mediana {res['icc_mediana']:.3f}")
    (root / "confiabilidade.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()