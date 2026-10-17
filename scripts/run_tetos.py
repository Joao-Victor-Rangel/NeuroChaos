"""Tetos de identificação no MICA-MICs (passo 1 do plano de melhoria; não altera nenhum resultado anterior).

Teto funcional: identificar cada sujeito comparando a FC da 1ª metade do BOLD com a FC da 2ª metade de todos
(FC → FC, mesma modalidade). Medido nas três parcelações, com e sem remover a média do grupo.
Interpretação: limita quanto da identidade individual a FC de ~3,5 min carrega.

  python scripts/run_tetos.py
"""

import json

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import binomtest

from neurochaos import hopf, mica
from neurochaos.config import resolve
from neurochaos.jsonutil import to_jsonable

TR = 0.6


def ident(A, B, demean=True):
    """A[i] identifica B[i]? Retorna acurácia (A→B), (B→A) e identificabilidade diferencial."""
    if demean:
        A = A - A.mean(0)
        B = B - B.mean(0)
    C = np.corrcoef(A, B)[: len(A), len(A):]
    acc_ab = float(np.mean(np.argmax(C, axis=1) == np.arange(len(A))))
    acc_ba = float(np.mean(np.argmax(C, axis=0) == np.arange(len(A))))
    idiff = float(np.mean(np.diag(C)) - np.mean(C[~np.eye(len(A), dtype=bool)]))
    return acc_ab, acc_ba, idiff


def main():
    out = resolve("results/tetos")
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for parc in ("aparc", "schaefer200", "schaefer400"):
        subs = mica.subjects()
        FA, FB = [], []
        for s in subs:
            ts, _ = mica.load(s, parc, "timeseries")
            h = len(ts) // 2
            n = ts.shape[1]
            iu = np.triu_indices(n, 1)
            FA.append(np.arctanh(np.clip(hopf.fc(hopf.bandpass(ts[:h], TR))[iu], -0.999, 0.999)))
            FB.append(np.arctanh(np.clip(hopf.fc(hopf.bandpass(ts[h:], TR))[iu], -0.999, 0.999)))
        FA, FB = np.array(FA), np.array(FB)
        r = {}
        for dm in (False, True):
            ab, ba, idf = ident(FA, FB, demean=dm)
            k = "com_media_removida" if dm else "bruto"
            r[k] = {"acc_A_para_B": ab, "acc_B_para_A": ba, "idiff": idf,
                    "p_binomial": float(binomtest(int(round(ab * len(subs))), len(subs), 1 / len(subs),
                                                  alternative="greater").pvalue)}
        rel = np.mean([np.corrcoef(FA[i], FB[i])[0, 1] for i in range(len(subs))])
        r["fc_split_half_r_medio"] = float(rel)
        res[parc] = r
        print(f"{parc:12s} FC→FC identificação: bruto {r['bruto']['acc_A_para_B']:.0%} | média removida "
              f"{r['com_media_removida']['acc_A_para_B']:.0%} (B→A {r['com_media_removida']['acc_B_para_A']:.0%}) | "
              f"r split-half {rel:.3f}", flush=True)
    (out / "teto_fc.json").write_text(json.dumps(to_jsonable(res), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()