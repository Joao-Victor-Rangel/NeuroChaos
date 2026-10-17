"""Aumento de dados para conectividade funcional (pré-registro em docs/preregistro_gan_nested.md).

A FC de um sujeito é uma ESTIMATIVA ruidosa (série BOLD finita). Reamostrar a própria série por bootstrap
circular em blocos gera outras estimativas plausíveis da FC do mesmo sujeito, preservando a autocorrelação
temporal dentro de cada bloco. O alvo (SC) não muda. Nunca misturar sujeitos nem sair do fold.
"""

import numpy as np


def block_bootstrap_fc(ts, n, block=25, rng=None):
    """n FCs de bootstrap circular em blocos. ts: T x N. Retorna n x N x N."""
    rng = rng or np.random.default_rng()
    T, N = ts.shape
    nb = int(np.ceil(T / block))
    out = np.empty((n, N, N))
    idx_base = np.arange(block)
    for k in range(n):
        starts = rng.integers(0, T, size=nb)
        idx = ((starts[:, None] + idx_base[None, :]) % T).ravel()[:T]
        out[k] = np.corrcoef(ts[idx].T)
    return out