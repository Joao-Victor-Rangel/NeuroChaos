"""Conectividade efetiva (EC) pelo modelo de Hopf LINEARIZADO (adendo do pré-registro, seção D).

Perto do ponto fixo (a < 0) o modelo de Hopf é um processo de Ornstein–Uhlenbeck 2N-dimensional:
    d[x; y] = J [x; y] dt + σ dW,   J = [[A, −Ω], [Ω, A]],  A = a·I + G (C − diag(C·1)),  Ω = diag(2π f)
A covariância estacionária Σ resolve J Σ + Σ Jᵀ + σ² I = 0 (Lyapunov), sem simular. A FC do modelo é a
correlação do bloco x. A EC é obtida ajustando C iterativamente até a FC do modelo reproduzir a FC empírica.
Como o laplaciano é semidefinido negativo, J é estável para qualquer C ≥ 0 com a < 0.
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov


def model_fc(C, freqs_hz, G=1.0, a=-0.02, sigma=0.02):
    N = len(C)
    A = a * np.eye(N) + G * (C - np.diag(C.sum(1)))
    Om = np.diag(2 * np.pi * np.asarray(freqs_hz))
    J = np.block([[A, -Om], [Om, A]])
    S = solve_continuous_lyapunov(J, -(sigma ** 2) * np.eye(2 * N))
    Sx = S[:N, :N]
    d = np.sqrt(np.diag(Sx))
    return Sx / np.outer(d, d)


def fit_ec(C0, fc_emp, freqs_hz, mask, n_iter=100, eps=0.01, G=1.0, log_every=0):
    """C ← max(0, C + ε (FC_emp − FC_mod)) nas arestas de `mask`.

    Retorna (EC, histórico): a EC é a da iteração com maior correlação com `fc_emp` (emenda 1 do adendo:
    o ajuste não é monotônico). `fc_emp` deve ser a FC de AJUSTE (1ª metade), nunca a de avaliação.
    """
    C = C0.copy()
    iu = np.triu_indices(len(C), 1)
    hist = []
    best, best_r, best_it = C.copy(), -np.inf, 0
    for it in range(n_iter + 1):
        F = model_fc(C, freqs_hz, G)
        r = float(np.corrcoef(F[iu], fc_emp[iu])[0, 1])
        if r > best_r:
            best, best_r, best_it = C.copy(), r, it
        if log_every and (it % log_every == 0 or it == n_iter):
            hist.append((it, r))
        if it == n_iter:
            hist.append(("melhor", best_it, best_r))
            return best, hist
        D = (fc_emp - F) * mask
        np.fill_diagonal(D, 0.0)
        C = np.maximum(0.0, C + eps * D)
        C = (C + C.T) / 2