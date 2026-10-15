"""Modelo de massa neural de Hopf (forma normal) para BOLD em repouso (Deco et al., 2017).

Cada região i é um oscilador perto da bifurcação de Hopf:
    dz_i/dt = (a + iω_i) z_i − |z_i|² z_i + G Σ_j C_ij (z_j − z_i) + σ η_i(t)
x_i = Re(z_i) é comparado ao BOLD filtrado. ω_i vem do pico espectral do BOLD empírico da região
(0,04–0,07 Hz). C = SC escalada para máximo 0,2. G (acoplamento global) é o único parâmetro ajustado.

Limitação: o Hopf modela diretamente o sinal lento filtrado (não há modelo hemodinâmico
Balloon–Windkessel); é a prática padrão para FC de repouso, mas não descreve a atividade neural rápida.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def bandpass(ts, tr, lo=0.01, hi=0.1, order=2):
    sos = butter(order, [lo, hi], btype="bandpass", fs=1.0 / tr, output="sos")
    return sosfiltfilt(sos, ts - ts.mean(0), axis=0)


def fc(ts):
    return np.corrcoef(ts.T)


def intrinsic_freqs(ts, tr, band=(0.04, 0.07)):
    """Frequência de pico por região no BOLD filtrado em `band` (Hz)."""
    x = bandpass(ts, tr, *band)
    f, p = welch(x, fs=1.0 / tr, nperseg=min(len(x), 128), axis=0)
    m = (f >= band[0]) & (f <= band[1])
    return f[m][np.argmax(p[m], axis=0)]


def coupling(sc, transform="raw"):
    """C escalada para máximo 0,2. `log` = log1p(SC/média): a SC do micapipe tem cauda muito pesada
    (máx ~1e5 × mediana), e na escala linear quase todas as arestas ficam ≈ 0."""
    C = np.array(sc, float)
    np.fill_diagonal(C, 0.0)
    C = (C + C.T) / 2
    if transform == "log":
        off = ~np.eye(len(C), dtype=bool)
        C = np.log1p(C / C[off].mean())
        np.fill_diagonal(C, 0.0)
    return 0.2 * C / C.max()


def simulate(C, omega_hz, G, n_tr, tr, a=-0.02, sigma=0.02, dt=0.1, transient_s=60.0, seed=0):
    """Euler–Maruyama em lote: G é um vetor (B,). Retorna x amostrado no TR, forma (B, n_tr, N)."""
    rng = np.random.default_rng(seed)
    G = np.atleast_1d(np.asarray(G, float))[:, None]
    N = len(C)
    w = 2 * np.pi * np.asarray(omega_hz)[None, :]
    deg = C.sum(1)[None, :]
    x = 0.1 * rng.standard_normal((len(G), N))
    y = 0.1 * rng.standard_normal((len(G), N))
    every = int(round(tr / dt))
    n_tr_steps = n_tr * every
    n_trans = int(round(transient_s / dt))
    out = np.empty((len(G), n_tr, N))
    sq = np.sqrt(dt) * sigma
    np.seterr(over="ignore", invalid="ignore")  # G alto pode divergir; a FC vira NaN e o G é descartado
    for k in range(n_trans + n_tr_steps):
        r2 = x * x + y * y
        cx = x @ C.T - deg * x
        cy = y @ C.T - deg * y
        dx = (a - r2) * x - w * y + G * cx
        dy = (a - r2) * y + w * x + G * cy
        x = x + dt * dx + sq * rng.standard_normal(x.shape)
        y = y + dt * dy + sq * rng.standard_normal(y.shape)
        j = k - n_trans
        if j >= 0 and j % every == 0:
            out[:, j // every] = x
    return out


def fit_score(fc_sim, fc_emp):
    iu = np.triu_indices(len(fc_emp), 1)
    return float(np.corrcoef(fc_sim[iu], fc_emp[iu])[0, 1])


def sim_fc_grid(C, omega, G_grid, n_tr, tr, n_rep=2, seed=0):
    """FC simulada (média de n_rep realizações de ruído) para cada G da grade: (B, N, N)."""
    acc = None
    for r in range(n_rep):
        ts = simulate(C, omega, G_grid, n_tr, tr, seed=seed + 1000 * r)
        with np.errstate(all="ignore"):
            f = np.array([fc(bandpass(t, tr)) if np.all(np.isfinite(t)) else np.full((t.shape[1],) * 2, np.nan)
                          for t in ts])
        acc = f if acc is None else acc + f
    return acc / n_rep