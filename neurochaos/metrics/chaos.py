"""Diagnósticos de caos determinístico (guia §8.3).

1. Trajetórias gêmeas: referência vs. perturbação mínima aplicada APÓS o transiente,
   com a MESMA realização de ruído (ruído congelado). A taxa de crescimento de ln δ(t)
   estima o maior expoente de Lyapunov condicional (λ_max, em 1/s).
2. Rosenstein (1993) em séries temporais — usado só na validação: ele também dá λ > 0
   para ruído, o que ilustra por que espectro largo/divergência não é prova de caos.
3. Seção estroboscópica (Poincaré) na fase theta, mapa de retorno de ISIs e máximos
   locais para diagramas de bifurcação.

Os estimadores são validados primeiro em Lorenz (λ ≈ 0,906) e num ciclo-limite (λ ≈ 0).
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

LORENZ_LAMBDA = 0.9056


# ----------------------------------------------------------------- sistemas de referência
def lorenz_rhs(s, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = s
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def van_der_pol_rhs(s, mu=1.0):
    x, y = s
    return np.array([y, mu * (1 - x * x) * y - x])


def integrate_rk4(f, x0, dt, n):
    out = np.empty((n, len(x0)))
    x = np.array(x0, dtype=float)
    for k in range(n):
        out[k] = x
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


# ----------------------------------------------------------------- trajetórias gêmeas
def divergence(ref, pert):
    """δ(t) = ||x_ref(t) − x_pert(t)||₂ sobre todas as variáveis (linhas = variáveis)."""
    return np.sqrt(np.sum((np.asarray(ref) - np.asarray(pert)) ** 2, axis=0))


def log_envelope(delta, win, floor=1e-300):
    """ln do envelope de δ (máximo móvel): suprime oscilações rápidas causadas por spikes."""
    env = maximum_filter1d(np.maximum(delta, floor), size=max(1, int(win)), mode="nearest")
    return np.log(env)


def fit_growth(tau, L, skip, growth_thresh=3.0, sat_frac=0.7, min_points=10, floor=None):
    """Ajusta a inclinação de L(τ) = ln δ na região de crescimento exponencial.

    - Se L cresce mais de `growth_thresh` (unidades de ln), ajusta de `skip` até atingir
      `sat_frac` do caminho até o platô de saturação (mediana dos últimos 20%).
    - Caso contrário (estável/neutro), ajusta em todo o horizonte após `skip`.
    - `floor` (em ln): pontos depois que L atinge o piso numérico são excluídos, senão o
      platô de arredondamento achata a inclinação de trajetórias que convergem.
    Retorna (λ, r², τ_ini, τ_fim, saturou).
    """
    tau, L = np.asarray(tau), np.asarray(L)
    m0 = tau >= skip
    if floor is not None:
        hit = np.where(m0 & (L <= floor))[0]
        if len(hit):
            m0 &= tau < tau[hit[0]]
    if m0.sum() < min_points:
        return float("nan"), float("nan"), float("nan"), float("nan"), False
    L0 = L[m0][: max(3, min_points // 2)].mean()
    plateau = np.median(L[int(0.8 * len(L)):])
    grew = plateau - L0 > growth_thresh
    if grew:
        target = L0 + sat_frac * (plateau - L0)
        idx = np.where(m0 & (L >= target))[0]
        end = tau[idx[0]] if len(idx) else tau[-1]
        m = m0 & (tau <= end)
        if m.sum() < min_points:
            m = m0 & (tau <= tau[np.where(m0)[0][0] + min_points])
    else:
        m = m0
    x, y = tau[m], L[m]
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (res[0] / ss) if len(res) and ss > 0 else float("nan")
    return float(coef[0]), float(r2), float(x[0]), float(x[-1]), bool(grew)


def ensemble_lyapunov(curves, dt, skip, **kw):
    """curves: lista de L(τ) (mesmo comprimento). Retorna λ da média e λ individuais."""
    curves = np.asarray(curves)
    tau = np.arange(curves.shape[1]) * dt
    mean_L = curves.mean(axis=0)
    lam, r2, a, b, grew = fit_growth(tau, mean_L, skip, **kw)

    indiv = [fit_growth(tau, c, skip, **kw)[0] for c in curves]
    return {"lambda": lam, "r2": r2, "fit_from": a, "fit_to": b, "saturated": grew,
            "lambda_individual": indiv, "tau": tau, "mean_L": mean_L}


def twin_lorenz(n_pairs=20, dt=0.01, horizon=25.0, eps=1e-8, seed=0, transient=50.0):
    """Validação do estimador de trajetórias gêmeas no atrator de Lorenz."""
    rng = np.random.default_rng(seed)
    x = integrate_rk4(lorenz_rhs, [1.0, 1.0, 1.0], dt, int(transient / dt))[-1]
    n = int(horizon / dt)
    curves = []
    for _ in range(n_pairs):
        x = integrate_rk4(lorenz_rhs, x, dt, int(rng.uniform(1, 5) / dt))[-1]
        d = rng.standard_normal(3)
        ref = integrate_rk4(lorenz_rhs, x, dt, n).T
        per = integrate_rk4(lorenz_rhs, x + eps * d / np.linalg.norm(d), dt, n).T
        curves.append(log_envelope(divergence(ref, per), win=int(0.5 / dt)))
    return ensemble_lyapunov(curves, dt, skip=0.5)


def twin_van_der_pol(n_pairs=10, dt=0.01, horizon=25.0, eps=1e-8, seed=0):
    rng = np.random.default_rng(seed)
    x = integrate_rk4(van_der_pol_rhs, [2.0, 0.0], dt, int(50 / dt))[-1]
    n = int(horizon / dt)
    curves = []
    for _ in range(n_pairs):
        x = integrate_rk4(van_der_pol_rhs, x, dt, int(rng.uniform(1, 5) / dt))[-1]
        d = rng.standard_normal(2)
        ref = integrate_rk4(van_der_pol_rhs, x, dt, n).T
        per = integrate_rk4(van_der_pol_rhs, x + eps * d / np.linalg.norm(d), dt, n).T
        curves.append(log_envelope(divergence(ref, per), win=int(0.5 / dt)))
    return ensemble_lyapunov(curves, dt, skip=0.5)


# ----------------------------------------------------------------- Rosenstein
def delay_embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    return np.stack([x[i * tau: i * tau + n] for i in range(m)], axis=1)


def rosenstein(x, dt, m, tau, theiler, k_max, fit_range, n_ref=800, seed=0):
    """Maior expoente de Lyapunov de uma série escalar (Rosenstein et al., 1993)."""
    x = (np.asarray(x, float) - np.mean(x)) / (np.std(x) + 1e-300)
    Y = delay_embed(x, m, tau)
    N = len(Y) - k_max
    rng = np.random.default_rng(seed)
    refs = rng.choice(N, size=min(n_ref, N), replace=False)
    logs = np.zeros(k_max)
    counts = np.zeros(k_max)
    for i in refs:
        d = np.linalg.norm(Y[:N] - Y[i], axis=1)
        d[max(0, i - theiler): i + theiler + 1] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            continue
        dk = np.linalg.norm(Y[i: i + k_max] - Y[j: j + k_max], axis=1)
        ok = dk > 0
        logs[ok] += np.log(dk[ok])
        counts[ok] += 1
    curve = logs / np.maximum(counts, 1)
    k = np.arange(k_max) * dt
    m_ = (k >= fit_range[0]) & (k <= fit_range[1])
    slope = np.polyfit(k[m_], curve[m_], 1)[0]
    return float(slope), k, curve


# ----------------------------------------------------------------- seções / mapas
def stroboscopic(t_ms, x, freq_hz, phase_rad=0.0):
    """Amostra x a cada período do forçamento, na fase 0 do seno (seção de Poincaré)."""
    t_ms = np.asarray(t_ms)
    period = 1000.0 / freq_hz
    t0 = (-phase_rad / (2 * np.pi)) * period
    k0 = int(np.ceil((t_ms[0] - t0) / period))
    k1 = int(np.floor((t_ms[-1] - t0) / period))
    ts = t0 + np.arange(k0, k1 + 1) * period
    return ts, np.interp(ts, t_ms, x)


def local_maxima(x, smooth=1):
    xs = uniform_filter1d(np.asarray(x, float), max(1, int(smooth)))
    pk, _ = find_peaks(xs)
    return xs[pk]


def isi_return_map(spike_times):
    isi = np.diff(np.sort(spike_times))
    return isi[:-1], isi[1:]