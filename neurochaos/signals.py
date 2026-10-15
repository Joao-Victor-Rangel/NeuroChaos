"""Sinais agregados a partir de spikes e V (guia §8.1–8.2).

O passo de integração (dt) é diferente do passo do sinal (fs = 1 kHz por padrão).
Não há LFP sintético aqui: o guia exige método de LFP especificado e validado.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def time_axis(t0_ms, t1_ms, fs_hz):
    dt = 1000.0 / fs_hz
    return np.arange(t0_ms, t1_ms, dt)


def population_rate(spkt, spkid, gids, t0_ms, t1_ms, fs_hz, sigma_ms):
    """Taxa populacional (Hz/célula) em bins de 1/fs, suavizada por gaussiana de sigma_ms."""
    dt = 1000.0 / fs_hz
    edges = np.arange(t0_ms, t1_ms + dt / 2, dt)
    mask = np.isin(spkid, gids) & (spkt >= t0_ms) & (spkt < t1_ms)
    counts, _ = np.histogram(spkt[mask], bins=edges)
    rate = counts / (len(gids) * dt / 1000.0)
    if sigma_ms > 0:
        rate = gaussian_filter1d(rate.astype(float), sigma_ms / dt, mode="reflect")
    return rate


def cell_rates(spkt, spkid, gids, t0_ms, t1_ms):
    T = (t1_ms - t0_ms) / 1000.0
    m = (spkt >= t0_ms) & (spkt < t1_ms)
    ids = spkid[m]
    return np.array([np.count_nonzero(ids == g) / T for g in gids])


def isi_cv(spkt, spkid, gids, t0_ms, t1_ms, min_spikes=3):
    cvs = []
    m = (spkt >= t0_ms) & (spkt < t1_ms)
    for g in gids:
        st = np.sort(spkt[m & (spkid == g)])
        if len(st) >= min_spikes:
            isi = np.diff(st)
            cvs.append(isi.std() / isi.mean())
    return float(np.mean(cvs)) if cvs else float("nan")


def mean_v(vall, vall_gids, gids, t0_ms, t1_ms, dt_ms):
    idx = np.isin(vall_gids, gids)
    i0, i1 = int(round(t0_ms / dt_ms)), int(round(t1_ms / dt_ms))
    return vall[idx, i0:i1].mean(axis=0)


def spike_shape(t_ms, v, thr=0.0):
    """Duração do PA na meia-amplitude (ms) e potencial basal (mV) a partir de um traço."""
    up = np.where((v[:-1] < thr) & (v[1:] >= thr))[0]
    if len(up) < 1:
        return float("nan"), float(np.median(v))
    dt = t_ms[1] - t_ms[0]
    widths = []
    for k in up[: min(len(up), 20)]:
        a, b = max(0, k - int(5 / dt)), min(len(v), k + int(5 / dt))
        seg = v[a:b]
        base, peak = seg[: int(2 / dt)].min(), seg.max()
        half = base + (peak - base) / 2
        above = np.where(seg >= half)[0]
        if len(above):
            widths.append((above[-1] - above[0]) * dt)
    return float(np.median(widths)) if widths else float("nan"), float(np.percentile(v, 10))