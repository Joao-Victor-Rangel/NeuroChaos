"""Modulation Index de Tort et al. (2010): amplitude gamma condicionada à fase theta.

Surrogates por deslocamento temporal circular da amplitude em relação à fase.
Atenção a harmônicos de ondas não senoidais e a gamma injetada diretamente: PAC > 0
num cenário com gamma externo não demonstra acoplamento theta–gamma emergente.
"""

import numpy as np

from .sync import analytic, random_shifts


def modulation_index(phase, amp, n_bins=18):
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    if np.any(cnt == 0):
        return float("nan"), None
    mean_amp = np.bincount(idx, weights=amp, minlength=n_bins) / cnt
    if mean_amp.sum() <= 0:
        return float("nan"), mean_amp
    p = mean_amp / mean_amp.sum()
    p = np.clip(p, 1e-300, None)
    H = -np.sum(p * np.log(p))
    return float((np.log(n_bins) - H) / np.log(n_bins)), mean_amp


def pac_test(x, fs_hz, phase_band, amp_band, n_bins, n_surr, min_shift_s, edge_s, rng):
    phase = np.angle(analytic(x, fs_hz, phase_band, edge_s))
    amp = np.abs(analytic(x, fs_hz, amp_band, edge_s))
    mi, dist = modulation_index(phase, amp, n_bins)
    out = {"mi": mi}
    if n_surr > 0 and np.isfinite(mi):
        shifts = random_shifts(len(amp), n_surr, int(min_shift_s * fs_hz), rng)
        surr = np.array([modulation_index(phase, np.roll(amp, s), n_bins)[0] for s in shifts])
        sd = np.nanstd(surr)
        out.update({
            "z": float((mi - np.nanmean(surr)) / sd) if sd > 0 else float("nan"),
            "p": float((1 + np.sum(surr >= mi)) / (1 + n_surr)),
            "surr_mean": float(np.nanmean(surr)),
        })
    out["amp_by_phase"] = None if dist is None else dist.tolist()
    return out