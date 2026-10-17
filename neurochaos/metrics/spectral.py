"""PSD por Welch: reportar resolução, bandas e normalização (guia §8.2)."""

import numpy as np
from scipy.signal import welch

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renomeou trapz


def psd(x, fs_hz, nperseg_s):
    nper = int(min(len(x), round(nperseg_s * fs_hz)))
    f, p = welch(x - np.mean(x), fs=fs_hz, nperseg=nper, noverlap=nper // 2, detrend="constant")
    return f, p, fs_hz / nper


def band_power(f, p, band):
    m = (f >= band[0]) & (f <= band[1])
    return float(_trapz(p[m], f[m])) if m.any() else float("nan")


def summarize(x, fs_hz, nperseg_s, bands, fmax=200.0):
    f, p, res = psd(x, fs_hz, nperseg_s)
    total = band_power(f, p, (1.0, fmax))
    out = {"resolution_hz": res}
    m = (f >= 1.0) & (f <= fmax)
    out["peak_hz"] = float(f[m][np.argmax(p[m])]) if m.any() else float("nan")
    for name, b in bands.items():
        bp = band_power(f, p, b)
        out[f"power_{name}"] = bp
        out[f"relpower_{name}"] = bp / total if total > 0 else float("nan")
        mb = (f >= b[0]) & (f <= b[1])
        out[f"peak_{name}_hz"] = float(f[mb][np.argmax(p[mb])]) if mb.any() else float("nan")
    return out, f, p