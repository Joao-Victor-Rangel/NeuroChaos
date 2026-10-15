"""Corrente periódica I(t) = I_bias + Aθ sin(2π fθ t + φ) + Aγ sin(2π fγ t + ψ)  (guia §7)."""

import numpy as np


def periodic_current(t_ms, bias_nA, theta, gamma):
    t = np.asarray(t_ms, dtype=float) / 1000.0
    i = np.full_like(t, float(bias_nA))
    if theta["amp_nA"]:
        i += theta["amp_nA"] * np.sin(2 * np.pi * theta["freq_hz"] * t + theta["phase_rad"])
    if gamma["amp_nA"]:
        i += gamma["amp_nA"] * np.sin(2 * np.pi * gamma["freq_hz"] * t + gamma["phase_rad"])
    return i


def drive_phase(t_ms, comp):
    """Fase instantânea (rad, em (-π, π]) do componente senoidal do estímulo."""
    t = np.asarray(t_ms, dtype=float) / 1000.0
    return np.angle(np.exp(1j * (2 * np.pi * comp["freq_hz"] * t + comp["phase_rad"])))


def is_periodic(stim):
    return bool(stim["theta"]["amp_nA"] or stim["gamma"]["amp_nA"])
