"""PLV (sincronização de fase entre sinais) com surrogates por deslocamento circular.

PLV mede fase–fase entre sinais na MESMA banda; não confundir com PAC (fase–amplitude
entre bandas). Entrada comum (o mesmo theta injetado em todas as regiões) infla o PLV:
compare sempre com o controle sem estímulo e com o PLV de cada região ao estímulo.
"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def bandpass(x, fs_hz, band, order=4):
    sos = butter(order, [band[0], band[1]], btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, x - np.mean(x))


def analytic(x, fs_hz, band, edge_s):
    """Sinal analítico da banda, com bordas descartadas após a filtragem."""
    z = hilbert(bandpass(x, fs_hz, band))
    e = int(round(edge_s * fs_hz))
    return z[e: len(z) - e] if e > 0 else z


def plv(ph1, ph2):
    return float(np.abs(np.mean(np.exp(1j * (ph1 - ph2)))))


def rayleigh_p(R, n):
    """p do teste de Rayleigh para resultante média R com n amostras independentes (Zar, 1999)."""
    n = max(float(n), 1.0)
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))
    return float(min(1.0, max(p, 0.0)))


def entrainment_test(ph_signal, ph_drive, n_cycles):
    """Travamento de fase ao estímulo periódico.

    Deslocamentos circulares NÃO servem de nulo aqui: deslocar uma senoide perfeita só gira a
    diferença de fase e preserva o PLV. Usa-se Rayleigh com N efetivo = número de ciclos
    (conservador) e, na análise, a comparação com o controle sem estímulo.
    """
    r = plv(ph_signal, ph_drive)
    return {"plv": r, "p_rayleigh": rayleigh_p(r, n_cycles), "n_cycles": float(n_cycles)}


def random_shifts(n, n_surr, min_shift, rng):
    if n <= 2 * min_shift:
        min_shift = n // 4
    return rng.integers(min_shift, n - min_shift, size=n_surr)


def plv_test(ph1, ph2, n_surr, min_shift, rng):
    """PLV entre sinais com nulo por deslocamento circular de ph2.

    Se ambos os sinais estão travados ao MESMO estímulo periódico, o deslocamento preserva
    esse travamento: z > 0 indica sincronia além da explicada pela periodicidade comum.
    """
    obs = plv(ph1, ph2)
    if n_surr <= 0:
        return {"plv": obs}
    d = np.exp(1j * ph1)
    surr = np.array([np.abs(np.mean(d * np.exp(-1j * np.roll(ph2, s))))
                     for s in random_shifts(len(ph2), n_surr, min_shift, rng)])
    sd = surr.std()
    return {
        "plv": obs,
        "z": float((obs - surr.mean()) / sd) if sd > 0 else float("nan"),
        "p": float((1 + np.sum(surr >= obs)) / (1 + n_surr)),
        "surr_mean": float(surr.mean()),
    }