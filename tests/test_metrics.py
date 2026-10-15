import numpy as np

from neurochaos.metrics import chaos, pac, spectral, sync
from neurochaos.stimulus import drive_phase, periodic_current

FS = 1000.0


def _t(sec):
    return np.arange(0, sec, 1 / FS)


def test_plv_locked_vs_independent():
    rng = np.random.default_rng(0)
    t = _t(20)
    base = np.sin(2 * np.pi * 6 * t)
    x = base + 0.3 * rng.standard_normal(len(t))
    y = np.sin(2 * np.pi * 6 * t + 0.7) + 0.3 * rng.standard_normal(len(t))
    # sinal independente: fase theta de ruído filtrado
    w = rng.standard_normal(len(t))
    px = np.angle(sync.analytic(x, FS, (4, 8), 0.5))
    py = np.angle(sync.analytic(y, FS, (4, 8), 0.5))
    pw = np.angle(sync.analytic(w, FS, (4, 8), 0.5))
    assert sync.plv(px, py) > 0.9
    assert sync.plv(px, pw) < 0.3


def test_rayleigh_p():
    assert sync.rayleigh_p(0.9, 100) < 1e-10
    assert sync.rayleigh_p(0.05, 100) > 0.5


def _noisy_theta(rng, n):
    # theta não estacionário (ruído filtrado em 4–8 Hz): os surrogates por deslocamento são válidos
    th = sync.bandpass(rng.standard_normal(n), FS, (4, 8))
    return th / th.std()


def test_pac_detects_coupling_not_independent():
    rng = np.random.default_rng(1)
    t = _t(60)
    th = _noisy_theta(rng, len(t))
    g = np.sin(2 * np.pi * 40 * t)
    coupled = th + (1 + 0.6 * np.tanh(th)) * 0.3 * g + 0.1 * rng.standard_normal(len(t))
    indep = th + 0.3 * g + 0.1 * rng.standard_normal(len(t))
    a = pac.pac_test(coupled, FS, (4, 8), (35, 45), 18, 100, 1.0, 0.5, rng)
    b = pac.pac_test(indep, FS, (4, 8), (35, 45), 18, 100, 1.0, 0.5, rng)
    assert a["mi"] > 10 * b["mi"]
    assert a["p"] < 0.05
    assert b["p"] > 0.01


def test_shift_surrogates_invalid_for_strictly_periodic_phase():
    """Documenta a limitação: com theta estritamente periódico, deslocar a amplitude só gira a
    fase preferida e o MI é preservado — o nulo por deslocamento não rejeita nem PAC verdadeiro."""
    rng = np.random.default_rng(2)
    t = _t(30)
    th = np.sin(2 * np.pi * 6 * t)
    coupled = th + (1 + 0.8 * th) * 0.3 * np.sin(2 * np.pi * 40 * t)
    a = pac.pac_test(coupled, FS, (4, 8), (35, 45), 18, 100, 1.0, 0.5, rng)
    assert a["mi"] > 1e-3
    assert a["p"] > 0.05

def test_psd_peak():
    t = _t(10)
    s, _, _ = spectral.summarize(np.sin(2 * np.pi * 40 * t), FS, 2.0, {"gamma": (35, 45)})
    assert abs(s["peak_hz"] - 40) <= s["resolution_hz"]
    assert s["relpower_gamma"] > 0.9


def test_stimulus_waveform_and_phase():
    t = np.arange(0, 1000, 0.5)
    th = {"amp_nA": 0.04, "freq_hz": 6.0, "phase_rad": 0.0}
    ga = {"amp_nA": 0.0, "freq_hz": 40.0, "phase_rad": 0.0}
    i = periodic_current(t, 0.01, th, ga)
    assert np.isclose(i.mean(), 0.01, atol=1e-3)
    assert np.isclose(i.max(), 0.05, atol=1e-4)
    ph = drive_phase(t, th)
    assert np.allclose(np.sin(ph), np.sin(2 * np.pi * 6 * t / 1000))


def test_twin_lorenz_and_periodic():
    lz = chaos.twin_lorenz(n_pairs=8, seed=2)
    assert abs(lz["lambda"] - chaos.LORENZ_LAMBDA) / chaos.LORENZ_LAMBDA < 0.2
    vdp = chaos.twin_van_der_pol(n_pairs=5, seed=2)
    assert abs(vdp["lambda"]) < 0.1


def test_fit_growth_respects_floor():
    tau = np.arange(0, 100, 1.0)
    L = np.maximum(-5 - 0.5 * tau, -23.0)  # decai até o piso de float64
    lam, *_ = chaos.fit_growth(tau, L, skip=1.0, floor=-22.9)
    assert np.isclose(lam, -0.5, rtol=0.05)


def test_stroboscopic_periodic_signal_is_constant():
    t = np.arange(0, 5000, 1.0)
    x = np.sin(2 * np.pi * 6 * t / 1000)
    _, s = chaos.stroboscopic(t, x, 6.0, 0.0)
    assert np.ptp(s) < 0.05