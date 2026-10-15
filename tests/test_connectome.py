import numpy as np

from neurochaos import connectome as cx
from neurochaos.config import load_config

try:
    import pytest

    fixture = pytest.fixture
except ImportError:  # executor mínimo em tests/run_tests.py
    def fixture(f):
        return f


@fixture
def cfg():
    return load_config("configs/mvp.yaml")


def test_distance_connectome_properties(cfg):
    c = cx.build_connectome(cfg)
    assert cx.validate(c)
    n = len(c["labels"])
    assert c["P"].shape == (n, n)
    assert np.allclose(c["P"], c["P"].T)
    assert np.isclose(c["P"].max(), cfg["connectome"]["p0"])  # normalizado pelo máximo
    # delay = d / v
    off = ~np.eye(n, dtype=bool)
    assert np.allclose(c["delay_ms"][off], c["distance_mm"][off] / cfg["connectome"]["velocity_m_per_s"])


def test_shuffle_preserves_weight_multiset_and_symmetry(cfg):
    cfg["connectome"]["topology"] = "distance"
    P = cx.build_connectome(cfg)["P"]
    S = cx.shuffle_edges(P, np.random.default_rng(3))
    iu = np.triu_indices(len(P), 1)
    assert np.allclose(np.sort(P[iu]), np.sort(S[iu]))
    assert np.allclose(S, S.T)
    assert not np.allclose(S, P)
    assert np.all(np.diag(S) == 0)


def test_homogeneous_keeps_mean(cfg):
    P = cx.build_connectome(cfg)["P"]
    H = cx.homogeneous(P)
    off = ~np.eye(len(P), dtype=bool)
    assert np.isclose(H[off].mean(), P[off].mean())
    assert np.allclose(H[off], H[off][0])


def test_shuffled_delays_stay_with_geometry(cfg):
    cfg["connectome"]["topology"] = "shuffled"
    c = cx.build_connectome(cfg)
    off = ~np.eye(len(c["P"]), dtype=bool)
    assert np.allclose(c["delay_ms"][off], c["distance_mm"][off] / cfg["connectome"]["velocity_m_per_s"])


def test_geometry_scale_scales_distances(cfg):
    d1 = cx.build_connectome(cfg)["distance_mm"]
    cfg["geometry"]["scale"] = 1.2
    d2 = cx.build_connectome(cfg)["distance_mm"]
    assert np.allclose(d2, 1.2 * d1)