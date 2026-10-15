"""Testes do carregador MICA-MICs (pulados se os dados não foram extraídos)."""

import numpy as np

from neurochaos import mica


def _have_data():
    return len(mica.subjects()) > 0


def test_layout_and_alignment():
    if not _have_data():
        return
    s = mica.subjects()[0]
    for parc, n in (("schaefer200", 200), ("aparc", 70)):
        d = mica.load_subject(s, parc)
        assert len(d["labels"]) == n
        assert not any("medial_wall" in lab.lower() for lab in d["labels"])
        for k in ("sc", "edgeLength", "fc", "gd", "mpc"):
            assert d[k].shape == (n, n)
            assert np.allclose(d[k], d[k].T)
        ts, lab = mica.load(s, parc, "timeseries")
        assert lab == d["labels"]
        iu = np.triu_indices(n, 1)
        assert np.max(np.abs(np.corrcoef(ts.T)[iu] - d["fc"][iu])) < 1e-4  # ordem dos nós correta


def test_subcortical_option():
    if not _have_data():
        return
    s = mica.subjects()[0]
    M, lab = mica.load(s, "aparc", "sc", include_subcortical=True)
    assert M.shape == (84, 84)
    assert lab[0] == "Left-Thalamus-Proper"
