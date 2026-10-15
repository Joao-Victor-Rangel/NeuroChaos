"""Conectividade inter-regional (guia §6).

Convenção: matrizes N×N com linhas = região de origem, colunas = região alvo.
W_ij = exp(-d_ij/ell); P_ij = clip(p0·W_ij/max(W), 0, pmax); delay_ij = d_ij / v.
O conectoma macroscópico fornece pesos entre regiões, não sinapses entre células.
"""

import csv

import numpy as np

from .config import resolve

TOPOLOGIES = ("distance", "shuffled", "homogeneous", "atlas")


def region_table(cfg):
    regs = cfg["geometry"]["regions"]
    labels = [r["label"] for r in regs]
    xyz = np.array([r["xyz_mm"] for r in regs], dtype=float)
    # Escala anatômica isotrópica em torno do centróide (manipulação de geometria macroscópica).
    c = xyz.mean(axis=0)
    xyz = c + cfg["geometry"].get("scale", 1.0) * (xyz - c)
    return labels, xyz


def distance_matrix(xyz):
    return np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)


def distance_weights(D, ell_mm):
    W = np.exp(-D / ell_mm)
    np.fill_diagonal(W, 0.0)
    return W


def to_probabilities(W, p0, pmax):
    off = ~np.eye(len(W), dtype=bool)
    wmax = W[off].max()
    P = np.zeros_like(W) if wmax <= 0 else np.clip(p0 * W / wmax, 0.0, pmax)
    np.fill_diagonal(P, 0.0)
    return P


def delays_ms(D, velocity_m_per_s):
    # 1 m/s = 1 mm/ms
    return D / velocity_m_per_s


def _is_symmetric(M):
    return np.allclose(M, M.T)


def shuffle_edges(M, rng, max_tries=100):
    """Permuta as arestas fora da diagonal preservando a distribuição de pesos.

    Matrizes simétricas são embaralhadas no triângulo superior e espelhadas (grau binário e
    multiconjunto de pesos preservados). Rejeita permutações idênticas à original.
    """
    n = len(M)
    sym = _is_symmetric(M)
    idx = np.triu_indices(n, 1) if sym else np.where(~np.eye(n, dtype=bool))
    vals = M[idx]
    for _ in range(max_tries):
        out = np.zeros_like(M)
        out[idx] = vals[rng.permutation(len(vals))]
        if sym:
            out = out + out.T
        if not np.allclose(out, M) or np.allclose(vals, vals[0]):
            return out
    return out


def homogeneous(M):
    off = ~np.eye(len(M), dtype=bool)
    out = np.where(off, M[off].mean(), 0.0)
    return out


def load_atlas(atlas_dir):
    """Lê weights.csv, distances_mm.csv e region_labels.tsv (saída de tck2connectome + atlas)."""
    d = resolve(atlas_dir)
    W = np.loadtxt(d / "weights.csv", delimiter=",")
    D = np.loadtxt(d / "distances_mm.csv", delimiter=",")
    with open(d / "region_labels.tsv", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if r]
    labels = [r[-1] for r in rows if not r[0].lower().startswith("index")]
    if W.shape != D.shape or W.shape[0] != W.shape[1] or W.shape[0] != len(labels):
        raise ValueError(f"Conectoma inconsistente: W{W.shape}, D{D.shape}, {len(labels)} rótulos")
    np.fill_diagonal(W, 0.0)
    return labels, W, D


def build_connectome(cfg):
    cc = cfg["connectome"]
    topo = cc["topology"]
    if topo not in TOPOLOGIES:
        raise ValueError(f"topologia desconhecida: {topo}")
    rng = np.random.default_rng([cfg["seeds"]["conn"], 7919])

    # A fonte (sintética por distância ou atlas/conectoma real) é a base também dos controles
    # `shuffled` e `homogeneous`: assim os controles preservam a distribuição de pesos da fonte real.
    source = "atlas" if topo == "atlas" else cc.get("source", "synthetic")
    if source == "atlas":
        labels, W, D = load_atlas(cc["atlas_dir"])
        xyz = None
        D = D * cfg["geometry"].get("scale", 1.0)
        if cc.get("atlas_transform", "log") == "log":
            # contagens de streamlines têm cauda pesada: log1p(W/média) evita que poucas arestas dominem P
            off = ~np.eye(len(W), dtype=bool)
            W = np.where(off, np.log1p(W / W[off].mean()), 0.0)
    else:
        labels, xyz = region_table(cfg)
        D = distance_matrix(xyz)
        W = distance_weights(D, cc["ell_mm"])

    P_ref = to_probabilities(W, cc["p0"], cc["pmax"])
    if topo == "shuffled":
        P = shuffle_edges(P_ref, rng)
    elif topo == "homogeneous":
        P = homogeneous(P_ref)
    else:
        P = P_ref
    # Atrasos ficam presos à geometria física (d_ij), mesmo quando os pesos são embaralhados:
    # isola o efeito da topologia de pesos do efeito da condução.
    De = D if not cc.get("shuffle_delays", False) else shuffle_edges(D, rng)
    delay = delays_ms(De, cc["velocity_m_per_s"])
    np.fill_diagonal(delay, 0.0)
    return {
        "topology": topo,
        "source": source,
        "labels": labels,
        "xyz_mm": xyz,
        "distance_mm": D,
        "W": W,
        "P": P,
        "delay_ms": delay,
    }


def validate(conn):
    P, D = conn["P"], conn["delay_ms"]
    n = len(conn["labels"])
    assert P.shape == (n, n) and D.shape == (n, n), "matriz não quadrada / rótulos inconsistentes"
    assert np.all(np.diag(P) == 0), "diagonal deve ser zero (conexões locais são tratadas à parte)"
    assert np.all((P >= 0) & (P <= 1)), "probabilidades fora de [0,1]"
    assert np.all(D[~np.eye(n, dtype=bool)] > 0), "atrasos inter-regionais devem ser > 0"
    return True
