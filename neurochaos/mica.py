"""Carregador das matrizes do MICA-MICs (micapipe v0.1, arquivos .txt).

Layout verificado nos arquivos (sub-HC001, Schaefer-200 e Desikan-Killiany):
- SC, edgeLength, FC: N_sub (14) + [parede medial E + córtex E] + [parede medial D + córtex D];
  só o triângulo superior é armazenado (SC/edgeLength com diagonal ≠ 0).
- timeseries: T × (mesmas colunas de FC).
- GD (geodésica) e MPC: só córtex, [parede medial E + córtex E] + [parede medial D + córtex D];
  GD vem preenchida (não simétrica exata), MPC só no triângulo superior.
Rótulos: lookup tables do micapipe v0.1.1 em data/derivatives/mica_mics/lut/.
"""

import csv
from pathlib import Path

import numpy as np

from .config import resolve

ROOT = "data/derivatives/mica_mics"
N_SUBCORTICAL = 14  # 7 E + 7 D (tálamo, caudado, putâmen, pálido, hipocampo, amígdala, accumbens)
LUT = {"schaefer200": "lut_schaefer-200_mics.csv", "schaefer400": "lut_schaefer-400_mics.csv",
       "aparc": "lut_aparc_mics.csv"}
KIND_DIR = {"sc": "dwi", "edgeLength": "dwi", "fc": "func", "timeseries": "func",
            "gd": "anat/geo_dist", "mpc": "anat/micro_profiles"}
KIND_SPACE = {"sc": "dwinative", "edgeLength": "dwinative", "fc": "fsnative", "timeseries": "fsnative",
              "gd": "fsnative", "mpc": "fsnative"}
WITH_SUBCORTICAL = ("sc", "edgeLength", "fc", "timeseries")


def subjects(root=ROOT):
    return sorted(p.name for p in (resolve(root) / "micapipe").glob("sub-HC*") if p.is_dir())


def path(sub, parc, kind, root=ROOT, ses="ses-01"):
    return (resolve(root) / "micapipe" / sub / ses / KIND_DIR[kind] /
            f"{sub}_{ses}_space-{KIND_SPACE[kind]}_atlas-{parc}_desc-{kind}.txt")


def _read(p):
    first = open(p, encoding="utf-8").readline()
    return np.loadtxt(p, delimiter="," if "," in first else None)


def cortical_labels(parc, root=ROOT):
    """Rótulos corticais na ordem das matrizes, com as paredes mediais marcadas."""
    with open(resolve(root) / "lut" / LUT[parc], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["label"].strip() for r in rows]


def subcortical_labels(root=ROOT):
    with open(resolve(root) / "lut" / "lut_subcortical-cerebellum_mics.csv", encoding="latin-1") as f:  # arquivo não é UTF-8
        rows = list(csv.DictReader(f))
    return [r["label"].strip() for r in rows[:N_SUBCORTICAL]]


def node_index(parc, kind, include_subcortical=False, root=ROOT):
    """Índices (no arquivo) e rótulos dos nós mantidos, sem paredes mediais."""
    ctx = cortical_labels(parc, root)
    off = N_SUBCORTICAL if kind in WITH_SUBCORTICAL else 0
    keep = [(off + i, lab) for i, lab in enumerate(ctx) if "medial_wall" not in lab.lower()]
    if include_subcortical and kind in WITH_SUBCORTICAL:
        keep = list(zip(range(N_SUBCORTICAL), subcortical_labels(root))) + keep
    idx, labels = zip(*keep)
    return np.array(idx), list(labels)


def load(sub, parc, kind, include_subcortical=False, root=ROOT):
    """Matriz N×N simétrica (ou T×N para timeseries) restrita aos nós válidos."""
    M = _read(path(sub, parc, kind, root))
    idx, labels = node_index(parc, kind, include_subcortical, root)
    expected = N_SUBCORTICAL * (kind in WITH_SUBCORTICAL) + len(cortical_labels(parc, root))
    ncol = M.shape[1]
    if ncol != expected:
        raise ValueError(f"{sub} {parc} {kind}: {ncol} colunas, esperado {expected}")
    if kind == "timeseries":
        return M[:, idx], labels
    if kind == "gd":
        M = (M + M.T) / 2.0
    else:
        M = np.triu(M, 1) + np.triu(M, 1).T  # triângulo superior → simétrica, diagonal zerada
    M = M[np.ix_(idx, idx)]
    if kind in ("gd", "sc", "edgeLength", "mpc"):
        np.fill_diagonal(M, 0.0)
    else:
        np.fill_diagonal(M, 1.0)
    return M, labels


def load_subject(sub, parc, kinds=("sc", "edgeLength", "fc", "gd", "mpc"), root=ROOT):
    """Todas as modalidades no MESMO conjunto de nós (só córtex, sem paredes mediais)."""
    out = {}
    labels = None
    for k in kinds:
        M, lab = load(sub, parc, k, include_subcortical=False, root=root)
        if labels is None:
            labels = lab
        elif lab != labels:
            raise ValueError(f"rótulos divergentes entre modalidades em {sub}/{parc}/{k}")
        out[k] = M
    out["labels"] = labels
    return out

def read_folds(path="data/manifests/subjects_split.csv"):
    with open(resolve(path), encoding="utf-8") as f:
        return {r["subject_id"]: int(r["fold"]) for r in csv.DictReader(f) if r["dataset"] == "mica_mics"}


def load_matrices(parc, root=ROOT):
    """{sujeito: {sc, fc, gd, mpc, labels}} em matrizes N×N alinhadas (só córtex)."""
    return {s: load_subject(s, parc, kinds=("sc", "fc", "gd", "mpc"), root=root) for s in subjects(root)}