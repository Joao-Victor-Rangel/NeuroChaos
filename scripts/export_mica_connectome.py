"""Exporta conectomas do MICA-MICs no formato da topologia `atlas` (data/README.md).

Por sujeito e para a média do grupo, em data/connectomes/mica_<parc>/<id>/:
  weights.csv      SC (pesos da tractografia), linhas = origem, colunas = alvo (simétrica)
  distances_mm.csv comprimento médio das fibras (edgeLength) em mm; arestas sem fibra recebem a
                   mediana (irrelevante: P = 0 nelas), para que todo atraso seja > 0
  region_labels.tsv índice<TAB>rótulo

  python scripts/export_mica_connectome.py --parc aparc
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from neurochaos import mica
from neurochaos.config import resolve


def write(dst, sc, length, labels):
    dst.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst / "weights.csv", sc, delimiter=",", fmt="%.6g")
    off = ~np.eye(len(sc), dtype=bool)
    L = length.copy()
    L[off & (L <= 0)] = np.median(L[off & (L > 0)])
    np.fill_diagonal(L, 0.0)
    np.savetxt(dst / "distances_mm.csv", L, delimiter=",", fmt="%.4f")
    with open(dst / "region_labels.tsv", "w", encoding="utf-8") as f:
        f.write("index\tlabel\n")
        for i, lab in enumerate(labels):
            f.write(f"{i}\t{lab}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc", default="aparc")
    a = ap.parse_args()
    root = resolve("data/connectomes") / f"mica_{a.parc}"
    scs, lens = [], []
    for s in mica.subjects():
        d = mica.load_subject(s, a.parc, kinds=("sc", "edgeLength"))
        write(root / s, d["sc"], d["edgeLength"], d["labels"])
        scs.append(d["sc"])
        lens.append(d["edgeLength"])
        labels = d["labels"]
    L = np.array(lens)
    with np.errstate(invalid="ignore"):
        mean_len = np.where((L > 0).sum(0) > 0, L.sum(0) / np.maximum((L > 0).sum(0), 1), 0.0)
    write(root / "group", np.mean(scs, axis=0), mean_len, labels)
    print(f"{len(scs)} sujeitos + group exportados em {root} ({len(labels)} regiões)")


if __name__ == "__main__":
    main()