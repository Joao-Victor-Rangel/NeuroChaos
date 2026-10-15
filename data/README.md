# data/

Estrutura do guia (§3). **Nada aqui é versionado** além deste arquivo e do cabeçalho do manifesto
(`.gitignore` exclui `raw/`, `processed/`, `atlases/`, `electrophysiology/` e os CSV/TSV de conectoma).

```text
data/
├── raw/{cq500_ct, seg_cq500, ct_ich, ixi_mri, hcp_dwi}/
├── atlases/{mni152, harvard_oxford}/
├── electrophysiology/{allen_nwb, morphology_swc, modeldb_mod}/
├── processed/{ct_nifti, ct_masks, atlas_registered, gan_pairs}/
├── connectomes/          # weights.csv, distances_mm.csv, region_labels.tsv
└── manifests/datasets.csv
```

O MVP atual **não usa nenhum dataset**: regiões e conectividade são sintéticas (`configs/mvp.yaml`).

## Plugar um conectoma de atlas (topologia `atlas`)

Coloque em `data/connectomes/` (mesma ordem de regiões em linhas, colunas e rótulos):

- `weights.csv` — matriz N×N sem cabeçalho (ex.: `tck2connectome`), linhas = origem, colunas = alvo;
- `distances_mm.csv` — N×N, comprimento médio das trajetórias (`tck2connectome -scale_length -stat_edge mean`);
- `region_labels.tsv` — `índice<TAB>rótulo`, uma linha por região.

Depois rode com `--set connectome.topology=atlas`. Contagem de streamlines ≠ número de axônios ≠ força
sináptica: teste normalizações alternativas e relate o viés da tractografia. Uma rede assim é
*atlas-informada*, não personalizada quanto à conectividade.