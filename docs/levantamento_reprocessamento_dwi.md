# Levantamento — reprocessar a difusão do MICA-MICs (nada foi baixado)

**Data:** 15/10/2026. Objetivo: reduzir o ruído da SC (lado mais fraco da identificação FC → SC).

## O que existe
- Derivados já baixados (`micapipe_MICs_v1.1.zip`): **só matrizes em texto** (SC, FC, GD, MPC, séries). Nenhuma imagem
  (sem DWI pré-processada, sem parcelação no espaço da difusão, sem superfícies). Verificado no índice do zip.
- Dados brutos: componente público do OSF **z8mu5** ("Raw data"), 10 arquivos `raw_HCxxx-HCyyy.tar.gz`, 4,44–4,68 GB
  cada, **~45 GB no total**, todas as modalidades (T1w, qT1, DWI, rs-fMRI) de 5 sujeitos por arquivo, BIDS, CC0.
- Dá para extrair só a DWI e o T1w em fluxo (`curl | tar -xz --wildcards`), sem guardar os .tar.gz — mas o download
  completo (~45 GB) é inevitável.

## Pipeline necessário por sujeito (estimativa nesta máquina: 8 núcleos no WSL, 5 GB de RAM)
| Etapa | Ferramenta | Tempo estimado |
|---|---|---|
| Parcelação cortical no espaço nativo | FreeSurfer recon-all (licença gratuita, exige cadastro do usuário) ou FastSurfer na GPU | 6–10 h (FreeSurfer) / ~1 h (FastSurfer) |
| Alternativa mais rápida e menos precisa | registro volumétrico do atlas Schaefer (MNI) para o T1 com ANTs | ~0,5 h |
| Pré-processamento da DWI | MRtrix3 dwidenoise, mrdegibbs; FSL topup/eddy (depende de haver b0 com fase invertida — não verificado) | 1–1,5 h |
| Modelo e tractografia | MRtrix3 dwi2response dhollander, msmt CSD, tckgen 10 M fibras, tcksift2 | 1–2 h |
| Conectoma | tck2connectome (pesos SIFT2 + comprimento) | minutos |
- **Total: ~3–4 h por sujeito sem FreeSurfer (≈ 150–200 h de CPU para 50)**; com 2–3 sujeitos em paralelo (limite
  de RAM do eddy), **~3–5 dias de processamento contínuo**.
- Instalações: FSL (~6 GB), MRtrix3 (~0,5 GB), ANTs (~1 GB), FastSurfer opcional. Disco: tractogramas de 10 M fibras
  têm 1–2 GB cada → apagar após gerar o conectoma ou reduzir fibras; ~50 GB livres hoje.

## Ganho esperado e limites
- Ganha: SIFT2, mais fibras, parâmetros sob nosso controle; permite medir o **ruído estocástico da própria
  tractografia** (duas tractografias com sementes diferentes do mesmo sujeito → auto-identificação da SC).
- Não ganha: não é reteste de aquisição (o ruído de aquisição continua desconhecido) e o n continua 50.
- Risco: dias de processamento para um ganho de identificação incerto.

## Comparação com o HCP
O HCP-YA oferece difusão já pré-processada, FreeSurfer pronto, ~1.000 sujeitos e **reteste de aquisição**; ainda exige
tractografia (horas por sujeito), mas mede diretamente o teto e dá poder estatístico. Depende do cadastro do usuário
no BALSA.

## Opção piloto (barata)
Baixar 1 lote (4,7 GB, 5 sujeitos), rodar a tractografia duas vezes por sujeito com sementes diferentes e medir a
auto-identificação da SC: quantifica o ruído da tractografia antes de decidir pelos 50.