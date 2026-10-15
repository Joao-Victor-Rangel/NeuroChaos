# NeuroChaos — MVP de rede HH multirregional

> **Direção atual do projeto:** ver [docs/plano_v2_fmri_gan.md](docs/plano_v2_fmri_gan.md) — fMRI/T1 → GAN →
> conectividade estrutural → simulação, validado contra a difusão dos mesmos sujeitos (MICA-MICs, HCP).
> O simulador abaixo é o estágio final (escala rápida/caos) desse fluxo.

Implementação do **estágio MVP** do guia *NeuroChaos — dados, modelos e ajustes experimentais*
(§6–§8): 4 regiões sintéticas × (8 E + 2 I) neurônios Hodgkin–Huxley de um compartimento,
sinapses `Exp2Syn`, conectoma sintético por distância e seus controles, estímulo theta/gama via
`Vector.play` e análises de PLV, PAC, PSD e caos (trajetórias gêmeas) com validação prévia.

> **O que é observação e o que é hipótese.** Nada aqui vem de imagem. Regiões, coordenadas,
> conectividade local/longa, proporção E/I, fundo Poisson e parâmetros celulares são **hipóteses
> de modelo** (`configs/mvp.yaml`). Nenhum dataset é usado nem foi baixado.

## Ambiente

NEURON não roda nativamente via pip no Windows; tudo é executado no **WSL2**:

```bash
conda env create -f environment.yml   # ou reutilize um env com NEURON 8.2 + NetPyNE 1.0
conda activate neurochaos
cd "/mnt/c/Users/Joao Victor/Documents/Capitiria/NeuroChaos"
```

## Parte de dados reais (plano v2) — ambiente `neurochaos-ml` (PyTorch 2.6 + CUDA)

| Passo | Comando | Saída |
|---|---|---|
| Extrair MICA-MICs | `python scripts/prepare_mica.py` | `data/derivatives/mica_mics/micapipe/` |
| Integridade + folds | `python scripts/check_mica.py` | `data/manifests/` |
| Baselines FC→SC | `python scripts/run_baselines.py` | `results/baselines/` |
| GAN + ablação U-Net | `python scripts/run_gan.py --seeds 0 1 2 --epochs 300` | `results/gan/` |

## Ordem de execução (simulador NEURON, ambiente `neuron-env`)

| Passo | Comando | Saída |
|---|---|---|
| 1. Testes | `python tests/run_tests.py` (ou `pytest`) | 14 testes |
| 2. Validar estimadores de caos | `python scripts/validate_chaos.py` | `results/validation/` |
| 3. Calibrar (F–I, reobase, K_base) | `python scripts/calibrate.py --workers 4` | `configs/calibration_mvp.json` |
| 4. Convergência numérica (dt vs dt/2) | `python scripts/convergence_dt.py` | `results/convergence_dt/report.json` |
| 5. Execução isolada + figura | `python scripts/run_single.py --out results/single/x --set stimulus.theta.amp_nA=0.04` | `summary.png` |
| 6. Grade de depuração (4 s) | `python scripts/run_grid.py --grid configs/grid_debug.yaml --workers 4` | `results/grid_debug/` |
| 7. Análise | `python scripts/analyze.py --grid configs/grid_debug.yaml` | `results/grid_debug/analysis/` |
| 8. Grade científica (62 s) | idem com `configs/grid_science.yaml` | ~4–6 h em 4 núcleos |

Qualquer parâmetro pode ser sobrescrito com `--set chave.pontuada=valor` (ex.: `geometry.scale=1.2`,
`connectome.velocity_m_per_s=2`, `connectome.topology=shuffled`, `background.mode=none`).
A grade é retomável: jobs com `metrics.json` já existente são pulados.

## Desenho

- **Conectoma** (`neurochaos/connectome.py`): `W = exp(-d/ℓ)`, `P = clip(p0·W/max W, 0, pmax)`,
  `delay = d/v`. Controles: `shuffled` (arestas permutadas, mesma distribuição de pesos, simetria
  preservada; atrasos permanecem presos à distância física) e `homogeneous` (mesma média).
  `atlas` lê `data/connectomes/` (ver `data/README.md`). `geometry.scale` manipula a geometria macroscópica.
- **Rede** (`neurochaos/model.py`): NetPyNE monta células, conexões e estímulos; cada célula recebe
  `I(t) = I_bias + Aθ sin(2π fθ t + φ) + Aγ sin(2π fγ t + ψ)` via `Vector.play` em `IClamp.amp`
  (a corrente efetivamente entregue é gravada). Fundo Poisson **congelado** (mesma realização por semente).
- **Sementes separadas**: `seeds.conn` (conexões/embaralhamento), `seeds.stim` (ruído e I_bias), `seeds.loc`.
- **Métricas** (`neurochaos/metrics/`): PLV com surrogates por deslocamento circular; PLV ao estímulo com
  Rayleigh; PAC (MI de Tort) com surrogates; PSD de Welch; trajetórias gêmeas (λ linear + probabilidade
  de divergência de tamanho finito); seção estroboscópica na fase theta para diagramas de bifurcação.
- **Estatística** (`scripts/analyze.py`): média por condição com IC95% bootstrap sobre 10 sementes;
  contrastes `distance − shuffled` e `distance − homogeneous` por (K, Aθ) com permutação e Holm sobre
  as métricas primárias pré-especificadas (`analysis.primary_metrics`).

## Cuidados metodológicos já incorporados

- **Determinismo verificado em toda execução**: a referência é repetida e deve coincidir bit a bit
  (`chaos.determinism_max_abs_mV == 0`); antes do instante da perturbação, δ deve ser exatamente 0.
- **Dois regimes de instabilidade**: perturbações mínimas (δV ≈ 1e-5 mV) reconvergem (λ linear < 0),
  mas perturbações finitas (1e-3 a 1e-1 mV) podem inverter spikes e descorrelacionar a rede. O pipeline
  reporta `lambda_linear_per_s` e `p_diverge_*` separadamente; misturá-los numa média de ln δ é enganoso.
- **Nulos inválidos sob forçamento periódico**: com theta injetado, deslocar a série preserva o travamento
  e o MI. Nessas condições `analyze.py` descarta z/p de surrogate e a inferência vem da comparação entre
  condições/sementes (e o PLV ao estímulo usa Rayleigh). Documentado em `tests/test_metrics.py`.
- **Gama injetado ≠ gama emergente**: os controles só-gama e theta+gama estão na família `controls`.
- **dt**: taxas convergem (< 2 % entre dt e dt/2 em 20 s), mas os trens de spikes descorrelacionam em
  ~300 ms — consistente com a instabilidade de tamanho finito; compare estatísticas, não trajetórias.
- **Duração**: 4 s é só depuração; inferências de PAC/caos exigem a grade científica (60 s).

## Estrutura

```text
configs/      mvp.yaml (base), calibration_mvp.json (gerado), grid_debug.yaml, grid_science.yaml
neurochaos/   config, connectome, stimulus, model, signals, experiment, worker, runner, grid, metrics/
scripts/      fi_curve, calibrate, validate_chaos, convergence_dt, run_single, run_grid, analyze, plot_run
tests/        test_connectome.py, test_metrics.py, run_tests.py
data/         estrutura do guia §3 (não versionada) + manifests/datasets.csv
results/      saídas (não versionadas)
```