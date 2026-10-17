# Pré-registro — Adendo 4: gêmeo digital funcional (identificação pelo modelo ajustado à fMRI)

Escrito em 17/10/2026, **antes** de qualquer resultado deste teste. Hash SHA-256 deste arquivo registrado no diário
(item 28) e gravado no JSON de resultados. Não altera nenhum resultado anterior.

## Motivação
Os testes anteriores mostraram que a FC (fMRI) é muito individual (teto FC→FC: 80 % em Schaefer-200, 92 % em
Schaefer-400), mas que prever a SC individual a partir da FC é fraco (12–30 %) e que a SC individual não muda as
simulações. Hipótese reformulada: o que é individual e recuperável é a **rede efetiva** ajustada à função. Se um
modelo mecanístico (Hopf linearizado; `neurochaos/ec.py`) ajustado à fMRI de uma pessoa reproduz e reconhece essa
pessoa, ele é um "gêmeo digital" funcional que pode depois receber perturbações (lesões virtuais, canais iônicos
no NEURON).

## Dados e pré-processamento (fixos)
MICA-MICs, 50 sujeitos, parcelação **Schaefer-200** (primária; a mesma do adendo D), TR = 0,6 s. BOLD dividido em
metades A e B; cada metade com filtro 0,01–0,1 Hz e frequências intrínsecas (0,04–0,07 Hz) próprias.

## Gêmeo (variante primária: "grupo")
C0 = 0,2·SC_grupo/max e máscara = (SC_grupo > 0), com SC_grupo = média micapipe dos sujeitos de **treino** do fold
(5 folds de `data/manifests/subjects_split.csv`). `fit_ec` com ε = 0,01, 100 iterações, escolhendo a melhor
iteração **na própria metade de ajuste** (emenda 1). Para cada sujeito ajustam-se dois gêmeos independentes:
EC_A (na FC_A) e EC_B (na FC_B). F_A = FC do modelo com EC_A.

## Representações e identificação
FC (empírica e do modelo): triângulo superior, z de Fisher (clip ±0,999). EC: log1p(EC/média) no triângulo
superior. Antes de correlacionar, remove-se a média entre sujeitos de cada aresta (como em `run_tetos.py`).
Acurácia A→B = fração de sujeitos cujo maior r é com eles mesmos (acaso 1/50).

## Testes primários (Holm sobre P1 e P2, α = 0,05; binomial unilateral contra 1/50)
- **P1 — o gêmeo reconhece a pessoa:** F_A(i) × FC_B(j).
- **P2 — os parâmetros do gêmeo são uma impressão digital confiável:** EC_A(i) × EC_B(j).

## Comparações descritivas pré-especificadas
- Teto FC_A → FC_B (80 %, já medido) ao lado de P1, com McNemar exato (espera-se gêmeo ≤ teto); razão de retenção
  = acurácia P1 / teto.
- r médio F_A × FC_A (ajuste) e F_A × FC_B (generalização), como no adendo D.
- Controle C1: gêmeo **sem ajuste** (C0 do grupo; só as frequências variam por sujeito) — espera-se ≈ acaso.

## Secundários (sem correção conjunta; relatados todos)
- **S1 "próprio":** C0/máscara a partir da SC micapipe do próprio sujeito. P1/P2 comparados com a variante "grupo"
  por McNemar exato.
- **S2 sensibilidade do ajuste:** ε = 0,002, 500 iterações (variante "grupo").
- **S3 Schaefer-400** (variante "grupo"), se o tempo permitir.
- **S4:** quando as 50 SCs reconstruídas (`results/dwi_mica50`) estiverem prontas, S1 com a SC reconstruída
  (Schaefer-400).

## Regras de interpretação (fixadas agora)
- P1 e P2 significativos → o gêmeo é pessoal e seus parâmetros são estáveis entre metades.
- P1 sim, P2 não → a saída é individual, mas os parâmetros não são identificáveis (problema mal posto); o gêmeo
  serve para prever, não para interpretar arestas.
- P1 não → o modelo linearizado com esta parametrização perde a individualidade da FC.
- Limitação declarada: A e B são metades da **mesma sessão**; confirmação exige reteste (HCP).