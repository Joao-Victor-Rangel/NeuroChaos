# NeuroChaos v2 — BOLD/anatomia → GAN → conectividade → simulação

**Data:** 14/10/2026. Substitui a entrada por TC do guia v1: a TC não contém informação funcional e
quase não distingue substância cinzenta/branca; o fMRI (BOLD) mede atividade regional ao longo do tempo.

## 1. Pergunta e fluxo

> A partir de exames de **um indivíduo** (T1 + fMRI em repouso), uma GAN consegue inferir a
> **conectividade estrutural** entre áreas cerebrais dessa pessoa — e a rede inferida, simulada,
> reproduz a **fisiologia** (BOLD) observada?

```text
T1 (anatomia) ─┐
               ├─► GAN condicional ─► SC prevista ─► modelo de massa neural + BOLD ─► FC simulada
BOLD (função) ─┘   (FC → SC)              │                                             │
                                          ▼                                             ▼
                          comparar com SC da difusão              comparar com FC empírica
                          DO MESMO SUJEITO (gabarito)             DO MESMO SUJEITO
```

Siglas: **FC** = conectividade funcional (correlação entre séries BOLD); **SC** = conectividade
estrutural (tractografia de difusão); **GD** = distância geodésica cortical; **MPC** = covariância de
perfis microestruturais.

## 2. Datasets — o que foi verificado

### 2.1 MICA-MICs — **começar por aqui** (verificado)

| Item | Valor |
|---|---|
| Sujeitos | 50 adultos saudáveis (23 mulheres; 29,5 ± 5,6 anos) |
| Anatomia | T1w e T1 quantitativo (qT1) a 3 T |
| Difusão | 3 shells: b = 300/700/2000 s/mm², 10/40/90 direções |
| fMRI repouso | 1 corrida de 7 min, TR = 600 ms, ~700 volumes |
| Já processado (micapipe) | matrizes **FC, SC, MPC, GD** por sujeito, em **18 parcelações** (70 a 1000 nós; Desikan-Killiany, Destrieux, Schaefer 100–1000, Glasser 360…); FC/SC incluem subcórtex e hipocampo; arquivos `.txt` |
| Brutos | BIDS (NIfTI + JSON); no OSF em lotes de 5 sujeitos |
| Onde | OSF `osf.io/j532r` e CONP `portal.conp.ca` |
| Licença | **CC0 1.0** (arquivo `license.txt` no OSF); citar Royer et al., *Sci Data* 2022 |

Por que é a melhor primeira fonte: as **três** medidas existem para **as mesmas 50 pessoas** e as matrizes
já estão prontas — dá para treinar/validar sem pré-processar imagem.

Limitação: 50 sujeitos é pouco para uma GAN → validação cruzada por sujeito e baselines fortes obrigatórios.

### 2.2 HCP Young Adult — fase de escala (verificado parcialmente)

| Item | Valor |
|---|---|
| Acesso | **BALSA** (`balsa.wustl.edu`): criar conta (de preferência e-mail institucional) → aba ConnectomeDB → aceitar *Data Use Terms* do "HCP-Young Adult 2025". **Você precisa fazer isso pessoalmente.** |
| Dados | MRI processado de 1.071 sujeitos; brutos de 1.113; 889 com as 4 modalidades 3T completas (incluindo rfMRI e difusão) |
| MEG | 95 sujeitos com MEG de repouso e/ou tarefa — **sobreposição exata com dMRI/rfMRI a conferir** na lista de sujeitos |
| Atenção | Inclui gêmeos/irmãos: dividir treino/teste **por família** (informação restrita) ou usar subconjuntos de não aparentados |
| Status | O ConnectomeDB antigo será desligado; a versão de agosto/2025 está no BALSA |

### 2.3 Gabarito animal (opcional, validação de método)

Em humanos a difusão é o melhor gabarito *in vivo*, mas tem fibras falsas/perdidas e não mede força
sináptica. Em macaco há conectividade medida por **traçadores retrógrados**: Markov et al. 2014
(*Cereb Cortex*), 91 áreas, 29 injetadas, pesos FLNe e distâncias — `core-nets.org`
(**arquivo e termos a conferir**; o site não respondeu na verificação). Para usar como teste do
método FC→SC seria preciso fMRI de macaco no mesmo espaço de parcelação (a pesquisar).

## 3. Desenho experimental

### 3.1 Divisão
- MICA-MICs: validação cruzada **5-fold por sujeito** (10 sujeitos de teste por fold), sementes fixas.
- HCP: conjunto de teste final separado, por família, nunca visto em ajuste de hiperparâmetros.

### 3.2 Modelos (todos com a mesma divisão)

| Modelo | Entrada → saída | Papel |
|---|---|---|
| B0 — SC média do grupo de treino | — → SC | piso: nada individual |
| B1 — Modelo por distância `exp(-GD/ℓ)` | GD → SC | já implementado em `neurochaos/connectome.py` |
| B2 — FC limiarizada / regressão ridge | FC (+GD) → SC | baseline linear |
| **GAN** — pix2pix condicional em matrizes (simetria imposta, diagonal zero) | FC + GD (+MPC) → SC | modelo proposto |
| Controle | GAN com FC de **outro** sujeito | testa se a GAN usa informação individual |

### 3.3 Métricas — nível estrutural (gabarito = SC da difusão do mesmo sujeito)
- Correlação aresta-a-aresta (Pearson/Spearman no triângulo superior, log dos pesos).
- AUC para existência de aresta.
- **Identificabilidade**: a SC prevista do sujeito *i* é mais parecida com a SC real de *i* do que com a de
  outros? (acurácia de identificação; diferencia "aprendeu a média" de "aprendeu o indivíduo").
- Propriedades de grafo: grau, força, modularidade, *rich club*, distribuição de pesos.

### 3.4 Métricas — nível fisiológico (simulação)
- Modelo de massa neural por região (oscilador de Hopf ou Wong-Wang reduzido) + modelo hemodinâmico
  (Balloon–Windkessel) → BOLD simulado com o mesmo TR.
- Ajuste de 1 parâmetro global (acoplamento G) por sujeito **no treino**; comparar FC simulada × FC
  empírica com SC real, SC da GAN, SC média, SC embaralhada (controles do v1).
- Critério de sucesso: SC da GAN reproduz a FC do sujeito melhor que B0/B2 e próximo da SC real.

### 3.5 Escala rápida (theta/gama, caos) — fase posterior
O BOLD é lento (TR 0,6–0,72 s): não resolve theta (6 Hz) nem gama (40 Hz). Para isso usar o MEG do HCP
(95 sujeitos) e o simulador de rede/caos já implementado no v1.

## 4. Onde colocar os dados

```text
data/
├── raw/
│   ├── mica_mics/        # BIDS bruto (opcional; só se for reprocessar)
│   └── hcp_ya/           # downloads do BALSA (após aceitar os termos)
├── derivatives/
│   └── mica_mics/        # matrizes micapipe: FC, SC, MPC, GD por sujeito e parcelação
├── atlases/              # parcelações (Schaefer, DK…) usadas pelas matrizes
└── manifests/
    ├── datasets.csv
    └── subjects_split.csv  # sujeito → fold (gerado por script, versionado)
```

## 5. Ordem de aquisição

1. [x] **MICA-MICs** — `micapipe_MICs_v1.1.zip` (2,74 GB, SHA-256 conferido com o OSF) baixado; extraídas
   Schaefer-200 e Desikan-Killiany: SC, edgeLength, FC, séries BOLD, GD, MPC (`scripts/prepare_mica.py`).
2. [x] Integridade (`scripts/check_mica.py`): 50 sujeitos, 0 problemas; a FC recalculada das séries BOLD
   coincide com a FC fornecida (erro < 1e-5) → ordem dos nós confirmada. `subjects_split.csv` com 5 folds.
   Atenção: SC muito densa (96 % das arestas > 0 em Schaefer-200) → avaliar pesos, não só existência.
3. [x] Baselines (`scripts/run_baselines.py`, resultados em `results/baselines/`). Schaefer-200, 5 folds:
   a média do grupo já dá r = 0,94 com a SC individual → r total NÃO discrimina modelos. A métrica que
   importa é a parte individual: B2 (linear FC+GD+MPC) r_desvio = 0,040 [0,030–0,050] e identificação
   12 % (acaso 2 %, p = 5e-4); com a FC de outro sujeito cai para 0,006 / 0 %. **Meta da GAN: superar
   r_desvio ≈ 0,04 e identificação ≈ 12 %.** Em Desikan-Killiany (70 nós) o sinal individual é ainda menor.
4. [x] GAN pix2pix (FC+GD+MPC+prior → desvio da SC) e ablação U-Net só-L1, 5 folds × 3 sementes × 300
   épocas, GPU (`scripts/run_gan.py`, `results/gan/`). **Resultado negativo**: GAN r_desvio = 0,013 e
   identificação 2 % (acaso) contra B2 linear 0,040 / 12 % (Wilcoxon p = 3e-5). A parte adversarial ajuda
   pouco sobre a U-Net só-L1 (0,013 vs 0,005). L1 de treino cai enquanto o r de validação fica ≈ 0 →
   com 35 sujeitos de treino a rede memoriza desvios em vez de aprender um mapeamento generalizável.
   - **Correção: GAN residual sobre o B2** (`run_gan.py --residual-b2`, `results/gan/schaefer200_resB2/`): a previsão
     do B2 (ajustado só no treino do fold) entra como canal e a rede aprende só o resíduo; última camada iniciada em
     zero, então a época 0 = B2 e a validação pode mantê-lo. Resultado: r_desvio 0,042 vs 0,040 do B2 pareado
     (p = 0,83), identificação 14 % vs 10 % (n.s.). Em 8 de 15 treinos a validação escolheu a época 0 (= B2).
     **Conclusão: com 50 sujeitos, a GAN não acrescenta informação além do modelo linear.**
5. [x] Simulação de Hopf (`scripts/run_hopf.py`, `results/hopf/schaefer200/`): G ajustado na 1ª metade do BOLD,
   avaliado na 2ª; FC do modelo estimada em 30 min simulados; o B2 usa só a FC da 1ª metade (sem vazamento).
   FC simulada × empírica: real 0,151 · B2 0,155 · grupo 0,152 · SC de OUTRA pessoa 0,150 · embaralhada 0,010.
   Referências: correlação direta log SC × FC 0,221; teto (FC metade A × B) 0,552.
   **A topologia importa muito (embaralhar derruba o ajuste em 50/50 sujeitos), mas a SC individual não
   explica a FC individual melhor que a SC média ou a de outra pessoa.** O pequeno ganho do B2 sobre o grupo
   (+0,003, 39/50, p_holm = 5e-4) vem da FC da 1ª metade embutida na SC prevista, não de estrutura melhor.
5b. [x] **Rede biofísica NEURON/NetPyNE sobre o conectoma real** (`configs/cortical_atlas.yaml`,
   `scripts/run_cortical_subjects.py`, `results/cortical_subjects/K2_th0/`). Canais em NMODL (`mod/`): Na⁺ e K⁺
   Traub–Miles + corrente M (Pospischil et al. 2008), 36 °C. Piramidal RS: reobase 0,25 nA, início contínuo
   (3,3 Hz), PA½ 0,72 ms, adaptação (ISI 14,6→33,4 ms); interneurônio FS: 0,33 nA, PA½ 0,60 ms, sem adaptação.
   70 regiões Desikan-Killiany × (8 RS + 2 FS) = 700 neurônios; P das conexões longas = log1p(SC); atrasos =
   comprimento das fibras / 5 m/s. Calibração: transição abrupta para atividade descontrolada entre 0,002 e
   0,004 µS; usado K = 2 (0,002 µS), o maior valor estável. dt = 0,05 ms (taxas a < 0,3 % de dt/2).
   10 sujeitos × 4 conectomas × 20 s. FC lenta simulada × FC do BOLD: própria 0,100 · outra pessoa 0,109 ·
   grupo 0,129 · embaralhada −0,017 (própria vs embaralhada 10/10, p_holm = 0,035; vs outra/grupo n.s.).
   **Replica o resultado do Hopf com um modelo biofísico independente.**
5c. [x] **Rede NEURON com 50 sujeitos** (CoreNEURON, `results/cortical_subjects/K2_th0_core` e `K2_th0.1_core`):
   espontâneo — FC lenta × BOLD: própria 0,122 · outra 0,110 · grupo 0,152 · embaralhada −0,024 (própria vs outra
   p_holm = 0,24; vs embaralhada 50/50, p ≈ 1e-14; grupo > própria, p ≈ 1e-6). Theta 0,1 nA: 0,177 · 0,177 · 0,200 ·
   0,012; PAC ~10× maior com theta, mas igual com conectoma embaralhado (vem da dinâmica local, não da topologia).
5d. [x] **Adendo pré-registrado** (`docs/preregistro_adendo_modelos.md`, `results/escada/`, Holm nos 3 primários):
   - Busca aninhada GAN/BrainNetCNN (`docs/preregistro_gan_nested.md`): r_desvio 0,052 vs B2 0,040, 37/50, p = 0,003;
     identificação 16 %. Escolhida BrainNetCNN + perda de correlação; o adversarial não ajudou. Exploratório.
   - GNN: 0,052 vs 0,040, p = 0,041 (p_holm 0,041); identificação 16 %. Perda de correlação melhor em 8/8 pares.
   - PLS de kernel: r_desvio 0,078 (p_holm 0,021) porém identificação 0 %: ganho populacional, não individual
     (post hoc: previsões 4× mais parecidas entre si que as verdades) → identificação é a métrica de individualidade.
   - Curva de aprendizado: B2 satura em N = 10; PLS cresce até N = 40.
   - Conectividade efetiva (Hopf linearizado, emenda 1): FC × 2ª metade 0,207 → 0,330 (teto 0,552), mas identificação
     estrutural 2 % e r_desvio 0,015 < B2 (p_holm 0,003): a individualidade funcional não aponta para a SC individual.
   - Caos na rede cortical (10 sujeitos): λ ≈ −5/s (perto da neutralidade), divergência finita 1–7 %; própria =
     embaralhada (p > 0,3). Sem evidência de caos determinístico.
5e. [x] **Robustez de parcelação**: B2 identifica 2 % (Desikan, 70), 12 % (Schaefer-200), 22 % (Schaefer-400; p = 4e-9) —
   a informação individual cresce com a resolução. Hopf na Desikan: própria 0,333 = outra 0,333; embaralhada 0,036;
   supera a correlação direta SC×FC (0,304).
6. [ ] **Você**: conta no BALSA + aceite dos termos do HCP-YA (pode levar dias se houver restritos).
7. [ ] Replicar no HCP (teste final por família) e, depois, MEG para a escala rápida.

## 6. Limitações a declarar

- FC ≠ SC: correlação funcional pode vir de caminhos indiretos ou entrada comum.
- A SC da difusão é proxy (vieses de tractografia; contagem de streamlines ≠ axônios ≠ força sináptica).
- A GAN gera **plausibilidade**; o mérito é medido contra o gabarito do mesmo sujeito e contra baselines.
- 50 sujeitos (MICA-MICs) limita conclusões; replicação no HCP é necessária.

## Fontes verificadas

- Royer et al. 2022, *An Open MRI Dataset for Multiscale Neuroscience*, Sci Data — https://pmc.ncbi.nlm.nih.gov/articles/PMC9477866/
- OSF MICA-MICs — https://osf.io/j532r/ (license.txt: CC0 1.0)
- HCP-YA no BALSA (ago/2025) — https://humanconnectome.org/study/hcp-young-adult/article/updated-hcp-young-adult-data-released-connectomedb-powered-balsa
- HCP 1200 Subjects Release (MEG: 95 sujeitos) — https://humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release
- Markov et al. 2014 no Scalable Brain Atlas — https://scalablebrainatlas.incf.org/macaque/MERetal14