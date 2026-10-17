# Diário metodológico — NeuroChaos

Registro cronológico de TODAS as análises, inclusive as substituídas, com erros encontrados e correções.
Nada é removido deste arquivo; resultados novos entram no fim. Datas em 2026.

## Fase 1 — MVP de simulação (14/10)
1. **Rede HH multirregional** (4 regiões sintéticas × 8E+2I, NEURON 8.2.2 + NetPyNE 1.0.3.1).
   Validação: corrente entregue (theta 0,040 nA / gama 0,010 nA conferidas no IClamp), determinismo bit a bit,
   estimadores de caos em Lorenz (λ = 0,869 vs 0,906) e van der Pol (≈ 0), Rosenstein dá λ > 0 para ruído (controle).
2. **Calibração**: reobase HH 0,070 nA; fundo Poisson 400 Hz → taxas altas demais; reduzido a 150 Hz (exploração em
   `results/dev`, **apagada** depois — scripts e parâmetros preservados em `configs/mvp.yaml`). K_base = 0,0006 µS.
3. **Convergência dt/dt÷2**: taxas < 2 %; trens de spikes descorrelacionam em ~300 ms (instabilidade finita).
4. **Achados de método**: (a) perturbações mínimas reconvergem (λ < 0), finitas divergem → λ linear e p_divergência
   separados; (b) surrogates por deslocamento são inválidos com theta injetado → Rayleigh/comparação entre condições.
5. **Grade 600 simulações v0 — SUBSTITUÍDA**: λ linear enviesado (ln δ sem piso; reconvergência bit a bit a δ = 0
   gerava ln(1e-300) e dominava a média). Recheck em 16 jobs: λ mudava até 40 % (ex.: −65,9 → −40,8/s);
   p_divergência idêntica. Resultados v0 **apagados**; grade refeita com piso numérico (`results/grid_debug`).
   Conclusões mantidas: topologia sem efeito (0/192 contrastes após Holm), K aumenta taxa e PLV gama.

## Fase 2 — mudança de pergunta: TC → fMRI/GAN (14/10)
6. Usuário esclarece o objetivo (exame → GAN → topologia → simulação). TC sem informação funcional → fMRI + T1 + difusão.
7. **MICA-MICs** (OSF j532r, CC0, zip 2,74 GB, SHA-256 conferido). Integridade: 50 sujeitos, 0 problemas; FC recalculada
   das séries BOLD bate com a fornecida (erro < 1e-5) → ordem dos nós correta. SC 96 % densa.

## Fase 3 — inferir SC individual (14–15/10)
8. **Baselines** (Schaefer-200): média do grupo r = 0,94 → r total não discrimina; métrica = r do desvio e identificação.
   B2 linear: 0,040 / 12 %; controle com FC de outro sujeito: 0 %.
9. **GAN pix2pix v1**: 0,013 / 2 % (pior que B2). Diagnóstico: memoriza (L1 treino cai, validação ≈ 0).
10. **GAN residual sobre B2** (zero-init, época 0 = B2): 0,042 / 14 % vs B2 0,040 / 10 % (p = 0,83). Época 0 escolhida
    em 8/15 treinos.
11. **Busca aninhada pré-registrada** (`docs/preregistro_gan_nested.md`, hash gravado): BrainNetCNN + perda de correlação
    escolhida; 0,052 / 16 % vs B2 0,040 (p = 0,003). Adversarial não ajudou. Exploratório (MICA já visto).
12. **Adendo pré-registrado** (`docs/preregistro_adendo_modelos.md`): GNN 0,052 / 16 % (p_holm 0,041); PLS 0,078 / **0 %**
    (ganho populacional — post hoc `scripts/posthoc_componente_comum.py`); curva de aprendizado (B2 satura em N = 10,
    PLS sobe); conectividade efetiva (emenda 1: n_iter 100, melhor iteração na 1ª metade).

## Fase 4 — simulação fisiológica com dados reais (15/10)
13. **Hopf v0 — SUBSTITUÍDO**: SC do B2 prevista com a FC do exame inteiro (incluía a metade de teste) → vazamento.
    Vantagem do B2 sobre o grupo (+0,003) era parcialmente vazamento. Resultado v0 **apagado**; refeito com FC da 1ª metade.
14. **Hopf corrigido** (FC do modelo em 30 min simulados — com 4 min o ajuste era ~0,09): própria 0,151 = outra 0,150 =
    grupo 0,152; embaralhada 0,010. Desikan: 0,333 = 0,333; embaralhada 0,036 (supera SC×FC direta 0,304).
15. **Canais NMODL** (Na/K Traub–Miles + M, Pospischil 2008; 36 °C): RS reobase 0,25 nA, início contínuo 3,3 Hz,
    adaptação 14,6 → 33,4 ms; FS 0,33 nA, sem adaptação. (Primeira compilação falhou por acento em comentário NMODL.)
16. **Rede cortical sobre conectoma real** (70 regiões × 10 neurônios): transição abrupta para atividade descontrolada
    entre 0,002 e 0,004 µS; K = 2. 10 sujeitos (NEURON) e depois 50 sujeitos (CoreNEURON, espontâneo e theta):
    própria = outra pessoa; grupo > própria; embaralhada destrói o ajuste. PAC sobe ~10× com theta, igual no embaralhado.
    Incidentes: notebook suspendeu (rodada retomada); WSL reiniciou por falta de RAM (8 → 6 processos); 1 arquivo
    `data.npz` corrompido (separado em `_corrompido_*`, refeito).
17. **CoreNEURON**: 1,5× mais rápido, spikes idênticos ao NEURON (Random123); **não grava V com Dt próprio** → análise de
    caos usa NEURON padrão (1ª e 2ª tentativas falharam; pastas `results/escada/_falha*_caos_*` preservadas).
18. **Caos na rede cortical**: λ ≈ −5/s, divergência finita 1–7 %, própria = embaralhada. Sem caos determinístico.

## Fase 5 — robustez e tetos (15/10)
19. **Parcelação**: B2 identifica 2 % (70), 12 % (200), 22 % (400).
20. **Teto funcional FC → FC** (`results/tetos/teto_fc.json`): 60 % (70), 80 % (200), 92 % (400) com ~3,5 min por metade.
    → A FC é muito individual; o gargalo da identificação FC → SC está na estrutura (tractografia) ou na ligação
    função–estrutura, não na duração do fMRI.21. **Adendo 2 — maximizar identificação** (`docs/preregistro_adendo2_identificacao.md`, `results/identificacao/`):
    - Seleção de arestas pela confiabilidade da FC (critério e fração por validação interna; `fc_rel` escolhido em
      10/10 folds): Schaefer-200 16 % vs B2 12 %; Schaefer-400 **26 %** vs B2 22 % (McNemar p ≈ 0,7; n.s.).
      Selecionar pela variabilidade da SC piorou (validação interna 4–17 %).
    - CCA em espaço comum: falhou (2 % e 0 %); pior que o B2 na Schaefer-400 (p_holm = 0,004). Com ~30 sujeitos de
      treino o espaço comum FC–SC sobreajusta e não carrega identidade.
    - Leitura: truques de método honestos dão ganhos pequenos (+4 pp); o teto prático com estes dados está em ~25 %
      (13× o acaso). A distância para o teto funcional (80–92 %) está no lado estrutural / acoplamento.
22. **Adendo 3 — combinação** (`docs/preregistro_adendo3_combinado.md`, `results/identificacao/combinado/`): GNN residual
    (λ_corr 10) + arestas com FC confiável + Schaefer-400. Validação interna 25–48 % (otimista: mesma partição escolhe
    época e mede), **teste externo 30 % (15/50; 15× o acaso)** vs B2 22 % (McNemar 8 × 4, p = 0,39) e vs B2+arestas
    26 % (p = 0,77). Ganhos se somam, mas não são significativos com n = 50.
    Trajetória da identificação FC → SC: 2 % (GAN v1) → 12 % (B2, S200) → 16 % (GNN/BrainNetCNN, S200) → 22 % (B2, S400)
    → 26 % (+arestas) → 30 % (combinado). Teto funcional FC → FC: 92 % (S400).
23. **Piloto de reprocessamento da difusão** (`docs/levantamento_reprocessamento_dwi.md`, `scripts/dwi_pilot.sh`):
    lote bruto `raw_HC001-HC005.tar.gz` (4,68 GB, SHA-256 conferido); ambiente `neurochaos-dwi` (MRtrix3 3.0.8, ANTs,
    FSL topup/eddy/bet/flirt); atlas Schaefer-400 volumétrico (CBIG) + MNI152NLin6Asym (TemplateFlow).
    Incidentes: (a) nome de pacote errado (`fsl-eddy-cpu` → `fsl-eddy`); (b) eddy recusou os dados ("Data not shelled",
    b-valores com pequena variação) → `--data_is_shelled`; (c) `eddy_cpu` usava 1 núcleo → `--nthr`; (d) eddy em CPU
    muito lento (> 5 h por sujeito com 1,6 mm/143 volumes, swap pela RAM de 5 GB) → instalado `eddy_cuda` (ambiente
    separado `neurochaos-eddycuda`, wrapper `~/dwi_work/bin/eddy_cuda`); teste em paralelo travou o WSL por memória
    e foi encerrado (GPU chegou a 6 % de uso); fila (HC003–HC005) configurada para usar a GPU, com recaída automática
    na CPU pelo dwifslpreproc. HC001/HC002 seguem com eddy em CPU para não perder ~5 h de processamento.

24. **Pausa do piloto (16/10, ~00:25, a pedido do usuário)** — estado ao parar:
    - sub-HC001: topup concluído; eddy na GPU (`eddy_cuda`, wrapper corrigido: a versão GPU só aceita `--nthr=1`)
      rodava há ~70 min quando foi interrompido. Etapas iniciais (`dwi_dg.mif`) guardadas em `~/dwi_work/proc/sub-HC001`.
    - sub-HC002: eddy em CPU (~6 h, 3 threads) interrompido antes, para liberar RAM; etapas iniciais guardadas.
    - sub-HC003: prévia sem eddy (`PREVIA=1`) nas etapas iniciais; HC004/HC005 não iniciados.
    - Nenhum conectoma reconstruído ainda. Lição: com 5 GB de RAM no WSL, rodar só UM eddy por vez; o eddy em
      CPU nestes dados (1,6 mm, 143 volumes) leva > 5 h por sujeito; usar a GPU.
    - **Como retomar:** prévia rápida primeiro (≈ 1 h, sem eddy, para um número inicial do ruído da tractografia):
      `PREVIA=1 bash scripts/dwi_pilot.sh sub-HC003 2M 6` e depois `sub-HC004`; em seguida
      `python scripts/dwi_reliability.py results/dwi_pilot_previa`. Piloto completo: um sujeito por vez,
      `bash scripts/dwi_pilot.sh sub-HC00X 2M 4` (eddy na GPU via `~/dwi_work/bin/eddy_cuda`).
25. **Prévia do reprocessamento (16/10, sem topup/eddy, `PREVIA=1`; `results/dwi_pilot_previa/`)**. Mais 4 incidentes de
    ambiente corrigidos e registrados em `erros.txt`: faltava `fsl-avwutils` (fslval/fslstats); `bet -R` exigia `dc`
    (instalado `bc` do conda-forge) e, ao rodar, deixava crânio/pescoço (máscara de ~3,2 L) → trocado por máscara do
    MNI levada ao T1 via ANTs (volume conferido: 1,41 L em HC004); `dwi2response` ignorava `-force` antes do subcomando;
    `antsRegistrationSyNQuick.sh` quebrava com espaço no caminho do projeto → atlas copiados para `~/dwi_work/atlases`.
    **Resultado (2 sujeitos, 2 M fibras, SIFT2, Schaefer-400 volumétrico):** r entre sementes da mesma pessoa
    0,995 / 0,994; r entre pessoas diferentes 0,836; densidade 33–35 %.
    Leitura: o ruído estocástico da tractografia é ~25× menor que a diferença entre pessoas → **não é o gargalo**.
    A SC deste pipeline é mais esparsa e mais distinta entre pessoas que a do micapipe (onde a média do grupo explica
    r ≈ 0,94). Ressalva: parte da diferença entre pessoas pode vir do registro do atlas por sujeito; separar isso exige
    reteste de aquisição (HCP).26. **Reconstrução das 50 SCs (16/10, `scripts/dwi_mica50.sh` → `results/dwi_mica50/`)**. Decisão: modo sem eddy
    (`PREVIA=1`), 1 tractografia (semente 1, 2 M fibras, SIFT2), Schaefer-400 volumétrico, para ter as 50 SCs em ~2 dias;
    eddy na GPU fica como análise de sensibilidade num subconjunto (HC001/HC002 têm os intermediários em `~/dwi_work/proc/`).
    Download em fluxo do OSF z8mu5 (só DWI + T1w run-1, SHA-256 conferido, até 3 tentativas por lote; falha gera
    `.fail` para o processamento não travar). HC003/HC004 reaproveitados da prévia (mesmo pipeline, semente 1;
    `origem.txt`). **Espaço em disco:** com o C: em 26 GB livres, com autorização do usuário foram descartados os
    intermediários regeneráveis da prévia (`~/dwi_work/proc_previa`, 12 GB; SCs/máscaras/logs preservados em
    `results/dwi_pilot_previa/`) e o cache de pacotes do conda (`conda clean -a`); os intermediários de cada sujeito
    passam a ser descartados após a SC ser salva (`CLEAN=1`). Incidente: a API do OSF devolveu 502 na hora do
    lançamento → listagem passa a tentar indefinidamente a cada 60 s.
27. **OSF em manutenção (16/10)**: a API respondia 502 e depois 503; a status.cos.io informa "Cloud Infrastructure
    Migration" de 16/10 09:00 até 19/10 09:00 (horário de Nova York). O orquestrador das 50 SCs parou em 5/50
    (HC001–HC005, os já baixados) e retoma sozinho quando o OSF voltar. Uso do tempo parado: rodar o pipeline
    **completo com topup + eddy na GPU** nos mesmos 5 sujeitos (1 semente, 2 M fibras, `CLEAN=1`;
    `results/dwi_pilot/`, fila em `_fila_eddy.log`), que é a análise de sensibilidade eddy × sem eddy planejada.
    Os intermediários de HC001/HC002 em `~/dwi_work/proc/` vinham de um piloto com eddy que nunca terminou
    (CPU lento; teste na GPU em paralelo congelou o WSL); agora roda um sujeito por vez.
    **HC001 com eddy (16/10, 17:11→19:57, ~2 h 45 min na GPU; pico de RAM do eddy_cuda ~3,9 GB de 4,9 GB):**
    densidade 0,360 (sem eddy 0,361); r(log) com a SC sem eddy da mesma pessoa = 0,911; com as SCs sem eddy de
    HC002–HC005 = 0,78–0,81 (média entre pessoas sem eddy 0,801). Leitura preliminar (n=1): o eddy muda a SC bem mais
    que o ruído de semente (0,995), mas a pessoa continua reconhecível (0,91 > 0,80). Comparação feita com
    script avulso (scratchpad); a análise formal virá com os 5 sujeitos.
    **HC002 com eddy (19:57→22:30):** r com a própria SC sem eddy 0,931; com as outras 0,77–0,82; r entre HC001 e HC002 ambos com eddy 0,786. 2/2 identificadas (eddy → sem eddy).
    **Comparação preliminar com o micapipe (17/10, 5 sujeitos sem eddy, Schaefer-400; script avulso no scratchpad):**
    r(log SC) entre pessoas: nosso pipeline 0,801 × micapipe 0,849 (o nosso é um pouco mais individual);
    nossa SC × SC micapipe da mesma pessoa 0,70–0,74 (pipelines bem diferentes); identificação dos desvios
    nossa → micapipe 5/5 (confirma que a ordem dos 400 rótulos bate). Identificação FC → SC por correlação bruta de
    desvios (sem modelo): nossa 0/5, micapipe 2/5, acaso 1/5 — n = 5 não permite conclusão.
28. **Mudança de rumo (17/10): "gêmeo digital funcional" + vertente CT.** Discussão com o usuário: um modelo
    generativo não cria informação individual ausente no exame; a fMRI carrega individualidade funcional (teto
    80–92 %), não anatômica. Duas frentes novas, sem descartar nada do que veio antes:
    (a) **Adendo 4** (`docs/preregistro_adendo4_gemeo.md`, SHA-256
    `6fe09d7d2cdb8a8ce868564544f5b3ce498a385f478df9e992139eb56ad7d77e`, escrito antes de rodar): o Hopf linearizado
    ajustado em cada metade do BOLD reconhece a pessoa (P1) e tem parâmetros estáveis (P2)? Script
    `scripts/run_gemeo.py`, saída em `results/gemeo/`. Rodando: variante "grupo" (primária) e "próprio" (S1).
    (b) **Levantamento CT → lesão → simulação** (`docs/levantamento_ct_lesao.md`): não há base pública com CT e fMRI
    nos mesmos pacientes; desenho em três elos (AISD/APIS: CT → lesão; ISLES'24: CT → desfecho mRS/NIHSS;
    ARC: lesão → FC real), com o conectoma de grupo justificado pelo nosso achado de que a SC do grupo serve tão bem
    quanto a individual.
    **Resultado primário do adendo 4 (17/10, Schaefer-200, variante "grupo", 918 s):**
    P1 gêmeo_A × FC_B = **44 %** (p_holm 2e-24); P2 EC_A × EC_B = **70 %** (p_holm 1e-47); teto FC_A × FC_B 80 %;
    controle sem ajuste 4 % (≈ acaso). McNemar P1 vs teto: 0 × 18 (p = 8e-6), retenção 0,55. r médio: gêmeo × A
    0,659, gêmeo × B 0,328, FC A × B 0,599, EC A × EC B 0,607. Iteração escolhida: A 9,8, B 24,5 em média.
    **Checagem pós-hoc (não pré-registrada):** como o C0 muda por fold, testou-se se o fold "ajuda" a identificar:
    só candidatos de outros folds P1 46 % / P2 72 %; removendo a média de cada fold P1 44 % / P2 72 % → sem confusão
    por fold. **Leitura:** os critérios pré-registrados de P1 e P2 foram atingidos; os parâmetros do gêmeo são uma
    impressão digital estável entre metades. Ressalva honesta: com poucas iterações, a EC é próxima de
    C0 + ε·k·(FC_A − F0), então boa parte da individualidade é herdada da FC (P2 70 % < teto 80 %); o ganho do gêmeo
    não é identificar melhor que a FC, e sim ser um modelo **mecanístico** que preserva a pessoa e pode ser perturbado.
29. **Queda da sessão (~01:10 de 17/10)**: o app/WSL parou (provável reinício ou suspensão do computador; não
    há registro de OOM). Perdidos: S1 do gêmeo (estava em HC009; log preservado como
    `results/gemeo/log_proprio_interrompido_0110.txt`) e o eddy de HC004 (estava no início; diretório de trabalho
    parcial descartado, regenerável). Relançados às 10:30: S1 → S2 → eddy HC004/HC005 em sequência (evita somar a
    memória do eddy com a dos ajustes) e o orquestrador das 50 (aguardando o OSF, ainda em manutenção).
    **Eddy HC003 (22:30→01:07):** r com a própria SC sem eddy 0,925; com as outras 0,77–0,80; 3/3 identificadas;
    r entre pessoas com eddy 0,780.
30. **Adendo 5 — lesão virtual no gêmeo** (`docs/preregistro_adendo5_lesao_virtual.md`, SHA-256
    `35ee39b86a2e438322620631b3c2bbef2d4af711465d6eaa3dd494c7b35c8e93`, escrito antes de rodar). Zera a EC de cada
    região e mede o impacto à distância; L1: vetor de impacto identifica a pessoa entre metades? L2: e depois de
    tirar a força do nó? Descritivo: gêmeo individual × gêmeo de grupo. Script `scripts/run_lesao.py`, saída
    `results/lesao_virtual/`. Uma solução de Lyapunov (N = 200) leva 0,165 s; ~20 mil soluções no total.
    **S1 do adendo 4 (variante "próprio", C0/máscara da SC micapipe do próprio sujeito; 513 s):** P1 = 44 % (igual à
    variante grupo; McNemar vs teto idêntico); P2 = 100 %; controle sem ajuste 6 %. **P2 = 100 % NÃO é evidência de
    gêmeo melhor:** EC_A e EC_B partem da mesma C0 (a SC da pessoa) e da mesma máscara, e o ajuste para em poucas
    iterações (A 16, B 30), então as duas EC compartilham a SC da pessoa e se reconhecem trivialmente — é a mesma
    matriz de partida identificando a si mesma. Por isso P2 só é interpretável na variante "grupo" (C0 comum a todos
    do fold). Leitura válida de S1: usar a SC individual **não melhora** o que o gêmeo prevê da fMRI (P1 44 % = 44 %),
    em linha com os achados anteriores (SC individual não muda as simulações). Checagem: as EC das duas variantes não
    são iguais (r 0,94–0,95 entre variantes); a coincidência de r médio EC_A × EC_B = 0,607 nas duas foi conferida.
    **Resultado do adendo 5 (17/10, 1652 s):** L1 impacto A × B = **12 %** (p_holm 5e-4); L2 resíduo da força do nó =
    **26 %** (p_holm 3e-11); acaso 2 %. r médio I_A × I_B: mesma pessoa 0,420, entre pessoas 0,317. **Gêmeo de grupo
    prevê o impacto da metade B melhor que o gêmeo individual da metade A:** r 0,488 × 0,420; individual > grupo em só
    18/50 (Wilcoxon p = 0,018, a favor do grupo). r impacto × força do nó = −0,63 (regiões mais conectadas na EC
    causam, em média, menos efeito à distância neste modelo linear). Maior impacto: regiões do modo padrão
    (PFC, PFC dorsomedial, temporal, pCun/PCC) e OFC límbico.
    **Leitura pela regra pré-registrada:** existe componente individual reprodutível do efeito da lesão (L1, L2), mas
    ele é pequeno diante do ruído de estimar o gêmeo com ~3,5 min de BOLD; para prever o efeito de uma lesão numa
    pessoa, o gêmeo de grupo é melhor → **o conectoma/gêmeo de referência basta** para a vertente CT → lesão →
    simulação (onde não há fMRI). A parte individual é mais visível depois de remover a força do nó (26 % > 12 %).
31. **Incidente (17/10, 11:08):** para adiar o eddy de HC004/HC005 (liberar memória para o gêmeo no NEURON),
    encerrei o shell que encadeava S2 → eddy; o S2 (ε = 0,002, 500 iterações) caiu junto (estava em HC033, 28 min).
    Log parcial preservado em `results/gemeo/log_grupo_eps0.002_it500_interrompido.txt`; S2 relançado do zero
    desacoplado (`setsid nohup`). Eddy HC004/HC005 adiado (3/5 sujeitos com eddy já prontos).
32. **Adendo 6 — gêmeo biofísico no NEURON/NetPyNE** (`docs/preregistro_adendo6_gemeo_neuron.md`; hash original
    `e55bc66c9df622d7be1daf859ae9f1f4a3b10b3dba64613f43148a1f3a843868`; com a emenda 1: `5a99537f341a9cd58df0c40cdbd1935a604c6fb6853c3335006a6fe19de4476e`). A EC individual
    (metade A) vira a fiação da rede RS/FS com canais (aparc, 70 regiões, 10 células/região, CoreNEURON, 20 s);
    desfecho: FC lenta simulada × FC da metade B. Teste de fumaça revelou totais de conexão diferentes entre
    condições (gêmeo de grupo disparando a 216 Hz) → emenda 1: total de probabilidade igualado ao da SC do grupo
    (opção nova `connectome.match_total_P`, desligada por padrão; resultados anteriores não mudam). 250 simulações,
    script `scripts/run_gemeo_neuron.py`, saída `results/gemeo_neuron/`.
    **S2 do adendo 4 (ε = 0,002, 500 iterações; 3224 s):** P1 46 %, P2 68 %, controle 4 %; retenção 0,57; r gêmeo × B
    0,334; r EC_A × EC_B 0,639; iterações escolhidas A 38, B 93. Praticamente igual à primária (44 % / 70 %): o
    resultado não depende do passo do ajuste.
