# Pré-registro — Adendo 6: gêmeo biofísico (EC individual como fiação da rede NEURON/NetPyNE)

Escrito em 17/10/2026, **antes** de rodar. Hash SHA-256 no diário (item 32) e no JSON de resultados.

## Pergunta
O gêmeo funcional (adendo 4) reconhece a pessoa num modelo linear. A individualidade sobrevive quando a rede
efetiva da pessoa vira a **fiação de uma rede de neurônios com canais iônicos** (NEURON 8.2 + NetPyNE; células RS com
Na/Kd Traub–Miles + corrente M e FS; sinapses AMPA/GABA; `configs/cortical_atlas.yaml`)? Se sim, a rede biofísica
personalizada é a base para manipular canais (ex.: corrente M) pessoa a pessoa no próximo passo.

## Dados e gêmeo (fixos)
MICA-MICs, 50 sujeitos, parcelação **Desikan-Killiany (aparc, 70 regiões)** — a da rede NEURON já calibrada.
BOLD dividido em metades A/B (filtro 0,01–0,1 Hz por metade). EC_A = `fit_ec` na FC_A, com C0 = 0,2·SC_grupo/max
e máscara SC_grupo > 0 (SC micapipe média dos sujeitos de treino do fold), ε = 0,01, 100 iterações, melhor iteração
na metade A. A metade **B nunca é usada para construir nada**.

## Fiação no NEURON
A EC é escrita como `weights.csv` em `data/connectomes/gemeo_aparc/<id>/`; o construtor existente aplica
log1p(W/média) → probabilidade de conexão (p0 = 0,2). Atrasos: comprimentos médios do grupo (iguais em todas as
condições). 20 s simulados (2 s de transiente descartados), K = 2·K_base, sem theta, CoreNEURON, **mesmas sementes**
em todas as condições de um sujeito.

Condições por sujeito: `gemeo_proprio` (EC_A da pessoa) · `gemeo_outro` (EC_A de outra pessoa, permutação fixa sem
ponto fixo, semente 20261017) · `gemeo_grupo` (média das EC_A dos sujeitos de treino do fold) · `sc_grupo` (SC
micapipe média; melhor condição anatômica nos testes anteriores) · `gemeo_embaralhado` (EC_A da pessoa com arestas
permutadas).

## Desfecho
FC simulada = correlação entre regiões das flutuações lentas (< 2 Hz) da taxa E (mesma função de
`run_cortical_subjects.py`). Desfecho: r(FC simulada, FC_B empírica da pessoa).

## Testes primários (Holm sobre N1–N3, α = 0,05)
- **N1:** `gemeo_proprio` > `gemeo_outro` (Wilcoxon pareado, bilateral).
- **N2:** `gemeo_proprio` > `sc_grupo` (Wilcoxon pareado, bilateral).
- **N3:** identificação FC simulada (`gemeo_proprio`, i) × FC_B (j), média entre sujeitos removida; binomial contra
  1/50.

## Secundários (relatados todos)
`gemeo_proprio` × `gemeo_grupo` e × `gemeo_embaralhado`; retenção da identificação em relação ao gêmeo linear na
mesma parcelação (FC do modelo linear A × FC_B); taxas de disparo, PLV gama e PAC por condição (checagem de que a
rede continua no regime calibrado).

## Interpretação (fixada agora)
- N1 e N3 significativos → a rede NEURON personalizada carrega a individualidade funcional da pessoa; segue o
  experimento de canais por pessoa.
- N1/N3 não → a individualidade do gêmeo linear se perde na rede biofísica com esta escala (10 células por região,
  conexões sorteadas); o próximo passo seria aumentar células por região ou o tempo simulado, não manipular canais.
- Limitações: metades da mesma sessão; 20 s simulados para FC lenta é curto; 70 regiões têm teto FC→FC de 60 %.
## Emenda 1 (17/10/2026, antes da rodada principal)
Um teste de fumaça (1 sujeito, 3 s simulados, saída no scratchpad, descartável) mostrou que a conversão
EC → probabilidade (p0·W/máx) dá **totais de conexão muito diferentes** entre condições (soma de P: SC do grupo
92,5; gêmeos individuais 127–154; gêmeo de grupo 246), levando a taxas de disparo fora do regime calibrado
(gêmeo de grupo: 216 Hz nas E). Isso confundiria fiação com excitabilidade. Correção: em todas as condições, P é
reescalado (`connectome.match_total_P`, `neurochaos/connectome.py`) para a mesma soma da SC do grupo (92,5),
respeitando pmax; só o **padrão** de conexões muda entre condições. Com isso, no mesmo teste, todas as condições
ficaram em ~11 Hz (E) e ~22 Hz (I). Os valores de r do teste (n = 1, 3 s) foram vistos, mas não têm valor
inferencial e não mudaram nenhuma escolha. Nada mais do desenho foi alterado.
