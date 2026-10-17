# Pré-registro — Adendo 5: lesão virtual no gêmeo digital

Escrito em 17/10/2026, **antes** de rodar. Hash SHA-256 no diário (item 30) e no JSON de resultados.
Depende do adendo 4 (gêmeo "grupo", Schaefer-200, ε = 0,01, 100 iterações), cujo resultado primário já é conhecido
(P1 44 %, P2 70 %); este adendo não muda nada dele.

## Pergunta
O efeito de uma mesma lesão é **individual e reprodutível**? Se o gêmeo de cada pessoa responde à lesão de um jeito
próprio e estável entre metades do exame, a simulação personalizada carrega informação que um conectoma de grupo
não teria — e isso define se a vertente CT → lesão → simulação precisa de dado individual ou se o conectoma de
referência basta.

## Procedimento (fixo)
Para cada sujeito e cada metade (A, B): gêmeo = EC ajustada como no adendo 4 (mesmos folds, C0, máscara, ε, iterações,
frequências intrínsecas da metade). Para cada região k (200): lesão = zerar linha e coluna k da EC (a região
continua com sua dinâmica local, mas desconectada). FC do modelo com e sem lesão (Lyapunov, sem simulação).
**Impacto da lesão k** = média de |ΔFC| sobre as arestas que **não** envolvem k (efeito à distância, "diaschisis").
Cada gêmeo vira um vetor de impacto de 200 posições (I_A, I_B).

## Testes primários (Holm sobre L1 e L2, α = 0,05)
- **L1 — impacto individual e reprodutível:** identificação I_A(i) × I_B(j) (média entre sujeitos removida;
  binomial unilateral contra 1/50).
- **L2 — além da força do nó:** a mesma identificação usando o **resíduo** de I após regredir, em cada gêmeo, o
  impacto sobre a força do nó na EC (soma da linha k). Testa se a individualidade não é só "quem tem mais conexões".

## Descritivos pré-especificados
- r médio entre I e força do nó (dentro do gêmeo).
- r médio I_A × I_B da mesma pessoa e entre pessoas.
- **Gêmeo de grupo:** impacto calculado na EC média dos gêmeos A dos sujeitos de treino do fold; r entre esse
  impacto de grupo e o I_B individual, comparado com o r entre I_A individual e I_B (quanto o individual acrescenta
  sobre o grupo; Wilcoxon pareado).
- As 10 regiões de maior impacto médio (rótulos do Schaefer).

## Interpretação (fixada agora)
- L1 e L2 significativos e individual > grupo → a lesão virtual precisa do gêmeo individual; a vertente CT deve
  usar dado funcional individual quando houver.
- L1 sim e L2 não → a individualidade do impacto é essencialmente força do nó.
- L1 não, ou individual ≈ grupo → o conectoma de referência basta para simular lesões (favorável à vertente CT,
  onde não há fMRI).
- Limitação: metades da mesma sessão; modelo linearizado (o efeito da lesão na dinâmica não linear/NEURON é passo
  posterior).