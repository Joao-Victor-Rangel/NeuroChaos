# Adendo 3 ao pré-registro — combinação GNN + arestas confiáveis + Schaefer-400

**Escrito em 15/10/2026, antes de executar.** Hash SHA-256 gravado em `results/identificacao/combinado/`.
Status: exploratório no MICA-MICs; confirmação no HCP com escolhas congeladas.

## Motivação (resultados já vistos)
Maior resolução ajuda (B2: 12 % → 22 %), arestas com FC confiável ajudam pouco (+4 pp), GNN + perda de correlação
supera o linear na Schaefer-200 (16 % vs 10–12 %). Pergunta: os ganhos se somam na Schaefer-400?

## Procedimento
- Parcelação Schaefer-400; folds externos e internos idênticos aos anteriores (sementes 500+fold).
- Modelo: GNN residual sobre o B2 (como no adendo 1), oculta 64, λ_corr = 10, L1 λ = 100, 200 épocas, Adam 2e-4,
  aumento de dados com banco de **10** FCs por sujeito (reduzido de 20 por limite de memória com 400 regiões) + ruído σ 0,02.
- Grade interna: camadas de mensagem ∈ {1, 2} × fração de arestas confiáveis q ∈ {0,25; 0,50; 1,0}
  (critério `fc_rel` calculado nos sujeitos de treino interno; q só afeta a identificação, não o treino).
- Seleção: maior acurácia de identificação média nos 4 folds internos (candidatos = 40 do treino externo);
  empate → 1 camada, depois maior q.
- Treino final: 35 + 5 validação (época pelo r do desvio), 3 sementes, média das previsões; identificação no teste
  entre os 50 candidatos, nas arestas selecionadas com os 40 de treino.

## Desfecho primário (um teste)
Acurácia de identificação, combinado vs B2 (Schaefer-400, todas as arestas), McNemar exato bilateral, α = 0,05.
Secundário: combinado vs B2 + arestas confiáveis (adendo 2), descritivo.