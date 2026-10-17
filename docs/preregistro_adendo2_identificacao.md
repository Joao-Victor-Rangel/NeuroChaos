# Adendo 2 ao pré-registro — maximizar a identificação FC → SC sem inflar

**Escrito em 15/10/2026, antes de executar.** Hash SHA-256 gravado em `results/identificacao/`. Status: exploratório
no MICA-MICs (já visto); confirmação obrigatória no HCP com as escolhas congeladas.

## Motivação (dados já obtidos, sem olhar resultados destas análises)
Teto funcional FC → FC (`results/tetos`): 80 % (Schaefer-200), 92 % (Schaefer-400). Melhor FC → SC atual: 16 % / 22 %.

## Desfecho primário
**Acurácia de identificação** no teste externo: para cada sujeito de teste, o "candidato" escolhido entre as SCs dos
50 sujeitos (desvio em relação à média do treino do fold) deve ser ele próprio. Acaso = 2 %.
Comparação com o B2 da mesma parcelação: teste exato de McNemar (pares discordantes acerto/erro por sujeito).
Duas parcelações (Schaefer-200 e Schaefer-400) × 2 métodos = 4 testes primários, corrigidos por Holm.

## Método 1 — B2 com seleção de arestas confiáveis (ES)
Critérios calculados SÓ nos sujeitos de treino do fold:
- `fc_rel`: confiabilidade da FC por aresta = correlação, entre sujeitos, do valor da aresta na 1ª vs 2ª metade do BOLD.
- `sc_var`: arestas presentes (SC > 0) em ≥ 95 % dos sujeitos de treino, ordenadas pelo coeficiente de variação
  entre sujeitos de log1p(SC/média).
Grade: critério ∈ {fc_rel, sc_var} × fração mantida q ∈ {0,10; 0,25; 0,50; 1,0}. O B2 é ajustado e a identificação
é feita só nas arestas mantidas. Escolha por validação interna (4 folds nos 40 de treino; candidatos = os 40),
pela acurácia de identificação; empate → maior q (menos seleção), depois `fc_rel`.

## Método 2 — identificação em espaço comum (CCA regularizada)
- No treino: desvios de FC (arctanh) e de SC (alvo log1p) em relação às médias do treino; PCA separada em cada
  modalidade (k_pca componentes), CCA entre os escores das PCAs (k_cca componentes).
- No teste: FC do sujeito → espaço CCA; SCs dos 50 candidatos → espaço CCA (pesos aprendidos só no treino);
  identificação por correlação no espaço CCA.
- Grade: k_pca ∈ {10, 20, 30} × k_cca ∈ {2, 5, 10} (k_cca ≤ k_pca). Escolha por validação interna (acurácia);
  empate → menos componentes.

## Regras
- Todas as configurações testadas são reportadas (escores internos).
- Nenhuma escolha usa os sujeitos de teste do fold externo.
- Resultados v. anteriores (B2, GAN, GNN, PLS) permanecem como estão; esta é uma análise adicional.