# Pré-registro — busca aninhada de hiperparâmetros da GAN (FC + anatomia → SC)

**Escrito em 15/10/2026, ANTES de qualquer execução desta busca.** O hash SHA-256 deste arquivo é
gravado em `results/gan_nested/` no início da execução; qualquer alteração posterior fica detectável.

## Contexto e status do conjunto de dados
- MICA-MICs (50 sujeitos) já foi usado para avaliar a GAN v1 e a GAN residual (resultados vistos).
  Portanto **todo resultado desta busca no MICA-MICs é EXPLORATÓRIO / de desenvolvimento**.
- A confirmação será feita no HCP com a configuração congelada, sem novos ajustes.

## Dados e alvo (inalterados)
- Parcelação Schaefer-200 (200 regiões corticais), 50 sujeitos, folds externos de `data/manifests/subjects_split.csv`.
- Alvo: y = log1p(SC/média) no triângulo superior. Entradas: FC, GD, MPC, prior do grupo, máscara inter-hemisférica
  e a previsão do B2 (modo residual: a rede aprende y − B2; última camada iniciada em zero).

## Aumento de dados (aplicado em TODAS as configurações, só nos sujeitos de treino de cada fold)
1. **FC reamostrada do BOLD**: bootstrap circular em blocos de 25 volumes (15 s) da série temporal do próprio
   sujeito; banco de 20 FCs por sujeito; a cada época cada sujeito de treino usa uma FC sorteada do banco.
2. **Ruído nas entradas**: gaussiano σ = 0,02 somado aos canais FC (arctanh) e GD (/100 mm).
- Espelhamento de hemisférios: NÃO usado (Schaefer-200 não é homóloga entre hemisférios).
- FCs reamostradas de um sujeito nunca saem do fold dele.

## Grade (8 configurações, fixa)
| fator | níveis |
|---|---|
| arquitetura do gerador | U-Net 2D (16 canais base) · BrainNetCNN (2 camadas aresta-a-aresta, 16 canais) |
| adversarial | λ_adv ∈ {0 (só reconstrução), 1 (GAN, PatchGAN com spectral norm)} |
| perda de reconstrução | L1 (λ = 100) · L1 (λ = 100) + correlação (λ_corr = 10 × (1 − r do desvio)) |
Fixos: Adam lr = 2e-4, β = (0,5, 0,999), batch 4, 200 épocas, avaliação a cada 10 épocas, 1 semente na busca interna.

## Procedimento aninhado
- **Externo**: 5 folds (10 sujeitos de teste cada), intocados durante a seleção.
- **Interno**: nos 40 sujeitos de treino de cada fold externo, 4 folds internos (10 de validação cada).
  Para cada configuração: B2 ajustado nos 30 de treino interno; a GAN treina nesses 30; a época é escolhida no
  fold de validação interno; o escore é o r do desvio (previsão final − μ vs verdade − μ) nesse fold.
- **Regra de seleção**: configuração com maior média do escore nos 4 folds internos (empate → mais simples:
  λ_adv = 0, depois L1 sem correlação, depois BrainNetCNN por ter menos parâmetros).
- **Treino final** de cada fold externo: configuração escolhida, 35 sujeitos de treino + 5 de validação para a
  época (sorteio fixo, semente 1000+fold), 3 sementes (0, 1, 2), média das previsões.

## Desfecho e teste (um único teste primário)
- **Primário**: r do desvio individual por sujeito, procedimento ajustado vs B2 ajustado nos mesmos 35 sujeitos,
  Wilcoxon pareado bilateral nos 50 sujeitos, α = 0,05.
- Secundários (descritivos, sem correção formal): r total, AUC das arestas fortes, identificação (acurácia e
  p binomial), configuração escolhida em cada fold, curvas de perda de G e D.
- Todas as 8 configurações têm seus escores internos reportados, não só a escolhida.

## Interpretação definida a priori
- Se p ≥ 0,05 ou o efeito for negativo: "com 50 sujeitos, nem a GAN ajustada com aumento de dados supera o
  modelo linear" (reforça o resultado negativo).
- Se p < 0,05 a favor da GAN: resultado exploratório promissor, a confirmar no HCP antes de qualquer afirmação.