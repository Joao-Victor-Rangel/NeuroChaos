# Adendo ao pré-registro — escada de modelos, curva de aprendizado, conectividade efetiva e caos

**Escrito em 15/10/2026, antes de executar qualquer análise aqui descrita.** Hash SHA-256 gravado em
`results/escada/` no início. Vale o mesmo status do pré-registro principal: no MICA-MICs tudo é
**exploratório/desenvolvimento**; a confirmação é no HCP, com configurações congeladas.

Comuns a todos os modelos de aprendizado: Schaefer-200, folds externos de `subjects_split.csv`, alvo
y = log1p(SC/média), métrica primária = r do desvio individual (previsão − μ vs verdade − μ), identificação
(acurácia e p binomial) como secundária; seleção sempre por validação interna (4 folds nos 40 de treino).

## A. PLS de kernel (degrau 2)
- Entradas por sujeito (desvios em relação à média do treino, padronizados): FC (arctanh), GD, MPC.
- Alvo: desvio y − μ. Kernel linear; PLS de kernel (Rännar et al., 1994).
- Grade: nº de componentes ∈ {1, 2, 3, 5, 10}; escolha pela média do r do desvio nos 4 folds internos.
- Teste primário A: PLS vs B2, Wilcoxon pareado no r do desvio (50 sujeitos).

## B. Rede de grafos — GNN (degrau 3)
- Nós = 200 regiões; atributos do nó = linhas de FC, GD e MPC do sujeito + embedding aprendido por região (16).
- Codificador: MLP + camadas de passagem de mensagem com adjacência = |FC| normalizada por linha (k camadas).
- Decodificador de aresta: MLP simétrico sobre [h_i ⊙ h_j, |h_i − h_j|, FC_ij, GD_ij, desvio do B2_ij].
- Modo residual sobre o B2 (última camada zerada; época 0 = B2), aumento de dados idêntico ao da GAN.
- Grade (8): dimensão oculta ∈ {32, 64} × camadas de mensagem ∈ {1, 2} × λ_corr ∈ {0, 10}; L1 λ = 100,
  Adam 2e-4, 200 épocas. Treino final: 35 + 5 validação, 3 sementes.
- Teste primário B: GNN vs B2 (Wilcoxon pareado, r do desvio).

## C. Curva de aprendizado
- B2 e PLS treinados com N ∈ {10, 20, 30, 40} sujeitos sorteados do treino de cada fold externo
  (5 sorteios por N, sementes fixas); avaliação no teste externo.
- Leitura: se o r do desvio cresce sem estabilizar até N = 40 → mais dados devem ajudar; se estabiliza → limite
  de sinal. Sem teste formal; inclinação e IC bootstrap reportados.

## D. Conectividade efetiva (EC) pelo modelo de Hopf linearizado (degrau 5)
- Modelo de Hopf linearizado em torno do ponto fixo (a = −0,02, ω_i do BOLD da 1ª metade): covariância
  analítica pela equação de Lyapunov; FC do modelo = correlação da parte real.
- Início: SC média do grupo de treino (escalada para máx 0,2), G fixo = 1. Atualização iterativa
  C_ij ← max(0, C_ij + ε (FC_emp − FC_mod)_ij) só nas arestas com SC média > 0; ε = 0,01; 300 iterações.
- Ajuste com a FC da 1ª metade do BOLD; avaliação: (i) FC do modelo × FC da 2ª metade; (ii) estrutura:
  r do desvio entre EC e SC individual, e identificação EC → SC (mesma métrica dos modelos de aprendizado).
- Teste primário D: identificação EC vs acaso (binomial) e r do desvio EC vs B2 (Wilcoxon).

## E. Caos na rede cortical com conectoma real (secundário, exploratório)
- 10 sujeitos (os mesmos da primeira rodada cortical) × {SC própria, embaralhada}; rede cortical CoreNEURON,
  K = 2, sem estímulo; trajetórias gêmeas com ε ∈ {1e-6, 1e-4, 1e-2} nA, t_perturb ∈ {3000, 4000, 5000} ms,
  horizonte 1000 ms. Desfechos: λ linear e probabilidade de divergência finita. Wilcoxon pareado, sem correção
  formal (exploratório).

## Correção para múltiplos testes
- Testes primários A, B e D (3 comparações com o B2) corrigidos por Holm. A GAN segue seu próprio pré-registro.
## Emenda 1 (15/10/2026, antes da execução completa da seção D)
Motivo, observado num único sujeito e APENAS no ajuste à FC da 1ª metade (dados de ajuste, nenhuma métrica de
avaliação consultada): (i) o ajuste não é monotônico — máximo em ~20 iterações (r = 0,595), queda lenta depois;
(ii) cada iteração custa segundos (equação de Lyapunov 400×400), tornando 300 iterações × 50 sujeitos inviável.
Mudança: n_iter = 100 e a EC usada é a da iteração com MAIOR correlação com a FC da 1ª metade. Todo o resto
(ε, início, máscara, avaliação na 2ª metade e contra a SC) permanece como pré-registrado.
