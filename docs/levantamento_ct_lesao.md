# Levantamento — bases públicas para a vertente "CT → lesão → simulação" (17/10/2026)

Pergunta: existe base pública com CT **e** fMRI (ou desfecho clínico) nos mesmos pacientes, para testar
CT → lesão → conectoma de referência → simulação → consequência funcional?

## Conclusão curta
Não encontrei base pública com CT e fMRI de repouso nos mesmos pacientes. O desenho viável encadeia três bases,
cada uma validando um elo, com a máscara de lesão como "moeda comum" (ela não depende da modalidade).

| Elo | Base | O que tem | Tamanho / acesso (a confirmar no download) |
|---|---|---|---|
| 1. CT → lesão confiável | **AISD** | CT sem contraste + DWI em até 24 h; lesão anotada pela MRI | 397 casos; aberto |
| 1. CT → lesão confiável | **APIS** | CT sem contraste pareado com ADC; anotação na ADC | 96 estudos; CC BY 4.0 |
| 1b. CT → anatomia | **SynthRAD2023 (cérebro)** | CT + T1 pareados (pacientes de radioterapia) | 180; usado na validação do SynthSeg em CT (Dice mediano 0,76) |
| 2. CT → desfecho (o mais próximo da ideia) | **ISLES'24** | CT sem contraste, angio-CT, perfusão-CT; MRI de seguimento (DWI/ADC) com máscara de infarto; NIHSS na admissão; **mRS em 3 meses** | 149 casos públicos (treino); CC BY-NC-SA 4.0 (Zenodo / Hugging Face) |
| 2b. Escala (só MRI) | **SOOP** | MRI aguda, máscaras de lesão, NIHSS | ~1.700 pacientes; OpenNeuro |
| 3. Lesão → mudança real da FC | **ARC (Aphasia Recovery Cohort)** | AVC crônico; T1/T2, lesões desenhadas à mão, DWI e **fMRI de repouso** em subconjunto, escores de afasia | 230 pacientes; OpenNeuro |

## Desenho proposto
1. **Elo 1:** segmentar a lesão na CT (AISD/APIS) e comparar com a máscara da MRI (Dice, volume, regiões
   atingidas no Schaefer). Critério: as regiões atingidas pela lesão da CT batem com as da MRI.
2. **Elo 2 (ISLES'24):** lesão → conectoma de referência (média de pessoas saudáveis; nosso achado de que a SC do
   grupo serve tão bem quanto a individual justifica isso) → simulação (Hopf; depois NEURON) com e sem a lesão →
   índice de dano dinâmico → prever NIHSS/mRS. **Linhas de base obrigatórias:** volume da lesão e "lesion network
   mapping" sem dinâmica. A simulação só vale se superar essas linhas de base.
3. **Elo 3 (ARC):** a lesão virtual prevê a FC real do paciente (versus a FC de controles)? É o teste de que a
   dinâmica simulada corresponde à realidade.

## Riscos
- ISLES'24 tem n = 149 e mRS é grosseiro; o volume da lesão sozinho costuma prever bem o desfecho — superar isso é
  difícil.
- No ISLES'24 a MRI de seguimento é pós-tratamento (2–9 dias).
- ARC é afasia (lesões no território da artéria cerebral média esquerda): bom para linguagem, limitado em diversidade.
- Licenças: CC BY-NC-SA (ISLES'24) permite uso acadêmico; conferir termos de cada base antes de baixar.

## Fontes
- AISD/APIS/ISLES (visão geral): https://arxiv.org/pdf/2301.06793 ; https://github.com/m-aryayi/Medical-Imaging-Datasets
- ISLES'24: https://arxiv.org/pdf/2408.10966 ; https://zenodo.org/records/16748089 ;
  https://huggingface.co/datasets/hugging-science/isles24-stroke
- SynthSeg em CT com MRI pareada: https://arxiv.org/abs/2406.17423v1
- ARC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11384737/
- SOOP: https://pmc.ncbi.nlm.nih.gov/articles/PMC11297183