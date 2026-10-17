#!/usr/bin/env bash
# Piloto de reprocessamento da difusão (docs/levantamento_reprocessamento_dwi.md) para UM sujeito.
# Pré-processamento completo (denoise, Gibbs, topup+eddy, viés), FODs multi-tecido e DUAS tractografias com
# sementes diferentes -> SC Schaefer-400 (pesos SIFT2) por semente: mede o ruído estocástico da tractografia.
#
#   bash scripts/dwi_pilot.sh sub-HC001 [n_fibras=2M] [threads=4]
set -euo pipefail
S="$1"; NSTREAM="${2:-2M}"; NT="${3:-4}"
REPO="/mnt/c/Users/Joao Victor/Documents/Capitiria/NeuroChaos"
ENV="$HOME/miniforge3/envs/neurochaos-dwi"
export PATH="$ENV/bin:$PATH" FSLDIR="$ENV" FSLOUTPUTTYPE=NIFTI_GZ MRTRIX_NTHREADS="$NT" OMP_NUM_THREADS="$NT"
# eddy na GPU quando disponível (wrapper em ~/dwi_work/bin); o dwifslpreproc recai na CPU se ele falhar
[ -x "$HOME/dwi_work/bin/eddy_cuda" ] && export PATH="$HOME/dwi_work/bin:$PATH"
B="$HOME/dwi_work/bids/$S/ses-01"
# PREVIA=1: versão rápida SEM topup/eddy, só para uma prévia do ruído da tractografia (saídas separadas)
# OUTNAME: pasta de saída alternativa (ex.: dwi_mica50) para a reconstrução das 50 SCs
if [ -n "${OUTNAME:-}" ]; then W="$HOME/dwi_work/proc_${OUTNAME}/$S"; OUT="$REPO/results/${OUTNAME}/$S"
elif [ "${PREVIA:-0}" = 1 ]; then W="$HOME/dwi_work/proc_previa/$S"; OUT="$REPO/results/dwi_pilot_previa/$S"
else W="$HOME/dwi_work/proc/$S"; OUT="$REPO/results/dwi_pilot/$S"; fi
SEEDS="${SEEDS:-1 2}"   # sementes de tractografia (o piloto usa 2 para medir o ruído; as 50 SCs usam 1)
mkdir -p "$W" "$OUT"; cd "$W"
log() { echo "[$(date +%H:%M:%S)] $S: $*" | tee -a "$OUT/log.txt"; }
TRT=0.05282   # TotalReadoutTime dos JSON (AP = j-, PA = j)
ATL="$HOME/dwi_work/atlases"   # cópia sem espaços no caminho (antsRegistrationSyNQuick.sh quebra com espaços)

log "início ($NSTREAM fibras por semente, $NT threads)"
if [ ! -f dwi_pre.mif ]; then
  if [ ! -f dwi_dg.mif ]; then   # etapas iniciais são reaproveitadas se já existirem
    for acq in b300-11 b700-41 b2000-91; do
      f="$B/dwi/${S}_ses-01_acq-${acq}_dir-AP_dwi"
      mrconvert -quiet -force "$f.nii.gz" -fslgrad "$f.bvec" "$f.bval" "${acq}.mif"
    done
    dwicat -quiet -force b300-11.mif b700-41.mif b2000-91.mif dwi_raw.mif
    dwidenoise -quiet -force dwi_raw.mif dwi_den.mif -noise noise.mif
    mrdegibbs -quiet -force dwi_den.mif dwi_dg.mif
  fi
  p="$B/dwi/${S}_ses-01_dir-PA_dwi"
  mrconvert -quiet -force "$p.nii.gz" -fslgrad "$p.bvec" "$p.bval" pa.mif
  dwiextract -quiet -force dwi_dg.mif -bzero - | mrmath -quiet -force - mean b0_ap.mif -axis 3
  dwiextract -quiet -force pa.mif -bzero - | mrmath -quiet -force - mean b0_pa.mif -axis 3
  mrcat -quiet -force b0_ap.mif b0_pa.mif b0_pair.mif -axis 3
  if [ "${PREVIA:-0}" = 1 ]; then
    log "PRÉVIA: sem topup/eddy"
    dwibiascorrect -quiet -force ants dwi_dg.mif dwi_pre.mif
  else
  log "topup + eddy"
  dwifslpreproc -force dwi_dg.mif dwi_ec.mif -rpe_pair -se_epi b0_pair.mif -pe_dir j- \
    -readout_time "$TRT" -align_seepi -eddy_options " --slm=linear --data_is_shelled --nthr=$NT" -scratch "$W/tmp_fsl" > "$OUT/dwifslpreproc.log" 2>&1
  dwibiascorrect -quiet -force ants dwi_ec.mif dwi_pre.mif
  fi
fi
log "máscara e FODs"
dwi2mask -quiet -force dwi_pre.mif mask.mif
dwi2response dhollander dwi_pre.mif wm.txt gm.txt csf.txt -force -quiet  # -force depois do algoritmo (subcomando)
dwi2fod -quiet -force msmt_csd dwi_pre.mif -mask mask.mif wm.txt wmfod.mif gm.txt gm.mif csf.txt csf.mif
mtnormalise -quiet -force wmfod.mif wmfod_n.mif gm.mif gm_n.mif csf.mif csf_n.mif -mask mask.mif
dwiextract -quiet -force dwi_pre.mif -bzero - | mrmath -quiet -force - mean meanb0.mif -axis 3
mrconvert -quiet -force meanb0.mif meanb0.nii.gz
mrconvert -quiet -force mask.mif mask.nii.gz

log "atlas Schaefer-400: MNI -> T1 (SyN) -> DWI (rígido)"
T1="$B/anat/${S}_ses-01_run-1_T1w.nii.gz"
if [ ! -f atlas_dwi.nii.gz ]; then
  N4BiasFieldCorrection -d 3 -i "$T1" -o t1_n4.nii.gz > /dev/null
  # Registro MNI (cabeça inteira) -> T1 (cabeça inteira): dispensa o bet, que deixava crânio/pescoço (~3 L)
  antsRegistrationSyNQuick.sh -d 3 -n "$NT" -f t1_n4.nii.gz \
    -m "$ATL/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz" -o mni2t1_ > /dev/null
  mrcalc -quiet -force "$ATL/tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz" 0 -gt mni_mask.nii.gz
  antsApplyTransforms -d 3 -n GenericLabel -r t1_n4.nii.gz -i mni_mask.nii.gz \
    -t mni2t1_1Warp.nii.gz -t mni2t1_0GenericAffine.mat -o t1_mask.nii.gz
  mrcalc -quiet -force t1_n4.nii.gz t1_mask.nii.gz -mult t1_brain.nii.gz
  VOL=$(mrstats -quiet t1_mask.nii.gz -output count -ignorezero)
  VOX=$(mrinfo t1_mask.nii.gz -spacing | awk '{print $1*$2*$3}')
  log "volume do cérebro (máscara MNI levada ao T1): $(awk -v n="$VOL" -v v="$VOX" 'BEGIN{printf "%.2f", n*v/1e6}') L"
  antsApplyTransforms -d 3 -n GenericLabel -r t1_n4.nii.gz \
    -i "$ATL/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz" \
    -t mni2t1_1Warp.nii.gz -t mni2t1_0GenericAffine.mat -o atlas_t1.nii.gz
  mrcalc -quiet -force meanb0.mif mask.mif -mult b0_brain.nii.gz
  antsRegistrationSyNQuick.sh -d 3 -t r -n "$NT" -f b0_brain.nii.gz -m t1_brain.nii.gz -o t12dwi_ > /dev/null
  antsApplyTransforms -d 3 -n GenericLabel -r meanb0.nii.gz -i atlas_t1.nii.gz -t t12dwi_0GenericAffine.mat \
    -o atlas_dwi.nii.gz
fi
NLAB=$(mrstats -quiet atlas_dwi.nii.gz -output max)
log "atlas no espaço da DWI: máximo rótulo = $NLAB"

for SEED in $SEEDS; do
  [ -f "$OUT/sc_seed${SEED}.csv" ] && continue
  log "tractografia semente $SEED"
  MRTRIX_RNG_SEED=$((1000 + SEED)) tckgen -quiet -force wmfod_n.mif tracks.tck -algorithm iFOD2 \
    -seed_dynamic wmfod_n.mif -select "$NSTREAM" -maxlength 250 -mask mask.mif
  tcksift2 -quiet -force tracks.tck wmfod_n.mif sift.txt
  tck2connectome -quiet -force tracks.tck atlas_dwi.nii.gz "$OUT/sc_seed${SEED}.csv" -tck_weights_in sift.txt \
    -symmetric -zero_diagonal
  tck2connectome -quiet -force tracks.tck atlas_dwi.nii.gz "$OUT/len_seed${SEED}.csv" -tck_weights_in sift.txt \
    -symmetric -zero_diagonal -scale_length -stat_edge mean
  rm -f tracks.tck   # tractograma intermediário é grande; o conectoma fica salvo
done
cp -f mask.nii.gz atlas_dwi.nii.gz "$OUT/" 2>/dev/null || true
cp -f t1_mask.nii.gz "$OUT/" 2>/dev/null || true
if [ "${CLEAN:-0}" = 1 ]; then
  # intermediários regeneráveis (~4-5 GB por sujeito), só depois de as saídas estarem copiadas
  for f in "$OUT"/sc_seed*.csv; do [ -s "$f" ] || { log "saída ausente; intermediários mantidos"; exit 1; }; done
  find "$W" -mindepth 1 -delete
  log "intermediários descartados"
fi
log "fim"
