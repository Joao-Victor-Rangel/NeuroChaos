#!/usr/bin/env bash
# Reconstrução das SCs das 50 pessoas do MICA-MICs com o pipeline próprio (modo sem eddy; 1 tractografia; SIFT2;
# Schaefer-400 volumétrico). Download em fluxo (só DWI + T1w run-1, SHA-256 conferido) em paralelo ao processamento.
#   bash scripts/dwi_mica50.sh
set -u
REPO="/mnt/c/Users/Joao Victor/Documents/Capitiria/NeuroChaos"
OUTN=dwi_mica50; OUT="$REPO/results/$OUTN"; BIDS="$HOME/dwi_work/bids"
mkdir -p "$OUT/_downloads" "$BIDS"
LOG="$OUT/_orquestrador.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# SCs já feitas com o mesmo pipeline na prévia (PREVIA=1, 2M fibras, semente 1) são reaproveitadas
for S in sub-HC003 sub-HC004; do
  if [ -s "$REPO/results/dwi_pilot_previa/$S/sc_seed1.csv" ] && [ ! -s "$OUT/$S/sc_seed1.csv" ]; then
    mkdir -p "$OUT/$S"; cp "$REPO/results/dwi_pilot_previa/$S/"{sc_seed1.csv,len_seed1.csv,atlas_dwi.nii.gz,mask.nii.gz} "$OUT/$S/" 2>/dev/null
    echo "reaproveitado da prévia (mesmo pipeline)" > "$OUT/$S/origem.txt"; log "$S reaproveitado da prévia"
  fi
done

# ---- downloader (em segundo plano)
(
  python3 - "$OUT/_downloads/lista.tsv" <<'PY'
import json, subprocess, sys, time
url = "https://api.osf.io/v2/nodes/z8mu5/files/osfstorage/?page[size]=100"
while True:  # o OSF às vezes devolve 502; tenta até responder
    out = subprocess.run(["curl", "-g", "-s", "--max-time", "120", url], capture_output=True, text=True).stdout
    try:
        d = json.loads(out); break
    except Exception:
        print("OSF indisponível; nova tentativa em 60 s", flush=True)
        time.sleep(60)
with open(sys.argv[1], "w") as f:
    for x in sorted(d["data"], key=lambda x: x["attributes"]["name"]):
        a = x["attributes"]
        f.write(f"{a['name']}\t{x['links']['download']}\t{a['extra']['hashes']['sha256']}\n")
PY
  while IFS=$'\t' read -r NAME LINK SHA; do
    FIRST=$(echo "$NAME" | sed -E 's/raw_HC([0-9]+)-.*/\1/')
    DONE_MARK="$OUT/_downloads/$NAME.ok"
    [ -f "$DONE_MARK" ] && continue
    if [ "$NAME" = "raw_HC001-HC005.tar.gz" ] && [ -d "$BIDS/sub-HC005" ]; then
      echo "lote extraído antes (SHA-256 conferido no piloto)" > "$DONE_MARK"; continue
    fi
    for TRY in 1 2 3; do
      log "baixando $NAME (tentativa $TRY)"
      curl -L -s --retry 5 -A "Mozilla/5.0" "$LINK" | tee >(sha256sum | cut -d" " -f1 > "$OUT/_downloads/$NAME.sha256") \
        | tar -xz -C "$BIDS" --wildcards "*/dwi/*" "*/anat/*run-1_T1w*"
      sleep 2
      GOT=$(cat "$OUT/_downloads/$NAME.sha256" 2>/dev/null)
      if [ "$GOT" = "$SHA" ]; then echo "$SHA" > "$DONE_MARK"; log "$NAME ok (SHA-256 confere)"; break; fi
      log "$NAME: SHA-256 diferente ($GOT vs $SHA); repetindo"
    done
    [ -f "$DONE_MARK" ] || { echo falhou > "$OUT/_downloads/$NAME.fail"; log "$NAME: FALHOU após 3 tentativas"; }
  done < "$OUT/_downloads/lista.tsv"
  log "downloads concluídos"
) &

# ---- processamento (um sujeito por vez, na ordem; espera o lote do sujeito ser conferido)
batch_of() { local n=$((10#${1#sub-HC})); local a=$(( (n-1)/5*5+1 )); printf "raw_HC%03d-HC%03d.tar.gz" $a $((a+4)); }
for N in $(seq 1 50); do
  S=$(printf "sub-HC%03d" $N)
  [ -s "$OUT/$S/sc_seed1.csv" ] && continue
  B=$(batch_of "$S")
  until [ -f "$OUT/_downloads/$B.ok" ] || [ -f "$OUT/_downloads/$B.fail" ]; do sleep 30; done
  if [ ! -d "$BIDS/$S/ses-01/dwi" ]; then log "$S: sem DWI no lote; pulando"; continue; fi
  log "$S: processando"
  OUTNAME=$OUTN PREVIA=1 SEEDS=1 CLEAN=1 bash "$REPO/scripts/dwi_pilot.sh" "$S" 2M 6 > "$HOME/dwi_work/run50_$S.log" 2>&1 \
    && log "$S: ok" || log "$S: FALHOU (ver ~/dwi_work/run50_$S.log)"
done
wait
log "FIM DAS 50"
