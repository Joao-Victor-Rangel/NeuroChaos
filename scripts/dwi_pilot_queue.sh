#!/usr/bin/env bash
# Fila do piloto DWI: espera não haver eddy rodando e processa os sujeitos restantes UM POR VEZ (limite de RAM do WSL).
set -u
REPO="/mnt/c/Users/Joao Victor/Documents/Capitiria/NeuroChaos"
while pgrep -f "dwi_pilot.sh sub-HC00[12]" > /dev/null; do sleep 120; done
for S in sub-HC003 sub-HC004 sub-HC005; do
  bash "$REPO/scripts/dwi_pilot.sh" "$S" 2M 6 > "$HOME/dwi_work/run_${S#sub-}.log" 2>&1 || echo "$S falhou" >> "$REPO/results/dwi_pilot/fila_erros.txt"
done
echo "fila concluída $(date)" >> "$REPO/results/dwi_pilot/fila_erros.txt"
