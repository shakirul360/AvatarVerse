#!/bin/bash
# RUN FROM YOUR LAPTOP (repo root). Pulls rendered videos + manifest back from VSC scratch.
set -euo pipefail
VSC=${VSC_HOST:-vsc}
SRC=${AVATARVERSE_OUT_REMOTE:-'$VSC_SCRATCH/avatarverse/out'}
DEST=${1:-Datasets/THuman2.0/mos_pilot_dataset_v5}
mkdir -p "$DEST"
rsync -av --progress "$VSC:$SRC/" "$DEST/"
echo "pulled into $DEST"
