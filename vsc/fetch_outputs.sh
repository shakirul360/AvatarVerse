#!/bin/bash
# RUN FROM YOUR LAPTOP (repo root). Pulls the final merged dataset (single/ + mixed/ + the two
# manifests) back from VSC scratch - not the whole avatarverse/out tree, which also has
# throwaway smoke-test/work directories.
set -euo pipefail
VSC=${VSC_HOST:-vsc}
# Concrete path, NOT $VSC_SCRATCH - same non-interactive-shell profile-sourcing gotcha as
# everywhere else in vsc/ (see RUNBOOK.md). vsc39181's $VSC_SCRATCH is /scratch/leuven/391/vsc39181.
SRC=${AVATARVERSE_OUT_REMOTE:-/scratch/leuven/391/vsc39181/avatarverse/out/final}
DEST=${1:-Datasets/THuman2.0/mos_dataset_textured_v1}
mkdir -p "$DEST"
rsync -av --progress "$VSC:$SRC/" "$DEST/"
echo "pulled into $DEST"
