#!/bin/bash
# RUN FROM YOUR LAPTOP (repo root). Pushes the ~250 MB of inputs the pipeline needs to VSC.
# Requires an ssh alias `vsc` -> login.vscentrum.be (see RUNBOOK.md).
set -euo pipefail

VSC=${VSC_HOST:-vsc}
# Concrete path, NOT $VSC_DATA - a non-interactive `ssh host "cmd"` doesn't reliably source the
# profile scripts that set $VSC_DATA (same class of bug the 2026-09-14 session hit with
# `srun --pty bash -c`; see RUNBOOK.md). vsc39181's $VSC_DATA is /data/leuven/391/vsc39181.
DEST=${AVATARVERSE_DATA_REMOTE:-/data/leuven/391/vsc39181/projects/avatarverse-data}

SUBJECTS=(0000 0100 0500)
AMASS_CLIPS=(
  "ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz"
  "ACCAD/Male2MartialArtsKicks_c3d/G3_-_front_kick_stageii.npz"
  "BMLmovi/Subject_1_F_MoSh/Subject_1_F_19_stageii.npz"
)

remote() { ssh "$VSC" "$1"; }

echo ">> creating remote tree"
remote "mkdir -p $DEST/models/models/smplx $DEST/thuman/smplx $DEST/amass"

echo ">> SMPL-X model"
rsync -av models/models/smplx/SMPLX_NEUTRAL.npz models/models/smplx/version.txt \
  "$VSC:$DEST/models/models/smplx/"

for s in "${SUBJECTS[@]}"; do
  echo ">> THuman subject $s (scan + texture + SMPL-X fit)"
  rsync -av "Datasets/THuman2.0/$s/" "$VSC:$DEST/thuman/$s/"
  remote "mkdir -p $DEST/thuman/smplx/$s"
  rsync -av "Datasets/THuman2.0/smplx/$s/" "$VSC:$DEST/thuman/smplx/$s/"
done

for c in "${AMASS_CLIPS[@]}"; do
  echo ">> AMASS $c"
  remote "mkdir -p $DEST/amass/$(dirname "$c")"
  rsync -av "Datasets/AMASS/$c" "$VSC:$DEST/amass/$c"
done

echo
echo "done. inputs staged under $DEST on VSC."
