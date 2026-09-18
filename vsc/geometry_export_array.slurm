#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc39181
#SBATCH --partition=batch
#SBATCH --job-name=av-geom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/av-geom-%A_%a.out
# Phase 3 (interactive-viewer track): full run, 6 subjects x 5 clips = 30 combos, one per array
# task (37 instances each: 1 reference + 24 single + 12 mixed - see distortions.py, motion_blur
# dropped for this track).
#
# --time=06:00:00, NOT a quick job despite no Playwright/Chromium: measured directly on wICE
# (job 62064715, node s28c11n1) at ~4.2 min/instance average for the single-distortion batch -
# 2x slower than pipeline.metrics.compute_geometry_metrics()'s own isolated benchmark (41s/
# instance there vs an implied much larger per-instance cost in the real job). Best working
# explanation: writing ~48 small files/instance (base.drc + ~45 frame .drc files + texture +
# meta) to $VSC_SCRATCH's shared filesystem has real per-file metadata overhead that neither
# earlier isolated test exercised (the single-instance srun test wrote to local /tmp; the
# metrics benchmark never wrote a bundle at all). Full combo extrapolates to ~2.5-3h - 6h leaves
# real margin given that extrapolation is unconfirmed at the time this was set. Revisit this
# once a full combo actually completes and the true number is in (see vsc/RUNBOOK.md).
#
# Same per-task-subdirectory reasoning as render_array.slurm: each task writes its own
# $AVATARVERSE_OUT/final_geom/task_<idx>/ so concurrent tasks' manifest_geometry.csv writes
# don't clobber each other. Run vsc/merge_geometry_outputs.py once ALL 30 tasks have completed
# (check with `sacct`) to combine into the final single/ + mixed/ layout.
#
#   cd $VSC_DATA/projects/AvatarVerse && mkdir -p logs && sbatch vsc/geometry_export_array.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source vsc/env.sh

SUBJECTS=(0000 0100 0500 0450 0150 0250)
CLIPS=(walk front_kick bmlmovi_walk run pickup_box)
N_CLIPS=${#CLIPS[@]}

SUBJECT_IDX=$(( SLURM_ARRAY_TASK_ID / N_CLIPS ))
CLIP_IDX=$(( SLURM_ARRAY_TASK_ID % N_CLIPS ))
SUBJECT=${SUBJECTS[$SUBJECT_IDX]}
CLIP=${CLIPS[$CLIP_IDX]}

echo "task $SLURM_ARRAY_TASK_ID -> subject=$SUBJECT clip=$CLIP"

TASK_OUT="$AVATARVERSE_OUT/final_geom/task_${SLURM_ARRAY_TASK_ID}"
WORK="$AVATARVERSE_OUT/work_geom_${SUBJECT}_${CLIP}"
mkdir -p "$TASK_OUT" "$WORK"

python -m pipeline.generate_geometry_dataset --subjects "$SUBJECT" --clips "$CLIP" \
    --out "$TASK_OUT" --work-dir "$WORK"

rm -rf "$WORK"   # scratch decimation/texture cache, not needed after this task's bundles exist
echo "task $SLURM_ARRAY_TASK_ID done"
