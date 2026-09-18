#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc39181
#SBATCH --partition=batch
#SBATCH --job-name=av-geom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/av-geom-%A_%a.out
# Phase 3 (interactive-viewer track): full run, 6 subjects x 5 clips = 30 combos, one per array
# task (37 instances each: 1 reference + 24 single + 12 mixed - see distortions.py, motion_blur
# dropped for this track). No Playwright/Chromium in this job at all - export is pure NumPy/
# Draco, so expect this to run noticeably faster per-task than render_array.slurm's video job
# (~21-26 min/task) - see vsc/RUNBOOK.md for the actual number once Phase 2's local timing run
# (pipeline/metrics.py's redundant-cKDTree fix applied) confirms it; --time above is a
# conservative placeholder carried over from the video job until that's in.
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
