#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc39181
#SBATCH --partition=batch
#SBATCH --job-name=av-render
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/av-render-%A_%a.out
# Phase 3: full run, 6 subjects x 5 clips = 30 combos, one per array task (40 files each: 28
# single + 12 mixed). ~21-26 min/task observed in Phase 1/2 testing (macOS and VSC).
#
# Each task writes into its OWN subdirectory ($AVATARVERSE_OUT/final/task_<idx>/), not a shared
# one - generate_dataset.py's manifest write (`open(..., 'w')`) would clobber itself if multiple
# concurrent array tasks shared one --out directory. Run vsc/merge_outputs.py once ALL 30 tasks
# have completed (check with `sacct`) to combine into the final single/ + mixed/ layout.
#
#   cd $VSC_DATA/projects/AvatarVerse && mkdir -p logs && sbatch vsc/render_array.slurm
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

TASK_OUT="$AVATARVERSE_OUT/final/task_${SLURM_ARRAY_TASK_ID}"
WORK="$AVATARVERSE_OUT/work_${SUBJECT}_${CLIP}"
mkdir -p "$TASK_OUT" "$WORK"

python -m pipeline.generate_dataset --subjects "$SUBJECT" --clips "$CLIP" \
    --out "$TASK_OUT" --work-dir "$WORK"

rm -rf "$WORK"   # scratch buffers/decimation-cache, not needed after this task's renders exist
echo "task $SLURM_ARRAY_TASK_ID done"
