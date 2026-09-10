#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc39181
#SBATCH --partition=batch
#SBATCH --job-name=av-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# Phase 1: prove the environment. Skins + renders ONE textured walk clip on a wICE node.
#   cd $VSC_DATA/projects/AvatarVerse && sbatch vsc/smoke_test.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source vsc/env.sh

WORK="$AVATARVERSE_OUT/_smoke"
mkdir -p "$WORK"

echo "=== skin ==="
python -m pipeline.skin_scan --subject 0000 --clip walk --speed 0.75 --out "$WORK/buf"

echo "=== render ==="
python -m pipeline.render --buffers "$WORK/buf" --out "$WORK/0000_walk_0.75x.mp4" --loops 3

echo "=== result ==="
ls -la "$WORK/0000_walk_0.75x.mp4"
python - <<PY
import imageio.v2 as iio
r = iio.get_reader("$WORK/0000_walk_0.75x.mp4")
m = r.get_meta_data()
print("size", m.get("size"), "fps", m.get("fps"), "frames", r.count_frames())
PY
echo "OK - fetch with:  rsync -av vsc:$WORK/0000_walk_0.75x.mp4 ."
