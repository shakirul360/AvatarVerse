#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc39181
#SBATCH --partition=batch
#SBATCH --job-name=av-phase2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# Phase 2 validation: one full (subject,clip) combo - all 40 files (28 single + 12 mixed) -
# proving the pymeshlab mesh_decimation + warmup-race fixes hold on VSC's Linux SwiftShader,
# not just macOS. Every frame of every file gets checked, not spot-checks (see RUNBOOK.md).
#   cd $VSC_DATA/projects/AvatarVerse && sbatch vsc/phase2_smoke_test.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source vsc/env.sh

WORK="$AVATARVERSE_OUT/_phase2_smoke"
rm -rf "$WORK"
mkdir -p "$WORK"

echo "=== generate ==="
python -m pipeline.generate_dataset --subjects 0000 --clips walk --out "$WORK/out" --work-dir "$WORK/work"

echo "=== validate every frame of every file ==="
python - <<PY
import imageio.v2 as imageio, numpy as np, glob, os
work = "$WORK"
files = sorted(glob.glob(f'{work}/out/single/*.mp4') + glob.glob(f'{work}/out/mixed/*.mp4'))
print(f'checking {len(files)} files, all frames each')
bad = []
for fp in files:
    r = imageio.get_reader(fp)
    fps = r.get_meta_data().get('fps')
    frames = [f for f in r]
    n = len(frames)
    whites = np.array([(f.mean(axis=2) > 240).mean() for f in frames])
    blacks = np.array([(f.mean(axis=2) < 8).mean() for f in frames])
    issues = []
    if fps is None or abs(fps - 30) > 0.5: issues.append(f'BAD_FPS={fps}')
    if (whites > 0.5).any(): issues.append(f'WHITE_FRAMES={int((whites > 0.5).sum())}/{n}')
    if (blacks > 0.999).any(): issues.append(f'BLANK_FRAMES={int((blacks > 0.999).sum())}/{n}')
    if n < 130: issues.append(f'TOO_FEW_FRAMES={n}')
    if issues:
        bad.append((os.path.basename(fp), issues))
print(f'{len(bad)} problem files out of {len(files)}')
for name, issues in bad:
    print(' ', name, issues)
assert not bad, f'{len(bad)} files failed validation'
print('ALL CLEAN')
PY
echo "=== done ==="
