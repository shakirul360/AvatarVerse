#!/bin/bash -l
# ONE-TIME, ON A VSC LOGIN NODE (has internet; compute nodes do not).
#   cd $VSC_DATA/projects/AvatarVerse && bash vsc/setup_env.sh
set -euo pipefail

source "$(dirname "$0")/env.sh"

if [ ! -d "$AVATARVERSE_VENV" ]; then
    echo ">> creating venv at $AVATARVERSE_VENV"
    python -m venv "$AVATARVERSE_VENV"
fi
source "$AVATARVERSE_VENV/bin/activate"
python -m pip install --upgrade pip wheel

echo ">> torch (CPU build)"
pip install --index-url https://download.pytorch.org/whl/cpu torch

echo ">> pipeline requirements"
pip install -r "$AVATARVERSE_REPO/pipeline/requirements.txt"

echo ">> chromium for playwright (into $PLAYWRIGHT_BROWSERS_PATH)"
python -m playwright install chromium

echo ">> smoke-import"
python - <<'PY'
import numpy, scipy, torch, smplx, trimesh, PIL, imageio, playwright
print("imports OK  torch", torch.__version__, "cpu" if not torch.cuda.is_available() else "cuda")
PY

echo
echo "done. next: bash vsc/stage_inputs.sh   (run that from your LAPTOP, not here)"
