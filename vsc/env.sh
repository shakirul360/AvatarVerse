# Sourced by every VSC job script and interactive session before running the pipeline.
#   source vsc/env.sh
# Assumes it is run from the repo root ($AVATARVERSE_REPO) or that AVATARVERSE_REPO is set.

: "${AVATARVERSE_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export AVATARVERSE_REPO

# --- Lmod: not initialised in srun --pty shells, so pull it in defensively ---
if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh \
             /usr/share/lmod/lmod/init/bash /apps/leuven/etc/profile.d/lmod.sh; do
        [ -f "$f" ] && source "$f" && break
    done
fi
module --quiet purge 2>/dev/null || true
module load Python/3.13.5-GCCcore-14.3.0

# --- data locations on VSC ($VSC_DATA persists, $VSC_SCRATCH is fast + purged) ---
export AVATARVERSE_DATA="${AVATARVERSE_DATA:-$VSC_DATA/projects/avatarverse-data}"
export AVATARVERSE_MODELS="$AVATARVERSE_DATA/models/models"
export AVATARVERSE_THUMAN="$AVATARVERSE_DATA/thuman"
export AVATARVERSE_AMASS="$AVATARVERSE_DATA/amass"
export AVATARVERSE_OUT="${AVATARVERSE_OUT:-$VSC_SCRATCH/avatarverse/out}"
mkdir -p "$AVATARVERSE_OUT"

# --- python env + playwright browser cache (must be visible from compute nodes) ---
export AVATARVERSE_VENV="${AVATARVERSE_VENV:-$VSC_DATA/venvs/avatar}"
export PLAYWRIGHT_BROWSERS_PATH="$AVATARVERSE_VENV/pw-browsers"
[ -f "$AVATARVERSE_VENV/bin/activate" ] && source "$AVATARVERSE_VENV/bin/activate"

export PYTHONUNBUFFERED=1
