# AvatarVerse — final dataset on VSC (wICE)

Runbook for generating the **textured** MOS pilot dataset on the KU Leuven VSC.
If you're a fresh Claude Code session opened over Remote-SSH: read this top to bottom, then
`git log` and the `pipeline/` sources for current state.

## The deliverable

3D-avatar MOS quality-assessment dataset (IMLEX internship). Pairwise-comparison stimuli:
distorted vs. reference short clips of an avatar in motion, for MOS collection + analysis.

- **Bodies:** real THuman2.1 textured scans (clothing, skin, hair), subjects `0000 / 0100 / 0500`.
- **Motion:** AMASS clips `walk` (ACCAD B3_walk1), `front_kick` (ACCAD G3_front_kick),
  `bmlmovi_walk` (BMLmovi Subject_1_F_19).
- **Skinning:** SMPL-X *surface wrap* — each scan vertex = barycentric point on the subject's
  posed SMPL-X surface + signed normal offset; re-posed by evaluating that against SMPL-X posed
  to each AMASS frame. Robust vs. LBS-weight propagation (no joint tearing). See
  `pipeline/skin_scan.py`.
- **Speed:** 0.75× real. AMASS is 120 fps → `stride = round(0.75 * 120 / 30) = 3`, played at 30 fps.
- **Render:** three.js + headless Chromium (SwiftShader, CPU). `pipeline/render.py`,
  `pipeline/three.min.js` vendored. Fixed tripod camera, ~52 % vertical fill, Z-up,
  1280×1280, H.264. (Plotly is retired — can't do UV textures, and the three.js renderer was
  independently shown to be more geometrically faithful; see notebook `09`.)
- **Loop:** ~46 unique frames (one gait cycle) × 3 → ~4.6 s clips.

### Distortions (9 types × 3 severities, per subject×clip)

Pose-space (apply to the AMASS pose before skinning — port unchanged from
`pipeline/generate_mos_pilot_dataset.py`): `jitter`, `joint_noise`, `frame_drop`, `smoothing`,
`param_quantization`.
Mesh/appearance-space: `vertex_quantization`, `motion_blur` (frame domain),
`mesh_decimation` (UV-aware, via `pymeshlab` quadric-collapse-with-texture),
`texture_compression` (JPEG-quality reduction of `material0.jpeg` — **new**, added because
texture quality is first-order for textured avatars).
→ trial pairs: 45 per (subject,clip) = 24 ref-vs-distorted + 21 adjacent-severity;
9 (subject,clip) combos → 405 pairs total. `pipeline/generate_trial_pairs.py` needs its
`DISTORTION_TYPES` + hardcoded `DATASET_DIR` updated for this.

## Status

- [x] Skinning + textured render proven locally (`pipeline/skin_scan.py`, `pipeline/render.py`)
      — one clip, `sample_textured_0000_walk_0.75x.mp4`.
- [ ] Phase 1: environment smoke test on wICE (`vsc/smoke_test.slurm`).
- [ ] Phase 2: port the 9 distortions to the textured path in a new
      `pipeline/generate_dataset.py` (reuse pose-space distortion fns from
      `generate_mos_pilot_dataset.py`).
- [ ] Phase 3: Slurm array over all subject×clip×distortion×severity + `fetch_outputs.sh`.

## VSC facts

- ID `vsc39181`, account `intro_vsc39181`, cluster **wICE**. Login `login.vscentrum.be`
  (lands on `tier2-p-login-*`).
- Verified working: `srun -M wice -A intro_vsc39181 -p interactive --cpus-per-task=4 --time=01:00:00 --pty bash`
- `$VSC_DATA` = `/data/leuven/391/vsc39181`, venv at `$VSC_DATA/venvs/avatar`.
- Module: `Python/3.13.5-GCCcore-14.3.0`. **Lmod is not initialised in `srun --pty` shells** —
  `vsc/env.sh` re-sources it.
- Compute nodes have **no internet** — all `pip install` / `playwright install` happen on the
  login node (`vsc/setup_env.sh`), browser cached at `$VSC_DATA/venvs/avatar/pw-browsers`.
- GPU exists (A100/H100) but this job is CPU (embarrassingly parallel, tiny per-frame compute).

## Data layout on VSC  (`$VSC_DATA/projects/avatarverse-data`, set by `vsc/env.sh`)

```
models/models/smplx/SMPLX_NEUTRAL.npz
thuman/{0000,0100,0500}/{<id>.obj, material0.jpeg, material0.mtl}
thuman/smplx/{0000,0100,0500}/{smplx_param.pkl, mesh_smplx.obj}
amass/ACCAD/.../*.npz   amass/BMLmovi/.../*.npz
```
Outputs → `$VSC_SCRATCH/avatarverse/out` (`AVATARVERSE_OUT`).
Code (`generate_mos_pilot_dataset.py`) reads `AVATARVERSE_MODELS/THUMAN/AMASS/OUT` env vars,
falling back to the laptop layout.

## Steps

### 0. Remote-SSH (laptop, once)
`~/.ssh/config`:
```
Host vsc
    HostName login.vscentrum.be
    User vsc39181
    IdentityFile ~/.ssh/<your-vsc-key>
    ServerAliveInterval 60
```
VS Code → Remote-SSH: Connect to Host → `vsc`. If your public IP changed, first whitelist it at
https://firewall.vscentrum.be .

### 1. Code onto VSC (login node)
```
mkdir -p $VSC_DATA/projects && cd $VSC_DATA/projects
git clone https://github.com/shakirul360/AvatarVerse.git
cd AvatarVerse
```

### 2. Build the env (login node — has internet)
```
bash vsc/setup_env.sh
```

### 3. Stage inputs (laptop)
```
bash vsc/stage_inputs.sh
```

### 4. Phase 1 smoke test (login node)
```
sbatch vsc/smoke_test.slurm
# watch: squeue --clusters=wice -u $USER ; tail -f av-smoke-*.out
```
Then from the laptop: `rsync -av vsc:'$VSC_SCRATCH/avatarverse/out/_smoke/*.mp4' .` and eyeball it.

### 5. Phases 2–3
Build `pipeline/generate_dataset.py` + `vsc/render_array.slurm`, then
`bash vsc/fetch_outputs.sh` and rerun `pipeline/generate_trial_pairs.py`.

## Quota

`quota -s` on VSC spews permission noise from other users' snapshots — ignore it. Use
`du -sh $VSC_DATA $VSC_SCRATCH` or the OnDemand dashboard. Footprint is ~250 MB in + ~1–3 GB out;
default wICE allocations (75 GB `$VSC_DATA`, large `$VSC_SCRATCH`) are plenty.
