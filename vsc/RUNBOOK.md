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
- [x] Phase 1: environment smoke test on wICE (`vsc/smoke_test.slurm`) — job 62007832,
      `0000_walk_0.75x.mp4` (1280×1280, 30fps, 138 frames), copied to
      `avatarverse-data/smoke_output/` for laptop-side viewing. See below.
- [ ] Phase 2: port the 9 distortions to the textured path in a new
      `pipeline/generate_dataset.py` (reuse pose-space distortion fns from
      `generate_mos_pilot_dataset.py`).
- [ ] Phase 3: Slurm array over all subject×clip×distortion×severity + `fetch_outputs.sh`.

### 2026-09-14 session notes

- The venv at `$VSC_DATA/venvs/avatar` (created 2026-07-29) was stale/broken: bound to the
  **icelake** Python build but only `pip` itself was ever installed — `setup_env.sh` had never
  actually completed. Deleted and rebuilt from scratch on a **skylake** login node
  (`tier2-p-login-2`); this now works on both skylake login nodes and the icelake `batch`
  partition nodes (icelake is a superset of skylake's instruction set — build on the older/lower
  arch, never the newer one, if you land on a different login node next time).
- **Login-node vs. compute-node CPU arch mismatch is real and will bite you again**: the login
  pool (`login.hpc.kuleuven.be` → currently `login-genius`, internal hostname
  `tier2-p-login-2`) is Skylake (Xeon Gold 6140). The default `batch`/`interactive` partitions on
  wICE are **icelake**. A venv (or anything with compiled/arch-tuned binaries) built while
  `srun`'d onto an icelake compute node will `Illegal instruction` crash if later run directly on
  a skylake login node, and vice versa isn't a problem. Always build the venv on whatever login
  node you're currently on — don't build it inside an interactive compute-node session.
- Playwright's bundled Chromium ("BEWARE: your OS is not officially supported... downloading
  fallback build for ubuntu24.04-x64") **will crash with `SIGTRAP`/`int3` if launched directly on
  the login node** — its PartitionAlloc tries to reserve ~48 GB of address space (mostly just
  reservation, not RSS) and the login node hard-caps `ulimit -v` to ~47.6 GB (soft==hard, not
  raisable). This is **not a real problem for the actual pipeline** — confirmed `ulimit -v` is
  `unlimited` on the `interactive`/`batch` compute nodes where rendering actually runs. Don't run
  `pipeline.render` (or any playwright smoke test) directly on a login node; only via
  `srun`/`sbatch`.
- **If driving `srun --pty` from a session with no real TTY** (e.g. this Claude Code session):
  SLURM prints `Not using a pseudo-terminal, disregarding --pty option` and falls back to a
  plain non-interactive shell, which on this site does **not** source the profile.d scripts that
  set `$VSC_DATA`/`$VSC_SCRATCH`/`MODULEPATH`. Fix: use `srun ... --pty bash -l -c '...'` (force
  a login shell) instead of bare `--pty bash -c '...'`. `sbatch` scripts are unaffected — they
  already use the `#!/bin/bash -l` shebang.
- Data staged (241 MB via `vsc/stage_inputs.sh` from the laptop): all 3 THuman subjects
  (scan+texture+SMPL-X fit), all 3 AMASS clips, the SMPL-X model.
- Two dependency bugs found by actually running the job (not caught by the local proof-of-concept
  since that ran with an already-complete laptop env):
  - `skin_scan.py` imports `generate_mos_pilot_dataset.py` for its `THUMAN_ROOT`/`AMASS_ROOT`/
    `SMPLX_MODEL_DIR`/`CLIPS` constants, but that module had a top-level `import plotly` — a
    dependency the "Plotly is retired" note above says is gone. Fixed: made the plotly imports
    lazy (moved inside the two legacy plotly-rendering functions, which the textured path never
    calls) instead of reinstalling a retired dependency.
  - `trimesh`'s `nearest.on_surface()` (used by the surface-wrap skinning) needs `rtree` for its
    spatial index; it's an optional trimesh extra that wasn't in `pipeline/requirements.txt`.
    Added `rtree>=1.2` there and installed it into the venv.
- Phase 1 passed end-to-end on job 62007832 (icelake `batch` node `s28c11n3`): skin 302,021 verts
  / 46 unique frames in ~18s, render 46 captured frames in 25s → looped 138-frame
  1280×1280@30fps `.mp4`. Output copied to `avatarverse-data/smoke_output/0000_walk_0.75x.mp4`
  (inside the `$VSC_DATA`-mounted tree, so it's visible from the laptop side without `rsync`).
  Minor cosmetic-only ffmpeg warning ("Multiple -pix_fmt options specified") in `render.py` —
  didn't affect output, not yet investigated.

## VSC facts

- ID `vsc39181`, account `intro_vsc39181`, cluster **wICE**. Login `login.hpc.kuleuven.be`
  (`login.vscentrum.be` doesn't resolve - stale/wrong hostname, don't use it) - lands on
  `tier2-p-login-*`, a shared KU Leuven Tier-2 login pool; pick the cluster per-job via `-M`.
- IP-whitelist firewall (`firewall.vscentrum.be`) gates the login node - if `ssh` hangs or is
  refused, re-whitelist your current public IP there first (it prompts with a URL on connect).
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

### 0. Remote-SSH (laptop, once) — already done, kept here for reference
`~/.ssh/config`:
```
Host vsc
    HostName login.hpc.kuleuven.be
    User vsc39181
    IdentityFile ~/.ssh/id_rsa_vsc
    ServerAliveInterval 60
```
Test with `ssh vsc` before trying VS Code — if it hangs/refuses, whitelist your current IP at
`firewall.vscentrum.be` first (your IP changes when you switch networks, e.g. hotspot <-> wifi).
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
bash -l vsc/setup_env.sh
```
**Must be `bash -l`, not plain `bash`.** `$VSC_DATA`/`$VSC_SCRATCH` and Lmod's `cluster/wice/*`
context (itself a module you must load before `Python/3.13.5-GCCcore-14.3.0` becomes visible -
`module --quiet purge` wipes it) both come from a profile.d chain that only runs for an actual
login shell - naming `bash` explicitly on the command line bypasses the script's own
`#!/bin/bash -l` shebang, so `bash vsc/setup_env.sh` silently loses all of it (found the hard
way: `bash vsc/setup_env.sh` had apparently "worked" for Phase 1 only because that session's
login node happened to already have the right modules loaded by chance, not because the
invocation was actually correct). Same rule for ANY ad-hoc one-off command on a login node
that needs `$VSC_DATA`/`module` - wrap it in `bash -l -c '...'`. `sbatch` scripts are unaffected
(Slurm execs them directly, honoring the shebang).

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

### 5. Phase 2 (code done, needs validating on VSC hardware)
`pipeline/distortions.py` + `pipeline/generate_dataset.py` exist and pass full per-frame
validation locally (macOS) for one full (subject,clip) combo (40 files: 28 single + 12 mixed).
Two real rendering bugs were found and fixed there (degenerate-triangle NaN normals from
aggressive mesh_decimation; a SwiftShader texture-upload warmup race producing a leading run of
blank frames) - both fixes are defensive/general (sanitize NaN normals regardless of cause,
retry any blank frame regardless of cause), so they should hold on VSC's Linux SwiftShader too,
but the warmup race in particular is timing-dependent and hasn't been proven there yet. Before
the real 30-combo run: `pip install pymeshlab` into the venv (not yet installed on VSC - only
`rtree` was needed for Phase 1), then rerun the same one-combo validation
(`python -m pipeline.generate_dataset --subjects 0000 --clips walk --out <scratch dir>`) via
`sbatch`, and re-check every frame of every file the same way the local validation did before
trusting a full run.

### 6. Phase 3
`vsc/render_array.slurm` (not yet written) + `bash vsc/fetch_outputs.sh` (update its
`AVATARVERSE_OUT_REMOTE` for the `single`/`mixed` split), then rerun
`pipeline/generate_trial_pairs.py` (needs updating for the 9-type + 6-mixed-pair catalog and
the two-folder layout - still has the old 8-type/single-folder assumptions).

## Getting code updates onto VSC

VSC has no GitHub credentials configured (the initial `git clone` is a public, read-only HTTPS
clone - fine; `git fetch`/`git pull`/`git push` against `github.com` all fail with
`could not read Username`). Updates flow laptop -> VSC directly over SSH instead, bypassing
GitHub: from the laptop, `git push vsc:$VSC_DATA/projects/AvatarVerse main` (needs
`receive.denyCurrentBranch=updateInstead` set once on the VSC-side repo, already done - it
updates VSC's working tree files too, not just the git history, so no separate `git reset --hard`
needed after). If a fresh VSC-side session makes local commits there, pull them the same way in
reverse: from the laptop, `git fetch vsc:$VSC_DATA/projects/AvatarVerse main` then
`git merge --ff-only FETCH_HEAD`, then push to `origin` from the laptop (which does have
credentials).

## Quota

`quota -s` on VSC spews permission noise from other users' snapshots — ignore it. Use
`du -sh $VSC_DATA $VSC_SCRATCH` or the OnDemand dashboard. Footprint is ~250 MB in + ~1–3 GB out;
default wICE allocations (75 GB `$VSC_DATA`, large `$VSC_SCRATCH`) are plenty.
