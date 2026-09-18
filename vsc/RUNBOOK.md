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

### 5. Phase 2 - DONE, validated on both macOS and VSC
`pipeline/distortions.py` + `pipeline/generate_dataset.py` pass full per-frame validation (every
frame of every file, not spot-checks) for one full (subject,clip) combo (40 files: 28 single +
12 mixed) on both macOS (locally) and VSC (`vsc/phase2_smoke_test.slurm`, job 62008646, 0/40
problem files, 25m38s). `pymeshlab` was already present on VSC (pulled in by
`pipeline/requirements.txt` during Phase 1's `setup_env.sh`, even though Phase 1 itself never
used it). The two rendering bugs found locally (degenerate-triangle NaN normals from aggressive
mesh_decimation; a SwiftShader texture-upload warmup race producing a leading run of blank
frames) do NOT recur on VSC's Linux SwiftShader - confirms both fixes were made at the right
level (defensive/general: sanitize NaN normals and retry blank frames regardless of cause)
rather than papering over something macOS-specific.

### 6. Phase 3 - DONE

Job 62009167 (30-task array, `vsc/render_array.slurm`): all 30 tasks COMPLETED, 0 failures,
~25 min total wall-clock (all tasks scheduled concurrently - no queueing). `vsc/merge_outputs.py`
combined them: 840 single + 360 mixed = 1,200 files, 30/30 combos with exact expected counts
(28+12 each). Validated: 80-file full-frame random sample, 0 problems. Pulled to the laptop via
`vsc/fetch_outputs.sh` (845 MB) -> `Datasets/THuman2.0/mos_dataset_textured_v1/`.
`pipeline/generate_trial_pairs.py` -> 1,710 trial pairs, every video_a/video_b reference
verified to exist on disk.

**The deliverable is complete**: 6 subjects x 5 motions x (9 single-distortion types x 3
severities + 6 mixed pairs x 2 severity presets) = 1,200 videos, 1,710 pairwise-comparison
trial pairs, at `Datasets/THuman2.0/mos_dataset_textured_v1/{single,mixed}/` +
`trial_pairs.csv`. Next step from here is handing this to whatever tool actually runs the MOS
study session (the professor's reference pairwise-comparison tool) - not yet wired up.

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

## Interactive-viewer track (2026-09-18/19): pairwise video retired, single-stimulus + free-orbit 3D adopted

New track, parallel to the completed video pipeline above (that deliverable is left untouched -
see `pipeline/generate_geometry_dataset.py`'s docstring). Participants now rate ONE live,
free-orbit interactive 3D avatar per trial (1-5 ACR) instead of comparing two videos -
Bradley-Terry pairwise ranking is retired for this track. Mixed (compound) distortions are kept
as their own single-stimulus items. `motion_blur` is dropped entirely (no 3D-geometry
equivalent - see `distortions.py`'s `FRAME_LEVELS` comment) - catalog is 8 distortion types now,
37 instances/combo (1 reference + 8x3 + 6x2).

**Phase 0 (de-risking, DONE)**: real Draco round-trip spike on subject `0000` (302,021 verts)
found point-cloud compression only hits 2.4-3.2x (not the 5-20x mesh-connectivity-prediction
number that assumes topology is present), because per-frame data has no connectivity to predict
from. Validated fix: delta-from-frame-0 point-cloud encoding (deltas cluster near zero) +
absolute-position mesh encoding for frame 0 only. Final recipe: `base.drc` (connectivity+UV+
frame-0, 14-bit, ~8.2MB) + `frames/*.drc` (per-frame deltas, 10-bit, ~800KB avg) = **~45MB/
instance**. User explicitly accepted this bandwidth cost (~900MB-1.1GB at the chosen 20-instance
session cap) over a uniform delivery-resolution cap or reverting to a multi-angle video switcher,
after seeing all three options' real numbers.

**Phase 1 (one instance, DONE)**: `pipeline/geometry_export.py` implements the recipe above.
`viewer/` (three.js r160 ES modules - OrbitControls/DRACOLoader are jsm-only by r160, confirmed;
vendored via `npm install three@0.160.1`, not CDN-loaded, same reasoning as the vendored
`three.min.js`) live-decodes and animates a bundle with free camera control. Verified in a real
Chromium via Playwright: correct front-facing default view (the pending "avatar faces backward"
fix landed here - one-line camera-eye flip in `viewer.js`), orbit confirmed by dragging to the
back view, zero console errors, no NaN/degenerate-triangle artifacts.

**Phase 2 (one full combo pilot, DONE)**: `pipeline/metrics.py` (Chamfer/Hausdorff via
`scipy.spatial.cKDTree`, same nearest-neighbor pattern `distortions.build_decimation()` already
used) + `pipeline/generate_geometry_dataset.py` (mirrors `generate_dataset.py`'s STAGE_OF
sequencing, swaps `renderer.render()` for `geometry_export.write_geometry_bundle()` +
`metrics.compute_geometry_metrics()`). Ran for real: subject `0000`/`walk`, 37/37 instances,
1.3GB total. Sanity-checked: reference reads exactly 0.0 chamfer/hausdorff (comparing to
itself); jitter's chamfer_mean scales monotonically with severity (6.9mm -> 15.9mm -> 30.7mm
mild/moderate/severe) - physically sensible. Pilot output lives at
`Datasets/THuman2.0/mos_dataset_geometry_pilot_0000_walk/`.

Found and fixed a real inefficiency during this run: `compute_geometry_metrics()` was building
each frame's two cKDTrees twice (once inside `chamfer_distance()`, once inside
`hausdorff_distance()`, both called separately) - fixed to build each tree once and reuse it for
both metrics. First full-combo run (pre-fix): **1h56m**. Timing re-run with the fix, for the
real Phase 3 Slurm `--time` budget: [update once the background re-run finishes].

`mos_app/` reworked for single-stimulus: new `pipeline/generate_trial_items.py` (no pairing
logic needed - one row per manifest instance), `mos_app/github_store.py`'s `RESPONSES_FIELDS`
changed to the single-stimulus schema, writing to a NEW file
(`mos_app/data/responses_geometry_v1.csv`) rather than the old pairwise `responses.csv`, which
already has real committed participant data under the old schema and is left untouched.
`MAIN_TRIALS_LIMIT = 20` (down from the full 37/combo) per the user's explicit session-length
call after seeing the bandwidth numbers. Verified end-to-end locally via Playwright: consent ->
demographics -> 3 practice trials (real live viewer embedded via
`st.components.v1.iframe`) -> a real main trial -> response recorded with the correct schema.

**Phase 3 (full VSC regen, IN PROGRESS)**: `vsc/geometry_export_array.slurm` (same 30-task
array/div-mod structure as `render_array.slurm`, calls `generate_geometry_dataset` instead of
`generate_dataset` - no Playwright/Chromium needed at all for this track, so expect it to run
faster per-task than the video job, `--time` placeholder carried over from that job until the
timing re-run confirms a real number), `vsc/merge_geometry_outputs.py` (same validate-then-merge
shape as `merge_outputs.py`, but validates each instance's directory contents - `meta.json`
parses, `base.drc`/`texture.jpg` exist, `frames/` has exactly `n_delta_frames` files - since a
truncated instance here is a bad file inside a directory, not a missing top-level file),
`vsc/fetch_geometry_outputs.sh` (pulls to `Datasets/THuman2.0/mos_dataset_geometry_v1/`).
`pipeline/requirements.txt` gained `DracoPy>=2.1.0` (confirmed building cleanly on macOS/arm64;
**VSC login-node build not yet verified** - builds from source, needs a C++ toolchain + cmake,
no prebuilt Linux/x86_64 wheel as of 2.1.0).

**Blocked on VSC connectivity as of 2026-09-19**: `ssh vsc` refused
(`login.hpc.kuleuven.be port 22`) from the current network - the IP-whitelist firewall
(`firewall.vscentrum.be`) needs re-whitelisting after a network change, same as documented
above. Can't verify the DracoPy build on the actual login node, or submit the array job, until
that's resolved.
