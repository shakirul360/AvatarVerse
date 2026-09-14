"""Phase 2: generate the textured MOS dataset - single distortions and mixed (compound) ones.

Per (subject, clip):
  1. build_wrap(subject)              - once per subject, reused across every clip
  2. load_pose_sequence(clip)         - clean AMASS pose
  3. render reference                 - clean pose -> clean skin -> clean texture
  4. precompute, once: 3 decimation topologies + 3 compressed textures (subject-level, reused
     by every clip/severity/mixed-pair that needs them - never recomputed per instance)
  5. 27 single-distortion instances (9 types x 3 severities) -> single/
  6. 12 mixed instances (6 curated pairs x 2 severity presets) -> mixed/, reusing whichever
     pose-distorted skin was already computed in step 5 for that (type, severity)

Output layout:
  <out_dir>/single/<subject>_<clip>_reference.mp4
  <out_dir>/single/<subject>_<clip>_<type>_<severity>.mp4
  <out_dir>/mixed/<subject>_<clip>_<type1>+<type2>_<sev1>+<sev2>.mp4
  <out_dir>/manifest_single.csv, manifest_mixed.csv

CLI:  python -m pipeline.generate_dataset --subjects 0000,0100,0500 --clips walk,front_kick,bmlmovi_walk --out <dir>
"""
import os, sys, csv, json, time, argparse, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_mos_pilot_dataset as gen
import skin_scan
import distortions as dist
import render as renderer

STAGE_OF = {t: 'pose' for t in dist.POSE_TYPES}
STAGE_OF.update(vertex_quantization='mesh', mesh_decimation='mesh',
                texture_compression='texture', motion_blur='frame')

PANEL = 1280
LOOPS = 3


def _skin_variant(wrap, root_orient, pose_body, pose_specs):
    """Apply any pose-space specs, then pose the scan. pose_specs: list of (type, severity)."""
    go, bp = root_orient, pose_body
    for t, sev in pose_specs:
        go, bp = dist.apply_pose_distortion(t, go, bp, sev)
    return skin_scan.pose_scan(wrap, go, bp)


def _render_positions(positions, faces, uv, texture_path, out_path, meta_extra,
                      frame_spec=None, work_dir=None):
    """positions/faces/uv/texture -> buffers -> render (with optional post-render motion_blur)."""
    buf_dir = tempfile.mkdtemp(dir=work_dir, prefix='buf_')
    try:
        meta = dict(n_verts=positions.shape[1], n_faces=len(faces), n_frames=len(positions),
                   playback_fps=meta_extra['playback_fps'])
        skin_scan.write_buffers(buf_dir, positions, faces, uv, texture_path, meta)
        if frame_spec is None:
            n = renderer.render(buf_dir, out_path, loops=LOOPS, panel=PANEL)
        else:
            ftype, fsev = frame_spec
            frames, meta = renderer.capture_frames(buf_dir, panel=PANEL)
            params = dist.FRAME_LEVELS[ftype][fsev]
            frames = list(dist.apply_motion_blur(frames, **params))
            n = renderer.encode_video(frames, out_path, meta['playback_fps'], loops=LOOPS)
        return n
    finally:
        shutil.rmtree(buf_dir, ignore_errors=True)


def render_instance(wrap, clean_root_orient, clean_pose_body, ref_positions, decimations,
                    compressed_textures, specs, out_path, work_dir, playback_fps, pose_cache=None):
    """specs: list of (type, severity), len 1 (single) or 2 (mixed). Applies each spec at its
    own pipeline stage (pose -> mesh -> texture -> frame), reusing ref_positions / precomputed
    decimations / precomputed compressed textures wherever a stage isn't touched."""
    pose_specs = [(t, s) for t, s in specs if STAGE_OF[t] == 'pose']
    mesh_specs = [(t, s) for t, s in specs if STAGE_OF[t] == 'mesh']
    tex_specs = [(t, s) for t, s in specs if STAGE_OF[t] == 'texture']
    frame_specs = [(t, s) for t, s in specs if STAGE_OF[t] == 'frame']

    cache_key = tuple(pose_specs)
    if not pose_specs:
        positions = ref_positions
    elif pose_cache is not None and cache_key in pose_cache:
        positions = pose_cache[cache_key]
    else:
        positions = _skin_variant(wrap, clean_root_orient, clean_pose_body, pose_specs)
        if pose_cache is not None:
            pose_cache[cache_key] = positions

    faces, uv = wrap.Fd, wrap.UVd
    for t, sev in mesh_specs:
        if t == 'vertex_quantization':
            positions = dist.apply_vertex_quantization(positions, **gen.LEVELS[t][sev])
        elif t == 'mesh_decimation':
            positions, faces, uv = dist.apply_mesh_decimation(positions, decimations[sev])

    texture_path = wrap.texture_path
    for t, sev in tex_specs:
        texture_path = compressed_textures[sev]

    frame_spec = frame_specs[0] if frame_specs else None
    return _render_positions(positions, faces, uv, texture_path, out_path,
                             dict(playback_fps=playback_fps), frame_spec, work_dir)


def generate_combo(subject, clip, out_dir, work_dir=None, wrap=None, speed=0.75,
                   playback_fps=30, n_frames=46, verbose=True):
    t0 = time.time()

    def log(m):
        if verbose:
            print(f'[{subject}/{clip} {time.time()-t0:5.0f}s] {m}', flush=True)

    single_dir = os.path.join(out_dir, 'single'); os.makedirs(single_dir, exist_ok=True)
    mixed_dir = os.path.join(out_dir, 'mixed'); os.makedirs(mixed_dir, exist_ok=True)
    work_dir = work_dir or tempfile.mkdtemp(prefix='avgen_')
    os.makedirs(work_dir, exist_ok=True)

    if wrap is None:
        wrap = skin_scan.build_wrap(subject, verbose=verbose)
    root_orient, pose_body, meta = skin_scan.load_pose_sequence(clip, speed, playback_fps, n_frames)
    playback_fps = meta['playback_fps']
    log(f'pose sequence loaded, speed {meta["speed"]}x')

    ref_positions = skin_scan.pose_scan(wrap, root_orient, pose_body)
    log('reference skinned')

    rows = []

    def out(folder, fname):
        return os.path.join(folder, fname)

    ref_name = f'{subject}_{clip}_reference.mp4'
    n = _render_positions(ref_positions, wrap.Fd, wrap.UVd, wrap.texture_path,
                          out(single_dir, ref_name), dict(playback_fps=playback_fps), None, work_dir)
    rows.append(dict(file=ref_name, subject=subject, clip=clip, distortion_type='none',
                     severity='none', params='', n_frames=n))
    log(f'reference rendered -> {ref_name}')

    decim_dir = os.path.join(work_dir, 'decim', subject)
    decimations = {}
    for sev in dist.SEVERITIES:
        tf = dist.MESH_LEVELS['mesh_decimation'][sev]['target_faces']
        if not os.path.exists(os.path.join(decim_dir, sev, 'faces.npy')):
            d = dist.build_decimation(wrap.Fd, wrap.UVd, wrap.texture_path,
                                      ref_positions[0].astype(np.float64), tf,
                                      os.path.join(decim_dir, sev, '_pymeshlab_tmp'))
            os.makedirs(os.path.join(decim_dir, sev), exist_ok=True)
            np.save(os.path.join(decim_dir, sev, 'kept_indices.npy'), d['kept_indices'])
            np.save(os.path.join(decim_dir, sev, 'faces.npy'), d['faces'])
            np.save(os.path.join(decim_dir, sev, 'uv.npy'), d['uv'])
        decimations[sev] = dict(
            kept_indices=np.load(os.path.join(decim_dir, sev, 'kept_indices.npy')),
            faces=np.load(os.path.join(decim_dir, sev, 'faces.npy')),
            uv=np.load(os.path.join(decim_dir, sev, 'uv.npy')))
    log('mesh_decimation topologies ready (3 severities)')

    tex_dir = os.path.join(work_dir, 'tex', subject)
    os.makedirs(tex_dir, exist_ok=True)
    compressed_textures = {}
    for sev in dist.SEVERITIES:
        p = os.path.join(tex_dir, f'{sev}.jpg')
        if not os.path.exists(p):
            q = dist.TEXTURE_LEVELS['texture_compression'][sev]['quality']
            dist.build_compressed_texture(wrap.texture_path, q, p)
        compressed_textures[sev] = p
    log('compressed textures ready (3 severities)')

    pose_cache = {}   # (type,severity) tuple-of-one -> posed positions, for mixed-pair reuse

    for dist_type in dist.DISTORTION_TYPES:
        for sev in dist.SEVERITIES:
            fname = f'{subject}_{clip}_{dist_type}_{sev}.mp4'
            n = render_instance(wrap, root_orient, pose_body, ref_positions, decimations,
                                compressed_textures, [(dist_type, sev)],
                                out(single_dir, fname), work_dir, playback_fps, pose_cache)
            params = dist.LEVELS[dist_type][sev]
            rows.append(dict(file=fname, subject=subject, clip=clip, distortion_type=dist_type,
                             severity=sev, params=str(params), n_frames=n))
    log(f'{len(dist.DISTORTION_TYPES)*3} single-distortion instances rendered -> {single_dir}')

    mixed_rows = []
    for t1, t2 in dist.MIXED_PAIRS:
        for sev in dist.MIXED_SEVERITY_PRESETS:
            fname = f'{subject}_{clip}_{t1}+{t2}_{sev}+{sev}.mp4'
            n = render_instance(wrap, root_orient, pose_body, ref_positions, decimations,
                                compressed_textures, [(t1, sev), (t2, sev)],
                                out(mixed_dir, fname), work_dir, playback_fps, pose_cache)
            params = dict(t1=dist.LEVELS[t1][sev], t2=dist.LEVELS[t2][sev])
            mixed_rows.append(dict(file=fname, subject=subject, clip=clip,
                                   distortion_type=f'{t1}+{t2}', severity=f'{sev}+{sev}',
                                   reference=ref_name, params=str(params), n_frames=n))
    log(f'{len(dist.MIXED_PAIRS)*len(dist.MIXED_SEVERITY_PRESETS)} mixed instances rendered -> {mixed_dir}')

    return rows, mixed_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', default='0000')
    ap.add_argument('--clips', default='walk')
    ap.add_argument('--out', required=True)
    ap.add_argument('--work-dir', default=None)
    a = ap.parse_args()
    subjects = a.subjects.split(',')
    clips = a.clips.split(',')
    os.makedirs(a.out, exist_ok=True)

    single_rows, mixed_rows = [], []
    for subject in subjects:
        wrap = skin_scan.build_wrap(subject)
        for clip in clips:
            r, m = generate_combo(subject, clip, a.out, work_dir=a.work_dir, wrap=wrap)
            single_rows.extend(r); mixed_rows.extend(m)

    with open(os.path.join(a.out, 'manifest_single.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'subject', 'clip', 'distortion_type', 'severity', 'params', 'n_frames'])
        w.writeheader(); w.writerows(single_rows)
    with open(os.path.join(a.out, 'manifest_mixed.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'subject', 'clip', 'distortion_type', 'severity', 'reference', 'params', 'n_frames'])
        w.writeheader(); w.writerows(mixed_rows)
    print(f'{len(single_rows)} single + {len(mixed_rows)} mixed = {len(single_rows)+len(mixed_rows)} files written to {a.out}')


if __name__ == '__main__':
    main()
