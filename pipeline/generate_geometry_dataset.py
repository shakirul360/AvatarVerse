"""Phase 2 (interactive-viewer track): generate the geometry-bundle MOS dataset - single
distortions and mixed (compound) ones, exported as Draco-compressed bundles for the live
free-orbit viewer instead of rendered mp4s. Mirrors pipeline/generate_dataset.py's structure and
reuses its pose/mesh/texture-distortion sequencing (STAGE_OF, _skin_variant) and
pipeline/distortions.py's distortion functions unchanged - only the final "turn positions into
an artifact" step differs (geometry_export.write_geometry_bundle() instead of renderer.render()),
plus a metrics.json (Chamfer/Hausdorff vs. the reference) written alongside every instance.

Per (subject, clip):
  1. build_wrap(subject)              - once per subject, reused across every clip
  2. load_pose_sequence(clip)         - clean AMASS pose
  3. export reference                 - clean pose -> clean skin -> geometry bundle
  4. precompute, once: 3 decimation topologies + 3 compressed textures (subject-level, reused
     by every clip/severity/mixed-pair that needs them - never recomputed per instance)
  5. 24 single-distortion instances (8 types x 3 severities) -> single/
  6. 12 mixed instances (6 curated pairs x 2 severity presets) -> mixed/, reusing whichever
     pose-distorted skin was already computed in step 5 for that (type, severity)
  = 37 instances/combo total.

Output layout:
  <out_dir>/single/<subject>_<clip>_reference/{base.drc, frames/*.drc, texture.jpg, meta.json, metrics.json}
  <out_dir>/single/<subject>_<clip>_<type>_<severity>/...
  <out_dir>/mixed/<subject>_<clip>_<type1>+<type2>_<sev1>+<sev2>/...
  <out_dir>/manifest_geometry.csv

CLI:  python -m pipeline.generate_geometry_dataset --subjects 0000 --clips walk --out <dir>
"""
import os, sys, csv, json, time, argparse, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_mos_pilot_dataset as gen
import skin_scan
import distortions as dist
import geometry_export
import metrics as geom_metrics

STAGE_OF = {t: 'pose' for t in dist.POSE_TYPES}
STAGE_OF.update(vertex_quantization='mesh', mesh_decimation='mesh', texture_compression='texture')


def _skin_variant(wrap, root_orient, pose_body, pose_specs):
    """Apply any pose-space specs, then pose the scan. pose_specs: list of (type, severity).
    Identical logic to generate_dataset.py's helper of the same name - inlined rather than
    imported so this module has no import-time dependency on the Playwright-based renderer
    generate_dataset.py pulls in, which this geometry-export track never needs at all."""
    go, bp = root_orient, pose_body
    for t, sev in pose_specs:
        go, bp = dist.apply_pose_distortion(t, go, bp, sev)
    return skin_scan.pose_scan(wrap, go, bp)


def _export_positions(positions, faces, uv, texture_path, out_dir, meta_extra, ref_positions):
    """positions/faces/uv/texture -> geometry bundle + metrics.json (Chamfer/Hausdorff vs.
    ref_positions - for the reference instance itself, ref_positions IS positions, giving the
    correct trivial 0.0 rather than needing a special case)."""
    meta = dict(subject=meta_extra['subject'], clip=meta_extra['clip'],
               playback_fps=meta_extra['playback_fps'])
    stats = geometry_export.write_geometry_bundle(out_dir, positions, faces, uv, texture_path, meta)
    m = geom_metrics.compute_geometry_metrics(ref_positions, positions)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    return stats, m


def export_instance(wrap, clean_root_orient, clean_pose_body, ref_positions, decimations,
                    compressed_textures, specs, out_dir, clip, playback_fps, pose_cache=None):
    """specs: list of (type, severity), len 1 (single) or 2 (mixed). Applies each spec at its
    own pipeline stage (pose -> mesh -> texture), reusing ref_positions / precomputed
    decimations / precomputed compressed textures wherever a stage isn't touched - identical
    sequencing to generate_dataset.py's render_instance(), minus the frame stage (motion_blur
    doesn't exist in this track - see distortions.py's FRAME_LEVELS docstring)."""
    pose_specs = [(t, s) for t, s in specs if STAGE_OF[t] == 'pose']
    mesh_specs = [(t, s) for t, s in specs if STAGE_OF[t] == 'mesh']
    tex_specs = [(t, s) for t, s in specs if STAGE_OF[t] == 'texture']

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

    meta_extra = dict(subject=wrap.subject, clip=clip, playback_fps=playback_fps)
    return _export_positions(positions, faces, uv, texture_path, out_dir, meta_extra, ref_positions)


def export_combo(subject, clip, out_dir, work_dir=None, wrap=None, speed=0.75,
                 playback_fps=30, n_frames=46, verbose=True):
    t0 = time.time()

    def log(m):
        if verbose:
            print(f'[{subject}/{clip} {time.time()-t0:5.0f}s] {m}', flush=True)

    single_dir = os.path.join(out_dir, 'single'); os.makedirs(single_dir, exist_ok=True)
    mixed_dir = os.path.join(out_dir, 'mixed'); os.makedirs(mixed_dir, exist_ok=True)
    work_dir = work_dir or tempfile.mkdtemp(prefix='avgeom_')
    os.makedirs(work_dir, exist_ok=True)

    if wrap is None:
        wrap = skin_scan.build_wrap(subject, verbose=verbose)
    root_orient, pose_body, meta = skin_scan.load_pose_sequence(clip, speed, playback_fps, n_frames)
    playback_fps = meta['playback_fps']
    log(f'pose sequence loaded, speed {meta["speed"]}x')

    ref_positions = skin_scan.pose_scan(wrap, root_orient, pose_body)
    log('reference skinned')

    rows = []

    ref_name = f'{subject}_{clip}_reference'
    stats, m = _export_positions(ref_positions, wrap.Fd, wrap.UVd, wrap.texture_path,
                                 os.path.join(single_dir, ref_name),
                                 dict(subject=subject, clip=clip, playback_fps=playback_fps),
                                 ref_positions)
    rows.append(dict(asset_id=ref_name, kind='single', subject=subject, clip=clip,
                     distortion_type='none', severity='none', reference='', params='',
                     bundle_bytes=stats['bundle_bytes_total'], chamfer_mean=m['chamfer_mean'],
                     chamfer_max=m['chamfer_max'], hausdorff_max=m['hausdorff_max']))
    log(f'reference exported -> {ref_name}  ({stats["bundle_bytes_total"]/1e6:.1f} MB)')

    # decimation topologies + compressed textures: identical on-disk caching pattern to
    # generate_dataset.py's generate_combo() (subject-level, reused across every clip/severity).
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

    def out(folder, name):
        return os.path.join(folder, name)

    for dist_type in dist.DISTORTION_TYPES:
        for sev in dist.SEVERITIES:
            asset_id = f'{subject}_{clip}_{dist_type}_{sev}'
            stats, m = export_instance(wrap, root_orient, pose_body, ref_positions, decimations,
                                       compressed_textures, [(dist_type, sev)],
                                       out(single_dir, asset_id), clip, playback_fps, pose_cache)
            params = dist.LEVELS[dist_type][sev]
            rows.append(dict(asset_id=asset_id, kind='single', subject=subject, clip=clip,
                             distortion_type=dist_type, severity=sev, reference='',
                             params=str(params), bundle_bytes=stats['bundle_bytes_total'],
                             chamfer_mean=m['chamfer_mean'], chamfer_max=m['chamfer_max'],
                             hausdorff_max=m['hausdorff_max']))
    log(f'{len(dist.DISTORTION_TYPES)*3} single-distortion instances exported -> {single_dir}')

    for t1, t2 in dist.MIXED_PAIRS:
        for sev in dist.MIXED_SEVERITY_PRESETS:
            asset_id = f'{subject}_{clip}_{t1}+{t2}_{sev}+{sev}'
            stats, m = export_instance(wrap, root_orient, pose_body, ref_positions, decimations,
                                       compressed_textures, [(t1, sev), (t2, sev)],
                                       out(mixed_dir, asset_id), clip, playback_fps, pose_cache)
            params = dict(t1=dist.LEVELS[t1][sev], t2=dist.LEVELS[t2][sev])
            rows.append(dict(asset_id=asset_id, kind='mixed', subject=subject, clip=clip,
                             distortion_type=f'{t1}+{t2}', severity=f'{sev}+{sev}',
                             reference=ref_name, params=str(params),
                             bundle_bytes=stats['bundle_bytes_total'], chamfer_mean=m['chamfer_mean'],
                             chamfer_max=m['chamfer_max'], hausdorff_max=m['hausdorff_max']))
    log(f'{len(dist.MIXED_PAIRS)*len(dist.MIXED_SEVERITY_PRESETS)} mixed instances exported -> {mixed_dir}')

    return rows


MANIFEST_FIELDS = ['asset_id', 'kind', 'subject', 'clip', 'distortion_type', 'severity',
                   'reference', 'params', 'bundle_bytes', 'chamfer_mean', 'chamfer_max', 'hausdorff_max']


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

    rows = []
    for subject in subjects:
        wrap = skin_scan.build_wrap(subject)
        for clip in clips:
            rows.extend(export_combo(subject, clip, a.out, work_dir=a.work_dir, wrap=wrap))

    with open(os.path.join(a.out, 'manifest_geometry.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} instances written to {a.out}')


if __name__ == '__main__':
    main()
