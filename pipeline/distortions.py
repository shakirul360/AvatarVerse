"""Distortion functions for the textured pipeline, organized by the stage each applies at (see
pipeline/generate_dataset.py for how they're sequenced). Pose-space distortions are the exact,
already-validated functions from generate_mos_pilot_dataset.py - this module just adapts them
to skin_scan's (root_orient, pose_body) tensor-pair representation instead of the old
list-of-dict one. The four mesh/texture/frame-space types are new - the untextured pipeline
never needed them (no UVs, no real texture map, no frame buffer worth blurring).
"""
import os, sys, tempfile, shutil
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_mos_pilot_dataset as gen

# ---------------------------------------------------------------- pose-space (before skinning)
POSE_TYPES = ['jitter', 'joint_noise', 'frame_drop', 'smoothing', 'param_quantization']
_POSE_FNS = {'jitter': gen.apply_jitter, 'joint_noise': gen.apply_joint_noise,
            'frame_drop': gen.apply_frame_drop, 'smoothing': gen.apply_smoothing,
            'param_quantization': gen.apply_param_quantization}


def apply_pose_distortion(dist_type, root_orient, pose_body, severity):
    """(T,3)/(T,63) clean -> (T,3)/(T,63) distorted. Thin adapter onto the validated
    list-of-dict functions in generate_mos_pilot_dataset.py - unchanged logic, just repacked."""
    params = gen.LEVELS[dist_type][severity]
    results = [dict(frame_idx=i, body_pose=pose_body[i:i + 1], global_orient=root_orient[i:i + 1])
               for i in range(len(root_orient))]
    out = _POSE_FNS[dist_type](results, **params)
    new_bp = torch.cat([r['body_pose'] for r in out], dim=0)
    new_go = torch.cat([r['global_orient'] for r in out], dim=0)
    return new_go, new_bp


# ---------------------------------------------------------------- mesh-space (posed positions)
MESH_LEVELS = {
    'vertex_quantization': gen.LEVELS['vertex_quantization'],   # step_m: .003 / .01 / .03
    'mesh_decimation': {'mild': dict(target_faces=100_000),
                       'moderate': dict(target_faces=30_000),
                       'severe': dict(target_faces=8_000)},
}


def apply_vertex_quantization(positions, step_m):
    """positions: (T, N, 3). Independent per frame, no topology change - unchanged from the
    untextured pipeline's version, just operating on a plain array instead of a trimesh object."""
    return np.round(positions / step_m) * step_m


def build_decimation(faces, uv, texture_path, ref_positions, target_faces, tmp_dir):
    """ONE-TIME per (subject, target_faces): decimate the topology from a single reference
    frame's geometry via pymeshlab's texture-aware quadric collapse, then snap every surviving
    vertex to its nearest ORIGINAL vertex index. All 46 frames share identical face connectivity
    (only positions move), so this index set + face list is reused for every frame of every
    clip/distortion of this subject - never re-run pymeshlab per frame.

    Calibration notes (found empirically): must load through a temp .obj file, not construct a
    pymeshlab.Mesh from arrays directly - the array constructor doesn't wire up per-face texture
    indices correctly and every decimation call fails with 'inconsistent tex coordinates'.
    preserveboundary=True hard-locks every UV-seam edge (this scan's atlas has many), which
    floors decimation at ~95k faces no matter the target; preserveboundary=False + the default
    boundaryweight=1.0 (a soft penalty, not a lock) actually reaches low targets."""
    import pymeshlab
    os.makedirs(tmp_dir, exist_ok=True)
    mtl_path = os.path.join(tmp_dir, 'ref.mtl')
    with open(mtl_path, 'w') as f:
        f.write('newmtl m0\nKa 0.2 0.2 0.2\nKd 1 1 1\nKs 1 1 1\nNs 0\nmap_Kd texture.jpg\n')
    shutil.copy(texture_path, os.path.join(tmp_dir, 'texture.jpg'))
    obj_path = os.path.join(tmp_dir, 'ref.obj')
    with open(obj_path, 'w') as f:
        f.write('mtllib ref.mtl\nusemtl m0\n')
        np.savetxt(f, ref_positions, fmt='v %.6f %.6f %.6f')
        np.savetxt(f, uv, fmt='vt %.6f %.6f')
        fv = (faces + 1).astype(str)   # obj is 1-indexed; v index == vt index (per-vertex UV)
        lines = 'f ' + fv[:, 0] + '/' + fv[:, 0] + ' ' + fv[:, 1] + '/' + fv[:, 1] + ' ' + fv[:, 2] + '/' + fv[:, 2] + '\n'
        f.writelines(lines.tolist())

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=target_faces, preserveboundary=False, boundaryweight=1.0,
        preservenormal=True, optimalplacement=True, qualitythr=0.3)
    dm = ms.current_mesh()
    v_decimated, f_decimated = dm.vertex_matrix(), dm.face_matrix().astype(np.int64)

    # optimalplacement=True (needed - optimalplacement=False segfaults this pymeshlab build)
    # moves collapsed vertices to an interpolated point, not always one of the two original
    # endpoints, so snapping to the nearest original vertex can send two different decimated
    # vertices to the SAME original index - degenerate (zero-area) triangles wherever that
    # happens, which upload NaN normals to the GPU and render solid white under SwiftShader.
    # Which triangles collide is fixed by the index snap alone (identical every frame, since
    # positions[:, kept_indices] always gives those slots the same shared position) - filter
    # once here, never per frame.
    kept_indices = cKDTree(ref_positions).query(v_decimated, k=1)[1]
    v0f, v1f, v2f = (ref_positions[kept_indices[f_decimated[:, c]]] for c in range(3))
    area2 = np.linalg.norm(np.cross(v1f - v0f, v2f - v0f), axis=1)
    degenerate = area2 < 1e-10
    if degenerate.any():
        f_decimated = f_decimated[~degenerate]
    return dict(kept_indices=kept_indices, faces=f_decimated, uv=uv[kept_indices],
               n_degenerate_dropped=int(degenerate.sum()))


def apply_mesh_decimation(positions, decimation):
    """positions: (T, N, 3) -> (T, N_kept, 3), plus the decimated faces/uv (constant across T)."""
    return positions[:, decimation['kept_indices'], :], decimation['faces'], decimation['uv']


# ---------------------------------------------------------------- texture-space
TEXTURE_LEVELS = {'texture_compression': {'mild': dict(quality=50), 'moderate': dict(quality=25),
                                          'severe': dict(quality=10)}}


def build_compressed_texture(texture_path, quality, out_path):
    """JPEG re-encode at a lower quality - precomputed once per (subject, severity), independent
    of pose/frame, then just swapped in at render time in place of the clean texture."""
    Image.open(texture_path).convert('RGB').save(out_path, quality=quality)
    return out_path


# ---------------------------------------------------------------- frame-space (after render)
# motion_blur (post-render RGB pixel averaging - see generate_mos_pilot_dataset.apply_motion_blur)
# has no 3D-geometry equivalent and can't exist in the free-orbit interactive-viewer track, so
# it's deliberately absent from FRAME_LEVELS/LEVELS/MIXED_PAIRS below. The legacy video track's
# already-shipped mp4s (Datasets/THuman2.0/mos_dataset_textured_v1/) don't depend on this
# catalog at runtime - this module is edited in place rather than forked. apply_motion_blur
# itself (gen.apply_motion_blur) is untouched and still importable if ever needed again.
FRAME_LEVELS = {}


# ---------------------------------------------------------------- catalog
LEVELS = {**{k: gen.LEVELS[k] for k in POSE_TYPES}, **MESH_LEVELS, **TEXTURE_LEVELS, **FRAME_LEVELS}
DISTORTION_TYPES = list(LEVELS.keys())   # 8: 5 pose + vertex_quantization + mesh_decimation + texture_compression
SEVERITIES = ['mild', 'moderate', 'severe']

# 6 curated cross-category pairs (one pose-space + one mesh/texture-space - same-category pairs
# like jitter+joint_noise are largely redundant), each motivated by a real co-occurring
# degradation. 2 matched-severity presets per pair (not all 3) to keep session length bounded -
# see the 2026-09 design discussion. motion_blur+vertex_quantization (the video track's 6th
# pair) has no equivalent here since motion_blur is dropped for this track - replaced with a
# second vertex_quantization-involving pairing so mesh-space distortions stay represented twice,
# same as the other mesh-space type (mesh_decimation).
MIXED_PAIRS = [
    ('jitter', 'texture_compression'),      # noisy tracking + bandwidth-limited texture (streaming)
    ('frame_drop', 'mesh_decimation'),      # low frame rate + reduced geometry LOD
    ('joint_noise', 'vertex_quantization'), # noisy pose estimation + coarse geometry precision
    ('smoothing', 'texture_compression'),   # over-filtered motion + compressed texture
    ('param_quantization', 'mesh_decimation'),  # quantized pose + reduced mesh (compression codec)
    ('jitter', 'vertex_quantization'),      # noisy tracking + coarse geometry precision
]
MIXED_SEVERITY_PRESETS = ['mild', 'severe']   # matched: (mild,mild) and (severe,severe)
