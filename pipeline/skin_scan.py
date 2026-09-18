"""Skin a THuman textured scan to AMASS motion via SMPL-X surface wrap.

Surface wrap (robust vs. propagating LBS weights): every scan vertex is stored as a barycentric
point on the subject's posed SMPL-X surface plus a signed offset along the interpolated surface
normal. Re-posing evaluates that same point + offset on SMPL-X posed to each AMASS frame, so the
model's own skinning + pose blendshapes carry the deformation - joints don't tear and loose
clothing rides the body.

Split in two so distortions can be applied and the expensive part reused:
  build_wrap(subject)                     - Procrustes + surface wrap, ONCE per subject. Reused
                                             across every clip and every distortion variant of
                                             that subject.
  load_pose_sequence(clip_key, ...)       - AMASS loading, ONCE per clip. Pose-space distortions
                                             (jitter etc.) are applied to this before posing.
  pose_scan(wrap, root_orient, pose_body) - the actual per-frame reconstruction. Called once for
                                             the clean sequence and again for each distorted one.

Speed: AMASS clips are 120 fps mocap. stride = round(target_speed * src_fps / playback_fps),
so stride 3 at 30 fps playback == 0.75x real speed.

CLI (unchanged, for the Phase-1 smoke test):
  python -m pipeline.skin_scan --subject 0000 --clip walk --speed 0.75 --out <dir>
  writes  positions.f32 (n_frames, n_verts, 3) | faces.u32 | uv.f32 | texture.jpg | meta.json
"""
import os, sys, json, pickle, time, argparse
from dataclasses import dataclass
import numpy as np
import torch
import trimesh
import smplx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_mos_pilot_dataset as gen   # THUMAN_ROOT / AMASS_ROOT / SMPLX_MODEL_DIR / CLIPS

TEX_MAX = 4096


def _procrustes(src, dst):
    """similarity transform s,R,t with  s*(src@R.T)+t ~= dst   (src,dst in 1:1 correspondence)"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(Xs.T @ Xd)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Vt[-1] *= -1
    R = Vt.T @ U.T
    s = S.sum() / (Xs ** 2).sum()
    t = mu_d - s * (R @ mu_s)
    return s, R, t


@dataclass
class Wrap:
    """Everything needed to pose one subject's scan, independent of which motion clip."""
    subject: str
    model: object                 # the shared SMPL-X model instance
    betas: torch.Tensor
    hands: dict                   # left_hand_pose / right_hand_pose
    Jreg: np.ndarray
    faces_x: np.ndarray           # SMPL-X faces
    Vd: np.ndarray                # scan verts, metric space (len = n_scan_verts)
    Fd: np.ndarray                # scan faces (constant across every frame/distortion)
    UVd: np.ndarray                # scan per-vertex UV (constant)
    texture_path: str
    f0: np.ndarray; f1: np.ndarray; f2: np.ndarray   # per-scan-vertex enclosing SMPL-X triangle
    b0: np.ndarray; b1: np.ndarray; b2: np.ndarray   # barycentric coords within that triangle
    offset: np.ndarray            # signed normal offset (the wrap's "how far off the body surface")


def build_wrap(subject, verbose=True):
    t0 = time.time()

    def log(m):
        if verbose:
            print(f'[wrap {time.time()-t0:5.1f}s] {m}', flush=True)

    scan = trimesh.load(os.path.join(gen.THUMAN_ROOT, subject, f'{subject}.obj'), process=False)
    Vd = np.asarray(scan.vertices, np.float64)
    Fd = np.asarray(scan.faces, np.int64)
    UVd = np.asarray(scan.visual.uv, np.float64)
    log(f'scan {len(Vd)} verts, {len(Fd)} faces')

    with open(os.path.join(gen.THUMAN_ROOT, 'smplx', subject, 'smplx_param.pkl'), 'rb') as f:
        p = pickle.load(f, encoding='latin1')
    T = lambda k: torch.tensor(np.asarray(p[k], np.float32).reshape(1, -1))
    betas, th_go, th_bp = T('betas')[:, :10], T('global_orient'), T('body_pose')
    hands = dict(left_hand_pose=T('left_hand_pose'), right_hand_pose=T('right_hand_pose'))
    model = smplx.create(gen.SMPLX_MODEL_DIR, model_type='smplx', gender='neutral',
                         use_pca=False, num_betas=10, batch_size=1)
    faces_x = model.faces.astype(np.int64)
    Jreg = model.J_regressor.detach().numpy()

    def smplx_verts(go, bp):
        return model(betas=betas, global_orient=go, body_pose=bp, **hands).vertices[0].detach().numpy()

    Vx_scan = smplx_verts(th_go, th_bp)
    Vx_reg = np.asarray(trimesh.load(os.path.join(gen.THUMAN_ROOT, 'smplx', subject,
                                                  'mesh_smplx.obj'), process=False).vertices, np.float64)
    s, R, t = _procrustes(Vx_scan, Vx_reg)
    Vd_metric = ((Vd - t) @ R) / s
    resid = np.linalg.norm((s * (Vx_scan @ R.T) + t) - Vx_reg, axis=1).mean()
    log(f'procrustes scale {s:.4f}  resid {resid*100:.3f} cm')

    mesh0 = trimesh.Trimesh(Vx_scan, faces_x, process=False)
    closest, _, tri_id = mesh0.nearest.on_surface(Vd_metric)
    bary = trimesh.triangles.points_to_barycentric(Vx_scan[faces_x[tri_id]], closest)
    vn0 = mesh0.vertex_normals
    n0 = np.einsum('nb,nbc->nc', bary, vn0[faces_x[tri_id]])
    n0 /= np.linalg.norm(n0, axis=1, keepdims=True) + 1e-9
    offset = np.einsum('nc,nc->n', Vd_metric - closest, n0)
    log(f'wrapped  (|offset| mean {np.abs(offset).mean()*100:.2f} cm, max {np.abs(offset).max()*100:.1f} cm)')

    f0, f1, f2 = faces_x[tri_id].T
    b0, b1, b2 = bary.T
    return Wrap(subject=subject, model=model, betas=betas, hands=hands, Jreg=Jreg, faces_x=faces_x,
               Vd=Vd, Fd=Fd, UVd=UVd, texture_path=os.path.join(gen.THUMAN_ROOT, subject, 'material0.jpeg'),
               f0=f0, f1=f1, f2=f2, b0=b0, b1=b1, b2=b2, offset=offset)


def load_pose_sequence(clip_key, target_speed=0.75, playback_fps=30, n_unique_frames=46):
    """AMASS root_orient/pose_body at the target playback speed - the sequence pose-space
    distortions (jitter etc.) are applied to, before pose_scan()."""
    clip_rel = gen.CLIPS[clip_key] if clip_key in gen.CLIPS else clip_key
    d = np.load(os.path.join(gen.AMASS_ROOT, clip_rel), allow_pickle=True)
    src_fps = float(d['mocap_frame_rate'])
    stride = max(1, round(target_speed * src_fps / playback_fps))
    speed = round(playback_fps * stride / src_fps, 3)
    root_orient = torch.tensor(d['root_orient'][::stride][:n_unique_frames], dtype=torch.float32)
    pose_body = torch.tensor(d['pose_body'][::stride][:n_unique_frames], dtype=torch.float32)
    meta = dict(clip=clip_key, clip_rel=clip_rel, n_frames=len(root_orient),
               playback_fps=playback_fps, speed=speed, stride=stride, src_fps=src_fps)
    return root_orient, pose_body, meta


def pose_scan(wrap: Wrap, root_orient, pose_body, verbose=False):
    """Reconstruct the scan for every frame of a (possibly distorted) pose sequence."""
    n = len(root_orient)
    positions = np.empty((n, len(wrap.Vd), 3), np.float32)
    for i in range(n):
        out = wrap.model(betas=wrap.betas, global_orient=root_orient[i:i + 1],
                         body_pose=pose_body[i:i + 1], **wrap.hands)
        Vx = out.vertices[0].detach().numpy()
        vn = trimesh.Trimesh(Vx, wrap.faces_x, process=False).vertex_normals
        cp = (wrap.b0[:, None] * Vx[wrap.f0] + wrap.b1[:, None] * Vx[wrap.f1]
              + wrap.b2[:, None] * Vx[wrap.f2])
        ni = (wrap.b0[:, None] * vn[wrap.f0] + wrap.b1[:, None] * vn[wrap.f1]
              + wrap.b2[:, None] * vn[wrap.f2])
        ni /= np.linalg.norm(ni, axis=1, keepdims=True) + 1e-9
        v = cp + wrap.offset[:, None] * ni
        v -= (wrap.Jreg @ Vx)[0]     # pelvis-center (matches the untextured pipeline)
        positions[i] = v
        if verbose and (i + 1) % 10 == 0:
            print(f'  posed {i+1}/{n}', flush=True)
    return positions


def skin_sequence(subject, clip_key, target_speed=0.75, playback_fps=30,
                  n_unique_frames=46, verbose=True):
    """Convenience one-shot wrapper (clean sequence only) - kept for the Phase-1 smoke test."""
    wrap = build_wrap(subject, verbose=verbose)
    root_orient, pose_body, meta = load_pose_sequence(clip_key, target_speed, playback_fps, n_unique_frames)
    positions = pose_scan(wrap, root_orient, pose_body, verbose=verbose)
    meta = dict(subject=subject, n_verts=len(wrap.Vd), n_faces=len(wrap.Fd), **meta)
    return positions, wrap.Fd, wrap.UVd, wrap.texture_path, meta


def write_texture(texture_src, out_path, quality=92):
    """Resize-if-needed + JPEG-encode a texture. Shared by write_buffers() (legacy video track)
    and geometry_export.write_geometry_bundle() (interactive-viewer track) - one place to change
    TEX_MAX/quality instead of two copies drifting apart."""
    from PIL import Image
    tex = Image.open(texture_src).convert('RGB')
    if max(tex.size) > TEX_MAX:
        tex = tex.resize((TEX_MAX, TEX_MAX), Image.LANCZOS)
    tex.save(out_path, quality=quality)


def write_buffers(out_dir, positions, faces, uv, texture_src, meta):
    os.makedirs(out_dir, exist_ok=True)
    positions.astype(np.float32).tofile(os.path.join(out_dir, 'positions.f32'))
    faces.astype(np.uint32).tofile(os.path.join(out_dir, 'faces.u32'))
    uv.astype(np.float32).tofile(os.path.join(out_dir, 'uv.f32'))
    write_texture(texture_src, os.path.join(out_dir, 'texture.jpg'))
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', default='0000')
    ap.add_argument('--clip', default='walk')
    ap.add_argument('--speed', type=float, default=0.75)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--frames', type=int, default=46)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = skin_sequence(a.subject, a.clip, a.speed, a.fps, a.frames)
    write_buffers(a.out, *res)
    print('wrote', a.out)
    print(json.dumps(res[-1], indent=2))
