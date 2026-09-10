"""Skin a THuman textured scan to an AMASS motion clip via SMPL-X surface wrap.

Surface wrap (robust vs. propagating LBS weights): every scan vertex is stored as a barycentric
point on the subject's posed SMPL-X surface plus a signed offset along the interpolated surface
normal. Re-posing evaluates that same point + offset on SMPL-X posed to each AMASS frame, so the
model's own skinning + pose blendshapes carry the deformation - joints don't tear and loose
clothing rides the body.

Speed: AMASS clips are 120 fps mocap. stride = round(target_speed * src_fps / playback_fps),
so stride 3 at 30 fps playback == 0.75x real speed.

CLI:  python -m pipeline.skin_scan --subject 0000 --clip walk --speed 0.75 --out <dir>
writes  positions.f32 (n_frames, n_verts, 3) | faces.u32 | uv.f32 | texture.jpg | meta.json
"""
import os, sys, json, pickle, time, argparse
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


def skin_sequence(subject, clip_key, target_speed=0.75, playback_fps=30,
                  n_unique_frames=46, verbose=True):
    t0 = time.time()
    clip_rel = gen.CLIPS[clip_key] if clip_key in gen.CLIPS else clip_key

    def log(m):
        if verbose:
            print(f'[{time.time()-t0:5.0f}s] {m}', flush=True)

    # 1. full-res scan (decimating corrupts UV seams -> torn dark patches on the black atlas)
    scan = trimesh.load(os.path.join(gen.THUMAN_ROOT, subject, f'{subject}.obj'), process=False)
    Vd = np.asarray(scan.vertices, np.float64)
    Fd = np.asarray(scan.faces, np.int64)
    UVd = np.asarray(scan.visual.uv, np.float64)
    log(f'scan {len(Vd)} verts, {len(Fd)} faces')

    # 2. SMPL-X in the scan-fit pose (metric) + Procrustes to the provided registration
    with open(os.path.join(gen.THUMAN_ROOT, 'smplx', subject, 'smplx_param.pkl'), 'rb') as f:
        p = pickle.load(f, encoding='latin1')
    T = lambda k: torch.tensor(np.asarray(p[k], np.float32).reshape(1, -1))
    betas, th_go, th_bp, lh, rh = (T('betas')[:, :10], T('global_orient'), T('body_pose'),
                                   T('left_hand_pose'), T('right_hand_pose'))
    model = smplx.create(gen.SMPLX_MODEL_DIR, model_type='smplx', gender='neutral',
                         use_pca=False, num_betas=10, batch_size=1)
    faces_x = model.faces.astype(np.int64)
    Jreg = model.J_regressor.detach().numpy()

    def smplx_verts(go, bp):
        return model(betas=betas, global_orient=go, body_pose=bp,
                     left_hand_pose=lh, right_hand_pose=rh).vertices[0].detach().numpy()

    Vx_scan = smplx_verts(th_go, th_bp)
    Vx_reg = np.asarray(trimesh.load(os.path.join(gen.THUMAN_ROOT, 'smplx', subject,
                                                  'mesh_smplx.obj'), process=False).vertices, np.float64)
    s, R, t = _procrustes(Vx_scan, Vx_reg)
    Vd_metric = ((Vd - t) @ R) / s
    resid = np.linalg.norm((s * (Vx_scan @ R.T) + t) - Vx_reg, axis=1).mean()
    log(f'procrustes scale {s:.4f}  resid {resid*100:.3f} cm')

    # 3. wrap scan onto scan-pose SMPL-X surface
    mesh0 = trimesh.Trimesh(Vx_scan, faces_x, process=False)
    closest, _, tri_id = mesh0.nearest.on_surface(Vd_metric)
    bary = trimesh.triangles.points_to_barycentric(Vx_scan[faces_x[tri_id]], closest)
    vn0 = mesh0.vertex_normals
    n0 = np.einsum('nb,nbc->nc', bary, vn0[faces_x[tri_id]])
    n0 /= np.linalg.norm(n0, axis=1, keepdims=True) + 1e-9
    offset = np.einsum('nc,nc->n', Vd_metric - closest, n0)
    log(f'wrapped  (|offset| mean {np.abs(offset).mean()*100:.2f} cm, max {np.abs(offset).max()*100:.1f} cm)')

    # 4. AMASS motion at target speed
    d = np.load(os.path.join(gen.AMASS_ROOT, clip_rel), allow_pickle=True)
    src_fps = float(d['mocap_frame_rate'])
    stride = max(1, round(target_speed * src_fps / playback_fps))
    speed = round(playback_fps * stride / src_fps, 3)
    ro = torch.tensor(d['root_orient'][::stride][:n_unique_frames], dtype=torch.float32)
    pb = torch.tensor(d['pose_body'][::stride][:n_unique_frames], dtype=torch.float32)
    n = len(ro)
    log(f'{src_fps:.0f}fps src, stride {stride} -> {speed}x real speed, {n} unique frames')

    # 5. reconstruct scan per frame
    f0, f1, f2 = faces_x[tri_id].T
    b0, b1, b2 = bary.T
    positions = np.empty((n, len(Vd), 3), np.float32)
    for i in range(n):
        Vx = smplx_verts(ro[i:i + 1], pb[i:i + 1])
        vn = trimesh.Trimesh(Vx, faces_x, process=False).vertex_normals
        cp = b0[:, None] * Vx[f0] + b1[:, None] * Vx[f1] + b2[:, None] * Vx[f2]
        ni = b0[:, None] * vn[f0] + b1[:, None] * vn[f1] + b2[:, None] * vn[f2]
        ni /= np.linalg.norm(ni, axis=1, keepdims=True) + 1e-9
        v = cp + offset[:, None] * ni
        v -= (Jreg @ Vx)[0]                     # pelvis-center (matches the untextured pipeline)
        positions[i] = v
    log(f'posed {n} frames')

    meta = dict(subject=subject, clip=clip_key, clip_rel=clip_rel, n_frames=n, n_verts=len(Vd),
                n_faces=len(Fd), playback_fps=playback_fps, speed=speed, stride=stride)
    return positions, Fd, UVd, os.path.join(gen.THUMAN_ROOT, subject, 'material0.jpeg'), meta


def write_buffers(out_dir, positions, faces, uv, texture_src, meta):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    positions.astype(np.float32).tofile(os.path.join(out_dir, 'positions.f32'))
    faces.astype(np.uint32).tofile(os.path.join(out_dir, 'faces.u32'))
    uv.astype(np.float32).tofile(os.path.join(out_dir, 'uv.f32'))
    tex = Image.open(texture_src).convert('RGB')
    if max(tex.size) > TEX_MAX:
        tex = tex.resize((TEX_MAX, TEX_MAX), Image.LANCZOS)
    tex.save(os.path.join(out_dir, 'texture.jpg'), quality=92)
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
