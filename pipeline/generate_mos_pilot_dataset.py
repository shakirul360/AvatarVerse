"""Milestone 5 - batch-generate a small pilot dataset for the MOS study.

Loops SUBJECTS x CLIPS x DISTORTION_TYPES x SEVERITIES, rendering one animation per
combination plus one clean reference per (subject, clip). Reuses the exact retargeting
(04_AMASS_Pose_Retarget.ipynb) and distortion (05_Distortion_Compression.ipynb) mechanisms -
nothing new is invented here, this just scales them up and writes a manifest so every
generated file can be traced back to its (subject, clip, distortion_type, severity) later.

Run: python3 generate_mos_pilot_dataset.py
"""
import os, csv, pickle, io, time, sys
import numpy as np
import torch
import smplx
import trimesh
import plotly.graph_objects as go
import plotly.io as pio
import imageio.v2 as imageio
from PIL import Image

pio.renderers.default = None

# Paths: env-overridable so the same code runs on the laptop and on VSC. Defaults are the
# laptop layout. On VSC set AVATARVERSE_MODELS / AVATARVERSE_THUMAN / AVATARVERSE_AMASS /
# AVATARVERSE_OUT (see vsc/env.sh).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMPLX_MODEL_DIR = os.environ.get('AVATARVERSE_MODELS', os.path.join(_REPO, 'models', 'models'))
THUMAN_ROOT = os.environ.get('AVATARVERSE_THUMAN', os.path.join(_REPO, 'Datasets', 'THuman2.0'))
AMASS_ROOT = os.environ.get('AVATARVERSE_AMASS', os.path.join(_REPO, 'Datasets', 'AMASS'))
# v2: video (.mp4) instead of GIF - see save_video()'s docstring for why. ("v3" loop/duration
# change was made in-place in the v2 dir, never got its own folder. v4 = aspectmode dolly-zoom
# fix + 1280px - but its full run never finished and it has a since-found anisotropic-squash bug
# from pairing per-frame smoothed ranges with a fixed aspectratio; delete THuman2.0/
# mos_pilot_dataset_v4/. v5 = UNIVERSAL_RANGE on every frame (no squash) + a much bigger body
# in frame, CAMERA_EYE pulled in to ~50-55% fill. Each version gets its own directory so older,
# now-known-wrong renders stay around for before/after comparison.)
OUT_DIR = os.environ.get('AVATARVERSE_OUT', os.path.join(THUMAN_ROOT, 'mos_pilot_dataset_v5'))
os.makedirs(OUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = smplx.create(SMPLX_MODEL_DIR, model_type='smplx', gender='neutral', use_pca=False, batch_size=1).to(device)

# Pilot scope, as agreed: 3 subjects x 3 clips x 8 distortion types x 3 severities
SUBJECTS = ['0000', '0100', '0500']
CLIPS = {
    # QkWalk1 (originally here) turned out to be a poor fit for a fixed-camera stimulus: its
    # global_orient cycles by up to 0.4 rad every ~50 frames (a genuine repeated turn in the
    # capture) and its stance width grows from 0.41 to 0.87 in the first 28 frames alone -
    # real motion, confirmed present even in the undistorted reference, not a rendering bug,
    # but exactly what produced the reported "thin then thick, repeating" look. B3_-_walk1 is
    # a plain, consistent walk with none of that (global_orient std 0.03 vs. QkWalk1's ~0.15).
    'walk': 'ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz',
    'front_kick': 'ACCAD/Male2MartialArtsKicks_c3d/G3_-_front_kick_stageii.npz',
    'bmlmovi_walk': 'BMLmovi/Subject_1_F_MoSh/Subject_1_F_19_stageii.npz',
}
# v3: short unique content, looped, instead of v2's long single pass. v2 sampled ~2.5s of real
# motion (150 frames) - multiple full gait cycles - which is exactly what made natural width
# variation (arm swing repeating several times over 30s) so visible. v3 samples roughly ONE
# gait cycle's worth of unique motion and repeats it (see LOOP_REPEATS/loop_sequence below)
# rather than showing a long stretch of unique-but-repetitive-anyway motion. stride=2 keeps
# consecutive sampled frames close together (smooth, not choppy).
STRIDE = 2
MAX_FRAMES = 40

LEVELS = {
    'jitter':              {'mild': dict(sigma_body=0.02, sigma_root=0.01),
                             'moderate': dict(sigma_body=0.05, sigma_root=0.025),
                             'severe': dict(sigma_body=0.10, sigma_root=0.05)},
    'joint_noise':         {'mild': dict(sigma=0.15, window=5),
                             'moderate': dict(sigma=0.30, window=5),
                             'severe': dict(sigma=0.55, window=5)},
    'frame_drop':          {'mild': dict(hold=2), 'moderate': dict(hold=4), 'severe': dict(hold=8)},
    'smoothing':           {'mild': dict(window=3), 'moderate': dict(window=7), 'severe': dict(window=13)},
    'motion_blur':         {'mild': dict(window=2), 'moderate': dict(window=4), 'severe': dict(window=7)},
    'mesh_decimation':     {'mild': dict(fraction=0.5), 'moderate': dict(fraction=0.2), 'severe': dict(fraction=0.05)},
    'vertex_quantization': {'mild': dict(step_m=0.003), 'moderate': dict(step_m=0.01), 'severe': dict(step_m=0.03)},
    'param_quantization':  {'mild': dict(bits=8), 'moderate': dict(bits=5), 'severe': dict(bits=3)},
}
DISTORTION_TYPES = list(LEVELS.keys())


# ---------------------------------------------------------------- core helpers (Milestone 3/4)
def load_subject(subject_id):
    param_path = os.path.join(THUMAN_ROOT, 'smplx', subject_id, 'smplx_param.pkl')
    with open(param_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    return {k: np.asarray(v).reshape(1, -1).astype(np.float32) for k, v in raw.items()}


def recon_pose(betas, body_pose, global_orient):
    with torch.no_grad():
        out = model(betas=betas, body_pose=body_pose, global_orient=global_orient, return_verts=True)
    v = out.vertices[0].cpu().numpy()
    mesh = trimesh.Trimesh(v, model.faces, process=False)
    mesh.root = out.joints[0, 0].cpu().numpy()  # pelvis - see orient_mesh's docstring
    return mesh


def orient_mesh(mesh):
    """Center on the pelvis joint, not the vertex mean. Measured directly: under moderate
    jitter, frame-to-frame vertex-mean drift reaches 10.5cm (vs ~0.7cm for the clean reference)
    because a limb flung out by noise drags the *mean of every vertex* toward it. Recentering
    each frame on that unstable mean, then viewing through a fixed perspective camera, is what
    produced the reported "weird camera movement" (lateral drift) and "thin then thick" (drift
    toward/away from the camera changing apparent size) - not two bugs, one cause. The pelvis
    joint is a single stable anatomical point that distortions don't drag around."""
    oriented = mesh.copy()
    center = getattr(mesh, 'root', None)
    if center is None:
        center = mesh.vertices.mean(axis=0)  # fallback for meshes without a tracked root
    oriented.vertices -= center
    return oriented


NO_AXES = dict(visible=False, showgrid=False, showticklabels=False, showbackground=False,
               zeroline=False, showspikes=False)


def fixed_scene(axis_range):
    """A CLEAN_SCENE with an explicit, pre-computed axis range instead of per-frame auto-fitting.
    Fixes two *separate* dolly-zoom sources, not one:

    1. (originally fixed) aspectmode='data' recomputing the scene's effective *scale* from
       whatever range/data was passed in per frame - a frame with a larger pose extent (e.g.
       mid-kick) auto-scaled to fill the same panel as a compact standing pose, so the same
       fixed camera.eye ended up at a different *effective* distance depending on that single
       frame's own bounding box. Fixed by using one shared range (UNIVERSAL_RANGE) instead of
       auto-fit for every frame, clip, and subject.

    2. (found via the three.js ground-truth comparison, 09_ThreeJS_Mesh_Viewer.ipynb) even with
       that fixed range, aspectmode='data' STILL re-derives the drawn x:y:z *proportions* from
       each frame's own mesh content - range only controls what part of space is visible, never
       the proportions it's drawn in. Measured directly: a walk clip whose real mesh-space
       vertical extent varies by 2.8% across 40 frames was rendered by aspectmode='data' as a
       1.25x pixel-height swing (a ~9x amplification) - a real, confirmed dolly-zoom that
       survived every earlier "fixed camera" fix because none of them touched aspectmode.
       aspectmode='manual' with FIXED_ASPECTRATIO computed once (never re-derived per frame)
       drops that swing back to 1.05x, matching an independent three.js reference render and
       the mesh's own ground truth. This is what actually made the compared renders look
       "irregular" throughout this whole investigation."""
    return dict(aspectmode='manual', aspectratio=FIXED_ASPECTRATIO, bgcolor='black',
                xaxis=dict(range=axis_range['x'], **NO_AXES),
                yaxis=dict(range=axis_range['y'], **NO_AXES),
                zaxis=dict(range=axis_range['z'], **NO_AXES))


# A fixed camera box, chosen once by hand and used identically for every clip, subject, and
# distortion - not computed from any clip's content at all. Three rounds of "compute the right
# box from this clip's own motion" (forced cube, then per-axis, then robust-percentile) each
# fixed one failure mode and exposed another, because the box kept changing based on what a
# specific clip's specific frames happened to contain. A real tripod camera doesn't renegotiate
# its field of view based on the subject's bounding box either - it just points somewhere fixed,
# and the subject's apparent size varies naturally with their real pose, which is correct, not a
# bug to engineer around. BASE_HALF_EXTENT is generous enough to contain any normal standing/
# moving human pose (pelvis-centered) without clipping; DISTANCE_SCALE and CAMERA_EYE together
# set how big the body reads in frame - see CAMERA_EYE.
BASE_HALF_EXTENT = {'x': 0.75, 'y': 0.75, 'z_lo': 1.15, 'z_hi': 1.05}
DISTANCE_SCALE = 1.0
UNIVERSAL_RANGE = {
    'x': [-BASE_HALF_EXTENT['x'] * DISTANCE_SCALE, BASE_HALF_EXTENT['x'] * DISTANCE_SCALE],
    'y': [-BASE_HALF_EXTENT['y'] * DISTANCE_SCALE, BASE_HALF_EXTENT['y'] * DISTANCE_SCALE],
    'z': [-BASE_HALF_EXTENT['z_lo'] * DISTANCE_SCALE, BASE_HALF_EXTENT['z_hi'] * DISTANCE_SCALE],
}

# The x:y:z proportions fixed_scene draws the scene in - computed ONCE, here, from
# UNIVERSAL_RANGE's own spans, and never re-derived from any mesh/frame's actual content. That
# "never re-derived per frame" is the whole point (see fixed_scene's docstring); because every
# frame is now rendered with this same UNIVERSAL_RANGE too, the range and the aspectratio have
# identical proportions, so Plotly maps data -> box isotropically (no axis-dependent stretch).
_range_span = {k: UNIVERSAL_RANGE[k][1] - UNIVERSAL_RANGE[k][0] for k in UNIVERSAL_RANGE}
_max_span = max(_range_span.values())
FIXED_ASPECTRATIO = {k: _range_span[k] / _max_span for k in _range_span}

# How far the (fixed) camera sits from the body, as a fraction of Plotly's default eye distance
# along the (1.8, -1.8, 1.2) direction. DISTANCE_SCALE alone saturates around ~40% frame fill
# below ~0.75 (the range shrinks but Plotly's eye is specified relative to the range, so it
# follows the range in and the view barely changes); pulling the eye itself in is what actually
# zooms. 0.6 measured ~50-55% frame-height fill on the tallest walk / front_kick poses with
# ~22% margin top and bottom - a much bigger body than the old ~15%, for judging mesh-level
# distortions (decimation, vertex quantization) in the MOS study, without clipping.
CAMERA_EYE = dict(x=1.8 * 0.6, y=-1.8 * 0.6, z=1.2 * 0.6)


def make_mesh_trace(mesh, color='royalblue', opacity=1.0):
    mesh = orient_mesh(mesh)
    return go.Mesh3d(x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
                      i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                      color=color, opacity=opacity, hoverinfo='skip',
                      # ambient dropped from 0.55 -> 0.22: high ambient is flat, uniform fill
                      # light everywhere, which is exactly what was washing out the volumetric
                      # gradient that makes a body read as a rounded 3D form rather than a flat
                      # cutout - a real reference (the AMASS showcase video) makes this same
                      # natural width variation look far less alarming purely because strong
                      # shading tells the eye "one solid body," not "changing outline."
                      lighting=dict(ambient=0.22, diffuse=0.9, specular=0.45, roughness=0.55,
                                    fresnel=0.15),
                      lightposition=dict(x=180, y=-60, z=90))  # more raking/side-lit than
                                                                # before, for a clearer gradient


def render_mesh_rgb(mesh, width=1280, height=1280):
    fig = go.Figure(data=[make_mesh_trace(mesh)])
    fig.update_layout(width=width, height=height, margin=dict(l=0, r=0, t=0, b=0),
                       paper_bgcolor='black',
                       scene=dict(camera=dict(eye=CAMERA_EYE), **fixed_scene(UNIVERSAL_RANGE)))
    png_bytes = pio.to_image(fig, format='png')
    return np.array(Image.open(io.BytesIO(png_bytes)).convert('RGB'))


def _render_chunk_rgb(meshes, panel_size, axis_ranges):
    """One kaleido call for a chunk of frames - see render_mesh_batch_rgb's docstring for why
    batching matters at all. Chunked (rather than one call for an entire ~150-frame clip) keeps
    each call's image at a sane, fast-to-render width instead of one extreme-aspect-ratio image.
    axis_ranges: one range dict per mesh (in practice always UNIVERSAL_RANGE for every frame -
    the per-mesh list is kept so callers can still pass something else if they need to)."""
    from plotly.subplots import make_subplots
    n = len(meshes)
    fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'scene'}] * n], horizontal_spacing=0)
    for i, mesh in enumerate(meshes):
        fig.add_trace(make_mesh_trace(mesh), row=1, col=i + 1)
        key = f'scene{i + 1}' if i > 0 else 'scene'
        fig.update_layout(**{key: dict(camera=dict(eye=CAMERA_EYE), **fixed_scene(axis_ranges[i]))})
    fig.update_layout(width=panel_size * n, height=panel_size, margin=dict(l=0, r=0, t=0, b=0),
                       paper_bgcolor='black')
    png_bytes = pio.to_image(fig, format='png')
    strip = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    w_each = strip.size[0] // n
    return [np.array(strip.crop((i * w_each, 0, (i + 1) * w_each, strip.size[1]))) for i in range(n)]


def render_mesh_batch_rgb(meshes, panel_size=1280, chunk_size=25):  # panel_size: multiple of 16
    """Render a whole animation's meshes in a small number of kaleido calls (chunks of
    `chunk_size` frames per call, batched as a 1xN subplot strip), then slice back into
    per-frame RGB arrays. kaleido 1.x has a large fixed per-call overhead (~5.5s, mostly
    independent of image content - measured directly) that dominates one-call-per-frame
    rendering; batching amortizes it, which is what makes rendering long, many-frame clips at
    225-video scale practical at all.

    chunk_size=25 (not the 448px-panel era's chunk_size=50) - at panel_size=1280 each chunk is
    a chunk_size*1280-wide strip, and a wider strip costs real per-pixel encode/transfer time on
    top of kaleido's fixed per-call overhead, so the two no longer trade off the same way:
    measured directly at 1280px, chunk_size=25 gave ~0.49s/frame (best), 20 gave ~0.54s/frame,
    and 30 fell off a cliff to ~4.9s/frame (some kaleido/Chromium internal limit on strip width,
    not a gradual slowdown) - don't push chunk_size up further without re-measuring.

    Every frame uses the one fixed UNIVERSAL_RANGE (a genuinely static tripod camera). An
    earlier version instead tracked each clip's real extent with a slowly-adapting per-frame
    range ('compute_smoothed_ranges', EMA + SMOOTH_MARGIN headroom) to damp the natural gait
    width-swing. That was removed once the aspectmode='data' bug was found (fixed_scene docstring
    #2): the swing it fought was mostly that bug's ~9x amplification, not real motion - a
    genuinely fixed camera shows the true swing at ~1.05x, which is fine. Worse, feeding a
    per-frame range whose x:y:z proportions differ from FIXED_ASPECTRATIO made Plotly stretch
    the body anisotropically (measured: rendered width:height 0.69 vs. the mesh's true 0.35 -
    a 2x horizontal squash). UNIVERSAL_RANGE everywhere keeps range and aspectratio proportions
    identical, so there's no stretch."""
    axis_ranges = [UNIVERSAL_RANGE] * len(meshes)
    frames = []
    for start in range(0, len(meshes), chunk_size):
        frames.extend(_render_chunk_rgb(meshes[start:start + chunk_size], panel_size,
                                         axis_ranges[start:start + chunk_size]))
    return frames


def load_amass_clip(rel_path):
    d = np.load(os.path.join(AMASS_ROOT, rel_path), allow_pickle=True)
    return dict(root_orient=torch.tensor(d['root_orient'], dtype=torch.float32, device=device),
                pose_body=torch.tensor(d['pose_body'], dtype=torch.float32, device=device),
                n_frames=d['poses'].shape[0])


def retarget_amass_sequence(rel_path, stride=STRIDE, max_frames=MAX_FRAMES):
    clip = load_amass_clip(rel_path)
    frame_indices = list(range(0, min(clip['n_frames'], stride * max_frames), stride))
    return [dict(frame_idx=idx, body_pose=clip['pose_body'][idx:idx + 1],
                  global_orient=clip['root_orient'][idx:idx + 1]) for idx in frame_indices]


# ---------------------------------------------------------------- distortion functions (Milestone 4)
def _deep_copy(results):
    return [dict(frame_idx=r['frame_idx'], body_pose=r['body_pose'].clone(),
                  global_orient=r['global_orient'].clone()) for r in results]


LOOP_REPEATS = 2   # short unique content (MAX_FRAMES), repeated - see loop_sequence
BLEND_FRAMES = 8   # ~1/5 of MAX_FRAMES - how many frames the seam transition spans


def blend_loop_seam(seq, blend_frames):
    """A copy of seq whose last `blend_frames` frames ease toward seq's own first frame's pose,
    so repeating this sequence back-to-back has no visible "pop" at the seam. A straight loop of
    a non-cyclic walk segment almost never has matching start/end poses; blending in POSE SPACE
    (not a pixel crossfade, which would ghost two different limb configurations on top of each
    other) produces one genuinely correct intermediate pose instead of a double-exposure."""
    out = _deep_copy(seq)
    n = len(seq)
    start_body, start_go = seq[0]['body_pose'], seq[0]['global_orient']
    for i in range(blend_frames):
        idx = n - blend_frames + i
        t = (i + 1) / blend_frames  # ramps from near-0 (mostly the real tail) to 1 (start pose)
        out[idx]['body_pose'] = seq[idx]['body_pose'] * (1 - t) + start_body * t
        out[idx]['global_orient'] = seq[idx]['global_orient'] * (1 - t) + start_go * t
    return out


def loop_sequence(seq, n_repeats=LOOP_REPEATS, blend_frames=BLEND_FRAMES):
    """Repeat a short sequence n_repeats times to reach a real viewing duration, with a
    pose-space blend at every seam except the last (which keeps its natural, unblended ending).
    Applied once, right after retargeting and before any distortion - every distortion type
    downstream (jitter's per-frame noise included) then just operates on this longer sequence
    exactly as it would on any other, no special-casing needed."""
    if n_repeats <= 1:
        return seq
    blended = blend_loop_seam(seq, blend_frames)
    out = []
    for _ in range(n_repeats - 1):
        out.extend(_deep_copy(blended))
    out.extend(_deep_copy(seq))
    for i, r in enumerate(out):
        r['frame_idx'] = i  # renumber sequentially across the whole looped output
    return out


def apply_jitter(results, sigma_body, sigma_root, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    out = _deep_copy(results)
    for r in out:
        r['body_pose'] = r['body_pose'] + torch.randn(r['body_pose'].shape, generator=g, device=device) * sigma_body
        r['global_orient'] = r['global_orient'] + torch.randn(r['global_orient'].shape, generator=g, device=device) * sigma_root
    return out


def apply_joint_noise(results, sigma, window, seed=1):
    torch.manual_seed(seed)
    T = len(results)
    raw = torch.randn(T, 63, device=device) * sigma
    pad = window // 2
    padded = torch.cat([raw[:1].repeat(pad, 1), raw, raw[-1:].repeat(pad, 1)], dim=0)
    smoothed = torch.stack([padded[i:i + window].mean(dim=0) for i in range(T)])
    out = _deep_copy(results)
    for i, r in enumerate(out):
        r['body_pose'] = r['body_pose'] + smoothed[i:i + 1]
    return out


def apply_frame_drop(results, hold):
    out = _deep_copy(results)
    for i in range(len(out)):
        src = (i // hold) * hold
        out[i]['body_pose'] = results[src]['body_pose'].clone()
        out[i]['global_orient'] = results[src]['global_orient'].clone()
    return out


def apply_smoothing(results, window):
    T = len(results)
    body = torch.cat([r['body_pose'] for r in results], dim=0)
    root = torch.cat([r['global_orient'] for r in results], dim=0)
    pad = window // 2
    body_p = torch.cat([body[:1].repeat(pad, 1), body, body[-1:].repeat(pad, 1)], dim=0)
    root_p = torch.cat([root[:1].repeat(pad, 1), root, root[-1:].repeat(pad, 1)], dim=0)
    body_s = torch.stack([body_p[i:i + window].mean(dim=0) for i in range(T)])
    root_s = torch.stack([root_p[i:i + window].mean(dim=0) for i in range(T)])
    out = _deep_copy(results)
    for i, r in enumerate(out):
        r['body_pose'] = body_s[i:i + 1]
        r['global_orient'] = root_s[i:i + 1]
    return out


def apply_param_quantization(results, bits):
    step = 2 * np.pi / (2 ** bits)
    out = _deep_copy(results)
    for r in out:
        r['body_pose'] = torch.round(r['body_pose'] / step) * step
        r['global_orient'] = torch.round(r['global_orient'] / step) * step
    return out


def apply_mesh_decimation(mesh, fraction):
    target = max(4, int(len(mesh.faces) * fraction))
    decimated = mesh.simplify_quadric_decimation(face_count=target)
    decimated.root = getattr(mesh, 'root', None)  # simplification doesn't move the pelvis
    return decimated


def apply_vertex_quantization(mesh, step_m):
    m = mesh.copy()
    m.vertices = np.round(m.vertices / step_m) * step_m
    m.root = getattr(mesh, 'root', None)  # .copy() doesn't preserve custom attributes
    return m


def apply_motion_blur(rgb_frames, window):
    arr = np.stack(rgb_frames).astype(np.float32)
    pad = window // 2
    arr_p = np.concatenate([np.repeat(arr[:1], pad, axis=0), arr, np.repeat(arr[-1:], pad, axis=0)], axis=0)
    blurred = np.stack([arr_p[i:i + window].mean(axis=0) for i in range(len(rgb_frames))])
    return np.clip(blurred, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- unified renderer
POSE_SPACE_TYPES = {
    'jitter': apply_jitter, 'joint_noise': apply_joint_noise,
    'frame_drop': apply_frame_drop, 'smoothing': apply_smoothing,
    'param_quantization': apply_param_quantization,
}
MESH_SPACE_TYPES = {'mesh_decimation': apply_mesh_decimation, 'vertex_quantization': apply_vertex_quantization}


VIDEO_FPS = 8   # v3: moderated back up from v2's 5fps now that duration comes from looping
                # (MAX_FRAMES=40 x LOOP_REPEATS=2 = 80 frames / 8fps = 10s) rather than from one
                # long slow pass - 5fps was slow enough to start looking floaty/unnatural on its
                # own. 40 frames at stride=2 = 80 native (120fps) frames = 0.67s of real motion,
                # played back over 5s (40 frames / 8fps): ~7.5x slow motion for the unique
                # content, well short of v2's ~12x. Total output lands at ~10s, matching the
                # reference MOS tool's own clip length.


def render_animation(betas, sequence, out_path, fps=VIDEO_FPS):
    """Render a pose sequence (list of {body_pose, global_orient}) to an MP4."""
    meshes = [recon_pose(betas, r['body_pose'], r['global_orient']) for r in sequence]
    frames = render_mesh_batch_rgb(meshes)
    save_video(frames, out_path, fps)
    return len(frames)


def save_video(frames, out_path, fps=VIDEO_FPS):
    """Write frames as an H.264 MP4 - not a GIF. GIF's 256-color palette would quantize/dither
    every stimulus (reference included), which is exactly the kind of uncontrolled compression
    artifact we can't have sitting underneath distortion types we're trying to study in a
    controlled way. Real video also gives native browser playback controls (play/pause/scrub),
    which the pairwise MOS tool needs anyway."""
    h, w = frames[0].shape[:2]
    # H.264 requires even width/height
    w, h = w - (w % 2), h - (h % 2)
    with imageio.get_writer(out_path, format='FFMPEG', fps=fps, codec='libx264',
                             output_params=['-crf', '18', '-pix_fmt', 'yuv420p']) as writer:
        for f in frames:
            writer.append_data(f[:h, :w])


def render_distorted(betas, reference_seq, dist_type, severity, out_path):
    params = LEVELS[dist_type][severity]

    if dist_type in POSE_SPACE_TYPES:
        distorted_seq = POSE_SPACE_TYPES[dist_type](reference_seq, **params)
        n = render_animation(betas, distorted_seq, out_path)

    elif dist_type in MESH_SPACE_TYPES:
        fn = MESH_SPACE_TYPES[dist_type]
        meshes = [fn(recon_pose(betas, r['body_pose'], r['global_orient']), **params) for r in reference_seq]
        frames = render_mesh_batch_rgb(meshes)
        save_video(frames, out_path)
        n = len(frames)

    elif dist_type == 'motion_blur':
        meshes = [recon_pose(betas, r['body_pose'], r['global_orient']) for r in reference_seq]
        raw_frames = render_mesh_batch_rgb(meshes)
        blurred = apply_motion_blur(raw_frames, **params)
        save_video(list(blurred), out_path)
        n = len(blurred)

    else:
        raise ValueError('unknown distortion type: ' + dist_type)

    return n, params


# ---------------------------------------------------------------- main batch loop
def main():
    manifest_path = os.path.join(OUT_DIR, 'manifest.csv')
    manifest_rows = []
    t_start = time.time()
    total_jobs = len(SUBJECTS) * len(CLIPS) * (1 + len(DISTORTION_TYPES) * 3)
    done = 0

    for subject in SUBJECTS:
        betas = torch.tensor(load_subject(subject)['betas'], device=device)
        for clip_name, clip_path in CLIPS.items():
            reference_seq = loop_sequence(retarget_amass_sequence(clip_path))

            ref_fname = f'{subject}_{clip_name}_reference.mp4'
            ref_out = os.path.join(OUT_DIR, ref_fname)
            n_frames = render_animation(betas, reference_seq, ref_out)
            manifest_rows.append(dict(file=ref_fname, subject=subject, clip=clip_name,
                                       distortion_type='none', severity='none', params='', n_frames=n_frames))
            done += 1
            print(f'[{done}/{total_jobs}] {ref_fname}  ({time.time()-t_start:.0f}s elapsed)')

            for dist_type in DISTORTION_TYPES:
                for severity in ['mild', 'moderate', 'severe']:
                    fname = f'{subject}_{clip_name}_{dist_type}_{severity}.mp4'
                    out_path = os.path.join(OUT_DIR, fname)
                    n_frames, params = render_distorted(betas, reference_seq, dist_type, severity, out_path)
                    manifest_rows.append(dict(file=fname, subject=subject, clip=clip_name,
                                               distortion_type=dist_type, severity=severity,
                                               params=str(params), n_frames=n_frames))
                    done += 1
                    if done % 10 == 0 or done == total_jobs:
                        print(f'[{done}/{total_jobs}] {fname}  ({time.time()-t_start:.0f}s elapsed)')

    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'subject', 'clip', 'distortion_type', 'severity', 'params', 'n_frames'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f'\nDone. {len(manifest_rows)} files written to {OUT_DIR}')
    print(f'Manifest: {manifest_path}')
    print(f'Total time: {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
