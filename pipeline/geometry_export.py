"""Geometry export for the interactive free-orbit-viewer MOS track: persists Draco-compressed
per-instance geometry bundles instead of (or in addition to) rendering to video.

Validated encoding recipe (Phase 0 spike, real numbers measured on subject 0000/walk, 302,021
verts, 46 frames - see /private/tmp/.../scratchpad/draco_spike*.py for the throwaway spike
scripts this was derived from):

  - Draco's real compression strength (5-20x) comes from mesh-CONNECTIVITY-based position
    prediction. A per-frame point cloud has no connectivity, so it only gets quantization
    savings (~2.4-3.2x measured) - and re-encoding full connectivity every frame would cost
    ~8MB/frame on its own (worse than not doing it), since faces/uv are identical every frame.
  - So: connectivity + UV + frame-0's absolute positions are Draco-MESH-encoded exactly once
    (base.drc), near-lossless. Every other frame is Draco-POINT-CLOUD-encoded as
    positions[i] - positions[0] (a delta) - deltas cluster tightly near zero (mean |delta| ~5mm
    vs an ~181cm body bbox for a walk cycle), which compresses better than absolute positions
    even though point clouds get no connectivity prediction.
  - Result: ~45MB/instance (8.2MB base + ~36.8MB frames for 46 frames), delta reconstruction
    error ~0.9mm - well below the smallest distortion severity's own effect size
    (vertex_quantization mild step_m=3mm), so Draco's own error doesn't confound the study.

CLI (Phase 1 smoke test - ONE instance, verify before scaling further):
  python -m pipeline.geometry_export --subject 0000 --clip walk --out <dir>
"""
import os, sys, json, argparse, time
import numpy as np
import DracoPy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skin_scan

BASE_QUANTIZATION_BITS = 14   # frame-0 mesh (connectivity+UV+positions) - near-lossless (~0.1mm)
DELTA_QUANTIZATION_BITS = 10  # per-frame position deltas - error ~0.9mm, see module docstring


def encode_base_mesh(positions_frame0, faces, uv, quantization_bits=BASE_QUANTIZATION_BITS):
    """One Draco MESH encode: connectivity + UV + absolute frame-0 positions. The only place
    connectivity is ever encoded - see module docstring for why every other frame is a point
    cloud instead."""
    return DracoPy.encode(
        positions_frame0.astype(np.float32), faces.astype(np.uint32),
        tex_coord=uv.astype(np.float64), quantization_bits=quantization_bits,
        compression_level=7, preserve_order=True,
    )


def encode_frame_deltas(positions, quantization_bits=DELTA_QUANTIZATION_BITS):
    """positions: (n_frames, n_verts, 3). Returns Draco POINT CLOUD encodes of
    positions[i] - positions[0] for i in 1..n_frames-1 (frame 0 is NOT re-encoded here - its
    absolute positions live in base.drc; the viewer treats it as delta-zero). One shared
    quantization box across the whole clip, not per-frame auto-fit, so decoded deltas don't pick
    up frame-to-frame quantization-grid drift that would look like injected jitter."""
    deltas = positions[1:] - positions[0:1]
    d_min = deltas.reshape(-1, 3).min(axis=0).astype(np.float64)
    d_max = deltas.reshape(-1, 3).max(axis=0)
    d_range = float((d_max - d_min).max())
    return [
        DracoPy.encode(
            deltas[i].astype(np.float32), None,
            quantization_bits=quantization_bits, quantization_range=d_range, quantization_origin=d_min,
            compression_level=1, preserve_order=True,
        )
        for i in range(len(deltas))
    ]


def write_geometry_bundle(out_dir, positions, faces, uv, texture_src, meta,
                          base_quantization_bits=BASE_QUANTIZATION_BITS,
                          delta_quantization_bits=DELTA_QUANTIZATION_BITS):
    """positions: (n_frames, n_verts, 3) float32, already posed (+ any pose/mesh distortion
    applied). faces/uv: constant for this subject+severity. Writes:
      base.drc               - connectivity + UV + frame-0 absolute positions
      frames/frame_%04d.drc  - per-frame position DELTAS for frames 1..n_frames-1 (frame 0 has
                                no file here - it's already in base.drc)
      texture.jpg, meta.json
    Returns byte-size stats for logging/validation."""
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    base_bytes = encode_base_mesh(positions[0], faces, uv, base_quantization_bits)
    with open(os.path.join(out_dir, 'base.drc'), 'wb') as f:
        f.write(base_bytes)

    frame_encodings = encode_frame_deltas(positions, delta_quantization_bits)
    for i, enc in enumerate(frame_encodings):
        with open(os.path.join(frames_dir, f'frame_{i + 1:04d}.drc'), 'wb') as f:
            f.write(enc)

    skin_scan.write_texture(texture_src, os.path.join(out_dir, 'texture.jpg'))

    n_frames, n_verts, _ = positions.shape
    bundle_meta = dict(meta, n_verts=n_verts, n_faces=len(faces), n_frames=n_frames,
                       n_delta_frames=n_frames - 1, draco=True,
                       base_quantization_bits=base_quantization_bits,
                       delta_quantization_bits=delta_quantization_bits)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(bundle_meta, f, indent=2)

    frame_bytes_total = sum(len(e) for e in frame_encodings)
    return dict(base_bytes=len(base_bytes), frame_bytes_total=frame_bytes_total,
               bundle_bytes_total=len(base_bytes) + frame_bytes_total,
               raw_baseline_bytes=n_verts * 3 * 4 * n_frames)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', default='0000')
    ap.add_argument('--clip', default='walk')
    ap.add_argument('--speed', type=float, default=0.75)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--frames', type=int, default=46)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    t0 = time.time()
    positions, faces, uv, texture_src, meta = skin_scan.skin_sequence(
        a.subject, a.clip, a.speed, a.fps, a.frames)
    print(f'posed {positions.shape[0]} frames, {positions.shape[1]} verts  ({time.time()-t0:.1f}s)')

    stats = write_geometry_bundle(a.out, positions, faces, uv, texture_src, meta)
    print(f'wrote {a.out}')
    print(f"  base.drc:          {stats['base_bytes']/1e6:.2f} MB")
    print(f"  frames/ total:     {stats['frame_bytes_total']/1e6:.2f} MB")
    print(f"  bundle total:      {stats['bundle_bytes_total']/1e6:.2f} MB "
         f"(raw baseline {stats['raw_baseline_bytes']/1e6:.1f} MB, "
         f"{stats['raw_baseline_bytes']/stats['bundle_bytes_total']:.1f}x compression)")
    print(f'  total time:        {time.time()-t0:.1f}s')
