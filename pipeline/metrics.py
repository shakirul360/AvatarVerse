"""Objective geometric quality metrics (Chamfer/Hausdorff distance), computed from the RAW
(pre-Draco) posed positions - not the compressed geometry bundle - so the metric isn't itself
contaminated by delivery-compression error (same reasoning as why the delivery encoding
shouldn't corrupt the reference condition: keep the two concerns separate).

True bidirectional nearest-neighbor distance, not index-aligned diffing, since mesh_decimation
instances have a different vertex count than the reference - the same pattern
distortions.build_decimation() already uses internally (cKDTree(ref_positions).query(...)).
"""
import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of (mean nearest-neighbor distance a->b) and (mean nearest-neighbor distance b->a).
    a, b: (N_a, 3) / (N_b, 3) point sets - need not be the same size."""
    tree_b = cKDTree(b)
    d_ab = tree_b.query(a, k=1)[0]
    tree_a = cKDTree(a)
    d_ba = tree_a.query(b, k=1)[0]
    return float((d_ab.mean() + d_ba.mean()) / 2)


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance: max over both directions of the max nearest-neighbor
    distance - the single worst-case point-to-surface gap between the two point sets."""
    tree_b = cKDTree(b)
    d_ab = tree_b.query(a, k=1)[0]
    tree_a = cKDTree(a)
    d_ba = tree_a.query(b, k=1)[0]
    return float(max(d_ab.max(), d_ba.max()))


def compute_geometry_metrics(ref_positions: np.ndarray, positions: np.ndarray) -> dict:
    """ref_positions, positions: (n_frames, N_ref, 3) / (n_frames, N, 3) - same frame count,
    possibly different vertex count (mesh_decimation). Per-frame chamfer/hausdorff, summarized
    across the clip.

    Builds each frame's two cKDTrees once and reuses them for both metrics, rather than calling
    chamfer_distance()/hausdorff_distance() separately (each of which builds its own pair) - that
    redundant 4-trees-per-frame version was measured taking ~2x longer per instance during the
    Phase 2 pilot (a real cost at 37 instances/combo x 30 combos), with identical results either
    way since both metrics are pure functions of the same two nearest-neighbor distance arrays."""
    n_frames = ref_positions.shape[0]
    chamfer = np.empty(n_frames)
    hausdorff = np.empty(n_frames)
    for i in range(n_frames):
        ref_i, pos_i = ref_positions[i], positions[i]
        tree_ref, tree_pos = cKDTree(ref_i), cKDTree(pos_i)
        d_ref_to_pos = tree_pos.query(ref_i, k=1)[0]
        d_pos_to_ref = tree_ref.query(pos_i, k=1)[0]
        chamfer[i] = (d_ref_to_pos.mean() + d_pos_to_ref.mean()) / 2
        hausdorff[i] = max(d_ref_to_pos.max(), d_pos_to_ref.max())
    return dict(
        chamfer_mean=float(chamfer.mean()), chamfer_max=float(chamfer.max()),
        hausdorff_max=float(hausdorff.max()),
    )
