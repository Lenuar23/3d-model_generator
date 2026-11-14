"""Refine an indexed surface with visibility-checked normal observations."""
from dataclasses import replace

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import lsqr

from .hull_quality import agreement, rasterize
from .mesh_generator import validate
from .texture_mapper import bilinear


def decode(rgb, mask):
    """Decode front OpenGL normals in Photo2Mesh's +X right, +Y up, +Z front basis."""
    vectors = np.asarray(rgb, dtype=float) / 127.5 - 1
    lengths = np.linalg.norm(vectors, axis=-1)
    valid = mask & (lengths > .65) & (lengths < 1.35)
    return vectors / np.maximum(lengths[..., None], 1e-9), valid


def vertex_normals(vertices, faces):
    """Area-weighted normals, independent of the initial tessellation."""
    p = vertices[faces]
    cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    normals = np.zeros_like(vertices)
    for corner in faces.T:
        np.add.at(normals, corner, cross)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _mesh_edges(faces):
    """Extract sorted unique undirected edge vertex pairs from triangle faces."""
    pairs = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
    return np.unique(np.sort(pairs, axis=1), axis=0)


def _fuse_observations(hull, observations, basis):
    """Accumulate multi-view normal targets under one coordinate sign convention."""
    p, n = hull.vertices, hull.normals
    total = np.zeros_like(p)
    weight = np.zeros(len(p))
    visible_weight = np.zeros(len(p))
    view_hits = np.zeros(len(p), dtype=int)
    seen_samples = 0
    compat_samples = 0
    for view, rgb in observations:
        predicted, valid = decode(rgb, view.mask)
        predicted *= basis
        depth = rasterize(p, hull.faces, view)
        x, y, z, px, py, inside = view.camera.pixels(p, view.mask.shape)
        facing = view.camera.facing(p, n)
        ids = np.flatnonzero(
            inside
            & (facing > 0.12)
            & (distance_transform_edt(valid)[py, px] > 1.5)
            & np.isfinite(depth[py, px])
            & (z >= depth[py, px] - 0.65 * hull.step)
        )
        if not len(ids):
            continue
        view_boost = 1.25 if getattr(view, "name", "") == "front" else 1.0
        base_scores = view_boost * (facing[ids] ** 1.5) * getattr(view, "texture_weight", 1.0)
        visible_weight[ids] += base_scores
        normals = bilinear(predicted, x[ids], y[ids])
        length = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(length[:, None], 1e-8)
        compatible = np.einsum("ij,ij->i", normals, n[ids]) > 0.38
        compatible &= view.camera.facing(p[ids], normals) > 0.08
        seen_samples += len(ids)
        compat_samples += int(np.count_nonzero(compatible))
        ids, normals, scores = ids[compatible], normals[compatible], base_scores[compatible]
        if not len(ids):
            continue
        total[ids] += normals * scores[:, None]
        weight[ids] += scores
        view_hits[ids] += 1
    if not seen_samples or compat_samples < 0.65 * seen_samples:
        return np.zeros_like(p), np.zeros(len(p)), np.zeros(len(p), dtype=int)
    length = np.linalg.norm(total, axis=1)
    confidence = np.where((visible_weight > 0) & (length >= 0.76 * visible_weight), np.minimum(weight, 1.0), 0.0)
    desired = total / np.maximum(length[:, None], 1e-9)
    return desired, confidence, view_hits


def _regularize_targets(desired, confidence, faces):
    """Fill coherent 1-ring gaps and denoise with edge-stopping bilateral normal filtering."""
    edges = _mesh_edges(faces)
    a, b = edges.T
    desired = desired.copy()
    confidence = confidence.copy()
    # One-ring infill only where confident neighbors agree strongly on a single tangent plane.
    missing = np.flatnonzero(confidence <= 0)
    if len(missing):
        conf_a = np.where(confidence[a] > 0, confidence[a], 0.0)
        conf_b = np.where(confidence[b] > 0, confidence[b], 0.0)
        vec_sum = np.zeros_like(desired)
        w_sum = np.zeros(len(desired))
        np.add.at(vec_sum, a, desired[b] * conf_b[:, None])
        np.add.at(vec_sum, b, desired[a] * conf_a[:, None])
        np.add.at(w_sum, a, conf_b)
        np.add.at(w_sum, b, conf_a)
        v_norm = np.linalg.norm(vec_sum[missing], axis=1)
        agree = np.divide(v_norm, np.maximum(w_sum[missing], 1e-9))
        fill = (w_sum[missing] > 0.1) & (agree >= 0.88)
        fill_ids = missing[fill]
        desired[fill_ids] = vec_sum[fill_ids] / np.maximum(v_norm[fill][:, None], 1e-9)
        confidence[fill_ids] = np.minimum(0.45, 0.5 * w_sum[fill_ids])
    # Three edge-stopping bilateral passes denoise planar walls and curves while preserving creases.
    active = confidence > 0
    for _ in range(3):
        both = active[a] & active[b]
        dot = np.clip(np.einsum("ij,ij->i", desired[a], desired[b]), -1.0, 1.0)
        w_edge = np.where(both, np.exp(-np.maximum(0.0, 1.0 - dot) / 0.22), 0.0)
        accum = desired * confidence[:, None]
        w_tot = confidence.copy()
        wa = w_edge * confidence[b]
        wb = w_edge * confidence[a]
        np.add.at(accum, a, desired[b] * wa[:, None])
        np.add.at(accum, b, desired[a] * wb[:, None])
        np.add.at(w_tot, a, wa)
        np.add.at(w_tot, b, wb)
        norms = np.linalg.norm(accum, axis=1, keepdims=True)
        desired = np.where(active[:, None] & (norms > 1e-8), accum / np.maximum(norms, 1e-9), desired)
    return desired, confidence


def targets(hull, observations, basis=(1, 1, 1), return_info=False, auto_basis=False):
    """Fuse predicted normals only on visible, mask-interior, consistent surfaces."""
    primary = tuple(int(v) for v in basis)
    desired, confidence, hits = _fuse_observations(hull, observations, primary)
    chosen = primary
    support = int(np.count_nonzero(confidence))
    if auto_basis and support < 0.15 * len(hull.vertices):
        alt = (-primary[0], primary[1], -primary[2])
        if alt != primary:
            alt_d, alt_c, alt_h = _fuse_observations(hull, observations, alt)
            alt_support = int(np.count_nonzero(alt_c))
            if alt_support >= max(20, 2 * support) and alt_support >= 0.20 * len(hull.vertices):
                desired, confidence, hits, chosen = alt_d, alt_c, alt_h, alt
    desired, confidence = _regularize_targets(desired, confidence, hull.faces)
    if return_info:
        return desired, confidence, {
            "basis_used": list(chosen),
            "fused_vertices": int(np.count_nonzero(confidence)),
            "multi_view_vertices": int(np.count_nonzero(hits >= 2)),
        }
    return desired, confidence


def _silhouette_constraints(p, n, f, views, depths=()):
    """Build linear rim constraints anchoring silhouette vertices onto view mask boundaries."""
    if not views:
        return coo_matrix((0, len(p))).tocsr(), np.zeros(0)
    rows, cols, vals, targets_rhs = [], [], [], []
    row_id = 0
    for idx, view in enumerate(views):
        depth = depths[idx] if idx < len(depths) else rasterize(p, f, view)
        x, y, z, px, py, inside = view.camera.pixels(p, view.mask.shape)
        facing = view.camera.facing(p, n)
        sdf_px = np.where(view.mask, distance_transform_edt(view.mask) - 0.5,
                          -(distance_transform_edt(~view.mask) - 0.5))
        grad_y, grad_x = np.gradient(sdf_px)
        local_sdf = np.zeros(len(p))
        in_ids = np.flatnonzero(inside)
        if not len(in_ids):
            continue
        local_sdf[in_ids] = bilinear(sdf_px[..., None], x[in_ids], y[in_ids])[:, 0]
        valid_depth = depth[np.isfinite(depth)]
        depth_tol = 0.08 * max(1e-3, float(np.ptp(valid_depth))) if len(valid_depth) else 0.05
        rim = (
            inside
            & np.isfinite(depth[py, px])
            & (np.abs(facing) < 0.42)
            & (np.abs(local_sdf) <= 2.2)
            & (z >= depth[py, px] - depth_tol)
        )
        rim_ids = np.flatnonzero(rim)
        if not len(rim_ids):
            continue
        axes = view.camera.axes
        gx = grad_x[py[rim_ids], px[rim_ids]]
        gy = grad_y[py[rim_ids], px[rim_ids]]
        inward_3d = gx[:, None] * axes[0][None, :] - gy[:, None] * axes[1][None, :]
        inward_norm = np.linalg.norm(inward_3d, axis=1, keepdims=True)
        good = inward_norm[:, 0] > 0.2
        if not good.any():
            continue
        rim_ids = rim_ids[good]
        inward_3d = inward_3d[good] / inward_norm[good]
        coeff = np.einsum("ij,ij->i", n[rim_ids], inward_3d)
        # Ignore vertices whose displacement vector is strictly perpendicular to the image rim normal.
        keep = np.abs(coeff) > 0.15
        if not keep.any():
            continue
        rim_ids = rim_ids[keep]
        coeff = coeff[keep]
        if view.camera.distance is None:
            scale = view.camera.pixels_per_unit
        else:
            axial = np.maximum(1e-3, view.camera.distance - z[rim_ids])
            scale = view.camera.focal_length / axial
        # Negative sub-pixel SDF is outside, so (-local_sdf / scale) shifts cleanly onto contour.
        goal = np.clip(-local_sdf[rim_ids] / np.maximum(scale, 1e-6), -0.05, 0.05)
        w = 0.65 * (0.45 if view.inferred else 1.0)
        count = len(rim_ids)
        rows.extend(range(row_id, row_id + count))
        cols.extend(rim_ids.tolist())
        vals.extend((w * coeff).tolist())
        targets_rhs.extend((w * goal).tolist())
        row_id += count
    if not rows:
        return coo_matrix((0, len(p))).tocsr(), np.zeros(0)
    return coo_matrix((vals, (rows, cols)), shape=(row_id, len(p))).tocsr(), np.asarray(targets_rhs, float)


def _planar_coplanarity_constraints(p, n, ref_n, desired, confidence, edges, edge_div, crease_edge, planarity):
    """Build 2-ring coplanar chord constraints across high-confidence same-plane regions."""
    a, b = edges.T
    same_plane = (confidence[a] > 0.05) & (confidence[b] > 0.05) & (crease_edge < 0.10) & (edge_div < 0.03)
    if not np.any(same_plane):
        return coo_matrix((0, len(p))).tocsr(), np.zeros(0)
    sa, sb = a[same_plane], b[same_plane]
    adj = coo_matrix((np.ones(2 * len(sa)), (np.r_[sa, sb], np.r_[sb, sa])), shape=(len(p), len(p))).tocsr()
    adj2 = (adj @ adj).tocoo()
    mask2 = (adj2.row < adj2.col) & (planarity[adj2.row] > 0.25) & (planarity[adj2.col] > 0.25)
    r2, c2 = adj2.row[mask2], adj2.col[mask2]
    if not len(r2):
        return coo_matrix((0, len(p))).tocsr(), np.zeros(0)
    # Drop direct 1-ring edges already covered by primary tangent constraints.
    direct = set(zip(a.tolist(), b.tolist()))
    non_edge = np.array([(ri, ci) not in direct for ri, ci in zip(r2.tolist(), c2.tolist())], bool)
    r2, c2 = r2[non_edge], c2[non_edge]
    if not len(r2):
        return coo_matrix((0, len(p))).tocsr(), np.zeros(0)
    pair_dot = np.einsum("ij,ij->i", ref_n[r2], ref_n[c2])
    flat_pairs = (pair_dot > 0.975) & (confidence[r2] > 0.05) & (confidence[c2] > 0.05)
    r2, c2 = r2[flat_pairs], c2[flat_pairs]
    if not len(r2):
        return coo_matrix((0, len(p))).tocsr(), np.zeros(0)
    deg = np.bincount(np.r_[r2, c2], minlength=len(p))
    avg_n = desired[r2] + desired[c2]
    avg_n /= np.maximum(np.linalg.norm(avg_n, axis=1, keepdims=True), 1e-9)
    signed2 = np.einsum("ij,ij->i", p[c2] - p[r2], avg_n)
    na2 = np.einsum("ij,ij->i", n[r2], avg_n)
    nb2 = np.einsum("ij,ij->i", n[c2], avg_n)
    w2 = 0.55 * np.sqrt(planarity[r2] * planarity[c2] * confidence[r2] * confidence[c2]) / np.sqrt(
        np.maximum(1.0, 0.5 * (deg[r2] + deg[c2]))
    )
    ids2 = np.arange(len(r2))
    mat2 = coo_matrix((np.r_[-w2 * na2, w2 * nb2], (np.r_[ids2, ids2], np.r_[r2, c2])),
                      shape=(len(r2), len(p))).tocsr()
    return mat2, -w2 * signed2


def refine(hull, desired, confidence, views=(), strength=.65, basis_used=None):
    """Solve anisotropic 5-term feature-preserving least squares with safety backtracking."""
    if not 0 <= strength <= 1:
        raise ValueError("Normal refinement strength must be between zero and one.")
    p = hull.vertices.astype(float)
    n = hull.normals
    f = hull.faces
    confidence = np.asarray(confidence, dtype=float)
    if not strength or np.count_nonzero(confidence) < 10:
        return hull, {"accepted": False, "reason": "Insufficient enabled normal evidence"}

    edges = _mesh_edges(f)
    a, b = edges.T
    ref_n = np.where(confidence[:, None] > 0.05, desired, n)
    ref_n = ref_n / np.maximum(np.linalg.norm(ref_n, axis=1, keepdims=True), 1e-9)
    edge_div = 1.0 - np.clip(np.einsum("ij,ij->i", ref_n[a], ref_n[b]), -1.0, 1.0)
    crease_edge = np.clip((edge_div - 0.06) / 0.25, 0.0, 1.0)

    # Estimate per-vertex planarity from maximum local target normal divergence on edges.
    max_div = np.zeros(len(p))
    np.maximum.at(max_div, a, edge_div)
    np.maximum.at(max_div, b, edge_div)
    planarity = np.where(confidence > 0.05, np.clip(1.0 - max_div / 0.04, 0.0, 1.0), 0.0)
    pair_planarity = 0.5 * (planarity[a] + planarity[b])

    # 1. Normal-guided tangent alignment with downweighted cross-crease chamfering.
    edge_normal = desired[a] + desired[b]
    edge_normal /= np.maximum(np.linalg.norm(edge_normal, axis=1, keepdims=True), 1e-9)
    w_tan = np.sqrt(confidence[a] * confidence[b]) * (1.0 - 0.85 * crease_edge)
    ids = np.arange(len(edges))
    signed = np.einsum("ij,ij->i", p[b] - p[a], edge_normal)
    na = np.einsum("ij,ij->i", n[a], edge_normal)
    nb = np.einsum("ij,ij->i", n[b], edge_normal)
    mat_normal = coo_matrix((np.r_[-w_tan * na, w_tan * nb], (np.r_[ids, ids], np.r_[a, b])),
                            shape=(len(edges), len(p))).tocsr()
    rhs_normal = -w_tan * signed

    initial_depths = [rasterize(p, f, view) for view in views]
    initial_masks = [np.isfinite(d) for d in initial_depths]

    # 2. Silhouette contour regularization on grazing rim vertices.
    mat_sil, rhs_sil = _silhouette_constraints(p, n, f, views, initial_depths)

    # 3. Feature-preserving anisotropic displacement smoothing (reduced across creases & flat patches).
    w_smooth = 0.42 * (1.0 - 0.82 * crease_edge) * (1.0 - 0.55 * pair_planarity)
    mat_smooth = coo_matrix((np.r_[-w_smooth, w_smooth], (np.r_[ids, ids], np.r_[a, b])),
                            shape=(len(edges), len(p))).tocsr()
    rhs_smooth = np.zeros(len(edges))

    # 4. Extended 2-ring coplanar constraints on flat faces.
    mat_feature, rhs_feature = _planar_coplanarity_constraints(
        p, n, ref_n, desired, confidence, edges, edge_div, crease_edge, planarity
    )

    # 5. Anchor regularizer preventing unbounded drift.
    mat_anchor = 0.20 * eye(len(p))
    rhs_anchor = np.zeros(len(p))

    system = vstack((mat_normal, mat_sil, mat_smooth, mat_feature, mat_anchor)).tocsr()
    rhs = np.r_[rhs_normal, rhs_sil, rhs_smooth, rhs_feature, rhs_anchor]
    delta = lsqr(system, rhs, atol=1e-6, btol=1e-6, iter_lim=120)[0]
    delta = np.clip(delta, -1.15 * hull.step, 1.15 * hull.step) * strength

    def angular(normals):
        angles = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", normals, desired), -1, 1)))
        return float(np.average(angles, weights=confidence))

    before = angular(n)
    objective_before = float(np.sum(rhs ** 2))
    trace = []
    initial_faces = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])

    def signed_volume(points):
        triangles = points[f]
        return np.einsum("ij,ij->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6

    volume = signed_volume(p)
    for factor in (1.0, 0.7, 0.45, 0.25, 0.125):
        candidate = p + factor * delta[:, None] * n
        normals = vertex_normals(candidate, f)
        error = angular(normals)
        residual = system @ (factor * delta) - rhs
        objective = float(np.sum(residual ** 2))
        trace.append({"step_scale": factor, "angle": error, "objective": objective})
        if error >= before or objective >= objective_before:
            continue
        new_faces = np.cross(candidate[f[:, 1]] - candidate[f[:, 0]], candidate[f[:, 2]] - candidate[f[:, 0]])
        if np.any(np.einsum("ij,ij->i", initial_faces, new_faces) <= 0):
            continue
        volume_change = abs(signed_volume(candidate) - volume) / max(abs(volume), 1e-12)
        if volume_change > 0.06:
            continue
        projected = [np.isfinite(rasterize(candidate, f, view)) for view in views]
        overlaps = [agreement(new, initial) for new, initial in zip(projected, initial_masks)]
        if overlaps and min(overlaps) < 0.965:
            continue
        if any(not view.inferred and agreement(new, view.mask) < agreement(old, view.mask) - 0.005
               for view, new, old in zip(views, projected, initial_masks)):
            continue
        result = replace(hull, vertices=candidate, normals=normals)
        checks = validate(result)
        if (checks["degenerate_triangles"] or not checks["unit_normals"]
                or not checks["watertight"] or checks["connected_components"] != 1):
            continue
        return result, {
            "accepted": True,
            "angular_before": before,
            "angular_after": error,
            "silhouette_stability": overlaps,
            "support_vertices": int(np.count_nonzero(confidence)),
            "planar_vertices": int(np.count_nonzero(planarity >= 0.45)),
            "curved_vertices": int(np.count_nonzero((planarity < 0.45) & (confidence > 0))),
            "sharp_feature_edges": int(np.count_nonzero(crease_edge >= 0.5)),
            "basis_used": list(basis_used) if basis_used is not None else [1, 1, 1],
            "relative_volume_change": float(volume_change),
            "objective_before": objective_before,
            "trials": trace,
        }
    return hull, {
        "accepted": False,
        "angular_before": before,
        "basis_used": list(basis_used) if basis_used is not None else [1, 1, 1],
        "trials": trace,
        "reason": "No bounded update met angle, objective, and silhouette checks",
    }

