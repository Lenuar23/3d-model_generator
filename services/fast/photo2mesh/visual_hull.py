"""Intersect calibrated silhouettes on a bounded regular voxel grid."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from skimage.measure import marching_cubes

from .camera import Camera
from .mesh_generator import generate


@dataclass
class View:
    rgb: np.ndarray
    mask: np.ndarray
    camera: Camera
    name: str
    inferred: bool = False
    texture_weight: float = 1.0

    def validate(self) -> None:
        if (self.mask.ndim != 2 or self.mask.dtype != bool or not self.mask.any()
            or self.rgb.shape != (*self.mask.shape, 3) or self.rgb.dtype != np.uint8):
            raise ValueError(f"{self.name}: expected a nonempty silhouette and matching RGB image.")


@dataclass
class Hull:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    occupied_points: np.ndarray
    step: float
    occupancy: np.ndarray
    diagnostics: dict


def consistency(views: list[View]) -> list[str]:
    """Flag changes in scale-normalized extent and foreground color."""
    heights, colors = [], []
    for view in views:
        view.validate()
        heights.append((np.ptp(np.where(view.mask.any(axis=1))[0]) + 1) / view.camera.pixels_per_unit)
        colors.append(np.median(view.rgb[view.mask], axis=0))
    typical_height = np.median(heights)
    typical_color = np.median(colors, axis=0)
    warnings = []
    for view, height, color in zip(views, heights, colors):
        if abs(view.camera.elevation) < 8 and abs(height / typical_height - 1) > 0.25:
            warnings.append(f"{view.name}: projected height differs substantially. Check scale and object identity.")
        if np.linalg.norm(color - typical_color) > 110:
            warnings.append(f"{view.name}: foreground color differs substantially. Check lighting and identity.")
    return warnings


def carve(views: list[View], resolution: int = 48, extent: float = 1.6, prior=None, polyhedral=None, *, weighted=False) -> Hull:
    """Keep voxels supported by silhouettes and discard unsupported components."""
    if not isinstance(resolution, int) or not 16 <= resolution <= 128:
        raise ValueError("Voxel resolution must be an integer between 16 and 128.")
    if not np.isfinite(extent) or not 0.1 <= extent <= 10:
        raise ValueError("Volume side length must be between 0.1 and 10 scene units.")
    if not 3 <= len(views) <= 8:
        raise ValueError("Provide 3–8 usable views with explicit camera parameters.")
    for view in views:
        view.validate()
    headings = np.array([view.camera.azimuth % 180 for view in views])
    separations = np.abs(headings[:, None] - headings)
    if np.max(np.minimum(separations, 180 - separations)) < 25:
        raise ValueError("Camera directions are too similar. Add a substantially different side view.")

    step = extent / resolution
    axis = (np.arange(resolution) + 0.5) * step - extent / 2
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    tested = np.zeros(len(points), np.uint8)
    occupied = np.ones(len(points), bool)
    if polyhedral is not None:
        rot, poly_center, half_ext = polyhedral
        occupied &= (np.abs((points - poly_center) @ rot) <= half_ext[None, :] + 0.5 * step).all(axis=1)
    ortho_only = not any(view.camera.distance is not None for view in views)
    robust_perspective = all(view.inferred for view in views[1:]) and not views[0].inferred and not ortho_only
    misses = np.zeros(len(points), np.uint8)
    support = np.zeros(len(points), np.float32) if weighted else None
    visible_weight = np.zeros(len(points), np.float32) if weighted else None
    processed_masks = []
    stats = []
    for view_idx, view in enumerate(views):
        ids = np.flatnonzero(occupied)
        _, _, _, x, y, visible = view.camera.pixels(points[ids], view.mask.shape)
        checked = ids[visible]
        tested[checked] += 1
        mask = view.mask
        if robust_perspective and view.inferred:
            mask = ndimage.binary_dilation(mask, iterations=2)
        elif view.inferred:
            # Close 1px foreground pinholes on inferred orthographic masks without outward dilation.
            mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
        processed_masks.append(mask)
        rejected = checked[~mask[y[visible], x[visible]]]
        tolerant_side = (robust_perspective and view.inferred) or (
            polyhedral is not None and view_idx >= 2 and view.inferred
        )
        if weighted and view.inferred:
            weight = view.texture_weight
            visible_weight[checked] += weight
            support[checked] += weight * mask[y[visible], x[visible]]
        elif tolerant_side:
            misses[rejected] += 1
        else:
            occupied[rejected] = False
        stats.append({"view": view.name, "tested": len(checked),
                      "carved": 0 if tolerant_side or (weighted and view.inferred) else len(rejected),
                      "background_votes": len(rejected),
                      "outside_frame": int((~visible).sum())})
    unknown = int(np.count_nonzero(occupied & (tested < 2)))
    if weighted:
        occupied &= (tested >= 3) & (visible_weight > 0) & (support >= 0.75 * visible_weight)
    elif robust_perspective:
        occupied &= (tested >= 2) & (misses <= np.maximum(0, (tested.astype(int) - 1) // 4))
    elif polyhedral is not None:
        occupied &= (tested >= 2) & (misses <= 1)
    else:
        occupied &= (tested >= 2)
    if prior is not None:
        heights, radii = prior
        radial = np.hypot(points[:, 0], points[:, 2])
        envelope = np.interp(points[:, 1], heights, radii, left=-1, right=-1)
        occupied &= radial <= 1.10 * envelope + step
    volume = occupied.reshape((resolution,) * 3)
    labels, count = ndimage.label(volume)
    if not count:
        raise ValueError("Silhouettes do not overlap in the volume. Verify camera angles, scale and centers.")
    sizes = np.bincount(labels.ravel())
    volume = labels == (1 + sizes[1:].argmax())
    if volume.sum() < 8:
        raise ValueError("Only a tiny intersection remains. Check the silhouettes and camera settings.")
    touches = any(np.take(volume, index, axis=dim).any() for dim in range(3) for index in (0, -1))
    pad_volume = np.pad(volume, 1)
    if ortho_only and not weighted:
        struct = np.ones((3, 3, 3), dtype=bool)
        band_mask = ndimage.binary_dilation(pad_volume, structure=struct) & ~ndimage.binary_erosion(
            pad_volume, structure=struct)
        other_components = np.pad((labels > 0) & ~volume, 1)
        blocked_island = ndimage.binary_dilation(other_components, structure=struct) & ~pad_volume
        band_mask[0, :, :] = band_mask[-1, :, :] = False
        band_mask[:, 0, :] = band_mask[:, -1, :] = False
        band_mask[:, :, 0] = band_mask[:, :, -1] = False
        bi, bj, bk = np.nonzero(band_mask)
        band_pts = (np.column_stack((bi, bj, bk)) - 1 + 0.5) * step - extent / 2
        if polyhedral is not None:
            rot, poly_center, half_ext = polyhedral
            d_poly = np.abs((band_pts - poly_center) @ rot) - half_ext[None, :]
            min_sdf = -(
                np.linalg.norm(np.maximum(d_poly, 0.0), axis=1) + np.minimum(np.max(d_poly, axis=1), 0.0)
            ).astype(np.float32) / step
            tested_band = np.full(len(band_pts), 2, dtype=np.uint8)
            sdf_views = list(zip(views[:2], processed_masks[:2]))
        else:
            min_sdf = np.full(len(band_pts), np.inf, dtype=np.float32)
            tested_band = np.zeros(len(band_pts), dtype=np.uint8)
            sdf_views = list(zip(views, processed_masks))
        for view, pmask in sdf_views:
            sdf_px = np.where(pmask, ndimage.distance_transform_edt(pmask) - 0.5,
                              -(ndimage.distance_transform_edt(~pmask) - 0.5))
            sdf_px = ndimage.gaussian_filter(sdf_px, sigma=0.85)
            sdf_vox = (sdf_px / max(1e-6, view.camera.pixels_per_unit * step)).astype(np.float32)
            px, py, _, _, _, valid = view.camera.pixels(band_pts, pmask.shape)
            tested_band[valid] += 1
            sampled = np.where(
                valid,
                ndimage.map_coordinates(sdf_vox, [py, px], order=1, mode="nearest"),
                -1.5,
            )
            min_sdf = np.minimum(min_sdf, sampled)
        min_sdf[tested_band < 2] = -1.5
        field = np.where(pad_volume, 1.5, -1.0).astype(np.float32)
        field[bi, bj, bk] = np.clip(0.5 + min_sdf, -1.0, 2.0)
        field[blocked_island] = -1.0
        field = np.where(np.abs(field - 0.5) < 1e-5, 0.5 + 1e-4, field)
        if np.count_nonzero(field > 0.5) < 8:
            field = pad_volume.astype(np.float32)
    else:
        # Break half-level ties between diagonally touching noisy voxels.
        field = ndimage.gaussian_filter(pad_volume.astype(np.float32), sigma=0.55)
    vertices, faces, _, _ = marching_cubes(field, level=0.5,
                                           spacing=(step,) * 3, allow_degenerate=False)
    vertices += axis[0] - step
    raw_vertices = vertices.copy()
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, incidence = np.unique(edges, axis=0, return_counts=True)
    if np.any(incidence != 2):
        raise ValueError(f"The voxel surface has {np.count_nonzero(incidence != 2)} ambiguous edges. "
                         "Adjust the masks or voxel resolution.")
    # Normal-similarity bilateral Taubin passes regularize planar faces without rounding sharp corners.
    p_raw = vertices[faces]
    raw_cross = np.cross(p_raw[:, 1] - p_raw[:, 0], p_raw[:, 2] - p_raw[:, 0])
    vn = np.zeros_like(vertices)
    for corner in faces.T:
        np.add.at(vn, corner, raw_cross)
    vn /= np.maximum(1e-12, np.linalg.norm(vn, axis=1, keepdims=True))
    a, b = edges.T
    dot_ab = np.sum(vn[a] * vn[b], axis=1)
    weights = np.maximum(1e-3, np.exp(-np.maximum(0.0, 1.0 - dot_ab) / 0.25))
    graph = coo_matrix((np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])),
                       shape=(len(vertices), len(vertices))).tocsr()
    degree = np.maximum(1e-12, np.asarray(graph.sum(axis=1)))
    passes = (0.30, -0.31, 0.30, -0.31) if ortho_only else (0.33, -0.34, 0.33, -0.34)
    for factor in passes:
        vertices += factor * ((graph @ vertices) / degree - vertices)
    smoothing_rms = float(np.sqrt(np.mean(np.sum((vertices - raw_vertices) ** 2, axis=1))))
    if prior is not None:
        heights, radii = prior
        model_y = np.linspace(heights[0], heights[-1], resolution)
        model_r = np.interp(model_y, heights, radii)
        sectors = max(64, resolution)
        shell = generate(model_y, model_r, sectors)
        # Take broad radial and axial cap evidence from the carved volume while preserving shape.
        azimuth = shell.angles[None, :]
        yy = np.broadcast_to(model_y[:, None], (resolution, sectors))
        sampled = model_r[:, None] * np.linspace(.62, 1.38, 21)[None, :]
        x = sampled[:, None, :] * np.cos(azimuth)[:, :, None]
        z = sampled[:, None, :] * np.sin(azimuth)[:, :, None]
        coords = [(coordinate - axis[0]) / step for coordinate in
                  np.broadcast_arrays(x, yy[:, :, None], z)]
        valid = ndimage.map_coordinates(volume.astype(np.float32), coords, order=1) >= .5
        observed_radius = np.max(np.where(valid, sampled[:, None, :], 0), axis=2)
        deviation = np.divide(observed_radius, model_r[:, None]) - 1
        deviation[~valid.any(axis=2)] = 0
        deviation = ndimage.gaussian_filter(np.clip(deviation, -.22, .22), (2, 2), mode=("nearest", "wrap"))
        influence = np.sin(azimuth) ** 2 * np.clip(model_r[:, None] / model_r.max() * 3, 0, 1)
        rings = shell.vertices[:resolution * sectors].reshape(resolution, sectors, 3)
        rings[:, :, [0, 2]] *= (1 + .55 * deviation * influence)[:, :, None]
        mid_idx = np.flatnonzero(np.abs(axis) <= 1.5 * step)
        if not len(mid_idx):
            mid_idx = np.array([int(np.argmin(np.abs(axis)))])
        col_occ = volume[np.ix_(mid_idx, np.arange(resolution), mid_idx)].any(axis=(0, 2))
        if col_occ.any():
            occ_ys = axis[col_occ]
            span = float(heights[-1] - heights[0])
            carved_top = float(occ_ys.max()) + 0.5 * step
            carved_bot = float(occ_ys.min()) - 0.5 * step
            if carved_top < heights[-1] - 0.5 * step:
                shell.vertices[-1, 1] += float(np.clip(carved_top - heights[-1], -0.14 * span, 0.0))
            if carved_bot > heights[0] + 0.5 * step:
                shell.vertices[-2, 1] += float(np.clip(carved_bot - heights[0], 0.0, 0.12 * span))
        vertices, faces = shell.vertices, shell.faces
    p = vertices[faces]
    if np.einsum("ij,ij->", p[:, 0], np.cross(p[:, 1], p[:, 2])) < 0:
        faces = faces[:, ::-1].copy()
        p = vertices[faces]
    normals = np.zeros_like(vertices)
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) <= 1e-12):
        raise ValueError("Surface smoothing produced a degenerate triangle.")
    for corner in faces.T:
        np.add.at(normals, corner, cross)
    normals /= np.maximum(1e-12, np.linalg.norm(normals, axis=1, keepdims=True))
    return Hull(vertices, faces, normals, points[volume.ravel()], step, volume, {
        "occupied_voxels": int(volume.sum()), "unresolved_voxels": unknown,
        "voxel_components_before_cleanup": count, "volume_boundary_touched": bool(touches),
        "carving": stats,
        "unresolved_ratio": unknown / len(points),
        "smoothing_rms_displacement": smoothing_rms,
    })

