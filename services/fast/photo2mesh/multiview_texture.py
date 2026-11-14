"""Bake a triangle atlas from depth-tested camera image samples."""
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter

from .texture_mapper import Texture, bilinear
from .visual_hull import Hull, View
from .hull_quality import rasterize
from .texture_consensus import blend_inferred


def bake(hull: Hull, views: list[View], *, consensus: bool = False) -> tuple[Texture, dict, np.ndarray]:
    """Blend visible vertex samples and interpolate them into padded UV islands."""
    points, normals = hull.vertices, hull.normals
    colors = np.zeros_like(points)
    weights = np.zeros(len(points))
    inferred_weights = np.zeros(len(points))
    inferred_colors = np.zeros_like(points)
    inferred_samples, inferred_scores = [], []
    prepared = []
    observed_medians = [np.median(view.rgb[view.mask], axis=0) for view in views if not view.inferred]
    observed_color = np.median(observed_medians, axis=0) if observed_medians else None
    for view in views:
        if view.texture_weight <= 0:
            continue
        x, y, depth, px, py, in_frame = view.camera.pixels(points, view.mask.shape)
        h, w = view.mask.shape
        front = view.camera.facing(points, normals)
        active = np.flatnonzero((front > 0.13) & in_frame)
        if not len(active):
            continue
        # Conservative depth proxy from occupied voxel centers and a screen-space splat.
        buffer = rasterize(points, hull.faces, view)
        # Fill rasterization cracks without spreading foreground depth across boundaries.
        buffer = np.where(np.isfinite(buffer), buffer, maximum_filter(buffer, size=3))
        coords = py[active], px[active]
        active = active[view.mask[coords] & np.isfinite(buffer[coords])
                        & (depth[active] >= buffer[coords] - .65*hull.step)]
        if not len(active):
            continue
        distances = distance_transform_edt(view.mask)
        # EDT above measures inside distances. A separate nearest-foreground fill prevents halos.
        nearest = distance_transform_edt(~view.mask, return_distances=False, return_indices=True)
        rgb = view.rgb[nearest[0], nearest[1]]
        if view.inferred and observed_color is not None:
            offset = np.clip((observed_color - np.median(view.rgb[view.mask], axis=0))*.35,-22,22)
            rgb = np.clip(rgb.astype(np.float32)+offset,0,255)
        edge_weight = np.clip(distances[py[active], px[active]] / 5, 0.1, 1)
        weight = front[active]**4 * edge_weight * view.texture_weight
        sample = bilinear(rgb, x[active], y[active])
        prepared.append((view, buffer, distances, rgb))
        target, total = (inferred_colors, inferred_weights) if view.inferred else (colors, weights)
        target[active] += sample * weight[:, None]
        total[active] += weight
        if consensus and view.inferred:
            samples = np.zeros(points.shape, dtype=np.float32)
            scores = np.zeros(len(points), dtype=np.float32)
            samples[active], scores[active] = sample, weight
            inferred_samples.append(samples)
            inferred_scores.append(scores)
    observed = weights > 0
    inferred = ~observed & (inferred_weights > 0)
    colors[observed] /= weights[observed, None]
    colors[inferred] = inferred_colors[inferred] / inferred_weights[inferred, None]
    if inferred_samples:
        colors[inferred] = blend_inferred(np.array(inferred_samples)[:, inferred],
                                          np.array(inferred_scores)[:, inferred])
    known = observed | inferred
    colors[~known] = np.median(colors[known], axis=0) if known.any() else [145, 145, 145]
    faces = hull.faces
    n = len(faces)
    tile = 12 if len(faces) < 100000 else 8
    columns = int(np.ceil(np.sqrt(n)))
    size = columns * tile
    if size > 4096:
        raise ValueError("The surface is too complex for the texture atlas. Reduce voxel resolution.")
    atlas = np.zeros((size, size, 3), np.uint8)
    provenance = np.zeros((size, size, 3), np.uint8)
    uv = np.empty((n, 3, 2))
    yy, xx = np.mgrid[:tile, :tile]
    span = tile - 5
    u, v = np.clip((xx - 2) / span, 0, 1), np.clip((yy - 2) / span, 0, 1)
    total = np.maximum(1, u + v)
    bary = np.stack((1 - (u + v)/total, u/total, v/total), axis=-1)
    counts = np.zeros(3, dtype=int)
    observed_weight_sum = 0.0
    interior = (xx >= 2) & (yy >= 2) & (xx + yy <= span+4)
    for start in range(0, n, 512):
        stop = min(n, start + 512)
        ids = np.arange(start, stop)
        tx, ty = tile*(ids % columns), tile*(ids // columns)
        patches = np.einsum("yxi,fic->fyxc", bary, colors[faces[start:stop]])
        shape = patches.shape[:-1]
        xyz = np.einsum("yxi,fic->fyxc", bary, points[faces[start:stop]]).reshape(-1,3)
        normal = np.einsum("yxi,fic->fyxc", bary, normals[faces[start:stop]]).reshape(-1,3)
        normal /= np.maximum(1e-8,np.linalg.norm(normal,axis=1,keepdims=True))
        sampled = np.zeros((2,len(xyz),3))
        scores = np.zeros((2,len(xyz)))
        inferred_samples, inferred_scores = [], []
        for view, buffer, distances, rgb in prepared:
            x, y, z, px, py, visible = view.camera.pixels(xyz, view.mask.shape)
            facing = view.camera.facing(xyz, normal)
            ids = np.flatnonzero(visible & (facing > .13) & (distances[py,px] > 1.5)
                                 & (z >= buffer[py,px]-.65*hull.step))
            if not len(ids):
                continue
            channel = int(view.inferred)
            weight = facing[ids]**6 * np.clip(distances[py[ids],px[ids]]/6,0,1)*view.texture_weight
            color = bilinear(rgb, x[ids], y[ids])
            sampled[channel,ids] += color*weight[:,None]
            scores[channel,ids] += weight
            if consensus and view.inferred:
                samples = np.zeros(xyz.shape, dtype=np.float32)
                confidence = np.zeros(len(xyz), dtype=np.float32)
                samples[ids], confidence[ids] = color, weight
                inferred_samples.append(samples)
                inferred_scores.append(confidence)
        direct = scores[0] > 0
        synthetic = (scores[1] > 0) if consensus else (~direct & (scores[1] > 0))
        provenance_ids = np.full(len(xyz), 2)
        observed_mix = np.zeros(len(xyz))
        flat = patches.reshape(-1,3)
        for channel, mask in ((1, synthetic), (0, direct)):
            pixels = sampled[channel,mask] / scores[channel,mask,None]
            if channel == 1 and inferred_samples:
                pixels = blend_inferred(np.array(inferred_samples)[:, mask],
                                         np.array(inferred_scores)[:, mask])
            strength = np.clip(scores[channel,mask]*(24 if channel == 0 else 10),0,1)[:,None]
            if consensus and channel == 0:
                strength = np.clip(scores[channel,mask]*90,0,1)[:,None]
                strength = strength**2 * (3 - 2*strength)
            if channel == 0:
                observed_mix[mask] = strength[:,0]
            flat[mask] += strength*(pixels-flat[mask])
            if consensus and channel == 0:
                provenance_ids[np.flatnonzero(mask)[strength[:,0] >= .5]] = channel
            else:
                provenance_ids[mask] = channel
        provenance_ids = provenance_ids.reshape(shape)
        counts += np.bincount(provenance_ids[:,interior].ravel(), minlength=3)
        observed_weight_sum += observed_mix.reshape(shape)[:,interior].sum()
        atlas[ty[:, None, None] + yy, tx[:, None, None] + xx] = np.uint8(np.rint(patches))
        provenance[ty[:, None, None] + yy, tx[:, None, None] + xx] = (
            np.array([[44,171,127],[245,158,60],[143,146,151]])[provenance_ids]
        )
        uv[start:stop, :, 0] = (tx[:, None] + [2.5, 2.5+span, 2.5]) / size
        uv[start:stop, :, 1] = 1 - (ty[:, None] + [2.5, 2.5, 2.5+span]) / size
    indices = faces.ravel()
    return Texture(atlas, points[indices], np.arange(3*n).reshape(-1, 3),
                   normals[indices], uv.reshape(-1, 2)), {
        "observed_texture_vertices": int(observed.sum()),
        "inferred_texture_vertices": int(inferred.sum()),
        "unobserved_texture_vertices": int((~known).sum()),
        "atlas_size": size,
        "observed_texel_fraction": float(counts[0]/counts.sum()),
        "inferred_texel_fraction": float(counts[1]/counts.sum()),
        "fallback_texel_fraction": float(counts[2]/counts.sum()),
        "inferred_texture_consensus": consensus,
        "mean_direct_photo_blend_weight": float(observed_weight_sum/counts.sum()),
    }, provenance

