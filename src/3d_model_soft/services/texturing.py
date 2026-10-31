"""Visibility-aware vertex colors from calibrated RGB views, using NumPy."""

from collections.abc import Sequence

import numpy as np

from ..domain.models import Camera, Mesh, View
from .camera_math import extrinsics, project
from .meshing import vertex_normals


def _clip_near(triangle: np.ndarray, near: float) -> list[np.ndarray]:
    """Clip a camera-space triangle to z >= near, returning a polygon."""
    polygon = []
    for previous, current in zip(np.roll(triangle, 1, axis=0), triangle):
        previous_in, current_in = previous[2] >= near, current[2] >= near
        if previous_in != current_in:
            fraction = (near - previous[2]) / (current[2] - previous[2])
            polygon.append(previous + fraction * (current - previous))
        if current_in:
            polygon.append(current)
    return polygon


def _sample_depths(mesh: Mesh, camera: Camera, pixels: np.ndarray) -> np.ndarray:
    """Rasterize triangle depths at continuous query positions, not pixel centers.

    This sparse depth buffer samples the exact projected vertex coordinates,
    avoiding missed coverage at silhouettes and subpixel-sized triangles.
    Reciprocal depth interpolation is perspective correct. Both sides of
    triangles occlude. Only depth > 0 is considered.
    """
    rotation, translation = extrinsics(camera)
    points = mesh.vertices @ rotation.T + translation
    nearest = np.full(len(pixels), np.inf)
    if not len(pixels):
        return nearest
    focal = camera.height / (2 * np.tan(np.radians(camera.fov_deg) / 2))
    center = np.array([(camera.width - 1) / 2, (camera.height - 1) / 2])
    # Keep a strictly positive clipping plane scaled to scene depth.
    near = max(float(np.max(np.abs(points[:, 2]), initial=0)) * 1e-12, np.finfo(float).tiny)
    for face in mesh.faces:
        triangle = points[face]
        if (triangle[:, 2] < near).all():
            continue
        polygon = list(triangle) if (triangle[:, 2] >= near).all() else _clip_near(triangle, near)
        for index in range(1, len(polygon) - 1):
            tri = np.array([polygon[0], polygon[index], polygon[index + 1]])
            uv = tri[:, :2] / tri[:, 2, None] * focal + center
            lower, upper = uv.min(axis=0) - 1e-8, uv.max(axis=0) + 1e-8
            candidates = np.flatnonzero(((pixels >= lower) & (pixels <= upper)).all(axis=1))
            if not len(candidates):
                continue
            a, b, c = uv
            ab, ac = b - a, c - a
            denominator = ab[0] * ac[1] - ab[1] * ac[0]
            if abs(denominator) < 1e-14:
                continue
            relative = pixels[candidates] - a
            beta = (relative[:, 0] * ac[1] - relative[:, 1] * ac[0]) / denominator
            gamma = (ab[0] * relative[:, 1] - ab[1] * relative[:, 0]) / denominator
            alpha = 1 - beta - gamma
            inside = (alpha >= -1e-8) & (beta >= -1e-8) & (gamma >= -1e-8)
            reciprocal = alpha / tri[0, 2] + beta / tri[1, 2] + gamma / tri[2, 2]
            valid = inside & (reciprocal > 0)
            targets = candidates[valid]
            nearest[targets] = np.minimum(nearest[targets], 1 / reciprocal[valid])
    return nearest


def texture_mesh(
    mesh: Mesh, views: Sequence[View], *, fallback_color: tuple[int, int, int] = (128, 128, 128),
    depth_tolerance: float = 1e-5,
) -> Mesh:
    """Blend visible front-facing RGB observations into uint8 sRGB vertex colors.

    Colors use nearest-pixel sampling, mask filtering and cosine angle weights.
    Blending happens in linear RGB. The relative depth tolerance is multiplied
    by each vertex's camera depth. Unobserved vertices receive fallback_color.
    Input geometry and colors are copied. This produces vertex colors, not UVs.
    """
    if not views or any(not isinstance(view, View) or view.image is None for view in views):
        raise ValueError("at least one View with an RGB image is required; every view needs an image")
    fallback = np.asarray(fallback_color)
    if (fallback.shape != (3,) or fallback.dtype.kind not in "iu"
            or (fallback < 0).any() or (fallback > 255).any()):
        raise ValueError("fallback_color must contain three integers in [0, 255]")
    if not np.isfinite(depth_tolerance) or depth_tolerance < 0:
        raise ValueError("depth_tolerance must be finite and nonnegative")
    normals = vertex_normals(mesh)
    accumulated = np.zeros((len(mesh.vertices), 3), dtype=float)
    weights = np.zeros(len(mesh.vertices), dtype=float)
    for view in views:
        rotation, translation = extrinsics(view.camera)
        camera_position = -rotation.T @ translation
        towards_camera = camera_position - mesh.vertices
        lengths = np.linalg.norm(towards_camera, axis=1)
        directions = np.zeros_like(towards_camera)
        np.divide(towards_camera, lengths[:, None], out=directions, where=lengths[:, None] > 0)
        cosine = np.einsum("ij,ij->i", normals, directions)
        pixels = project(mesh.vertices, view.camera)
        valid = (np.isfinite(pixels).all(axis=1) & (cosine > 1e-8)
                 & (pixels[:, 0] >= -0.5) & (pixels[:, 0] < view.camera.width - 0.5)
                 & (pixels[:, 1] >= -0.5) & (pixels[:, 1] < view.camera.height - 0.5))
        indices = np.flatnonzero(valid)
        uv = np.floor(pixels[indices] + 0.5).astype(np.intp)
        foreground = view.mask[uv[:, 1], uv[:, 0]]
        indices, uv = indices[foreground], uv[foreground]
        if not len(indices):
            continue
        depths = (mesh.vertices[indices] @ rotation.T + translation)[:, 2]
        nearest = _sample_depths(mesh, view.camera, pixels[indices])
        visible = np.isfinite(nearest) & (depths <= nearest + depth_tolerance * depths)
        indices, uv = indices[visible], uv[visible]
        srgb = view.image[uv[:, 1], uv[:, 0]].astype(float) / 255
        linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055)**2.4)
        accumulated[indices] += linear * cosine[indices, None]
        weights[indices] += cosine[indices]
    colors = np.tile(fallback.astype(np.uint8), (len(mesh.vertices), 1))
    observed = weights > 0
    linear = accumulated[observed] / weights[observed, None]
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear**(1 / 2.4) - 0.055)
    colors[observed] = np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)
    return Mesh(mesh.vertices.copy(), mesh.faces.copy(), colors)
