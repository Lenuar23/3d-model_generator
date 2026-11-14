"""Project the source photograph into a cylindrical UV atlas."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

from .mesh_generator import Mesh
from .silhouette import Profile


@dataclass
class Texture:
    pixels: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    rear_row_jump: float = 0
    seam_difference: int = 0

    def validate(self) -> None:
        """Reject malformed atlas data before exporting it."""
        if (self.uv.shape != (len(self.vertices), 2) or not np.isfinite(self.uv).all()
            or np.any((self.uv < 0) | (self.uv > 1))):
            raise ValueError("UV coordinates must be finite pairs in [0, 1], one per vertex.")
        if (self.pixels.ndim != 3 or self.pixels.shape[2] != 3
            or not self.pixels.size or self.pixels.dtype != np.uint8):
            raise ValueError("The texture atlas must be a nonempty RGB byte image.")


def bilinear(rgb: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate RGB at fractional image coordinates."""
    x, y = np.broadcast_arrays(np.clip(x, 0, rgb.shape[1] - 1),
                               np.clip(y, 0, rgb.shape[0] - 1))
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, rgb.shape[1] - 1), np.minimum(y0 + 1, rgb.shape[0] - 1)
    wx = (x - x0).astype(np.float32)[..., None]
    wy = (y - y0).astype(np.float32)[..., None]
    return ((1 - wy) * ((1 - wx) * rgb[y0, x0] + wx * rgb[y0, x1])
            + wy * ((1 - wx) * rgb[y1, x0] + wx * rgb[y1, x1]))


def project(
    rgb: np.ndarray, mask: np.ndarray, profile: Profile, mesh: Mesh,
    mirrored: bool = False, size: int = 512,
) -> Texture:
    """Bake visible colors and approximate occluded colors using the same image."""
    if not isinstance(size, int) or not 64 <= size <= 1024:
        raise ValueError("The atlas size must be an integer between 64 and 1024 pixels.")
    if mask.dtype != bool or rgb.shape != (*mask.shape, 3) or not mask.any():
        raise ValueError("Texture projection requires RGB pixels and a matching foreground mask.")
    # Nearest foreground fill keeps background out of bilinear samples near edges.
    region = np.s_[profile.rows[0]:profile.rows[-1] + 1]
    nearest = distance_transform_edt(~mask[region], return_distances=False, return_indices=True)
    filled = rgb[region][nearest[0], nearest[1]]
    padding = 4
    side_end = size - max(12, size // 16) - 1
    theta = np.linspace(1.5 * np.pi, 3.5 * np.pi, size)
    rows = np.linspace(profile.rows[0], profile.rows[-1], side_end + 1)
    left = np.interp(rows, profile.rows, profile.left)
    right = np.interp(rows, profile.rows, profile.right)
    mid = (left + right) / 2
    fraction = np.cos(theta)[None, :]
    x = mid[:, None] + (right - left)[:, None] * 0.5 * fraction
    source_y = rows[:, None] - profile.rows[0]
    colors = bilinear(filled, x, source_y)
    if mirrored:
        # The same projected horizontal coordinate is reused across the visibility boundary.
        side = colors
    else:
        # Blend to robust per-row colors across the first part of the hidden hemisphere.
        average = np.median(bilinear(
            filled, mid[:, None] + (right - left)[:, None] * np.linspace(-0.4, 0.4, 17),
            source_y,
        ), axis=1)
        widths = np.maximum(1, right - left).astype(np.float32)
        sigma = max(1, len(rows) * 0.025)
        average = (gaussian_filter1d(average * widths[:, None], sigma, axis=0)
                   / gaussian_filter1d(widths, sigma)[:, None])
        blend = np.clip(-np.sin(theta) * 2, 0, 1).astype(np.float32)[None, :, None]
        blend = blend**2 * (3 - 2 * blend)
        side = colors * (1 - blend) + average[:, None, :] * blend
    side[:, -1] = side[:, 0]

    pixels = np.empty((size, size, 3), dtype=np.uint8)
    pixels[:side_end + 1] = np.rint(side).astype(np.uint8)
    pixels[side_end + 1:side_end + 1 + padding] = pixels[side_end]
    bottom_color = np.median(side[-1], axis=0)
    top_color = np.median(side[0], axis=0)
    pixels[side_end + 1 + padding:, :size // 2] = np.rint(bottom_color)
    pixels[side_end + 1 + padding:, size // 2:] = np.rint(top_color)

    # Include a seam duplicate per ring and separate the cap rims for hard normals.
    sectors = len(mesh.angles)
    sector = np.arange(sectors + 1)
    canonical = np.arange(mesh.ring_count)[:, None] * sectors + sector % sectors
    vertices = [mesh.vertices[canonical.ravel()]]
    normals = [mesh.normals[canonical.ravel()]]
    u = (0.5 + sector * (size - 1) / sectors) / size
    v = 1 - (0.5 + np.linspace(side_end, 0, mesh.ring_count)) / size
    uv = [np.stack(np.broadcast_arrays(u, v[:, None]), axis=-1).reshape(-1, 2)]
    a = np.arange(mesh.ring_count - 1)[:, None] * (sectors + 1) + sector[:-1]
    b = a + 1
    stride = sectors + 1
    faces = [np.stack((a, a + stride, b), axis=-1).reshape(-1, 3),
             np.stack((b, a + stride, b + stride), axis=-1).reshape(-1, 3)]
    offset = len(vertices[0])
    for index, top in enumerate(mesh.cap_ends):
        ring = (mesh.ring_count - 1 if top else 0) * sectors + sector[:-1]
        center_id = mesh.ring_count * sectors + index
        vertices.append(mesh.vertices[np.r_[ring, center_id]])
        normals.append(np.tile([0, 1 if top else -1, 0], (stride, 1)))
        center_u = 0.75 if top else 0.25
        cap_height = size - side_end - 1 - padding
        center_v = cap_height / (2 * size)
        rim_u = center_u + 0.2 * np.cos(mesh.angles)
        rim_v = center_v + (center_v - 1.5 / size) * np.sin(mesh.angles)
        uv.append(np.vstack((np.column_stack((rim_u, rim_v)), [center_u, center_v])))
        rim = offset + sector[:-1]
        following = offset + (sector[:-1] + 1) % sectors
        center = np.full(sectors, offset + sectors, dtype=int)
        faces.append(np.column_stack((center, following, rim) if top else (center, rim, following)))
        offset += stride
    rear = pixels[:side_end + 1, 0].astype(np.float32)
    seam = int(np.abs(rear - pixels[:side_end + 1, -1]).max())
    return Texture(pixels, np.vstack(vertices), np.vstack(faces), np.vstack(normals), np.vstack(uv),
                   float(np.max(np.abs(np.diff(rear, axis=0)))), seam)
