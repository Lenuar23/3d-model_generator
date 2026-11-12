"""Project photo colors onto the front of a Hunyuan mesh."""

import numpy as np


def apply_photo_colors(mesh, image, *, mirror_back=False):
    """Fit an orthographic projection to the foreground and color visible vertices."""
    rgba = np.asarray(image.convert("RGBA"))
    foreground = rgba[..., 3] >= 128
    rows, columns = np.nonzero(foreground)
    if not len(rows):
        raise ValueError("The photo has no opaque foreground for coloring.")

    # Hunyuan's front view looks along -Z in the exported mesh coordinates.
    vertices = mesh.vertices
    lower = vertices[:, :2].min(axis=0)
    upper = vertices[:, :2].max(axis=0)
    span = np.maximum(upper - lower, 1e-8)
    image_size = np.array([np.ptp(columns) + 1, np.ptp(rows) + 1])
    scale = np.min(image_size / span)
    center = np.array([(columns.min() + columns.max()) / 2,
                       (rows.min() + rows.max()) / 2])
    pixels = (vertices[:, :2] - (lower + upper) / 2) * [scale, -scale] + center
    x, y = np.rint(pixels).astype(int).T
    x = np.clip(x, 0, rgba.shape[1] - 1)
    y = np.clip(y, 0, rgba.shape[0] - 1)

    # A coarse depth buffer on projected vertices suppresses hidden layers.
    # It approximates visibility without rasterizing every triangle on CPU.
    bins = np.minimum((pixels - pixels.min(axis=0)).astype(int) // max(1, int(scale / 96)), 511)
    depth = np.full((512, 512), -np.inf)
    bx, by = bins.T
    np.maximum.at(depth, (by, bx), vertices[:, 2])
    visible = vertices[:, 2] >= depth[by, bx] - 0.02 * np.ptp(vertices[:, 2])

    rgb = rgba[..., :3] / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    base = np.median(linear[foreground], axis=0)
    weight = np.clip(mesh.vertex_normals[:, 2] * 2, 0, 1) * visible * foreground[y, x]
    if mirror_back:
        from scipy.ndimage import distance_transform_edt

        # Extend edge colors outside the photo silhouette before projecting
        # both sides. This also covers small differences between the outlines.
        nearest_y, nearest_x = distance_transform_edt(
            ~foreground, return_distances=False, return_indices=True,
        )
        y, x = nearest_y[y, x], nearest_x[y, x]
        rear_depth = np.full_like(depth, np.inf)
        np.minimum.at(rear_depth, (by, bx), vertices[:, 2])
        rear_visible = vertices[:, 2] <= rear_depth[by, bx] + 0.02 * np.ptp(vertices[:, 2])
        weight = (visible | rear_visible).astype(float)
    color = base + weight[:, None] * (linear[y, x] - base)
    mesh.visual.vertex_colors = np.column_stack([
        np.rint(color * 255).astype(np.uint8), np.full(len(vertices), 255, dtype=np.uint8),
    ])
