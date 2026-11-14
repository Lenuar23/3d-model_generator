"""Create reproducible diagnostic images, plots and wireframes."""
import numpy as np
import cv2
import plotly.graph_objects as go

from .mesh_generator import IndexedSurface
from .segmentation import Segmentation
from .silhouette import Profile


def images(rgb: np.ndarray, region: Segmentation) -> dict[str, np.ndarray]:
    """Return input, binary mask, masked foreground and contour overlay."""
    overlay = rgb.copy()
    contours, _ = cv2.findContours(region.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 240, 170), 2)
    return {
        "original": rgb,
        "mask": region.mask.astype(np.uint8) * 255,
        "foreground": np.dstack((rgb, region.mask.astype(np.uint8) * 255)),
        "contour": overlay,
    }


def radius_plot(profile: Profile, radius_factor: float = 1) -> go.Figure:
    """Compare observed row widths to the normalized radii used for geometry."""
    height = profile.rows[-1] - profile.rows[0]
    figure = go.Figure([
        go.Scatter(x=profile.radii / height, y=0.5 - (profile.rows - profile.rows[0]) / height,
                   name="Raw", mode="lines", line={"color": "#8298a9"}),
        go.Scatter(x=profile.sample_radii * radius_factor, y=profile.heights,
                   name="Reconstructed", mode="lines", line={"color": "#02977d"}),
    ])
    figure.update_layout(xaxis_title="Radius / object height", yaxis_title="Normalized height",
                         margin={"l": 55, "r": 20, "t": 30, "b": 50})
    return figure


def wireframe(mesh: IndexedSurface) -> go.Figure:
    """Display all unique triangular edges in one WebGL trace."""
    faces = mesh.faces
    edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lines = np.full((len(edges), 3, 3), np.nan)
    lines[:, :2] = mesh.vertices[edges]
    x, y, z = lines.reshape(-1, 3).T
    figure = go.Figure(go.Scatter3d(
        x=x, y=z, z=y, mode="lines", line={"width": 2, "color": "#008b77"}, hoverinfo="skip",
    ))
    figure.update_layout(
        scene={"aspectmode": "data", "xaxis_title": "X", "yaxis_title": "Z", "zaxis_title": "Y"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=525,
    )
    return figure
