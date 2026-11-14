"""Construct and validate an indexed surface of revolution."""
from dataclasses import dataclass

import numpy as np
from typing import Protocol
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    angles: np.ndarray
    ring_count: int
    cap_ends: tuple[bool, ...]


def generate(
    heights: np.ndarray, radii: np.ndarray, segments: int = 96,
    close_bottom: bool = True, close_top: bool = True,
) -> Mesh:
    """Build outward-wound quad strips and optionally triangulate flat end disks."""
    heights = np.asarray(heights, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if (heights.ndim != 1 or len(heights) < 2 or radii.shape != heights.shape
        or not isinstance(segments, (int, np.integer)) or segments < 3
        or not np.isfinite(heights).all() or not np.isfinite(radii).all()
        or np.any(np.diff(heights) <= 0) or np.any(radii <= 0)):
        raise ValueError("Mesh needs increasing heights, positive radii and at least three sectors.")
    angles = 1.5 * np.pi + np.arange(segments) * (2 * np.pi / segments)
    cosine, sine = np.cos(angles), np.sin(angles)
    rings = np.stack(np.broadcast_arrays(
        radii[:, None] * cosine, heights[:, None], radii[:, None] * sine
    ), axis=-1)
    vertices = rings.reshape(-1, 3)
    slopes = np.gradient(radii, heights)
    normals = np.stack(np.broadcast_arrays(cosine, -slopes[:, None], sine), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals.reshape(-1, 3)
    sector = np.arange(segments)
    a = np.arange(len(heights) - 1)[:, None] * segments + sector
    b = np.arange(len(heights) - 1)[:, None] * segments + (sector + 1) % segments
    faces = [np.stack((a, a + segments, b), axis=-1).reshape(-1, 3),
             np.stack((b, a + segments, b + segments), axis=-1).reshape(-1, 3)]
    ends = []
    for top, enabled in enumerate((close_bottom, close_top)):
        if enabled:
            row = len(heights) - 1 if top else 0
            ring = row * segments + sector
            next_ring = row * segments + (sector + 1) % segments
            center = np.full(segments, len(vertices), dtype=int)
            triangle = (center, next_ring, ring) if top else (center, ring, next_ring)
            faces.append(np.column_stack(triangle))
            vertices = np.vstack((vertices, [0, heights[row], 0]))
            normals = np.vstack((normals, [0, 1 if top else -1, 0]))
            ends.append(bool(top))
    return Mesh(vertices, np.vstack(faces), normals, angles, len(heights), tuple(ends))


class IndexedSurface(Protocol):
    """Shared geometry arrays needed for diagnostics."""
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray


def validate(mesh: IndexedSurface) -> dict:
    """Check triangle area, edge incidence, orientation and normal lengths."""
    faces = mesh.faces
    vertex_count = len(mesh.vertices)
    indices_valid = (faces.ndim == 2 and faces.shape[1] == 3 and faces.size > 0
                     and np.issubdtype(faces.dtype, np.integer)
                     and faces.min() >= 0 and faces.max() < vertex_count)
    if not indices_valid:
        raise ValueError("The mesh has out-of-range triangle indices.")
    points = mesh.vertices[faces]
    area_vectors = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
    low, high = np.sort(edges, axis=1).T
    _, inverse, counts = np.unique(low.astype(np.int64) * vertex_count + high,
                                   return_inverse=True, return_counts=True)
    directions = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    areas = np.linalg.norm(area_vectors, axis=1) * 0.5
    used = np.unique(faces)
    graph = coo_matrix((np.ones(len(low), dtype=bool), (low, high)),
                       shape=(vertex_count, vertex_count))
    isolated = vertex_count - len(used)
    components = connected_components(graph, directed=False, return_labels=False) - isolated
    ordered_faces = np.sort(faces.astype(np.int64), axis=1)
    # Lexicographic grouping avoids integer overflow for standalone meshes.
    duplicate_faces = len(faces) - len(np.unique(ordered_faces, axis=0))
    agreements = np.einsum("ij,ij->i", area_vectors, mesh.normals[faces].sum(axis=1))
    return {
        "valid_indices": bool(indices_valid),
        "finite_vertices": bool(np.isfinite(mesh.vertices).all()),
        "degenerate_triangles": int(np.count_nonzero(areas == 0)),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "inconsistent_edges": int(np.count_nonzero((counts == 2) & (directions != 0))),
        "unit_normals": bool(np.allclose(np.linalg.norm(mesh.normals, axis=1), 1)),
        "duplicate_faces": duplicate_faces,
        "isolated_vertices": isolated,
        "connected_components": components,
        "opposed_face_normals": int(np.count_nonzero(agreements < 0)),
        "triangle_area_min_median_max": [float(areas.min()), float(np.median(areas)), float(areas.max())],
        "bounding_box_xyz": np.ptp(mesh.vertices, axis=0).tolist(),
        "watertight": boundary == nonmanifold == duplicate_faces == 0,
    }
