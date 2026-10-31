"""Classic marching cubes on binary cell-center samples and mesh processing.

The standard 256-case table has ambiguous saddle configurations; this is
not a topology-guaranteed Lewiner/MC33 implementation. Occupancy is padded
with empty samples to close surfaces touching the grid boundary.
"""

from numbers import Integral

import numpy as np

from ..domain.models import Mesh, VoxelGrid
from ._marching_cubes_table import CASES

_CORNERS = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
_EDGES = np.array(((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                   (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)))
_MIDPOINTS_X2 = _CORNERS[_EDGES].sum(axis=1)


def _empty_mesh() -> Mesh:
    return Mesh(np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.intp))


def marching_cubes(grid: VoxelGrid) -> Mesh:
    """Extract the 0.5 isosurface; return welded vertices in world coordinates.

    Inputs are boolean samples at voxel centers, not filled cubic blocks.
    Binary interpolation places intersections at edge midpoints. Integer
    doubled coordinates weld shared edges exactly without float tolerances.
    """
    if not grid.occupancy.any():
        return _empty_mesh()
    samples = np.pad(grid.occupancy, 1)
    shape = np.asarray(samples.shape) - 1
    cases = np.zeros(tuple(shape), dtype=np.uint8)
    for bit, offset in enumerate(_CORNERS):
        slices = tuple(slice(int(offset[axis]), int(offset[axis] + shape[axis])) for axis in range(3))
        cases |= samples[slices].astype(np.uint8) << bit
    active = np.argwhere((cases != 0) & (cases != 255))
    rows = CASES[cases[tuple(active.T)]]
    cube_indices, slots = np.nonzero(rows >= 0)
    doubled = active[cube_indices] * 2 + _MIDPOINTS_X2[rows[cube_indices, slots]]
    coordinates, inverse = np.unique(doubled, axis=0, return_inverse=True)
    vertices = (coordinates / 2 - 0.5) * np.asarray(grid.spacing) + np.asarray(grid.origin)
    # The classic table winds towards high values; occupancy=1 is inside.
    faces = inverse.reshape(-1, 3)[:, ::-1].copy()
    return Mesh(vertices, faces)


def _compact(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    if not len(faces):
        return _empty_mesh()
    used, remapped = np.unique(faces, return_inverse=True)
    return Mesh(vertices[used].copy(), remapped.reshape(-1, 3))


def clean_mesh(mesh: Mesh, *, min_component_faces: int = 0,
               min_component_ratio: float = 0.01) -> Mesh:
    """Remove degenerate/duplicate faces and small vertex-connected components.

    Keep components with at least max(min_component_faces, largest*ratio)
    triangles. Set both thresholds to zero to retain every valid component.
    """
    if (isinstance(min_component_faces, bool) or not isinstance(min_component_faces, Integral)
            or min_component_faces < 0):
        raise ValueError("min_component_faces must be a nonnegative integer")
    if not np.isfinite(min_component_ratio) or not 0 <= min_component_ratio <= 1:
        raise ValueError("min_component_ratio must be between zero and one")
    if not len(mesh.faces):
        return _empty_mesh()
    triangles = mesh.vertices[mesh.faces]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    faces = mesh.faces[np.any(area_vectors != 0, axis=1)]
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(unique)]
    if not len(faces):
        return _empty_mesh()
    parent = np.arange(len(mesh.vertices))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for a, b, c in faces:
        root = find(int(a))
        parent[find(int(b))] = root
        parent[find(int(c))] = root
    labels = np.array([find(int(a)) for a in faces[:, 0]])
    _, components, counts = np.unique(labels, return_inverse=True, return_counts=True)
    threshold = max(min_component_faces, counts.max() * min_component_ratio)
    return _compact(mesh.vertices, faces[counts[components] >= threshold])


def _edges(faces: np.ndarray) -> np.ndarray:
    return np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]])), axis=1), axis=0)


def smooth_mesh(mesh: Mesh, *, iterations: int = 3,
                lambda_factor: float = 0.5, mu_factor: float = -0.53) -> Mesh:
    """Apply uniform Taubin lambda/mu smoothing; topology stays unchanged.

    This reduces Laplacian shrinkage but does not guarantee volume preservation
    or prevent self-intersections. Boundary and isolated vertices are fixed.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not (0 < lambda_factor < 1 and -1 < mu_factor < -lambda_factor):
        raise ValueError("require 0 < lambda < 1 and -1 < mu < -lambda")
    vertices = mesh.vertices.copy()
    if not len(mesh.faces) or not iterations:
        return Mesh(vertices, mesh.faces.copy())
    edges = _edges(mesh.faces)
    directed = np.concatenate((edges, edges[:, ::-1]))
    degree = np.bincount(directed[:, 0], minlength=len(vertices))
    face_edges = np.sort(np.concatenate((mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]],
                                         mesh.faces[:, [2, 0]])), axis=1)
    all_edges, counts = np.unique(face_edges, axis=0, return_counts=True)
    movable = degree > 0
    movable[np.unique(all_edges[counts != 2])] = False
    for _ in range(iterations):
        for factor in (lambda_factor, mu_factor):
            sums = np.zeros_like(vertices)
            np.add.at(sums, directed[:, 0], vertices[directed[:, 1]])
            vertices[movable] += factor * (sums[movable] / degree[movable, None] - vertices[movable])
    return Mesh(vertices, mesh.faces.copy())


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """Return unit area-weighted vertex normals; isolated/degenerate normals are zero."""
    normals = np.zeros_like(mesh.vertices)
    triangles = mesh.vertices[mesh.faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, mesh.faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    np.divide(normals, lengths[:, None], out=normals, where=lengths[:, None] > 0)
    return normals


def build_mesh(grid: VoxelGrid, *, smoothing_iterations: int = 3,
               min_component_faces: int = 0, min_component_ratio: float = 0.01) -> Mesh:
    """Extract, remove small artifacts, and smooth a voxel reconstruction."""
    mesh = clean_mesh(marching_cubes(grid), min_component_faces=min_component_faces,
                      min_component_ratio=min_component_ratio)
    return smooth_mesh(mesh, iterations=smoothing_iterations)
