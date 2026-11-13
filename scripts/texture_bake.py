"""CPU UV baking with triangle visibility and a generated triplanar material."""
import numpy as np
from PIL import Image


def triangle_pixels(points, width, height):
    low = np.maximum(np.ceil(points.min(axis=0)).astype(int), 0)
    high = np.minimum(np.floor(points.max(axis=0)).astype(int), [width - 1, height - 1])
    if np.any(high < low):
        return None
    a, b, c = points
    denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
    if abs(denominator) < 1e-10:
        return None
    y, x = np.mgrid[low[1]:high[1]+1, low[0]:high[0]+1]
    w0 = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / denominator
    w1 = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / denominator
    weights = np.stack([w0, w1, 1-w0-w1], axis=-1)
    inside = np.all(weights >= -1e-6, axis=-1)
    return y[inside], x[inside], weights[inside]


def linear(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)


def bake_texture(mesh, photo, material, *, size=1024):
    import trimesh
    import xatlas
    from scipy.ndimage import distance_transform_edt, map_coordinates
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Expected a nonempty finite triangle mesh")
    rgba = np.asarray(photo.convert("RGBA")) / 255.
    mask = rgba[..., 3] >= .5
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Photo has no opaque foreground")
    vertices = np.asarray(mesh.vertices)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    span = np.maximum(upper-lower, 1e-8)
    scale = min((np.ptp(cols)+1)/span[0], (np.ptp(rows)+1)/span[1])
    center = np.array([(cols.min()+cols.max())/2, (rows.min()+rows.max())/2])
    def project(points):
        return (points[:, :2]-(lower[:2]+upper[:2])/2)*[scale, -scale]+center
    projected = project(vertices)
    h, w = mask.shape
    depth = np.full((h, w), -np.inf)
    for face in mesh.faces:
        pixels = triangle_pixels(projected[face], w, h)
        if pixels is not None:
            y, x, weights = pixels
            depth[y, x] = np.maximum(depth[y, x], weights @ vertices[face, 2])
    print("Photo visibility rasterized", flush=True)
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), np.asarray(mesh.faces, dtype=np.uint32))
    options = xatlas.PackOptions()
    options.resolution = size
    options.padding = 4
    atlas.generate(pack_options=options)
    mapping, faces, uv = atlas[0]
    points = vertices[mapping]
    normals = mesh.vertex_normals[mapping]
    uv_pixels = uv * [size-1, -(size-1)] + [0, size-1]
    result = np.zeros((size, size, 3))
    filled = np.zeros((size, size), dtype=bool)
    tile = linear(np.asarray(material.convert("RGB"))/255.)
    # Mirrored repetition makes opposite tile boundaries meet continuously.
    def sample_tile(coords):
        coords = 1-np.abs(np.mod(coords, 2)-1)
        return np.stack([map_coordinates(tile[..., i], [coords[:, 1]*(tile.shape[0]-1),
                         coords[:, 0]*(tile.shape[1]-1)], order=1, mode="nearest") for i in range(3)], axis=1)
    photo_rgb = linear(rgba[..., :3])
    # Match generated material's average color to observed foreground.
    color_scale = np.clip(np.median(photo_rgb[mask], axis=0) /
                          np.maximum(np.median(tile.reshape(-1, 3), axis=0), 1e-6), .25, 4)
    for face in faces:
        pixels = triangle_pixels(uv_pixels[face], size, size)
        if pixels is None:
            continue
        y, x, bary = pixels
        if not len(x):
            continue
        pos = bary @ points[face]
        normal = bary @ normals[face]
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        coords = (pos-lower)/span * 2
        weights = np.abs(normal)**4
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        generated = (sample_tile(coords[:, [1, 2]])*weights[:, 0:1] +
                     sample_tile(coords[:, [0, 2]])*weights[:, 1:2] +
                     sample_tile(coords[:, [0, 1]])*weights[:, 2:3]) * color_scale
        screen = project(pos)
        px = np.clip(np.rint(screen[:, 0]).astype(int), 0, w-1)
        py = np.clip(np.rint(screen[:, 1]).astype(int), 0, h-1)
        visible = (pos[:, 2] >= depth[py, px] - max(span)*.003) & mask[py, px]
        visible &= (screen[:, 0] >= 0) & (screen[:, 0] < w) & (screen[:, 1] >= 0) & (screen[:, 1] < h)
        blend = np.clip((normal[:, 2]-.15)/.6, 0, 1)
        blend = (blend*blend*(3-2*blend)*visible)[:, None]
        observed = np.stack([map_coordinates(photo_rgb[..., i], [screen[:, 1], screen[:, 0]],
                            order=1, mode="nearest") for i in range(3)], axis=1)
        result[y, x] = observed*blend + generated*(1-blend)
        filled[y, x] = True
    if not filled.any():
        raise ValueError("UV rasterization produced no pixels")
    nearest = distance_transform_edt(~filled, return_distances=False, return_indices=True)
    result[~filled] = result[nearest[0][~filled], nearest[1][~filled]]
    srgb = np.where(result <= .0031308, result*12.92, 1.055*np.maximum(result, 0)**(1/2.4)-.055)
    image = Image.fromarray(np.rint(np.clip(srgb, 0, 1)*255).astype(np.uint8))
    visual = trimesh.visual.texture.TextureVisuals(uv=uv,
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0., roughnessFactor=.65))
    output = trimesh.Trimesh(vertices=points, faces=faces, vertex_normals=normals, visual=visual, process=False)
    return output, image

