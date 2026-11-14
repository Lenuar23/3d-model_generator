"""Write a textured GLB, a portable OBJ bundle and an untextured preview."""
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial, SimpleMaterial

from .texture_mapper import Texture


@dataclass
class Exports:
    glb: Path
    obj: Path
    bundle: Path
    geometry: Path


def export_model(texture: Texture, directory: Path) -> Exports:
    """Export a shared triangulation with explicit normals and texture coordinates."""
    texture.validate()
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(texture.pixels)
    material = PBRMaterial(
        name="Photo2Mesh", baseColorTexture=image, metallicFactor=0, roughnessFactor=0.9,
        doubleSided=True,
    )
    mesh = trimesh.Trimesh(
        vertices=texture.vertices, faces=texture.faces, vertex_normals=texture.normals,
        visual=trimesh.visual.texture.TextureVisuals(uv=texture.uv, material=material),
        process=False,
    )
    glb = directory / "photo2mesh.glb"
    glb_bytes = trimesh.exchange.gltf.export_glb(mesh.scene(), include_normals=True)
    obj_material = SimpleMaterial(image=image, name="Photo2Mesh", diffuse=[255, 255, 255, 255])
    mesh.visual.material = obj_material
    obj_text, assets = trimesh.exchange.obj.export_obj(
        mesh, include_normals=True, return_texture=True, mtl_name="photo2mesh.mtl"
    )
    material_data = assets.get("photo2mesh.mtl", b"").decode("utf-8")
    texture_names = [line.split(maxsplit=1)[1] for line in material_data.splitlines()
                     if line.startswith("map_Kd ")]
    if len(texture_names) != 1 or not assets.get(texture_names[0]):
        raise ValueError("The OBJ exporter did not provide the referenced texture image.")
    glb.write_bytes(glb_bytes)
    obj = directory / "photo2mesh.obj"
    obj_bytes = obj_text.encode("utf-8")
    obj.write_bytes(obj_bytes)
    for name, content in assets.items():
        (directory / name).write_bytes(content)
    bundle = directory / "photo2mesh-obj.zip"
    with ZipFile(bundle, "w") as archive:
        for name, data in [(obj.name, obj_bytes), *sorted(assets.items())]:
            archive.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED, compresslevel=1)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=[150, 173, 193, 255])
    geometry = directory / "geometry.glb"
    geometry.write_bytes(trimesh.exchange.gltf.export_glb(mesh.scene(), include_normals=True))
    return Exports(glb, obj, bundle, geometry)
