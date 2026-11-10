"""Work around Model3D's cancelled lazy import in pinned Gradio 6.31.0."""

from pathlib import Path


def repair_model3d():
    import gradio

    if gradio.__version__ != "6.31.0":
        return
    assets = Path(gradio.__file__).parent / "templates" / "frontend" / "assets"
    # File metadata updates clean up the effect while the module is loading.
    # The imported component is independent of that file and must still be
    # registered. Otherwise the signature guard skips retrying the import.
    original = "d||v(i,e.default,!0)"
    replacement = "v(i,e.default,!0)"
    for asset in assets.glob("Model3D-*.js"):
        source = asset.read_text(encoding="utf-8")
        if original in source:
            asset.write_text(source.replace(original, replacement, 1), encoding="utf-8")
