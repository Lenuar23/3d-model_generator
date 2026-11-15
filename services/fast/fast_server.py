"""Run Photo2Mesh's AI-Assisted workflow behind the generation API."""
import logging
from http.server import HTTPServer

from models import ensure_models
from photo2mesh.multiview import HullValidationError
from photo2mesh.wonder3d_adapter import generate_wonder, reconstruct_wonder
from server import Handler


class FastHandler(Handler):
    def generate(self, photo, model, options):
        ensure_models()
        folder, _ = generate_wonder(photo, folder=model.parent)
        attempts = [(resolution, ()) for resolution in (64, 80, 96, 48)]
        original_attempts = len(attempts)
        for index, (resolution, excluded) in enumerate(attempts):
            try:
                result, _, _, _ = reconstruct_wonder(
                    photo, resolution=resolution, reuse=folder, excluded=excluded,
                )
                logging.info("Photo2Mesh exported a validated hull at resolution %d, excluded=%s",
                             resolution, excluded)
                return result.exports.glb
            except HullValidationError:
                logging.warning("Photo2Mesh hull failed validation at resolution %d, excluded=%s",
                                resolution, excluded)
                if index == original_attempts - 1:
                    try:
                        result, _, _, _ = reconstruct_wonder(
                            photo, reuse=folder, weighted=True,
                        )
                        logging.info("Photo2Mesh exported a hull using weighted silhouette support")
                        return result.exports.glb
                    except ValueError as error:
                        logging.info("Weighted silhouette reconstruction failed: %s", error)
                    ranked = reconstruct_wonder(photo, reuse=folder, rank_only=True)
                    logging.info("Photo2Mesh view exclusion order: %s", ranked)
                    attempts.extend((64, (name,)) for name in ranked)
                if index == len(attempts) - 1:
                    raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    HTTPServer(("0.0.0.0", 8080), FastHandler).serve_forever()
