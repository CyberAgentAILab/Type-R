from type_r.util.loader import load_class_lazily

from .base import BaseTextEraser


def build_inpaintor(inpaint_model, model_params) -> BaseTextEraser:
    """Build an inpainting model."""
    MODELS = {
        "lama": "type_r.eraser.lama.LamaWrapper",
        "padding": "type_r.eraser.padding.PadInpaintor",
        "garnet": "type_r.eraser.garnet.GarnetWrapper",
    }
    return load_class_lazily(MODELS[inpaint_model])(**model_params)
