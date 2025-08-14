from type_r.util.loader import load_class_lazily

from .base import BaseTextEditor


def build_textinpaintor(textinpaint_model: str, model_params: dict) -> BaseTextEditor:
    """Build a text inpainting model."""
    MODELS = {
        "anytext": "type_r.editor.anytext.AnyTextWrapper",
        "udifftext": "type_r.editor.udifftext.UdiffTextWrapper",
        "textctrl": "type_r.editor.textctrl.TextCtrlWrapper",
        "mostel": "type_r.editor.mostel.MostelWrapper",
        "fluxfill": "type_r.editor.fluxfill.FluxFillWrapper",
        "skia": "type_r.editor.skia.SkiaEditor",
    }
    return load_class_lazily(MODELS[textinpaint_model])(**model_params)
