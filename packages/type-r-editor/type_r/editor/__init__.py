from .anytext import AnyTextWrapper
from .fluxfill import FluxFillWrapper
from .mostel import MostelWrapper
from .skia import SkiaEditor
from .textctrl import TextCtrlWrapper
from .udifftext import UdiffTextWrapper

TEXTINPAINTOR_FACTORY = {
    "anytext": AnyTextWrapper,
    "udifftext": UdiffTextWrapper,
    "textctrl": TextCtrlWrapper,
    "mostel": MostelWrapper,
    "fluxfill": FluxFillWrapper,
    "skia": SkiaEditor,
}


def build_textinpaintor(textinpaint_model, model_params):
    return TEXTINPAINTOR_FACTORY[textinpaint_model](**model_params)
