from .lama import LamaWrapper
from .garnet import GarnetWrapper
from .pad import PadInpaintor


INPAINTOR_FACTORY = {
    "lama": LamaWrapper,
    "padding": PadInpaintor,
    "garnet": GarnetWrapper,
}


def build_inpaintor(inpaint_model, model_params):
    return INPAINTOR_FACTORY[inpaint_model](**model_params)
