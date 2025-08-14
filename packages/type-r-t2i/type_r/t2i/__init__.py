from type_r.util.loader import load_class_lazily

from .base import BaseT2I


def build_t2i(model_name: str, model_params: dict) -> BaseT2I:
    MODELS = {
        "dalle3": "type_r.t2i.dalle3.DALLE3",
        "flux": "type_r.t2i.flux.Flux",
        "geminiimg": "type_r.t2i.geminiimg.GeminiImage",
        "gptimg": "type_r.t2i.gptimg.GPTImage",
        "sd3": "type_r.t2i.sd3.SD3",
        "sd3dot5": "type_r.t2i.sd3dot5.SD3dot5",
    }
    module = load_class_lazily(MODELS[model_name])
    return module.build_t2i(model_params)
