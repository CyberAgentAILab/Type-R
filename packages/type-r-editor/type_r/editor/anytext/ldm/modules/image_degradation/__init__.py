from .bsrgan import degradation_bsrgan_variant as degradation_fn_bsr
from .bsrgan_light import degradation_bsrgan_variant as degradation_fn_bsr_light

__all__ = ["degradation_fn_bsr", "degradation_fn_bsr_light"]