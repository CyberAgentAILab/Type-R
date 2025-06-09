import numpy as np
from type_r.eraser import build_inpaintor


def test_padinpaintor():
    inpaintor = build_inpaintor(
        "padding",
        {"dilate_kernel_size": 0, "dilate_iteration": 0, "polygon_dilation": False},
    )
    img = np.zeros((512, 512, 3))
    inpaintor(img, [[(0, 0), (30, 0), (30, 30), (0, 30)]], [])
