import numpy as np
from type_r.util.cv_func import draw_pos


def test_paddle(
    sample_image,
    sample_paddle_polygons,
    paddle,
):
    paddle_outs = paddle(sample_image)
    for i, (polygon, polygon_sample) in enumerate(
        zip(paddle_outs.polygons, sample_paddle_polygons)
    ):
        if i == 0:
            mask = draw_pos(np.array(polygon), 1.0) * 255.0
            mask_sample = draw_pos(np.array(polygon_sample), 1.0) * 255.0
        else:
            mask += draw_pos(np.array(polygon), 1.0) * 255.0
            mask_sample += draw_pos(np.array(polygon_sample), 1.0) * 255.0
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    mask_sample = np.clip(mask_sample, 0, 255).astype(np.uint8)
    assert np.allclose(mask, mask_sample, atol=1), "Paddle OCR result is not correct"
