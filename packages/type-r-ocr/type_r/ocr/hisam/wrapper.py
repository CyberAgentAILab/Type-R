import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from ..base import BaseOCRDet, OCRDetOut
from .modeling.auto_mask_generator import AutoMaskGenerator
from .modeling.build import model_registry


def unclip(p, unclip_ratio=2.0):
    try:
        poly = Polygon(p)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset_polygon = offset.Execute(distance)
        expanded = np.array(offset_polygon)
    except Exception as e:
        print("error:", e)
        print("distance:", distance)
        print("offset_polygon:", offset_polygon)
        return np.array(offset_polygon[0])
    return expanded


class HiSAMOCRDet(BaseOCRDet):
    def __init__(self, weight_path, *args, **kwargs):
        hisam = model_registry["vit_h"](
            checkpoint=weight_path,
            hier_det=True,
            model_type="vit_h",
            prompt_len=12,
            attn_layers=1,
        )
        hisam.eval()
        hisam.to("cuda")
        self.model = AutoMaskGenerator(hisam, efficient_hisam=False)

    def inference(self, image: np.ndarray) -> OCRDetOut:
        img_h, img_w = image.shape[:2]

        self.model.set_image(image)
        masks, _, _ = self.model.predict(
            from_low_res=False,
            fg_points_num=1500,
            batch_points_num=100,
            score_thresh=0.5,
            nms_thresh=0.5,
        )  # only return word masks here
        if masks is None:
            polygons = []
        else:
            masks = (masks[:, 0, :, :]).astype(np.uint8)  # word masks, (n, h, w)
            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                for cont in contours:
                    epsilon = 0.002 * cv2.arcLength(cont, True)
                    approx = cv2.approxPolyDP(cont, epsilon, True)
                    points = approx.reshape((-1, 2))
                    if points.shape[0] < 4:
                        continue
                    pts = unclip(points)
                    if len(pts) != 1:
                        continue
                    pts = pts[0].astype(np.int32)
                    if Polygon(pts).area < 32:
                        continue
                    pts[:, 0] = np.clip(pts[:, 0], 0, img_w)
                    pts[:, 1] = np.clip(pts[:, 1], 0, img_h)
                    cnt_list = pts.tolist()
                    polygons.append(cnt_list)

        return OCRDetOut(
            polygons=polygons,
            modelname="hisam",
        )
