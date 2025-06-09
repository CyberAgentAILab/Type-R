import logging
from functools import lru_cache

import numpy as np
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import Polygons, SegmentationMask
from type_r.util.cv_func import draw_pos, min_bounding_rect

from ..base import BaseOCRDet, BaseOCRRecog, OCRDetOut
from .masktextspotterv3 import TextDemo


@lru_cache(maxsize=None)
def load_masktextspooterv3(weight_file):
    logging.getLogger("maskrcnn_benchmark.utils.checkpoint").setLevel(logging.CRITICAL)
    logging.getLogger("maskrcnn_benchmark.utils.model_serialization").setLevel(
        logging.CRITICAL
    )
    text_demo = TextDemo(
        weight_file, min_image_size=800, confidence_threshold=0.0, output_polygon=True
    )
    return text_demo


class MaskTextSpotterV3Det(BaseOCRDet):
    def __init__(self, weight_path, *args, **kwargs):
        self.model = load_masktextspooterv3(weight_path)

    def convert_polygon_format(self, polygons):
        # [[x0,y0,...], ...] -> [[[x0,x1,...], [y0,y1,...]], ...]
        return [
            np.reshape(polygon, (len(polygon) // 2, 2)).tolist() for polygon in polygons
        ]

    def polygons2proposalitems(self, polygons):
        proposal_items = []
        for polygon in polygons:
            polygon_xy = np.reshape(polygon, (len(polygon) // 2, 2))
            x_min, x_max = np.min(polygon_xy[:, 0]), np.max(polygon_xy[:, 0])
            y_min, y_max = np.min(polygon_xy[:, 1]), np.max(polygon_xy[:, 1])
            polygon_box = [x_min, y_min, x_max, y_max]  # xyxy mode
            proposal_items.append((polygon_box, polygon))
        return proposal_items

    def inference(self, image: np.ndarray) -> OCRDetOut:
        image = image[:, :, ::-1]
        polygons, _, _ = self.model.run_on_opencv_image(image)
        polygons = self.convert_polygon_format(polygons)
        return OCRDetOut(
            polygons=polygons,
            modelname="masktextspotterv3_detection",
        )


class MaskTextSpotterV3Recog(BaseOCRRecog):
    def __init__(self, weight_path, *args, **kwargs):
        self.model = load_masktextspooterv3(weight_path)

    def _boxpolygon2proposal(self, proposal_boxes, proposal_polygons, img_size):
        # proposal_boxes [[x0,y0,x1,y1], ..., [x0,y0,x1,y1]]
        # proposal_polygons [[polygon], ..., [polygon]]
        proposal_polygons = [Polygons([p], img_size, None) for p in proposal_polygons]
        boxlist_obj = BoxList(
            torch.Tensor(proposal_boxes).to(self.model.device), img_size, mode="xyxy"
        )
        masks_obj = SegmentationMask(proposal_polygons, img_size)
        boxlist_obj.add_field("masks", masks_obj)
        proposals = [boxlist_obj]
        return proposals

    def polygons2boxes(self, polygons):
        boxes = []
        for polygon in polygons:
            mask = draw_pos(polygon, 1.0) * 255.0
            box = min_bounding_rect(mask)
            boxes.append(box)
        return boxes

    def polygons2proposal(self, polygons, img_size):
        # proposal_boxes [[x0,y0,x1,y1], ..., [x0,y0,x1,y1]]
        # proposal_polygons [[polygon], ..., [polygon]]
        proposal_polygons = [Polygons([p], img_size, None) for p in polygons]
        boxes = self.polygons2boxes(polygons)
        boxlist_obj = BoxList(
            torch.Tensor(boxes).to(self.model.device), img_size, mode="xyxy"
        )
        masks_obj = SegmentationMask(proposal_polygons, img_size)
        boxlist_obj.add_field("masks", masks_obj)
        proposals = [boxlist_obj]
        return proposals

    def proposalitems2proposals(self, proposal_items: list, image_size: tuple):
        proposal_boxes = [x[0] for x in proposal_items]
        proposal_polygons = [x[1] for x in proposal_items]
        proposals = self.boxpolygon2proposal(
            proposal_boxes, proposal_polygons, image_size
        )
        return proposals

    def __call__(self, image: np.ndarray, polygons: list) -> list[str]:
        image = image[:, :, ::-1]
        image_size = (image.shape[1], image.shape[0])
        proposal_items = self.polygons2proposal(polygons)
        proposals = self.proposalitems2proposals(proposal_items, image_size)
        words = self.model.run_on_opencv_image(image, proposals=proposals)
        return words
