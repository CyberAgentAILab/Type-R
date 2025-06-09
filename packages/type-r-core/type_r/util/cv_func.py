import math
from typing import TypeAlias

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform._geometric import _umeyama as get_sym_mat

Point: TypeAlias = tuple[int, int]  # (x, y)
Polygon: TypeAlias = list[Point]  # List of polygons [[x,y],...] for each text


def arr2tensor(arr, bs):
    arr = np.transpose(arr, (2, 0, 1))
    _arr = torch.from_numpy(arr.copy()).float().cuda()
    _arr = torch.stack([_arr for _ in range(bs)], dim=0)
    return _arr


def draw_pos(polygon, prob=1.0, height=512, width=512):
    img = np.zeros((height, width, 1))
    pts = polygon.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color=255)
    return img / 255.0


def pre_process(img_list, shape="3, 48, 320"):
    numpy_list = []
    img_num = len(img_list)
    assert img_num > 0
    for idx in range(0, img_num):
        # rotate
        img = img_list[idx]
        h, w = img.shape[1:]
        if h > w * 1.2:
            img = torch.transpose(img, 1, 2).flip(dims=[1])
            img_list[idx] = img
            h, w = img.shape[1:]
        # resize
        imgC, imgH, imgW = (int(i) for i in shape.strip().split(","))
        assert imgC == img.shape[0]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(imgH, resized_w),
            mode="bilinear",
            align_corners=True,
        )
        # padding
        padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32)
        padding_im[:, :, 0:resized_w] = resized_image[0]
        numpy_list += [padding_im.permute(1, 2, 0).cpu().numpy()]  # HWC ,numpy
    return numpy_list


def min_bounding_rect(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        print("Bad contours, using fake bbox...")
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    # sort
    x_sorted = sorted(box, key=lambda x: x[0])
    left = x_sorted[:2]
    right = x_sorted[2:]
    left = sorted(left, key=lambda x: x[1])
    (tl, bl) = left
    right = sorted(right, key=lambda x: x[1])
    (tr, br) = right
    if tl[1] > bl[1]:
        (tl, bl) = (bl, tl)
    if tr[1] > br[1]:
        (tr, br) = (br, tr)
    return np.array([tl, tr, br, bl])


def adjust_image(box, img):
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    width = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
    height = max(np.linalg.norm(pts1[0] - pts1[3]), np.linalg.norm(pts1[1] - pts1[2]))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # get transform matrix
    M = get_sym_mat(pts1, pts2, estimate_scale=True)
    C, H, W = img.shape
    T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
    theta = (
        torch.from_numpy(theta[:2, :]).unsqueeze(0).type(torch.float32).to(img.device)
    )
    grid = F.affine_grid(theta, torch.Size([1, C, H, W]), align_corners=True)
    result = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    # crop
    result = result[:, : int(height), : int(width)]
    return result


def crop_image(src_img, mask):
    box = min_bounding_rect(mask)
    result = adjust_image(box, src_img)
    if len(result.shape) == 2:
        result = torch.stack([result] * 3, axis=-1)
    return result


def harmonize_mask(
    img: np.ndarray,
    bg: np.ndarray,
    mask_paste: np.ndarray,
    mask_keep: np.ndarray,
    blur_param: int = 10,
    iteration: int = 5,
) -> np.ndarray:
    assert img.shape == bg.shape
    assert img.shape[:2] == mask_paste.shape
    assert img.shape[:2] == mask_keep.shape
    assert len(mask_paste.shape) == 2
    assert len(mask_keep.shape) == 2
    mask_paste = np.tile(mask_paste[:, :, np.newaxis].astype(np.float32), (1, 1, 3))
    mask_keep = np.tile(mask_keep[:, :, np.newaxis].astype(np.float32), (1, 1, 3))
    prior = cv2.blur(mask_paste, (blur_param, blur_param))
    prior = np.tanh(3 * prior)
    prior = np.maximum(prior, mask_paste)
    prior[mask_keep == 1] = 0
    harmonized_img = img.copy()
    for _ in range(iteration):
        harmonized_img = harmonized_img * (1 - prior) + bg * prior
    return harmonized_img


def paste_inpainted_results(
    img: np.ndarray,
    inpainted_img: np.ndarray,
    mask_paste: np.ndarray,
    mask_keep: np.ndarray,
    paste_option: str = "simple",
):
    if paste_option == "harmonize":
        _img = harmonize_mask(img, inpainted_img, mask_paste, mask_keep)
    elif paste_option == "simple":
        mask_paste = np.tile(mask_paste[:, :, np.newaxis], (1, 1, 3))
        mask_keep = np.tile(mask_keep[:, :, np.newaxis], (1, 1, 3))
        mask_paste[mask_keep == 1] = 0
        _img = img.copy()
        _img[mask_paste == 1] = inpainted_img[mask_paste == 1]
    else:
        raise
    return _img


def polygons2mask(img: np.ndarray, polygons: list[Polygon]):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for polygon in polygons:
        pos = draw_pos(np.array(polygon), 1.0, height=img.shape[0], width=img.shape[1])
        mask += pos[:, :, 0]
    mask = np.clip(mask, 0, 1)
    return mask


def polygons2maskdilate(
    img: np.ndarray,
    polygons: list[Polygon],
    dilation: bool = True,
    dilate_kernel_size: int = 10,
    dilate_iteration: int = 1,
):
    mask_erase = polygons2mask(img, polygons)
    if dilation is True:
        mask_erase = cv2.dilate(
            mask_erase,
            np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8),
            iterations=dilate_iteration,
        )
    return mask_erase


def polygons2bboxes(polygons: list[Polygon]) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for polygon in polygons:
        points = np.array(polygon)  # [n x 2]
        xpoints = points[:, 0]
        ypoints = points[:, 1]
        bbox = (
            ypoints.min(),
            ypoints.max(),
            xpoints.min(),
            xpoints.max(),
        )  # y0,y1,x0,x1
        bboxes.append(bbox)
    return bboxes


def crop_image_w_bbox(image: np.ndarray, bbox: tuple) -> np.array:
    y0, y1, x0, x1 = bbox
    return image[y0:y1, x0:x1]
