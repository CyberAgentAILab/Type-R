import tempfile
from typing import Any

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from type_r.util.cv_func import adjust_image, draw_pos, min_bounding_rect

from ..base import BaseOCRRecog
from .dataset import AlignCollate, RawDataset
from .model import Model
from .utils import AttnLabelConverter


def get_recog_opt_and_model(tmpdirname, weight_dir):
    opt = {
        "image_folder": f"{tmpdirname}/",
        "workers": 0,
        "batch_size": 1,
        "saved_model": f"{weight_dir}/TPS-ResNet-BiLSTM-Attn.pth",
        "batch_max_length": 25,
        "imgH": 32,
        "imgW": 100,
        "rgb": False,
        "character": "0123456789abcdefghijklmnopqrstuvwxyz",
        "sensitive": False,
        "PAD": False,
        "Transformation": "TPS",
        "FeatureExtraction": "ResNet",
        "SequenceModeling": "BiLSTM",
        "Prediction": "Attn",
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 256,
    }
    opt["num_gpu"] = torch.cuda.device_count()

    class DictDotNotation(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    opt = DictDotNotation(opt)

    """model configuration"""
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    model.eval()

    return opt, model


def recognition(opt: Any, model: Any):
    converter = AttnLabelConverter(opt.character)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
    )
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo,
        pin_memory=True,
    )

    # predict
    results = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(
                device
            )
            text_for_pred = (
                torch.LongTensor(batch_size, opt.batch_max_length + 1)
                .fill_(0)
                .to(device)
            )

            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(
                image_path_list, preds_str, preds_max_prob
            ):
                if "Attn" in opt.Prediction:
                    pred_EOS = pred.find("[s]")
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                results.append(
                    {
                        "image_path": img_name,
                        "predicted_labels": pred,
                        "confidence_score": confidence_score,
                    }
                )
    return results


class ClovaRecogBench(BaseOCRRecog):
    def __init__(self, weight_path, *args, **kwargs):
        self.opt, self.model = get_recog_opt_and_model("tmp", weight_path)

    def inference(self, image: np.ndarray, polygons: list) -> list[str]:
        image = torch.from_numpy(image).float()
        image = einops.rearrange(image, "h w c-> c h w")
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i, polygon in enumerate(polygons):
                self.opt.image_folder = f"{tmpdirname}/"
                mask = (
                    draw_pos(
                        np.array(polygon),
                        1.0,
                        height=image.shape[1],
                        width=image.shape[2],
                    )
                    * 255.0
                )
                box = min_bounding_rect(mask.astype(np.uint8))
                inp = adjust_image(box, image).data.numpy()
                inp = np.transpose(inp, (1, 2, 0))
                Image.fromarray(inp.astype(np.uint8)).save(f"{tmpdirname}/{i}.jpg")
            results = recognition(self.opt, self.model)
        words = []
        for r in results:
            words.append(r["predicted_labels"])
        return words
