import json
import os
import warnings

warnings.simplefilter("ignore", FutureWarning)

import logging
import shutil
import tempfile

import cv2
import hydra
import numpy as np
import pandas as pd
from hydra.utils import call, instantiate
from loguru import logger
from omegaconf import DictConfig
from type_r.eval.misc.clipscore import calc_clipscore
from type_r.eval.misc.fid_score import calculate_fid_given_paths
from type_r.eval.ocr.ocrscore import _get_key_words, get_key_words, get_p_r_acc
from type_r.eval.vlm.dataset import DirectoryDataset
from type_r.eval.vlm.tester import BaseTester
from type_r.eval.vlm.util import validate_path
from type_r.eval.vlm.vlmeval_helper import VLMEvalWrapper
from type_r.ocr.masktextspotterv3.masktextspotterv3 import TextDemo


def ocreval(evaldataset: DirectoryDataset, masktextspotterv3_weight: str) -> dict:
    logging.getLogger("maskrcnn_benchmark.utils.checkpoint").setLevel(logging.CRITICAL)
    logging.getLogger("maskrcnn_benchmark.utils.model_serialization").setLevel(
        logging.CRITICAL
    )
    text_demo = TextDemo(
        masktextspotterv3_weight,
        min_image_size=800,
        confidence_threshold=0.7,
        output_polygon=True,
    )
    # load image and then run prediction

    ##########################
    # main loop
    ##########################
    ocr_results = {}
    results = {"cnt": 0, "p": 0, "r": 0, "f": 0, "acc": 0}
    for element in evaldataset:
        image_path = element.image_paths[0]
        image = cv2.imread(image_path)
        _, result_words, _ = text_demo.run_on_opencv_image(image)
        ocr_results[element.id] = result_words
        prompt = element.prompt
        if "ChineseDrawText" in element.id or "DrawBenchText" in element.id:
            keywords = _get_key_words(prompt)
        else:
            keywords = get_key_words(prompt)
        p, r, acc = get_p_r_acc(result_words, keywords)
        results["cnt"] += 1
        results["p"] += p
        results["r"] += r
        results["acc"] += acc
    results["acc"] /= results["cnt"]
    results["p"] /= results["cnt"]
    results["r"] /= results["cnt"]
    results["f"] = (
        2 * results["p"] * results["r"] / (results["p"] + results["r"] + 1e-8)
    )
    return results


def visualize_rating_results(df: pd.DataFrame, id_key: str) -> tuple[float, float]:
    """
    Visualize the rating results from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the evaluation results.
        id_key (str): The key to identify the image IDs in the DataFrame.
    Returns:
        tuple: Mean and standard deviation of the scores.
    """

    df = df.replace(-1, 1)  # -1 means failed to evaluate, thus give the lowest
    df = df.drop(columns=[id_key])
    flag = False
    for name, series in df.items():
        if name != "score":
            continue
        mean = series.mean().item()
        std = series.std().item()
        logger.info(f"VLM Evaluation {name}: {mean:.2f} ± {std:.2f}")
        flag = True
        break
    if not flag:
        raise ValueError("The DataFrame does not contain the expected 'score' column.")
    return mean, std


def vlmeval(
    evaldataset: DirectoryDataset, tester: BaseTester, vlm_name: str, root_res: str
) -> tuple[float, float, str]:
    model = VLMEvalWrapper(
        vlm_name=vlm_name,
        output_object=tester.pydantic_object,
        system_prompt=tester.system_prompt,
        # **parse_unknown(unknown),  # TODO: how to feed vlm-specific arguments?
    )
    output_path = os.path.join(root_res, "evaluation", "eval.csv")
    output_path = validate_path(output_path)

    results: list[dict] = []
    for input_ in evaldataset:
        result = tester(model=model, input_=input_)
        results.append(result)
    df = pd.DataFrame.from_dict(results)
    df.to_csv(str(output_path), index=False)
    # df = pd.read_csv(str(output_path), dtype={"id": object})
    mean, std = visualize_rating_results(df, id_key="id")
    return mean, std, str(output_path)


def eval_clip_score(
    evaldataset: DirectoryDataset,
    device: str = "cuda:0",
) -> float:
    image_id_list, image_list, prompt_list = [], [], []
    for x in evaldataset:
        image_path = x.image_paths[0]
        image_id_list.append(x.id)
        image_list.append(image_path)
        prompt_list.append(x.prompt)

    score = calc_clipscore(
        image_ids=image_id_list,
        image_paths=image_list,
        text_list=prompt_list,
        device=device,
    )
    clip_score = np.mean([s["CLIPScore"] for s in score.values()])
    return clip_score


def eval_fid_score(evaldata, hfds) -> float:
    with tempfile.TemporaryDirectory() as tmpdirname_gt:
        with tempfile.TemporaryDirectory() as tmpdirname_pred:
            img_list = [elm.image_paths[0] for elm in evaldata]
            for i, img in enumerate(img_list):
                _from = img
                _to = f"{tmpdirname_pred}/{i}.jpg"
                shutil.copy(_from, _to)
            for i, elm in enumerate(hfds):
                img = elm["image"]
                if img is not None:
                    _to = f"{tmpdirname_gt}/{i}.jpg"
                    img.save(_to)
            if (
                len(os.listdir(tmpdirname_gt)) == 0
                or len(os.listdir(tmpdirname_pred)) == 0
            ):
                logger.info("No images found for FID evaluation.")
                return "n/a"
            # Calculate FID
            fid = calculate_fid_given_paths(
                paths=[tmpdirname_gt, tmpdirname_pred], num_workers=0
            )
    return fid


@hydra.main(version_base=None, config_path="../config", config_name="evaluation")
def evaluation(config: DictConfig):
    root_res = config.output_dir
    os.makedirs(os.path.join(root_res, "evaluation"), exist_ok=True)
    hfds = call(config.dataset)(image_dir=config.evaluation_img_dir)()
    evaldataset = instantiate(config.evaldata)(
        image_dir=config.evaluation_img_dir, hfds=hfds
    )
    tester = instantiate(config.evaluation)()

    if config.evalbyvlm is True:
        vlm_score_mean, vlm_score_std, vlm_output_path = vlmeval(
            evaldataset,
            tester,
            config.vlm_name,
            root_res,
        )
    else:
        vlm_output_path = "n/a"
        vlm_score_mean = 0.0
        vlm_score_std = 0.0

    logger.info(f"vlm score: {vlm_score_mean:.2f} ± {vlm_score_std:.2f}")

    ocrres = ocreval(evaldataset, config.masktextspotterv3_weight)
    logger.info(f"ocr score: {ocrres}")
    fid_score = "n/a"
    if len(evaldataset) <= 1:
        logger.info("FID evaluation requires more than one image")
    elif config.dataset.sub_set != "test":
        logger.info("FID evaluation is for only test subset")
    else:
        logger.info("FID evaluation")
        fid_score = eval_fid_score(evaldataset, hfds)
    logger.info(f"fid score: {fid_score}")
    clip_score = eval_clip_score(
        evaldataset,
    )
    logger.info(f"clip score: {clip_score}")
    result_json = {
        "vlm_output_path": vlm_output_path,
        "vlm_score": f"{vlm_score_mean:.2f} +/- {vlm_score_std:.2f}",
        "ocrres": ocrres,
        "fid_score": fid_score,
        "clip_score": clip_score,
    }
    output_path = os.path.join(root_res, "evaluation", "eval.json")
    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=4)


if __name__ == "__main__":
    logger.info("start evaluation")
    evaluation()
