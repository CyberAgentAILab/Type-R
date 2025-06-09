import copy
import re


def get_p_r_acc(pred, gt):
    pred = [p.strip().lower() for p in pred]
    gt = [g.strip().lower() for g in gt]

    pred_orig = copy.deepcopy(pred)
    gt_orig = copy.deepcopy(gt)

    pred_length = len(pred)
    gt_length = len(gt)

    for p in pred:
        if p in gt_orig:
            pred_orig.remove(p)
            gt_orig.remove(p)

    p = (pred_length - len(pred_orig)) / (pred_length + 1e-8)
    r = (gt_length - len(gt_orig)) / (gt_length + 1e-8)

    pred_sorted = sorted(pred)
    gt_sorted = sorted(gt)
    if "".join(pred_sorted) == "".join(gt_sorted):
        acc = 1
    else:
        acc = 0

    return p, r, acc


def get_key_words(text: str):
    words = []
    text = text
    # print(text)
    matches = re.findall(r"'(.*?)'", text)  # find the keywords enclosed by ''
    if matches:
        for match in matches:
            words.extend(match.split())

    return words


def _get_key_words(text: str):
    words = []
    text = text
    # print(text)
    matches = re.findall(r'"(.*?)"', text)  # find the keywords enclosed by ''
    if matches:
        for match in matches:
            words.extend(match.split())
    return words
