from typing import TypeAlias

import Levenshtein as lev
import numpy as np
from scipy.optimize import linear_sum_assignment

Point: TypeAlias = tuple[int, int]
Polygon: TypeAlias = list[Point]  # List of polygons [[x,y],...] for each text


def normalize_cost(cost: np.ndarray) -> np.ndarray:
    mean = np.mean(cost)
    std = np.std(cost)
    if std > 0:
        for i in range(len(cost)):
            cost[i] = (cost[i] - mean) / std
    return cost


def compute_text_similarity_matrix(
    prompt_words: list[str], ocr_words: list[str], word_num: int, normalize: bool = True
) -> np.array:
    cost = np.zeros((word_num, word_num)) + 1000
    for i in range(len(prompt_words)):
        for j in range(len(ocr_words)):
            cost[i][j] = lev.distance(prompt_words[i], ocr_words[j])
    if normalize:
        cost = normalize_cost(cost)
    return cost


def get_cost(
    prompt_words: list[str],
    ocr_words: list[str],
) -> np.array:
    """
    cost is a matrix of shape (word_num, word_num)
    """
    word_num = max(len(prompt_words), len(ocr_words))
    cost = np.zeros((word_num, word_num))  # source: prompt, target: ocr
    cost = compute_text_similarity_matrix(prompt_words, ocr_words, word_num)
    return cost


def wordchain(
    words_prompt: list[str],
    words_ocr: list[str],
) -> tuple[list[int], list[int]]:
    """
    Matches words from two lists based on a cost function and returns the indexes of the matched words.
    Args:
        words_prompt (list[str]): A list of words from the prompt.
        words_ocr (list[str]): A list of words from OCR (Optical Character Recognition).
    Returns:
        tuple[list[int], list[int]]: Two lists of indexes. The first list contains the indexes of the matched words from `words_prompt`,
                                     and the second list contains the indexes of the matched words from `words_ocr`.
    """
    _words_prompt = [prompt_word.lower() for prompt_word in words_prompt]
    _words_ocr = [ocr_word.lower() for ocr_word in words_ocr]
    cost = get_cost(_words_prompt, _words_ocr)
    src_indexes, tgt_indexes = linear_sum_assignment(cost, maximize=False)
    return src_indexes.tolist(), tgt_indexes.tolist()


def wordchain_mapping(
    words_prompt: list[str],
    words_ocr: list[str],
    src_indexes: list[int],
    tgt_indexes: list[int],
) -> dict[int, int]:
    """
    The length of src_indexes and tgt_indexes are [word_num]
    pad with -1 if ocr_word_index is out of range for words_ocr
    """
    prompt2ocr = {}
    for prompt_word_index in range(len(words_prompt)):
        ocr_word_index = tgt_indexes[src_indexes.index(prompt_word_index)]
        if len(words_ocr) > ocr_word_index:
            prompt2ocr[prompt_word_index] = ocr_word_index
        else:
            prompt2ocr[prompt_word_index] = -1
    return prompt2ocr


def get_unmatched_ocr_ids(
    words_ocr: list[str], prompt2ocr: dict[int, int]
) -> list[int]:
    """
    Get unmatched OCR IDs based on the prompt2ocr mapping.
    Args:
        words_ocr (list[str]): List of words from OCR.
        prompt2ocr (dict[int, int]): Mapping of prompt word indexes to OCR word indexes.
    Returns:
        list[int]: List of unmatched OCR IDs.
    """
    unmatched_ids = []
    _matched_ids = prompt2ocr.values()
    for ocr_word_index in range(len(words_ocr)):
        if ocr_word_index not in _matched_ids:
            unmatched_ids.append(ocr_word_index)
    return unmatched_ids
