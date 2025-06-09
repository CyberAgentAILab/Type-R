from type_r.adjuster.wordchain import wordchain

from type_r_app.launcher.layout_correction import get_wordchain


def test_wordchain(word_combination_samples) -> None:
    source, target, src_indexes, tgt_indexes = word_combination_samples
    _src_indexes, _tgt_indexes = wordchain(source, target)
    print(_src_indexes, _tgt_indexes)
    assert list(_src_indexes) == src_indexes
    assert list(_tgt_indexes) == tgt_indexes


def test_wordchain_sample(sample_wordmapping):
    ocr_words, prompt_words, ocr_words_match, prompt_words_match = sample_wordmapping
    prompt2ocr = get_wordchain(prompt_words, ocr_words)
    matched_ocr_ids = [ocr_id for ocr_id in prompt2ocr.values() if ocr_id != -1]
    _ocr_words_match = [ocr_words[matched_id] for matched_id in matched_ocr_ids]
    _prompt_words_match = prompt_words
    ocr_words_match = sorted([word.lower() for word in ocr_words_match])
    _ocr_words_match = sorted([word.lower() for word in _ocr_words_match])
    assert "".join(ocr_words_match) == "".join(_ocr_words_match), (
        "The word matching result is not correct"
    )
    _prompt_words_match = sorted([word.lower() for word in _prompt_words_match])
    prompt_words_match = sorted([word.lower() for word in prompt_words_match])
    assert "".join(_prompt_words_match) == "".join(prompt_words_match), (
        "The word matching result is not correct"
    )
