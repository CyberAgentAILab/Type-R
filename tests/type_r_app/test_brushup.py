import pytest

from type_r_app.launcher.layout_correction import get_word_mapping
from type_r_app.launcher.typo_correction import brushup_all, get_text_editing_inputs


@pytest.mark.gpu
def test_typocorrection(
    sample_image,
    sample_prompt,
    anytext,
    paddle,
    modelscope,
):
    word_mapping = get_word_mapping(
        sample_prompt,
        sample_image,
        paddle,
        modelscope,
        filter=True,
        filtering_size_rate=0.04,
    )
    text_editing_inputs = get_text_editing_inputs(word_mapping)

    brushup_all(
        sample_prompt,
        sample_image,
        anytext,
        modelscope,
        text_editing_inputs,
        trial_num=1,
    )
