import numpy as np
import pytest
import skia
import yaml
from hydra.utils import instantiate
from PIL import Image

word_sample0 = ["category", "dog", "boat"]
word_sample1 = ["done", "catalog", "boy"]
params = {
    "sample0": (
        word_sample0,
        word_sample1,
        [0, 1, 2],
        [1, 0, 2],
    ),
    "sample1": (
        word_sample0,
        word_sample1[0:2],
        [0, 1, 2],
        [1, 0, 2],
    ),
}


@pytest.fixture(params=list(params.values()))
def word_combination_samples(request):
    yield request.param


word_sample0 = ["category", "dig", "boy"]
word_sample1 = ["dog", "boys", "catalog", "catalog"]

box_dummy = [
    [(0, 0), (0, 110), (110, 110), (110, 0)],
    [(0, 0), (0, 90), (90, 90), (90, 0)],
    [(0, 0), (0, 10), (10, 10), (10, 0)],
    [(0, 0), (0, 100), (100, 100), (100, 0)],
]

params = {
    "sample0": (
        word_sample0,
        word_sample1,
        box_dummy,
        0.1,
        1.0,
        [0, 1, 2, 3],
        [3, 0, 1, 2],
    ),
}


@pytest.fixture(params=list(params.values()))
def word_box_combination_samples(request):
    yield request.param


def pytest_addoption(parser):
    parser.addoption(
        "--gpufunc", action="store_true", default=False, help="run cpu func"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as gpu to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpufunc"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_gpu = pytest.mark.skip(reason="need --gpufunc option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def font_skia(pytestconfig):
    font_path = pytestconfig.getoption("font_path")
    fontmgr = skia.FontMgr()
    font_type_face = fontmgr.makeFromFile(font_path, 0)
    return skia.Font(font_type_face, 60, 1, 1e-20)


@pytest.fixture
def sample_image():
    return np.array(Image.open("tests/sample/0.jpg"))


@pytest.fixture
def sample_image_brushup():
    return np.array(Image.open("tests/sample/brushup_0.jpg"))


@pytest.fixture
def sample_image_woinpaintbg():
    return np.array(Image.open("tests/sample/woinpaintbg_0.jpg"))


@pytest.fixture
def sample_prompt():
    return "'Bury My Heart At Wounded Knee' Lp"


@pytest.fixture
def sample_wordmapping():
    prompt_words = ["Bury", "My", "Heart", "At", "Wounded", "Knee"]
    ocr_words = [
        "BURYY",
        "HEART",
        "AT",
        "KINE",
        "WOUNDED",
        "AT",
        "WOUNIDED",
        "D",
        "KINE",
        "Lu",
    ]
    ocr_words_match = ["BURYY", "HEART", "AT", "KINE", "WOUNDED", "AT"]
    prompt_words_match = ["Bury", "Heart", "My", "Knee", "Wounded", "At"]
    return ocr_words, prompt_words, ocr_words_match, prompt_words_match


@pytest.fixture
def sample_paddle_words():
    return [
        "BURYY",
        "HEART",
        "AT",
        "KINE",
        "WOUNDED",
        "AT",
        "WOUNIDED",
        "D",
        "KINE",
        "Lu",
    ]


@pytest.fixture
def sample_paddle_polygons():
    polygons = [
        [(68, 48), (253, 48), (253, 116), (68, 116)],
        [(267, 46), (408, 46), (408, 117), (267, 117)],
        [(424, 39), (478, 48), (466, 122), (412, 113)],
        [(327, 130), (436, 130), (436, 203), (327, 203)],
        [(63, 140), (292, 135), (293, 199), (64, 204)],
        [(65, 229), (112, 229), (112, 266), (65, 266)],
        [(123, 233), (289, 233), (289, 261), (123, 261)],
        [(299, 229), (328, 229), (328, 264), (299, 264)],
        [(336, 230), (423, 230), (423, 263), (336, 263)],
        [(430, 225), (476, 229), (473, 267), (427, 263)],
    ]
    return polygons


@pytest.fixture
def sample_unformatted_polygons():
    polygons = [
        [68, 48, 253, 48, 253, 116, 68, 116],
    ]
    return polygons


@pytest.fixture
def sample_masktextspotterv3_words():
    return [
        "wounided",
        "kine",
        "l",
        "d",
        "at",
        "wounded",
        "kine",
        "buryiy",
        "heart",
        "at",
    ]


@pytest.fixture
def sample_masktextspotterv3_polygons():
    polygons = [
        [(117, 234), (122, 261), (290, 261), (290, 233), (284, 230)],
        [(335, 229), (335, 259), (422, 259), (422, 229)],
        [(434, 228), (434, 261), (468, 261), (468, 228)],
        [
            (303, 229),
            (301, 231),
            (301, 235),
            (303, 236),
            (301, 239),
            (301, 260),
            (325, 260),
            (325, 231),
            (323, 229),
        ],
        [
            (65, 233),
            (65, 259),
            (106, 259),
            (106, 230),
            (93, 230),
            (92, 232),
            (88, 230),
            (68, 230),
        ],
        [(59, 133), (59, 204), (293, 204), (293, 133)],
        [(325, 134), (328, 199), (432, 199), (430, 131)],
        [(63, 48), (63, 116), (252, 116), (252, 44)],
        [(265, 44), (265, 116), (407, 113), (407, 45)],
        [(468, 43), (421, 46), (418, 114), (470, 114)],
    ]
    return polygons


@pytest.fixture
def sample_craft_polygons():
    polygons = [
        [
            (14, 24),
            (18, 25),
            (25, 24),
            (29, 25),
            (35, 24),
            (40, 23),
            (45, 22),
            (46, 27),
            (41, 29),
            (35, 30),
            (30, 31),
            (25, 30),
            (19, 31),
            (15, 30),
        ],
        [(66, 47), (473, 48), (473, 115), (66, 114)],
        [(60, 137), (296, 134), (297, 202), (61, 204)],
        [(327, 137), (433, 135), (435, 197), (328, 199)],
        [(67, 231), (471, 230), (472, 262), (67, 263)],
    ]
    return polygons


@pytest.fixture
def sample_modelscope_words():
    return ["BURYY", "HEART", "三", "KINE", "WOUNDED", "AT"]


@pytest.fixture
def craft():
    with open(
        "src/type_r_app/config/ocr_detection/craft.yaml", "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)
    return instantiate(config)()


@pytest.fixture
def hisam():
    with open(
        "src/type_r_app/config/ocr_detection/hisam.yaml", "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)
    return instantiate(config)()


@pytest.fixture
def deepsolo():
    with open(
        "src/type_r_app/config/ocr_detection/deepsolo.yaml", "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)
    return instantiate(config)()


@pytest.fixture
def modelscope():
    with open(
        "src/type_r_app/config/ocr_recognition/modelscope.yaml", "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)
    return instantiate(config)()


@pytest.fixture
def paddle():
    with open(
        "src/type_r_app/config/ocr_detection/paddleocr.yaml", "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)
    return instantiate(config)()


@pytest.fixture
def trocr():
    with open(
        "src/type_r_app/config/ocr_recognition/trocr.yaml", "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)
    return instantiate(config)()


@pytest.fixture
def sample_trocr_words():
    return ["BURYLY", "HEART", "AT", "KINE", "WOUNDED", "AT"]


@pytest.fixture
def clovarecog():
    with open(
        "src/type_r_app/config/ocr_recognition/clovarecog.yaml", "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)
    return instantiate(config)()


@pytest.fixture
def sample_clovarecog_words():
    return ["buryly", "heart", "at", "kinie", "wounded", "at"]


@pytest.fixture
def anytext():
    with open(
        "src/type_r_app/config/text_editor/anytext.yaml", "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)
    return instantiate(config)(font_path="resources/data/LiberationSans-Regular.ttf")


@pytest.fixture
def masktextspotterv3_detection():
    with open(
        "src/type_r_app/config/ocr_detection/masktextspotterv3.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)
    return instantiate(config)()
