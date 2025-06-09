from pathlib import Path


def load_chardictpath(input_json: str) -> str:
    if "wukong" in input_json:
        rec_char_dict_path = str(Path(__file__).parent / "ppocr_keys_v1.txt")
    elif "laion" in input_json:
        rec_char_dict_path = str(Path(__file__).parent / "en_dict.txt")
    return rec_char_dict_path
