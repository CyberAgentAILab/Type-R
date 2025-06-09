import json
from typing import TypeAlias

from pydantic import BaseModel

Point: TypeAlias = tuple[int, int]  # Point [x,y]
Polygon: TypeAlias = list[Point]  # List of points
Polygons: TypeAlias = list[Polygon]  # List of polygons


class JsonIOModel(BaseModel):
    @classmethod
    def load_json(cls, json_path: str) -> "JsonIOModel":
        with open(json_path, "r") as f:
            json_str = f.read()
        return cls.model_validate_json(json_str)

    def dump_json(self, json_path: str) -> None:
        with open(json_path, "w") as f:
            json.dump(self.model_dump(), f, indent=4, ensure_ascii=False)


class WordMapping(JsonIOModel):
    """Word mapping info."""

    words_ocr: list[str]
    words_prompt: list[str]
    polygons: list[Polygon]
    prompt2ocr: dict[int, int]  # ocr index 2 prompt index
