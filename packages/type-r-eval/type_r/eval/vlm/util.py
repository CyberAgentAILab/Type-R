import json
from copy import deepcopy
from pathlib import Path
from typing import Any, NamedTuple


class TesterInput(NamedTuple):
    image_paths: list[str]
    id: str | None  # may not be necessary but kept for debug
    prompt: str | None = None


def get_image_path_list(directories: list[str]) -> list[list[str]]:
    assert len(directories) >= 2, "At least two directories are required."
    output = []
    directories_ = [Path(d) for d in directories]
    names = sorted([f.name for f in Path(directories[0]).glob("*.*")])
    for name in names:
        images = []
        for dir_ in directories_:
            path = dir_ / name
            if not path.exists():
                raise FileNotFoundError(f"{path} not found.")
            images.append(str(path))
        output.append(images)
    return output


def validate_path(path: str | Path) -> Path:
    assert path not in ["", None]
    if isinstance(path, str):
        path = Path(path)

    if "." in path.name:
        dir_ = path.parent
    else:
        dir_ = path
    if not dir_.exists():
        dir_.mkdir(parents=True, exist_ok=True)

    return path


JSON_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
{schema}
"""


def get_format_instructions(schema: dict[str, Any]) -> str:
    """Return the format instructions for the JSON output.
    Returns:
        The format instructions for the JSON output.
    """
    # Copy schema to avoid altering original Pydantic schema.
    schema = deepcopy(schema)
    # Remove extraneous fields.
    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    if "type" in reduced_schema:
        del reduced_schema["type"]
    # Ensure json in context is well-formed with double quotes.
    schema_str = json.dumps(reduced_schema)
    return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)
