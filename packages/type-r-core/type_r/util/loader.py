import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_module_lazily(module_name: str) -> Any:
    """Loading module lazily."""
    logger.debug(f"Loading module {module_name} lazily.")
    module = importlib.import_module(module_name)
    return module


def load_class_lazily(full_class_name: str) -> Any:
    """Loading class lazily.
    
    Full class name must have the package and module prefix.
    """
    split_name = full_class_name.rsplit('.', 1)
    if len(split_name) != 2:
        raise ValueError(f"Invalid class name: {full_class_name}")
    module_name, class_name = tuple(split_name)
    module = load_module_lazily(module_name)
    return getattr(module, class_name)