import os

import streamlit as st
from huggingface_hub import snapshot_download
from streamlit.logger import get_logger

logger = get_logger(__name__)


def download_resources(output_dir: str = "resources/") -> None:
    """Download resources from Hugging Face Hub."""
    logger.info("Downloading resources from Hugging Face Hub...")
    allowed_patterns = [
        "weight/ocr/modelscope/configuration.json",
        "weight/ocr/modelscope/pytorch_model.pt",
        "weight/ocr/modelscope/vocab.txt",
        "weight/text_eraser/big-lama.pt",
        "weight/text_editor/anytext.ckpt",
        "example.txt",
        "data/*.ttf",
        "README.md",
    ]
    if os.path.exists(output_dir):
        logger.info(f"Output directory {output_dir} already exists. Skipping download.")
        return

    with st.spinner("Downloading model weights..."):
        snapshot_download(
            repo_id="cyberagent/type-r",
            repo_type="model",
            local_dir=output_dir,
            allow_patterns=allowed_patterns,
        )
