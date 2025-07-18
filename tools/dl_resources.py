import argparse
import logging

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download Type-R resources.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download all the resources for Type-R.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="resources",
        help="The local output directory to save the downloaded resources.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    allowed_patterns = ["*"] if args.full else [
        "weight/ocr/modelscope/configuration.json",
        "weight/ocr/modelscope/pytorch_model.pt",
        "weight/ocr/modelscope/vocab.txt",
        "weight/text_eraser/big-lama.pt",
        "weight/text_editor/anytext.ckpt",
        "data/*.ttf",
        "README.md",
    ]
    snapshot_download(
        repo_id="cyberagent/type-r",
        repo_type="model",
        local_dir=args.output_dir,
        allow_patterns=allowed_patterns,
    )
