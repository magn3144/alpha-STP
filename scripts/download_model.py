"""Download a Hugging Face model into the repository's models directory."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY = Path(__file__).resolve().parents[1]
MODELS_DIR = REPOSITORY / "models"
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    """Parse model, revision, and destination inputs and return them."""

    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model into models/.",
    )
    parser.add_argument("model_id", nargs="?", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face branch, tag, or commit to download.",
    )
    return parser.parse_args()


def download_model(model_id: str, revision: str) -> Path:
    """Download one model revision and return its local models directory."""

    model_dir = MODELS_DIR / model_id.rsplit("/", 1)[-1]
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=model_dir,
    )
    return model_dir


def main() -> None:
    """Download the requested model and print its local path."""

    args = parse_args()
    print(download_model(args.model_id, args.revision))


if __name__ == "__main__":
    main()
