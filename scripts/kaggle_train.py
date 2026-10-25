"""Kaggle entry point: train the model, archive outputs, optionally upload artifacts."""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

from sahayak.ml.train import main as train_main

LOGGER = logging.getLogger("sahayak.kaggle_train")
OUTPUT_DIR = Path("/kaggle/working/outputs")
MODEL_REPOSITORY = "SrishtiPandey1203/sahayak-flood-unet"
UPLOAD_FILES = ("best.pt", "metrics.json", "config.json", "model_card.md")


def main() -> int:
    """Train in Kaggle's working directory and perform best-effort model upload."""
    import torch

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: unavailable; training will use CPU.")
    ensure_hf_token()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = train_main(["--dataset", "sen1floods11", "--out", str(OUTPUT_DIR)])
    finally:
        create_outputs_archive()
    if result != 0:
        return result
    upload_outputs()
    return 0


def create_outputs_archive() -> Path:
    """Zip the current output directory, including partial outputs after a failure."""
    archive = OUTPUT_DIR.parent / "sahayak-flood-unet-outputs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        for path in OUTPUT_DIR.rglob("*"):
            if path.is_file() and path != archive:
                output_zip.write(path, path.relative_to(OUTPUT_DIR))
    print(f"Output archive: {archive}")
    return archive


def ensure_hf_token() -> bool:
    """Load an optional Hugging Face token from Kaggle Secrets without displaying it."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = token
    available = bool(token)
    print(f"HF_TOKEN available: {'yes' if available else 'no'}")
    return available


def upload_outputs() -> None:
    """Upload requested artifacts if HF_TOKEN exists; never fail training on upload."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(
            repo_id=MODEL_REPOSITORY,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        for filename in UPLOAD_FILES:
            path = OUTPUT_DIR / filename
            if not path.is_file():
                raise FileNotFoundError(f"Expected upload artifact is missing: {filename}")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=filename,
                repo_id=MODEL_REPOSITORY,
                repo_type="model",
                token=token,
            )
        print(f"Uploaded model artifacts to {MODEL_REPOSITORY}.")
    except Exception as error:
        LOGGER.warning(
            "Hugging Face upload failed (%s); local training outputs were preserved.",
            type(error).__name__,
        )
        print(f"Hugging Face upload failed ({type(error).__name__}); local outputs are preserved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
