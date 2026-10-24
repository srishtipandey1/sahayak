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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = train_main(["--dataset", "sen1floods11", "--out", str(OUTPUT_DIR)])
    if result != 0:
        return result

    archive = OUTPUT_DIR.parent / "sahayak-flood-unet-outputs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        for path in OUTPUT_DIR.rglob("*"):
            if path.is_file() and path != archive:
                output_zip.write(path, path.relative_to(OUTPUT_DIR))
    print(f"Output archive: {archive}")
    upload_outputs()
    return 0


def upload_outputs() -> None:
    """Upload requested artifacts if HF_TOKEN exists; never fail training on upload."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Hugging Face upload skipped: HF_TOKEN is not set.")
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
