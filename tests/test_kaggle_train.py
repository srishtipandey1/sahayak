"""Tests for Kaggle notebook structure and safe secret handling."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_train.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("sahayak_kaggle_train", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise ImportError(f"Unable to load Kaggle runner from {RUNNER_PATH}")
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)
ensure_hf_token = RUNNER_MODULE.ensure_hf_token


def test_kaggle_notebook_has_four_safe_code_cells() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_train.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    all_code = "\n".join(code_cells)

    assert len(code_cells) == 4
    assert "--no-deps" not in all_code
    assert "!pip install -q -e ." in code_cells[0]
    assert "!python -m sahayak.ml.train --dry-run" in code_cells[2]
    assert "!python scripts/kaggle_train.py" in code_cells[2]
    assert "print(hf_token)" not in all_code
    assert "print(os.environ" not in all_code
    assert "print(type(error).__name__)" in code_cells[2]
    for package in (
        "segmentation-models-pytorch",
        "rasterio",
        "pyogrio",
        "huggingface_hub",
        "pandas",
        "tqdm",
        "matplotlib",
        "requests",
        "scipy",
    ):
        assert package in code_cells[0]


def test_runner_loads_kaggle_secret_without_printing_value(
    monkeypatch,
    capsys,
) -> None:
    token = "test-token-that-must-not-be-printed"

    class FakeUserSecretsClient:
        def get_secret(self, key: str) -> str:
            assert key == "HF_TOKEN"
            return token

    kaggle_secrets = ModuleType("kaggle_secrets")
    kaggle_secrets.UserSecretsClient = FakeUserSecretsClient
    monkeypatch.setitem(sys.modules, "kaggle_secrets", kaggle_secrets)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert ensure_hf_token()
    output = capsys.readouterr().out

    assert output == "HF_TOKEN available: yes\n"
    assert token not in output
    assert os.environ["HF_TOKEN"] == token


def test_runner_reports_missing_token_without_showing_secret_errors(
    monkeypatch,
    capsys,
) -> None:
    class MissingSecretsClient:
        def get_secret(self, _key: str) -> str:
            raise RuntimeError("secret retrieval failure")

    kaggle_secrets = ModuleType("kaggle_secrets")
    kaggle_secrets.UserSecretsClient = MissingSecretsClient
    monkeypatch.setitem(sys.modules, "kaggle_secrets", kaggle_secrets)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    assert not ensure_hf_token()

    assert capsys.readouterr().out == "HF_TOKEN available: no\n"


def test_runner_creates_zip_even_when_training_raises(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    kaggle_train = RUNNER_MODULE
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "partial-metrics.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(kaggle_train, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        kaggle_train,
        "train_main",
        lambda _args: (_ for _ in ()).throw(RuntimeError("evaluation failed")),
    )
    torch = ModuleType("torch")
    torch.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    try:
        kaggle_train.main()
    except RuntimeError as error:
        assert str(error) == "evaluation failed"
    else:
        raise AssertionError("Training error was not propagated after archiving.")

    archive = output_dir.parent / "sahayak-flood-unet-outputs.zip"
    assert archive.is_file()
    assert "Output archive:" in capsys.readouterr().out
