"""Focused no-network tests for the optional Sentinel-1 ML pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sahayak.ml.baselines import otsu_vv_metrics, otsu_vv_threshold  # noqa: E402
from sahayak.ml.data import (  # noqa: E402
    IGNORE_INDEX,
    Sen1ChipDataset,
    build_dataset_splits,
    parse_split_csv,
    random_crop,
)
from sahayak.ml.losses import bce_dice_loss  # noqa: E402
from sahayak.ml.metrics import (  # noqa: E402
    bootstrap_intervals,
    confusion_counts,
    metrics_from_counts,
    select_threshold,
)
from sahayak.ml.train import (  # noqa: E402
    run_config,
    time_budget_expired,
    train_batch,
    train_model,
)


def test_loss_and_metrics_ignore_minus_one_and_255_labels() -> None:
    logits = torch.tensor([[[[0.5, -0.5, 3.0, -3.0]]]], dtype=torch.float32)
    labels = torch.tensor([[[1, 0, IGNORE_INDEX, -1]]], dtype=torch.int64)

    loss = bce_dice_loss(logits, labels)
    expected = bce_dice_loss(logits[..., :2], labels[..., :2])
    assert torch.isclose(loss, expected)

    counts = confusion_counts(
        np.asarray([[True, False, True, False]]),
        np.asarray([[1, 0, IGNORE_INDEX, -1]]),
    )
    assert counts == {
        "tn": 1,
        "fp": 0,
        "fn": 0,
        "tp": 1,
        "valid_pixels": 2,
        "positive_pixels": 1,
    }


def test_otsu_baseline_separates_synthetic_bimodal_chip() -> None:
    vv = np.asarray([[-22.0, -21.0, -20.0, -4.0, -3.0, -2.0]], dtype=np.float32)
    labels = np.asarray([[1, 1, 1, 0, 0, 0]], dtype=np.uint8)

    threshold = otsu_vv_threshold(vv, labels)
    output = otsu_vv_metrics(vv, labels)

    assert -20.0 < threshold < -4.0
    assert output["counts"]["tp"] == 3
    assert output["counts"]["tn"] == 3
    assert output["flood_iou"] == 1.0


def test_metrics_threshold_selection_and_bootstrap_on_tiny_arrays() -> None:
    counts = confusion_counts(
        np.asarray([[True, False], [False, False]]),
        np.asarray([[1, 0], [0, 255]], dtype=np.uint8),
    )
    result = metrics_from_counts(counts)
    threshold = select_threshold(
        [np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)],
        [np.asarray([[1, 0], [0, 255]], dtype=np.uint8)],
    )
    intervals = bootstrap_intervals([counts, counts], resamples=40, seed=19)

    assert result["confusion_matrix"] == [[2, 0], [0, 1]]
    assert threshold == pytest.approx(0.9)
    for interval in intervals.values():
        assert interval["lower_95"] <= interval["upper_95"]


def test_threshold_selection_rejects_no_valid_labeled_pixels() -> None:
    with pytest.raises(ValueError, match="valid labeled pixels"):
        select_threshold(
            [np.asarray([[0.2, 0.8]])],
            [np.asarray([[IGNORE_INDEX, -1]])],
        )


def test_random_crop_and_split_csv_parsing() -> None:
    image = np.arange(2 * 12 * 14, dtype=np.float32).reshape(2, 12, 14)
    labels = np.zeros((12, 14), dtype=np.uint8)
    cropped_image, cropped_labels = random_crop(
        image,
        labels,
        8,
        np.random.default_rng(42),
    )
    csv_text = (
        "s1,label\n"
        "train_a_S1Hand.tif,train_a_LabelHand.tif\n"
        "train_b_S1Hand.tif,train_b_LabelHand.tif\n"
    )
    parsed = parse_split_csv(csv_text)

    assert cropped_image.shape == (2, 8, 8)
    assert cropped_labels.shape == (8, 8)
    assert len(parsed) == 2


def test_split_derives_bolivia_only_from_official_test_rows() -> None:
    train = "train_a_S1Hand.tif,train_a_LabelHand.tif\n"
    valid = "valid_a_S1Hand.tif,valid_a_LabelHand.tif\n"
    test = (
        "test_a_S1Hand.tif,test_a_LabelHand.tif\n"
        "Bolivia_event_1_S1Hand.tif,Bolivia_event_1_LabelHand.tif\n"
    )

    splits = build_dataset_splits(train, valid, test)

    assert [pair.s1_name for pair in splits.test] == ["test_a_S1Hand.tif"]
    assert [pair.s1_name for pair in splits.bolivia] == ["Bolivia_event_1_S1Hand.tif"]
    assert set(splits.split_hashes) == {"train", "valid", "test"}


def test_official_bolivia_csv_is_used_and_must_not_overlap_other_splits() -> None:
    train = "train_a_S1Hand.tif,train_a_LabelHand.tif\n"
    valid = "valid_a_S1Hand.tif,valid_a_LabelHand.tif\n"
    test = "test_a_S1Hand.tif,test_a_LabelHand.tif\n"
    bolivia = (
        "Bolivia_a_S1Hand.tif,Bolivia_a_LabelHand.tif\n"
        "Bolivia_b_S1Hand.tif,Bolivia_b_LabelHand.tif\n"
    )
    source_url = (
        "https://storage.googleapis.com/sen1floods11/"
        "v1.1/splits/flood_handlabeled/flood_bolivia_data.csv"
    )

    splits = build_dataset_splits(
        train,
        valid,
        test,
        bolivia_csv=bolivia,
        bolivia_source_url=source_url,
    )

    assert splits.bolivia_split_method == "official_bolivia_csv"
    assert splits.bolivia_source_url == source_url
    assert len(splits.bolivia) == 2
    assert len(splits.test) == 1
    assert "bolivia" in splits.split_hashes
    with pytest.raises(ValueError, match="overlap"):
        build_dataset_splits(
            train,
            valid,
            "Bolivia_a_S1Hand.tif,Bolivia_a_LabelHand.tif\n",
            bolivia_csv=bolivia,
            bolivia_source_url=source_url,
        )


def test_run_config_records_official_bolivia_source_hash_and_small_split_warning(
    tmp_path: Path,
) -> None:
    train = "train_a_S1Hand.tif,train_a_LabelHand.tif\n"
    valid = "valid_a_S1Hand.tif,valid_a_LabelHand.tif\n"
    test = "test_a_S1Hand.tif,test_a_LabelHand.tif\n"
    bolivia = "Bolivia_a_S1Hand.tif,Bolivia_a_LabelHand.tif\n"
    source_url = "https://storage.googleapis.com/sen1floods11/flood_bolivia_data.csv"
    splits = build_dataset_splits(
        train,
        valid,
        test,
        bolivia_csv=bolivia,
        bolivia_source_url=source_url,
    )
    split_path = tmp_path / "flood_bolivia_data.csv"
    split_path.write_text(bolivia, encoding="utf-8")
    args = SimpleNamespace(
        dataset="sen1floods11",
        encoder_weights="none",
        max_minutes=240,
    )

    config = run_config(
        args,
        torch.device("cpu"),
        ((-18.0, -24.0), (4.0, 4.0)),
        {"train": 1, "valid": 1, "test": 1, "bolivia": 1},
        splits,
        {"bolivia": split_path},
        False,
    )

    assert config["bolivia_split"]["source_url"] == source_url
    assert config["bolivia_split"]["sha256"] == (
        hashlib.sha256(split_path.read_bytes()).hexdigest()
    )
    assert config["bolivia_split"]["chip_count"] == 1
    assert config["bolivia_split"]["warning"] == (
        "Bolivia split contains only 1 chip; fewer than 5."
    )


def test_cpu_training_loop_reduces_synthetic_loss_in_twenty_iterations() -> None:
    torch.manual_seed(42)
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    features = torch.ones((1, 3, 8, 8), dtype=torch.float32)
    labels = torch.ones((1, 8, 8), dtype=torch.int64)
    labels[:, 0, 0] = IGNORE_INDEX
    batch = (features, labels, torch.zeros((1, 8, 8), dtype=torch.float32))

    losses = [
        train_batch(
            model,
            optimizer,
            batch,
            torch.device("cpu"),
            scaler,
            amp_enabled=False,
        )
        for _ in range(20)
    ]

    assert losses[-1] < losses[0]


def test_training_time_budget_boundary() -> None:
    assert not time_budget_expired(10.0, 1.0, now=69.9)
    assert time_budget_expired(10.0, 1.0, now=70.0)


def test_expired_training_budget_still_writes_non_final_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sahayak.ml.train.build_model",
        lambda _weights: torch.nn.Conv2d(3, 1, kernel_size=1),
    )
    monkeypatch.setattr(
        "sahayak.ml.train.Settings",
        lambda: SimpleNamespace(resolved_data_dir=tmp_path / "data"),
    )
    monkeypatch.setattr("sahayak.ml.train.time_budget_expired", lambda *_args: True)
    output_dir = tmp_path / "outputs"
    args = argparse.Namespace(
        dataset="sen1floods11",
        max_minutes=1.0,
        resume=False,
        encoder_weights="none",
        synthetic=True,
    )

    result = train_model(args, output_dir)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert result == 0
    assert metrics["stopping_reason"] == "max_minutes_before_first_epoch"
    assert metrics["final_metrics"] is False
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "training_log.csv").is_file()
    assert len(list((output_dir / "qualitative").glob("*.png"))) == 8


def test_sen1_dataset_crop_shape_with_synthetic_pair_loader(monkeypatch) -> None:
    image = np.zeros((2, 512, 512), dtype=np.float32)
    labels = np.zeros((512, 512), dtype=np.uint8)
    monkeypatch.setattr(
        "sahayak.ml.data.read_chip",
        lambda *_paths: (image.copy(), labels.copy()),
    )
    dataset = Sen1ChipDataset(
        [(Path("synthetic-s1.tif"), Path("synthetic-label.tif"))],
        (-20.0, -25.0),
        (4.0, 4.0),
        training=True,
        seed=42,
    )

    features, mask, vv = dataset[0]

    assert features.shape == (3, 256, 256)
    assert mask.shape == (256, 256)
    assert vv.shape == (256, 256)
