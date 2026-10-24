"""Train, validate, test, and package the Sentinel-1 flood segmentation model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from sahayak.config import Settings
from sahayak.ml.baselines import fixed_vv_metrics, otsu_vv_metrics
from sahayak.ml.data import (
    BOLIVIA_SPLIT_FILE,
    IGNORE_INDEX,
    SPLIT_FILES,
    DatasetSplits,
    Sen1ChipDataset,
    compute_train_statistics,
    download_sen1floods11,
    download_sen1floods11_splits,
    split_paths,
)
from sahayak.ml.losses import bce_dice_loss
from sahayak.ml.metrics import (
    aggregate_counts,
    bootstrap_intervals,
    chip_metric_distribution,
    confusion_counts,
    metrics_from_counts,
    select_threshold,
)
from sahayak.ml.model import build_model

LOGGER = logging.getLogger(__name__)
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TRAIN_CROP_SIZE = 256
EVALUATION_TILE_SIZE = 256
BOOTSTRAP_RESAMPLES = 1000
SEED = 42
IGNORE_VALUE = IGNORE_INDEX


class SyntheticChipDataset:
    """Small deterministic in-memory dataset for smoke runs only."""

    def __init__(
        self,
        count: int,
        *,
        seed: int,
        mean: tuple[float, float],
        standard_deviation: tuple[float, float],
        training: bool,
    ) -> None:
        self.count = count
        self.rng = np.random.default_rng(seed)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.standard_deviation = np.asarray(standard_deviation, dtype=np.float32)[:, None, None]
        self.training = training
        self.arrays = [self._make_chip(index) for index in range(count)]

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, labels = self.arrays[index]
        image = image.copy()
        labels = labels.copy()
        if self.training:
            from sahayak.ml.data import augment_chip, random_crop

            image, labels = random_crop(
                image,
                labels,
                min(TRAIN_CROP_SIZE, image.shape[1], image.shape[2]),
                self.rng,
            )
            image, labels = augment_chip(image, labels, self.rng)
        normalized = (image - self.mean) / self.standard_deviation
        features = np.concatenate(
            (normalized, normalized[0:1] - normalized[1:2]),
            axis=0,
        ).astype(np.float32)
        return features, labels, image[0].copy()

    def _make_chip(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        height = width = 64
        vv = self.rng.normal(-18.0, 2.0, size=(height, width)).astype(np.float32)
        vh = self.rng.normal(-24.0, 2.0, size=(height, width)).astype(np.float32)
        labels = np.zeros((height, width), dtype=np.uint8)
        top = 18 + index % 5
        left = 20 + index % 7
        labels[top : top + 14, left : left + 18] = 1
        vv[labels == 1] -= 7.0
        vh[labels == 1] -= 6.0
        labels[0, 0] = IGNORE_INDEX
        return np.stack((vv, vh)), labels


def parser() -> argparse.ArgumentParser:
    """Build the model training CLI parser."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--dataset",
        choices=("sen1floods11",),
        default="sen1floods11",
        help="Training dataset (Kuro Siwo lacks a documented sub-10-GB labeled subset).",
    )
    argument_parser.add_argument("--out", type=Path)
    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download split CSVs only and print chip counts without downloading imagery.",
    )
    argument_parser.add_argument("--max-minutes", type=float, default=240.0)
    argument_parser.add_argument("--resume", action="store_true")
    argument_parser.add_argument(
        "--encoder-weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet pretrained ResNet-34 weights or random initialization.",
    )
    argument_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run one-epoch synthetic smoke training; all resulting metrics are non-final.",
    )
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    """Execute one complete training/evaluation/package run."""
    args = parser().parse_args(argv)
    if args.max_minutes <= 0:
        raise SystemExit("--max-minutes must be greater than zero.")
    if args.dry_run:
        settings = Settings()
        _, splits = download_sen1floods11_splits(settings.resolved_data_dir)
        print_split_counts(splits)
        return 0
    if args.out is None:
        parser().error("--out is required unless --dry-run is selected.")
    output_dir = args.out.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return train_model(args, output_dir)


def train_model(args: argparse.Namespace, output_dir: Path) -> int:
    """Train the U-Net, select validation threshold, evaluate, and package results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError('Install the optional extra with `pip install -e ".[ml]"`.') from error

    set_seed(SEED, torch)
    settings = Settings()
    synthetic = bool(args.synthetic)
    if synthetic:
        splits: DatasetSplits | None = None
        split_paths_by_name: dict[str, Path] = {}
        split_pairs: dict[str, Any] = {
            "train": 8,
            "valid": 4,
            "test": 8,
            "bolivia": 4,
        }
        train_mean = (-18.0, -24.0)
        train_std = (4.0, 4.0)
        train_dataset = SyntheticChipDataset(
            split_pairs["train"],
            seed=SEED,
            mean=train_mean,
            standard_deviation=train_std,
            training=True,
        )
        valid_dataset = SyntheticChipDataset(
            split_pairs["valid"],
            seed=SEED + 1,
            mean=train_mean,
            standard_deviation=train_std,
            training=False,
        )
        test_dataset = SyntheticChipDataset(
            split_pairs["test"],
            seed=SEED + 2,
            mean=train_mean,
            standard_deviation=train_std,
            training=False,
        )
        bolivia_dataset = SyntheticChipDataset(
            split_pairs["bolivia"],
            seed=SEED + 3,
            mean=train_mean,
            standard_deviation=train_std,
            training=False,
        )
    else:
        local_paths, splits = download_sen1floods11(settings.resolved_data_dir)
        split_paths_by_name = {
            name: settings.resolved_data_dir / "ml" / "sen1floods11" / "splits" / filename
            for name, filename in {
                **SPLIT_FILES,
                "bolivia": BOLIVIA_SPLIT_FILE,
            }.items()
            if (settings.resolved_data_dir / "ml" / "sen1floods11" / "splits" / filename).is_file()
        }
        train_pairs = split_paths(local_paths, splits.train)
        valid_pairs = split_paths(local_paths, splits.valid)
        test_pairs = split_paths(local_paths, splits.test)
        bolivia_pairs = split_paths(local_paths, splits.bolivia)
        train_mean, train_std = compute_train_statistics(train_pairs)
        train_dataset = Sen1ChipDataset(
            train_pairs, train_mean, train_std, training=True, seed=SEED
        )
        valid_dataset = Sen1ChipDataset(
            valid_pairs, train_mean, train_std, training=False, seed=SEED
        )
        test_dataset = Sen1ChipDataset(
            test_pairs, train_mean, train_std, training=False, seed=SEED
        )
        bolivia_dataset = Sen1ChipDataset(
            bolivia_pairs, train_mean, train_std, training=False, seed=SEED
        )
        split_pairs = {
            "train": len(train_pairs),
            "valid": len(valid_pairs),
            "test": len(test_pairs),
            "bolivia": len(bolivia_pairs),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(None if args.encoder_weights == "none" else args.encoder_weights)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=amp_enabled,
    )

    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    log_path = output_dir / "training_log.csv"
    start_epoch = 0
    best_iou = -1.0
    best_threshold = 0.5
    no_improvement = 0
    log_rows: list[dict[str, Any]] = []
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError("--resume was set but last.pt does not exist in --out.")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        best_iou = float(checkpoint["best_iou"])
        best_threshold = float(checkpoint["best_threshold"])
        no_improvement = int(checkpoint["no_improvement"])
        log_rows = list(checkpoint.get("log_rows", []))
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"])
        if checkpoint.get("train_dataset_rng") is not None:
            train_dataset.rng.bit_generator.state = checkpoint["train_dataset_rng"]
        if torch.cuda.is_available() and checkpoint.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        LOGGER.info("Resuming at epoch %d", start_epoch + 1)

    run_started = time.monotonic()
    stop_reason = "max_epochs"
    epochs_to_run = 1 if synthetic else MAX_EPOCHS
    for epoch in range(start_epoch, epochs_to_run):
        if time_budget_expired(run_started, args.max_minutes):
            stop_reason = "max_minutes"
            break
        model.train()
        train_loss_sum = 0.0
        batch_count = 0
        for batch in loader:
            loss_value = train_batch(
                model,
                optimizer,
                batch,
                device,
                scaler,
                amp_enabled=amp_enabled,
            )
            train_loss_sum += loss_value
            batch_count += 1
        if batch_count == 0:
            stop_reason = "max_minutes"
            break

        scheduler.step()
        valid_arrays = evaluate_model(model, valid_dataset, device, torch)
        threshold = select_threshold(valid_arrays["probabilities"], valid_arrays["labels"])
        valid_counts = [
            confusion_counts(probability >= threshold, labels)
            for probability, labels in zip(
                valid_arrays["probabilities"],
                valid_arrays["labels"],
                strict=True,
            )
        ]
        valid_metrics = metrics_from_counts(aggregate_counts(valid_counts))
        valid_loss = evaluate_loss(model, valid_dataset, device, torch, amp_enabled)
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / batch_count,
            "valid_loss": valid_loss,
            "valid_threshold": threshold,
            "valid_flood_iou": valid_metrics["flood_iou"],
            "valid_f1": valid_metrics["f1"],
            "elapsed_minutes": (time.monotonic() - run_started) / 60.0,
        }
        log_rows.append(epoch_row)
        write_training_log(log_path, log_rows)

        improved = float(valid_metrics["flood_iou"]) > best_iou
        if improved:
            best_iou = float(valid_metrics["flood_iou"])
            best_threshold = threshold
            no_improvement = 0
            save_checkpoint(
                best_path,
                _checkpoint_state(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    best_iou,
                    best_threshold,
                    no_improvement,
                    log_rows,
                    train_dataset.rng.bit_generator.state,
                ),
            )
        else:
            no_improvement += 1

        save_checkpoint(
            last_path,
            _checkpoint_state(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                best_iou,
                best_threshold,
                no_improvement,
                log_rows,
                train_dataset.rng.bit_generator.state,
            ),
        )
        LOGGER.info(
            "epoch=%d train_loss=%.6f valid_iou=%.6f threshold=%.3f",
            epoch + 1,
            epoch_row["train_loss"],
            valid_metrics["flood_iou"],
            threshold,
        )
        if synthetic:
            stop_reason = "synthetic_smoke_one_epoch"
            break
        if no_improvement >= EARLY_STOPPING_PATIENCE:
            stop_reason = "early_stopping"
            break
        if time_budget_expired(run_started, args.max_minutes):
            stop_reason = "max_minutes"
            break

    if not log_rows:
        initial_state = _checkpoint_state(
            model,
            optimizer,
            scheduler,
            scaler,
            start_epoch,
            best_iou,
            best_threshold,
            no_improvement,
            log_rows,
            train_dataset.rng.bit_generator.state,
        )
        if not (args.resume and best_path.is_file()):
            save_checkpoint(best_path, initial_state)
        if not args.resume:
            save_checkpoint(last_path, initial_state)
        best_threshold = 0.5
        stop_reason = "max_minutes_before_first_epoch"
    write_training_log(log_path, log_rows)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    best_threshold = float(best_checkpoint.get("best_threshold", best_threshold))
    validation = evaluate_model(model, valid_dataset, device, torch)
    best_threshold = select_threshold(validation["probabilities"], validation["labels"])
    test = evaluate_model(model, test_dataset, device, torch)
    bolivia = evaluate_model(model, bolivia_dataset, device, torch)
    metrics = {
        "dataset": args.dataset,
        "synthetic": synthetic,
        "final_metrics": not synthetic and bool(log_rows),
        "status": (
            "synthetic smoke test; not final performance metrics"
            if synthetic
            else (
                "evaluated on official test and derived held-out Bolivia chips"
                if log_rows
                else "no epoch completed; scores are from untrained initialization"
            )
        ),
        "device": str(device),
        "stopping_reason": stop_reason,
        "elapsed_minutes": (time.monotonic() - run_started) / 60.0,
        "validation_threshold": best_threshold,
        "validation": _evaluate_output(validation, best_threshold),
        "test": _evaluate_output(test, best_threshold),
        "held_out_bolivia": _evaluate_output(bolivia, best_threshold),
        "baselines": {
            "test": evaluate_baselines(test),
            "held_out_bolivia": evaluate_baselines(bolivia),
        },
        "bootstrap": {
            "test_model": bootstrap_intervals(
                _counts_at_threshold(test, best_threshold),
                resamples=BOOTSTRAP_RESAMPLES,
                seed=SEED,
            ),
            "bolivia_model": bootstrap_intervals(
                _counts_at_threshold(bolivia, best_threshold),
                resamples=BOOTSTRAP_RESAMPLES,
                seed=SEED,
            ),
            "test_otsu": bootstrap_intervals(
                test["otsu_counts"],
                resamples=BOOTSTRAP_RESAMPLES,
                seed=SEED,
            ),
            "test_fixed_minus_15_db": bootstrap_intervals(
                test["fixed_minus_15_db_counts"],
                resamples=BOOTSTRAP_RESAMPLES,
                seed=SEED,
            ),
            "bolivia_otsu": bootstrap_intervals(
                bolivia["otsu_counts"],
                resamples=BOOTSTRAP_RESAMPLES,
                seed=SEED,
            ),
            "bolivia_fixed_minus_15_db": bootstrap_intervals(
                bolivia["fixed_minus_15_db_counts"],
                resamples=BOOTSTRAP_RESAMPLES,
                seed=SEED,
            ),
        },
    }
    config = run_config(
        args,
        device,
        (train_mean, train_std),
        split_pairs,
        splits,
        split_paths_by_name,
        synthetic,
    )
    config["final_metrics"] = metrics["final_metrics"]
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "config.json", config)
    model_card = render_model_card(metrics, config)
    (output_dir / "model_card.md").write_text(model_card, encoding="utf-8")
    qualitative_items = attach_qualitative(test, best_threshold)
    if len(qualitative_items) < 8:
        qualitative_items.extend(attach_qualitative(bolivia, best_threshold))
    if len(qualitative_items) < 8:
        raise ValueError("At least eight held-out chips are required for qualitative outputs.")
    write_qualitative_images(output_dir / "qualitative", qualitative_items[:8])
    LOGGER.info("Outputs written to %s", output_dir)
    return 0


def train_batch(
    model: Any,
    optimizer: Any,
    batch: tuple[Any, Any, Any],
    device: Any,
    scaler: Any,
    *,
    amp_enabled: bool,
) -> float:
    """Run one optimizer step and return the measured combined loss."""
    import torch

    features, labels, _ = batch
    features = features.to(device=device, dtype=torch.float32, non_blocking=True)
    labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        logits = model(features)
        loss = bce_dice_loss(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().item())


def evaluate_model(model: Any, dataset: Any, device: Any, torch_module: Any) -> dict[str, Any]:
    """Run tiled full-chip inference and collect probabilities, labels, and raw VV."""
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    vv_chips: list[np.ndarray] = []
    with torch_module.no_grad():
        for index in range(len(dataset)):
            features, target, vv_db = dataset[index]
            probability = predict_tiled(model, features, device, torch_module)
            probabilities.append(probability)
            labels.append(np.asarray(target, dtype=np.uint8))
            vv_chips.append(np.asarray(vv_db, dtype=np.float32))
    return {
        "probabilities": probabilities,
        "labels": labels,
        "vv_db": vv_chips,
    }


def predict_tiled(
    model: Any,
    features: np.ndarray,
    device: Any,
    torch_module: Any,
    tile_size: int = EVALUATION_TILE_SIZE,
) -> np.ndarray:
    """Infer over the full chip in non-overlapping tiles and reassemble probabilities."""
    if features.ndim != 3 or features.shape[0] != 3:
        raise ValueError("Model input must have shape (3, height, width).")
    if tile_size < 1:
        raise ValueError("Evaluation tile size must be positive.")
    _, height, width = features.shape
    output = np.empty((height, width), dtype=np.float32)
    model.eval()
    with torch_module.no_grad():
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)
                tensor = torch_module.from_numpy(
                    np.ascontiguousarray(features[:, top:bottom, left:right])
                ).unsqueeze(0).to(device)
                logits = model(tensor)
                output[top:bottom, left:right] = (
                    torch_module.sigmoid(logits[0, 0]).detach().cpu().numpy()
                )
    return output


def evaluate_loss(model: Any, dataset: Any, device: Any, torch_module: Any, amp: bool) -> float:
    """Measure mean validation loss over complete chips."""
    model.eval()
    losses = []
    with torch_module.no_grad():
        for index in range(len(dataset)):
            features, labels, _ = dataset[index]
            input_tensor = torch_module.from_numpy(features).unsqueeze(0).to(device)
            target_tensor = torch_module.from_numpy(labels.astype(np.int64)).unsqueeze(0).to(device)
            with torch_module.autocast(device_type=device.type, enabled=amp):
                logits = model(input_tensor)
                loss = bce_dice_loss(logits, target_tensor)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def _evaluate_output(evaluation: dict[str, Any], threshold: float) -> dict[str, Any]:
    counts = _counts_at_threshold(evaluation, threshold)
    aggregate = aggregate_counts(counts)
    result = {
        **metrics_from_counts(aggregate),
        "per_chip": chip_metric_distribution(counts),
        "chip_count": len(counts),
    }
    return result


def _counts_at_threshold(
    evaluation: dict[str, Any],
    threshold: float,
) -> list[dict[str, int]]:
    return [
        confusion_counts(probability >= threshold, labels)
        for probability, labels in zip(
            evaluation["probabilities"],
            evaluation["labels"],
            strict=True,
        )
    ]


def evaluate_baselines(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Evaluate per-chip Otsu and fixed -15 dB baselines."""
    outputs: dict[str, Any] = {}
    for method in ("otsu", "fixed_minus_15_db"):
        rows = []
        for vv, labels in zip(evaluation["vv_db"], evaluation["labels"], strict=True):
            result = (
                otsu_vv_metrics(vv, labels)
                if method == "otsu"
                else fixed_vv_metrics(vv, labels)
            )
            rows.append(result["counts"])
        outputs[method] = {
            **metrics_from_counts(aggregate_counts(rows)),
            "per_chip": chip_metric_distribution(rows),
            "chip_count": len(rows),
        }
        evaluation[f"{method}_counts"] = rows
    return outputs


def time_budget_expired(started_at: float, max_minutes: float, now: float | None = None) -> bool:
    """Return whether the elapsed monotonic time reached the training budget."""
    current = time.monotonic() if now is None else now
    return current - started_at >= max_minutes * 60.0


def write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the measured per-epoch training history as CSV."""
    fields = (
        "epoch",
        "train_loss",
        "valid_loss",
        "valid_threshold",
        "valid_flood_iou",
        "valid_f1",
        "elapsed_minutes",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_state(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_iou: float,
    best_threshold: float,
    no_improvement: int,
    log_rows: list[dict[str, Any]],
    train_dataset_rng: dict[str, Any],
) -> dict[str, Any]:
    """Capture all state required for resume and reproducibility."""
    import torch

    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_iou": best_iou,
        "best_threshold": best_threshold,
        "no_improvement": no_improvement,
        "log_rows": log_rows,
        "train_dataset_rng": train_dataset_rng,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically write one model checkpoint."""
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_config(
    args: argparse.Namespace,
    device: Any,
    normalization: tuple[tuple[float, float], tuple[float, float]],
    split_sizes: dict[str, int],
    splits: DatasetSplits | None,
    split_paths_by_name: dict[str, Path],
    synthetic: bool,
) -> dict[str, Any]:
    """Build run metadata without implying that synthetic smoke metrics are final."""
    import torch

    versions = {}
    for package_name in (
        "torch",
        "segmentation-models-pytorch",
        "numpy",
        "rasterio",
        "pandas",
        "tqdm",
        "matplotlib",
        "huggingface_hub",
    ):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    try:
        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit_hash = None
    return {
        "dataset": args.dataset,
        "synthetic": synthetic,
        "final_metrics": not synthetic,
        "hyperparameters": {
            "architecture": "segmentation_models_pytorch.Unet",
            "encoder": "resnet34",
            "encoder_weights": args.encoder_weights,
            "input_channels": ["VV", "VH", "VV_minus_VH"],
            "output_logits": 1,
            "loss": "BCEWithLogits + soft Dice; ignore -1/255",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "max_minutes": args.max_minutes,
            "training_crop_size": TRAIN_CROP_SIZE,
            "evaluation_tile_size": EVALUATION_TILE_SIZE,
            "amp_enabled": device.type == "cuda",
            "seed": SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "threshold_grid_step": 1 / 1000,
        },
        "normalization": {
            "source": "training split only" if not synthetic else "synthetic training split only",
            "VV_mean_db": normalization[0][0],
            "VH_mean_db": normalization[0][1],
            "VV_std_db": normalization[1][0],
            "VH_std_db": normalization[1][1],
            "clip_db": [-30.0, 5.0],
        },
        "split_sizes_chips": split_sizes,
        "split_file_sha256": (
            {name: path_sha256(path) for name, path in split_paths_by_name.items()}
            if split_paths_by_name
            else {}
        ),
        "split_content_sha256": splits.split_hashes if splits is not None else {},
        "bolivia_split_definition": (
            splits.bolivia_split_method if splits is not None else "synthetic_smoke"
        ),
        "bolivia_split": (
            {
                "method": splits.bolivia_split_method,
                "source_url": splits.bolivia_source_url,
                "sha256": (
                    path_sha256(split_paths_by_name["bolivia"])
                    if "bolivia" in split_paths_by_name
                    else splits.split_hashes.get("bolivia")
                ),
                "chip_count": len(splits.bolivia),
                "warning": (
                    _small_split_warning(len(splits.bolivia))
                    if len(splits.bolivia) < 5
                    else None
                ),
            }
            if splits is not None
            else {
                "method": "synthetic_smoke",
                "source_url": None,
                "sha256": None,
                "chip_count": split_sizes["bolivia"],
                "warning": None,
            }
        ),
        "library_versions": versions,
        "git_commit": commit_hash,
        "device": str(device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def print_split_counts(splits: DatasetSplits) -> None:
    """Print split chip counts without reading or downloading raster imagery."""
    counts = {
        "train": len(splits.train),
        "valid": len(splits.valid),
        "test": len(splits.test),
        "Bolivia": len(splits.bolivia),
    }
    print("Sen1Floods11 split dry run (CSV files only; imagery not downloaded)")
    print(f"Bolivia split method: {splits.bolivia_split_method}")
    if splits.bolivia_source_url:
        print(f"Bolivia source: {splits.bolivia_source_url}")
    print(f"{'Split':<12} | Chip count")
    print("-" * 25)
    for name, count in counts.items():
        print(f"{name:<12} | {count}")
    if counts["Bolivia"] < 5:
        print(f"WARNING: {_small_split_warning(counts['Bolivia'])}")


def _small_split_warning(count: int) -> str:
    noun = "chip" if count == 1 else "chips"
    return f"Bolivia split contains only {count} {noun}; fewer than 5."


def write_json(path: Path, value: Any) -> None:
    """Write reproducibility and metric JSON files."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def path_sha256(path: Path) -> str:
    """Calculate SHA-256 for a downloaded split file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int, torch_module: Any) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def render_model_card(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    """Render a model card that flags synthetic runs and scientific limitations."""
    if metrics["synthetic"]:
        results = (
            "This is a synthetic CPU smoke run. Its values are not final performance "
            "metrics and must not be cited as model evaluation."
        )
    elif not metrics["final_metrics"]:
        results = (
            "The time budget expired before an epoch completed. These scores are from "
            "untrained initialization and are not final performance metrics."
        )
    else:
        test = metrics["test"]
        bolivia = metrics["held_out_bolivia"]
        results = (
            f"Test flood IoU: {test['flood_iou']:.4f}; F1: {test['f1']:.4f}; "
            f"precision: {test['precision']:.4f}; recall: {test['recall']:.4f}; "
            f"accuracy: {test['accuracy']:.4f}.\n\n"
            f"Held-out Bolivia flood IoU: {bolivia['flood_iou']:.4f}; "
            f"F1: {bolivia['f1']:.4f}; precision: {bolivia['precision']:.4f}; "
            f"recall: {bolivia['recall']:.4f}; accuracy: {bolivia['accuracy']:.4f}."
        )
    return f"""---
library_name: pytorch
tags:
- flood-mapping
- sentinel-1
- remote-sensing
---

# Sahayak Sentinel-1 Flood U-Net

## Intended use

Educational research prototype for screening open-water flood extent from paired
Sentinel-1 VV/VH measurements. It is not an operational rescue product or a debris-flow
damage detector.

## Training data

Sen1Floods11 hand-labeled Sentinel-1 GRD chips (`S1Hand`) and labels (`LabelHand`) from
the official split CSVs. The separate official Bolivia CSV is used as the geographic
holdout when available; otherwise Bolivia-prefixed chips in the official test CSV form
the holdout. Split overlap is rejected.
Citation: Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020),
"Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms
for Sentinel-1," CVPR Workshops 2020, pp. 210–211.

License note: the upstream README identifies the dataset and provides a public bucket,
but the README page consulted for this run does not state its license. Mirrors describe
CC BY 4.0; verify the applicable upstream terms before redistribution.

## Evaluation

{results}

Threshold is chosen on the validation split only. The output `metrics.json` includes
test and held-out Bolivia metrics and VV-only Otsu/fixed-threshold baselines on the same
labeled pixels. Synthetic smoke runs set `final_metrics` to false.

## Configuration and training metadata

See `config.json` for the measured split sizes, CSV hashes, normalization statistics,
software versions, commit identifier, and hyperparameters.

## Limitations

- Labels represent open-water flood extents; open-water river floods are not debris flows.
- Sigma0 versus gamma0 and preprocessing differences can shift backscatter distributions.
- Mountain shadow and layover can obscure or alter radar returns.
- Sen1Floods11 chips are 10 m GRD; narrow channels and fine-scale damage may be unresolved.
- Transfer to the Himalaya is unverified; this model is not ground-truth validation.
"""


def write_qualitative_images(output_dir: Path, items: list[dict[str, Any]]) -> None:
    """Write no-person VV/label/prediction PNG triptychs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, start=1):
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(item["vv_db"], cmap="gray", vmin=-30, vmax=5)
        axes[0].set_title("Sentinel-1 VV (dB)")
        label_image = np.ma.masked_where(
            np.isin(item["labels"], (-1, IGNORE_INDEX)),
            item["labels"],
        )
        axes[1].imshow(
            label_image,
            cmap=ListedColormap(["#202020", "#1976d2"]),
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Hand label")
        axes[2].imshow(
            item["probability"] >= item["threshold"],
            cmap=ListedColormap(["#202020", "#e53935"]),
            vmin=0,
            vmax=1,
        )
        axes[2].set_title(f"Prediction (threshold {item['threshold']:.3f})")
        for axis in axes:
            axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_dir / f"qualitative_{index:02d}.png", dpi=140)
        plt.close(figure)


def attach_qualitative(
    evaluation: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    """Prepare qualitative records from already-evaluated chips."""
    return [
        {
            "vv_db": vv,
            "labels": labels,
            "probability": probability,
            "threshold": threshold,
        }
        for vv, labels, probability in zip(
            evaluation["vv_db"],
            evaluation["labels"],
            evaluation["probabilities"],
            strict=True,
        )
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
