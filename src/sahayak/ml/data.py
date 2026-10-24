"""Sen1Floods11 hand-labeled Sentinel-1 dataset access and preprocessing."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
BUCKET = "sen1floods11"
BUCKET_PREFIX = "v1.1"
SPLIT_PREFIX = f"{BUCKET_PREFIX}/splits/flood_handlabeled"
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
}
BOLIVIA_SPLIT_FILE = "flood_bolivia_data.csv"
SPLIT_SOURCE_BASE_URL = (
    "https://storage.googleapis.com/sen1floods11/v1.1/splits/flood_handlabeled"
)
DATA_PREFIX = f"{BUCKET_PREFIX}/data/flood_events/HandLabeled"
S1_PREFIX = f"{DATA_PREFIX}/S1Hand"
LABEL_PREFIX = f"{DATA_PREFIX}/LabelHand"
IGNORE_INDEX = 255
CHIP_SIZE = 512
CHANNEL_NAMES = ("VV", "VH", "VV_minus_VH")
RESAMPLING_NOT_USED = "Original 512x512 georeferenced chips are retained without resampling."


@dataclass(frozen=True)
class ChipPair:
    """A paired VV/VH chip and hand-labeled mask."""

    s1_name: str
    label_name: str


@dataclass(frozen=True)
class DatasetSplits:
    """Leakage-checked training, validation, test, and Bolivia chip lists."""

    train: tuple[ChipPair, ...]
    valid: tuple[ChipPair, ...]
    test: tuple[ChipPair, ...]
    bolivia: tuple[ChipPair, ...]
    split_hashes: dict[str, str]
    bolivia_split_method: str
    bolivia_source_url: str | None


def parse_split_csv(text: str) -> list[ChipPair]:
    """Parse official two-column S1/label CSV rows, with or without a header."""
    rows = list(csv.reader(text.splitlines()))
    pairs: list[ChipPair] = []
    for row_number, row in enumerate(rows, start=1):
        values = [value.strip() for value in row if value.strip()]
        if len(values) < 2:
            if values:
                raise ValueError(f"Split CSV row {row_number} must contain S1 and label paths.")
            continue
        s1_name = Path(values[0].replace("\\", "/")).name
        label_name = Path(values[1].replace("\\", "/")).name
        if row_number == 1 and not (
            s1_name.endswith("_S1Hand.tif") and label_name.endswith("_LabelHand.tif")
        ):
            continue
        if not s1_name.endswith("_S1Hand.tif") or not label_name.endswith("_LabelHand.tif"):
            raise ValueError(
                f"Split CSV row {row_number} must pair *_S1Hand.tif with *_LabelHand.tif."
            )
        if s1_name.removesuffix("_S1Hand.tif") != label_name.removesuffix("_LabelHand.tif"):
            raise ValueError(f"Split CSV row {row_number} contains mismatched chip identifiers.")
        pairs.append(ChipPair(s1_name=s1_name, label_name=label_name))
    if not pairs:
        raise ValueError("Split CSV contains no hand-labeled Sentinel-1 chip pairs.")
    if len({pair.s1_name for pair in pairs}) != len(pairs):
        raise ValueError("Split CSV contains duplicate Sentinel-1 chip identifiers.")
    return pairs


def build_dataset_splits(
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    bolivia_csv: str | None = None,
    bolivia_source_url: str | None = None,
) -> DatasetSplits:
    """Parse dataset splits using the official Bolivia CSV when supplied."""
    train = parse_split_csv(train_csv)
    valid = parse_split_csv(valid_csv)
    official_test = parse_split_csv(test_csv)
    train_ids = {pair.s1_name for pair in train}
    valid_ids = {pair.s1_name for pair in valid}
    if train_ids & valid_ids:
        raise ValueError("Official train and validation splits overlap.")

    def is_bolivia(pair: ChipPair) -> bool:
        return pair.s1_name.casefold().startswith("bolivia_")

    if bolivia_csv is not None:
        bolivia = tuple(parse_split_csv(bolivia_csv))
        split_method = "official_bolivia_csv"
        if not bolivia_source_url:
            raise ValueError("The official Bolivia CSV source URL must be recorded.")
        if any(not is_bolivia(pair) for pair in bolivia):
            raise ValueError("The official Bolivia CSV contains a non-Bolivia chip identifier.")
        ordinary_test = tuple(official_test)
    else:
        bolivia = tuple(pair for pair in official_test if is_bolivia(pair))
        split_method = "derived_from_official_test_csv"
        ordinary_test = tuple(pair for pair in official_test if not is_bolivia(pair))
        if not bolivia:
            raise ValueError(
                "No official Bolivia CSV or Bolivia-named chips exist in the test split."
            )
    if any(is_bolivia(pair) for pair in (*train, *valid)):
        raise ValueError(
            "Bolivia chips occur in official training or validation; refusing geographic leakage."
        )
    if not ordinary_test:
        raise ValueError("The official test CSV contains no chips for evaluation.")
    all_splits = [train, valid, ordinary_test, bolivia]
    identifiers = [{pair.s1_name for pair in split} for split in all_splits]
    for first_index, first in enumerate(identifiers):
        for second in identifiers[first_index + 1 :]:
            if first & second:
                raise ValueError("Dataset splits overlap after deriving Bolivia holdout.")
    hashes = {
        "train": hashlib.sha256(train_csv.encode("utf-8")).hexdigest(),
        "valid": hashlib.sha256(valid_csv.encode("utf-8")).hexdigest(),
        "test": hashlib.sha256(test_csv.encode("utf-8")).hexdigest(),
    }
    if bolivia_csv is not None:
        hashes["bolivia"] = hashlib.sha256(bolivia_csv.encode("utf-8")).hexdigest()
    return DatasetSplits(
        tuple(train),
        tuple(valid),
        ordinary_test,
        bolivia,
        hashes,
        split_method,
        bolivia_source_url,
    )


def list_public_objects(prefix: str) -> list[dict[str, Any]]:
    """List public objects in a GCS prefix using its anonymous JSON API."""
    objects: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        query: dict[str, str] = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            query["pageToken"] = page_token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
            f"{urllib.parse.urlencode(query)}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "sahayak-ml/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise ValueError("Public Sen1Floods11 object listing returned an invalid response.")
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                objects.append(item)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return objects


def download_public_object(
    object_name: str,
    destination: Path,
    expected_size: int,
) -> None:
    """Download one public GCS object over HTTPS, resuming and checking its size."""
    if expected_size < 0:
        raise ValueError("Expected object size cannot be negative.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    encoded_name = urllib.parse.quote(object_name, safe="/")
    url = f"https://storage.googleapis.com/{BUCKET}/{encoded_name}"
    headers = {"User-Agent": "sahayak-ml/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset and response.status == 206:
                mode = "ab"
                start = offset
            else:
                mode = "wb"
                start = 0
            received = start
            with partial.open(mode) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    received += len(chunk)
            if received != expected_size:
                raise OSError(
                    f"Downloaded {object_name} has {received} bytes; expected {expected_size}."
                )
        os.replace(partial, destination)
    except (urllib.error.URLError, TimeoutError, OSError):
        LOGGER.exception("Failed downloading Sen1Floods11 object %s", object_name)
        raise


def download_sen1floods11_splits(
    data_root: Path,
) -> tuple[dict[str, Path], DatasetSplits]:
    """Download only split CSVs and return the parsed chip membership."""
    dataset_root = data_root / "ml" / "sen1floods11"
    split_root = dataset_root / "splits"
    split_paths: dict[str, Path] = {}
    source_files = {**SPLIT_FILES, "bolivia": BOLIVIA_SPLIT_FILE}
    for split, filename in source_files.items():
        object_name = f"{SPLIT_PREFIX}/{filename}"
        listed = list_public_objects(object_name)
        match = next((item for item in listed if item.get("name") == object_name), None)
        if match is None:
            if split == "bolivia":
                continue
            raise FileNotFoundError(
                f"Sen1Floods11 bucket does not list required split {filename}."
            )
        path = split_root / filename
        download_public_object(object_name, path, int(match["size"]))
        split_paths[split] = path

    splits = build_dataset_splits(
        *(split_paths[name].read_text(encoding="utf-8-sig") for name in ("train", "valid", "test")),
        (
            split_paths["bolivia"].read_text(encoding="utf-8-sig")
            if "bolivia" in split_paths
            else None
        ),
        (
            f"{SPLIT_SOURCE_BASE_URL}/{BOLIVIA_SPLIT_FILE}"
            if "bolivia" in split_paths
            else None
        ),
    )
    return split_paths, splits


def download_sen1floods11(data_root: Path) -> tuple[dict[str, Path], DatasetSplits]:
    """Download split CSVs and only the hand-labeled S1 chips and labels they reference."""
    split_paths, splits = download_sen1floods11_splits(data_root)
    dataset_root = data_root / "ml" / "sen1floods11"
    requested_pairs = {
        pair.s1_name: pair
        for split in (splits.train, splits.valid, splits.test, splits.bolivia)
        for pair in split
    }
    local_paths: dict[str, Path] = {}
    for prefix, suffix, member in (
        (S1_PREFIX, "_S1Hand.tif", "s1"),
        (LABEL_PREFIX, "_LabelHand.tif", "label"),
    ):
        objects = list_public_objects(f"{prefix}/")
        object_by_basename = {
            Path(str(item["name"])).name: item
            for item in objects
            if Path(str(item.get("name", ""))).name.endswith(suffix)
        }
        for chip in requested_pairs.values():
            name = chip.s1_name if member == "s1" else chip.label_name
            item = object_by_basename.get(name)
            if item is None:
                raise FileNotFoundError(f"Sen1Floods11 bucket is missing requested chip {name}.")
            path = dataset_root / ("S1Hand" if member == "s1" else "LabelHand") / name
            download_public_object(str(item["name"]), path, int(item["size"]))
            local_paths[name] = path
    return local_paths, splits


def split_paths(
    local_paths: dict[str, Path],
    split: tuple[ChipPair, ...],
) -> list[tuple[Path, Path]]:
    """Resolve split entries to the downloaded raster file paths."""
    result = []
    for pair in split:
        try:
            result.append((local_paths[pair.s1_name], local_paths[pair.label_name]))
        except KeyError as error:
            raise FileNotFoundError(f"Dataset chip {error.args[0]} was not downloaded.") from error
    return result


def compute_train_statistics(
    pairs: list[tuple[Path, Path]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute VV/VH dB means and standard deviations using train chips only."""
    sums = np.zeros(2, dtype=np.float64)
    squared_sums = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for s1_path, label_path in pairs:
        image, labels = read_chip(s1_path, label_path)
        valid = (labels != IGNORE_INDEX) & np.all(np.isfinite(image), axis=0)
        for band in range(2):
            samples = image[band][valid].astype(np.float64)
            sums[band] += samples.sum()
            squared_sums[band] += np.square(samples).sum()
            counts[band] += samples.size
    if np.any(counts == 0):
        raise ValueError(
            "Cannot compute train normalization: one or more bands have no valid samples."
        )
    means = sums / counts
    variances = np.maximum(squared_sums / counts - np.square(means), 0.0)
    standard_deviations = np.sqrt(variances)
    standard_deviations[standard_deviations < 1e-6] = 1.0
    return tuple(map(float, means)), tuple(map(float, standard_deviations))


def read_chip(s1_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate one two-band VV/VH chip and one hand-labeled mask."""
    import rasterio

    with rasterio.open(s1_path) as image_source:
        if image_source.count != 2:
            raise ValueError(f"{s1_path.name} has {image_source.count} bands; expected VV then VH.")
        image = image_source.read(out_dtype="float32")
    with rasterio.open(label_path) as label_source:
        if label_source.count != 1:
            raise ValueError(
                f"{label_path.name} has {label_source.count} bands; expected one label band."
            )
        labels = label_source.read(1)
    if image.shape[1:] != labels.shape:
        raise ValueError(f"S1 chip {s1_path.name} and label {label_path.name} do not align.")
    invalid_labels = ~np.isin(labels, (-1, 0, 1))
    if invalid_labels.any():
        values = np.unique(labels[invalid_labels]).tolist()
        raise ValueError(f"{label_path.name} contains undocumented label values: {values}.")
    finite_pixels = np.all(np.isfinite(image), axis=0)
    image = np.clip(image, -30.0, 5.0)
    labels = np.where(labels == -1, IGNORE_INDEX, labels).astype(np.uint8)
    labels[~finite_pixels] = IGNORE_INDEX
    return image, labels


class Sen1ChipDataset:
    """Raster-backed dataset with train-only crop and geometric/gain augmentation."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        mean: tuple[float, float],
        standard_deviation: tuple[float, float],
        *,
        training: bool,
        crop_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.pairs = pairs
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.standard_deviation = np.asarray(standard_deviation, dtype=np.float32)[:, None, None]
        self.training = training
        self.crop_size = crop_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, labels = read_chip(*self.pairs[index])
        if self.training:
            image, labels = random_crop(image, labels, self.crop_size, self.rng)
            image, labels = augment_chip(image, labels, self.rng)
        else:
            image, labels = evaluation_chip(image, labels)
        normalized = (image - self.mean) / self.standard_deviation
        normalized[~np.isfinite(normalized)] = 0.0
        features = np.concatenate(
            (normalized, (normalized[0:1] - normalized[1:2])),
            axis=0,
        ).astype(np.float32)
        return features, labels, image[0].copy()


def random_crop(
    image: np.ndarray,
    labels: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a random square training crop."""
    if crop_size <= 0 or image.shape[1] < crop_size or image.shape[2] < crop_size:
        raise ValueError(
            f"Crop size {crop_size} exceeds chip dimensions {image.shape[2]}x{image.shape[1]}."
        )
    top = int(rng.integers(0, image.shape[1] - crop_size + 1))
    left = int(rng.integers(0, image.shape[2] - crop_size + 1))
    return (
        image[:, top : top + crop_size, left : left + crop_size],
        labels[top : top + crop_size, left : left + crop_size],
    )


def evaluation_chip(
    image: np.ndarray,
    labels: np.ndarray,
    tile_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return full-chip arrays; model inference tiles them later."""
    if tile_size <= 0:
        raise ValueError("Evaluation tile size must be positive.")
    if image.shape[1:] != labels.shape:
        raise ValueError("Evaluation image and label dimensions differ.")
    return image, labels


def augment_chip(
    image: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply flips, right-angle rotations, and shared +/-1 dB S1 gain jitter."""
    if rng.random() < 0.5:
        image = np.flip(image, axis=2)
        labels = np.flip(labels, axis=1)
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        labels = np.flip(labels, axis=0)
    rotations = int(rng.integers(0, 4))
    image = np.rot90(image, rotations, axes=(1, 2))
    labels = np.rot90(labels, rotations, axes=(0, 1))
    gain = float(rng.uniform(-1.0, 1.0))
    image = np.clip(image + gain, -30.0, 5.0).copy()
    return image, labels.copy()


def split_size_bytes(paths: dict[str, Path]) -> dict[str, int]:
    """Return exact on-disk sizes for the three split CSVs."""
    return {name: path.stat().st_size for name, path in paths.items()}


def sha256_file(path: Path) -> str:
    """Hash a split file for reproducible run metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gsutil_available() -> bool:
    """Report whether optional gsutil is installed without requiring it."""
    import shutil

    return shutil.which("gsutil") is not None
