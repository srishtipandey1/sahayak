# Sentinel-1 flood segmentation

Sahayak includes an optional PyTorch training pipeline for an educational
Sentinel-1 open-water flood segmentation model. It is not an operational product,
does not map debris flows, and is not evidence of Himalayan performance.

## Dataset and provenance

The primary dataset is Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020),
available from the public Google Cloud Storage bucket `gs://sen1floods11/`.
The upstream v1.1 hand-labeled layout used here is:

- `v1.1/data/flood_events/HandLabeled/S1Hand/`: paired two-band Sentinel-1 chips.
- `v1.1/data/flood_events/HandLabeled/LabelHand/`: corresponding hand-labeled masks.
- `v1.1/splits/flood_handlabeled/flood_train_data.csv`
- `v1.1/splits/flood_handlabeled/flood_valid_data.csv`
- `v1.1/splits/flood_handlabeled/flood_test_data.csv`

The official training notebook uses the three split CSVs as paired S1 and label
paths. The notebook does not list a Bolivia split, but the bucket also contains
the official `flood_bolivia_data.csv` file. The implementation uses this
separate holdout rather than deriving Bolivia chips from the test list. It
checks that the Bolivia CSV has no overlap with train, validation, or test, and
rejects Bolivia-prefixed chips in training or validation.

Verified Bolivia split:

- Source: <https://storage.googleapis.com/sen1floods11/v1.1/splits/flood_handlabeled/flood_bolivia_data.csv>
- SHA-256: `4775d100fae1f1ca66a842b8af6d2a97b8bfc5847f29d88ff0affff2c1e22768`
- Chips: 15

The candidate GitHub raw URL for this CSV returned HTTP 404; the bucket URL
returned HTTP 200. The reviewed official `Train.ipynb` references train,
validation, and test CSVs only; the Bolivia split was verified directly in the
bucket. `config.json` records the selected split method, source URL, exact
downloaded file hash, chip count, and a warning whenever fewer than five chips
are available.

Run a CSV-only split dry run to display train, validation, test, and Bolivia
chip counts without downloading imagery:

```powershell
python -m sahayak.ml.train --dry-run
```

The verified counts at the time of this check are 252 train, 89 validation,
90 test, and 15 Bolivia chips.

The upstream materials document 512 by 512 chips, two Sentinel-1 bands ordered
VV then VH, backscatter in dB, and labels -1 (no data), 0 (not water), and 1
(water). The full bucket is described as approximately 14 GB. The size of only
the hand-labeled S1 chips, labels, and split files is not published in the
reviewed materials, so the implementation downloads only the referenced files
instead of downloading the complete bucket. Downloads use anonymous HTTPS,
resume partial files, and validate the server-reported byte size.

The cited paper is Bonafilia, D., Tellman, B., Anderson, T., and Issenberg, E.
(2020), “Sen1Floods11: a georeferenced dataset to train and test deep learning
flood algorithms for Sentinel-1,” CVPR Workshops 2020, pp. 210–211. Mirrors
describe the dataset as CC BY 4.0; the upstream README page reviewed for this
implementation does not state the license itself. Confirm the applicable
upstream terms before redistribution.

Kuro Siwo is not wired as a training option. Its reviewed README does not
document a labeled pre/post GRD subset that can be streamed or downloaded under
10 GB. No Kuro Siwo training is claimed.

## Preprocessing and training

Install the optional dependencies with `pip install -e ".[ml]"`. The package
supports Python 3.11 and runs on CPU for smoke tests or CUDA for training.

```powershell
python -m sahayak.ml.train --dataset sen1floods11 --out outputs\sen1floods11
```

`--max-minutes` defaults to 240 minutes, `--resume` restores the last checkpoint,
and `--encoder-weights none` avoids ImageNet initialization. Training uses
random 256-pixel crops, flips, right-angle rotations, and train-only +/-1 dB
gain jitter. Values are clipped to [-30, 5] dB. VV/VH means and standard
deviations are calculated from valid training pixels only; labels -1 become
ignore index 255. The model input is VV, VH, and VV minus VH.

The network is an SMP U-Net with ResNet-34 encoder and one output logit. The
combined loss is binary cross entropy with logits plus soft Dice, excluding
ignored labels. AdamW uses learning rate 0.0003 and weight decay 0.0001 with a
cosine schedule, batch size 16, maximum 60 epochs, early stopping patience 10
on validation flood IoU, seed 42, and AMP when CUDA is available. Every epoch
writes `best.pt` when validation IoU improves and writes `last.pt` for resume.
The time budget is checked at epoch boundaries; after stopping, the run writes
available evaluation artifacts.

To evaluate an existing checkpoint without training, pass `--eval-only
--weights PATH`. It still prepares the selected dataset splits, evaluates each
split independently, and writes the same metrics, configuration, model card,
qualitative images, and checkpoint copies.

The decision threshold is selected by pooled F1 on validation pixels using a
0.001 grid. It is frozen for test and held-out Bolivia scoring. Evaluation runs
full chips in tiles and reports flood IoU, F1, precision, recall, accuracy,
confusion matrices, valid pixel counts, and per-chip median/p10/p90 IoU and F1.
Per-chip VV Otsu and fixed VV < -15 dB baselines use the same labeled pixels.
Baseline chips with fewer than 100 valid labeled finite-VV pixels or constant
VV are skipped, logged with chip ID and valid/finite counts, and listed in
`metrics.json`. If every chip is skipped, that baseline is recorded as an
evaluation error without suppressing model scores or other output files.
Validation, test, Bolivia, and each baseline evaluation are isolated; failures
are logged with exception class and message and recorded in `metrics.json`.
The reported 95% intervals bootstrap chips with replacement, using 1,000
resamples.

## Outputs

Each run writes:

- `best.pt`, `last.pt`: best and most recent model/checkpoint state.
- `metrics.json`: validation threshold, test and Bolivia results, both VV
  baselines, confusion/pixel counts, per-chip distributions, and bootstrap
  intervals.
- `training_log.csv`: measured train/validation losses and validation IoU/F1 by
  epoch.
- `config.json`: hyperparameters, normalization values, split sizes and hashes,
  Git commit when available, package versions, and device metadata.
- `model_card.md`: intended use, training data/citation, license caveat,
  measured results, and limitations.
- `qualitative/qualitative_01.png` through `qualitative_08.png`: VV, hand label,
  and prediction panels. These contain no people.

## Kaggle

`notebooks/kaggle_train.ipynb` has four code cells: clean and clone the repository
then install it with dependencies, check GPU availability, load the optional
`HF_TOKEN` Kaggle secret without printing its value and run the CSV-only dry-run
before training, then list output files and the ZIP archive. The install cells
use `pip install -q -e .` followed by the missing ML/geospatial packages; they do
not use `--no-deps`, so the project runtime dependencies (including
`pydantic-settings`) are installed while Kaggle's preinstalled PyTorch remains
available to pip's dependency resolver.

`scripts/kaggle_train.py` also checks the process environment for `HF_TOKEN`,
then tries Kaggle Secrets if the environment value is absent and the
`kaggle_secrets` package is available. It prints only whether a token is
available, trains under `/kaggle/working/outputs`, creates a ZIP archive, and
optionally uploads `best.pt`, `metrics.json`, `config.json`, and `model_card.md`
to `SrishtiPandey1203/sahayak-flood-unet`. Upload errors are reported by
exception class and do not discard local outputs or fail the training run.

## Synthetic smoke runs

Use `--synthetic --encoder-weights none` to exercise the training and artifact
pipeline without downloading a dataset. The resulting `config.json` and
`metrics.json` set `synthetic: true` and `final_metrics: false`; these metrics
must not be reported as model performance.

```powershell
python -m sahayak.ml.train --dataset sen1floods11 --out outputs\smoke --synthetic --encoder-weights none
```

## Limitations

- Training labels describe open-water flood extent, not debris flow or damage.
- The model has not been evaluated on a verified held-out Himalayan event.
- Sigma0 versus gamma0 and upstream preprocessing can shift backscatter values.
- Mountain radar shadow and layover obscure or distort observations.
- Ten-metre GRD pixels cannot resolve narrow channels or small damage.
- Dataset license terms need confirmation from the applicable upstream source.
- Evaluation is only as representative as the dataset labels and geographic split.
