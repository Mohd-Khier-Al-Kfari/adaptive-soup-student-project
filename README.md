# Adaptive Soup Student Project

**Author:** Moh'd Khier Al Kfari

This repository prepares PAMAP2, RealWorld, and MHEALTH human-activity-
recognition data for cross-dataset, cross-gender, and cross-age UDA experiments.
The prepared files are ready for model training: students load train, validation,
and test arrays without repeating preprocessing.

## Installation

With Conda:

```bash
conda env create -f environment.yml
conda activate adaptive_soup_student
```

With pip:

```bash
python -m pip install -r requirements.txt
```

## Run the pipeline

From the repository root:

```bash
python run_pipeline.py
```

The pipeline asks whether each dataset should be downloaded, preprocessed, and
postprocessed. It creates repository-relative folders automatically and reports
the reason for any failed stage. Independent datasets run concurrently, and the
independent postprocessing targets for one dataset also run concurrently. Use
`--workers 1` for a fully serial run or `--workers 3` to use up to three workers.

## Load prepared data

```bash
python load_experiment.py
```

Choose `cross_dataset`, `cross_gender`, or `cross_age`. The loader returns every
dataset/group in the selected experiment. Each prepared entry contains:

- `features.pkl`: `train`, `val`, and `test` feature arrays grouped by subject.
- `evaluation_labels.pkl`: label map and labels corresponding to those arrays.

## Plot an experiment

```bash
python plot_experiment.py
```

Plots are saved under `data/prepared/plots/` and displayed interactively. The
figures use consistent wrist/ankle rows, running/walking/standing columns, and
shared accelerometer legends and axes.

## Repository layout

```text
src/preprocessing/       Download, cleaning, splitting, normalization, windowing
src/data_access/         Prepared-data loader and plotting utilities
src/training/             Future model and transfer-learning code
data/original_datasets/  Local downloaded archives and extracted sources
data/processed_datasets/ Intermediate local parser output
data/prepared/           Final experiment files and metadata
data/reports/            Local diagnostic reports
data/plots/              Additional generated figures
models/                  Experiment metrics and local checkpoints
```

Cross-dataset data is stored directly by dataset:

```text
data/prepared/cross_dataset/PAMAP2/
data/prepared/cross_dataset/RealWorld/
data/prepared/cross_dataset/MHEALTH/
```

Gender and age experiments retain their group folders, such as
`cross_gender/RealWorld/female/` and `cross_age/RealWorld/below30/`.

## Processing rules

- Subjects are assigned before windowing.
- Windows never cross recording or activity-segment boundaries.
- Normalization statistics are fitted from training data and applied to all splits.
- Windows and their labels are shuffled together with seed `52`.
- The shared activities are lying, sitting, standing, walking, and running;
  dataset-specific aliases are mapped to this vocabulary.

Large datasets, prepared arrays, logs, caches, and model checkpoints are ignored
by Git. Small metadata, configuration, documentation, and selected plots can be
committed safely.

## Experiment assignments

The reviewed assignments are stored in `src/preprocessing/config.toml`.

Cross-dataset uses PAMAP2 `101,103,106,107,108 / 105 / 102,104`, RealWorld
`proband1,2,3,5,6,7,12,13,14,15 / proband4,8 / proband9,10,11`, and MHEALTH
`1,3,4,6,10 / 2,7 / 5,8,9` for train/validation/test. PAMAP2 subject109 is
excluded because it has no selected five-class activity coverage.

Cross-gender uses separate male and female groups. PAMAP2 male is
`103,104,107,108 / 105 / 101,106`; its single female subject102 uses a
non-overlapping 80/10/10 temporal split. RealWorld male is
`proband2,4,5,7,9 / proband3 / proband10,14`, and female is
`proband1,6,8,12 / proband13 / proband11,15`.

Cross-age is RealWorld only: `below30` uses `6/2/2` subjects and `age30plus`
uses `3/1/1` subjects, as listed in the configuration.

## Data contract

The common activity map is `lying=0`, `sitting=1`, `standing=2`, `walking=3`,
and `running=4`. Jogging is mapped to running. Empty, unknown, other, and
transition labels are excluded. The prepared sensor arrays contain 18 channels:
accelerometer, gyroscope, and magnetometer x/y/z for the upper sensor followed
by the lower sensor. Plotting uses accelerometer indices `0:3` for the general
`wrist` row and `9:12` for the general `ankle` row; the source sensor names are
dataset-specific (hand/forearm and ankle/shin).

The parser first preserves continuous recording segments. Subjects and temporal
partitions are assigned before windowing, so a window never crosses a recording
or partition boundary. The configured sample rate is 50 Hz, window size is 150
samples, and shift is 75 samples. Normalization moments are fitted on the full
training recordings only, then applied to validation and test. Windows and their
labels are shuffled together with seed `52`.

`features.pkl` contains `train`, `val`, and `test` dictionaries keyed by subject,
with each value an array shaped `(windows, samples, channels)`. The matching
`evaluation_labels.pkl` contains `label_map`, split/subject labels, and a
fingerprint. Metadata JSON records the experiment, group, splits, activities,
channels, sampling, windowing, normalization, and fingerprint.

## Troubleshooting and reproducibility

If loading or plotting reports that no prepared experiments are available, run
`python run_pipeline.py` and enable preprocessing and postprocessing. If plotting
reports a missing Matplotlib dependency, install `requirements.txt` or create
`environment.yml`. Use `--workers 1` to debug serially; use up to `--workers 3`
for independent dataset and target jobs. All paths are derived from the
repository location, so the project can be cloned to another computer.

## Future models

Training code belongs in `src/training/`. Save experiment configurations and
small metrics under `models/cross_dataset`, `models/cross_gender`, or
`models/cross_age`. Large `.pt`, `.pth`, `.ckpt`, and `.bin` checkpoints stay
local and are ignored by Git.

## Training transfers

Install the pinned NumPy `2.4.6` and PyTorch `2.14.1` environment from
`requirements.txt` or `environment.yml`, then run:

```bash
python train_experiments.py
```

The command supports one transfer or `all`. It uses the attached SelfHAR CNN,
seed `52`, Adam (`lr=0.0003`, `weight_decay=0.0001`), batch size `32`, and 30
epochs. Prepared windows are already shuffled with their labels, so the
training loader does not shuffle again. Target labels are evaluation-only.
Final-epoch metrics, predictions, and `model_last.pt` are saved under the
corresponding `models/` transfer directory. Choosing `all` runs up to three
independent transfers concurrently on the selected GPU.

The training head is created from the selected experiment's `label_map`; the
loss is therefore not fixed to five classes. Cross-dataset uses the five shared
activities, while cross-gender and cross-age use their compatible activity map.
The source training labels are the only labels used for optimization. Source
validation/test and every target split are evaluation-only. Each transfer saves
`model_last.pt`, `config.json`, `metrics.json`, and prediction files under its
own directory in `models/`. The scheduler uses three concurrent workers on the
same selected GPU and reports failures per transfer.
