# Adaptive Soup Student Project

**Author:** Moh'd Khier Al Kfari

This repository prepares PAMAP2, RealWorld, and MHEALTH human activity data for three unsupervised domain adaptation (UDA) experiment families: cross-dataset, cross-gender, and cross-age. The prepared files are ready for loading, plotting, and training without repeating preprocessing.

## Installation

From the repository root, use either Conda:

```bash
conda env create -f environment.yml
conda activate adaptive_soup_student
```

or pip:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run every pipeline stage and answer the dataset prompts:

```bash
python run_pipeline.py
```

Load all groups in one experiment family:

```bash
python load_experiment.py
```

Create and save the plots for one experiment family:

```bash
python plot_experiment.py
```

Train one transfer or queue all registered transfers:

```bash
python train_experiments.py
```

Run these commands from the repository root. Paths are derived from the repository location, so the project can be cloned elsewhere without editing machine-specific paths.

## What the code does

- `run_pipeline.py` starts downloading, parsing, cleaning, splitting, normalizing, windowing, shuffling, and saving. It reports the dataset, stage, and reason for every failure.
- `load_experiment.py` lets the student select `cross_dataset`, `cross_gender`, or `cross_age`, then loads every prepared group for that family with train, validation, and test data.
- `plot_experiment.py` loads prepared windows, plots accelerometer examples for running, walking, and standing, displays the figure, and saves it under `data/prepared/plots/`.
- `train_experiments.py` selects one transfer or `all`, trains the SelfHAR CNN on source training data, evaluates source and target partitions, and saves final weights and metrics.

Implementation modules are organized as follows:

- `src/preprocessing/` downloads, parses, cleans, filters, splits, normalizes, windows, shuffles, and writes prepared data.
- `src/data_access/` loads prepared files and creates plots.
- `src/training/` defines the CNN, transfers, training lifecycle, metrics, scheduling, and checkpoint writing.

## Data locations

```text
data/
├── original_datasets/       raw archives and extracted source data
├── processed_datasets/      intermediate parser output
├── prepared/                final student-ready experiment data
│   ├── cross_dataset/
│   │   ├── PAMAP2/
│   │   ├── RealWorld/
│   │   └── MHEALTH/
│   ├── cross_gender/
│   │   ├── PAMAP2/male/
│   │   ├── PAMAP2/female/
│   │   ├── RealWorld/male/
│   │   └── RealWorld/female/
│   └── cross_age/
│       └── RealWorld/
│           ├── below30/
│           └── age30plus/
├── reports/                 preprocessing and coverage reports
└── prepared/plots/          generated figures

models/
├── cross_dataset/
├── cross_gender/
└── cross_age/
```

Raw archives belong in `data/original_datasets/`. Intermediate parser files belong in `data/processed_datasets/`. Final student-ready files belong in `data/prepared/`. Reports and generated plots are kept separately. Each trained transfer gets its own directory under `models/<experiment_family>/`.

## Experiment families and transfers

### Cross-dataset

Cross-dataset transfers use the five shared activities and include:

1. PAMAP2 → RealWorld
2. RealWorld → PAMAP2
3. PAMAP2 → MHEALTH
4. MHEALTH → PAMAP2
5. RealWorld → MHEALTH
6. MHEALTH → RealWorld

### Cross-gender

Cross-gender transfers stay within one dataset:

1. PAMAP2 male → PAMAP2 female
2. PAMAP2 female → PAMAP2 male
3. RealWorld male → RealWorld female
4. RealWorld female → RealWorld male

### Cross-age

Cross-age transfers use RealWorld age groups:

1. RealWorld below30 → RealWorld age30plus
2. RealWorld age30plus → RealWorld below30

## Activity contract

The parser first reads the native activities available in each dataset. It excludes empty labels, `0`, `null`, `unknown`, `other`, `transition`, `transitions`, and `other (transient activities)`. `jogging` is mapped to `running` where applicable.

Cross-dataset keeps exactly these five shared classes:

```text
lying = 0
sitting = 1
standing = 2
walking = 3
running = 4
```

Cross-gender and cross-age keep every remaining valid activity available in the selected dataset and group. Their classes are not assumed to be five fixed classes; students must read the actual `label_map` from that group's `evaluation_labels.pkl` and metadata. A source-to-target transfer is allowed only when the label maps are compatible.

## Sensor and channel contract

Every prepared feature array uses 18 channels in this order:

```text
Upper sensor:
  0:3   accelerometer x, y, z
  3:6   gyroscope x, y, z
  6:9   magnetometer x, y, z

Lower sensor:
  9:12  accelerometer x, y, z
  12:15 gyroscope x, y, z
  15:18 magnetometer x, y, z
```

The plots use general location names:

| General name | PAMAP2 source | RealWorld source | MHEALTH source |
|---|---|---|---|
| wrist | hand | forearm | forearm |
| ankle | ankle | shin | ankle |

Only accelerometer channels are plotted. For `wrist`, `acc_x`, `acc_y`, and `acc_z` are indices `0`, `1`, and `2`. For `ankle`, they are indices `9`, `10`, and `11`. Gyroscope and magnetometer channels remain available in the feature arrays.

## Processing rules

- Subject and group assignments happen before windowing.
- Windows never cross recording or activity-segment boundaries.
- Data are resampled to the configured rate and short missing gaps are interpolated according to the configuration.
- Normalization statistics are fitted from the full training data only and then applied to validation and test.
- Windows and labels are shuffled together with seed `52`.
- The configured sample rate is `50 Hz`.
- The window size is `150` samples and the shift is `75` samples.

## Prepared-file contract

Each prepared group contains `features.pkl` and `evaluation_labels.pkl`.

`features.pkl` contains `train`, `val`, and `test` dictionaries keyed by subject. Each subject value is an array shaped:

```text
(number_of_windows, window_size, number_of_channels)
```

`evaluation_labels.pkl` contains labels aligned with those windows, the group's `label_map`, split/subject information where available, and a data fingerprint. Group metadata records the dataset, experiment family, group, subject split, activities, channels, sampling and window settings, normalization provenance, and fingerprints.

Students should load these files directly. They should not repeat downloading, preprocessing, normalization, windowing, or shuffling before training.

## Training and checkpoints

Each transfer trains only on source `train`. Source `val` is used for validation; source `test` is used only for evaluation. Target `train`, `val`, and `test` are evaluation-only in the source-only baseline, so target labels are never used for optimization.

The default training configuration is:

- SelfHAR CNN from `src/training/model.py`.
- Dynamic classifier size from the selected label map.
- Adam optimizer.
- Learning rate `0.0003`.
- Weight decay `0.0001`.
- Batch size `32`.
- `30` epochs.
- Cross-entropy loss with the dynamic number of classes.
- Training and evaluation loaders do not reshuffle prepared windows.
- CUDA when available, otherwise CPU.
- Seed `52`.
- Final epoch only is saved by default.

A successful transfer writes to a unique directory under `models/`:

```text
models/<experiment_family>/<source>_to_<target>/
├── model_last.pt
├── config.json
├── metrics.json
└── predictions/
```

`config.json` records the transfer, label map, model and training settings, seed, device, and paths. `metrics.json` records final-epoch source and target results. Prediction files contain the evaluated predictions and true labels.

When `all` is selected, up to three independent jobs can run concurrently on one GPU. Each transfer has separate output paths and reports its own progress or failure. Use one worker while debugging if needed.

## Reproducibility and troubleshooting

Seed `52` is used for preprocessing shuffling and training randomness. Use `--workers 1` for serial debugging or up to three workers for independent jobs.

If prepared data are missing, run `python run_pipeline.py` and enable the required dataset stages. If an archive is missing, place it under `data/original_datasets/` or select download. If a label map is incompatible, use a compatible source and target group. Missing dependencies, missing files, CUDA errors, and out-of-memory errors identify the affected dataset or transfer in the console output.

## Git and generated files

The repository tracks source code, configuration, tests, and documentation. Raw datasets, processed/prepared arrays, reports, logs, generated plots, caches, virtual environments, and model checkpoints are generated locally and ignored by Git. This keeps the public repository small while preserving the commands and documentation needed to reproduce the workflow.
