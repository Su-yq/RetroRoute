# RetroBench Dataset

This directory stores the RetroBench data used to train and evaluate RetroRoute.

RetroBench is the multistep retrosynthesis benchmark introduced with FusionRetro and also used by RetroInText.

## Files Included in This Repository

The public repository contains:

```text
dataset/
├── test_dataset.json
├── valid_dataset.json
└── README.md
```

The full training set is not uploaded because of its file size.

## Required Training Data

Download the RetroBench training split from the FusionRetro/RetroBench release used by RetroInText and place it in this directory as:

```text
dataset/train_dataset.json
```

After downloading, the raw route files should be:

```text
dataset/
├── train_dataset.json
├── valid_dataset.json
└── test_dataset.json
```

The RetroInText release provides the RetroBench dataset through its associated data release.

## Starting Material Stock

Multistep search additionally requires the ZINC starting material stock used by RetroBench.

Download:

```text
zinc_stock_17_04_20.hdf5
```

and place it at:

```text
dataset/zinc_stock_17_04_20.hdf5
```

The resulting directory should contain:

```text
dataset/
├── train_dataset.json
├── valid_dataset.json
├── test_dataset.json
├── zinc_stock_17_04_20.hdf5
└── README.md
```

## Preprocessing

Raw RetroBench routes must be converted into the grouped multistep format used by RetroRoute.

From the repository root, run:

```bash
mkdir -p ./dataset/processed

python dataset_process/preprocess_multistep_retro.py \
  --input_dir ./dataset \
  --output_dir ./dataset/processed \
  --inner_path_as_route \
  --save_flat_route_level
```

The grouped files used in subsequent experiments are expected under:

```text
dataset/processed/
├── train_dataset_grouped.json
├── valid_dataset_grouped.json
└── test_dataset_grouped.json
```

The preprocessing script may additionally create route level files when `--save_flat_route_level` is enabled.

For the complete preprocessing pipeline, see:

```text
dataset_process/README.md
```
