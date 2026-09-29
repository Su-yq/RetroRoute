# RetroBench Dataset

This directory stores the RetroBench data used by RetroRoute.

RetroBench is a multistep retrosynthesis benchmark constructed from USPTO reaction pathways and used in FusionRetro and subsequent retrosynthetic planning studies.

## Files Included in This Repository

The GitHub repository contains:

```text
dataset/
├── test_dataset.json
├── valid_dataset.json
└── README.md
```

The training split is not stored in this repository because of its file size.

## Training Set

Download the RetroBench training split from the FusionRetro or RetroInText data release and place it at:

```text
dataset/train_dataset.json
```

After downloading the training set, the raw dataset directory should contain:

```text
dataset/
├── train_dataset.json
├── valid_dataset.json
└── test_dataset.json
```

## Starting Material Stock

Multistep route search additionally requires the ZINC starting material stock used by RetroBench.

Place:

```text
zinc_stock_17_04_20.hdf5
```

at:

```text
dataset/zinc_stock_17_04_20.hdf5
```

The complete dataset directory should therefore be:

```text
dataset/
├── train_dataset.json
├── valid_dataset.json
├── test_dataset.json
├── zinc_stock_17_04_20.hdf5
└── README.md
```

## Preprocessing

Raw routes must be processed before training RetroRoute.

From the repository root, run:

```bash
mkdir -p ./dataset/processed

nohup python dataset_process/preprocess_multistep_retro.py \
  --input_dir ./dataset \
  --output_dir ./dataset/processed \
  --inner_path_as_route \
  --save_flat_route_level \
  > preprocess_multistep.log 2>&1 &
```

The grouped multistep datasets used in later stages are:

```text
dataset/processed/
├── train_dataset_grouped.json
├── valid_dataset_grouped.json
└── test_dataset_grouped.json
```

Continue with:

```text
dataset_process/README.md
```

for single step extraction and cross split overlap removal.
