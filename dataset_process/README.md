# Dataset Processing

This directory contains the preprocessing scripts used to convert RetroBench routes into the data formats required by RetroRoute.

## Pipeline

```text
preprocess_multistep_retro.py
        |
        v
build_single_step_dataset.py
        |
        v
filter_single_step_overlaps.py
```

All commands below should be executed from the repository root.

## 1. Process Multistep Routes

`preprocess_multistep_retro.py` converts the raw RetroBench route files into the grouped multistep representation used during search and ChemDFM dataset construction.

```bash
mkdir -p ./dataset/processed

nohup python dataset_process/preprocess_multistep_retro.py \
  --input_dir ./dataset \
  --output_dir ./dataset/processed \
  --inner_path_as_route \
  --save_flat_route_level \
  > preprocess_multistep.log 2>&1 &
```

The primary grouped outputs are:

```text
dataset/processed/
├── train_dataset_grouped.json
├── valid_dataset_grouped.json
└── test_dataset_grouped.json
```

`--inner_path_as_route` treats each inner synthesis path as an individual route.

`--save_flat_route_level` additionally saves route level records when available.

## 2. Build the Single Step Dataset

Individual retrosynthetic reaction examples are extracted from the processed multistep routes.

```bash
mkdir -p ./dataset/single_step

nohup python dataset_process/build_single_step_dataset.py \
  --input_dir ./dataset/processed \
  --output_dir ./dataset/single_step \
  --dedup_mode reaction \
  > build_single_step.log 2>&1 &
```

Reaction level deduplication is enabled using:

```text
--dedup_mode reaction
```

## 3. Remove Cross Split Overlap

To avoid reaction overlap between training, validation, and test examples, run:

```bash
mkdir -p ./dataset/single_step_no_overlap

nohup python dataset_process/filter_single_step_overlaps.py \
  --data_dir ./dataset/single_step \
  --output_dir ./dataset/single_step_no_overlap \
  --key_type reaction \
  > filter_single_step_overlap.log 2>&1 &
```

The files used by subsequent proposal model experiments are:

```text
dataset/single_step_no_overlap/
├── train_single_step_dedup.json
├── valid_single_step_no_train_overlap.json
└── test_single_step_no_train_valid_overlap.json
```

## Next Step

Continue with:

```text
single_step_model/README.md
```

to train the Planning Scaffold MolT5 model and perform URPO refinement.
