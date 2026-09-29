# Dataset Processing

This directory contains the data preparation scripts used by RetroRoute.

The preprocessing pipeline converts raw RetroBench synthesis routes into grouped multistep examples, extracts single step reactions, and removes overlap between the training, validation, and test reaction sets.

All commands below should be executed from the repository root.

## Execution Order

```text
preprocess_multistep_retro.py
        |
        v
build_single_step_dataset.py
        |
        v
filter_single_step_overlaps.py
```

## 1. Multistep Route Preprocessing

`preprocess_multistep_retro.py` converts the raw RetroBench route files into the grouped representation used by the multistep search code.

```bash
mkdir -p ./dataset/processed

nohup python dataset_process/preprocess_multistep_retro.py \
  --input_dir ./dataset \
  --output_dir ./dataset/processed \
  --inner_path_as_route \
  --save_flat_route_level \
  > preprocess_multistep.log 2>&1 &
```

The primary outputs used later are:

```text
dataset/processed/train_dataset_grouped.json
dataset/processed/valid_dataset_grouped.json
dataset/processed/test_dataset_grouped.json
```

`--inner_path_as_route` interprets each inner synthesis path as an individual route.

`--save_flat_route_level` additionally stores route level representations when supported by the input data.

## 2. Build the Single Step Dataset

`build_single_step_dataset.py` extracts individual retrosynthetic reaction examples from the processed multistep routes.

```bash
mkdir -p ./dataset/single_step

nohup python dataset_process/build_single_step_dataset.py \
  --input_dir ./dataset/processed \
  --output_dir ./dataset/single_step \
  --dedup_mode reaction \
  > build_single_step.log 2>&1 &
```

Reaction level deduplication is enabled with:

```text
--dedup_mode reaction
```

The resulting files are used to construct the proposal model training, validation, and test sets.

## 3. Remove Cross Split Overlap

`filter_single_step_overlaps.py` removes overlapping reactions across the train, validation, and test splits.

```bash
mkdir -p ./dataset/single_step_no_overlap

nohup python dataset_process/filter_single_step_overlaps.py \
  --data_dir ./dataset/single_step \
  --output_dir ./dataset/single_step_no_overlap \
  --key_type reaction \
  > filter_overlap.log 2>&1 &
```

The key files used by the proposal model experiments are:

```text
dataset/single_step_no_overlap/
├── train_single_step_dedup.json
├── valid_single_step_no_train_overlap.json
└── test_single_step_no_train_valid_overlap.json
```

## Next Step

After preprocessing, train the planning conditioned MolT5 proposal model following:

```text
single_step_model/README.md
```
