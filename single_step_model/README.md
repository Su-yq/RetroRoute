# Single Step Proposal Model and URPO

This directory contains the MolT5 based proposal model training and Utility Regularized Positive Optimization pipeline used by RetroRoute.

The proposal pipeline contains four main stages:

```text
Planning Scaffold SFT
        |
        v
Top K evaluation
        |
        v
Top 20 candidate generation
        |
        v
Utility supported positive construction
        |
        v
URPO continued SFT
```

All commands below should be executed from the repository root.

## Prerequisites

Prepare the processed single step data:

```text
dataset/single_step_no_overlap/
├── train_single_step_dedup.json
├── valid_single_step_no_train_overlap.json
└── test_single_step_no_train_valid_overlap.json
```

Prepare the MolT5 model under:

```text
MolT5/model/
```

See `MolT5/README.md`.

## 1. Train the Planning Scaffold Proposal Model

`train_molt5_route_context_sft.py` fine tunes MolT5 using the planning scaffold containing the target molecule, current molecule, current depth, maximum depth, and route completion objective.

```bash
mkdir -p ./outputs/molt5_route_context

nohup python single_step_model/train_molt5_route_context_sft.py \
  --data_dir ./dataset/single_step_no_overlap \
  --model_dir ./MolT5/model \
  --output_dir ./outputs/molt5_route_context \
  --train_file train_single_step_dedup.json \
  --valid_file valid_single_step_no_train_overlap.json \
  --max_depth 14 \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --epochs 40 \
  --weight_decay 0.1 \
  --save_every_epochs 10 \
  --early_stop_patience 5 \
  --early_stop_min_delta 1e-4 \
  --fp16 \
  > train_route_context.log 2>&1 &
```

The best checkpoint is expected at:

```text
outputs/molt5_route_context/checkpoint-best
```

## 2. Evaluate Single Step Top K Accuracy

Use `eval_molt5_topk.py` to evaluate the planning scaffold model.

```bash
mkdir -p ./outputs/molt5_topk

nohup python single_step_model/eval_molt5_topk.py \
  --model_dir ./outputs/molt5_route_context/checkpoint-best \
  --data_file ./dataset/single_step_no_overlap/test_single_step_no_train_valid_overlap.json \
  --output_dir ./outputs/molt5_topk \
  --prompt_mode route_context \
  --max_depth 14 \
  --topk 10 \
  --batch_size 16 \
  --fp16 \
  > eval_molt5_topk.log 2>&1 &
```

## 3. Generate Top 20 Candidate Reactions

Generate alternative reactions for the train and validation splits using the trained planning scaffold model.

```bash
mkdir -p ./outputs/route_candidates_top20

nohup python single_step_model/generate_route_context_candidates.py \
  --data_dir ./dataset/single_step_no_overlap \
  --model_dir ./outputs/molt5_route_context/checkpoint-best \
  --output_dir ./outputs/route_candidates_top20 \
  --splits train valid \
  --topk 20 \
  --batch_size 8 \
  --max_depth 14 \
  --fp16 \
  > generate_candidates.log 2>&1 &
```

These candidates form the action pool used for utility evaluation.

## 4. Utility Scoring

Generated nonreference reactions are evaluated using multiple signals including molecular validity, forward reaction plausibility, route compatibility, reference alignment, building block availability, and undesirable action penalties.

The utility scoring stage used in the experiments produces scored files such as:

```text
outputs/utility_scores/train_scored_candidates.jsonl
```

and optionally:

```text
outputs/utility_scores/valid_scored_candidates.jsonl
```

The public file tree must contain either these precomputed scored candidates or the corresponding utility scoring script before the next step can be reproduced from scratch.

ReactionT5 is used as the forward reaction model during this stage. See:

```text
ReactionT5/README.md
```

## 5. Build Utility Supported Positive Data

`build_u_positive_sft_data.py` filters generated candidates according to the utility and admissibility criteria used by URPO.

```bash
mkdir -p ./outputs/urpo_data

python single_step_model/build_u_positive_sft_data.py \
  --train_json ./dataset/single_step_no_overlap/train_single_step_dedup.json \
  --valid_json ./dataset/single_step_no_overlap/valid_single_step_no_train_overlap.json \
  --train_scored_candidates ./outputs/utility_scores/train_scored_candidates.jsonl \
  --output_dir ./outputs/urpo_data \
  --max_pseudo_per_sample 1 \
  --max_pseudo_to_gold_ratio 0.5 \
  --min_valid_score 1.0 \
  --max_bad_action_penalty 0.0 \
  --min_forward_plausibility 0.5 \
  --min_utility 1.5
```

The main output files are:

```text
outputs/urpo_data/
├── train_u_positive_sft.json
└── valid_gold_sft.json
```

At most one supported alternative is added for each original training example, and generated positives are restricted to at most half the number of gold examples.

## 6. Train the URPO Refined Proposal Model

Continue training from the best Planning Scaffold checkpoint using the utility supported positive dataset.

```bash
mkdir -p ./outputs/molt5_urpo

nohup env CUDA_VISIBLE_DEVICES=3 python single_step_model/train_molt5_route_context_positive_sft.py \
  --init_model_dir ./outputs/molt5_route_context/checkpoint-best \
  --train_file ./outputs/urpo_data/train_u_positive_sft.json \
  --valid_file ./outputs/urpo_data/valid_gold_sft.json \
  --output_dir ./outputs/molt5_urpo \
  --epochs 30 \
  --lr 3e-6 \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --eval_batch_size 8 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --label_smoothing 0.02 \
  --max_source_length 512 \
  --max_target_length 256 \
  --max_depth 14 \
  --max_grad_norm 1.0 \
  --fp16 \
  --logging_steps 100 \
  > train_urpo.log 2>&1 &
```

The reported multistep experiments use:

```text
outputs/molt5_urpo/checkpoint-epoch-6
```

as the proposal checkpoint.

## Next Step

After proposal refinement, return to the repository root and construct the ChemDFM search state dataset:

```text
build_chemdfm_dataset.py
```

The corresponding commands are provided in the root `README.md`.
