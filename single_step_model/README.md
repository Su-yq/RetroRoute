# Proposal Model Training and URPO

This directory contains the MolT5 proposal training and Utility Regularized Positive Optimization pipeline used by RetroRoute.

RetroRoute first adapts MolT5 with a planning scaffold. The trained proposal model then generates alternative reactions that are evaluated using complementary chemistry and route related utility signals. Supported nonreference reactions are selected as additional positive supervision for continued proposal model training.

## Pipeline

```text
train_molt5_route_context_sft.py
        |
        v
eval_molt5_topk.py
        |
        v
generate_route_context_candidates.py
        |
        v
score_candidates_with_utility.py
        |
        v
build_u_positive_sft_data.py
        |
        v
train_molt5_route_context_positive_sft.py
```

All commands below should be executed from the repository root.

## Prerequisites

Prepare the overlap filtered single step dataset:

```text
dataset/single_step_no_overlap/
├── train_single_step_dedup.json
├── valid_single_step_no_train_overlap.json
└── test_single_step_no_train_valid_overlap.json
```

Prepare the MolT5 model:

```text
MolT5/model/
```

Prepare the ReactionT5 forward reaction model:

```text
ReactionT5/model/
```

Prepare the ZINC starting material stock:

```text
dataset/zinc_stock_17_04_20.hdf5
```

---

## 1. Planning Scaffold SFT

`train_molt5_route_context_sft.py` adapts MolT5 using a structured planning context.

The proposal input contains:

```text
target molecule
current molecule
current depth
maximum search depth
route completion objective
```

The model output remains the predicted reactant sequence.

Run:

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

The best checkpoint is saved as:

```text
outputs/molt5_route_context/checkpoint-best
```

---

## 2. Evaluate Single Step Proposal Accuracy

Evaluate the planning scaffold model on the held out single step test set.

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

---

## 3. Generate Top 20 Reaction Candidates

Generate alternative reaction candidates for the train and validation splits.

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

The expected outputs are:

```text
outputs/route_candidates_top20/
├── train_candidates_top20.jsonl
└── valid_candidates_top20.jsonl
```

These candidates form the action pool used during offline utility evaluation.

---

## 4. Score Candidates with Multi Signal Utility

`score_candidates_with_utility.py` evaluates each generated candidate using complementary chemistry and route related signals.

The script was named `build_dpo_pairs_with_u.py` during early experiments. It is renamed in the public RetroRoute repository because the final method does not perform DPO optimization.

The candidate utility contains the following signals:

```text
exact reaction agreement
forward reaction plausibility
molecular validity
route future compatibility
molecular representation similarity
building block availability
undesirable action penalty
```

The reported utility weights are:

```text
exact reaction agreement        4.0
forward reaction plausibility   1.0
molecular validity              0.5
route compatibility             0.5
representation similarity       0.3
building block availability     0.2
undesirable action penalty      1.0
```

ReactionT5 is used to evaluate forward reaction consistency.

Run:

```bash
mkdir -p ./outputs/utility_scores

nohup python single_step_model/score_candidates_with_utility.py \
  --candidate_dir ./outputs/route_candidates_top20 \
  --single_step_dir ./dataset/single_step_no_overlap \
  --output_dir ./outputs/utility_scores \
  --project_root . \
  --forward_model_path ./ReactionT5/model \
  --stock_path ./dataset/zinc_stock_17_04_20.hdf5 \
  --splits train valid \
  --forward_topk 5 \
  --forward_num_beams 5 \
  --forward_batch_size 16 \
  --forward_fp16 \
  > utility_scoring.log 2>&1 &
```

The important outputs for the final RetroRoute pipeline are:

```text
outputs/utility_scores/
├── train_scored_candidates.jsonl
└── valid_scored_candidates.jsonl
```

Each candidate record contains its utility score together with the individual utility components.

### Auxiliary Preference Pair Files

The current utility scoring implementation also retains functionality from earlier preference optimization experiments and may additionally generate files such as:

```text
train_dpo_pairs.jsonl
valid_dpo_pairs.jsonl
```

These files are **not used by the final RetroRoute training pipeline**.

URPO consumes only the scored candidate records.

### Molecular Representation Signal

The utility scorer supports the molecular representation model used in the original experimental environment.

The corresponding paths can be provided through:

```text
--fusion_root
--threed_config
--threed_checkpoint
```

If the external representation model cannot be loaded, the implementation falls back to a Morgan fingerprint representation.

For exact reproduction of a specific experimental environment, configure these paths to the corresponding FusionRetro or RetroInText resources before utility scoring.

---

## 5. Build Utility Supported Positive Training Data

`build_u_positive_sft_data.py` selects high utility generated reactions as additional positive targets.

It does not optimize against rejected candidates.

Run:

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

The reported configuration retains candidates satisfying:

```text
valid score                    >= 1.0
bad action penalty             <= 0.0
forward plausibility           >= 0.5
utility                        >= 1.5
```

At most one generated positive is retained for each original training example.

The total number of generated positive examples is restricted to at most half the number of gold examples.

The outputs are:

```text
outputs/urpo_data/
├── train_u_positive_sft.json
├── valid_gold_sft.json
└── u_positive_sft_data_report.json
```

---

## 6. URPO Proposal Refinement

Continue training from the Planning Scaffold checkpoint using gold reactions and the selected utility supported alternatives.

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

The proposal checkpoint used in the reported multistep experiments is:

```text
outputs/molt5_urpo/checkpoint-epoch-6
```

## Next Step

The URPO refined proposal model is next used to construct state aware ChemDFM training examples.

Return to the repository root and run:

```text
build_chemdfm_dataset.py
```

followed by:

```text
train_chemdfm.py
multistep_test.py
```

See the root `README.md` for these stages.
