# RetroRoute

Effective multistep retrosynthetic planning requires reaction decisions to account for their downstream influence on route completion. Conventional systems commonly optimize reaction proposal from isolated product reactant pairs and rely heavily on local prediction scores during search. This creates a mismatch between single step reaction plausibility and route level utility as sequential decisions accumulate.

We introduce **RetroRoute**, a route level agentic framework that integrates planning conditioned reaction generation, utility guided proposal refinement, and search state aware LLM decision making within an explicit multistep search process. First, RetroRoute conditions a MolT5 proposal model on the synthesis target, current molecule, search depth, and route completion objective to generate reaction candidates with planning relevant context. Second, **Utility Regularized Positive Optimization (URPO)** evaluates generated alternatives using complementary chemical and route utility signals and incorporates supported reactions as additional positive supervision. Third, a ChemDFM based decision model jointly observes the evolving search state and candidate reaction set and learns listwise preferences for branch selection.

Through this formulation, RetroRoute separates candidate coverage from state dependent branch evaluation while connecting both stages to the objective of route completion. The explicit search procedure preserves interpretable route construction and allows the frontier, route history, stock information, and search budget to influence subsequent decisions.

## Framework

The overall RetroRoute framework is available here:

![image](./Framework.png)

## Repository Structure

```text
RetroRoute/
├── ChemDFM/
│   └── README.md
├── MolT5/
│   └── README.md
├── ReactionT5/
│   └── README.md
├── dataset/
│   ├── README.md
│   ├── test_dataset.json
│   └── valid_dataset.json
├── dataset_process/
│   ├── README.md
│   ├── preprocess_multistep_retro.py
│   ├── build_single_step_dataset.py
│   └── filter_single_step_overlaps.py
├── single_step_model/
│   ├── README.md
│   ├── train_molt5_route_context_sft.py
│   ├── eval_molt5_topk.py
│   ├── generate_route_context_candidates.py
│   ├── score_candidates_with_utility.py
│   ├── build_u_positive_sft_data.py
│   └── train_molt5_route_context_positive_sft.py
├── build_chemdfm_dataset.py
├── train_chemdfm.py
├── multistep_test.py
├── Framework.pdf
├── LICENSE
└── README.md
```

## External Resources

The pretrained model weights and the complete RetroBench training data are not included in this repository.

RetroRoute requires the following external resources:

```text
MolT5/model/
ReactionT5/model/
ChemDFM/model/
dataset/train_dataset.json
dataset/zinc_stock_17_04_20.hdf5
```

The MolT5 initialization follows the checkpoint used by RetroInText. ReactionT5 is used as an independent forward reaction model during offline utility evaluation. ChemDFM is used as the search state aware chemistry language model.

See the corresponding README files for model preparation.

## Complete Reproduction Pipeline

All commands in this repository are intended to be executed from the repository root.

```text
RetroBench raw routes
        |
        v
Multistep route preprocessing
        |
        v
Single step reaction extraction
        |
        v
Cross split overlap removal
        |
        v
Planning Scaffold MolT5 SFT
        |
        v
Top 20 reaction candidate generation
        |
        v
Multi signal candidate utility scoring
        |
        v
Utility supported positive selection
        |
        v
URPO proposal refinement
        |
        v
ChemDFM search state dataset construction
        |
        v
ChemDFM listwise adaptation
        |
        v
Multistep RetroRoute evaluation
```

Dataset preprocessing is described in:

```text
dataset_process/README.md
```

Planning Scaffold training and URPO refinement are described in:

```text
single_step_model/README.md
```

The ChemDFM stages and final multistep evaluation are described below.

---

## 1. Build the ChemDFM Search State Dataset

The ChemDFM training data are constructed using the URPO refined MolT5 proposal model.

The proposal checkpoint used in the reported experiments is:

```text
outputs/molt5_urpo/checkpoint-epoch-6
```

Create the training split:

```bash
mkdir -p ./outputs/chemdfm_dataset

nohup env CUDA_VISIBLE_DEVICES=1 python build_chemdfm_dataset.py \
  --input_json ./dataset/processed/train_dataset_grouped.json \
  --input_format grouped \
  --split_name train \
  --model_dir ./outputs/molt5_urpo/checkpoint-epoch-6 \
  --stock_path ./dataset/zinc_stock_17_04_20.hdf5 \
  --output_jsonl ./outputs/chemdfm_dataset/train_chemdfm.jsonl \
  --report_json ./outputs/chemdfm_dataset/train_report.json \
  --checkpoint_json ./outputs/chemdfm_dataset/train_checkpoint.json \
  --topk 10 \
  --num_beams 10 \
  --batch_size 4 \
  --max_depth 14 \
  --max_reactants 5 \
  --max_history_steps 6 \
  --fp16 \
  --overwrite \
  > build_chemdfm_train.log 2>&1 &
```

Create the validation split:

```bash
nohup env CUDA_VISIBLE_DEVICES=3 python build_chemdfm_dataset.py \
  --input_json ./dataset/processed/valid_dataset_grouped.json \
  --input_format grouped \
  --split_name valid \
  --model_dir ./outputs/molt5_urpo/checkpoint-epoch-6 \
  --stock_path ./dataset/zinc_stock_17_04_20.hdf5 \
  --output_jsonl ./outputs/chemdfm_dataset/valid_chemdfm.jsonl \
  --report_json ./outputs/chemdfm_dataset/valid_report.json \
  --checkpoint_json ./outputs/chemdfm_dataset/valid_checkpoint.json \
  --topk 10 \
  --num_beams 10 \
  --batch_size 4 \
  --max_depth 14 \
  --max_reactants 5 \
  --max_history_steps 6 \
  --fp16 \
  --overwrite \
  > build_chemdfm_valid.log 2>&1 &
```

The resulting directory is:

```text
outputs/chemdfm_dataset/
├── train_chemdfm.jsonl
├── valid_chemdfm.jsonl
├── train_report.json
├── valid_report.json
├── train_checkpoint.json
└── valid_checkpoint.json
```

---

## 2. Train the Search State Aware ChemDFM

ChemDFM is adapted using LoRA and a listwise candidate ranking objective.

```bash
mkdir -p ./outputs/chemdfm_adapter

nohup env CUDA_VISIBLE_DEVICES=0 python train_chemdfm.py \
  --train_jsonl ./outputs/chemdfm_dataset/train_chemdfm.jsonl \
  --valid_jsonl ./outputs/chemdfm_dataset/valid_chemdfm.jsonl \
  --model_dir ./ChemDFM/model \
  --output_dir ./outputs/chemdfm_adapter \
  --max_candidates 10 \
  --max_history_steps 6 \
  --max_length 2048 \
  --include_molt5_rank \
  --include_validity \
  --include_stock \
  --randomize_candidate_ids \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --batch_size 1 \
  --eval_batch_size 2 \
  --grad_accum_steps 32 \
  --epochs 3 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --early_stop_patience 2 \
  --logging_steps 20 \
  --save_every_epoch \
  > train_chemdfm.log 2>&1 &
```

The trained LoRA checkpoints are saved under:

```text
outputs/chemdfm_adapter/
```

Use the selected checkpoint as `--chemdfm_adapter` during multistep evaluation.

---

## 3. Multistep RetroRoute Evaluation

The final evaluation combines the URPO refined MolT5 proposal model with the adapted ChemDFM decision model.

```bash
mkdir -p ./outputs/multistep_eval

nohup env CUDA_VISIBLE_DEVICES=3 python multistep_test.py \
  --test_multistep_json ./dataset/processed/test_dataset_grouped.json \
  --generator_model_dir ./outputs/molt5_urpo/checkpoint-epoch-6 \
  --chemdfm_base_model ./ChemDFM/model \
  --chemdfm_adapter ./outputs/chemdfm_adapter \
  --stock_path ./dataset/zinc_stock_17_04_20.hdf5 \
  --output_dir ./outputs/multistep_eval \
  --single_step_topk 10 \
  --single_step_num_beams 10 \
  --ranker_num_candidates 10 \
  --ranker_keep_topm 5 \
  --route_beam_size 5 \
  --eval_topk 5 \
  --ranker_score_weight 1.0 \
  --molt5_score_weight 0.0 \
  --length_norm_gamma 0.0 \
  --depth_penalty 0.0 \
  --max_depth 14 \
  --max_search_steps 14 \
  --search_depth_limit_mode global \
  --max_reactants 5 \
  --max_history_steps 6 \
  --expand_all_unsolved \
  --stop_when_enough_complete \
  --skip_bad_actions \
  --run_greedy_dfs \
  --greedy_depth_limit_mode gt \
  --greedy_num_candidates 5 \
  --sample_size -1 \
  --sample_seed 42 \
  --load_chemdfm_in_4bit \
  --chemdfm_bf16 \
  --fp16 \
  > multistep_test.log 2>&1 &
```

The main beam search uses a global maximum search depth of 14. Greedy DFS uses the ground truth route depth as the expansion limit following the RetroBench evaluation protocol.

## Recommended Output Structure

```text
outputs/
├── molt5_route_context/
├── molt5_topk/
├── route_candidates_top20/
├── utility_scores/
├── urpo_data/
├── molt5_urpo/
├── chemdfm_dataset/
├── chemdfm_adapter/
└── multistep_eval/
```

## Notes

The reported configuration uses:

```text
Proposal candidates per expansion      10
ChemDFM candidates                     10
Retained actions                        5
Route beam size                         5
Maximum search depth                   14
Maximum search rounds                  14
Maximum reactants                       5
Maximum route history                   6
Evaluation Top K                        5
Random seed                            42
```

The reported URPO proposal checkpoint is epoch 6.

## Acknowledgements

RetroRoute builds on RetroBench and FusionRetro, RetroInText, MolT5, ReactionT5, and ChemDFM. Please cite the corresponding original works when using these resources.
