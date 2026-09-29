# ReactionT5 Forward Reaction Model

RetroRoute uses ReactionT5 as an independent forward reaction model during offline utility evaluation.

The forward model is used to estimate whether a proposed retrosynthetic reaction is consistent with reconstructing the current product.

ReactionT5 is not used as the final multistep branch decision model.

## Model Source

The model follows the public ReactionT5 implementation:

```text
sagawatatsuya/ReactionT5
```

ReactionT5 is pretrained on chemical reaction data and supports forward reaction prediction.

## Expected Directory

Download the forward reaction checkpoint and place it under:

```text
ReactionT5/model/
```

For example:

```text
ReactionT5/
├── README.md
└── model/
    ├── config.json
    ├── tokenizer files
    └── model weights
```

## Usage in RetroRoute

ReactionT5 is used during utility scoring of MolT5 generated reaction candidates.

In the reported configuration, the forward evaluation settings are:

```text
forward_topk       = 5
forward_num_beams  = 5
forward_batch_size = 16
precision          = FP16
```

The resulting forward plausibility score is combined with other chemical and route related signals before utility supported alternatives are selected for URPO.

The scored candidate file required by the public training pipeline is expected at:

```text
outputs/utility_scores/train_scored_candidates.jsonl
```

See:

```text
single_step_model/README.md
```

for the complete URPO workflow.
