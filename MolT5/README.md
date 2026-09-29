# MolT5 Model

RetroRoute uses MolT5 as its retrosynthetic reaction proposal model.

The model weights are not stored in this repository.

## Model Source

The initialization used by RetroRoute follows the MolT5 checkpoint used in the RetroInText implementation.

Relevant upstream resources include:

```text
Kin-CL/RetroInText
blender-nlp/MolT5
laituan245/molt5-base
```

The RetroInText release also provides the MolT5 checkpoint used in its experiments.

## Expected Directory

Download the model and place the Hugging Face compatible checkpoint under:

```text
MolT5/model/
```

A typical directory should contain files such as:

```text
MolT5/
├── README.md
└── model/
    ├── config.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── spiece.model
    └── model weights
```

The exact model weight filename depends on the downloaded checkpoint.

## Usage in RetroRoute

The model is first adapted using:

```text
single_step_model/train_molt5_route_context_sft.py
```

with:

```text
--model_dir ./MolT5/model
```

The resulting planning scaffold model is subsequently refined with Utility Regularized Positive Optimization.

The proposal model generates reaction candidates during both ChemDFM dataset construction and final multistep search.

See:

```text
single_step_model/README.md
```

for the full proposal training pipeline.
