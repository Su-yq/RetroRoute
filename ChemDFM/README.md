# ChemDFM

RetroRoute uses ChemDFM as the chemistry language model for search state aware candidate reranking.

The pretrained ChemDFM weights are not included in this repository.

## Model Source

Download the pretrained model from the official ChemDFM release:

```text
OpenDFM/ChemDFM
```

Follow the upstream ChemDFM instructions to obtain a Hugging Face compatible local checkpoint.

## Expected Directory

Place the pretrained model under:

```text
ChemDFM/model/
```

For example:

```text
ChemDFM/
├── README.md
└── model/
    ├── config.json
    ├── tokenizer files
    └── model weights
```

## Role in RetroRoute

For each search decision, ChemDFM receives information about:

```text
final target molecule
current molecule
current search depth
maximum search depth
remaining search budget
unresolved frontier
recent route history
candidate reactions
MolT5 candidate rank
structural validity
purchasable reactant count
```

The candidate identifiers are randomized during training to reduce dependence on candidate position.

ChemDFM is adapted with LoRA using a listwise ranking objective over the candidate reaction set.

## Training Configuration

The reported configuration uses:

```text
Maximum candidates       10
Maximum history steps     6
Maximum input length   2048
Training epochs            3
Learning rate           1e-4
Training batch size         1
Evaluation batch size       2
Gradient accumulation      32
Weight decay             0.01
Warmup ratio             0.03
Maximum gradient norm     1.0
LoRA rank                  16
LoRA alpha                 32
LoRA dropout             0.05
```

The model is loaded with 4 bit quantization and trained with gradient checkpointing.

## Training

ChemDFM dataset construction and LoRA training are performed from the repository root using:

```text
build_chemdfm_dataset.py
train_chemdfm.py
```

The complete commands are provided in the root:

```text
README.md
```

The trained adapter is then supplied to:

```text
multistep_test.py
```

for the final RetroRoute multistep evaluation.
