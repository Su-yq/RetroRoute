#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-end multi-step retrosynthesis evaluation:
    MolT5 proposal generator
      -> route-contextual ChemDFM listwise reranker
      -> existing beam search
      -> grouped multi-reference exact-match evaluation

This script is standalone with respect to project-local Python modules.

Keys of multistep_test.py:
1. A ChemDFM LoRA/QLoRA adapter.
2. ChemDFM sees the FULL current search context:
      final target + current molecule + frontier + route history + candidates.
3. MolT5 raw proposals can be cached by (target, current, depth), but ChemDFM
   contextual reranking is NOT cached by that local key.
4. Supports deterministic random subset evaluation, e.g. 580 products.
5. Saves selected_sample_ids.json so different checkpoints/baselines can be
   evaluated on exactly the same subset.
6. Reports overall Top-k exact match AND depth-stratified Top-k exact match.
7. Optionally runs a RetroInText/FusionRetro-style Greedy DFS evaluation for
   Figure-2-style accuracy-vs-ground-truth-depth analysis.
8. Supports a GT-depth search cap (paper-compatible) or the original global
   max-depth cap.
"""

import argparse
import csv
import gc
import inspect
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel

try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False
    Chem = None


ATOM_MAP_RE = re.compile(r":\d+(?=\])")
TASK_PREFIX = "Please predict the reactant of the product:\n"
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(rows: List[Dict[str, Any]], path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def yesno(x: bool) -> str:
    return "yes" if bool(x) else "no"


# ---------------------------------------------------------------------------
# Chemistry / normalization
# ---------------------------------------------------------------------------

def strip_atom_mapping(smi: str) -> str:
    if smi is None:
        return ""
    return ATOM_MAP_RE.sub("", str(smi).strip())


def split_side(side: str) -> List[str]:
    if side is None:
        return []
    side = str(side).strip().replace(" ", "").rstrip(".")
    if not side:
        return []
    return [x for x in side.split(".") if x.strip()]


def canonicalize_smiles(smi: str) -> Optional[str]:
    if smi is None:
        return None

    smi = strip_atom_mapping(str(smi).strip().replace(" ", ""))
    if not smi:
        return None

    if not HAS_RDKIT:
        return smi

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def canonicalize_side(side: str) -> Optional[str]:
    parts = split_side(side)
    if not parts:
        return None

    out = []
    for p in parts:
        cp = canonicalize_smiles(p)
        if cp is None:
            return None
        out.append(cp)

    return ".".join(sorted(out))


def normalize_prediction_text(text: str) -> Optional[str]:
    if text is None:
        return None

    text = str(text).strip()
    if ">>" in text:
        text = text.split(">>", 1)[1]

    text = re.sub(r"\s+", "", text)
    return canonicalize_side(text)


def mol_to_inchikey_full(smiles: str) -> Optional[str]:
    if not HAS_RDKIT:
        return canonicalize_smiles(smiles)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)
        return key if key else None
    except Exception:
        return None


def mol_to_inchikey14(smiles: str) -> Optional[str]:
    key = mol_to_inchikey_full(smiles)
    return key[:14] if key else None


def smiles_list_to_full_key_set(smiles_list: List[str]) -> Set[str]:
    keys = set()
    for smi in smiles_list:
        key = mol_to_inchikey_full(smi)
        if key:
            keys.add(key)
    return keys


def smiles_list_to_key14_set(smiles_list: List[str]) -> Set[str]:
    keys = set()
    for smi in smiles_list:
        key = mol_to_inchikey14(smi)
        if key:
            keys.add(key)
    return keys


def valid_side(side: str) -> bool:
    parts = split_side(side)
    if not parts:
        return False

    if not HAS_RDKIT:
        return True

    for p in parts:
        try:
            mol = Chem.MolFromSmiles(p)
            if mol is None:
                return False
        except Exception:
            return False

    return True


# ---------------------------------------------------------------------------
# Stock
# ---------------------------------------------------------------------------

def load_stock_keys(stock_path: str) -> Set[str]:
    if not stock_path:
        return set()

    try:
        import pandas as pd
        df = pd.read_hdf(stock_path, key="table")
        return set(str(x)[:14] for x in df.inchi_key.values)
    except Exception as exc:
        print(f"[WARN] failed to load stock from {stock_path}: {exc}")
        return set()


def is_in_stock(smiles: str, stock_keys: Set[str]) -> bool:
    key14 = mol_to_inchikey14(smiles)
    return key14 is not None and key14 in stock_keys


# ---------------------------------------------------------------------------
# MolT5 prompt / basic candidate features
# ---------------------------------------------------------------------------

def build_molt5_route_context_prompt(
    target_smiles: str,
    current_smiles: str,
    current_depth: int,
    max_depth: int,
) -> str:
    target = canonicalize_smiles(target_smiles) or ""
    current = canonicalize_smiles(current_smiles) or ""

    return (
        f"{TASK_PREFIX}"
        f"<target> {target}\n"
        f"<current> {current}\n"
        f"<current_depth> {current_depth}\n"
        f"<max_depth> {max_depth}\n"
        f"<goal> purchasable_starting_materials"
    )


def compute_basic_features(
    candidate: str,
    current: str,
    beam_rank: int,
    topk: int,
    sequence_score: Optional[float],
    candidate_token_length: int,
    current_depth: int,
    stock_keys: Set[str],
    max_reactants: int,
) -> Dict[str, float]:
    parts = split_side(candidate)
    num_reactants = len(parts)

    valid_score = 1.0 if valid_side(candidate) else 0.0
    current_can = canonicalize_smiles(current) or current
    candidate_can = canonicalize_side(candidate) or ""

    copy_current = 1.0 if candidate_can == current_can else 0.0
    too_many_reactants = 1.0 if num_reactants > max_reactants else 0.0
    empty_candidate = 1.0 if not candidate_can else 0.0

    bad_action_penalty = 1.0 if (
        valid_score < 1.0
        or copy_current > 0.0
        or too_many_reactants > 0.0
        or empty_candidate > 0.0
    ) else 0.0

    if len(stock_keys) == 0 or num_reactants == 0:
        stock_count = 0
        stock_ratio = 0.0
    else:
        stock_count = sum(1 for p in parts if is_in_stock(p, stock_keys))
        stock_ratio = stock_count / max(1, num_reactants)

    generator_score = 0.0 if sequence_score is None else float(sequence_score)
    generator_score_per_token = generator_score / max(1, candidate_token_length)

    return {
        "beam_rank": float(beam_rank),
        "sequence_score": generator_score,
        "generator_score_per_token": float(generator_score_per_token),
        "valid_score": float(valid_score),
        "bad_action_penalty": float(bad_action_penalty),
        "copy_current": float(copy_current),
        "too_many_reactants": float(too_many_reactants),
        "num_reactants": float(num_reactants),
        "stock_count": float(stock_count),
        "stock_ratio": float(stock_ratio),
        "current_depth": float(current_depth),
    }


# ---------------------------------------------------------------------------
# Grouped multi-reference ground truth
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthReference:
    reference_id: str
    route_id: str
    starting_materials: List[str]
    intermediates: List[str]
    starting_material_keys: Set[str]
    intermediate_keys: Set[str]
    depth: int
    num_steps: int


@dataclass
class GroundTruthSample:
    sample_id: str
    product_name: str
    target_smiles: str
    route_references: List[GroundTruthReference]
    unique_starting_materials: List[List[str]]
    unique_starting_material_key_sets: List[Set[str]]
    depth: int
    declared_num_routes: int


def canonicalize_smiles_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = split_side(values)
    if not isinstance(values, (list, tuple, set)):
        return []

    canonical = []
    for value in values:
        smi = canonicalize_smiles(value)
        if smi:
            canonical.append(smi)
    return sorted(set(canonical))


def make_ground_truth_reference(
    route: Dict[str, Any],
    sample_id: str,
    route_index: int,
    default_depth: int,
) -> Optional[GroundTruthReference]:
    materials = canonicalize_smiles_list(route.get("materials", []))
    intermediates = canonicalize_smiles_list(route.get("intermediates", []))

    material_keys = smiles_list_to_full_key_set(materials)
    intermediate_keys = smiles_list_to_full_key_set(intermediates)
    if not material_keys:
        return None

    steps = route.get("steps", [])
    num_steps = len(steps) if isinstance(steps, list) else 0

    route_depth = safe_int(route.get("depth", -1), default=-1)
    if route_depth < 0:
        route_depth = num_steps if num_steps > 0 else default_depth

    route_id = str(route.get("route_id", route_index))

    return GroundTruthReference(
        reference_id=f"{sample_id}:route-{route_id}",
        route_id=route_id,
        starting_materials=materials,
        intermediates=intermediates,
        starting_material_keys=material_keys,
        intermediate_keys=intermediate_keys,
        depth=route_depth,
        num_steps=num_steps,
    )


def build_ground_truth_samples(raw_items: List[Dict[str, Any]]) -> List[GroundTruthSample]:
    if not isinstance(raw_items, list):
        raise ValueError("test_dataset_grouped.json 顶层必须是 list。")

    samples: List[GroundTruthSample] = []
    seen_products: Set[str] = set()

    invalid_products = 0
    invalid_routes = 0
    duplicate_products = 0
    route_count_mismatches = 0

    for item_index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            invalid_products += 1
            continue

        product = canonicalize_smiles(item.get("product", ""))
        routes = item.get("routes", None)

        if not product or not isinstance(routes, list):
            invalid_products += 1
            continue

        if product in seen_products:
            duplicate_products += 1
            continue
        seen_products.add(product)

        sample_id = str(item.get("id", item_index))
        product_depth = safe_int(item.get("depth", -1), default=-1)
        declared_num_routes = safe_int(
            item.get("num_routes", len(routes)),
            default=len(routes),
        )

        if declared_num_routes != len(routes):
            route_count_mismatches += 1

        route_references: List[GroundTruthReference] = []
        seen_route_pairs = set()
        unique_material_map: Dict[Tuple[str, ...], List[str]] = {}

        for route_index, route in enumerate(routes):
            if not isinstance(route, dict):
                invalid_routes += 1
                continue

            ref = make_ground_truth_reference(
                route=route,
                sample_id=sample_id,
                route_index=route_index,
                default_depth=product_depth,
            )
            if ref is None:
                invalid_routes += 1
                continue

            paired_key = (
                tuple(sorted(ref.starting_material_keys)),
                tuple(sorted(ref.intermediate_keys)),
            )
            if paired_key not in seen_route_pairs:
                seen_route_pairs.add(paired_key)
                route_references.append(ref)

            materials_key = tuple(sorted(ref.starting_material_keys))
            unique_material_map.setdefault(materials_key, ref.starting_materials)

        if not unique_material_map:
            invalid_products += 1
            continue

        samples.append(
            GroundTruthSample(
                sample_id=sample_id,
                product_name=str(item.get("product_name", "")),
                target_smiles=product,
                route_references=route_references,
                unique_starting_materials=list(unique_material_map.values()),
                unique_starting_material_key_sets=[
                    set(x) for x in unique_material_map.keys()
                ],
                depth=product_depth,
                declared_num_routes=declared_num_routes,
            )
        )

    print("===== Grouped ground-truth parsing =====")
    print(f"raw product rows              : {len(raw_items)}")
    print(f"valid unique products         : {len(samples)}")
    print(f"invalid product rows          : {invalid_products}")
    print(f"invalid route rows            : {invalid_routes}")
    print(f"duplicate canonical products  : {duplicate_products}")
    print(f"num_routes mismatches         : {route_count_mismatches}")

    return samples


def match_route_references(
    predicted_starting_keys: Set[str],
    predicted_intermediate_keys: Set[str],
    references: List[GroundTruthReference],
) -> Tuple[List[str], List[str]]:
    starting_matches = []
    paired_matches = []

    for ref in references:
        if predicted_starting_keys == ref.starting_material_keys:
            starting_matches.append(ref.reference_id)
            if predicted_intermediate_keys == ref.intermediate_keys:
                paired_matches.append(ref.reference_id)

    return starting_matches, paired_matches


def match_unique_starting_material_reference(
    predicted_starting_keys: Set[str],
    reference_sets: List[Set[str]],
) -> List[int]:
    return [
        idx
        for idx, ref_keys in enumerate(reference_sets)
        if predicted_starting_keys == ref_keys
    ]


# ---------------------------------------------------------------------------
# Search state / route score
# ---------------------------------------------------------------------------

@dataclass
class SearchState:
    frontier: List[Tuple[str, int]]
    expanded: Set[str]
    score: float
    actions: List[Dict[str, Any]] = field(default_factory=list)

    def is_complete(self, stock_keys: Set[str]) -> bool:
        return all(is_in_stock(mol, stock_keys) for mol, _ in self.frontier)

    def starting_materials(self) -> List[str]:
        return sorted([mol for mol, _ in self.frontier])

    def intermediates(self, root_target: str) -> List[str]:
        return sorted([mol for mol in self.expanded if mol != root_target])

    def state_key(self) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[str, ...]]:
        return (
            tuple(sorted(self.frontier)),
            tuple(sorted(self.expanded)),
        )


def compute_route_score(actions, args):
    n = max(1, len(actions))
    denom = float(n) ** float(args.length_norm_gamma)

    ranker_sum = 0.0
    molt5_sum = 0.0

    for a in actions:
        ranker_sum += safe_float(a.get("ranker_logprob", 0.0), 0.0)

        if args.molt5_score_field == "sequence_score":
            molt5_sum += safe_float(a.get("sequence_score", 0.0), 0.0)
        else:
            molt5_sum += safe_float(
                a.get("generator_score_per_token", 0.0),
                0.0,
            )

    ranker_score = ranker_sum / denom
    molt5_score = molt5_sum / denom

    score = (
        float(args.ranker_score_weight) * ranker_score
        + float(args.molt5_score_weight) * molt5_score
        - float(args.depth_penalty) * float(n)
    )

    return float(score)


# ---------------------------------------------------------------------------
# ChemDFM prompt and candidate-ID scoring
# ---------------------------------------------------------------------------

def history_from_search_state(
    state: SearchState,
    max_history_steps: int,
) -> List[Dict[str, Any]]:
    actions = state.actions[-max_history_steps:] if max_history_steps > 0 else state.actions
    history = []

    start = len(state.actions) - len(actions)

    for offset, action in enumerate(actions, start=1):
        reactants = action.get("reactants", []) or []
        reactants = [
            canonicalize_smiles(x) or str(x)
            for x in reactants
            if str(x).strip()
        ]
        reactants = sorted(reactants)

        history.append({
            "step": start + offset,
            "product_smiles": action.get("current", ""),
            "reactants_str": ".".join(reactants),
            "depth": safe_int(action.get("current_depth", 0), 0),
        })

    return history


def build_chemdfm_prompt(
    target: str,
    current: str,
    current_depth: int,
    state: SearchState,
    candidates: List[Dict[str, Any]],
    stock_keys: Set[str],
    max_depth: int,
    max_history_steps: int,
) -> str:
    lines = [
        "[Round 0]",
        "Human: You are a retrosynthesis candidate reranker.",
        "",
        "Select the most promising one-step retrosynthetic candidate for the CURRENT molecule, considering its role in the FULL multi-step synthesis.",
        "The objective is to maximize the likelihood of completing the final target from purchasable starting materials within the search budget.",
        "Use the route context when comparing candidates.",
        "Do not propose new reactants. Choose only from the candidates provided.",
        "",
        "[FINAL TARGET]",
        target,
        "",
        "[SEARCH STATE]",
        f"Current molecule: {current}",
        f"Current depth: {current_depth}",
        f"Maximum search depth: {max_depth}",
        f"Remaining depth budget: {max(0, max_depth - current_depth)}",
        "",
        "Current frontier:",
    ]

    if state.frontier:
        for i, (mol, depth) in enumerate(state.frontier, start=1):
            lines.append(
                f"  F{i}. {mol} | depth={depth} | purchasable={yesno(is_in_stock(mol, stock_keys))}"
            )
    else:
        lines.append("  None")

    lines.extend(["", "[ROUTE HISTORY]"])

    history = history_from_search_state(
        state=state,
        max_history_steps=max_history_steps,
    )

    if history:
        for h in history:
            lines.append(
                f"  Step {h['step']}: {h['product_smiles']} -> {h['reactants_str']}"
            )
    else:
        lines.append("  None")

    lines.extend(["", "[CANDIDATES]"])

    for idx, cand in enumerate(candidates):
        cid = LABELS[idx]
        feats = cand.get("features", {}) or {}
        n_reactants = int(safe_float(feats.get("num_reactants", 0.0), 0.0))
        stock_count = int(safe_float(feats.get("stock_count", 0.0), 0.0))
        valid = safe_float(feats.get("valid_score", 0.0), 0.0) > 0.5

        lines.append(f"{cid}. Reactants: {cand['reactants_str']}")
        lines.append(f"   MolT5 rank: {int(cand.get('beam_rank', idx + 1))}")
        lines.append(f"   Structurally valid: {yesno(valid)}")
        lines.append(
            f"   Purchasable reactants: {stock_count}/{max(0, n_reactants)}"
        )

    lines.extend([
        "",
        "[OUTPUT]",
        "Return only the ID of the preferred candidate.",
        "Assistant:",
    ])

    return "\n".join(lines)


def infer_single_token_label_ids(
    tokenizer,
    num_candidates: int,
) -> Tuple[List[str], List[int]]:
    base = LABELS[:num_candidates]

    for texts in (
        [f" {x}" for x in base],
        base,
    ):
        ids = []
        ok = True

        for text in texts:
            enc = tokenizer.encode(text, add_special_tokens=False)
            if len(enc) != 1:
                ok = False
                break
            ids.append(int(enc[0]))

        if ok and len(set(ids)) == len(ids):
            return texts, ids

    raise RuntimeError(
        "Could not infer one-token candidate labels A..J from ChemDFM tokenizer."
    )


def detect_logits_keep_arg(model) -> Optional[str]:
    objects = [model]

    if hasattr(model, "get_base_model"):
        try:
            objects.append(model.get_base_model())
        except Exception:
            pass

    for obj in objects:
        try:
            params = inspect.signature(obj.forward).parameters
        except Exception:
            continue

        if "logits_to_keep" in params:
            return "logits_to_keep"
        if "num_logits_to_keep" in params:
            return "num_logits_to_keep"

    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MultiStepChemDFMEvaluator:
    def __init__(self, args):
        self.args = args

        self.device = torch.device(
            args.device
            if torch.cuda.is_available() and args.device.startswith("cuda")
            else "cpu"
        )

        self.stock_keys = load_stock_keys(args.stock_path)
        print(f"[INFO] loaded stock keys: {len(self.stock_keys)}")

        # ---------------- MolT5 ----------------
        print(f"[INFO] loading MolT5 generator: {args.generator_model_dir}")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(
            args.generator_model_dir
        )
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            args.generator_model_dir
        )
        self.generator.to(self.device)

        if args.fp16 and self.device.type == "cuda":
            self.generator.half()

        self.generator.eval()

        # ---------------- ChemDFM ----------------
        print(f"[INFO] loading ChemDFM base: {args.chemdfm_base_model}")

        if args.chemdfm_bf16:
            chemdfm_dtype = torch.bfloat16
        elif args.chemdfm_fp16:
            chemdfm_dtype = torch.float16
        else:
            chemdfm_dtype = torch.float32

        quant_config = None
        if args.load_chemdfm_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if args.chemdfm_bf16
                    else torch.float16
                ),
            )

        base_kwargs = {
            "trust_remote_code": True,
        }

        if quant_config is not None:
            base_kwargs["quantization_config"] = quant_config
            base_kwargs["device_map"] = {"": 0}
        else:
            base_kwargs["torch_dtype"] = chemdfm_dtype

        self.chemdfm = AutoModelForCausalLM.from_pretrained(
            args.chemdfm_base_model,
            **base_kwargs,
        )

        if quant_config is None:
            self.chemdfm.to(self.device)

        print(f"[INFO] loading ChemDFM adapter: {args.chemdfm_adapter}")
        self.chemdfm = PeftModel.from_pretrained(
            self.chemdfm,
            args.chemdfm_adapter,
        )
        self.chemdfm.eval()

        self.chemdfm_tokenizer = AutoTokenizer.from_pretrained(
            args.chemdfm_base_model,
            trust_remote_code=True,
            use_fast=False,
        )

        if self.chemdfm_tokenizer.pad_token_id is None:
            if self.chemdfm_tokenizer.eos_token_id is not None:
                self.chemdfm_tokenizer.pad_token = self.chemdfm_tokenizer.eos_token
            else:
                self.chemdfm_tokenizer.pad_token = self.chemdfm_tokenizer.unk_token

        self.chemdfm_tokenizer.padding_side = "left"
        self.chemdfm_tokenizer.truncation_side = "left"

        # Prefer the exact label token IDs saved by the trainer.
        trainer_state_path = Path(args.chemdfm_adapter) / "trainer_state.json"

        if trainer_state_path.exists():
            trainer_state = load_json(str(trainer_state_path))
            saved_ids = trainer_state.get("label_token_ids", None)
            saved_texts = trainer_state.get("label_texts", None)

            if (
                isinstance(saved_ids, list)
                and len(saved_ids) >= args.ranker_num_candidates
            ):
                self.label_token_ids = [
                    int(x)
                    for x in saved_ids[:args.ranker_num_candidates]
                ]
                self.label_texts = (
                    saved_texts[:args.ranker_num_candidates]
                    if isinstance(saved_texts, list)
                    else LABELS[:args.ranker_num_candidates]
                )
            else:
                self.label_texts, self.label_token_ids = (
                    infer_single_token_label_ids(
                        self.chemdfm_tokenizer,
                        args.ranker_num_candidates,
                    )
                )
        else:
            self.label_texts, self.label_token_ids = (
                infer_single_token_label_ids(
                    self.chemdfm_tokenizer,
                    args.ranker_num_candidates,
                )
            )

        print(
            "[INFO] ChemDFM label tokens: "
            + str(list(zip(self.label_texts, self.label_token_ids)))
        )

        self.label_token_tensor = torch.tensor(
            self.label_token_ids,
            dtype=torch.long,
            device=self.device,
        )

        self.logits_keep_arg = detect_logits_keep_arg(self.chemdfm)
        print(
            f"[INFO] last-logits optimization arg: {self.logits_keep_arg}"
        )

        # MolT5 proposals depend only on (target, current, depth), so this cache
        # is inference-safe. We intentionally do NOT cache contextual ChemDFM
        # rankings under this local key.
        self.generator_cache: Dict[
            Tuple[str, str, int],
            List[Dict[str, Any]]
        ] = {}

        self.stats = {
            "molt5_generation_calls": 0,
            "molt5_cache_hits": 0,
            "chemdfm_rerank_calls": 0,
        }

    @torch.no_grad()
    def generate_raw_candidates(
        self,
        target: str,
        current: str,
        current_depth: int,
    ) -> List[Dict[str, Any]]:
        cache_key = (target, current, current_depth)

        if cache_key in self.generator_cache:
            self.stats["molt5_cache_hits"] += 1
            return [
                dict(x)
                for x in self.generator_cache[cache_key]
            ]

        self.stats["molt5_generation_calls"] += 1

        prompt = build_molt5_route_context_prompt(
            target_smiles=target,
            current_smiles=current,
            current_depth=current_depth,
            max_depth=self.args.max_depth,
        )

        enc = self.gen_tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_source_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        gen_kwargs = {
            "num_beams": self.args.single_step_num_beams,
            "num_return_sequences": self.args.single_step_topk,
            "early_stopping": True,
            "max_length": self.args.max_target_length,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        if self.args.fp16 and self.device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                gen_out = self.generator.generate(
                    **enc,
                    **gen_kwargs,
                )
        else:
            gen_out = self.generator.generate(
                **enc,
                **gen_kwargs,
            )

        decoded = self.gen_tokenizer.batch_decode(
            gen_out.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if (
            hasattr(gen_out, "sequences_scores")
            and gen_out.sequences_scores is not None
        ):
            seq_scores = (
                gen_out.sequences_scores
                .detach()
                .cpu()
                .tolist()
            )
        else:
            seq_scores = [None for _ in decoded]

        candidates = []
        seen = set()

        for j, raw in enumerate(decoded):
            beam_rank = j + 1
            reactants = normalize_prediction_text(raw)

            if not reactants:
                continue

            if reactants in seen:
                continue

            seen.add(reactants)

            cand_token_length = len(
                self.gen_tokenizer(
                    reactants,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )

            features = compute_basic_features(
                candidate=reactants,
                current=current,
                beam_rank=beam_rank,
                topk=self.args.single_step_topk,
                sequence_score=seq_scores[j],
                candidate_token_length=cand_token_length,
                current_depth=current_depth,
                stock_keys=self.stock_keys,
                max_reactants=self.args.max_reactants,
            )

            candidates.append({
                "reactants_str": reactants,
                "raw_text": raw,
                "beam_rank": beam_rank,
                "sequence_score": (
                    None
                    if seq_scores[j] is None
                    else float(seq_scores[j])
                ),
                "features": features,
            })

        self.generator_cache[cache_key] = [
            dict(x)
            for x in candidates
        ]

        del enc, gen_out
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return candidates

    @torch.no_grad()
    def rerank_with_chemdfm(
        self,
        target: str,
        current: str,
        current_depth: int,
        state: SearchState,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        self.stats["chemdfm_rerank_calls"] += 1

        num_candidates = min(
            self.args.ranker_num_candidates,
            len(candidates),
        )

        candidates = candidates[:num_candidates]

        if not candidates:
            return []

        prompt = build_chemdfm_prompt(
            target=target,
            current=current,
            current_depth=current_depth,
            state=state,
            candidates=candidates,
            stock_keys=self.stock_keys,
            max_depth=self.args.max_depth,
            max_history_steps=self.args.max_history_steps,
        )

        enc = self.chemdfm_tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.chemdfm_max_length,
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }

        if self.logits_keep_arg:
            kwargs[self.logits_keep_arg] = 1

        amp_enabled = bool(
            self.device.type == "cuda"
            and (
                self.args.chemdfm_bf16
                or self.args.chemdfm_fp16
            )
        )

        amp_dtype = (
            torch.bfloat16
            if self.args.chemdfm_bf16
            else torch.float16
        )

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = self.chemdfm(**kwargs)

        last_logits = outputs.logits[:, -1, :].float()[0]

        label_ids = self.label_token_tensor[:num_candidates]
        valid_scores = last_logits.index_select(
            dim=-1,
            index=label_ids,
        )

        log_probs = F.log_softmax(
            valid_scores,
            dim=0,
        ).detach().cpu().tolist()

        order = torch.argsort(
            valid_scores.detach().cpu(),
            descending=True,
        ).tolist()

        ranked = []

        for rank_pos, idx in enumerate(order, start=1):
            c = dict(candidates[idx])
            c["ranker_rank"] = rank_pos
            c["ranker_score"] = float(
                valid_scores[idx].detach().cpu()
            )
            c["ranker_logprob"] = float(log_probs[idx])
            c["chemdfm_candidate_id"] = LABELS[idx]
            ranked.append(c)

        return ranked

    def get_search_depth_limit(self, gt: GroundTruthSample) -> int:
        if (
            self.args.search_depth_limit_mode == "gt"
            and gt.depth is not None
            and int(gt.depth) >= 0
        ):
            return int(gt.depth)
        return int(self.args.max_depth)

    def expand_state(
        self,
        gt: GroundTruthSample,
        state: SearchState,
        depth_limit: Optional[int] = None,
    ) -> List[SearchState]:
        new_states = []

        unsolved = [
            (idx, mol, depth)
            for idx, (mol, depth) in enumerate(state.frontier)
            if not is_in_stock(mol, self.stock_keys)
        ]

        if not unsolved:
            return new_states

        if not self.args.expand_all_unsolved:
            unsolved = unsolved[:1]

        if depth_limit is None:
            depth_limit = self.get_search_depth_limit(gt)

        for idx, current, current_depth in unsolved:
            if self.args.search_depth_limit_mode == "gt":
                # Match RetroInText/FusionRetro: stop only when depth > GT depth.
                if current_depth > depth_limit:
                    continue
            else:
                # Preserve the original behavior for the global --max_depth cap.
                if current_depth >= depth_limit:
                    continue

            raw_candidates = self.generate_raw_candidates(
                target=gt.target_smiles,
                current=current,
                current_depth=current_depth,
            )

            ranked_candidates = self.rerank_with_chemdfm(
                target=gt.target_smiles,
                current=current,
                current_depth=current_depth,
                state=state,
                candidates=raw_candidates,
            )

            if not ranked_candidates:
                continue

            kept = ranked_candidates[
                :self.args.ranker_keep_topm
            ]

            for cand in kept:
                features = cand.get("features", {}) or {}

                if (
                    self.args.skip_bad_actions
                    and safe_float(
                        features.get(
                            "bad_action_penalty",
                            0.0,
                        )
                    ) > 0.0
                ):
                    continue

                reactants = split_side(
                    cand["reactants_str"]
                )

                if not reactants:
                    continue

                if current in reactants:
                    continue

                new_frontier = list(state.frontier)
                new_frontier.pop(idx)

                for r in reactants:
                    cr = canonicalize_smiles(r)
                    if cr:
                        new_frontier.append(
                            (cr, current_depth + 1)
                        )

                new_frontier = sorted(new_frontier)

                new_expanded = set(state.expanded)
                new_expanded.add(current)

                action = {
                    "current": current,
                    "current_depth": current_depth,
                    "reactants": sorted(reactants),
                    "ranker_rank": cand.get("ranker_rank"),
                    "ranker_score": cand.get("ranker_score"),
                    "ranker_logprob": cand.get("ranker_logprob"),
                    "chemdfm_candidate_id": cand.get(
                        "chemdfm_candidate_id"
                    ),
                    "sequence_score": cand.get("sequence_score"),
                    "generator_score_per_token": (
                        cand.get("features", {})
                        .get(
                            "generator_score_per_token"
                        )
                    ),
                    "beam_rank": cand.get("beam_rank"),
                }

                new_actions = state.actions + [action]

                new_score = compute_route_score(
                    new_actions,
                    self.args,
                )

                new_states.append(
                    SearchState(
                        frontier=new_frontier,
                        expanded=new_expanded,
                        score=new_score,
                        actions=new_actions,
                    )
                )

        return new_states

    def search_one(
        self,
        gt: GroundTruthSample,
    ) -> List[SearchState]:
        root = gt.target_smiles

        init_state = SearchState(
            frontier=[(root, 0)],
            expanded=set(),
            score=0.0,
            actions=[],
        )

        active = [init_state]
        completed = []
        depth_limit = self.get_search_depth_limit(gt)

        for _ in range(self.args.max_search_steps):
            all_next = []

            for state in active:
                if state.is_complete(self.stock_keys):
                    completed.append(state)
                    continue

                expansions = self.expand_state(
                    gt,
                    state,
                    depth_limit=depth_limit,
                )
                all_next.extend(expansions)

            if not all_next:
                break

            # Keep the old deterministic state deduplication rule unchanged.
            best_by_key = {}

            for st in all_next:
                key = st.state_key()

                if (
                    key not in best_by_key
                    or st.score > best_by_key[key].score
                ):
                    best_by_key[key] = st

            candidates = sorted(
                best_by_key.values(),
                key=lambda x: x.score,
                reverse=True,
            )

            active = candidates[
                :self.args.route_beam_size
            ]

            for st in active:
                if st.is_complete(self.stock_keys):
                    completed.append(st)

            comp_best = {}

            for st in completed:
                key = st.state_key()

                if (
                    key not in comp_best
                    or st.score > comp_best[key].score
                ):
                    comp_best[key] = st

            completed = sorted(
                comp_best.values(),
                key=lambda x: x.score,
                reverse=True,
            )

            if (
                len(completed) >= self.args.route_beam_size
                and self.args.stop_when_enough_complete
            ):
                break

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            gc.collect()

        completed = sorted(
            completed,
            key=lambda x: x.score,
            reverse=True,
        )

        return completed[:self.args.eval_topk]

    def greedy_dfs_one(
        self,
        gt: GroundTruthSample,
    ) -> Dict[str, Any]:
        """RetroInText/FusionRetro-style greedy depth-first route construction."""
        depth_limit = (
            int(gt.depth)
            if (
                self.args.greedy_depth_limit_mode == "gt"
                and gt.depth is not None
                and int(gt.depth) >= 0
            )
            else int(self.args.max_depth)
        )

        # Next unsolved molecule is always pending[0].  New unsolved children are
        # prepended, reproducing the depth-first ordering in RetroInText's code.
        pending: List[Tuple[str, int]] = [(gt.target_smiles, 0)]
        stock_frontier: List[Tuple[str, int]] = []
        expanded: Set[str] = set()
        actions: List[Dict[str, Any]] = []
        failure_reason = ""

        while pending:
            current, current_depth = pending.pop(0)

            # Official RetroInText Greedy_DFS.py stops when depth > GT depth.
            if current_depth > depth_limit:
                failure_reason = "depth_exceeded"
                break

            if is_in_stock(current, self.stock_keys):
                stock_frontier.append((current, current_depth))
                continue

            # ChemDFM remains route-contextual in our model.
            prompt_frontier = sorted(
                stock_frontier + [(current, current_depth)] + pending
            )
            state = SearchState(
                frontier=prompt_frontier,
                expanded=set(expanded),
                score=(
                    compute_route_score(actions, self.args)
                    if actions
                    else 0.0
                ),
                actions=list(actions),
            )

            raw_candidates = self.generate_raw_candidates(
                target=gt.target_smiles,
                current=current,
                current_depth=current_depth,
            )

            # RetroInText's Greedy_DFS uses beam_size=5.  The normal generator
            # still produces single_step_topk candidates, but the greedy run only
            # gives the first greedy_num_candidates to ChemDFM (default 5).
            greedy_pool = raw_candidates[: self.args.greedy_num_candidates]
            ranked_candidates = self.rerank_with_chemdfm(
                target=gt.target_smiles,
                current=current,
                current_depth=current_depth,
                state=state,
                candidates=greedy_pool,
            )

            chosen = None
            for cand in ranked_candidates:
                features = cand.get("features", {}) or {}
                if (
                    self.args.skip_bad_actions
                    and safe_float(
                        features.get("bad_action_penalty", 0.0),
                        0.0,
                    ) > 0.0
                ):
                    continue
                reactants = split_side(cand.get("reactants_str", ""))
                if not reactants:
                    continue
                if current in reactants:
                    continue
                chosen = cand
                break

            if chosen is None:
                failure_reason = "no_eligible_candidate"
                break

            reactants = []
            for r in split_side(chosen["reactants_str"]):
                cr = canonicalize_smiles(r)
                if cr:
                    reactants.append(cr)
            reactants = sorted(set(reactants))

            if not reactants:
                failure_reason = "empty_reactants"
                break

            action = {
                "current": current,
                "current_depth": current_depth,
                "reactants": reactants,
                "ranker_rank": chosen.get("ranker_rank"),
                "ranker_score": chosen.get("ranker_score"),
                "ranker_logprob": chosen.get("ranker_logprob"),
                "chemdfm_candidate_id": chosen.get("chemdfm_candidate_id"),
                "sequence_score": chosen.get("sequence_score"),
                "generator_score_per_token": (
                    chosen.get("features", {}).get(
                        "generator_score_per_token"
                    )
                ),
                "beam_rank": chosen.get("beam_rank"),
            }
            actions.append(action)
            expanded.add(current)

            # Same ordering as the official implementation: reactants are sorted,
            # then each unsolved child is prepended. Thus the last sorted child is
            # expanded first on the next iteration.
            for reactant in reactants:
                child = (reactant, current_depth + 1)
                if is_in_stock(reactant, self.stock_keys):
                    stock_frontier.append(child)
                else:
                    pending.insert(0, child)

        completed = (not pending) and (failure_reason == "")
        predicted_starting_materials = (
            sorted(mol for mol, _ in stock_frontier)
            if completed
            else []
        )

        # Two exact-match variants are stored at no extra inference cost:
        # full InChIKey = same strict metric used by this script's main Top-k;
        # first 14 chars = the comparison used by official RetroInText Greedy_DFS.py.
        pred_full = smiles_list_to_full_key_set(
            predicted_starting_materials
        )
        pred_key14 = smiles_list_to_key14_set(
            predicted_starting_materials
        )

        full_match_indices = (
            match_unique_starting_material_reference(
                predicted_starting_keys=pred_full,
                reference_sets=gt.unique_starting_material_key_sets,
            )
            if completed
            else []
        )

        ref14_sets = [
            {key[:14] for key in ref_set}
            for ref_set in gt.unique_starting_material_key_sets
        ]
        key14_match_indices = (
            [
                idx
                for idx, ref_set in enumerate(ref14_sets)
                if pred_key14 == ref_set
            ]
            if completed
            else []
        )

        return {
            "sample_id": gt.sample_id,
            "product_name": gt.product_name,
            "target_smiles": gt.target_smiles,
            "ground_depth": gt.depth,
            "depth_limit_used": depth_limit,
            "completed": completed,
            "failure_reason": failure_reason,
            "predicted_starting_materials": predicted_starting_materials,
            "exact_match_full_inchikey": bool(full_match_indices),
            "exact_match_connectivity14": bool(key14_match_indices),
            "matched_full_reference_indices": full_match_indices,
            "matched_connectivity14_reference_indices": key14_match_indices,
            "num_actions": len(actions),
            "actions": actions,
        }

    def evaluate_greedy_dfs(
        self,
        samples: List[GroundTruthSample],
    ) -> Dict[str, Any]:
        details = []
        for gt in tqdm(
            samples,
            desc="Greedy DFS (Figure-2 style)",
            dynamic_ncols=True,
        ):
            details.append(self.greedy_dfs_one(gt))

        total = len(details)
        completed_count = sum(
            int(x["completed"]) for x in details
        )
        full_hits = sum(
            int(x["exact_match_full_inchikey"]) for x in details
        )
        key14_hits = sum(
            int(x["exact_match_connectivity14"]) for x in details
        )

        by_depth: Dict[int, Dict[str, Any]] = {}
        for item in details:
            depth = safe_int(item.get("ground_depth", -1), -1)
            bucket = by_depth.setdefault(
                depth,
                {
                    "ground_depth": depth,
                    "num_samples": 0,
                    "completed_count": 0,
                    "full_exact_count": 0,
                    "connectivity14_exact_count": 0,
                },
            )
            bucket["num_samples"] += 1
            bucket["completed_count"] += int(item["completed"])
            bucket["full_exact_count"] += int(
                item["exact_match_full_inchikey"]
            )
            bucket["connectivity14_exact_count"] += int(
                item["exact_match_connectivity14"]
            )

        depth_rows = []
        for depth in sorted(by_depth):
            row = by_depth[depth]
            n = row["num_samples"]
            row["search_success_rate"] = (
                row["completed_count"] / n if n else 0.0
            )
            row["accuracy_full_inchikey"] = (
                row["full_exact_count"] / n if n else 0.0
            )
            row["accuracy_full_inchikey_percent"] = (
                100.0 * row["accuracy_full_inchikey"]
            )
            row["accuracy_connectivity14"] = (
                row["connectivity14_exact_count"] / n if n else 0.0
            )
            row["accuracy_connectivity14_percent"] = (
                100.0 * row["accuracy_connectivity14"]
            )
            depth_rows.append(row)

        metrics = {
            "test_evaluated_products": total,
            "greedy_depth_limit_mode": self.args.greedy_depth_limit_mode,
            "greedy_num_candidates": self.args.greedy_num_candidates,
            "search_success_count": completed_count,
            "search_success_rate": (
                completed_count / total if total else 0.0
            ),
            "exact_full_inchikey_count": full_hits,
            "exact_full_inchikey_rate": (
                full_hits / total if total else 0.0
            ),
            "exact_connectivity14_count": key14_hits,
            "exact_connectivity14_rate": (
                key14_hits / total if total else 0.0
            ),
            "figure2_primary_metric": (
                "Greedy-DFS complete-route starting-material exact match by "
                "ground-truth depth. accuracy_connectivity14 reproduces the "
                "first-14-InChIKey comparison in RetroInText Greedy_DFS.py."
            ),
            "by_depth": depth_rows,
        }
        return {
            "metrics": metrics,
            "details": details,
        }

    def evaluate_samples(
        self,
        samples: List[GroundTruthSample],
    ) -> Dict[str, Any]:
        total = 0
        success = 0

        exact_hits = {
            k: 0
            for k in range(
                1,
                self.args.eval_topk + 1,
            )
        }

        exact14_hits = {
            k: 0
            for k in range(
                1,
                self.args.eval_topk + 1,
            )
        }

        exact_inter_hits = {
            k: 0
            for k in range(
                1,
                self.args.eval_topk + 1,
            )
        }

        details = []

        for gt in tqdm(
            samples,
            desc="multi-step ChemDFM eval",
            dynamic_ncols=True,
        ):
            total += 1

            completed_routes = self.search_one(gt)

            if completed_routes:
                success += 1

            pred_infos = []

            for st in completed_routes:
                pred_start_mats = st.starting_materials()
                pred_intermediates = st.intermediates(
                    gt.target_smiles
                )

                pred_start_keys = (
                    smiles_list_to_full_key_set(
                        pred_start_mats
                    )
                )

                pred_start_key14 = (
                    smiles_list_to_key14_set(
                        pred_start_mats
                    )
                )

                pred_inter_keys = (
                    smiles_list_to_full_key_set(
                        pred_intermediates
                    )
                )

                unique_ref_indices = (
                    match_unique_starting_material_reference(
                        predicted_starting_keys=pred_start_keys,
                        reference_sets=gt.unique_starting_material_key_sets,
                    )
                )

                ref14_sets = [
                    {key[:14] for key in ref_set}
                    for ref_set in gt.unique_starting_material_key_sets
                ]
                unique_ref14_indices = [
                    idx
                    for idx, ref_set in enumerate(ref14_sets)
                    if pred_start_key14 == ref_set
                ]

                (
                    route_start_ids,
                    paired_route_ids,
                ) = match_route_references(
                    predicted_starting_keys=pred_start_keys,
                    predicted_intermediate_keys=pred_inter_keys,
                    references=gt.route_references,
                )

                pred_infos.append({
                    "score": st.score,
                    "starting_materials": pred_start_mats,
                    "intermediates": pred_intermediates,
                    "exact_starting_material_match": bool(
                        unique_ref_indices
                    ),
                    "exact_starting_material_match_connectivity14": bool(
                        unique_ref14_indices
                    ),
                    "exact_with_intermediate_match": bool(
                        paired_route_ids
                    ),
                    "matched_unique_material_reference_indices": (
                        unique_ref_indices
                    ),
                    "matched_unique_material_reference_indices_connectivity14": (
                        unique_ref14_indices
                    ),
                    "matched_route_reference_ids_by_materials": (
                        route_start_ids
                    ),
                    "matched_paired_route_reference_ids": (
                        paired_route_ids
                    ),
                    "num_actions": len(st.actions),
                    "actions": st.actions,
                })

            for k in range(
                1,
                self.args.eval_topk + 1,
            ):
                topk_infos = pred_infos[:k]

                if any(
                    x["exact_starting_material_match"]
                    for x in topk_infos
                ):
                    exact_hits[k] += 1

                if any(
                    x["exact_starting_material_match_connectivity14"]
                    for x in topk_infos
                ):
                    exact14_hits[k] += 1

                if any(
                    x["exact_with_intermediate_match"]
                    for x in topk_infos
                ):
                    exact_inter_hits[k] += 1

            details.append({
                "sample_id": gt.sample_id,
                "product_name": gt.product_name,
                "target_smiles": gt.target_smiles,
                "ground_depth": gt.depth,
                "search_depth_limit_used": self.get_search_depth_limit(gt),
                "declared_num_routes": gt.declared_num_routes,
                "num_route_references": len(
                    gt.route_references
                ),
                "num_unique_starting_material_sets": len(
                    gt.unique_starting_material_key_sets
                ),
                "success": bool(completed_routes),
                "num_completed_routes": len(
                    completed_routes
                ),
                "predicted_routes": pred_infos,
            })

        total_route_refs = sum(
            len(x.route_references)
            for x in samples
        )

        total_unique_material_refs = sum(
            len(x.unique_starting_material_key_sets)
            for x in samples
        )

        metrics = {
            "test_evaluated_products": total,
            "ground_truth_route_reference_count": total_route_refs,
            "ground_truth_unique_material_set_count": total_unique_material_refs,
            "mean_routes_per_product": (
                total_route_refs / total
                if total
                else 0.0
            ),
            "mean_unique_material_sets_per_product": (
                total_unique_material_refs / total
                if total
                else 0.0
            ),
            "success_count": success,
            "success_rate": (
                success / total
                if total
                else 0.0
            ),
            "primary_metric": (
                "top-k complete-route starting-material exact match against any "
                "unique routes[*].materials set for the same product, using full InChIKey"
            ),
            "retrointext_compatible_metric": (
                "same top-k complete-route starting-material exact match, using "
                "the first 14 InChIKey characters as in RetroInText/FusionRetro"
            ),
            "search_depth_limit_mode": self.args.search_depth_limit_mode,
        }

        for k in range(
            1,
            self.args.eval_topk + 1,
        ):
            metrics[
                f"top{k}_starting_material_exact_count"
            ] = exact_hits[k]

            metrics[
                f"top{k}_starting_material_exact_rate"
            ] = (
                exact_hits[k] / total
                if total
                else 0.0
            )

            metrics[
                f"top{k}_starting_material_exact_connectivity14_count"
            ] = exact14_hits[k]

            metrics[
                f"top{k}_starting_material_exact_connectivity14_rate"
            ] = (
                exact14_hits[k] / total
                if total
                else 0.0
            )

            metrics[
                f"top{k}_starting_material_intermediate_exact_count"
            ] = exact_inter_hits[k]

            metrics[
                f"top{k}_starting_material_intermediate_exact_rate"
            ] = (
                exact_inter_hits[k] / total
                if total
                else 0.0
            )

        metrics.update(self.stats)

        return {
            "metrics": metrics,
            "details": details,
        }


def aggregate_topk_by_depth(
    details: List[Dict[str, Any]],
    eval_topk: int,
) -> List[Dict[str, Any]]:
    """Create Table-4-style GT-depth x Top-k exact-match statistics."""
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for item in details:
        depth = safe_int(item.get("ground_depth", -1), -1)
        buckets.setdefault(depth, []).append(item)

    rows = []
    for depth in sorted(buckets):
        items = buckets[depth]
        n = len(items)
        row: Dict[str, Any] = {
            "ground_depth": depth,
            "num_samples": n,
            "search_success_count": sum(
                int(x.get("success", False)) for x in items
            ),
        }
        row["search_success_rate"] = (
            row["search_success_count"] / n if n else 0.0
        )

        for k in range(1, eval_topk + 1):
            exact_count = 0
            exact14_count = 0
            exact_inter_count = 0
            for item in items:
                preds = item.get("predicted_routes", [])[:k]
                exact_count += int(any(
                    p.get("exact_starting_material_match", False)
                    for p in preds
                ))
                exact14_count += int(any(
                    p.get("exact_starting_material_match_connectivity14", False)
                    for p in preds
                ))
                exact_inter_count += int(any(
                    p.get("exact_with_intermediate_match", False)
                    for p in preds
                ))

            row[f"top{k}_exact_count"] = exact_count
            row[f"top{k}_exact_rate"] = (
                exact_count / n if n else 0.0
            )
            row[f"top{k}_exact_percent"] = (
                100.0 * row[f"top{k}_exact_rate"]
            )
            row[f"top{k}_retrointext14_exact_count"] = exact14_count
            row[f"top{k}_retrointext14_exact_rate"] = (
                exact14_count / n if n else 0.0
            )
            row[f"top{k}_retrointext14_exact_percent"] = (
                100.0 * row[f"top{k}_retrointext14_exact_rate"]
            )
            row[f"top{k}_start_intermediate_exact_count"] = (
                exact_inter_count
            )
            row[f"top{k}_start_intermediate_exact_rate"] = (
                exact_inter_count / n if n else 0.0
            )

        rows.append(row)

    return rows


def make_table4_csv_rows(
    depth_rows: List[Dict[str, Any]],
    eval_topk: int,
) -> List[Dict[str, Any]]:
    rows = []
    for x in depth_rows:
        row = {
            "ground_depth": x["ground_depth"],
            "num_samples": x["num_samples"],
        }
        for k in range(1, eval_topk + 1):
            row[f"top{k}_full_exact_percent"] = round(
                float(x[f"top{k}_exact_percent"]),
                6,
            )
            row[f"top{k}_retrointext14_exact_percent"] = round(
                float(x[f"top{k}_retrointext14_exact_percent"]),
                6,
            )
        rows.append(row)
    return rows


def make_overall_topk_csv_rows(
    metrics: Dict[str, Any],
    eval_topk: int,
) -> List[Dict[str, Any]]:
    row: Dict[str, Any] = {
        "num_samples": metrics.get("test_evaluated_products", 0),
        "search_depth_limit_mode": metrics.get("search_depth_limit_mode", ""),
    }
    for k in range(1, eval_topk + 1):
        row[f"top{k}_full_exact_percent"] = 100.0 * safe_float(
            metrics.get(f"top{k}_starting_material_exact_rate", 0.0),
            0.0,
        )
        row[f"top{k}_retrointext14_exact_percent"] = 100.0 * safe_float(
            metrics.get(
                f"top{k}_starting_material_exact_connectivity14_rate",
                0.0,
            ),
            0.0,
        )
    return [row]


# ---------------------------------------------------------------------------
# Deterministic subset selection
# ---------------------------------------------------------------------------

def select_samples(
    samples: List[GroundTruthSample],
    sample_size: int,
    sample_seed: int,
    sample_ids_json: str,
) -> List[GroundTruthSample]:
    if sample_ids_json:
        ids_obj = load_json(sample_ids_json)

        if isinstance(ids_obj, dict):
            ids = ids_obj.get(
                "sample_ids",
                ids_obj.get(
                    "selected_sample_ids",
                    [],
                ),
            )
        else:
            ids = ids_obj

        ids = [str(x) for x in ids]
        id_set = set(ids)

        by_id = {
            str(x.sample_id): x
            for x in samples
        }

        selected = [
            by_id[x]
            for x in ids
            if x in by_id
        ]

        missing = [
            x
            for x in ids
            if x not in by_id
        ]

        if missing:
            raise ValueError(
                f"{len(missing)} sample IDs from {sample_ids_json} "
                f"were not found. First missing IDs: {missing[:10]}"
            )

        return selected

    if sample_size <= 0 or sample_size >= len(samples):
        return list(samples)

    rng = random.Random(sample_seed)
    selected_indices = sorted(
        rng.sample(
            range(len(samples)),
            sample_size,
        )
    )

    return [
        samples[i]
        for i in selected_indices
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_multistep_json",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--generator_model_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--chemdfm_base_model",
        type=str,
        default="../ChemDFM-local-model",
    )

    parser.add_argument(
        "--chemdfm_adapter",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--stock_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    # Candidate/search configuration
    parser.add_argument(
        "--single_step_topk",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--single_step_num_beams",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ranker_num_candidates",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ranker_keep_topm",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--route_beam_size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--eval_topk",
        type=int,
        default=5,
    )

    # Route score
    parser.add_argument(
        "--ranker_score_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--molt5_score_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--length_norm_gamma",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--depth_penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--molt5_score_field",
        type=str,
        default="generator_score_per_token",
        choices=[
            "generator_score_per_token",
            "sequence_score",
        ],
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--max_search_steps",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--search_depth_limit_mode",
        type=str,
        default="global",
        choices=["global", "gt"],
        help=(
            "Normal Top-k search depth cap: 'global' keeps the old behavior "
            "(--max_depth); 'gt' uses each product's ground-truth depth, as in "
            "the RetroInText/FusionRetro evaluation code."
        ),
    )
    parser.add_argument(
        "--max_reactants",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max_history_steps",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--chemdfm_max_length",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--expand_all_unsolved",
        action="store_true",
    )
    parser.add_argument(
        "--stop_when_enough_complete",
        action="store_true",
    )
    parser.add_argument(
        "--skip_bad_actions",
        action="store_true",
    )

    # Figure-2-style Greedy DFS
    parser.add_argument(
        "--run_greedy_dfs",
        action="store_true",
        help=(
            "Also run RetroInText/FusionRetro-style Greedy DFS and report "
            "accuracy by ground-truth route depth."
        ),
    )
    parser.add_argument(
        "--greedy_depth_limit_mode",
        type=str,
        default="gt",
        choices=["global", "gt"],
        help=(
            "Greedy DFS depth cap. Use 'gt' for the paper-style protocol."
        ),
    )
    parser.add_argument(
        "--greedy_num_candidates",
        type=int,
        default=5,
        help=(
            "Number of leading MolT5 proposals reranked at each greedy step. "
            "RetroInText Greedy_DFS uses beam_size=5."
        ),
    )

    # Subset selection
    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1,
        help="Randomly evaluate this many products; -1 means all.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--sample_ids_json",
        type=str,
        default="",
        help=(
            "Optional selected_sample_ids.json from another run. "
            "When set, this exact subset/order is reused."
        ),
    )

    # Precision/model loading
    parser.add_argument(
        "--load_chemdfm_in_4bit",
        action="store_true",
    )
    parser.add_argument(
        "--chemdfm_bf16",
        action="store_true",
    )
    parser.add_argument(
        "--chemdfm_fp16",
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 for MolT5.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.chemdfm_bf16 and args.chemdfm_fp16:
        raise ValueError(
            "Choose only one of --chemdfm_bf16 / --chemdfm_fp16"
        )

    if args.single_step_num_beams < args.single_step_topk:
        raise ValueError(
            "--single_step_num_beams must be >= --single_step_topk"
        )

    if args.ranker_num_candidates > args.single_step_topk:
        raise ValueError(
            "--ranker_num_candidates cannot exceed --single_step_topk"
        )

    if args.ranker_num_candidates > len(LABELS):
        raise ValueError(
            f"--ranker_num_candidates must be <= {len(LABELS)}"
        )

    if args.greedy_num_candidates <= 0:
        raise ValueError(
            "--greedy_num_candidates must be >= 1"
        )

    if args.greedy_num_candidates > args.ranker_num_candidates:
        raise ValueError(
            "--greedy_num_candidates cannot exceed --ranker_num_candidates"
        )

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        vars(args),
        str(out_dir / "config.json"),
    )

    raw_items = load_json(
        args.test_multistep_json
    )

    all_gt_samples = build_ground_truth_samples(
        raw_items
    )

    selected_samples = select_samples(
        samples=all_gt_samples,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        sample_ids_json=args.sample_ids_json,
    )

    selected_ids = [
        str(x.sample_id)
        for x in selected_samples
    ]

    write_json(
        {
            "sample_size": len(selected_ids),
            "sample_seed": args.sample_seed,
            "source_test_json": args.test_multistep_json,
            "sample_ids": selected_ids,
        },
        str(
            out_dir
            / "selected_sample_ids.json"
        ),
    )

    print("===== Multi-step ChemDFM Evaluation Setup =====")
    print(f"test_multistep_json : {args.test_multistep_json}")
    print(f"all valid products  : {len(all_gt_samples)}")
    print(f"selected products   : {len(selected_samples)}")
    print(f"sample_seed         : {args.sample_seed}")
    print(f"generator           : {args.generator_model_dir}")
    print(f"ChemDFM base        : {args.chemdfm_base_model}")
    print(f"ChemDFM adapter     : {args.chemdfm_adapter}")
    print(f"single_step_topk    : {args.single_step_topk}")
    print(f"keep_topm           : {args.ranker_keep_topm}")
    print(f"route_beam_size     : {args.route_beam_size}")
    print(f"eval_topk           : {args.eval_topk}")
    print(f"search depth mode   : {args.search_depth_limit_mode}")
    print(f"run Greedy DFS      : {args.run_greedy_dfs}")
    if args.run_greedy_dfs:
        print(f"greedy depth mode   : {args.greedy_depth_limit_mode}")
        print(f"greedy candidates   : {args.greedy_num_candidates}")

    evaluator = MultiStepChemDFMEvaluator(
        args
    )

    result = evaluator.evaluate_samples(
        selected_samples
    )

    result["metrics"][
        "test_total_products_full"
    ] = len(all_gt_samples)

    result["metrics"][
        "test_selected_products"
    ] = len(selected_samples)

    result["metrics"][
        "sample_seed"
    ] = args.sample_seed

    write_json(
        result["metrics"],
        str(out_dir / "metrics.json"),
    )
    write_csv(
        make_overall_topk_csv_rows(result["metrics"], args.eval_topk),
        str(out_dir / "overall_topk.csv"),
    )

    write_json(
        result["details"],
        str(out_dir / "details.json"),
    )

    # No extra inference: regroup the normal Top-k route results by GT depth.
    depth_rows = aggregate_topk_by_depth(
        result["details"],
        args.eval_topk,
    )
    write_json(
        {
            "search_depth_limit_mode": args.search_depth_limit_mode,
            "eval_topk": args.eval_topk,
            "by_depth": depth_rows,
        },
        str(out_dir / "metrics_by_depth.json"),
    )
    write_csv(
        make_table4_csv_rows(depth_rows, args.eval_topk),
        str(out_dir / "table4_depth_topk.csv"),
    )

    # Greedy DFS is a separate search trajectory.  It reuses the safe MolT5
    # proposal cache when possible, but reranks with the greedy route context.
    greedy_result = None
    if args.run_greedy_dfs:
        greedy_result = evaluator.evaluate_greedy_dfs(
            selected_samples
        )
        write_json(
            greedy_result["metrics"],
            str(out_dir / "greedy_dfs_metrics.json"),
        )
        write_json(
            greedy_result["details"],
            str(out_dir / "greedy_dfs_details.json"),
        )
        write_csv(
            greedy_result["metrics"].get("by_depth", []),
            str(out_dir / "figure2_greedy_depth_accuracy.csv"),
        )

    print("\n===== Metrics =====")
    print(
        json.dumps(
            result["metrics"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\n[完成] metrics: {out_dir / 'metrics.json'}"
    )
    print(
        f"[完成] overall Top-k CSV: {out_dir / 'overall_topk.csv'}"
    )
    print(
        f"[完成] details: {out_dir / 'details.json'}"
    )
    print(
        f"[完成] depth metrics: {out_dir / 'metrics_by_depth.json'}"
    )
    print(
        f"[完成] Table-4 CSV: {out_dir / 'table4_depth_topk.csv'}"
    )
    if greedy_result is not None:
        print("\n===== Greedy DFS / Figure-2-style =====")
        print(
            json.dumps(
                greedy_result["metrics"],
                ensure_ascii=False,
                indent=2,
            )
        )
        print(
            f"[完成] greedy metrics: {out_dir / 'greedy_dfs_metrics.json'}"
        )
        print(
            f"[完成] greedy details: {out_dir / 'greedy_dfs_details.json'}"
        )
        print(
            f"[完成] Figure-2 CSV: {out_dir / 'figure2_greedy_depth_accuracy.csv'}"
        )
    print(
        f"[完成] selected IDs: {out_dir / 'selected_sample_ids.json'}"
    )


if __name__ == "__main__":
    main()
