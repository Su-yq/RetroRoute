#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build route-contextual ChemDFM reranking data from multi-step retrosynthesis routes.

Important properties:
  1. Standalone: does NOT import any project-local Python module.
  2. Supports preprocessed grouped JSON (recommended) and the original raw JSON format.
  3. Reconstructs teacher-forced frontier/history from each linear reference route.
  4. Aggregates identical search states across reference routes, so multiple valid
     reference reactions become multi-positive labels instead of false negatives.
  5. Uses the same MolT5 route-context prompt format as the existing project.
  6. Uses the same InChIKey-14 stock matching convention as the existing search code.
  7. Keeps rows where no reference reaction appears in Top-K for diagnostics, with
     `trainable_v2=false`.
  8. Skips teacher-forced states whose current molecule is already in stock, because
     the current multi-step search would not expand such a molecule.

Example output row (abbreviated):
{
  "schema_version": "chemdfm_route_rerank_v2_v1",
  "sample_id": "train_...",
  "split": "train",
  "target_smiles": "...",
  "state": {
    "current_smiles": "...",
    "current_depth": 1,
    "max_depth": 14,
    "remaining_depth_budget": 13,
    "frontier": [...],
    "history": [...]
  },
  "candidates": [
    {
      "candidate_id": "A",
      "reactants_str": "...",
      "molt5_rank": 1,
      "sequence_score": -1.23,
      "generator_score_per_token": -0.05,
      "valid": true,
      "num_reactants": 2,
      "stock_count": 1,
      "stock_ratio": 0.5,
      "reactants": [{"smiles": "...", "in_stock": true}]
    }
  ],
  "supervision": {
    "source": "reference_route",
    "gold_reactants_options": ["..."],
    "positive_candidate_ids": ["B"],
    "matched_gold_reactants": {"B": ["..."]},
    "num_positive_candidates": 1,
    "trainable_v2": true
  },
  "rollout_supervision": null,
  "chemdfm_prompt": "..."
}
"""

import argparse
import gc
import hashlib
import json
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    Chem = None
    HAS_RDKIT = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ATOM_MAP_RE = re.compile(r":\d+(?=\])")
MOLT5_TASK_PREFIX = "Please predict the reactant of the product:\n"
SCHEMA_VERSION = "chemdfm_route_rerank_v2_v1"
PROMPT_VERSION = "route_context_listwise_en_v1"


# -----------------------------------------------------------------------------
# Reproducibility / IO
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def inspect_and_repair_jsonl(path: Path) -> Dict[str, int]:
    """Inspect an existing output JSONL and truncate a broken final line if needed."""
    stats = {
        "next_state_index": 0,
        "num_rows": 0,
        "num_trainable": 0,
        "num_multi_positive": 0,
    }

    if not path.exists() or path.stat().st_size == 0:
        return stats

    last_good_offset = 0
    max_index = -1

    with open(path, "rb") as f:
        while True:
            line_start = f.tell()
            line = f.readline()
            if not line:
                break

            if not line.endswith(b"\n"):
                break

            try:
                row = json.loads(line.decode("utf-8"))
            except Exception:
                break

            last_good_offset = f.tell()
            stats["num_rows"] += 1
            idx = int(row.get("_state_index", stats["num_rows"] - 1))
            max_index = max(max_index, idx)

            sup = row.get("supervision", {}) or {}
            if bool(sup.get("trainable_v2", False)):
                stats["num_trainable"] += 1
            if int(sup.get("num_positive_candidates", 0) or 0) > 1:
                stats["num_multi_positive"] += 1

    file_size = path.stat().st_size
    if last_good_offset < file_size:
        print(f"[WARN] repairing incomplete JSONL tail: {path}")
        with open(path, "rb+") as f:
            f.truncate(last_good_offset)

    stats["next_state_index"] = max_index + 1 if max_index >= 0 else 0
    return stats


# -----------------------------------------------------------------------------
# Chemistry normalization
# -----------------------------------------------------------------------------

def strip_atom_mapping(smiles: str) -> str:
    if smiles is None:
        return ""
    return ATOM_MAP_RE.sub("", str(smiles).strip())


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if smiles is None:
        return None

    smi = strip_atom_mapping(str(smiles).strip().replace(" ", ""))
    if not smi:
        return None

    if not HAS_RDKIT:
        return smi

    # First try normal sanitized parsing.
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        mol = None

    # Fallback mirrors the project's preprocessing behavior.
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is not None:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
        except Exception:
            mol = None

    if mol is None:
        return None

    try:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def split_side(side: str) -> List[str]:
    if side is None:
        return []
    text = str(side).strip().replace(" ", "").rstrip(".")
    if not text:
        return []
    return [x for x in text.split(".") if x.strip()]


def canonicalize_reactant_list(values: Sequence[str]) -> Tuple[List[str], Optional[str]]:
    clean: List[str] = []
    for value in values:
        smi = canonicalize_smiles(value)
        if smi is None:
            return [], None
        clean.append(smi)

    if not clean:
        return [], None

    # Keep multiplicity; only sort for order-invariant exact matching.
    clean = sorted(clean)
    return clean, ".".join(clean)


def canonicalize_side(side: str) -> Optional[str]:
    parts = split_side(side)
    if not parts:
        return None
    _, text = canonicalize_reactant_list(parts)
    return text


def normalize_prediction_text(text: str) -> Optional[str]:
    if text is None:
        return None

    value = str(text).strip()
    if ">>" in value:
        value = value.split(">>", 1)[1]
    value = re.sub(r"\s+", "", value)
    return canonicalize_side(value)


def valid_side(side: str) -> bool:
    parts = split_side(side)
    if not parts:
        return False
    if not HAS_RDKIT:
        return True

    for part in parts:
        try:
            if Chem.MolFromSmiles(part) is None:
                return False
        except Exception:
            return False
    return True


def parse_retro_reaction(reaction_smiles: str) -> Optional[Dict[str, Any]]:
    """Parse raw atom-mapped `product>>reactants` into canonical unmapped SMILES."""
    if not reaction_smiles or ">>" not in reaction_smiles:
        return None

    left, right = str(reaction_smiles).split(">>", 1)
    product = canonicalize_smiles(left)
    reactants, reactants_str = canonicalize_reactant_list(split_side(right))

    if not product or not reactants_str:
        return None

    return {
        "product_smiles": product,
        "reactants_smiles": reactants,
        "reactants_str": reactants_str,
    }


# -----------------------------------------------------------------------------
# Stock database: same InChIKey-14 convention as existing project
# -----------------------------------------------------------------------------

def mol_to_inchikey14(smiles: str) -> Optional[str]:
    if not HAS_RDKIT:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)
        return key[:14] if key else None
    except Exception:
        return None


def load_stock_keys(stock_path: str) -> Set[str]:
    if not stock_path:
        return set()

    import pandas as pd

    df = pd.read_hdf(stock_path, key="table")
    if "inchi_key" not in df.columns:
        raise KeyError(f"stock HDF5 has no 'inchi_key' column: {stock_path}")
    return set(str(x)[:14] for x in df.inchi_key.values)


def is_in_stock(smiles: str, stock_keys: Set[str]) -> bool:
    if not stock_keys:
        return False
    key14 = mol_to_inchikey14(smiles)
    return key14 is not None and key14 in stock_keys


# -----------------------------------------------------------------------------
# Input route adapters
# -----------------------------------------------------------------------------

def normalize_grouped_step(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    product = canonicalize_smiles(step.get("product_smiles", ""))
    reactants_raw = step.get("reactants_smiles", [])

    if isinstance(reactants_raw, str):
        reactants_raw = split_side(reactants_raw)

    if not isinstance(reactants_raw, (list, tuple)):
        return None

    reactants, reactants_str = canonicalize_reactant_list(list(reactants_raw))
    if not product or not reactants_str:
        return None

    return {
        "step_order": int(step.get("step_order", 0) or 0),
        "product_smiles": product,
        "reactants_smiles": reactants,
        "reactants_str": reactants_str,
    }


def iter_grouped_routes(data: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Yield one linear route object from preprocess_multistep_retro grouped JSON."""
    for product_index, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue

        target = canonicalize_smiles(obj.get("product", ""))
        if not target:
            continue

        declared_depth = obj.get("depth", None)
        routes = obj.get("routes", [])
        if not isinstance(routes, list):
            continue

        for route_index, route in enumerate(routes):
            if not isinstance(route, dict):
                continue

            steps: List[Dict[str, Any]] = []
            for raw_step in route.get("steps", []) or []:
                if not isinstance(raw_step, dict):
                    continue
                step = normalize_grouped_step(raw_step)
                if step is not None:
                    steps.append(step)

            if not steps:
                continue

            # Preserve the preprocessed order. Under --inner_path_as_route this is a
            # target-to-material linear path.
            steps = sorted(
                enumerate(steps),
                key=lambda x: (x[1].get("step_order", 0), x[0]),
            )
            steps = [x[1] for x in steps]

            materials = []
            for x in route.get("materials", []) or []:
                smi = canonicalize_smiles(x)
                if smi:
                    materials.append(smi)

            yield {
                "target_smiles": target,
                "product_index": product_index,
                "route_id": str(route.get("route_id", route_index)),
                "declared_depth": declared_depth,
                "steps": steps,
                "materials": sorted(materials),
                "source_format": "grouped",
            }


def iter_raw_routes(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield each original raw `retro_routes` inner list as one linear route."""
    for product_index, (raw_target, obj) in enumerate(data.items()):
        if not isinstance(obj, dict):
            continue

        target = canonicalize_smiles(raw_target)
        if not target:
            continue

        declared_depth = obj.get("depth", None)
        num_trees = int(obj.get("num_reaction_trees", 0) or 0)

        for tree_idx in range(1, num_trees + 1):
            tree = obj.get(str(tree_idx), {})
            if not isinstance(tree, dict):
                continue

            materials = []
            for x in tree.get("materials", []) or []:
                smi = canonicalize_smiles(x)
                if smi:
                    materials.append(smi)

            retro_routes = tree.get("retro_routes", []) or []
            for path_idx, path in enumerate(retro_routes):
                if isinstance(path, str):
                    path = [path]
                if not isinstance(path, list):
                    continue

                steps: List[Dict[str, Any]] = []
                valid = True
                for step_idx, reaction_smiles in enumerate(path, start=1):
                    parsed = parse_retro_reaction(reaction_smiles)
                    if parsed is None:
                        valid = False
                        break
                    parsed["step_order"] = step_idx
                    steps.append(parsed)

                if not valid or not steps:
                    continue

                yield {
                    "target_smiles": target,
                    "product_index": product_index,
                    "route_id": f"tree{tree_idx}_path{path_idx}",
                    "declared_depth": declared_depth,
                    "steps": steps,
                    "materials": sorted(materials),
                    "source_format": "raw",
                }


def detect_input_format(data: Any, requested: str) -> str:
    if requested != "auto":
        return requested

    if isinstance(data, list):
        return "grouped"
    if isinstance(data, dict):
        return "raw"
    raise TypeError("Cannot infer input format. Expected grouped list or raw dict.")


# -----------------------------------------------------------------------------
# Teacher-forced route-state reconstruction
# -----------------------------------------------------------------------------

def make_frontier_entry(smiles: str, depth: int, stock_keys: Set[str]) -> Dict[str, Any]:
    return {
        "smiles": smiles,
        "depth": int(depth),
        "in_stock": bool(is_in_stock(smiles, stock_keys)),
    }


def frontier_sort_key(entry: Dict[str, Any]) -> Tuple[int, str, int]:
    return (
        int(entry.get("depth", 0)),
        str(entry.get("smiles", "")),
        1 if bool(entry.get("in_stock", False)) else 0,
    )


def remove_one_frontier_molecule(
    frontier: List[Dict[str, Any]],
    smiles: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], str]:
    """
    Remove one frontier molecule matching a reference-step product.

    Matching order:
      1. exact canonical/isomeric SMILES;
      2. InChIKey first block (14 chars), which is connectivity-based and makes the
         reconstruction robust to tautomer/proton-placement representation differences
         already present between some dataset target strings and atom-mapped reactions.

    Returns (removed_entry, new_frontier, match_type).
    """
    for idx, entry in enumerate(frontier):
        if entry.get("smiles") == smiles:
            removed = dict(entry)
            new_frontier = list(frontier)
            new_frontier.pop(idx)
            return removed, new_frontier, "exact_smiles"

    query_key = mol_to_inchikey14(smiles)
    if query_key is not None:
        for idx, entry in enumerate(frontier):
            entry_smiles = str(entry.get("smiles", ""))
            entry_key = mol_to_inchikey14(entry_smiles)
            if entry_key is not None and entry_key == query_key:
                removed = dict(entry)
                new_frontier = list(frontier)
                new_frontier.pop(idx)
                return removed, new_frontier, "inchikey14"

    return None, list(frontier), "none"


def canonical_state_key(
    target: str,
    current: str,
    current_depth: int,
    frontier: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
) -> str:
    payload = {
        "target": target,
        "current": current,
        "current_depth": int(current_depth),
        "frontier": [
            {
                "smiles": x["smiles"],
                "depth": int(x["depth"]),
                "in_stock": bool(x["in_stock"]),
            }
            for x in sorted(frontier, key=frontier_sort_key)
        ],
        "history": [
            {
                "product": h["product_smiles"],
                "reactants": list(h["reactants_smiles"]),
                "depth": int(h["depth"]),
            }
            for h in history
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def extract_and_aggregate_states(
    routes: Iterable[Dict[str, Any]],
    stock_keys: Set[str],
    max_depth: int,
    split_name: str,
    max_products: int = -1,
    max_history_steps: int = 6,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Reconstruct reference states and merge identical states across routes.

    We intentionally do NOT expose future route materials/intermediates in the model
    input. They are reference annotations, not inference-time observations.
    """
    aggregated: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    stats = {
        "routes_seen": 0,
        "routes_used": 0,
        "states_raw": 0,
        "states_aggregated": 0,
        "route_inconsistencies": 0,
        "connectivity_equivalent_matches": 0,
        "skipped_current_in_stock": 0,
        "skipped_depth_limit": 0,
    }

    seen_product_indices: Set[int] = set()

    for route in routes:
        product_index = int(route.get("product_index", -1))
        if max_products > 0 and product_index >= max_products:
            continue

        seen_product_indices.add(product_index)
        stats["routes_seen"] += 1

        target = route["target_smiles"]
        route_id = str(route["route_id"])
        steps = route["steps"]

        frontier = [make_frontier_entry(target, 0, stock_keys)]
        history: List[Dict[str, Any]] = []
        route_used = False

        for step_idx, step in enumerate(steps):
            reference_product = step["product_smiles"]
            gold_reactants = step["reactants_smiles"]
            gold_reactants_str = step["reactants_str"]

            current_entry, frontier_without_current, match_type = remove_one_frontier_molecule(
                frontier,
                reference_product,
            )

            if current_entry is None:
                # In the current --inner_path_as_route preprocessing this should not
                # happen. Do not fabricate a state; abandon the remainder of this route.
                stats["route_inconsistencies"] += 1
                break

            if match_type == "inchikey14":
                stats["connectivity_equivalent_matches"] += 1

            # Use the actual frontier representation as CURRENT so training mirrors
            # inference. The reference reaction product may be a tautomeric/alternate
            # representation of the same connectivity.
            current = str(current_entry["smiles"])
            current_depth = int(current_entry["depth"])

            # Match inference behavior: stock molecules are not expanded.
            if bool(current_entry["in_stock"]):
                stats["skipped_current_in_stock"] += 1
                break

            if current_depth >= max_depth:
                stats["skipped_depth_limit"] += 1
                break

            state_frontier = sorted(frontier, key=frontier_sort_key)
            history_for_prompt = history[-max_history_steps:] if max_history_steps > 0 else list(history)

            state_key = canonical_state_key(
                target=target,
                current=current,
                current_depth=current_depth,
                frontier=state_frontier,
                history=history_for_prompt,
            )

            if state_key not in aggregated:
                digest = hashlib.sha1(state_key.encode("utf-8")).hexdigest()[:16]
                aggregated[state_key] = {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": f"{split_name}_{digest}",
                    "split": split_name,
                    "target_smiles": target,
                    "reference_route_ids": [],
                    "state": {
                        "current_smiles": current,
                        "current_depth": current_depth,
                        "max_depth": int(max_depth),
                        "remaining_depth_budget": max(0, int(max_depth) - current_depth),
                        "frontier": state_frontier,
                        "history": [dict(x) for x in history_for_prompt],
                    },
                    "gold_reactants_options": [],
                    "source_formats": [],
                }

            item = aggregated[state_key]
            if route_id not in item["reference_route_ids"]:
                item["reference_route_ids"].append(route_id)
            if gold_reactants_str not in item["gold_reactants_options"]:
                item["gold_reactants_options"].append(gold_reactants_str)
            source_format = str(route.get("source_format", "unknown"))
            if source_format not in item["source_formats"]:
                item["source_formats"].append(source_format)

            stats["states_raw"] += 1
            route_used = True

            # Teacher-force the reference action to obtain the next state.
            next_depth = current_depth + 1
            new_frontier = list(frontier_without_current)
            for reactant in gold_reactants:
                new_frontier.append(make_frontier_entry(reactant, next_depth, stock_keys))
            frontier = sorted(new_frontier, key=frontier_sort_key)

            history.append({
                "step": len(history) + 1,
                "product_smiles": current,
                "reactants_smiles": list(gold_reactants),
                "reactants_str": gold_reactants_str,
                "depth": current_depth,
            })

        if route_used:
            stats["routes_used"] += 1

    states = list(aggregated.values())
    for state in states:
        state["reference_route_ids"] = sorted(state["reference_route_ids"])
        state["gold_reactants_options"] = sorted(state["gold_reactants_options"])
        state["source_formats"] = sorted(state["source_formats"])

    stats["states_aggregated"] = len(states)
    stats["products_seen"] = len(seen_product_indices)
    return states, stats


# -----------------------------------------------------------------------------
# MolT5 generation
# -----------------------------------------------------------------------------

def build_molt5_prompt(
    target_smiles: str,
    current_smiles: str,
    current_depth: int,
    max_depth: int,
) -> str:
    """Exact route-context prompt convention used by the current MolT5 code."""
    return (
        f"{MOLT5_TASK_PREFIX}"
        f"<target> {target_smiles}\n"
        f"<current> {current_smiles}\n"
        f"<current_depth> {int(current_depth)}\n"
        f"<max_depth> {int(max_depth)}\n"
        f"<goal> purchasable_starting_materials"
    )


def candidate_id(index: int) -> str:
    if index < 0:
        raise ValueError("candidate index must be non-negative")
    if index < 26:
        return chr(ord("A") + index)
    return f"C{index + 1:02d}"


def build_candidate_record(
    reactants_str: str,
    raw_text: str,
    beam_rank: int,
    sequence_score: Optional[float],
    tokenizer,
    current_smiles: str,
    stock_keys: Set[str],
    max_reactants: int,
) -> Dict[str, Any]:
    reactants = split_side(reactants_str)
    reactant_rows = [
        {
            "smiles": smi,
            "in_stock": bool(is_in_stock(smi, stock_keys)),
        }
        for smi in reactants
    ]

    stock_count = sum(1 for x in reactant_rows if x["in_stock"])
    num_reactants = len(reactants)

    token_length = len(
        tokenizer(
            reactants_str,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )

    seq_score = None if sequence_score is None else float(sequence_score)
    score_per_token = (
        None if seq_score is None else float(seq_score / max(1, token_length))
    )

    candidate_can = canonicalize_side(reactants_str) or ""
    current_can = canonicalize_smiles(current_smiles) or current_smiles

    valid = bool(valid_side(reactants_str))
    copy_current = bool(candidate_can == current_can)
    too_many_reactants = bool(num_reactants > max_reactants)
    bad_action = bool((not valid) or copy_current or too_many_reactants or num_reactants == 0)

    return {
        "reactants_str": reactants_str,
        "raw_text": raw_text,
        "molt5_rank": int(beam_rank),
        "sequence_score": seq_score,
        "generator_score_per_token": score_per_token,
        "valid": valid,
        "bad_action": bad_action,
        "copy_current": copy_current,
        "too_many_reactants": too_many_reactants,
        "num_reactants": int(num_reactants),
        "stock_count": int(stock_count),
        "stock_ratio": float(stock_count / max(1, num_reactants)),
        "reactants": reactant_rows,
    }


def generate_batch_candidates(
    batch_states: List[Dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    topk: int,
    num_beams: int,
    max_source_length: int,
    max_target_length: int,
    fp16: bool,
    stock_keys: Set[str],
    max_reactants: int,
) -> List[List[Dict[str, Any]]]:
    prompts = [
        build_molt5_prompt(
            target_smiles=row["target_smiles"],
            current_smiles=row["state"]["current_smiles"],
            current_depth=row["state"]["current_depth"],
            max_depth=row["state"]["max_depth"],
        )
        for row in batch_states
    ]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = {
        "num_beams": int(num_beams),
        "num_return_sequences": int(topk),
        "early_stopping": True,
        "max_length": int(max_target_length),
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    with torch.inference_mode():
        if fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                gen_out = model.generate(**enc, **gen_kwargs)
        else:
            gen_out = model.generate(**enc, **gen_kwargs)

    decoded = tokenizer.batch_decode(
        gen_out.sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    if getattr(gen_out, "sequences_scores", None) is not None:
        seq_scores = gen_out.sequences_scores.detach().float().cpu().tolist()
    else:
        seq_scores = [None] * len(decoded)

    all_candidates: List[List[Dict[str, Any]]] = []

    for i, state in enumerate(batch_states):
        current = state["state"]["current_smiles"]
        raw_preds = decoded[i * topk:(i + 1) * topk]
        raw_scores = seq_scores[i * topk:(i + 1) * topk]

        candidates: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for j, (raw_text, seq_score) in enumerate(zip(raw_preds, raw_scores), start=1):
            normalized = normalize_prediction_text(raw_text)
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)

            record = build_candidate_record(
                reactants_str=normalized,
                raw_text=raw_text,
                beam_rank=j,
                sequence_score=seq_score,
                tokenizer=tokenizer,
                current_smiles=current,
                stock_keys=stock_keys,
                max_reactants=max_reactants,
            )
            candidates.append(record)

        # Candidate IDs follow the de-duplicated MolT5 order.
        for idx, cand in enumerate(candidates):
            cand["candidate_id"] = candidate_id(idx)

        all_candidates.append(candidates)

    return all_candidates


# -----------------------------------------------------------------------------
# ChemDFM prompt and supervision
# -----------------------------------------------------------------------------

def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def build_chemdfm_prompt(row: Dict[str, Any]) -> str:
    state = row["state"]
    candidates = row["candidates"]

    lines: List[str] = []
    lines.append("[Round 0]")
    lines.append("Human: You are a retrosynthesis candidate reranker.")
    lines.append("")
    lines.append(
        "Select the most promising one-step retrosynthetic candidate for the CURRENT "
        "molecule, considering its role in the FULL multi-step synthesis."
    )
    lines.append(
        "The objective is to maximize the likelihood of completing the final target "
        "from purchasable starting materials within the search budget."
    )
    lines.append("Use the route context when comparing candidates.")
    lines.append("Do not propose new reactants. Choose only from the candidates provided.")
    lines.append("")
    lines.append("[FINAL TARGET]")
    lines.append(row["target_smiles"])
    lines.append("")
    lines.append("[SEARCH STATE]")
    lines.append(f"Current molecule: {state['current_smiles']}")
    lines.append(f"Current depth: {state['current_depth']}")
    lines.append(f"Maximum search depth: {state['max_depth']}")
    lines.append(f"Remaining depth budget: {state['remaining_depth_budget']}")
    lines.append("")
    lines.append("Current frontier:")

    frontier = state.get("frontier", [])
    if frontier:
        for idx, item in enumerate(frontier, start=1):
            lines.append(
                f"  F{idx}. {item['smiles']} | depth={item['depth']} | "
                f"purchasable={yes_no(bool(item['in_stock']))}"
            )
    else:
        lines.append("  None")

    lines.append("")
    lines.append("[ROUTE HISTORY]")
    history = state.get("history", [])
    if history:
        for h in history:
            lines.append(
                f"  Step {h['step']}: {h['product_smiles']} -> {h['reactants_str']}"
            )
    else:
        lines.append("  None")

    lines.append("")
    lines.append("[CANDIDATES]")
    for cand in candidates:
        cid = cand["candidate_id"]
        lines.append(f"{cid}. Reactants: {cand['reactants_str']}")
        lines.append(f"   MolT5 rank: {cand['molt5_rank']}")
        lines.append(f"   Structurally valid: {yes_no(bool(cand['valid']))}")
        lines.append(
            f"   Purchasable reactants: {cand['stock_count']}/{cand['num_reactants']}"
        )

    lines.append("")
    lines.append("[OUTPUT]")
    lines.append("Return only the ID of the preferred candidate.")
    lines.append("Assistant:")
    return "\n".join(lines)


def attach_supervision_and_prompt(row: Dict[str, Any]) -> Dict[str, Any]:
    gold_options = set(row.pop("gold_reactants_options", []))
    candidates = row["candidates"]

    positive_ids: List[str] = []
    matched_gold: Dict[str, List[str]] = {}

    for cand in candidates:
        cid = cand["candidate_id"]
        reactants = cand["reactants_str"]
        if reactants in gold_options:
            positive_ids.append(cid)
            matched_gold.setdefault(cid, []).append(reactants)

    positive_ids = sorted(set(positive_ids), key=lambda x: [c["candidate_id"] for c in candidates].index(x))

    row["supervision"] = {
        "source": "reference_route",
        "gold_reactants_options": sorted(gold_options),
        "positive_candidate_ids": positive_ids,
        "matched_gold_reactants": matched_gold,
        "num_positive_candidates": len(positive_ids),
        "trainable_v2": len(positive_ids) > 0,
    }

    # Reserved for V3. Keep the model input unchanged when filling this later.
    row["rollout_supervision"] = None
    row["prompt_version"] = PROMPT_VERSION
    row["chemdfm_prompt"] = build_chemdfm_prompt(row)
    return row


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------

def build_dataset(
    states: List[Dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    output_jsonl: Path,
    checkpoint_json: Path,
    args,
    stock_keys: Set[str],
    resume_stats: Dict[str, int],
) -> Dict[str, Any]:
    start_index = min(int(resume_stats.get("next_state_index", 0)), len(states))

    num_rows = int(resume_stats.get("num_rows", 0))
    num_trainable = int(resume_stats.get("num_trainable", 0))
    num_multi_positive = int(resume_stats.get("num_multi_positive", 0))
    num_candidates_total = 0
    prompt_chars_total = 0

    # When resuming, these two averages only cover newly generated rows. They are
    # reported separately from the exact counts.
    newly_generated = 0

    progress = tqdm(
        total=len(states),
        initial=start_index,
        desc="Generating ChemDFM data",
        unit="state",
        dynamic_ncols=True,
    )

    for start in range(start_index, len(states), args.batch_size):
        batch_states = states[start:start + args.batch_size]

        all_candidates = generate_batch_candidates(
            batch_states=batch_states,
            model=model,
            tokenizer=tokenizer,
            device=device,
            topk=args.topk,
            num_beams=args.num_beams,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            fp16=args.fp16,
            stock_keys=stock_keys,
            max_reactants=args.max_reactants,
        )

        output_rows: List[Dict[str, Any]] = []

        for offset, (state, candidates) in enumerate(zip(batch_states, all_candidates)):
            row = dict(state)
            row["state"] = dict(state["state"])
            row["state"]["frontier"] = [dict(x) for x in state["state"]["frontier"]]
            row["state"]["history"] = [dict(x) for x in state["state"]["history"]]
            row["candidates"] = candidates
            row["_state_index"] = start + offset
            row["molt5_prompt"] = build_molt5_prompt(
                target_smiles=row["target_smiles"],
                current_smiles=row["state"]["current_smiles"],
                current_depth=row["state"]["current_depth"],
                max_depth=row["state"]["max_depth"],
            )

            row = attach_supervision_and_prompt(row)
            output_rows.append(row)

            num_rows += 1
            newly_generated += 1
            num_candidates_total += len(candidates)
            prompt_chars_total += len(row["chemdfm_prompt"])

            sup = row["supervision"]
            if bool(sup["trainable_v2"]):
                num_trainable += 1
            if int(sup["num_positive_candidates"]) > 1:
                num_multi_positive += 1

        append_jsonl(output_rows, output_jsonl)

        next_index = start + len(batch_states)
        checkpoint = {
            "next_state_index": next_index,
            "num_rows": num_rows,
            "num_trainable": num_trainable,
            "num_multi_positive": num_multi_positive,
            "output_jsonl": str(output_jsonl),
            "completed": next_index >= len(states),
        }
        atomic_write_json(checkpoint, checkpoint_json)

        progress.update(len(batch_states))

        del output_rows, all_candidates, batch_states
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    progress.close()

    return {
        "num_states": len(states),
        "num_rows": num_rows,
        "num_trainable_v2": num_trainable,
        "trainable_v2_rate": num_trainable / num_rows if num_rows else 0.0,
        "num_multi_positive": num_multi_positive,
        "multi_positive_rate": num_multi_positive / num_rows if num_rows else 0.0,
        "new_rows_this_run": newly_generated,
        "avg_unique_candidates_new_rows": (
            num_candidates_total / newly_generated if newly_generated else None
        ),
        "avg_prompt_chars_new_rows": (
            prompt_chars_total / newly_generated if newly_generated else None
        ),
    }




def summarize_generated_jsonl(path: Path) -> Dict[str, Any]:
    """Final exact summary computed from the output file (resume-safe)."""
    total = 0
    trainable = 0
    multi_positive = 0
    total_candidates = 0
    total_prompt_chars = 0
    max_prompt_chars = 0
    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0}
    depth_stats: Dict[str, Dict[str, int]] = {}

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1

            candidates = row.get("candidates", []) or []
            total_candidates += len(candidates)

            prompt_chars = len(str(row.get("chemdfm_prompt", "")))
            total_prompt_chars += prompt_chars
            max_prompt_chars = max(max_prompt_chars, prompt_chars)

            sup = row.get("supervision", {}) or {}
            positive_ids = set(sup.get("positive_candidate_ids", []) or [])
            is_trainable = bool(sup.get("trainable_v2", False))
            if is_trainable:
                trainable += 1
            if int(sup.get("num_positive_candidates", 0) or 0) > 1:
                multi_positive += 1

            positive_beam_ranks = [
                int(c.get("molt5_rank", 10**9))
                for c in candidates
                if c.get("candidate_id") in positive_ids
            ]
            for k in hit_counts:
                if any(rank <= k for rank in positive_beam_ranks):
                    hit_counts[k] += 1

            depth = str(int((row.get("state", {}) or {}).get("current_depth", -1)))
            bucket = depth_stats.setdefault(depth, {"rows": 0, "trainable": 0})
            bucket["rows"] += 1
            if is_trainable:
                bucket["trainable"] += 1

    for depth, bucket in depth_stats.items():
        bucket["trainable_rate"] = (
            bucket["trainable"] / bucket["rows"] if bucket["rows"] else 0.0
        )

    return {
        "num_rows": total,
        "num_trainable_v2": trainable,
        "trainable_v2_rate": trainable / total if total else 0.0,
        "num_multi_positive": multi_positive,
        "multi_positive_rate": multi_positive / total if total else 0.0,
        "avg_unique_candidates": total_candidates / total if total else 0.0,
        "avg_prompt_chars": total_prompt_chars / total if total else 0.0,
        "max_prompt_chars": max_prompt_chars,
        "reference_hit_at_1": hit_counts[1] / total if total else 0.0,
        "reference_hit_at_3": hit_counts[3] / total if total else 0.0,
        "reference_hit_at_5": hit_counts[5] / total if total else 0.0,
        "reference_hit_at_10": hit_counts[10] / total if total else 0.0,
        "by_current_depth": depth_stats,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build V2/V3-compatible ChemDFM route-context reranking dataset."
    )

    parser.add_argument(
        "--input_json",
        type=str,
        default="./train_dataset_grouped.json",
        help="Preprocessed grouped JSON (recommended) or original raw JSON.",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="auto",
        choices=["auto", "grouped", "raw"],
    )
    parser.add_argument("--split_name", type=str, default="train")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="./molt5_route_context_sft_u_positive_30epoch/checkpoint-epoch-6",
        help="Frozen route-context MolT5 checkpoint.",
    )
    parser.add_argument(
        "--stock_path",
        type=str,
        default="/workspace/kangchenglong/Multi-step/fusion/zinc_stock_17_04_20.hdf5",
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="./chemdfm_dataset_v2/train_chemdfm.jsonl",
    )
    parser.add_argument("--report_json", type=str, default="")
    parser.add_argument("--checkpoint_json", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max_depth", type=int, default=14)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--max_reactants", type=int, default=5)
    parser.add_argument("--max_history_steps", type=int, default=6)

    parser.add_argument(
        "--max_products",
        type=int,
        default=-1,
        help="Debugging: only use the first N target products; -1 means all.",
    )
    parser.add_argument(
        "--max_states",
        type=int,
        default=-1,
        help="Debugging: only generate the first N aggregated states; -1 means all.",
    )
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="Only reconstruct/inspect route states; do not load MolT5 or generate candidates.",
    )
    parser.add_argument(
        "--states_jsonl",
        type=str,
        default="",
        help="Optional path to save extracted aggregated teacher-forced states.",
    )

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.num_beams < args.topk:
        raise ValueError("--num_beams must be >= --topk")
    if args.topk <= 0:
        raise ValueError("--topk must be positive")

    set_seed(args.seed)

    input_path = Path(args.input_json)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    report_json = Path(args.report_json) if args.report_json else output_jsonl.with_suffix(".report.json")
    checkpoint_json = (
        Path(args.checkpoint_json)
        if args.checkpoint_json
        else output_jsonl.with_suffix(".checkpoint.json")
    )

    if args.overwrite:
        for path in (output_jsonl, report_json, checkpoint_json):
            if path.exists():
                path.unlink()

    print("===== Build ChemDFM Route-Context Dataset =====")
    print(f"input_json        : {input_path}")
    print(f"input_format      : {args.input_format}")
    print(f"split_name        : {args.split_name}")
    print(f"model_dir         : {args.model_dir}")
    print(f"stock_path        : {args.stock_path}")
    print(f"output_jsonl      : {output_jsonl}")
    print(f"topk / beams      : {args.topk} / {args.num_beams}")
    print(f"max_depth         : {args.max_depth}")
    print(f"max_history_steps : {args.max_history_steps}")
    print(f"rdkit_available   : {HAS_RDKIT}")

    if not input_path.exists():
        raise FileNotFoundError(f"input JSON not found: {input_path}")
    if not HAS_RDKIT:
        raise RuntimeError(
            "RDKit is required for this dataset builder because canonicalization and "
            "stock matching must be consistent with the existing pipeline."
        )

    raw_data = load_json(input_path)
    input_format = detect_input_format(raw_data, args.input_format)
    print(f"detected_format   : {input_format}")

    print(f"[INFO] loading stock database: {args.stock_path}")
    stock_keys = load_stock_keys(args.stock_path) if args.stock_path else set()
    print(f"[INFO] stock keys loaded: {len(stock_keys)}")

    if input_format == "grouped":
        if not isinstance(raw_data, list):
            raise TypeError("grouped input must be a JSON list")
        routes_iter = iter_grouped_routes(raw_data)
    else:
        if not isinstance(raw_data, dict):
            raise TypeError("raw input must be a JSON object keyed by target SMILES")
        routes_iter = iter_raw_routes(raw_data)

    states, extraction_stats = extract_and_aggregate_states(
        routes=routes_iter,
        stock_keys=stock_keys,
        max_depth=args.max_depth,
        split_name=args.split_name,
        max_products=args.max_products,
        max_history_steps=args.max_history_steps,
    )

    if args.max_states > 0:
        states = states[:args.max_states]

    print("\n===== Teacher-Forced State Extraction =====")
    print(json.dumps(extraction_stats, ensure_ascii=False, indent=2))
    print(f"states_after_max_states: {len(states)}")

    if args.states_jsonl:
        states_path = Path(args.states_jsonl)
        states_path.parent.mkdir(parents=True, exist_ok=True)
        with open(states_path, "w", encoding="utf-8") as f:
            for idx, state in enumerate(states):
                tmp = dict(state)
                tmp["_state_index"] = idx
                f.write(json.dumps(tmp, ensure_ascii=False) + "\n")
        print(f"[INFO] extracted states saved: {states_path}")

    if args.extract_only:
        report = {
            "schema_version": SCHEMA_VERSION,
            "input_json": str(input_path),
            "input_format": input_format,
            "split_name": args.split_name,
            "stock_path": args.stock_path,
            "max_depth": args.max_depth,
            "max_history_steps": args.max_history_steps,
            "extraction": extraction_stats,
            "num_states_after_limit": len(states),
            "extract_only": True,
        }
        write_json(report, report_json)
        print(f"[DONE] extract-only report: {report_json}")
        return

    model_path = Path(args.model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"MolT5 model directory not found: {model_path}")

    device = torch.device(
        args.device
        if torch.cuda.is_available() and str(args.device).startswith("cuda")
        else "cpu"
    )
    print(f"[INFO] device: {device}")

    print(f"[INFO] loading frozen MolT5: {args.model_dir}")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for MolT5 candidate generation. "
            "Install/use the same environment as your existing MolT5 scripts."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.to(device)
    if args.fp16 and device.type == "cuda":
        model.half()
    model.eval()

    resume_stats = inspect_and_repair_jsonl(output_jsonl)
    if resume_stats["next_state_index"] > 0:
        print(
            f"[RESUME] rows={resume_stats['num_rows']} "
            f"next_state_index={resume_stats['next_state_index']}"
        )
    else:
        print("[RESUME] starting from state 0")

    if resume_stats["next_state_index"] >= len(states):
        print("[DONE] output already covers all extracted states.")
        generation_stats = {
            "num_states": len(states),
            "num_rows": resume_stats["num_rows"],
            "num_trainable_v2": resume_stats["num_trainable"],
            "trainable_v2_rate": (
                resume_stats["num_trainable"] / resume_stats["num_rows"]
                if resume_stats["num_rows"] else 0.0
            ),
            "num_multi_positive": resume_stats["num_multi_positive"],
            "new_rows_this_run": 0,
        }
    else:
        generation_stats = build_dataset(
            states=states,
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_jsonl=output_jsonl,
            checkpoint_json=checkpoint_json,
            args=args,
            stock_keys=stock_keys,
            resume_stats=resume_stats,
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "input_json": str(input_path),
        "input_format": input_format,
        "split_name": args.split_name,
        "model_dir": args.model_dir,
        "stock_path": args.stock_path,
        "output_jsonl": str(output_jsonl),
        "topk": args.topk,
        "num_beams": args.num_beams,
        "max_depth": args.max_depth,
        "max_history_steps": args.max_history_steps,
        "max_reactants": args.max_reactants,
        "rdkit_available": HAS_RDKIT,
        "extraction": extraction_stats,
        "generation": generation_stats,
        "final_dataset_summary": summarize_generated_jsonl(output_jsonl),
        "prompt_version": PROMPT_VERSION,
        "completed": True,
    }

    atomic_write_json(report, report_json)
    atomic_write_json(
        {
            "next_state_index": len(states),
            "num_rows": generation_stats.get("num_rows", len(states)),
            "num_trainable": generation_stats.get("num_trainable_v2", 0),
            "num_multi_positive": generation_stats.get("num_multi_positive", 0),
            "output_jsonl": str(output_jsonl),
            "completed": True,
        },
        checkpoint_json,
    )

    print("\n===== Final Report =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[DONE] dataset : {output_jsonl}")
    print(f"[DONE] report  : {report_json}")
    print(f"[DONE] checkpoint: {checkpoint_json}")


if __name__ == "__main__":
    main()
