#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build DPO preference pairs from route-context SFT top-k candidates.

This script:
1. Reads generated top-20 candidates.
2. Computes:
   - exact_step
   - forward_plausibility using original ReactionT5 forward model
   - valid_score
   - route_future_proxy
   - 3DInfomax similarity to gold reactants
   - bb_ratio
   - bad_action_penalty
3. Computes utility U.
4. Builds DPO pairs:
   - gold > low-quality candidate
   - high-quality non-gold > low-quality non-gold, only when margin is large enough.

Recommended run environment:
  conda activate Retro_R1_forward
"""

import argparse
import gc
import json
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False
    Chem = None


# -------------------------
# Basic chemistry utilities
# -------------------------

def canonicalize_smiles(smiles: str, isomeric: bool = True) -> Optional[str]:
    if smiles is None:
        return None
    smiles = str(smiles).strip().replace(" ", "")
    if not smiles:
        return None
    if not HAS_RDKIT:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return None


def split_reactants(text: str) -> List[str]:
    if text is None:
        return []
    text = str(text).strip().replace(" ", "").rstrip(".")
    if not text:
        return []
    return [p for p in text.split(".") if p.strip()]


def canonicalize_reactants(text_or_list: Any) -> Tuple[Optional[str], Optional[Tuple[str, ...]]]:
    if isinstance(text_or_list, list):
        parts = text_or_list
    else:
        parts = split_reactants(str(text_or_list))

    out = []
    for p in parts:
        cs = canonicalize_smiles(p)
        if cs is None:
            return None, None
        out.append(cs)

    if not out:
        return None, None

    out = sorted(out)
    return ".".join(out), tuple(out)


def canonicalize_side(text: str) -> Optional[str]:
    return canonicalize_reactants(text)[0]


def mol_to_stock_key(smiles: str) -> Optional[str]:
    if not HAS_RDKIT:
        return canonicalize_smiles(smiles)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)
        if not key:
            return None
        return key[:14]
    except Exception:
        return None


def product_equal(a: str, b: str) -> bool:
    ca = canonicalize_side(a)
    cb = canonicalize_side(b)
    return bool(ca and cb and ca == cb)


def cosine_01(a: torch.Tensor, b: torch.Tensor) -> float:
    try:
        if a is None or b is None:
            return 0.0
        a = a.float().view(1, -1)
        b = b.float().view(1, -1)
        if a.shape[-1] != b.shape[-1]:
            return 0.0
        sim = F.cosine_similarity(a, b, dim=-1).item()
        if math.isnan(sim):
            return 0.0
        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
    except Exception:
        return 0.0


def build_route_context_prompt(sample: Dict[str, Any], max_depth: int = 14) -> str:
    target = sample.get("target_smiles", "")
    current = sample.get("current_smiles", sample.get("input_current", ""))
    step_order = sample.get("step_order", 1)

    try:
        current_depth = int(step_order) - 1
    except Exception:
        current_depth = 0

    return (
        "Please predict the reactant of the product:\n"
        f"<target> {target}\n"
        f"<current> {current}\n"
        f"<current_depth> {current_depth}\n"
        f"<max_depth> {max_depth}\n"
        "<goal> purchasable_starting_materials"
    )


# -------------------------
# JSONL helpers
# -------------------------

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Single-step metadata
# -------------------------

def load_single_step_index(single_step_dir: Path) -> Dict[str, Dict[str, Any]]:
    split_files = [
        single_step_dir / "train_single_step_dedup.json",
        single_step_dir / "valid_single_step_no_train_overlap.json",
        single_step_dir / "test_single_step_no_train_valid_overlap.json",
    ]

    index = {}

    for path in split_files:
        if not path.exists():
            continue
        data = load_json(path)
        for sample in data:
            sid = str(sample.get("id", ""))
            if sid:
                index[sid] = sample

    return index


def build_train_material_sets(single_step_dir: Path) -> Tuple[Set[str], Set[str]]:
    """
    Build fallback building-block sets from train materials only.
    Returns:
      material_smiles_set, material_stockkey_set
    """
    path = single_step_dir / "train_single_step_dedup.json"
    material_smiles_set: Set[str] = set()
    material_key_set: Set[str] = set()

    if not path.exists():
        return material_smiles_set, material_key_set

    data = load_json(path)

    for sample in tqdm(data, desc="build train material set"):
        materials = sample.get("materials", [])
        if not isinstance(materials, list):
            continue
        for m in materials:
            cm = canonicalize_smiles(m)
            if cm:
                material_smiles_set.add(cm)
                key = mol_to_stock_key(cm)
                if key:
                    material_key_set.add(key)

    return material_smiles_set, material_key_set


def get_future_reference_set(sample: Dict[str, Any]) -> Set[str]:
    """
    route_future_proxy reference:
      intermediates ∪ materials from the current route sample.
    """
    ref = set()

    for field in ["intermediates", "materials"]:
        xs = sample.get(field, [])
        if not isinstance(xs, list):
            continue
        for x in xs:
            cx = canonicalize_smiles(x)
            if cx:
                ref.add(cx)

    return ref


# -------------------------
# Stock loader
# -------------------------

def load_stock_keys(stock_path: str, project_root: str) -> Set[str]:
    """
    Prefer the project's load_stock if available.
    Fallback: direct pandas read_hdf with key='table'.
    """
    if not stock_path or not os.path.exists(stock_path):
        print(f"[WARN] stock_path not found: {stock_path}")
        return set()

    if project_root and project_root not in sys.path:
        sys.path.append(project_root)

    try:
        from rl_molt5_direct.chem_utils import load_stock
        stock = load_stock(stock_path)
        print(f"[INFO] stock loaded by rl_molt5_direct.chem_utils: {len(stock)}")
        return set(stock)
    except Exception as exc:
        print(f"[WARN] load_stock from project failed: {exc}")

    try:
        import pandas as pd
        stock = pd.read_hdf(stock_path, key="table")
        keys = set(str(x)[:14] for x in stock.inchi_key.values)
        print(f"[INFO] stock loaded by pandas: {len(keys)}")
        return keys
    except Exception as exc:
        print(f"[WARN] pandas read_hdf stock failed: {exc}")
        return set()


# -------------------------
# 3DInfomax / Morgan featurizer
# -------------------------

class MorganOnlyFeaturizer:
    def __init__(self, device: torch.device, fp_dim: int = 600):
        self.device = device
        self.fp_dim = fp_dim
        self.cache: Dict[str, torch.Tensor] = {}

    def _fp(self, smiles: str) -> torch.Tensor:
        if smiles in self.cache:
            return self.cache[smiles].to(self.device)

        arr = np.zeros(self.fp_dim, dtype=np.float32)

        if HAS_RDKIT:
            try:
                from rdkit.Chem import AllChem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.fp_dim)
                    arr[list(fp.GetOnBits())] = 1.0
            except Exception:
                pass

        emb = torch.from_numpy(arr).float()
        self.cache[smiles] = emb.cpu()
        return emb.to(self.device)

    def get_molecules_embedding(self, smiles_list: Sequence[str]) -> torch.Tensor:
        valid = []
        for s in smiles_list:
            cs = canonicalize_smiles(s)
            if cs:
                valid.append(cs)

        if not valid:
            return torch.zeros(self.fp_dim, dtype=torch.float32, device=self.device)

        return torch.stack([self._fp(s) for s in valid], dim=0).mean(dim=0)


def load_featurizer(args, device: torch.device):
    if args.disable_infomax:
        print("[INFO] 3DInfomax disabled. Use Morgan-only featurizer.")
        return MorganOnlyFeaturizer(device=device), "morgan_only"

    if args.project_root and args.project_root not in sys.path:
        sys.path.append(args.project_root)

    try:
        from rl_molt5_direct.chem_utils import load_3dinfomax_model, MoleculeFeaturizer

        value_model = load_3dinfomax_model(
            args.threed_config,
            args.threed_checkpoint,
            args.fusion_root,
            device,
        )

        featurizer = MoleculeFeaturizer(
            device=device,
            value_model=value_model,
            fp_dim=args.infomax_fp_dim,
        )

        if value_model is None:
            print("[WARN] 3DInfomax failed to load. MoleculeFeaturizer will fallback to Morgan FP.")
            return featurizer, "molecule_featurizer_morgan_fallback"

        print("[INFO] 3DInfomax featurizer loaded.")
        return featurizer, "3dinfomax"

    except Exception as exc:
        print(f"[WARN] Could not import/load 3DInfomax utilities: {exc}")
        print("[WARN] Use Morgan-only featurizer.")
        return MorganOnlyFeaturizer(device=device, fp_dim=args.infomax_fp_dim), "morgan_only"


# -------------------------
# Forward scorer
# -------------------------

class ReactionT5ForwardScorer:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        batch_size: int = 32,
        input_max_length: int = 400,
        output_max_length: int = 200,
        forward_topk: int = 5,
        num_beams: int = 5,
        fp16: bool = True,
        cache_path: Optional[Path] = None,
    ):
        self.model_path = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
        self.device = device
        self.batch_size = batch_size
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.forward_topk = forward_topk
        self.num_beams = max(num_beams, forward_topk)
        self.fp16 = fp16
        self.cache_path = cache_path

        self.cache: Dict[str, List[str]] = {}

        if cache_path and cache_path.exists():
            print(f"[INFO] Loading forward cache: {cache_path}")
            with open(cache_path, "rb") as f:
                self.cache = pickle.load(f)
            print(f"[INFO] Forward cache size: {len(self.cache)}")

        print(f"[INFO] Loading ReactionT5 forward model: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(device)
        self.model.eval()

    def save_cache(self):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)
        print(f"[INFO] Saved forward cache: {self.cache_path}, size={len(self.cache)}")

    def _build_input(self, reactants_str: str) -> str:
        return f"REACTANT:{reactants_str}REAGENT:"

    @torch.no_grad()
    def fill_cache(self, reactants_list: Sequence[str], save_every_batches: int = 200):
        todo = []
        for r in reactants_list:
            cr, _ = canonicalize_reactants(r)
            if cr and cr not in self.cache:
                todo.append(cr)

        todo = sorted(set(todo))
        print(f"[INFO] Forward cache todo unique reactants: {len(todo)}")

        for batch_idx, start in enumerate(tqdm(range(0, len(todo), self.batch_size), desc="forward cache")):
            batch_reactants = todo[start:start + self.batch_size]
            texts = [self._build_input(x) for x in batch_reactants]

            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.input_max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            gen_kwargs = {
                "num_beams": self.num_beams,
                "num_return_sequences": self.forward_topk,
                "max_length": self.output_max_length,
                "return_dict_in_generate": True,
                "output_scores": False,
                "early_stopping": True,
            }

            if self.fp16 and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model.generate(**enc, **gen_kwargs)
            else:
                out = self.model.generate(**enc, **gen_kwargs)

            decoded = self.tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)

            for i, r in enumerate(batch_reactants):
                preds = decoded[i * self.forward_topk:(i + 1) * self.forward_topk]
                norm_preds = []
                seen = set()
                for p in preds:
                    cp = canonicalize_side(p)
                    if cp and cp not in seen:
                        seen.add(cp)
                        norm_preds.append(cp)
                self.cache[r] = norm_preds

            del out, enc
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            if self.cache_path and save_every_batches > 0 and (batch_idx + 1) % save_every_batches == 0:
                self.save_cache()

        self.save_cache()

    def score(self, reactants_str: str, current_product: str) -> Tuple[float, Optional[int], List[str]]:
        cr, _ = canonicalize_reactants(reactants_str)
        cp = canonicalize_side(current_product)

        if not cr or not cp:
            return 0.0, None, []

        preds = self.cache.get(cr, [])

        hit_rank = None
        for idx, pred in enumerate(preds, start=1):
            if pred == cp:
                hit_rank = idx
                break

        if hit_rank is None:
            return 0.0, None, preds

        if hit_rank <= 1:
            return 1.0, hit_rank, preds
        if hit_rank <= 3:
            return 0.7, hit_rank, preds
        if hit_rank <= 5:
            return 0.5, hit_rank, preds
        if hit_rank <= 10:
            return 0.3, hit_rank, preds

        return 0.0, hit_rank, preds


# -------------------------
# Candidate scoring
# -------------------------

def compute_bb_ratio(
    reactant_tuple: Optional[Tuple[str, ...]],
    stock_keys: Set[str],
    fallback_material_keys: Set[str],
    fallback_material_smiles: Set[str],
) -> float:
    if not reactant_tuple:
        return 0.0

    hits = 0
    for smi in reactant_tuple:
        hit = False

        key = mol_to_stock_key(smi)
        if stock_keys:
            hit = key is not None and key in stock_keys
        elif fallback_material_keys:
            hit = key is not None and key in fallback_material_keys
        else:
            hit = smi in fallback_material_smiles

        if hit:
            hits += 1

    return hits / len(reactant_tuple)


def compute_route_future_proxy(
    reactant_tuple: Optional[Tuple[str, ...]],
    future_reference_set: Set[str],
) -> float:
    if not reactant_tuple or not future_reference_set:
        return 0.0
    hits = sum(1 for x in reactant_tuple if x in future_reference_set)
    return hits / len(reactant_tuple)


def compute_bad_action_penalty(
    raw_text: str,
    canonical_reactants: Optional[str],
    reactant_tuple: Optional[Tuple[str, ...]],
    current_product: str,
    valid: bool,
    max_reactants: int = 5,
) -> float:
    penalty = 0.0

    if raw_text is None or str(raw_text).strip() == "":
        penalty += 1.0

    if not valid or canonical_reactants is None or reactant_tuple is None:
        penalty += 1.0
        return penalty

    current_canon = canonicalize_side(current_product)

    if current_canon and current_canon in set(reactant_tuple):
        penalty += 1.0

    if len(reactant_tuple) > max_reactants:
        penalty += 0.5

    return penalty


def compute_infomax_similarity(
    featurizer,
    candidate_tuple: Optional[Tuple[str, ...]],
    gold_tuple: Optional[Tuple[str, ...]],
) -> float:
    if not candidate_tuple or not gold_tuple:
        return 0.0
    try:
        e_cand = featurizer.get_molecules_embedding(list(candidate_tuple))
        e_gold = featurizer.get_molecules_embedding(list(gold_tuple))
        return cosine_01(e_cand, e_gold)
    except Exception:
        return 0.0


def compute_utility(features: Dict[str, float], args) -> float:
    return (
        args.w_exact * features["exact_step"]
        + args.w_forward * features["forward_plausibility"]
        + args.w_valid * features["valid_score"]
        + args.w_route_future * features["route_future_proxy"]
        + args.w_infomax * features["infomax_similarity"]
        + args.w_bb * features["bb_ratio"]
        - args.w_bad * features["bad_action_penalty"]
    )


def make_feature_dict(
    *,
    exact_step: float,
    forward_plausibility: float,
    valid_score: float,
    route_future_proxy: float,
    infomax_similarity: float,
    bb_ratio: float,
    bad_action_penalty: float,
    forward_hit_rank: Optional[int],
) -> Dict[str, Any]:
    return {
        "exact_step": float(exact_step),
        "forward_plausibility": float(forward_plausibility),
        "valid_score": float(valid_score),
        "route_future_proxy": float(route_future_proxy),
        "infomax_similarity": float(infomax_similarity),
        "bb_ratio": float(bb_ratio),
        "bad_action_penalty": float(bad_action_penalty),
        "forward_hit_rank": forward_hit_rank,
    }


def score_generated_candidate(
    cand: Dict[str, Any],
    sample: Dict[str, Any],
    gold_tuple: Optional[Tuple[str, ...]],
    future_reference_set: Set[str],
    forward_scorer: Optional[ReactionT5ForwardScorer],
    featurizer,
    stock_keys: Set[str],
    fallback_material_keys: Set[str],
    fallback_material_smiles: Set[str],
    args,
) -> Dict[str, Any]:

    current_product = sample.get("current_smiles", sample.get("input_current", ""))

    raw_text = (
        cand.get("reactants_str")
        or cand.get("raw_text")
        or ""
    )

    canonical_reactants, reactant_tuple = canonicalize_reactants(raw_text)

    valid = bool(canonical_reactants and reactant_tuple)
    if "valid_smiles" in cand:
        valid = bool(cand.get("valid_smiles")) and valid

    exact_step = 1.0 if (reactant_tuple is not None and gold_tuple is not None and reactant_tuple == gold_tuple) else 0.0
    valid_score = 1.0 if valid else 0.0

    if valid and forward_scorer is not None:
        forward_plausibility, forward_hit_rank, forward_preds = forward_scorer.score(
            canonical_reactants,
            current_product,
        )
    else:
        forward_plausibility, forward_hit_rank, forward_preds = 0.0, None, []

    route_future_proxy = compute_route_future_proxy(
        reactant_tuple,
        future_reference_set,
    )

    infomax_similarity = compute_infomax_similarity(
        featurizer,
        reactant_tuple,
        gold_tuple,
    )

    bb_ratio = compute_bb_ratio(
        reactant_tuple,
        stock_keys,
        fallback_material_keys,
        fallback_material_smiles,
    )

    bad_action_penalty = compute_bad_action_penalty(
        raw_text=raw_text,
        canonical_reactants=canonical_reactants,
        reactant_tuple=reactant_tuple,
        current_product=current_product,
        valid=valid,
        max_reactants=args.max_reactants,
    )

    features = make_feature_dict(
        exact_step=exact_step,
        forward_plausibility=forward_plausibility,
        valid_score=valid_score,
        route_future_proxy=route_future_proxy,
        infomax_similarity=infomax_similarity,
        bb_ratio=bb_ratio,
        bad_action_penalty=bad_action_penalty,
        forward_hit_rank=forward_hit_rank,
    )

    utility = compute_utility(features, args)

    return {
        "raw_text": raw_text,
        "reactants_str": canonical_reactants if canonical_reactants else str(raw_text).replace(" ", "").rstrip("."),
        "reactant_tuple": list(reactant_tuple) if reactant_tuple else [],
        "valid_smiles": bool(valid),
        "is_gold": bool(exact_step == 1.0),
        "unique_rank": cand.get("unique_rank"),
        "raw_rank": cand.get("raw_rank"),
        "sequence_score": cand.get("sequence_score"),
        "features": features,
        "utility": float(utility),
        "forward_predictions": forward_preds[:args.forward_topk] if args.save_forward_predictions else [],
    }


def build_gold_item(
    sample: Dict[str, Any],
    gold_reactants: str,
    gold_tuple: Optional[Tuple[str, ...]],
    future_reference_set: Set[str],
    featurizer,
    stock_keys: Set[str],
    fallback_material_keys: Set[str],
    fallback_material_smiles: Set[str],
    args,
) -> Dict[str, Any]:

    current_product = sample.get("current_smiles", sample.get("input_current", ""))

    valid = gold_tuple is not None
    valid_score = 1.0 if valid else 0.0

    route_future_proxy = compute_route_future_proxy(
        gold_tuple,
        future_reference_set,
    )

    bb_ratio = compute_bb_ratio(
        gold_tuple,
        stock_keys,
        fallback_material_keys,
        fallback_material_smiles,
    )

    # Gold is known target supervision.
    # Do not let forward model or 3DInfomax lower gold.
    forward_plausibility = 1.0 if args.assume_gold_forward_plausible else 0.0
    infomax_similarity = 1.0

    bad_action_penalty = compute_bad_action_penalty(
        raw_text=gold_reactants,
        canonical_reactants=gold_reactants,
        reactant_tuple=gold_tuple,
        current_product=current_product,
        valid=valid,
        max_reactants=args.max_reactants,
    )

    features = make_feature_dict(
        exact_step=1.0,
        forward_plausibility=forward_plausibility,
        valid_score=valid_score,
        route_future_proxy=route_future_proxy,
        infomax_similarity=infomax_similarity,
        bb_ratio=bb_ratio,
        bad_action_penalty=bad_action_penalty,
        forward_hit_rank=1 if forward_plausibility > 0 else None,
    )

    utility = compute_utility(features, args)

    return {
        "raw_text": gold_reactants,
        "reactants_str": gold_reactants,
        "reactant_tuple": list(gold_tuple) if gold_tuple else [],
        "valid_smiles": bool(valid),
        "is_gold": True,
        "unique_rank": None,
        "raw_rank": None,
        "sequence_score": None,
        "features": features,
        "utility": float(utility),
        "forward_predictions": [],
    }


# -------------------------
# Pair construction
# -------------------------

def pair_row(
    sample: Dict[str, Any],
    prompt: str,
    chosen: Dict[str, Any],
    rejected: Dict[str, Any],
    pair_type: str,
) -> Dict[str, Any]:

    return {
        "id": sample.get("id"),
        "split": sample.get("split"),
        "route_id": sample.get("route_id"),
        "step_order": sample.get("step_order"),
        "depth": sample.get("depth"),
        "target_smiles": sample.get("target_smiles"),
        "current_smiles": sample.get("current_smiles", sample.get("input_current", "")),

        "prompt": prompt,
        "chosen": chosen["reactants_str"],
        "rejected": rejected["reactants_str"],

        "pair_type": pair_type,
        "chosen_utility": chosen["utility"],
        "rejected_utility": rejected["utility"],
        "utility_gap": chosen["utility"] - rejected["utility"],

        "chosen_features": chosen["features"],
        "rejected_features": rejected["features"],

        "chosen_rank": chosen.get("unique_rank"),
        "rejected_rank": rejected.get("unique_rank"),
    }


def select_gold_rejections(
    scored_candidates: List[Dict[str, Any]],
    max_pairs: int,
    max_invalid_pairs: int,
) -> List[Dict[str, Any]]:

    pool = [x for x in scored_candidates if not x["is_gold"] and x["reactants_str"]]

    # Prefer clearly bad / low utility.
    pool = sorted(
        pool,
        key=lambda x: (
            x["utility"],
            x["features"]["valid_score"],
            x["features"]["forward_plausibility"],
            x["features"]["route_future_proxy"],
            x["features"]["infomax_similarity"],
        ),
    )

    selected = []
    invalid_count = 0
    seen = set()

    for cand in pool:
        key = cand["reactants_str"]
        if key in seen:
            continue

        is_invalid = cand["features"]["valid_score"] < 0.5

        if is_invalid and invalid_count >= max_invalid_pairs:
            continue

        selected.append(cand)
        seen.add(key)

        if is_invalid:
            invalid_count += 1

        if len(selected) >= max_pairs:
            break

    return selected


def select_candidate_quality_pair(
    scored_candidates: List[Dict[str, Any]],
    margin: float,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:

    pool = [
        x for x in scored_candidates
        if not x["is_gold"]
        and x["reactants_str"]
        and x["features"]["valid_score"] > 0.5
    ]

    if len(pool) < 2:
        return None

    pool_sorted = sorted(pool, key=lambda x: x["utility"])
    worst = pool_sorted[0]
    best = pool_sorted[-1]

    if best["reactants_str"] == worst["reactants_str"]:
        return None

    if best["utility"] - worst["utility"] < margin:
        return None

    return best, worst


# -------------------------
# Split processing
# -------------------------

def collect_forward_reactants(candidate_file: Path, max_samples: int = -1) -> Set[str]:
    unique_reactants = set()

    count = 0
    for rec in tqdm(iter_jsonl(candidate_file), desc=f"collect forward inputs {candidate_file.name}"):
        count += 1
        if max_samples > 0 and count > max_samples:
            break

        for cand in rec.get("candidates", []):
            raw = cand.get("reactants_str") or cand.get("raw_text") or ""
            cr, rt = canonicalize_reactants(raw)

            if cr and rt:
                # If candidate file says invalid, skip forward scoring.
                if "valid_smiles" in cand and not cand.get("valid_smiles"):
                    continue
                unique_reactants.add(cr)

    return unique_reactants


def process_split(
    split: str,
    candidate_file: Path,
    sample_index: Dict[str, Dict[str, Any]],
    output_dir: Path,
    forward_scorer: Optional[ReactionT5ForwardScorer],
    featurizer,
    stock_keys: Set[str],
    fallback_material_keys: Set[str],
    fallback_material_smiles: Set[str],
    args,
) -> Dict[str, Any]:

    scored_path = output_dir / f"{split}_scored_candidates.jsonl"
    pair_path = output_dir / f"{split}_dpo_pairs.jsonl"

    stats = Counter()
    pair_type_counter = Counter()
    pair_gap_values = []
    utility_values = []

    with open(scored_path, "w", encoding="utf-8") as scored_f, open(pair_path, "w", encoding="utf-8") as pair_f:
        sample_count = 0

        for rec in tqdm(iter_jsonl(candidate_file), desc=f"score/build pairs {split}"):
            sample_count += 1
            if args.max_samples > 0 and sample_count > args.max_samples:
                break

            sid = str(rec.get("id", ""))
            sample = sample_index.get(sid, {}).copy()

            # Candidate file fields have priority if present.
            for k, v in rec.items():
                if k != "candidates":
                    sample[k] = v

            sample["split"] = split

            current_product = sample.get("current_smiles", sample.get("input_current", rec.get("current_smiles", "")))
            target_smiles = sample.get("target_smiles", rec.get("target_smiles", ""))

            if not current_product:
                stats["skip_no_current"] += 1
                continue

            gold_raw = (
                rec.get("gold_reactants")
                or rec.get("gold_reactants_str")
                or sample.get("reactants_str")
                or ""
            )

            gold_reactants, gold_tuple = canonicalize_reactants(gold_raw)

            if not gold_reactants or not gold_tuple:
                stats["skip_no_gold"] += 1
                continue

            sample["current_smiles"] = canonicalize_side(current_product) or current_product
            sample["target_smiles"] = canonicalize_side(target_smiles) or target_smiles
            sample["reactants_str"] = gold_reactants

            prompt = build_route_context_prompt(sample, max_depth=args.max_depth)
            future_reference_set = get_future_reference_set(sample)

            gold_item = build_gold_item(
                sample=sample,
                gold_reactants=gold_reactants,
                gold_tuple=gold_tuple,
                future_reference_set=future_reference_set,
                featurizer=featurizer,
                stock_keys=stock_keys,
                fallback_material_keys=fallback_material_keys,
                fallback_material_smiles=fallback_material_smiles,
                args=args,
            )

            scored_candidates = []
            generated_gold_found = False

            for cand in rec.get("candidates", []):
                item = score_generated_candidate(
                    cand=cand,
                    sample=sample,
                    gold_tuple=gold_tuple,
                    future_reference_set=future_reference_set,
                    forward_scorer=forward_scorer,
                    featurizer=featurizer,
                    stock_keys=stock_keys,
                    fallback_material_keys=fallback_material_keys,
                    fallback_material_smiles=fallback_material_smiles,
                    args=args,
                )
                scored_candidates.append(item)
                utility_values.append(item["utility"])
                if item["is_gold"]:
                    generated_gold_found = True

            if generated_gold_found:
                stats["samples_gold_in_candidates"] += 1
            else:
                stats["samples_gold_not_in_candidates"] += 1

            scored_record = {
                "id": sample.get("id"),
                "split": split,
                "target_smiles": sample.get("target_smiles"),
                "current_smiles": sample.get("current_smiles"),
                "gold_reactants": gold_reactants,
                "gold_item": gold_item,
                "num_candidates": len(scored_candidates),
                "gold_in_candidates": generated_gold_found,
                "candidates": scored_candidates,
            }
            scored_f.write(json.dumps(scored_record, ensure_ascii=False) + "\n")

            # 1) gold-anchored pairs
            rejections = select_gold_rejections(
                scored_candidates,
                max_pairs=args.max_gold_pairs_per_sample,
                max_invalid_pairs=args.max_invalid_pairs_per_sample,
            )

            for rejected in rejections:
                if gold_item["reactants_str"] == rejected["reactants_str"]:
                    continue

                row = pair_row(
                    sample=sample,
                    prompt=prompt,
                    chosen=gold_item,
                    rejected=rejected,
                    pair_type="gold_vs_generated_low_quality",
                )

                # Even if gap is negative, gold remains chosen by rule.
                pair_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                pair_type_counter[row["pair_type"]] += 1
                pair_gap_values.append(row["utility_gap"])
                stats["pairs"] += 1

            # 2) non-gold quality pairs
            if args.add_candidate_quality_pairs:
                q_pair = select_candidate_quality_pair(
                    scored_candidates,
                    margin=args.candidate_pair_margin,
                )
                if q_pair is not None:
                    chosen, rejected = q_pair

                    row = pair_row(
                        sample=sample,
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        pair_type="generated_high_u_vs_low_u",
                    )

                    pair_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    pair_type_counter[row["pair_type"]] += 1
                    pair_gap_values.append(row["utility_gap"])
                    stats["pairs"] += 1

            stats["samples"] += 1
            stats["generated_candidates"] += len(scored_candidates)

    summary = {
        "split": split,
        "candidate_file": str(candidate_file),
        "scored_candidates_path": str(scored_path),
        "dpo_pairs_path": str(pair_path),
        "stats": dict(stats),
        "pair_type_counter": dict(pair_type_counter),
        "utility_mean": float(np.mean(utility_values)) if utility_values else None,
        "utility_std": float(np.std(utility_values)) if utility_values else None,
        "pair_gap_mean": float(np.mean(pair_gap_values)) if pair_gap_values else None,
        "pair_gap_std": float(np.std(pair_gap_values)) if pair_gap_values else None,
    }

    return summary


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate_dir",
        type=str,
        default="/home/kangchenglong/suyuqing/rl_retrosynthesis/datasets/dpo_candidates_route_context_sft_top20",
    )
    parser.add_argument(
        "--single_step_dir",
        type=str,
        default="/home/kangchenglong/suyuqing/rl_retrosynthesis/datasets/single_step_no_overlap",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/kangchenglong/suyuqing/rl_retrosynthesis/datasets/dpo_pairs_u_v1",
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default="/home/kangchenglong/suyuqing/rl_retrosynthesis",
    )

    parser.add_argument("--splits", nargs="+", default=["train", "valid"])

    parser.add_argument(
        "--forward_model_path",
        type=str,
        default="/home/kangchenglong/suyuqing/rl_retrosynthesis/ReactionT5/model",
    )
    parser.add_argument("--disable_forward", action="store_true")
    parser.add_argument("--forward_topk", type=int, default=5)
    parser.add_argument("--forward_num_beams", type=int, default=5)
    parser.add_argument("--forward_batch_size", type=int, default=32)
    parser.add_argument("--forward_input_max_length", type=int, default=400)
    parser.add_argument("--forward_output_max_length", type=int, default=200)
    parser.add_argument("--forward_fp16", action="store_true")
    parser.add_argument("--forward_cache", type=str, default="")
    parser.add_argument("--save_forward_predictions", action="store_true")

    parser.add_argument(
        "--stock_path",
        type=str,
        default="/workspace/kangchenglong/Multi-step/fusion/zinc_stock_17_04_20.hdf5",
    )
    parser.add_argument(
        "--fusion_root",
        type=str,
        default="/workspace/kangchenglong/Multi-step",
    )
    parser.add_argument(
        "--threed_config",
        type=str,
        default="/workspace/kangchenglong/Multi-step/fusion/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/12.yml",
    )
    parser.add_argument(
        "--threed_checkpoint",
        type=str,
        default="/workspace/kangchenglong/Multi-step/fusion/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt",
    )
    parser.add_argument("--disable_infomax", action="store_true")
    parser.add_argument("--infomax_fp_dim", type=int, default=600)

    parser.add_argument("--max_depth", type=int, default=14)
    parser.add_argument("--max_reactants", type=int, default=5)

    parser.add_argument("--w_exact", type=float, default=4.0)
    parser.add_argument("--w_forward", type=float, default=1.0)
    parser.add_argument("--w_valid", type=float, default=0.5)
    parser.add_argument("--w_route_future", type=float, default=0.5)
    parser.add_argument("--w_infomax", type=float, default=0.3)
    parser.add_argument("--w_bb", type=float, default=0.2)
    parser.add_argument("--w_bad", type=float, default=1.0)

    parser.add_argument("--assume_gold_forward_plausible", action="store_true", default=True)

    parser.add_argument("--max_gold_pairs_per_sample", type=int, default=2)
    parser.add_argument("--max_invalid_pairs_per_sample", type=int, default=1)
    parser.add_argument("--add_candidate_quality_pairs", action="store_true", default=True)
    parser.add_argument("--candidate_pair_margin", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=-1)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.project_root and args.project_root not in sys.path:
        sys.path.append(args.project_root)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] device = {device}")

    candidate_dir = Path(args.candidate_dir)
    single_step_dir = Path(args.single_step_dir)

    print("[INFO] Loading single-step sample index...")
    sample_index = load_single_step_index(single_step_dir)
    print(f"[INFO] sample index size = {len(sample_index)}")

    print("[INFO] Building fallback train material sets...")
    fallback_material_smiles, fallback_material_keys = build_train_material_sets(single_step_dir)
    print(f"[INFO] fallback_material_smiles = {len(fallback_material_smiles)}")
    print(f"[INFO] fallback_material_keys   = {len(fallback_material_keys)}")

    print("[INFO] Loading stock keys...")
    stock_keys = load_stock_keys(args.stock_path, args.project_root)
    if stock_keys:
        print(f"[INFO] bb_ratio will use external stock keys: {len(stock_keys)}")
    else:
        print("[WARN] bb_ratio will fallback to train materials.")

    print("[INFO] Loading 3DInfomax/Morgan featurizer...")
    featurizer, featurizer_type = load_featurizer(args, device)

    forward_cache_path = Path(args.forward_cache) if args.forward_cache else output_dir / "forward_reactants_cache.pkl"

    forward_scorer = None
    if not args.disable_forward:
        forward_scorer = ReactionT5ForwardScorer(
            model_path=args.forward_model_path,
            device=device,
            batch_size=args.forward_batch_size,
            input_max_length=args.forward_input_max_length,
            output_max_length=args.forward_output_max_length,
            forward_topk=args.forward_topk,
            num_beams=args.forward_num_beams,
            fp16=args.forward_fp16,
            cache_path=forward_cache_path,
        )

        # First pass: collect unique candidate reactants and fill forward cache.
        all_forward_inputs = set()
        for split in args.splits:
            candidate_file = candidate_dir / f"{split}_candidates_top20.jsonl"
            if not candidate_file.exists():
                print(f"[WARN] candidate file not found: {candidate_file}")
                continue
            all_forward_inputs.update(
                collect_forward_reactants(candidate_file, max_samples=args.max_samples)
            )

        print(f"[INFO] total unique forward reactants to cache = {len(all_forward_inputs)}")
        forward_scorer.fill_cache(all_forward_inputs)

    summaries = {}

    for split in args.splits:
        candidate_file = candidate_dir / f"{split}_candidates_top20.jsonl"
        if not candidate_file.exists():
            print(f"[WARN] skip missing split file: {candidate_file}")
            continue

        summary = process_split(
            split=split,
            candidate_file=candidate_file,
            sample_index=sample_index,
            output_dir=output_dir,
            forward_scorer=forward_scorer,
            featurizer=featurizer,
            stock_keys=stock_keys,
            fallback_material_keys=fallback_material_keys,
            fallback_material_smiles=fallback_material_smiles,
            args=args,
        )

        summaries[split] = summary
        print(f"\n[SUMMARY {split}]")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    final_report = {
        "args": vars(args),
        "featurizer_type": featurizer_type,
        "stock_keys_size": len(stock_keys),
        "fallback_material_smiles_size": len(fallback_material_smiles),
        "fallback_material_keys_size": len(fallback_material_keys),
        "summaries": summaries,
    }

    write_json(final_report, output_dir / "dpo_pair_build_report.json")
    print(f"\n[DONE] output_dir = {output_dir}")
    print(f"[DONE] report = {output_dir / 'dpo_pair_build_report.json'}")


if __name__ == "__main__":
    main()