# build_single_step_dataset.py

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


ATOM_MAP_RE = re.compile(r":\d+(?=\])")

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    Chem = None


def strip_atom_mapping(smi: str) -> str:
    if smi is None:
        return ""
    return ATOM_MAP_RE.sub("", str(smi).strip())


def canonicalize_smiles(smi: str) -> str:
    """
    去原子映射 + RDKit canonical SMILES。
    如果 RDKit 不可用或解析失败，则退回到去 atom mapping 后的字符串。
    """
    smi = strip_atom_mapping(smi)
    if not smi:
        return ""

    if not HAS_RDKIT:
        return smi

    mol = None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        mol = None

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
        return smi

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return smi


def canonicalize_reactants(reactants: List[str]) -> Tuple[List[str], str]:
    """
    reactants 内部分子 canonicalize 后排序。
    reactants_str 用于 exact match 比较。
    """
    clean = [canonicalize_smiles(x) for x in reactants if x and str(x).strip()]
    clean = [x for x in clean if x]

    clean_sorted = sorted(clean)
    reactants_str = ".".join(clean_sorted)

    return clean_sorted, reactants_str


def build_route_context_input(
    target: str,
    current: str,
    depth_remaining: int,
    goal: str = "building_blocks",
) -> str:

    return (
        f"retrosynthesis: "
        f"<target> {target} "
        f"<current> {current} "
        f"<depth_remaining> {depth_remaining} "
        f"<goal> {goal}"
    )


def iter_route_level_objects(data: Any) -> Iterable[Dict[str, Any]]:
    """
    兼容两种格式：
    1. route-level flat:
       [
         {"product": ..., "route_id": ..., "steps": [...]}
       ]

    2. grouped:
       [
         {"product": ..., "routes": [{"route_id": ..., "steps": [...]}]}
       ]
    """
    if isinstance(data, list):
        for obj in data:
            if "steps" in obj:
                yield obj
            elif "routes" in obj:
                product = obj.get("product", "")
                depth = obj.get("depth", None)
                for r in obj.get("routes", []):
                    route_obj = dict(r)
                    route_obj["product"] = product
                    route_obj["depth"] = depth
                    yield route_obj
            else:
                continue

    elif isinstance(data, dict):
        for product, obj in data.items():
            if isinstance(obj, dict) and "routes" in obj:
                for r in obj.get("routes", []):
                    route_obj = dict(r)
                    route_obj["product"] = product
                    route_obj["depth"] = obj.get("depth", None)
                    yield route_obj
            elif isinstance(obj, dict) and "steps" in obj:
                route_obj = dict(obj)
                route_obj["product"] = product
                yield route_obj
    else:
        raise TypeError("JSON 顶层应为 list 或 dict。")


def make_single_step_samples(
    split_name: str,
    route_level_path: Path,
    dedup_mode: str = "reaction",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    dedup_mode:
    - reaction:
        按 current_product >> reactants 去重。
        适合评估原始 MolT5 current-only baseline。
    - context_reaction:
        按 target + current_product + depth_remaining + reactants 去重。
        适合后续 route-contextualized SFT。
    - none:
        不去重。
    """
    with open(route_level_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_count = 0
    kept_count = 0
    duplicate_count = 0

    seen = set()
    samples: List[Dict[str, Any]] = []

    duplicate_examples = []

    for route_obj in iter_route_level_objects(data):
        target = canonicalize_smiles(route_obj.get("product", ""))
        route_id = route_obj.get("route_id", None)
        depth = route_obj.get("depth", None)
        steps = route_obj.get("steps", [])

        if depth is None:
            depth = len(steps)

        materials = [
            canonicalize_smiles(x)
            for x in route_obj.get("materials", [])
            if x and str(x).strip()
        ]
        intermediates = [
            canonicalize_smiles(x)
            for x in route_obj.get("intermediates", [])
            if x and str(x).strip()
        ]

        for step in steps:
            raw_count += 1

            step_order = int(step.get("step_order", raw_count))
            current = canonicalize_smiles(step.get("product_smiles", ""))
            reactants, reactants_str = canonicalize_reactants(
                step.get("reactants_smiles", [])
            )

            if not current or not reactants_str:
                continue

            try:
                depth_int = int(depth)
            except Exception:
                depth_int = len(steps)

            depth_remaining = max(depth_int - step_order + 1, 0)

            input_current = current
            input_route_context = build_route_context_input(
                target=target,
                current=current,
                depth_remaining=depth_remaining,
                goal="building_blocks",
            )

            reaction_key = f"{current}>>{reactants_str}"
            context_reaction_key = (
                f"{target}||{current}||{depth_remaining}>>{reactants_str}"
            )

            if dedup_mode == "reaction":
                dedup_key = reaction_key
            elif dedup_mode == "context_reaction":
                dedup_key = context_reaction_key
            elif dedup_mode == "none":
                dedup_key = f"{split_name}_{raw_count}_{reaction_key}"
            else:
                raise ValueError(f"Unknown dedup_mode: {dedup_mode}")

            if dedup_key in seen:
                duplicate_count += 1
                if len(duplicate_examples) < 20:
                    duplicate_examples.append(dedup_key)
                continue

            seen.add(dedup_key)

            sample = {
                "id": f"{split_name}_{kept_count:08d}",
                "split": split_name,

                "target_smiles": target,
                "current_smiles": current,
                "reactants_smiles": reactants,
                "reactants_str": reactants_str,

                "depth": depth_int,
                "route_id": route_id,
                "step_order": step_order,
                "depth_remaining": depth_remaining,

                "materials": sorted(materials),
                "intermediates": sorted(intermediates),

                "input_current": input_current,
                "input_route_context": input_route_context,

                "reaction_key": reaction_key,
                "context_reaction_key": context_reaction_key,
            }

            samples.append(sample)
            kept_count += 1

    stats = {
        "split": split_name,
        "input_file": str(route_level_path),
        "dedup_mode": dedup_mode,
        "raw_step_count": raw_count,
        "kept_step_count": kept_count,
        "duplicate_count": duplicate_count,
        "duplicate_examples": duplicate_examples,
        "rdkit_available": HAS_RDKIT,
    }

    return samples, stats


def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help="route-level 数据所在目录。默认当前目录。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./single_step",
        help="single-step 数据输出目录。"
    )
    parser.add_argument(
        "--dedup_mode",
        type=str,
        default="reaction",
        choices=["reaction", "context_reaction", "none"],
        help=(
            "reaction: 按 current>>reactants 去重；"
            "context_reaction: 按 target/current/depth/reactants 去重；"
            "none: 不去重。"
        )
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="需要处理的 split 名。"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    split_reaction_keys = {}

    for split in args.splits:
        input_path = input_dir / f"{split}_dataset_route_level.json"

        if not input_path.exists():
            print(f"[跳过] 找不到文件：{input_path}")
            continue

        samples, stats = make_single_step_samples(
            split_name=split,
            route_level_path=input_path,
            dedup_mode=args.dedup_mode,
        )

        output_path = output_dir / f"{split}_single_step_dedup.json"
        write_json(samples, output_path)

        stats_path = output_dir / f"{split}_single_step_stats.json"
        write_json(stats, stats_path)

        all_stats[split] = stats
        split_reaction_keys[split] = set(x["reaction_key"] for x in samples)

        print(
            f"[完成] {split}: raw={stats['raw_step_count']} "
            f"kept={stats['kept_step_count']} "
            f"dup={stats['duplicate_count']} -> {output_path}"
        )

    # 检查 split 间是否有重复单步反应。
    overlap_report = {}
    split_names = list(split_reaction_keys.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a = split_names[i]
            b = split_names[j]
            overlap = split_reaction_keys[a] & split_reaction_keys[b]
            overlap_report[f"{a}_vs_{b}"] = {
                "num_overlap_reactions": len(overlap),
                "examples": sorted(list(overlap))[:20],
            }

    write_json(all_stats, output_dir / "all_single_step_stats.json")
    write_json(overlap_report, output_dir / "split_overlap_report.json")

    print("[完成] 总统计：", output_dir / "all_single_step_stats.json")
    print("[完成] split 重复检查：", output_dir / "split_overlap_report.json")


if __name__ == "__main__":
    main()
