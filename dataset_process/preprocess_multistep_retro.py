# preprocess_multistep_retro.py

import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set


try:
    from rdkit import Chem
except ImportError as e:
    raise ImportError(
        "请先安装 RDKit，例如：\n"
        "conda install -c conda-forge rdkit\n"
        "或：pip install rdkit-pypi"
    ) from e


ATOM_MAP_PATTERN = re.compile(r":\d+(?=\])")


def strip_atom_mapping_text(smiles: str) -> str:
    """
    兜底用：直接从字符串层面删除原子映射号。
    例如 [CH2:14] -> [CH2]
    """
    return ATOM_MAP_PATTERN.sub("", smiles)


def canonicalize_smiles(smiles: str) -> str:
    """
    删除 atom mapping，并做 canonical SMILES 标准化。

    如果 RDKit 无法正常解析，则退回到仅删除 atom mapping 的字符串。
    """
    if smiles is None:
        return ""

    smiles = smiles.strip()
    if not smiles:
        return ""

    candidate_smiles = [smiles, strip_atom_mapping_text(smiles)]

    for smi in candidate_smiles:
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
                    return Chem.MolToSmiles(
                        mol,
                        canonical=True,
                        isomericSmiles=True
                    )
            except Exception:
                mol = None

        if mol is not None:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

            try:
                return Chem.MolToSmiles(
                    mol,
                    canonical=True,
                    isomericSmiles=True
                )
            except Exception:
                continue

    return strip_atom_mapping_text(smiles)


def unique_sorted(smiles_list: List[str]) -> List[str]:
    """
    去重并排序，保证输出稳定。
    """
    cleaned = [s for s in smiles_list if s]
    return sorted(set(cleaned))


def split_reaction_side(side: str) -> List[str]:
    """
    将反应一侧按 '.' 拆分成多个分子。
    """
    if side is None:
        return []

    parts = [x.strip() for x in side.split(".") if x.strip()]
    return parts


def parse_retro_reaction(reaction_smiles: str) -> Dict[str, Any]:
    """
    解析一条逆合成反应。

    输入格式：
        product >> reactant1.reactant2...

    输出格式：
        {
            "product_smiles": canonical_product,
            "reactants_smiles": [canonical_reactant1, canonical_reactant2, ...]
        }
    """
    if ">>" not in reaction_smiles:
        raise ValueError(f"反应字符串中缺少 >> ：{reaction_smiles}")

    left, right = reaction_smiles.split(">>", 1)

    product_smiles = canonicalize_smiles(left)

    reactants = []
    for mol in split_reaction_side(right):
        reactants.append(canonicalize_smiles(mol))

    reactants = unique_sorted(reactants)

    return {
        "product_smiles": product_smiles,
        "reactants_smiles": reactants
    }


def build_route_from_retro_paths(
    route_id: int,
    retro_paths: List[List[str]],
    final_product_smiles: str,
    materials: List[str]
) -> Dict[str, Any]:
    """
    一个 reaction tree 可能包含多条从最终产物到不同起始原料的线性 retro path。

    这里把同一个 tree 下面的所有 retro_paths 合并成一条完整 route，
    并对重复出现的 reaction step 去重。

    仍然保持逆合成方向：
        product -> precursors
    """
    final_product = canonicalize_smiles(final_product_smiles)
    clean_materials = unique_sorted([canonicalize_smiles(x) for x in materials])

    seen_steps: Set[Tuple[str, Tuple[str, ...]]] = set()
    steps: List[Dict[str, Any]] = []

    for path in retro_paths:
        if isinstance(path, str):
            path = [path]

        for reaction_smiles in path:
            parsed = parse_retro_reaction(reaction_smiles)

            product = parsed["product_smiles"]
            reactants = parsed["reactants_smiles"]

            step_key = (product, tuple(reactants))

            if step_key in seen_steps:
                continue

            seen_steps.add(step_key)

            steps.append({
                "step_order": len(steps) + 1,
                "product_smiles": product,
                "reactants_smiles": reactants
            })

    all_compounds: Set[str] = set()
    all_compounds.add(final_product)

    for step in steps:
        all_compounds.add(step["product_smiles"])
        all_compounds.update(step["reactants_smiles"])

    material_set = set(clean_materials)

    intermediates = sorted(
        mol for mol in all_compounds
        if mol not in material_set and mol != final_product
    )

    return {
        "route_id": route_id,
        "steps": steps,
        "intermediates": intermediates,
        "materials": clean_materials
    }


def process_one_product(
    raw_product_smiles: str,
    product_data: Dict[str, Any],
    inner_path_as_route: bool = False
) -> Dict[str, Any]:
    """
    处理单个 product。

    默认逻辑：
        原始数据中的 "1", "2", ..., "num_reaction_trees"
        各自对应一条完整 route。

    如果 inner_path_as_route=True：
        则把每个 tree 里面的 retro_routes 的每条线性 path
        也单独拆成 route。
    """
    product = canonicalize_smiles(raw_product_smiles)
    depth = product_data.get("depth", None)

    num_reaction_trees = int(product_data.get("num_reaction_trees", 0))

    routes = []
    route_id = 0

    for tree_idx in range(1, num_reaction_trees + 1):
        tree_key = str(tree_idx)

        if tree_key not in product_data:
            continue

        tree_data = product_data[tree_key]

        retro_paths = tree_data.get("retro_routes", [])
        materials = tree_data.get("materials", [])

        if not retro_paths:
            continue

        if inner_path_as_route:
            for single_path in retro_paths:
                route = build_route_from_retro_paths(
                    route_id=route_id,
                    retro_paths=[single_path],
                    final_product_smiles=raw_product_smiles,
                    materials=materials
                )
                routes.append(route)
                route_id += 1
        else:
            route = build_route_from_retro_paths(
                route_id=route_id,
                retro_paths=retro_paths,
                final_product_smiles=raw_product_smiles,
                materials=materials
            )
            routes.append(route)
            route_id += 1

    return {
        "product": product,
        "depth": depth,
        "num_routes": len(routes),
        "routes": routes
    }


def process_dataset(
    input_json: Path,
    output_json: Path,
    inner_path_as_route: bool = False
) -> List[Dict[str, Any]]:
    """
    处理一个完整数据集文件。
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []

    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = []
        for obj in data:
            if "product" not in obj:
                raise ValueError("list 格式数据中，每个对象必须包含 product 字段。")
            iterable.append((obj["product"], obj))
    else:
        raise TypeError("输入 JSON 顶层应该是 dict 或 list。")

    for raw_product_smiles, product_data in iterable:
        processed_obj = process_one_product(
            raw_product_smiles=raw_product_smiles,
            product_data=product_data,
            inner_path_as_route=inner_path_as_route
        )
        processed.append(processed_obj)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    return processed


def flatten_to_route_level(grouped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将 grouped 格式拆成 route-level 格式。

    原格式：
        一个 product 对应多个 routes

    拆分后：
        一个对象只包含一个 product-route pair

    这个格式更适合训练多步逆合成模型。
    """
    flat_data = []

    for product_obj in grouped_data:
        product = product_obj["product"]
        depth = product_obj.get("depth", None)

        for route in product_obj.get("routes", []):
            flat_data.append({
                "product": product,
                "depth": depth,
                "route_id": route["route_id"],
                "steps": route["steps"],
                "intermediates": route["intermediates"],
                "materials": route["materials"]
            })

    return flat_data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help="原始 train/valid/test json 所在文件夹"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed",
        help="处理后数据集输出文件夹"
    )

    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "train_dataset.json",
            "valid_dataset.json",
            "test_dataset.json"
        ],
        help="需要处理的数据集文件名"
    )

    parser.add_argument(
        "--inner_path_as_route",
        action="store_true",
        help=(
            "默认按 num_reaction_trees 把每个 reaction tree 作为一条 route。"
            "如果打开该选项，则把 tree 内部每条 retro_routes 线性路径也单独作为 route。"
        )
    )

    parser.add_argument(
        "--save_flat_route_level",
        action="store_true",
        help="是否额外保存 route-level 拆分数据。"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in args.files:
        input_path = input_dir / filename

        if not input_path.exists():
            print(f"[跳过] 找不到文件：{input_path}")
            continue

        stem = input_path.stem

        grouped_output_path = output_dir / f"{stem}_grouped.json"

        grouped_data = process_dataset(
            input_json=input_path,
            output_json=grouped_output_path,
            inner_path_as_route=args.inner_path_as_route
        )

        print(f"[完成] grouped 数据已保存：{grouped_output_path}")

        if args.save_flat_route_level:
            flat_data = flatten_to_route_level(grouped_data)
            flat_output_path = output_dir / f"{stem}_route_level.json"

            with open(flat_output_path, "w", encoding="utf-8") as f:
                json.dump(flat_data, f, ensure_ascii=False, indent=2)

            print(f"[完成] route-level 数据已保存：{flat_output_path}")


if __name__ == "__main__":
    main()
