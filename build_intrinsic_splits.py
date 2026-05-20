import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must be a mapping")
    return cfg


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f)


def get_labels_from_dataset(dataset_path: Path) -> List[int]:
    data = read_json(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"{dataset_path} must be a JSON list")

    labels: List[int] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{dataset_path}: row {i} is not an object")
        if "label" not in row:
            raise KeyError(f"{dataset_path}: row {i} missing 'label'")
        lab = row["label"]
        if not isinstance(lab, (int, bool)):
            # allow string "0"/"1" just in case
            try:
                lab = int(str(lab).strip())
            except Exception:
                raise ValueError(f"{dataset_path}: row {i} label not int/bool: {row['label']}")
        lab = int(lab)
        if lab not in (0, 1):
            raise ValueError(f"{dataset_path}: row {i} label not in {{0,1}}: {lab}")
        labels.append(lab)
    return labels


def split_indices_60_20_20(n: int) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(n))
    train_size = int(n * 0.6)
    val_size = int(n * 0.2)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    return train_indices, val_indices, test_indices


def validate_index_list(name: str, idx: Sequence[int], n: int) -> None:
    for j in idx:
        if not isinstance(j, int):
            raise ValueError(f"{name}: index not int: {j}")
        if j < 0 or j >= n:
            raise ValueError(f"{name}: index out of range [0,{n-1}]: {j}")


def build_perplexity_splits(
    labels: List[int],
    perplexities: List[float],
    seed: int,
) -> Dict[str, List[int]]:
    n = len(labels)
    if len(perplexities) != n:
        raise ValueError(f"perplexities length {len(perplexities)} != labels length {n}")

    train_indices, val_indices, test_indices = split_indices_60_20_20(n)

    med = float(np.median(np.asarray(perplexities, dtype=float)))
    # paper uses: perplexity_labels = [_ < perplexity_median for _ in perplexities]
    # so label is bool; keep 0/1 form for comparisons
    ppl_short = [int(p < med) for p in perplexities]

    train_val_indices: List[int] = []
    for idx in train_indices + val_indices:
        # correlated env in train+val
        if labels[idx] == 0 and ppl_short[idx] == 0:
            train_val_indices.append(idx)
        if labels[idx] == 1 and ppl_short[idx] == 1:
            train_val_indices.append(idx)

    filtered_test: List[int] = []
    for idx in test_indices:
        # shift env for test (paper moves some label==0 into train_val and uses label==1 for filtered test)
        if labels[idx] == 0 and ppl_short[idx] == 1:
            train_val_indices.append(idx)
        if labels[idx] == 1 and ppl_short[idx] == 0:
            filtered_test.append(idx)

    if len(train_val_indices) == 0:
        raise ValueError("perplexity: train_val_indices is empty (cannot split)")

    rng = random.Random(seed)
    rng.shuffle(train_val_indices)

    # random control sampled from original train+val pool (same size as train_val_indices)
    random_indices = rng.sample(train_indices + val_indices, k=len(train_val_indices))

    # resplit 80/20
    train_size2 = int(len(train_val_indices) * 0.8)
    shortcut_train = train_val_indices[:train_size2]
    shortcut_val = train_val_indices[train_size2:]
    random_train = random_indices[:train_size2]
    random_val = random_indices[train_size2:]

    return {
        "train": shortcut_train,
        "val": shortcut_val,
        "filtered_test": filtered_test,
        "random_train": random_train,
        "random_val": random_val,
    }


def build_sentiment_splits(
    labels: List[int],
    sentiments: List[str],
    seed: int,
) -> Dict[str, List[int]]:
    n = len(labels)
    if len(sentiments) != n:
        raise ValueError(f"sentiments length {len(sentiments)} != labels length {n}")

    train_indices, val_indices, test_indices = split_indices_60_20_20(n)

    train_val_indices: List[int] = []
    for idx in train_indices + val_indices:
        s = str(sentiments[idx])
        # correlated env in train+val
        if labels[idx] == 0 and ("Negative" in s):
            train_val_indices.append(idx)
        if labels[idx] == 1 and ("Positive" in s):
            train_val_indices.append(idx)

    filtered_test: List[int] = []
    for idx in test_indices:
        s = str(sentiments[idx])
        # shift env for test
        if labels[idx] == 0 and ("Positive" in s):
            train_val_indices.append(idx)
        if labels[idx] == 1 and ("Negative" in s):
            filtered_test.append(idx)

    rng = random.Random(seed)
    rng.shuffle(train_val_indices)
    random_indices = rng.sample(train_indices + val_indices, k=len(train_val_indices))

    train_size2 = int(len(train_val_indices) * 0.8)
    shortcut_train = train_val_indices[:train_size2]
    shortcut_val = train_val_indices[train_size2:]
    random_train = random_indices[:train_size2]
    random_val = random_indices[train_size2:]

    return {
        "train": shortcut_train,
        "val": shortcut_val,
        "filtered_test": filtered_test,
        "random_train": random_train,
        "random_val": random_val,
    }


def save_split_pack(out_dir: Path, shortcut_name: str, splits: Dict[str, List[int]]) -> None:
    write_json(out_dir / f"{shortcut_name}_filtered_test.json", splits["filtered_test"])
    write_json(out_dir / f"{shortcut_name}_train.json", splits["train"])
    write_json(out_dir / f"{shortcut_name}_val.json", splits["val"])
    write_json(out_dir / f"{shortcut_name}_random_train.json", splits["random_train"])
    write_json(out_dir / f"{shortcut_name}_random_val.json", splits["random_val"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build intrinsic shortcut split indices (paper-style).")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument(
        "--shortcuts",
        nargs="*",
        default=None,
        choices=["sentiment", "perplexity"],
        help="Optional override. Example: --shortcuts sentiment",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    datasets_dir = Path(cfg["paths"]["datasets_dir"])
    data_dir = Path(cfg["paths"]["intrinsic_data_dir"])
    splits_root = Path(cfg.get("paths", {}).get("intrinsic_splits_dir", "intrinsic/splits"))

    dataset_names: List[str] = cfg["datasets"]

    # Use a dedicated split seed like the paper, unless you set one
    split_seed = int(cfg.get("intrinsic", {}).get("split_seed", 20250101))

    cfg_shortcuts = cfg.get("intrinsic", {}).get("signals", ["sentiment", "perplexity"])
    shortcuts = args.shortcuts if args.shortcuts else list(cfg_shortcuts)

    print(f"Datasets: {dataset_names}")
    print(f"Shortcuts: {shortcuts}")
    print(f"signals dir: {data_dir}")
    print(f"splits out:  {splits_root}")
    print(f"split_seed:  {split_seed}")

    for ds in dataset_names:
        dataset_path = datasets_dir / f"{ds}.json"
        labels = get_labels_from_dataset(dataset_path)
        n = len(labels)

        out_dir = splits_root / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        if "sentiment" in shortcuts:
            sent_path = data_dir / f"sentiment_{ds}.json"
            sentiments = read_json(sent_path)
            if not isinstance(sentiments, list):
                raise ValueError(f"{sent_path} must be a JSON list")
            splits = build_sentiment_splits(labels, [str(x) for x in sentiments], seed=split_seed)

            # validate
            for k, v in splits.items():
                validate_index_list(f"{ds}/sentiment/{k}", v, n)

            save_split_pack(out_dir, "sentiment", splits)
            print(
                f"[{ds}] sentiment: train={len(splits['train'])} val={len(splits['val'])} "
                f"filtered_test={len(splits['filtered_test'])} random_train={len(splits['random_train'])}"
            )

        if "perplexity" in shortcuts:
            ppl_path = data_dir / f"perplexity_{ds}.json"
            perplexities = read_json(ppl_path)
            if not isinstance(perplexities, list):
                raise ValueError(f"{ppl_path} must be a JSON list")
            ppl = [float(x) for x in perplexities]
            splits = build_perplexity_splits(labels, ppl, seed=split_seed)

            for k, v in splits.items():
                validate_index_list(f"{ds}/perplexity/{k}", v, n)

            save_split_pack(out_dir, "perplexity", splits)
            print(
                f"[{ds}] perplexity: train={len(splits['train'])} val={len(splits['val'])} "
                f"filtered_test={len(splits['filtered_test'])} random_train={len(splits['random_train'])}"
            )


if __name__ == "__main__":
    main()