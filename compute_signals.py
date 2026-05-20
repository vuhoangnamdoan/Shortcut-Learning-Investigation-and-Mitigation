import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from evaluate import load as load_metric
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must be a mapping")
    return cfg


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requested cuda but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{dataset_path} must contain a JSON list")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{dataset_path}: row {i} is not an object")
        if "claim" not in item:
            raise KeyError(f"{dataset_path}: row {i} missing 'claim'")
    return data


@torch.no_grad()
def predict_sentiment_labels(
    texts: List[str],
    model_name: str,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 8,
) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    sentiment_map = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive",
    }

    outputs: List[str] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="sentiment", leave=False):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            [t.lower() for t in batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)
        logits = model(**enc).logits
        preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).tolist()
        outputs.extend([sentiment_map[int(p)] for p in preds])

    return outputs


def compute_perplexities(
    texts: List[str],
    model_id: str,
    device: torch.device,
    max_length: int = 512,
) -> List[float]:
    metric = load_metric("perplexity", module_type="metric")
    # evaluate perplexity expects device string
    dev = "cuda" if device.type == "cuda" else "cpu"
    res = metric.compute(
        predictions=texts,
        model_id=model_id,
        device=dev,
        max_length=max_length,
    )
    ppl = res.get("perplexities")
    if ppl is None:
        raise RuntimeError("Perplexity metric did not return 'perplexities'")
    return [float(x) for x in ppl]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute intrinsic signals.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--signals",
        nargs="*",
        default=None,
        choices=["sentiment", "perplexity"],
        help="Optional override. Example: --signals sentiment",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    datasets_dir = Path(cfg["paths"]["datasets_dir"])
    out_dir = Path(cfg["paths"]["intrinsic_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names: List[str] = cfg["datasets"]
    cfg_signals: List[str] = cfg["intrinsic"]["signals"]
    signals = args.signals if args.signals else cfg_signals

    device = resolve_device(cfg["runtime"].get("device", "auto"))
    batch_size = int(cfg["runtime"].get("batch_size", 8))

    sent_cfg = cfg["intrinsic"]["sentiment"]
    ppl_cfg = cfg["intrinsic"]["perplexity"]

    print(f"Device: {device}")
    print(f"Datasets: {dataset_names}")
    print(f"Signals: {signals}")

    for ds in dataset_names:
        ds_path = datasets_dir / f"{ds}.json"
        data = read_dataset(ds_path)
        texts = [str(x["claim"]) for x in data]
        print(f"\n[{ds}] n={len(texts)}")

        if "sentiment" in signals:
            sentiment = predict_sentiment_labels(
                texts=texts,
                model_name=sent_cfg["model_name"],
                device=device,
                max_length=int(sent_cfg.get("max_length", 512)),
                batch_size=batch_size,
            )
            out_path = out_dir / f"sentiment_{ds}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(sentiment, f, ensure_ascii=False)
            print(f"  saved {out_path}")

        if "perplexity" in signals:
            perplexities = compute_perplexities(
                texts=texts,
                model_id=ppl_cfg["model_id"],
                device=device,
                max_length=int(ppl_cfg.get("max_length", 512)),
            )
            out_path = out_dir / f"perplexity_{ds}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(perplexities, f)
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()