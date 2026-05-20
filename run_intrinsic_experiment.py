import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import gc
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from typing import Tuple as TupleType


# ---------------------------------------------------------------------
# Config and data loading
# ---------------------------------------------------------------------
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path: Path) -> pd.DataFrame:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    df = pd.DataFrame(data)
    if "claim" not in df.columns or "label" not in df.columns:
        raise KeyError(f"{path} must have 'claim' and 'label' columns")
    df = df[["claim", "label"]].copy()
    df.rename(columns={"claim": "text"}, inplace=True)
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def subset_by_indices(df: pd.DataFrame, idx_list: Sequence[int]) -> pd.DataFrame:
    return df.iloc[list(idx_list)].copy().reset_index(drop=True)


def load_intrinsic_splits(
    splits_root: Path,
    dataset_name: str,
    shortcut: str,
) -> Dict[str, List[int]]:
    base = splits_root / dataset_name
    splits = {
        "train": read_json(base / f"{shortcut}_train.json"),
        "val": read_json(base / f"{shortcut}_val.json"),
        "filtered_test": read_json(base / f"{shortcut}_filtered_test.json"),
        "random_train": read_json(base / f"{shortcut}_random_train.json"),
        "random_val": read_json(base / f"{shortcut}_random_val.json"),
    }
    return splits


def release_adapter(adapter) -> None:
    if adapter is None:
        return
    for name in ("model", "tokenizer", "optimizer"):
        obj = getattr(adapter, name, None)
        if obj is not None:
            del obj
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_f1": float(f1),
        "macro_precision": float(p),
        "macro_recall": float(r),
    }


# ---------------------------------------------------------------------
# Encoder classifier adapter (BERT / DeBERTa)
# ---------------------------------------------------------------------
class TextClsDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_len: int = 256):
        self.texts = list(texts)
        self.labels = [int(l) for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


class EncoderClassifierAdapter:
    def __init__(self, model_cfg: Dict[str, Any], device: torch.device, runtime_cfg: Dict[str, Any]):
        self.model_key = model_cfg.get("name", "?")
        self.hf_model_id = model_cfg["hf_model_id"]
        self.max_len = int(model_cfg.get("max_seq_length", 256))
        self.device = device

        self.batch_size = int(runtime_cfg.get("batch_size", 16))
        self.num_workers = int(runtime_cfg.get("num_workers", 0))
        self.epochs = int(runtime_cfg.get("epochs", 4))
        self.lr = float(runtime_cfg.get("lr", 2e-5))
        self.weight_decay = float(runtime_cfg.get("weight_decay", 1e-2))
        self.early_stopping_patience = int(runtime_cfg.get("early_stopping_patience", 2))
        self.min_delta = float(runtime_cfg.get("min_delta", 1e-4))

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_model_id,
            num_labels=2,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _make_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = TextClsDataset(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        train_loader = self._make_loader(train_df, shuffle=True)

        best_val_f1 = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"[{self.model_key}] train epoch {epoch+1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())

            val_metrics = self.evaluate(val_df)
            val_f1 = val_metrics["macro_f1"]
            improved = val_f1 > (best_val_f1 + self.min_delta)

            if improved:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"[{self.model_key}] epoch {epoch+1}/{self.epochs} "
                f"train_loss={total_loss / max(1, len(train_loader)):.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_f1:.4f} "
                f"patience={patience_counter}/{self.early_stopping_patience}"
            )
            if patience_counter >= self.early_stopping_patience:
                print(f"[{self.model_key}] early stopping")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        loader = self._make_loader(df, shuffle=False)
        all_true: List[int] = []
        all_pred: List[int] = []

        for batch in tqdm(loader, desc=f"[{self.model_key}] predict"):
            labels = batch["labels"].cpu().numpy().tolist()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            logits = out.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            all_true.extend(labels)
            all_pred.extend(preds)

        return np.array(all_true), np.array(all_pred)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        y_true, y_pred = self.predict(df)
        return compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------
# LLM prompt-based classifier (Llama / Qwen) – zero-shot
# ---------------------------------------------------------------------
class LLMPromptClassifierAdapter:
    def __init__(self, model_cfg: Dict[str, Any], device: torch.device, runtime_cfg: Dict[str, Any]):
        self.model_key = model_cfg.get("name", "?")
        self.hf_model_id = model_cfg["hf_model_id"]
        self.device = device

        self.max_new_tokens = int(model_cfg.get("max_new_tokens", 8))
        self.temperature = float(model_cfg.get("temperature", 0.0))
        self.batch_size = int(runtime_cfg.get("llm_batch_size", 4))

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        # Chat models often need padding setup
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_id,
            torch_dtype=getattr(torch, "bfloat16", torch.float16),
        ).to(self.device)
        self.model.eval()

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        # no fine-tuning here (zero-shot), but you could implement few-shot caching if desired
        return

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        texts = df["text"].tolist()
        labels = df["label"].astype(int).tolist()
        preds: List[int] = []

        for start in tqdm(range(0, len(texts), self.batch_size), desc=f"[{self.model_key}] prompt"):
            batch_texts = texts[start : start + self.batch_size]
            prompts = [self._build_prompt(t) for t in batch_texts]

            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            outputs = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
            )

            # decode only the generated continuation
            for i, out_ids in enumerate(outputs):
                prompt_len = enc["input_ids"][i].shape[0]
                gen_ids = out_ids[prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                preds.append(self._parse_label(text))

        return np.array(labels), np.array(preds)

    def _build_prompt(self, claim: str) -> str:
        return (
            "You are a misinformation detection model.\n\n"
            "Classify the following claim as REAL or FAKE news.\n"
            "Respond with a single word: REAL or FAKE.\n\n"
            f"Claim: {claim}\n"
            "Answer:"
        )

    def _parse_label(self, output: str) -> int:
        o = output.strip().lower()
        if "fake" in o and "real" not in o:
            return 1
        if "real" in o and "fake" not in o:
            return 0
        # fallback: simple heuristic
        if o.startswith("fake"):
            return 1
        if o.startswith("real"):
            return 0
        # default to fake for ambiguous
        return 1

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        y_true, y_pred = self.predict(df)
        return compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------
@dataclass
class ResultRow:
    dataset: str
    model: str
    family: str
    setting: str         # "intrinsic"
    shortcut: str        # "sentiment" / "perplexity"
    split_type: str      # "shortcut" / "random_control"
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    n_train: int
    n_val: int
    n_test: int


# ---------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Run intrinsic shortcut experiments.")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument(
        "--shortcuts",
        nargs="*",
        default=None,
        choices=["sentiment", "perplexity"],
        help="Optional subset of intrinsic shortcuts to run.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    datasets_dir = Path(cfg["paths"]["datasets_dir"])
    splits_root = Path(cfg["paths"]["intrinsic_splits_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "raw" / "intrinsic_all_runs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dataset_names: List[str] = cfg["datasets"]
    all_models: List[str] = cfg["models"]
    model_registry: Dict[str, Dict[str, Any]] = cfg["model_registry"]

    intrinsic_cfg = cfg.get("intrinsic", {})
    cfg_shortcuts = intrinsic_cfg.get("signals", ["sentiment", "perplexity"])
    shortcuts = args.shortcuts if args.shortcuts else cfg_shortcuts

    runtime_cfg = {
        "batch_size": cfg.get("runtime", {}).get("batch_size", 16),
        "llm_batch_size": cfg.get("runtime", {}).get("llm_batch_size", 4),
        "num_workers": cfg.get("runtime", {}).get("num_workers", 0),
        "epochs": cfg.get("runtime", {}).get("epochs", 4),
        "lr": cfg.get("runtime", {}).get("lr", 2e-5),
        "weight_decay": cfg.get("runtime", {}).get("weight_decay", 1e-2),
        "early_stopping_patience": cfg.get("runtime", {}).get("early_stopping_patience", 2),
        "min_delta": cfg.get("runtime", {}).get("min_delta", 1e-4),
    }

    device = resolve_device(cfg.get("runtime", {}).get("device", "auto"))
    print(f"Device: {device}")
    print(f"Datasets: {dataset_names}")
    print(f"Models: {all_models}")
    print(f"Shortcuts: {shortcuts}")
    print(f"Writing results to: {out_csv}")

    rows: List[ResultRow] = []

    for ds in dataset_names:
        print(f"\n==================== Dataset: {ds} ====================")
        df = load_dataset(datasets_dir / f"{ds}.json")

        for shortcut in shortcuts:
            print(f"\n--- Shortcut: {shortcut} ---")
            splits = load_intrinsic_splits(splits_root, ds, shortcut)

            s_train_df = subset_by_indices(df, splits["train"])
            s_val_df = subset_by_indices(df, splits["val"])
            s_test_df = subset_by_indices(df, splits["filtered_test"])

            r_train_df = subset_by_indices(df, splits["random_train"])
            r_val_df = subset_by_indices(df, splits["random_val"])
            # random-control uses same filtered_test
            r_test_df = s_test_df

            for model_key in all_models:
                if model_key not in model_registry:
                    print(f"  [warn] model {model_key} not in model_registry; skipping")
                    continue
                m_cfg = dict(model_registry[model_key])
                m_cfg.setdefault("name", model_key)
                family = m_cfg.get("family", "encoder")

                print(f"\nModel: {model_key} ({family})")

                # build adapter
                if family == "encoder":
                    adapter = EncoderClassifierAdapter(m_cfg, device=device, runtime_cfg=runtime_cfg)
                elif family == "llm":
                    adapter = LLMPromptClassifierAdapter(m_cfg, device=device, runtime_cfg=runtime_cfg)
                else:
                    print(f"  [warn] unknown family '{family}' for {model_key}; skipping")
                    continue

                # --- shortcut split run ---
                print("  [shortcut] train/val on correlated env")
                adapter.fit(s_train_df, s_val_df)
                shortcut_metrics = adapter.evaluate(s_test_df)
                rows.append(
                    ResultRow(
                        dataset=ds,
                        model=model_key,
                        family=family,
                        setting="intrinsic",
                        shortcut=shortcut,
                        split_type="shortcut",
                        accuracy=shortcut_metrics["accuracy"],
                        macro_f1=shortcut_metrics["macro_f1"],
                        macro_precision=shortcut_metrics["macro_precision"],
                        macro_recall=shortcut_metrics["macro_recall"],
                        n_train=len(s_train_df),
                        n_val=len(s_val_df),
                        n_test=len(s_test_df),
                    )
                )
                print("    shortcut metrics:", shortcut_metrics)

                # --- random-control run ---
                print("  [random_control] train/val on random env")
                adapter.fit(r_train_df, r_val_df)
                random_metrics = adapter.evaluate(r_test_df)
                rows.append(
                    ResultRow(
                        dataset=ds,
                        model=model_key,
                        family=family,
                        setting="intrinsic",
                        shortcut=shortcut,
                        split_type="random_control",
                        accuracy=random_metrics["accuracy"],
                        macro_f1=random_metrics["macro_f1"],
                        macro_precision=random_metrics["macro_precision"],
                        macro_recall=random_metrics["macro_recall"],
                        n_train=len(r_train_df),
                        n_val=len(r_val_df),
                        n_test=len(r_test_df),
                    )
                )
                print("    random-control metrics:", random_metrics)

                # checkpoint save
                df_out = pd.DataFrame([r.__dict__ for r in rows])
                df_out.to_csv(out_csv, index=False)
                print(f"  [checkpoint saved] {out_csv} (rows={len(rows)})")

                release_adapter(adapter)
                adapter = None

    # final save
    df_out = pd.DataFrame([r.__dict__ for r in rows])
    df_out.to_csv(out_csv, index=False)
    print(f"\nDone. Final rows: {len(rows)}")
    print(df_out.sort_values(["dataset", "model", "shortcut", "split_type"]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()