import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    script_dir = Path(__file__).parent.parent
    config_file = script_dir / config_path

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def construct_length(text, tokenizer, max_length=512):
    """Truncate text to max_length using tokenizer."""
    tokens = tokenizer.encode(text)
    tokens = tokens[:max_length]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text[:max_length]


def split_train_test_data(data, test_size=0.2, seed=42):
    """Split a dataset into train/test with a fixed, reproducible 80/20 split."""
    if not data:
        return [], [], []

    indices = np.arange(len(data))
    labels = np.array([int(item["label"]) for item in data])

    label_counts = np.bincount(labels, minlength=2)
    stratify = labels if np.all(label_counts >= 2) else None

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    train_idx = sorted(train_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    train_data = [data[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]

    return train_data, test_data, test_idx


def subset_by_indices(data, indices):
    """Subset a list of rows by index."""
    return [data[i] for i in indices]


@torch.no_grad()
def get_llm_prediction(prompt, model, tokenizer, max_new_tokens=8):
    """Get binary prediction from LLM via prompt."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

    if hasattr(inputs, "keys") or isinstance(inputs, dict):
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        input_length = input_ids.shape[1]
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        inputs = inputs.to(model.device)
        input_length = inputs.shape[1]
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[:, input_length:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()

    false_idx = decoded.find("false")
    true_idx = decoded.find("true")
    false_idx = len(decoded) + 1 if false_idx == -1 else false_idx
    true_idx = len(decoded) + 1 if true_idx == -1 else true_idx

    return 1 if false_idx < true_idx else 0


def load_encoder_model(config, model_name, device="cuda"):
    """Load encoder model for classification."""
    model_config = config["model_registry"][model_name]
    hf_model_id = model_config["hf_model_id"]

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        num_labels=2,
        dtype=torch.float32,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_llm_model(config, model_name, device="cuda"):
    """Load LLM for zero-shot evaluation."""
    model_config = config["model_registry"][model_name]
    hf_model_id = model_config["hf_model_id"]

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_encoder_dataset(data, tokenizer, max_length=512):
    """Prepare dataset for encoder model."""
    texts = [item["claim"] for item in data]
    labels = [item["label"] for item in data]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )


def train_encoder_model(config, model, tokenizer, train_dataset, model_name, device="cuda"):
    """Train encoder model on the train split."""
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}_extrinsic",
        num_train_epochs=config["runtime"]["epochs"],
        per_device_train_batch_size=config["runtime"]["batch_size"],
        learning_rate=2e-5,
        logging_strategy="epoch",
        max_grad_norm=1.0,
        save_strategy="no",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    return model


@torch.no_grad()
def evaluate_encoder_model(model, tokenizer, test_dataset, device="cuda"):
    """Evaluate encoder model on test data."""
    model.eval()

    predictions = []
    labels_list = []

    for batch in test_dataset:
        input_ids = torch.tensor([batch["input_ids"]]).to(device)
        attention_mask = torch.tensor([batch["attention_mask"]]).to(device)
        label = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()

        predictions.append(pred)
        labels_list.append(label)

    accuracy = accuracy_score(labels_list, predictions)
    return accuracy, predictions


def evaluate_llm_zero_shot(data, model, tokenizer, config, model_name, device="cuda"):
    """Evaluate LLM on data using zero-shot prompting."""
    predictions = []
    labels_list = []

    max_new_tokens = config["model_registry"][model_name].get("max_new_tokens", 8)

    for item in tqdm(data, desc="LLM Evaluation", leave=False):
        claim = construct_length(item["claim"], tokenizer)
        prompt = (
            f"{claim}\n"
            "Please check if this claim is true or false. Just output 'True' or 'False'."
        )

        pred = get_llm_prediction(prompt, model, tokenizer, max_new_tokens)
        predictions.append(pred)
        labels_list.append(item["label"])

    accuracy = accuracy_score(labels_list, predictions)
    return accuracy, predictions


def load_json_data(file_path):
    """Load JSON dataset."""
    with open(file_path, "r") as f:
        return json.load(f)


def run_extrinsic_experiment(config, model_name, dataset_name, device="cuda", signals_filter=None):
    """Run one model-dataset experiment using an 80/20 split."""
    root_path = Path(__file__).parent.parent
    results = {}

    original_path = root_path / config["paths"]["datasets_dir"] / f"{dataset_name}.json"
    if not original_path.exists():
        print(f"Dataset not found: {original_path}")
        return None

    original_data = load_json_data(original_path)
    if not original_data:
        print(f"Dataset is empty: {original_path}")
        return None

    seed = int(config.get("seed", 42))
    train_data, test_data, test_indices = split_train_test_data(
        original_data,
        test_size=0.2,
        seed=seed,
    )

    model_config = config["model_registry"][model_name]
    model_family = model_config.get("family", "encoder")
    signals_to_eval = signals_filter if signals_filter else config["extrinsic"]["signals"]

    print(f"\n{'='*70}")
    print(f"Model: {model_name} ({model_family}) | Dataset: {dataset_name}")
    print(f"Split: 80% train / 20% test")
    print(f"{'='*70}")

    signal_dir_map = {
        "paraphrase": "llm-generation",
        "tone": "tone",
        "word-choice": "word-choice",
    }

    extrinsic_dir = root_path / "extrinsic/data"

    # ==================== ENCODER MODELS ====================
    if model_family == "encoder":
        print(f"\nLoading encoder model {model_name}...")
        model, tokenizer = load_encoder_model(config, model_name, device)

        print("Training on 80% train split...")
        train_dataset = prepare_encoder_dataset(train_data, tokenizer)
        model = train_encoder_model(config, model, tokenizer, train_dataset, model_name, device)

        print("Evaluating on 20% held-out original split...")
        test_dataset = prepare_encoder_dataset(test_data, tokenizer)
        original_acc, _ = evaluate_encoder_model(model, tokenizer, test_dataset, device)
        results["original"] = original_acc
        print(f"  Original accuracy (20% test): {original_acc:.4f}")

        print("\nEvaluating on extrinsic variants using the same 20% indices...")
        for signal in signals_to_eval:
            dir_name = signal_dir_map.get(signal, signal)
            signal_dir = extrinsic_dir / dir_name
            if not signal_dir.exists():
                continue

            for variant in config["extrinsic"]["profiles"].get(signal, []):
                variant_file = signal_dir / f"{dataset_name}_{variant}.json"
                if not variant_file.exists():
                    continue

                variant_data = load_json_data(variant_file)
                if len(variant_data) <= max(test_indices, default=-1):
                    print(f"  Skipping {signal}_{variant}: variant file shorter than test indices")
                    continue

                variant_test_data = subset_by_indices(variant_data, test_indices)
                variant_test_dataset = prepare_encoder_dataset(variant_test_data, tokenizer)
                variant_acc, _ = evaluate_encoder_model(
                    model,
                    tokenizer,
                    variant_test_dataset,
                    device,
                )

                result_key = f"{signal}_{variant}"
                results[result_key] = variant_acc

                drop = (original_acc - variant_acc) / original_acc * 100 if original_acc > 0 else 0
                print(f"  {signal}_{variant}: {variant_acc:.4f} (drop: {drop:.1f}%)")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================== LLM MODELS ====================
    else:
        print(f"\nLoading LLM model {model_name}...")
        model, tokenizer = load_llm_model(config, model_name, device)

        print("Evaluating on 20% held-out original split...")
        original_acc, _ = evaluate_llm_zero_shot(test_data, model, tokenizer, config, model_name, device)
        results["original"] = original_acc
        print(f"  Original accuracy (20% test): {original_acc:.4f}")

        print("\nEvaluating on extrinsic variants using the same 20% indices...")
        for signal in signals_to_eval:
            dir_name = signal_dir_map.get(signal, signal)
            signal_dir = extrinsic_dir / dir_name
            if not signal_dir.exists():
                continue

            for variant in config["extrinsic"]["profiles"].get(signal, []):
                variant_file = signal_dir / f"{dataset_name}_{variant}.json"
                if not variant_file.exists():
                    continue

                variant_data = load_json_data(variant_file)
                if len(variant_data) <= max(test_indices, default=-1):
                    print(f"  Skipping {signal}_{variant}: variant file shorter than test indices")
                    continue

                variant_test_data = subset_by_indices(variant_data, test_indices)
                variant_acc, _ = evaluate_llm_zero_shot(
                    variant_test_data,
                    model,
                    tokenizer,
                    config,
                    model_name,
                    device,
                )

                result_key = f"{signal}_{variant}"
                results[result_key] = variant_acc

                drop = (original_acc - variant_acc) / original_acc * 100 if original_acc > 0 else 0
                print(f"  {signal}_{variant}: {variant_acc:.4f} (drop: {drop:.1f}%)")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0", help="GPU device IDs")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Models to run (bert, deberta, llama, qwen) - default: all",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Datasets to run - default: from config",
    )
    parser.add_argument(
        "--signals",
        type=str,
        nargs="+",
        help="Signals to evaluate (paraphrase, tone, word-choice) - default: all",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config()

    models_to_run = args.models if args.models else config["models"]
    datasets_to_run = args.datasets if args.datasets else config["datasets"]
    signals_to_eval = args.signals if args.signals else config["extrinsic"]["signals"]

    root_path = Path(__file__).parent.parent
    results_dir = root_path / config["paths"]["results_dir"] / "extrinsic_shortcut"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_file = results_dir / "extrinsic_shortcut_summary.json"

    all_results = {}
    if summary_file.exists():
        with open(summary_file) as f:
            all_results = json.load(f)

    for model_name in models_to_run:
        if model_name not in config["model_registry"]:
            print(f"Warning: Model {model_name} not in config, skipping")
            continue

        if model_name not in all_results:
            all_results[model_name] = {}

        for dataset_name in datasets_to_run:
            results = run_extrinsic_experiment(
                config,
                model_name,
                dataset_name,
                device,
                signals_to_eval,
            )
            if results:
                if dataset_name not in all_results[model_name]:
                    all_results[model_name][dataset_name] = {}

                all_results[model_name][dataset_name].update(results)

                result_file = results_dir / f"{model_name}_{dataset_name}.json"
                with open(result_file, "w") as f:
                    json.dump(results, f, indent=2)

    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY: Extrinsic Shortcut Injection Results")
    print(f"{'='*70}")

    for model_name, datasets in all_results.items():
        print(f"\n{model_name.upper()}")
        for dataset_name, results in datasets.items():
            original_acc = results.get("original", 0)
            print(f"  {dataset_name}:")
            print(f"    Original: {original_acc:.4f}")

            for signal in config["extrinsic"]["signals"]:
                for variant in config["extrinsic"]["profiles"].get(signal, []):
                    key = f"{signal}_{variant}"
                    if key in results:
                        acc = results[key]
                        drop = (original_acc - acc) / original_acc * 100 if original_acc > 0 else 0
                        print(f"    {key}: {acc:.4f} (drop: {drop:.1f}%)")


if __name__ == "__main__":
    main()