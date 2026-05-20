#!/usr/bin/env python3
"""
Generate extrinsic shortcut variants with Ollama Cloud rewrites.

Outputs:
  extrinsic/data/word-choice/<dataset>_<profile>.json
  extrinsic/data/tone/<dataset>_<profile>.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml
from ollama import Client
from tqdm.auto import tqdm

DEFAULT_PROFILES: Dict[str, List[str]] = {
    "word-choice": ["simple", "complex"],
    "tone": ["formal", "informal"],
}

PROMPT_BUILDERS: Dict[str, str] = {
    "word-choice": (
        "Given a passage, please rewrite it without any explanations. "
        "The content should be the same. Make sure the word choice of the rewritten passage is {profile}. "
        "The passage is: {claim}"
    ),
    "tone": (
        "Given a passage, please rewrite it without any explanations. "
        "The content should be the same. Make sure the tone of the rewritten passage is {profile}. "
        "The passage is: {claim}"
    ),
}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must be a mapping")
    return cfg


def read_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{dataset_path} must contain a JSON list")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{dataset_path}: row {i} is not an object")
        if "claim" not in item or "label" not in item:
            raise KeyError(f"{dataset_path}: row {i} must have 'claim' and 'label'")
    return data


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def truncate_claim(text: str, max_chars: int) -> str:
    text = str(text)
    if max_chars <= 0:
        return text
    return text[:max_chars]


def extract_message_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", "")).strip()
        return str(getattr(message, "content", "")).strip()

    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", "")).strip()
    return ""


def build_ollama_client(host: str) -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Create a key at https://ollama.com/settings/keys "
            "and export OLLAMA_API_KEY before running this script."
        )
    return Client(
        host=host,
        headers={"Authorization": f"Bearer {api_key}"},
    )


def rewrite_claim(
    client: Client,
    *,
    model: str,
    prompt: str,
    max_new_tokens: int,
    request_timeout: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={
                    "temperature": 0,
                    "num_predict": max_new_tokens,
                },
            )
            content = extract_message_content(response)
            if content:
                return content
            raise RuntimeError("Ollama returned an empty rewrite")
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_backoff_sec * attempt)

    raise RuntimeError(f"Ollama rewrite failed after {max_retries} attempts: {last_error}") from last_error


def generate_variant_file(
    *,
    data: List[Dict[str, Any]],
    signal: str,
    profile: str,
    save_path: Path,
    client: Client,
    model: str,
    max_claim_chars: int,
    max_new_tokens: int,
    request_timeout: int,
    max_retries: int,
    retry_backoff_sec: float,
    overwrite: bool,
) -> None:
    if save_path.exists() and not overwrite:
        return

    prompt_template = PROMPT_BUILDERS[signal]
    rows: List[Dict[str, Any]] = []

    for item in tqdm(data, desc=f"{signal}:{profile}", leave=False):
        claim = truncate_claim(item["claim"], max_claim_chars)
        prompt = prompt_template.format(profile=profile, claim=claim)
        new_claim = rewrite_claim(
            client,
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        rows.append({"claim": new_claim, "label": item["label"]})

    write_json(save_path, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate extrinsic variants with Ollama Cloud.")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument(
        "--signals",
        nargs="*",
        default=None,
        choices=["tone", "word-choice"],
        help="Optional override. Example: --signals tone word-choice",
    )
    ap.add_argument("--overwrite", action="store_true", help="Regenerate files even if they already exist.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    datasets_dir = Path(cfg["paths"]["datasets_dir"])
    out_root = Path(cfg["paths"].get("extrinsic_data_dir", "extrinsic/data"))

    extrinsic_cfg = cfg.get("extrinsic", {})
    signals: Sequence[str] = args.signals or extrinsic_cfg.get("signals", ["tone", "word-choice"])
    host = str(extrinsic_cfg.get("ollama_host", "https://ollama.com"))
    model = str(extrinsic_cfg.get("ollama_model", "gpt-oss:120b-cloud"))
    max_claim_chars = int(extrinsic_cfg.get("max_claim_chars", 4000))
    max_new_tokens = int(extrinsic_cfg.get("max_new_tokens", 1000))
    request_timeout = int(extrinsic_cfg.get("request_timeout", 300))
    max_retries = int(extrinsic_cfg.get("max_retries", 3))
    retry_backoff_sec = float(extrinsic_cfg.get("retry_backoff_sec", 2.0))
    profiles_cfg = extrinsic_cfg.get("profiles", DEFAULT_PROFILES)

    client = build_ollama_client(host)

    for dataset_name in cfg["datasets"]:
        dataset_path = datasets_dir / f"{dataset_name}.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

        data = read_dataset(dataset_path)

        for signal in signals:
            profiles = list(profiles_cfg.get(signal, DEFAULT_PROFILES[signal]))
            for profile in profiles:
                save_path = out_root / signal / f"{dataset_name}_{profile}.json"
                generate_variant_file(
                    data=data,
                    signal=signal,
                    profile=profile,
                    save_path=save_path,
                    client=client,
                    model=model,
                    max_claim_chars=max_claim_chars,
                    max_new_tokens=max_new_tokens,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    retry_backoff_sec=retry_backoff_sec,
                    overwrite=args.overwrite,
                )


if __name__ == "__main__":
    main()