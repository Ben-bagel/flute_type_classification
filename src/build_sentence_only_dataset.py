

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


LABELS = ("metaphor", "simile", "idiom", "sarcasm", "non_figurative")
TYPE_MAP = {
    "Metaphor": "metaphor",
    "Simile": "simile",
    "Idiom": "idiom",
    "Sarcasm": "sarcasm",
    "CreativeParaphrase": "non_figurative",
}


def load_huggingface_records(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets first: python -m pip install datasets") from exc

    dataset = load_dataset(dataset_name)
    if hasattr(dataset, "keys"):
        if split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(f"Split {split!r} not found. Available splits: {available}")
        dataset = dataset[split]
    return [dict(row) for row in dataset]


def make_prompt(sentence: str) -> str:
    return (
        "Classify the following sentence by language type.\n\n"
        f"Sentence: {sentence}\n\n"
        "Choose exactly one label: metaphor, simile, idiom, sarcasm, non_figurative.\n\n"
        "Return valid JSON with keys label and brief_explanation."
    )


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_records = load_huggingface_records(args.input, args.split)
    grouped = {label: [] for label in LABELS}
    seen_sentences = set()

    for row in raw_records:
        label = TYPE_MAP.get(str(row.get("type", "")))
        if label not in grouped:
            continue
        sentence = str(row.get(args.sentence_col, "")).strip()
        if not sentence:
            continue
        dedupe_key = " ".join(sentence.lower().split())
        if dedupe_key in seen_sentences:
            continue
        seen_sentences.add(dedupe_key)
        grouped[label].append(
            {
                "id": f"sentence-only-{row.get('id', len(grouped[label]))}",
                "sentence": sentence,
                "gold_label": label,
                "gold_explanation": str(row.get("explanation", "")).strip(),
                "source_type": row.get("type", ""),
                "source_label": row.get("label", ""),
                "labels": list(LABELS),
                "prompt": make_prompt(sentence),
                "prompt_sentence_only": make_prompt(sentence),
            }
        )

    rng = random.Random(args.seed)
    sampled = []
    for label in LABELS:
        rows = grouped[label]
        if len(rows) < args.sample_per_class:
            raise ValueError(f"Not enough {label} examples: requested {args.sample_per_class}, found {len(rows)}.")
        rng.shuffle(rows)
        sampled.extend(rows[: args.sample_per_class])
    rng.shuffle(sampled)
    return sampled


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="ColumbiaNLP/FLUTE")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sentence-col", default="hypothesis")
    parser.add_argument("--sample-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records(args)
    write_jsonl(records, Path(args.output))
    print(f"Wrote {len(records)} sentence-only examples to {args.output}")


if __name__ == "__main__":
    main()
