
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable


LABELS = ("metaphor", "simile", "idiom", "sarcasm")
LABEL_ALIASES = {
    "metaphor": "metaphor",
    "metaphors": "metaphor",
    "simile": "simile",
    "similes": "simile",
    "idiom": "idiom",
    "idioms": "idiom",
    "sarcasm": "sarcasm",
    "sarcastic": "sarcasm",
}

FIELD_CANDIDATES = {
    "id": ("id", "uid", "example_id", "idx", "index"),
    "premise": ("premise", "sentence1", "text1", "context", "source"),
    "hypothesis": ("hypothesis", "sentence2", "text2", "target"),
    "nli_label": ("nli_label", "nli", "entailment_label", "gold_nli", "gold_label"),
    "type": ("figurative_type", "fig_type", "category", "phenomenon", "type", "class"),
    "explanation": ("explanation", "gold_explanation", "rationale", "reason"),
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


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".tsv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        for key in ("data", "records", "examples"):
            if isinstance(data, dict) and isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported input format: {path.suffix}")


def load_input_records(input_value: str, split: str) -> list[dict[str, Any]]:
    path = Path(input_value)
    if path.exists():
        return load_records(path)
    return load_huggingface_records(input_value, split)


def first_present(record: dict[str, Any], candidates: Iterable[str]) -> Any:
    lower_map = {str(key).lower(): key for key in record.keys()}
    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None and record.get(key) not in (None, ""):
            return record[key]
    return ""


def value_from(record: dict[str, Any], explicit: str | None, role: str) -> Any:
    if explicit:
        if explicit not in record:
            raise KeyError(f"Column {explicit!r} not found. Available columns: {sorted(record.keys())}")
        return record.get(explicit, "")
    return first_present(record, FIELD_CANDIDATES[role])


def normalize_type(value: Any) -> str | None:
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    return LABEL_ALIASES.get(text)


def make_prompt(premise: str, hypothesis: str, nli_label: str, include_nli: bool) -> str:
    nli_line = f"NLI label: {nli_label}\n" if include_nli else ""
    return (
        "Given the following FLUTE example, identify the type of figurative language involved.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"{nli_line}\n"
        "Question: What type of figurative language is used in this example?\n"
        "Choose one of the following labels: metaphor, simile, idiom, sarcasm.\n\n"
        "Return JSON with keys label and brief_explanation."
    )


def apply_balanced_sample(records: list[dict[str, Any]], sample_per_class: int, seed: int) -> list[dict[str, Any]]:
    if not sample_per_class:
        return records

    grouped = {label: [] for label in LABELS}
    for record in records:
        grouped[record["gold_label"]].append(record)

    rng = random.Random(seed)
    sampled = []
    for label in LABELS:
        rows = grouped[label]
        if len(rows) < sample_per_class:
            raise ValueError(f"Not enough {label} examples: requested {sample_per_class}, found {len(rows)}.")
        rng.shuffle(rows)
        sampled.extend(rows[:sample_per_class])
    rng.shuffle(sampled)
    return sampled


def normalize_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_records = load_input_records(args.input, args.split)
    if args.shuffle:
        random.Random(args.seed).shuffle(raw_records)
    if args.limit:
        raw_records = raw_records[: args.limit]

    normalized = []
    skipped = 0
    for index, record in enumerate(raw_records):
        fig_type = normalize_type(value_from(record, args.type_col, "type"))
        if fig_type not in LABELS:
            skipped += 1
            continue

        premise = str(value_from(record, args.premise_col, "premise")).strip()
        hypothesis = str(value_from(record, args.hypothesis_col, "hypothesis")).strip()
        nli_label = str(value_from(record, args.nli_col, "nli_label")).strip()
        explanation = str(value_from(record, args.explanation_col, "explanation")).strip()
        example_id = str(value_from(record, args.id_col, "id") or f"ex-{index}").strip()

        normalized.append(
            {
                "id": example_id,
                "premise": premise,
                "hypothesis": hypothesis,
                "nli_label": nli_label,
                "gold_label": fig_type,
                "gold_explanation": explanation,
                "prompt": make_prompt(premise, hypothesis, nli_label, include_nli=True),
                "prompt_with_nli": make_prompt(premise, hypothesis, nli_label, include_nli=True),
                "prompt_without_nli": make_prompt(premise, hypothesis, nli_label, include_nli=False),
            }
        )

    if not normalized:
        raise ValueError(
            "No valid examples found. Check the type/category column and ensure it contains "
            "metaphor, simile, idiom, or sarcasm."
        )
    normalized = apply_balanced_sample(normalized, args.sample_per_class, args.seed)
    print(f"Converted {len(normalized)} examples; skipped {skipped} rows with unsupported labels.")
    return normalized


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help='Raw FLUTE-style file, or a Hugging Face dataset id such as "ColumbiaNLP/FLUTE".',
    )
    parser.add_argument("--output", required=True, help="Output classification JSONL path.")
    parser.add_argument("--split", default="train", help="Hugging Face dataset split to load.")
    parser.add_argument("--sample-per-class", type=int, default=0, help="Balanced sample size per label.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of examples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--id-col")
    parser.add_argument("--premise-col")
    parser.add_argument("--hypothesis-col")
    parser.add_argument("--nli-col")
    parser.add_argument("--type-col")
    parser.add_argument("--explanation-col")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = normalize_records(args)
    write_jsonl(records, Path(args.output))


if __name__ == "__main__":
    main()
