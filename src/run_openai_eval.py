
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_LABELS = ("metaphor", "simile", "idiom", "sarcasm")

DEVELOPER_INSTRUCTIONS = """You are doing sentence classification.
Choose exactly one label from the options in the user prompt.
Use only the information shown in the user prompt. Do not infer any extra label.
Return only a JSON object with:
{"label": "...", "brief_explanation": "..."}
Use valid JSON with double-quoted keys and strings.
The explanation should be one concise sentence about the linguistic cue."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            snippet = match.group(0)
            try:
                data = json.loads(snippet)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(snippet)
                except (SyntaxError, ValueError):
                    repaired = re.sub(r"([{,]\s*)(label|brief_explanation)\s*:", r'\1"\2":', snippet)
                    repaired = repaired.replace("'", '"')
                    try:
                        data = json.loads(repaired)
                    except json.JSONDecodeError:
                        data = fallback_parse(text)
        else:
            data = fallback_parse(text)
    if not isinstance(data, dict):
        raise ValueError("Model output was not a JSON object.")
    return data


def fallback_parse(text: str) -> dict[str, str]:
    label_match = re.search(
        r"\b(metaphor|simile|idiom|sarcasm|non[-_\s]?figurative)\b",
        text,
        flags=re.IGNORECASE,
    )
    explanation_match = re.search(
        r"(?:brief[_\s-]*explanation|explanation)\s*[:=-]\s*['\"]?(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    explanation = explanation_match.group(1).strip() if explanation_match else text.strip()
    explanation = explanation.strip("`'\" \n")
    return {
        "label": normalize_label(label_match.group(1)) if label_match else "",
        "brief_explanation": explanation,
        "parse_warning": "fallback_parse_used",
    }


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    aliases = {
        "metaphors": "metaphor",
        "similes": "simile",
        "idioms": "idiom",
        "sarcastic": "sarcasm",
        "non-figurative": "non_figurative",
        "non figurative": "non_figurative",
        "nonfigurative": "non_figurative",
        "literal": "non_figurative",
    }
    return aliases.get(text, text)


def dry_run_prediction(example: dict[str, Any]) -> dict[str, str]:
    text = f"{example.get('sentence', '')} {example.get('premise', '')} {example.get('hypothesis', '')}".lower()
    if " like " in f" {text} " or " as " in f" {text} ":
        label = "simile"
        explanation = "The example uses an explicit comparison cue such as like or as."
    elif any(phrase in text for phrase in ("on cloud nine", "spill the beans", "piece of cake")):
        label = "idiom"
        explanation = "The example contains a fixed expression with a nonliteral meaning."
    elif any(word in text for word in ("great,", "wonderful,", "thanks a lot", "just perfect")):
        label = "sarcasm"
        explanation = "The positive wording is likely used ironically rather than literally."
    else:
        label = "metaphor"
        explanation = "The example directly maps one concept onto another without like or as."
    return {"label": label, "brief_explanation": explanation}


def make_prompt(example: dict[str, Any], setting: str) -> str:
    if setting == "sentence_only":
        if example.get("prompt_sentence_only"):
            return str(example["prompt_sentence_only"])
        if example.get("prompt"):
            return str(example["prompt"])

    if setting == "with_nli":
        if example.get("prompt_with_nli"):
            return str(example["prompt_with_nli"])
        if example.get("prompt"):
            return str(example["prompt"])
    if setting == "without_nli" and example.get("prompt_without_nli"):
        return str(example["prompt_without_nli"])

    premise = str(example.get("premise", "")).strip()
    hypothesis = str(example.get("hypothesis", "")).strip()
    nli_label = str(example.get("nli_label", "")).strip()
    nli_line = f"NLI label: {nli_label}\n" if setting == "with_nli" else ""
    return (
        "Identify the type of language involved.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"{nli_line}\n"
        "Question: What type of figurative language is used in this example?\n"
        "Choose one of the following labels: metaphor, simile, idiom, sarcasm.\n\n"
        "Return JSON with keys label and brief_explanation."
    )


def call_openai(model: str, prompt: str, reasoning_effort: str) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: python -m pip install -r requirements.txt") from exc

    client = OpenAI()
    request = {
        "model": model,
        "input": [
            {"role": "developer", "content": DEVELOPER_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    }
    if model.startswith(("o", "gpt-5")):
        request["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**request)
    output_text = response.output_text
    parsed = extract_json_object(output_text)
    parsed["raw_output"] = output_text
    return parsed


def build_result(
    example: dict[str, Any],
    model: str,
    setting: str,
    parsed: dict[str, Any],
    latency: float,
) -> dict[str, Any]:
    predicted = normalize_label(parsed.get("label", ""))
    gold = normalize_label(example.get("gold_label", ""))
    return {
        "id": example.get("id", ""),
        "model": model,
        "setting": setting,
        "gold_label": gold,
        "predicted_label": predicted,
        "correct": predicted == gold,
        "brief_explanation": str(parsed.get("brief_explanation", "")).strip(),
        "gold_explanation": str(example.get("gold_explanation", "")).strip(),
        "raw_output": parsed.get("raw_output", parsed),
        "latency_seconds": round(latency, 3),
        "premise": example.get("premise", ""),
        "hypothesis": example.get("hypothesis", ""),
        "nli_label": example.get("nli_label", ""),
        "sentence": example.get("sentence", ""),
    }


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for record in load_jsonl(path):
        keys.add(
            (
                str(record.get("id", "")),
                str(record.get("model", "")),
                str(record.get("setting", "with_nli")),
            )
        )
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Classification JSONL file.")
    parser.add_argument("--output", required=True, help="Prediction JSONL file.")
    parser.add_argument("--models", nargs="+", default=["gpt-5.5", "gpt-4.1-nano"])
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=("with_nli", "without_nli", "sentence_only"),
        default=["with_nli"],
        help="Run one or both prompt settings.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reasoning-effort", default="low", choices=("minimal", "low", "medium", "high"))
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--resume", action="store_true", help="Skip completed id/model pairs in output.")
    parser.add_argument("--dry-run", action="store_true", help="Use a deterministic local baseline instead of API calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY first, or pass --dry-run to test the pipeline without API calls.")

    examples = load_jsonl(Path(args.input))
    if args.limit:
        examples = examples[: args.limit]

    output_path = Path(args.output)
    completed = existing_keys(output_path) if args.resume else set()

    for model in args.models:
        for setting in args.settings:
            for example in examples:
                key = (str(example.get("id", "")), model, setting)
                if key in completed:
                    continue
                started = time.time()
                if args.dry_run:
                    parsed = dry_run_prediction(example)
                else:
                    parsed = call_openai(model, make_prompt(example, setting), args.reasoning_effort)
                result = build_result(example, model, setting, parsed, time.time() - started)
                append_jsonl(result, output_path)
                print(
                    f"{model} {setting} {result['id']}: "
                    f"{result['predicted_label']} correct={result['correct']}"
                )
                if args.sleep:
                    time.sleep(args.sleep)


if __name__ == "__main__":
    main()
