
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_ORDER = ("metaphor", "simile", "idiom", "sarcasm", "non_figurative")
THRESHOLDS = (50, 60)

JUDGE_INSTRUCTIONS = """You are grading explanations for a language type classification task.
Score whether the model explanation correctly justifies the gold label.

Use this rubric:
0 = incorrect, irrelevant, or missing explanation
1 = partially correct but vague, incomplete, or tied to the wrong cue
2 = correct explanation with a clear linguistic cue for the gold label

If the predicted label is different from the gold label, the score should usually be 0 or 1.
Return only valid JSON with double-quoted keys and strings: {"score": 0, "reason": "..."}"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            return {"score": 0, "reason": text.strip(), "parse_warning": "fallback_parse_used"}
        snippet = match.group(0)
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(snippet)
            except (SyntaxError, ValueError):
                repaired = re.sub(r"([{,]\s*)(score|reason)\s*:", r'\1"\2":', snippet)
                repaired = repaired.replace("'", '"')
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError:
                    return {"score": 0, "reason": text.strip(), "parse_warning": "fallback_parse_used"}
    if not isinstance(data, dict):
        raise ValueError("Judge output was not a JSON object.")
    return data


def judge_prompt(row: dict[str, Any]) -> str:
    return (
        f"Gold label: {row.get('gold_label', '')}\n"
        f"Predicted label: {row.get('predicted_label', '')}\n"
        f"Setting: {row.get('setting', 'with_nli')}\n"
        f"Sentence: {row.get('sentence', '')}\n"
        f"Premise: {row.get('premise', '')}\n"
        f"Hypothesis: {row.get('hypothesis', '')}\n"
        f"NLI label: {row.get('nli_label', '')}\n"
        f"Gold explanation: {row.get('gold_explanation', '')}\n"
        f"Model explanation: {row.get('brief_explanation', '')}\n\n"
        "Score the model explanation using the rubric."
    )


def compute_bertscore(records: list[dict[str, Any]]) -> list[float | None]:
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore is required for FLUTE-style explanation scoring. "
            "Install it with: python -m pip install bert-score"
        ) from exc

    valid_indices = []
    candidates = []
    references = []
    for index, row in enumerate(records):
        candidate = str(row.get("brief_explanation", "")).strip()
        reference = str(row.get("gold_explanation", "")).strip()
        if candidate and reference:
            valid_indices.append(index)
            candidates.append(candidate)
            references.append(reference)

    scores: list[float | None] = [None] * len(records)
    if not candidates:
        return scores

    _, _, f1_scores = bert_score(candidates, references, lang="en", verbose=False)
    for index, value in zip(valid_indices, f1_scores.tolist()):
        scores[index] = max(0.0, min(100.0, float(value) * 100.0))
    return scores


def compute_bleurt(records: list[dict[str, Any]], checkpoint: str) -> list[float | None]:
    try:
        from bleurt import score as bleurt_score
    except ImportError as exc:
        raise RuntimeError(
            "BLEURT is required for --explanation-scoring flute. "
            "Install BLEURT and provide a local checkpoint with --bleurt-checkpoint."
        ) from exc

    valid_indices = []
    candidates = []
    references = []
    for index, row in enumerate(records):
        candidate = str(row.get("brief_explanation", "")).strip()
        reference = str(row.get("gold_explanation", "")).strip()
        if candidate and reference:
            valid_indices.append(index)
            candidates.append(candidate)
            references.append(reference)

    scores: list[float | None] = [None] * len(records)
    if not candidates:
        return scores

    scorer = bleurt_score.BleurtScorer(checkpoint)
    raw_scores = scorer.score(references=references, candidates=candidates)
    for index, value in zip(valid_indices, raw_scores):
        # BLEURT is not strictly bounded. This normalization keeps the
        # average on a readable 0-100 scale for thresholding.
        normalized = (float(value) + 1.0) / 2.0 * 100.0
        scores[index] = max(0.0, min(100.0, normalized))
    return scores


def compute_judge_scores(
    records: list[dict[str, Any]],
    judge_model: str,
    reasoning_effort: str,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before using --explanation-scoring judge.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: python -m pip install -r requirements.txt") from exc

    client = OpenAI()
    enriched = []
    for row in records:
        request = {
            "model": judge_model,
            "input": [
                {"role": "developer", "content": JUDGE_INSTRUCTIONS},
                {"role": "user", "content": judge_prompt(row)},
            ],
        }
        if judge_model.startswith(("o", "gpt-5")):
            request["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(**request)
        parsed = extract_json_object(response.output_text)
        score = int(parsed.get("score", 0))
        score = max(0, min(2, score))

        updated = dict(row)
        updated["judge_model"] = judge_model
        updated["judge_score_0_to_2"] = score
        updated["judge_reason"] = str(parsed.get("reason", "")).strip()
        updated["judge_raw_output"] = response.output_text
        updated["explanation_score"] = score * 50.0
        enriched.append(updated)

        if sleep_seconds:
            time.sleep(sleep_seconds)
    return enriched


def attach_explanation_scores(
    records: list[dict[str, Any]],
    scoring: str,
    bleurt_checkpoint: str | None,
    judge_model: str,
    judge_reasoning_effort: str,
    judge_sleep: float,
) -> list[dict[str, Any]]:
    if scoring == "none":
        return records
    if scoring == "judge":
        return compute_judge_scores(records, judge_model, judge_reasoning_effort, judge_sleep)

    bert_scores = compute_bertscore(records)
    bleurt_scores: list[float | None] = [None] * len(records)
    if scoring == "flute":
        if not bleurt_checkpoint:
            raise RuntimeError("--bleurt-checkpoint is required when --explanation-scoring flute is used.")
        bleurt_scores = compute_bleurt(records, bleurt_checkpoint)

    enriched = []
    for row, bert_value, bleurt_value in zip(records, bert_scores, bleurt_scores):
        updated = dict(row)
        updated["bertscore_f1"] = bert_value
        updated["bleurt_score"] = bleurt_value
        score_parts = [value for value in (bert_value, bleurt_value) if value is not None]
        updated["explanation_score"] = sum(score_parts) / len(score_parts) if score_parts else None
        enriched.append(updated)
    return enriched


def labels_for(records: list[dict[str, Any]]) -> list[str]:
    labels = {str(row.get("gold_label", "")) for row in records}
    labels.update(str(row.get("predicted_label", "")) for row in records)
    return [label for label in LABEL_ORDER if label in labels]


def evaluate_model(records: list[dict[str, Any]], explanation_scoring: str) -> dict[str, Any]:
    labels = labels_for(records)
    total = len(records)
    correct = sum(1 for row in records if row.get("gold_label") == row.get("predicted_label"))
    confusion = {gold: {pred: 0 for pred in labels} for gold in labels}
    predicted_counts = Counter()
    gold_counts = Counter()
    explanation_scores = []
    threshold_correct = {threshold: 0 for threshold in THRESHOLDS}

    for row in records:
        gold = row.get("gold_label")
        pred = row.get("predicted_label")
        is_correct = gold == pred
        if gold in labels:
            gold_counts[gold] += 1
            if pred in labels:
                confusion[gold][pred] += 1
        if pred in labels:
            predicted_counts[pred] += 1
        if row.get("explanation_score") is not None:
            explanation_score = float(row["explanation_score"])
            explanation_scores.append(explanation_score)
            for threshold in THRESHOLDS:
                if is_correct and explanation_score >= threshold:
                    threshold_correct[threshold] += 1

    per_label = {}
    f1_values = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in labels if gold != label)
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_label[label] = {
            "support": gold_counts[label],
            "predicted": predicted_counts[label],
            "accuracy": safe_div(tp, gold_counts[label]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    has_explanation_scores = bool(explanation_scores)
    return {
        "n": total,
        "overall_accuracy": safe_div(correct, total),
        "macro_f1": sum(f1_values) / len(f1_values),
        "explanation_scoring": explanation_scoring,
        "mean_explanation_score": safe_div(sum(explanation_scores), len(explanation_scores))
        if has_explanation_scores
        else None,
        "acc_at_50": safe_div(threshold_correct[50], total) if has_explanation_scores else None,
        "acc_at_60": safe_div(threshold_correct[60], total) if has_explanation_scores else None,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "labels": labels,
    }


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def write_confusion_csv(group_name: str, confusion: dict[str, dict[str, int]], output_dir: Path) -> None:
    labels = list(confusion.keys())
    path = output_dir / f"confusion_matrix_{safe_filename(group_name)}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\predicted", *labels])
        for gold in labels:
            writer.writerow([gold, *[confusion[gold][pred] for pred in labels]])


def write_scored_predictions(records: list[dict[str, Any]], output_dir: Path) -> None:
    path = output_dir / "predictions_with_explanation_scores.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown(metrics: dict[str, Any], output_dir: Path) -> None:
    lines = ["# Evaluation Summary", ""]
    for model, model_metrics in metrics.items():
        lines.extend(
            [
                f"## {model}",
                "",
                f"- N: {model_metrics['n']}",
                f"- Overall accuracy: {model_metrics['overall_accuracy']:.3f}",
                f"- Macro-F1: {model_metrics['macro_f1']:.3f}",
                f"- Explanation scoring: {model_metrics['explanation_scoring']}",
            ]
        )
        if model_metrics["mean_explanation_score"] is not None:
            lines.extend(
                [
                    f"- Mean explanation score: {model_metrics['mean_explanation_score']:.3f} / 100",
                    f"- Acc@50: {model_metrics['acc_at_50']:.3f}",
                    f"- Acc@60: {model_metrics['acc_at_60']:.3f}",
                ]
            )
        else:
            lines.append("- Explanation score: not computed")

        lines.extend(
            [
                "",
                "| Label | Support | Accuracy | Precision | Recall | F1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label in model_metrics["labels"]:
            row = model_metrics["per_label"][label]
            lines.append(
                f"| {label} | {row['support']} | {row['accuracy']:.3f} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
            )
        lines.append("")
    (output_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics files.")
    parser.add_argument(
        "--explanation-scoring",
        choices=("none", "bertscore", "flute", "judge"),
        default="bertscore",
        help=(
            "none = classification metrics only; bertscore = BERTScore F1 on a 0-100 scale; "
            "flute = average of BERTScore and BLEURT, closest to the FLUTE paper; "
            "judge = GPT judge score mapped from 0/1/2 to 0/50/100."
        ),
    )
    parser.add_argument("--bleurt-checkpoint", help="Local BLEURT checkpoint path for --explanation-scoring flute.")
    parser.add_argument("--judge-model", default="gpt-4.1-nano", help="Model used for --explanation-scoring judge.")
    parser.add_argument(
        "--judge-reasoning-effort",
        default="low",
        choices=("minimal", "low", "medium", "high"),
        help="Reasoning effort for the judge model.",
    )
    parser.add_argument("--judge-sleep", type=float, default=0.0, help="Seconds to sleep between judge API calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.predictions))
    records = attach_explanation_scores(
        records,
        args.explanation_scoring,
        args.bleurt_checkpoint,
        args.judge_model,
        args.judge_reasoning_effort,
        args.judge_sleep,
    )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_name = f"{record.get('model', 'unknown')} | {record.get('setting', 'with_nli')}"
        by_model[str(group_name)].append(record)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {model: evaluate_model(rows, args.explanation_scoring) for model, rows in sorted(by_model.items())}
    for model, model_metrics in metrics.items():
        write_confusion_csv(model, model_metrics["confusion_matrix"], output_dir)
    write_scored_predictions(records, output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(metrics, output_dir)
    print(f"Wrote metrics for {len(metrics)} model(s) to {output_dir}")


if __name__ == "__main__":
    main()
