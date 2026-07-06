# FLUTE Figurative Language Type Classification

This project implements a term-paper experiment that converts FLUTE-style figurative NLI data into a figurative language type classification task. The goal is to test how well GPT-based models recognize different types of figurative language, and whether their performance changes when the original NLI context is removed.

The project contains two experiments:

1. **Experiment 1: FLUTE-context classification**
   The model sees the original FLUTE premise and hypothesis, with or without the NLI label.

2. **Experiment 2: Sentence-only classification**
   The model sees only one target sentence and must classify it into one of five language types, including a new `non_figurative` class.

The second experiment follows the supervisor's feedback more closely because it removes explicit FLUTE/NLI framing and evaluates a cleaner figurative language classification setting.

## Project Structure

```text
src/normalize_flute.py              Convert FLUTE into the original 4-way classification dataset
src/build_sentence_only_dataset.py  Build the revised sentence-only 5-way dataset
src/run_openai_eval.py              Query GPT models and save predictions
src/evaluate_results.py             Compute metrics and explanation scores
data/                               Processed datasets
results/                            Prediction files and evaluation outputs
```

## Dataset Construction

### Experiment 1 Dataset: FLUTE-Context Setting

The original dataset is built directly from Hugging Face FLUTE:

```powershell
python src\normalize_flute.py `
  --input ColumbiaNLP/FLUTE `
  --split train `
  --sample-per-class 100 `
  --output data\flute_100_per_class.jsonl
```

This creates a balanced 400-example dataset:

```text
metaphor 100
simile   100
idiom    100
sarcasm  100
```

Each example contains the original FLUTE premise, hypothesis, NLI label, figurative type label, and gold explanation.

### Experiment 2 Dataset: Sentence-Only Setting

The revised dataset is built with:

```powershell
python src\build_sentence_only_dataset.py `
  --output data\flute_sentence_only_100_per_class.jsonl
```

This creates a balanced 500-example dataset:

```text
metaphor        100
simile          100
idiom           100
sarcasm         100
non_figurative  100
```

The sentence-only setting uses one target sentence per example. The `non_figurative` class is sampled from FLUTE `CreativeParaphrase` examples. This keeps the data source consistent while adding a contrast class for literal/non-figurative language.

## Experimental Settings

Both experiments are framed as classification questions. The models are asked to return a JSON object with two fields:

```json
{
  "label": "one selected class label",
  "brief_explanation": "a short reason for the classification"
}
```

The answer is counted as correct only when the predicted `label` exactly matches the gold figurative language type. The `brief_explanation` is evaluated separately by a GPT judge, so the project measures both classification accuracy and explanation quality.

The question format differs across the two experiments:

```text
Experiment 1 question format:
Premise: ...
Hypothesis: ...
NLI label: ...        # only in the with_nli setting
Choose one label: metaphor, simile, idiom, sarcasm.

Experiment 2 question format:
Sentence: ...
Choose one label: metaphor, simile, idiom, sarcasm, non_figurative.
```

Experiment 1 therefore tests classification under the original FLUTE NLI-style context, while Experiment 2 tests a stricter sentence-only setting.

### Experiment 1: Original FLUTE-Context Experiment

This experiment tests whether the model benefits from the original FLUTE NLI structure.

```text
with_nli     = premise + hypothesis + NLI label
without_nli  = premise + hypothesis only
```

Run predictions:

```powershell
python src\run_openai_eval.py `
  --input data\flute_100_per_class.jsonl `
  --output results\gpt_predictions.jsonl `
  --models gpt-5.5 gpt-4.1-nano `
  --settings with_nli without_nli `
  --resume
```

Evaluate:

```powershell
python src\evaluate_results.py `
  --predictions results\gpt_predictions.jsonl `
  --output-dir results\gpt_metrics `
  --explanation-scoring judge `
  --judge-model gpt-5.5
```

### Experiment 2: Revised Sentence-Only Experiment

This experiment tests whether models can classify figurative language type from a single sentence, without FLUTE name, NLI label, premise, or hypothesis pair structure.

Run predictions:

```powershell
python src\run_openai_eval.py `
  --input data\flute_sentence_only_100_per_class.jsonl `
  --output results\sentence_only_predictions.jsonl `
  --models gpt-5.5 gpt-4.1-nano `
  --settings sentence_only `
  --resume
```

Evaluate:

```powershell
python src\evaluate_results.py `
  --predictions results\sentence_only_predictions.jsonl `
  --output-dir results\sentence_only_metrics `
  --explanation-scoring judge `
  --judge-model gpt-5.5
```

## Evaluation Metrics

The evaluation reports:

```text
Accuracy
Macro-F1
Per-class accuracy, precision, recall, and F1
Confusion matrix
Mean explanation score
Acc@50
Acc@60
```

Explanation quality is evaluated with a GPT judge instead of BLEURT, BERTScore, or human annotation. For each model prediction, the judge receives the original test item, the gold label, the model's predicted label, and the model's `brief_explanation`. It then decides whether the explanation gives a valid reason for the classification.

The judge uses a three-level rubric:

```text
0 = incorrect, irrelevant, or missing explanation
1 = partially correct explanation, but vague, incomplete, or only weakly tied to the label
2 = correct explanation that identifies a clear figurative-language cue and supports the label
```

The raw judge score is therefore `0`, `1`, or `2`. The script maps it to a 0-100 explanation score:

```text
judge score 0 -> explanation_score 0
judge score 1 -> explanation_score 50
judge score 2 -> explanation_score 100
```

The reported **Mean explanation score** is the average of these mapped 0-100 scores over all examples in the same model and setting. For example, a mean explanation score of `84.600` means that, on average, the judge considered the model's explanations to be mostly correct, with some partial or incorrect explanations.

This score evaluates the explanation text, not just the label. A prediction can have a correct label but a weak explanation, or an incorrect label with an explanation that only partially identifies a relevant cue. This is why explanation score is reported separately from classification accuracy.

`Acc@50` and `Acc@60` combine label correctness with explanation quality:

```text
Acc@50 = label is correct and explanation_score >= 50
Acc@60 = label is correct and explanation_score >= 60
```

In judge mode, `Acc@50` means the label is correct and the explanation receives at least partial credit. `Acc@60` means the label is correct and the explanation receives full credit.

## Results

### Experiment 1: FLUTE-Context Results

| Model | Setting | Accuracy | Macro-F1 | Explanation Score | Acc@50 | Acc@60 |
|---|---|---:|---:|---:|---:|---:|
| gpt-5.5 | with_nli | 0.885 | 0.886 | 92.125 | 0.885 | 0.877 |
| gpt-5.5 | without_nli | 0.885 | 0.884 | 92.375 | 0.885 | 0.875 |
| gpt-4.1-nano | with_nli | 0.787 | 0.789 | 81.125 | 0.785 | 0.688 |
| gpt-4.1-nano | without_nli | 0.802 | 0.800 | 82.375 | 0.802 | 0.715 |

In the FLUTE-context experiment, `gpt-5.5` performs strongly in both settings, reaching 0.885 accuracy with and without the NLI label. This suggests that the explicit NLI label is not necessary for the stronger model. Its explanation scores are also high, above 92/100 in both settings.

`gpt-4.1-nano` performs lower than `gpt-5.5`, but still reaches around 0.79-0.80 accuracy. Interestingly, removing the NLI label does not hurt the smaller model; its accuracy slightly increases from 0.787 to 0.802. This indicates that the premise-hypothesis pair itself already provides enough contextual information, and the NLI label may not be the main shortcut.

At the category level, both models perform best on relatively explicit categories such as simile and sarcasm. Metaphor is consistently harder, especially for the smaller model. This is expected because metaphors often require more semantic abstraction and can overlap with idiomatic meaning.

### Experiment 2: Sentence-Only Results

| Model | Setting | Accuracy | Macro-F1 | Explanation Score | Acc@50 | Acc@60 |
|---|---|---:|---:|---:|---:|---:|
| gpt-5.5 | sentence_only | 0.816 | 0.817 | 84.600 | 0.816 | 0.814 |
| gpt-4.1-nano | sentence_only | 0.510 | 0.495 | 53.100 | 0.510 | 0.440 |

The sentence-only experiment is more difficult than the FLUTE-context experiment. For `gpt-5.5`, accuracy drops from 0.885 in the original context setting to 0.816 in the revised sentence-only setting. This decrease is meaningful but not severe, showing that the stronger model can still identify figurative language types from the sentence itself.

For `gpt-4.1-nano`, performance drops sharply from about 0.80 in the FLUTE-context setting to 0.510 in the sentence-only setting. This suggests that the smaller model relied much more on the additional context provided by the original FLUTE pair structure. When only the target sentence is given, it struggles to separate figurative categories from literal language.

Per-class results show a clear difference between the two models:

| Model | Metaphor Acc. | Simile Acc. | Idiom Acc. | Sarcasm Acc. | Non-Fig. Acc. |
|---|---:|---:|---:|---:|---:|
| gpt-5.5 | 0.630 | 0.990 | 0.890 | 0.860 | 0.710 |
| gpt-4.1-nano | 0.380 | 0.560 | 0.590 | 0.100 | 0.920 |

`gpt-5.5` is very strong on simile, idiom, and sarcasm. Simile is the easiest class, likely because it often contains explicit comparison markers such as "like" or "as". Idiom is also relatively strong because idiomatic expressions often contain recognizable fixed phrases. Sarcasm remains strong for `gpt-5.5`, but the model still misses some cases when no explicit context is available.

Metaphor is the hardest figurative category for `gpt-5.5`, with 0.630 accuracy. Many metaphor errors are confused with idiom or non-figurative language, which suggests that metaphor recognition often depends on subtle semantic incongruity rather than surface-level cues.

`gpt-4.1-nano` shows a very different pattern. Its non-figurative accuracy is high at 0.920, but its precision for non-figurative is low. This means the model overpredicts `non_figurative`. The confusion matrix shows that many metaphors and sarcasm examples are incorrectly classified as non-figurative. In particular, sarcasm accuracy falls to 0.100, which indicates that the smaller model has difficulty detecting ironic intent from a single sentence.

## Comparison Between the Two Experiments

The two experiments together show that the task formulation strongly affects model performance.

In Experiment 1, the original FLUTE context provides rich information through the premise-hypothesis pair. Both models perform relatively well, and the presence or absence of the NLI label makes little difference. This suggests that the extra sentence context, rather than the NLI label alone, may make the task easier.

In Experiment 2, the sentence-only setup removes that context and introduces a non-figurative contrast class. This produces a stricter and more realistic classification task. The stronger model remains fairly robust, while the smaller model drops close to a weak baseline level. This contrast supports the conclusion that larger models are better able to use sentence-internal linguistic cues, while smaller models depend more heavily on contextual framing.

The revised experiment also better matches the research question. Instead of asking whether a model can infer a FLUTE category from an NLI-style example, it asks whether the model can recognize the type of figurative language in a standalone sentence.

## Main Findings

1. `gpt-5.5` consistently outperforms `gpt-4.1-nano` across both experiments.

2. The NLI label itself does not substantially improve performance in the original FLUTE-context experiment.

3. Removing the premise-hypothesis context makes the task harder, especially for the smaller model.

4. Simile is the easiest category, likely because it often has explicit lexical markers.

5. Metaphor is one of the hardest categories because it requires more abstract semantic interpretation.

6. Sarcasm is highly model-dependent: `gpt-5.5` handles it well, while `gpt-4.1-nano` fails badly in the sentence-only setting.

7. Adding `non_figurative` makes the task more realistic, but also reveals that weaker models may overpredict literal language when figurative cues are subtle.

## Limitations

The `non_figurative` class is sampled from FLUTE `CreativeParaphrase` examples, so it is still drawn from the same dataset distribution rather than from an entirely separate general corpus. This is useful for consistency, but future work could test non-figurative examples from broader corpora.

The explanation score is based on a GPT judge, so it is an automatic approximation rather than a fully human evaluation. To strengthen the paper, a small manual analysis of representative correct and incorrect explanations should be included.

The current comparison uses two GPT models. `gpt-4.1-nano` functions as a weaker model baseline, but a future version could add a non-LLM baseline, such as a keyword rule system or TF-IDF classifier.

## Output Files

The main generated files are:

```text
data/flute_100_per_class.jsonl
data/flute_sentence_only_100_per_class.jsonl
results/gpt_predictions.jsonl
results/gpt_metrics/metrics.md
results/gpt_metrics/metrics.json
results/sentence_only_predictions.jsonl
results/sentence_only_metrics/metrics.md
results/sentence_only_metrics/metrics.json
```

`metrics.md` is the easiest file to read for the paper. `metrics.json` is useful if tables or plots need to be generated later.
