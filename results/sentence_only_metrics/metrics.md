# Evaluation Summary

## gpt-4.1-nano | sentence_only

- N: 500
- Overall accuracy: 0.510
- Macro-F1: 0.495
- Explanation scoring: judge
- Mean explanation score: 53.100 / 100
- Acc@50: 0.510
- Acc@60: 0.440

| Label | Support | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| metaphor | 100 | 0.380 | 0.376 | 0.380 | 0.378 |
| simile | 100 | 0.560 | 0.918 | 0.560 | 0.696 |
| idiom | 100 | 0.590 | 0.894 | 0.590 | 0.711 |
| sarcasm | 100 | 0.100 | 0.909 | 0.100 | 0.180 |
| non_figurative | 100 | 0.920 | 0.352 | 0.920 | 0.510 |

## gpt-5.5 | sentence_only

- N: 500
- Overall accuracy: 0.816
- Macro-F1: 0.817
- Explanation scoring: judge
- Mean explanation score: 84.600 / 100
- Acc@50: 0.816
- Acc@60: 0.814

| Label | Support | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| metaphor | 100 | 0.630 | 0.887 | 0.630 | 0.737 |
| simile | 100 | 0.990 | 0.934 | 0.990 | 0.961 |
| idiom | 100 | 0.890 | 0.685 | 0.890 | 0.774 |
| sarcasm | 100 | 0.860 | 1.000 | 0.860 | 0.925 |
| non_figurative | 100 | 0.710 | 0.664 | 0.710 | 0.686 |
