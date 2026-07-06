# Evaluation Summary

## gpt-4.1-nano | with_nli

- N: 400
- Overall accuracy: 0.787
- Macro-F1: 0.789
- Explanation scoring: judge
- Mean explanation score: 81.125 / 100
- Acc@50: 0.785
- Acc@60: 0.688

| Label | Support | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| metaphor | 100 | 0.570 | 0.722 | 0.588 | 0.648 |
| simile | 100 | 0.840 | 0.884 | 0.840 | 0.862 |
| idiom | 100 | 0.810 | 0.723 | 0.810 | 0.764 |
| sarcasm | 100 | 0.930 | 0.838 | 0.930 | 0.882 |

## gpt-4.1-nano | without_nli

- N: 400
- Overall accuracy: 0.802
- Macro-F1: 0.800
- Explanation scoring: judge
- Mean explanation score: 82.375 / 100
- Acc@50: 0.802
- Acc@60: 0.715

| Label | Support | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| metaphor | 100 | 0.640 | 0.753 | 0.640 | 0.692 |
| simile | 100 | 0.850 | 0.895 | 0.850 | 0.872 |
| idiom | 100 | 0.810 | 0.750 | 0.810 | 0.779 |
| sarcasm | 100 | 0.910 | 0.812 | 0.910 | 0.858 |

## gpt-5.5 | with_nli

- N: 400
- Overall accuracy: 0.885
- Macro-F1: 0.886
- Explanation scoring: judge
- Mean explanation score: 92.125 / 100
- Acc@50: 0.885
- Acc@60: 0.877

| Label | Support | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| metaphor | 100 | 0.780 | 0.940 | 0.780 | 0.852 |
| simile | 100 | 0.960 | 0.923 | 0.960 | 0.941 |
| idiom | 100 | 0.920 | 0.760 | 0.920 | 0.833 |
| sarcasm | 100 | 0.880 | 0.957 | 0.880 | 0.917 |

## gpt-5.5 | without_nli

- N: 400
- Overall accuracy: 0.885
- Macro-F1: 0.884
- Explanation scoring: judge
- Mean explanation score: 92.375 / 100
- Acc@50: 0.885
- Acc@60: 0.875

| Label | Support | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| metaphor | 100 | 0.750 | 0.962 | 0.750 | 0.843 |
| simile | 100 | 0.980 | 0.925 | 0.980 | 0.951 |
| idiom | 100 | 0.910 | 0.784 | 0.910 | 0.843 |
| sarcasm | 100 | 0.900 | 0.900 | 0.900 | 0.900 |
