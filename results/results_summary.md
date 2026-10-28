# Results summary (test set, 2627 messages: 1646 safe, 981 phishing)

## Default threshold 0.5

| Model | Thr | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC | FP | FN | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | 0.50 | 98.59% | 98.56% | 97.66% | 0.9811 | 0.9983 | 0.9972 | 0.9699 | 14 | 23 | 0.85% | 2.34% |
| TextCNN (deep learning) | 0.50 | 98.40% | 98.65% | 97.04% | 0.9784 | 0.9980 | 0.9964 | 0.9658 | 13 | 29 | 0.79% | 2.96% |
| BiLSTM + Attention (deep learning) | 0.50 | 97.79% | 96.01% | 98.17% | 0.9708 | 0.9974 | 0.9955 | 0.9532 | 40 | 18 | 2.43% | 1.83% |
| Ensemble (average of all) | 0.50 | 98.71% | 98.17% | 98.37% | 0.9827 | 0.9985 | 0.9975 | 0.9724 | 18 | 16 | 1.09% | 1.63% |


## Cost-tuned threshold (chosen on validation, FN cost = 3 x FP cost)

| Model | Thr | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC | FP | FN | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | 0.32 | 98.29% | 96.71% | 98.78% | 0.9773 | 0.9983 | 0.9972 | 0.9637 | 33 | 12 | 2.00% | 1.22% |
| TextCNN (deep learning) | 0.12 | 98.10% | 96.41% | 98.57% | 0.9748 | 0.9980 | 0.9964 | 0.9597 | 36 | 14 | 2.19% | 1.43% |
| BiLSTM + Attention (deep learning) | 0.30 | 97.49% | 94.90% | 98.57% | 0.9670 | 0.9974 | 0.9955 | 0.9472 | 52 | 14 | 3.16% | 1.43% |
| Ensemble (average of all) | 0.21 | 97.68% | 94.75% | 99.29% | 0.9696 | 0.9985 | 0.9975 | 0.9515 | 54 | 7 | 3.28% | 0.71% |


## Error profile (threshold 0.5)

**TF-IDF + LogReg (baseline)**

| Group | Count | Median tokens | Contains URL | Mean confidence |
|---|---|---|---|---|
| false positives | 14 | 134 | 14.3% | 0.41 |
| false negatives | 23 | 155 | 13.0% | 0.36 |
| correct safe | 1632 | 175 | 21.0% | 0.92 |
| correct phishing | 958 | 120 | 6.1% | 0.85 |


**TextCNN (deep learning)**

| Group | Count | Median tokens | Contains URL | Mean confidence |
|---|---|---|---|---|
| false positives | 13 | 203 | 7.7% | 0.66 |
| false negatives | 29 | 97 | 6.9% | 0.61 |
| correct safe | 1633 | 174 | 21.1% | 0.99 |
| correct phishing | 952 | 120 | 6.2% | 0.97 |


**BiLSTM + Attention (deep learning)**

| Group | Count | Median tokens | Contains URL | Mean confidence |
|---|---|---|---|---|
| false positives | 40 | 188 | 22.5% | 0.68 |
| false negatives | 18 | 100 | 11.1% | 0.62 |
| correct safe | 1606 | 174 | 20.9% | 0.98 |
| correct phishing | 963 | 120 | 6.1% | 0.98 |


## Error rate by message length (threshold 0.5)

| Length (tokens) | baseline | bilstm | cnn |
|---|---|---|---|
| <30 | 3.38% | 4.83% | 3.38% |
| 30-100 | 1.33% | 2.00% | 1.47% |
| 100-300 | 1.12% | 1.83% | 1.22% |
| 300-1000 | 1.65% | 2.01% | 1.83% |
| >1000 | 0.00% | 2.90% | 1.45% |



Messages per bucket: <30: 207, 30-100: 750, 100-300: 985, 300-1000: 547, >1000: 138


## Error overlap

17 test messages are misclassified by **every** model (errors per model: baseline 37, cnn 42, bilstm 58).


**BiLSTM + Attention (deep learning) - most-attended words in false positives:** <unk> (26), numtoken (13), your (11), this (10), urltoken (8), please (6), http (6), at (5), you (5), ! (4), click (3), business (3), reply (3), am (3), news (3)


**BiLSTM + Attention (deep learning) - most-attended words in false negatives:** <unk> (7), numtoken (6), this (4), please (3), urltoken (3), at (2), your (2), news (2), ! (2), once (2), you (2), http (2), so (1), gave (1), voting (1)
