# Fuzzy Sentiment Analysis using VADER + scikit-fuzzy

A small Python project that performs sentiment analysis on tweets by combining VADER sentiment scores with a fuzzy logic decision layer (using `scikit-fuzzy`). The pipeline computes VADER polarity (pos/neg/neu), fuzzifies them, applies fuzzy rules, defuzzifies to an output score, and maps that score to `Negative` / `Neutral` / `Positive` classes.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset format](#dataset-format)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How it works (brief)](#how-it-works-brief)
- [Fuzzy system details](#fuzzy-system-details)
- [Output & evaluation](#output--evaluation)
- [Visualization](#visualization)
- [Troubleshooting & tips](#troubleshooting--tips)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project demonstrates how to combine a rule-based sentiment lexicon (VADER) with a fuzzy decision layer to (attempt to) improve robustness to boundary/uncertain cases. The notebook/script reads a CSV file with tweets, computes VADER scores, fuzzifies scores into linguistic variables (Low/Medium/High for `pos` and `neg`), applies fuzzy rules to get an aggregated output membership, and finally defuzzifies to a numeric output which is mapped to a sentiment label.

## Features

- Preprocessing (contractions, URL/handle removal, lowercasing)
- VADER polarity scoring
- Fuzzy membership functions for `pos`, `neg`, and output (`Neg`/`Neu`/`Pos`)
- Fuzzy rule evaluation and centroid defuzzification
- Output metrics: accuracy, precision, recall, F1 (macro/micro)
- Plotting of membership functions and aggregated outputs

## Dataset format

The script expects a CSV file with at least the following columns:

- `TweetText` — the raw tweet text (string)
- `Sentiment` — ground-truth label for evaluation (`Negative`, `Neutral`, `Positive`) or similar

Example (CSV header):

```
TweetText,Sentiment
"nuclear energy is great!",Positive
"i fear nuclear waste",Negative
```

Paths in the provided script example use:

```python
traindata = pd.read_csv("C:/MyData/PythonPractice/twitter_nuclear/nuclear.csv", encoding='ISO-8859-1')
```

Change the path to your CSV accordingly.

## Requirements

Minimum recommended packages (create a virtualenv):

- Python 3.8+ (tested)
- pandas
- numpy
- matplotlib
- scikit-fuzzy (`skfuzzy`)
- nltk (VADER lexicon)
- scikit-learn (for metrics)

You can install them with pip:

```bash
pip install pandas numpy matplotlib scikit-fuzzy nltk scikit-learn
```

_Note:_ You must download the VADER lexicon for NLTK before running (see Installation below).

## Installation

1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install packages:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install as shown above.

3. Download NLTK VADER lexicon (one-time):

```python
import nltk
nltk.download('vader_lexicon')
```

You can add this to a small bootstrap script or run it in a Python REPL/notebook.

## Usage

1. Place your CSV file in a known location and update the path in the script (or accept a CLI argument if you modify the script).

2. Run the script (or paste into a Jupyter Notebook cell):

```bash
python fuzzy_vader_sentiment.py
# or run in a notebook cell
```

3. The script will print per-tweet VADER results, fuzzy activations, defuzzified outputs and the final evaluation metrics (accuracy, precision, recall, F1). It will also show plots of the membership functions and the aggregated output for each tweet.

### Example output snippet

```
1 nuclear power is expensive --------------------------- {'neg': 0.0, 'neu': 0.6, 'pos': 0.4, 'compound': 0.1027}
Positive Score for each tweet :
0.4
Negative Score for each tweet :
0.0
...
Defuzzified Output: 6.75
Output after Defuzzification: Neutral
Doc sentiment: Neutral

Accuracy is: 78.21
Precision score (MACRO): 75.00
Recall score (MACRO): 74.11
F1 score (MACRO): 74.55
F1 score (MICRO): 78.21
Execution Time: 4.123 secs
```

(Values will vary depending on your dataset.)

## How it works (brief)

1. **Preprocessing:** tweets are lowercased, contraction-expanded, URLs and handles removed, hashtags stripped of `#` symbol.
2. **VADER scoring:** SentimentIntensityAnalyzer returns `pos`, `neg`, `neu`, `compound` values in `[0,1]`.
3. **Fuzzification:** `pos` and `neg` are fuzzified over the domain `[0,1)` into Low/Medium/High triangular MFs; output is defined on `[0,10)` with three triangular MFs (Neg/Neu/Pos).
4. **Rule evaluation:** The script builds nine rule combinations (pos × neg levels) and combines their firing strengths into three output activations (negative, neutral, positive) using `np.fmin` / `np.fmax` logic.
5. **Aggregation & defuzzification:** The three activated output MFs are aggregated into a single fuzzy set and defuzzified using centroid method to produce a numeric output in `[0,10)`. Numeric ranges are then bucketed into classes.

## Fuzzy system details

- **Input universes**
  - `x_p` (pos): `np.arange(0, 1, 0.1)`
  - `x_n` (neg): `np.arange(0, 1, 0.1)`
- **Output universe**
  - `x_op`: `np.arange(0, 10, 1)`

- **Membership functions**
  - `pos`: Low `trimf([0,0,0.5])`, Mid `trimf([0,0.5,1])`, High `trimf([0.5,1,1])`
  - `neg`: same as `pos`
  - `output`: Negative `trimf([0,0,5])`, Neutral `trimf([0,5,10])`, Positive `trimf([5,10,10])`

- **Rule summary**
  - Rules combine pos/neg levels. Intuitively:
    - High positive & low negative -> strongly Positive
    - High negative & low positive -> strongly Negative
    - Both low -> Neutral
    - Mixed strengths lead to intermediate activation depending on min/max operators used in the code

> The rules used in the script are implemented via `np.fmin` and `np.fmax` to compute firing strengths and output clipping.

## Output & evaluation

- The defuzzified output (`op`) is in range approximately `0`–`9`. The script maps numeric output to labels:
  - `0 < op < 3.33` → `Negative`
  - `3.34 < op < 6.66` → `Neutral`
  - `6.67 < op < 10` → `Positive`

- Evaluation prints:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1 (macro and micro)

**Important note on labels:**
- Make sure the ground-truth labels in the CSV match the three-class labels expected by the script (`Negative`, `Neutral`, `Positive`) or pre-map them before evaluation.

## Visualization

The script plots:

- Input membership functions (`pos`, `neg`) and output membership functions.
- For every tweet, an aggregated output membership set and the defuzzified result (as a vertical line). This can be commented out or selectively enabled if visual clutter is a concern.

If running on many tweets, consider saving plots to disk instead of showing them interactively.

## Troubleshooting & tips

- **`nltk` error**: If `SentimentIntensityAnalyzer` fails, run:

```python
import nltk
nltk.download('vader_lexicon')
```

- **`skfuzzy` import fails**: Install with `pip install scikit-fuzzy`.

- **Slow plotting**: The script creates a plot per tweet — comment out per-tweet plotting when running on a large dataset.

- **Precision/recall zero or errors**: Ensure `y_true` and `y_pred` contain exactly the same set of labels (case-sensitive) and at least one sample per class when using macro averaging.

- **Membership interpolation warnings**: If you pass exact 1.0 into the membership arrays, the original script clamps `1 -> 0.9` for plotting; consider normalizing or avoiding rounding before fuzzification.

## Customize

- You can tune the membership function shapes and the output mapping thresholds to better match your dataset.
- Replace rules or use a fuzzy control system API if you want a more declarative ruleset (e.g., `skfuzzy.control` module).

## License

This example is provided under the MIT License — adapt and reuse freely.

## Contact

If you need help adapting the script, include the CSV header and 5–10 sample rows and I can suggest tweaks to MFs/rules or help debug unexpected metric results.

