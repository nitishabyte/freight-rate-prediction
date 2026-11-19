# Freight Rate Prediction

Predicts `posted_rate` for freight loads with a gradient-boosted model validated on a time-based split.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Put the assessment files in `data/`:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

```bash
# 1. clean, validate, train final model, write predictions
python -m src.train --data-dir data --out-dir .

# 2. validate outputs and create the December chart
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# 3. build the report (reports/report.docx)
python -m src.report --out-dir .
```

Outputs:

| File | What |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 loads |
| `data/december_chart_inputs.csv` | filled December predictions (dates rewritten as ISO) |
| `scorer_results/candidate_december.png` | chart from `score.py` |
| `reports/` | EDA plots, time-CV results, data-quality summary, `report.docx` |

Smoke test with synthetic data (no real data needed):

```bash
python tests/make_synthetic.py /tmp/synth
python -m src.train --data-dir /tmp/synth --out-dir /tmp/synth_out
```

## Approach

- **Cleaning** (`src/data.py`): day-first date parsing, equipment label normalisation, invalid weights and distances set to NaN, duplicate removal, and robust rate-per-mile outlier removal (training data only).
- **Features** (`src/features.py`): distance, coordinates and circuity, equipment, weight with a missing flag, lane categories, out-of-fold smoothed lane target encodings, `market_index`, `quote_signal`, and calendar and holiday features.
- **Model** (`src/train.py`): `HistGradientBoostingRegressor` on log(rate per mile). The prediction is multiplied by distance to get dollars.
- **Validation**: rolling-origin CV over the last 3 labelled months (train on the past, test on the next month), compared with simple baselines. A random K-fold is reported only to show how much it overstates accuracy.
- **December**: `market_index` and `quote_signal` aren't supplied, so they are projected: a recent-level estimate plus a day-of-week effect for `market_index`, and an auxiliary model for `quote_signal`.
