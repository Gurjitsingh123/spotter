# Freight Rate Prediction — Spotter ML Engineer assessment

Predicts the posted rate (USD) of a truck load from its lane, distance, equipment, weight, date and daily market signals.

## Results (time-based validation)

Train on every month **before** the test month, test on that month (expanding window), so the model never sees the future:

| Test month | MAPE (clean rows) | MAE (clean rows) | MAE (all rows, incl. corrupt labels) |
|---|---|---|---|
| 2025-07 | 1.19% | $28 | $83 |
| 2025-08 | 5.71% | $112 | $160 |
| 2025-09 | 2.1% | $51 | $102 |
| 2025-10 | 1.97% | $46 | $105 |

**Unseen cities** (8 cities removed from training, tested on Sep–Oct loads that touch them): MAPE 2.38%, MAE $59.
This matters because 8 cities in `validation.csv` never appear in the training data.

## Approach in one paragraph
Validation covers Nov–Dec 2025 while training covers Jan–Oct 2025, so all validation is **time-based**. About 1.3% of labels are
corrupt (rate per mile 2–7× too high or too low) and are removed from training; negative weights are sign errors (abs), missing
weights get the equipment median, and missing `market_index` values get that day's mean (the index is ~constant within a day).
A **LightGBM** model predicts **rate per mile**, which is multiplied back by distance — this beat predicting log(rate) and a
ridge baseline. Features: distance, straight-line miles, coordinates (generalise to unseen cities), equipment, weight,
`market_index`, `quote_signal`, their daily means, and weekday. For the December chart, city coordinates come from the
training data and each day's market signals from the validation file (which already contains December dates).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py                       # validation report + final model + both prediction files
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`src/train.py` writes `validation_predictions.csv`, fills `data/december_chart_inputs.csv`, and saves metrics and charts in
`reports/`. `score.py` (provided by Spotter) validates both files and creates `scorer_results/candidate_december.png`.

## Layout
```
data/                     input CSVs (december_chart_inputs.csv is filled by train.py)
src/features.py           cleaning + feature engineering (shared by training and prediction)
src/train.py              validation, final training, prediction files, report charts
score.py                  Spotter's scorer
reports/                  metrics.json, charts, Freight_Rate_Report.pdf
validation_predictions.csv
```

Data: 48,000 labelled loads (2025-01-01 → 2025-10-31), 12,000 validation loads (2025-11-01 → 2025-12-31).
