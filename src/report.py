"""Builds reports/report.docx from the artefacts written by src.train and score.py.

Usage (after training and running score.py):
    python -m src.report --out-dir .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    out = Path(a.out_dir); rep = out / "reports"
    s = json.load(open(rep / "summary.json"))
    cv = pd.read_csv(rep / "time_cv_results.csv")
    dq = s["data_quality"]["train"]

    doc = Document()
    doc.add_heading("Freight Rate Prediction: Approach & Validation", 0)

    doc.add_heading("1. Data overview & quality", 1)
    doc.add_paragraph(
        f"Training file: {dq.get('train_rows_in')} rows in, {dq.get('train_rows_kept')} kept after cleaning. "
        "Dates are DD-MM-YYYY and are parsed explicitly (day-first) to avoid month/day swaps.")
    t = doc.add_table(rows=1, cols=2); t.style = "Light Grid Accent 1"
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Check", "Count / action"
    actions = {
        "duplicate_load_ids": "duplicate load_id - kept first",
        "duplicate_rows": "exact duplicate rows - removed",
        "dropped_bad_target_or_key_fields": "missing/non-positive rate, bad date or distance - removed",
        "dropped_rate_outliers": "rate-per-mile |robust z| > 5 within equipment x distance band - removed",
        "weight_missing": "missing weight - kept, left as NaN + missing flag",
        "weight_invalid": "weight <= 0 or > 80,000 lb - set to NaN",
        "distance_invalid": "distance <= 0 - set to NaN",
        "distance_vs_greatcircle_suspicious": "distance inconsistent with coordinates - flagged",
        "equipment_relabelled": "equipment label variants normalised",
    }
    for k, txt in actions.items():
        r = t.add_row().cells; r[0].text = txt; r[1].text = str(dq.get(k, 0))
    for img in ("eda_overview.png", "eda_time_trend.png", "eda_equipment.png"):
        if (rep / img).exists():
            doc.add_picture(str(rep / img), width=Inches(6.3))

    doc.add_heading("2. Train / validation split approach", 1)
    doc.add_paragraph(
        "The labelled data is a time series of loads, and the scoring set (validation.csv) lies strictly "
        "after the training period. A random split would let the model learn each day's market level "
        "from neighbouring loads of the same day, which overstates accuracy. I therefore validate with "
        "rolling-origin (expanding-window) cross-validation: for each of the last three months, the model "
        "is trained only on earlier months and tested on that month. This mirrors the real task of "
        "predicting the next, unseen month.")
    doc.add_paragraph(
        "Leakage controls: lane/pickup/delivery target encodings are fitted only on the training fold, "
        "and are computed out-of-fold for the training rows themselves. Outlier removal is applied to "
        "training data only; scoring rows are never dropped. After validation, the final model is refit on "
        "all labelled data before predicting validation.csv.")

    doc.add_heading("3. Validation results", 1)
    summ = cv.groupby("model")[["MAE", "RMSE", "MAPE_%", "R2"]].mean().round(3).reset_index()
    t = doc.add_table(rows=1, cols=len(summ.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(summ.columns): t.rows[0].cells[i].text = c
    for _, row in summ.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = str(v)
    doc.add_paragraph("Per-fold results:")
    t = doc.add_table(rows=1, cols=4); t.style = "Light Grid Accent 1"
    for i, c in enumerate(["Test month", "Model", "MAE", "MAPE %"]): t.rows[0].cells[i].text = c
    for _, r in cv.iterrows():
        c = t.add_row().cells
        c[0].text, c[1].text, c[2].text, c[3].text = r.fold_test_month, r.model, f"{r.MAE:.1f}", f"{r['MAPE_%']:.2f}"
    if s.get("random_kfold"):
        rk = s["random_kfold"]
        doc.add_paragraph(f"For comparison, a random 5-fold split gives MAE {rk['MAE']:.1f} / MAPE "
                          f"{rk['MAPE_%']:.2f}%. The time-based figures above are the honest estimate.")
    if (rep / "holdout_actual_vs_pred.png").exists():
        doc.add_picture(str(rep / "holdout_actual_vs_pred.png"), width=Inches(4.5))

    doc.add_heading("4. Model", 1)
    doc.add_paragraph(
        "Gradient-boosted trees (scikit-learn HistGradientBoostingRegressor) predicting log(rate per mile); "
        "the prediction is converted back to dollars by multiplying by distance. Modelling rate per mile on a "
        "log scale removes most of the distance-driven scale, makes errors roughly proportional, and keeps "
        "predictions positive. Features: distance, coordinates and circuity, equipment, weight (+ missing "
        "flag), pickup/delivery/lane categories and smoothed lane target encodings, market_index, "
        "quote_signal and quote_signal x distance, and calendar features (day of week, weekend, day of "
        "month, holiday proximity). Missing values are handled natively by the model. Benchmarks: "
        "quote_signal x distance and lane-median rate per mile.")

    doc.add_heading("5. December prediction chart", 1)
    doc.add_paragraph(
        "december_chart_inputs.csv does not contain market_index or quote_signal. Because validation.csv "
        "covers November-December, market_index for each December day was set to the observed daily mean "
        "from validation.csv features (no labels involved), with a recent-level + day-of-week projection as "
        "fallback; quote_signal was estimated with an auxiliary model from lane, equipment, "
        "distance, date and market_index. Dates are written back as ISO (YYYY-MM-DD) because score.py "
        "parses dates without a day-first setting.")
    chart = out / "scorer_results" / "candidate_december.png"
    if chart.exists():
        doc.add_picture(str(chart), width=Inches(6.5))
    else:
        doc.add_paragraph("[Run score.py to generate candidate_december.png, then rerun this script.]")

    doc.add_heading("6. Limitations", 1)
    doc.add_paragraph(
        "No December history exists in the labelled data, so holiday-season effects (e.g. the Christmas "
        "capacity squeeze) cannot be learned and are only approximated by calendar features. "
        "December market inputs are projections, so the chart reflects expected rather than realised "
        "market conditions.")
    doc.save(rep / "report.docx")
    print("Wrote", rep / "report.docx")


if __name__ == "__main__":
    main()
