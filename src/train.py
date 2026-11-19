"""End-to-end pipeline: clean -> validate (time-based) -> fit final model -> predict.

Usage:
    python -m src.train --data-dir data --out-dir .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .data import TARGET, clean, read_table
from .features import FeatureBuilder

SEED = 42
PARAMS = dict(learning_rate=0.05, max_iter=1500, max_leaf_nodes=63, min_samples_leaf=40,
              l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
              n_iter_no_change=50, categorical_features="from_dtype", random_state=SEED)


# --------------------------------------------------------------------------- metrics
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": float(np.mean(np.abs(y - p))),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "MAPE_%": float(np.mean(np.abs(y - p) / y) * 100),
            "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            "n": int(len(y))}


# --------------------------------------------------------------------------- model
class RateModel:
    """Predicts log(rate per mile) with gradient boosting; returns rate in $."""

    def fit(self, df: pd.DataFrame):
        self.fb = FeatureBuilder()
        X = self.fb.fit_transform(df)
        y = np.log(df[TARGET] / df["distance"])
        self.model = HistGradientBoostingRegressor(**PARAMS).fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.fb.transform(df)
        return np.exp(self.model.predict(X)) * df["distance"].to_numpy()


def baseline_quote(df):            # rate ~= quote_signal * distance
    return df["quote_signal"].to_numpy() * df["distance"].to_numpy()


def baseline_lane(train, test):    # median lane x equipment rate-per-mile
    rpm = train[TARGET] / train["distance"]
    key = lambda d: d["pickup"].astype(str) + "|" + d["delivery"].astype(str) + "|" + d["equipment"].astype(str)
    lane = rpm.groupby(key(train)).median()
    eq = rpm.groupby(train["equipment"].astype(str)).median()
    r = key(test).map(lane).fillna(test["equipment"].astype(str).map(eq)).fillna(rpm.median())
    return r.to_numpy() * test["distance"].to_numpy()


# --------------------------------------------------------------------------- validation
def time_validation(df: pd.DataFrame, n_months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin (expanding window) CV: for each of the last n months,
    train on all earlier months and test on that month."""
    months = sorted(df["date"].dt.to_period("M").unique())
    rows, last_preds = [], None
    for m in months[-n_months:]:
        tr = df[df["date"].dt.to_period("M") < m]
        te = df[df["date"].dt.to_period("M") == m]
        model = RateModel().fit(tr)
        p = model.predict(te)
        for name, pred in (("HGB model", p), ("quote_signal x distance", baseline_quote(te)),
                           ("lane median rpm", baseline_lane(tr, te))):
            rows.append({"fold_test_month": str(m), "train_rows": len(tr), "model": name, **metrics(te[TARGET], pred)})
        last_preds = te.assign(pred=p)
    return pd.DataFrame(rows), last_preds


def random_kfold(df: pd.DataFrame, k: int = 5) -> dict:
    """Shown only for comparison: random splits leak future market info."""
    ys, ps = [], []
    for tr_i, te_i in KFold(k, shuffle=True, random_state=SEED).split(df):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        ps.append(RateModel().fit(tr).predict(te)); ys.append(te[TARGET])
    return metrics(np.concatenate(ys), np.concatenate(ps))


# --------------------------------------------------------------------------- December inputs
def impute_market_features(dec: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """December rows lack market_index and quote_signal.

    market_index: daily mean over the last 28 observed days (train + unlabeled validation
                  features, no targets used), adjusted by a day-of-week effect.
    quote_signal: auxiliary model predicting quote_signal from lane/equipment/distance/date/
                  market_index, trained on every row where quote_signal is observed.
    """
    dec = dec.copy()
    daily = history.groupby("date")["market_index"].mean().sort_index()
    recent = daily[daily.index > daily.index.max() - pd.Timedelta(days=28)]
    dow_eff = (daily - daily.rolling(7, center=True, min_periods=4).mean()).groupby(daily.index.day_of_week).mean()
    projected = recent.mean() + dec["date"].dt.day_of_week.map(dow_eff).fillna(0)
    # validation.csv covers December, so use the observed daily market_index where available
    dec["market_index"] = dec["date"].map(daily).astype(float).fillna(projected)

    hist = history.dropna(subset=["quote_signal"]).copy()
    hist["posted_rate"] = hist["quote_signal"] * hist["distance"]  # dummy for FeatureBuilder.fit
    fb = FeatureBuilder().fit(hist)
    drop = ["quote_signal", "quote_x_distance", "lane_rpm_te", "pickup_rpm_te", "delivery_rpm_te"]
    Xh = fb.transform(hist).drop(columns=drop)
    aux = HistGradientBoostingRegressor(**PARAMS).fit(Xh, hist["quote_signal"])
    dec["quote_signal"] = aux.predict(fb.transform(dec.assign(quote_signal=np.nan)).drop(columns=drop))
    return dec


# --------------------------------------------------------------------------- EDA plots
def eda_plots(df: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(df[TARGET], bins=80, color="#064A56"); ax[0].set_title("posted_rate distribution")
    ax[1].scatter(df["distance"], df[TARGET], s=2, alpha=0.2, color="#064A56")
    ax[1].set_title("rate vs distance"); ax[1].set_xlabel("miles")
    rpm = df[TARGET] / df["distance"]
    ax[2].scatter(df["quote_signal"], rpm, s=2, alpha=0.2, color="#064A56")
    ax[2].set_title("rate-per-mile vs quote_signal"); ax[2].set_xlabel("quote_signal")
    fig.tight_layout(); fig.savefig(out / "eda_overview.png", dpi=130); plt.close(fig)

    daily = df.assign(rpm=rpm).groupby("date").agg(rpm=("rpm", "median"), mi=("market_index", "mean"))
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(daily.index, daily["rpm"], color="#064A56", label="median $/mile")
    ax2 = ax1.twinx(); ax2.plot(daily.index, daily["mi"], color="#C8553D", alpha=0.7, label="market_index")
    ax1.set_title("Daily median rate-per-mile and market_index"); fig.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out / "eda_time_trend.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    df.assign(rpm=rpm).boxplot(column="rpm", by="equipment", ax=ax, showfliers=False)
    ax.set_title("rate-per-mile by equipment"); plt.suptitle("")
    fig.tight_layout(); fig.savefig(out / "eda_equipment.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--skip-random-cv", action="store_true")
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out_dir)
    rep_dir = out / "reports"; rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. load + clean
    dq_train, dq_val, dq_dec = {}, {}, {}
    raw = read_table(data / "train_test.csv")
    train = clean(raw, is_train=True, report=dq_train)
    val_raw = read_table(data / "validation.csv")
    val = clean(val_raw, is_train=False, report=dq_val)
    dec_raw = read_table(data / "december_chart_inputs.csv")
    dec = clean(dec_raw.drop(columns=["predicted_rate"]), is_train=False, report=dq_dec)
    print("Train date range:", train.date.min().date(), "->", train.date.max().date(), "| rows", len(train))
    print("Validation date range:", val.date.min().date(), "->", val.date.max().date(), "| rows", len(val))
    eda_plots(train, rep_dir)

    # Lane coordinates for December (not supplied in that file): look up from known cities
    coords = pd.concat([
        train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).groupby("city")[["lat", "lon"]].median()
    for side in ("pickup", "delivery"):
        dec[f"{side}_lat"] = dec[side].map(coords["lat"]).astype(float)
        dec[f"{side}_lon"] = dec[side].map(coords["lon"]).astype(float)
    from .data import haversine_miles
    dec["gc_distance"] = haversine_miles(dec.pickup_lat, dec.pickup_lon, dec.delivery_lat, dec.delivery_lon)

    # 2. validation
    cv, last_fold = time_validation(train, n_months=3)
    cv.to_csv(rep_dir / "time_cv_results.csv", index=False)
    summary = cv.groupby("model")[["MAE", "RMSE", "MAPE_%", "R2"]].mean().round(3)
    print("\nRolling-origin time CV (mean of last 3 months):\n", summary)
    rnd = None if a.skip_random_cv else random_kfold(train)
    if rnd: print("\nRandom 5-fold (for comparison only):", {k: round(v, 3) for k, v in rnd.items()})

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(last_fold[TARGET], last_fold["pred"], s=3, alpha=0.3, color="#064A56")
    lim = [0, last_fold[TARGET].quantile(0.995)]
    ax.plot(lim, lim, "r--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual posted_rate"); ax.set_ylabel("predicted")
    ax.set_title(f"Hold-out month {last_fold.date.dt.to_period('M').iloc[0]}: actual vs predicted")
    fig.tight_layout(); fig.savefig(rep_dir / "holdout_actual_vs_pred.png", dpi=130); plt.close(fig)

    # 3. final model on all labelled data
    final = RateModel().fit(train)

    # 4. validation.csv predictions
    pv = final.predict(val)
    tmpl = read_table(data / "validation_predictions_template.csv")[["load_id"]]
    pred_map = pd.Series(pv, index=val["load_id"].astype(str))
    tmpl["predicted_rate"] = tmpl["load_id"].astype(str).map(pred_map).round(2)
    assert tmpl["predicted_rate"].notna().all() and (tmpl["predicted_rate"] > 0).all()
    tmpl.to_csv(out / "validation_predictions.csv", index=False)

    # 5. December chart inputs
    history = pd.concat([train, val], ignore_index=True)
    dec_f = impute_market_features(dec, history)
    dec_raw = dec_raw.copy()
    dec_raw["date"] = dec["date"].dt.strftime("%Y-%m-%d")   # ISO so score.py parses correctly
    dec_raw["predicted_rate"] = np.round(final.predict(dec_f), 2)
    dec_raw.to_csv(data / "december_chart_inputs.csv", index=False)
    dec_f[["date", "market_index", "quote_signal"]].to_csv(rep_dir / "december_imputed_inputs.csv", index=False)

    # 6. persist report artefacts
    json.dump({"data_quality": {"train": dq_train, "validation": dq_val, "december": dq_dec},
               "time_cv_summary": summary.reset_index().to_dict("records"),
               "random_kfold": rnd, "params": {k: str(v) for k, v in PARAMS.items()}},
              open(rep_dir / "summary.json", "w"), indent=2, default=str)
    print("\nWrote validation_predictions.csv and data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
