"""Feature engineering. All encodings are fitted on training data only."""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS = ["pickup", "delivery", "equipment"]  # lane (thousands of levels) enters via target encoding
NUM_COLS = [
    "distance", "log_distance", "weight", "weight_missing",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "gc_distance", "circuity",
    "market_index", "quote_signal", "quote_x_distance",
    "dow", "is_weekend", "dom", "month", "doy", "week", "days_since_start", "near_holiday",
    "lane_rpm_te", "pickup_rpm_te", "delivery_rpm_te",
]

US_HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-07-04", "2025-09-01",
    "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25", "2026-01-01",
])
START = pd.Timestamp("2025-01-01")


class FeatureBuilder:
    """Fits target encodings / category levels on train, applies to any frame."""

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        d = self._base(df)
        rpm = np.log(df["posted_rate"] / df["distance"])
        self.global_rpm = float(rpm.mean())
        self.te = {}
        for key in ("lane", "pickup", "delivery"):
            g = rpm.groupby(d[key].astype(str)).agg(["sum", "count"])
            self.te[key] = (g["sum"] + self.smoothing * self.global_rpm) / (g["count"] + self.smoothing)
        self.levels = {c: d[c].astype(str).value_counts().index[:250].tolist() for c in CAT_COLS}  # model limit: 255 levels
        return self

    @staticmethod
    def _base(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["lane"] = d["pickup"].astype(str) + "->" + d["delivery"].astype(str)
        return d

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self._base(df)
        out = pd.DataFrame(index=d.index)
        for c in CAT_COLS:
            out[c] = pd.Categorical(d[c].astype(str), categories=self.levels[c])
        out["distance"] = d["distance"]
        out["log_distance"] = np.log1p(d["distance"])
        out["weight"] = d.get("weight", np.nan)
        out["weight_missing"] = out["weight"].isna().astype(int)
        for c in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "gc_distance",
                  "market_index", "quote_signal"):
            out[c] = d[c] if c in d else np.nan
        out["circuity"] = out["distance"] / out["gc_distance"].replace(0, np.nan)
        out["quote_x_distance"] = out["quote_signal"] * out["distance"]
        dt = d["date"]
        out["dow"] = dt.dt.dayofweek
        out["is_weekend"] = (out["dow"] >= 5).astype(int)
        out["dom"] = dt.dt.day
        out["month"] = dt.dt.month
        out["doy"] = dt.dt.dayofyear
        out["week"] = dt.dt.isocalendar().week.astype(float).values
        out["days_since_start"] = (dt - START).dt.days
        gap = np.abs(dt.values[:, None] - US_HOLIDAYS_2025.values[None, :]).min(axis=1)
        out["near_holiday"] = (gap <= np.timedelta64(3, "D")).astype(int)
        for key in ("lane", "pickup", "delivery"):
            out[f"{key}_rpm_te"] = d[key].astype(str).map(self.te[key]).astype(float).fillna(self.global_rpm)
        return out[CAT_COLS + NUM_COLS]

    def fit_transform(self, df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
        """Fit on df; target-encoding columns for df itself are computed out-of-fold
        so a row never sees its own target (prevents leakage into the model)."""
        self.fit(df)
        X = self.transform(df)
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, n_folds, len(df))
        for k in range(n_folds):
            m = fold == k
            sub = FeatureBuilder(self.smoothing).fit(df[~m])
            Xk = sub.transform(df[m])
            for key in ("lane", "pickup", "delivery"):
                X.loc[m, f"{key}_rpm_te"] = Xk[f"{key}_rpm_te"].values
        return X
