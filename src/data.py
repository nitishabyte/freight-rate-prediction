"""Loading and cleaning of the freight-rate data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "posted_rate"
EQUIPMENT_CANON = {"dry van": "Dry Van", "dryvan": "Dry Van", "van": "Dry Van",
                   "reefer": "Reefer", "refrigerated": "Reefer",
                   "flatbed": "Flatbed", "flat bed": "Flatbed"}


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV (comma or tab separated, auto-detected)."""
    return pd.read_csv(path, sep=None, engine="python")


def parse_dates(s: pd.Series) -> pd.Series:
    """Source files use DD-MM-YYYY; also accept ISO dates."""
    out = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    iso = pd.to_datetime(s[out.isna()], format="%Y-%m-%d", errors="coerce")
    out.loc[iso.index] = iso
    return out


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame, is_train: bool, report: dict | None = None) -> pd.DataFrame:
    """Basic cleaning shared by train / validation / December inputs.

    Rows are only *dropped* for training data; scoring data is repaired, never dropped.
    """
    rep = report if report is not None else {}
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    n0 = len(df)

    for col in ("pickup", "delivery", "equipment"):
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    if "equipment" in df:
        norm = df["equipment"].str.lower().map(EQUIPMENT_CANON)
        rep["equipment_relabelled"] = int((norm.notna() & (norm != df["equipment"])).sum())
        df["equipment"] = norm.fillna(df["equipment"])

    df["date"] = parse_dates(df["date"].astype(str))
    rep["bad_dates"] = int(df["date"].isna().sum())

    for col in ("distance", "weight", "market_index", "quote_signal", "pickup_lat",
                "pickup_lon", "delivery_lat", "delivery_lon", TARGET):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impossible physical values -> missing (model handles NaN natively)
    if "weight" in df:
        neg = df["weight"] < 0
        rep["weight_sign_flipped"] = int(neg.sum())   # e.g. -47500 -> 47500 (sign typo)
        df.loc[neg, "weight"] = df.loc[neg, "weight"].abs()
        bad_w = (df["weight"] == 0) | (df["weight"] > 80_000)
        rep["weight_missing"] = int(df["weight"].isna().sum())
        rep["weight_invalid"] = int(bad_w.sum())
        df.loc[bad_w, "weight"] = np.nan
    bad_d = df["distance"] <= 0
    rep["distance_invalid"] = int(bad_d.sum())
    df.loc[bad_d, "distance"] = np.nan

    # Distance sanity check against great-circle distance from the coordinates
    if {"pickup_lat", "delivery_lat"} <= set(df.columns):
        gc = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
        df["gc_distance"] = gc
        ratio = df["distance"] / gc.replace(0, np.nan)
        rep["distance_vs_greatcircle_suspicious"] = int(((ratio < 0.9) | (ratio > 2.5)).sum())

    if is_train:
        # duplicates
        dup_id = df.duplicated("load_id", keep="first") if "load_id" in df else pd.Series(False, df.index)
        feat_cols = [c for c in df.columns if c != "load_id"]
        dup_row = df.duplicated(feat_cols, keep="first")
        rep["duplicate_load_ids"] = int(dup_id.sum())
        rep["duplicate_rows"] = int(dup_row.sum())
        df = df[~(dup_id | dup_row)]

        # target problems
        bad_t = df[TARGET].isna() | (df[TARGET] <= 0) | df["date"].isna() | df["distance"].isna()
        rep["dropped_bad_target_or_key_fields"] = int(bad_t.sum())
        df = df[~bad_t]

        # rate-per-mile outliers within equipment x distance band (robust z-score on log rpm)
        rpm = np.log(df[TARGET] / df["distance"])
        band = pd.cut(df["distance"], [0, 250, 500, 1000, 1500, 2000, np.inf])
        grp = [df["equipment"].astype(str), band.astype(str)]
        med = rpm.groupby(grp).transform("median")
        mad = (rpm - med).abs().groupby(grp).transform("median") * 1.4826
        z = (rpm - med) / mad.replace(0, np.nan)
        out = z.abs() > 5
        rep["dropped_rate_outliers"] = int(out.sum())
        df = df[~out.fillna(False)]
        rep["train_rows_in"] = n0
        rep["train_rows_kept"] = len(df)
    return df.reset_index(drop=True)
