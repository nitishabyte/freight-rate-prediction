"""Generates fake data with the assessment's schema, for smoke-testing only."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
cities = {"Lexington": (36.99, -85.0), "Fort Wayne": (41.0, -85.1), "Dallas": (31.83, -94.38),
          "Atlanta": (34.85, -86.29), "Phoenix": (28.76, -115.88), "Richmond": (38.09, -76.79),
          "Fresno": (33.08, -119.80), "Detroit": (43.62, -84.05), "Memphis": (35.57, -89.53),
          "Philadelphia": (39.22, -72.97)}
names = list(cities)
def gen(n, start, end, prefix, label):
    p = rng.choice(names, n); d = rng.choice(names, n)
    same = p == d; d[same] = np.roll(names, 1)[[names.index(x) for x in p[same]]]
    plat, plon = np.array([cities[x] for x in p]).T; dlat, dlon = np.array([cities[x] for x in d]).T
    gc = 69 * np.hypot(plat - dlat, (plon - dlon) * 0.8); dist = np.round(gc * rng.uniform(1.1, 1.3, n), 1)
    dates = pd.to_datetime(rng.integers(pd.Timestamp(start).value // 86400e9, pd.Timestamp(end).value // 86400e9 + 1, n), unit="D")
    t = (dates - pd.Timestamp("2025-01-01")).days.to_numpy()
    mi = 0.95 + 0.03 * np.sin(t / 40) - 0.0002 * t + rng.normal(0, 0.02, n)
    eq = rng.choice(["Dry Van", "Reefer", "Flatbed"], n, p=[.6, .25, .15])
    base = 1.6 + 0.4 * (eq == "Reefer") + 0.3 * (eq == "Flatbed") + 120 / (dist + 50)
    true_rpm = base * mi * (1 + 0.04 * (pd.DatetimeIndex(dates).dayofweek >= 5))
    qs = np.round(true_rpm * rng.normal(1, 0.05, n), 5)
    w = rng.integers(15000, 45000, n).astype(float); w[rng.random(n) < 0.03] = np.nan
    df = pd.DataFrame({"load_id": [f"{prefix}-{i:06d}" for i in range(1, n + 1)], "pickup": p, "delivery": d,
        "pickup_lat": plat, "pickup_lon": plon, "delivery_lat": dlat, "delivery_lon": dlon, "distance": dist,
        "equipment": eq, "weight": w, "date": dates.strftime("%d-%m-%Y"), "market_index": np.round(mi, 5), "quote_signal": qs})
    if label:
        df["posted_rate"] = np.round(true_rpm * dist * rng.normal(1, 0.06, n), 2)
        df.loc[rng.choice(n, 30), "posted_rate"] *= 10   # outliers
        df = pd.concat([df, df.sample(50, random_state=1)])  # duplicates
    return df
gen(40000, "2025-01-01", "2025-10-31", "TR", True).to_csv(out / "train_test.csv", index=False)
v = gen(12000, "2025-11-01", "2025-11-30", "TE", False); v.to_csv(out / "validation.csv", index=False)
v[["load_id"]].assign(predicted_rate="").to_csv(out / "validation_predictions_template.csv", index=False)
pd.DataFrame({"pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360, "equipment": "Dry Van",
    "weight": 32000, "date": pd.date_range("2025-12-01", "2025-12-31").strftime("%d-%m-%Y"), "predicted_rate": ""}
    ).to_csv(out / "december_chart_inputs.csv", index=False)
