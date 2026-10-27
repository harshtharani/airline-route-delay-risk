"""Forecast: score the LATEST month in the data and predict each route's risk class for the month after it.

    python forecast.py

Training uses every consecutive month pair in TRAIN_MONTHS + TEST_MONTHS (features of month t, outcome of month t+1).
The forecast month's outcomes do not exist yet, so nothing can be checked against them. This is a prediction,
not a verified result. The honest accuracy estimate is the held-out test month from train_eval.py.
"""
import json
import numpy as np
import pandas as pd
from build_graph import build_graph, airport_metrics
from config import PROCESSED, OUTPUTS, TRAIN_MONTHS, TEST_MONTHS, RISK_QUANTILES, RISK_NAMES, check_consecutive
from features import build_pair, feature_sets, load_weather
from gcn import train_gcn
from train_eval import fit_xgb, shap_3d


def main():
    OUTPUTS.mkdir(exist_ok=True)
    stats = pd.read_csv(PROCESSED / "route_month_stats.csv")
    months = TRAIN_MONTHS + TEST_MONTHS
    check_consecutive(months)
    last = months[-1]
    forecast_month = (pd.Period(last, "M") + 1).strftime("%Y-%m")
    pairs = list(zip(months[:-1], months[1:]))
    print(f"Training on {len(pairs)} month pair(s): {pairs}")
    print(f"Scoring features from {last} to forecast {forecast_month}")

    G0, _ = build_graph(stats, [months[0]])
    gcn = train_gcn(G0, airport_metrics(G0), verbose=False)
    weather = load_weather()
    train = pd.concat([build_pair(stats, f, l, gcn, weather) for f, l in pairs], ignore_index=True)
    cuts = np.quantile(train.label_rate, RISK_QUANTILES)
    to_class = lambda r: np.digitize(r, cuts)
    ytr = to_class(train.label_rate)
    _, graph_cols = feature_sets(train)
    model = fit_xgb(train[graph_cols], ytr)

    latest = build_pair(stats, last, None, gcn, weather)
    X = latest[graph_cols]
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)
    sv = shap_3d(model, X)
    own = sv[np.arange(len(X)), :, pred]
    names = np.array(graph_cols)
    top = np.argsort(-np.abs(own), axis=1)[:, :3]
    drivers = [", ".join(f"{names[j]} ({own[i, j]:+.2f})" for j in top[i]) for i in range(len(X))]

    out = latest[["Origin", "Dest", "feature_month", "flights", "disrupted_rate"]].copy()
    out["forecast_month"] = forecast_month
    out["pred_class"] = [RISK_NAMES[i] for i in pred]
    out["pred_persistence"] = [RISK_NAMES[i] for i in to_class(latest.disrupted_rate)]
    for j, n in enumerate(RISK_NAMES):
        out[f"p_{n}"] = proba[:, j].round(4)
    out["top_drivers"] = drivers
    out.to_csv(OUTPUTS / "forecast_next_month.csv", index=False)
    meta = {"feature_month": last, "forecast_month": forecast_month, "train_pairs": pairs,
            "risk_cutoffs": cuts.tolist(), "routes_scored": len(out)}
    (OUTPUTS / "forecast_meta.json").write_text(json.dumps(meta, indent=2))
    counts = out.pred_class.value_counts().to_dict()
    print(f"Scored {len(out)} routes for {forecast_month}: {counts}")
    print("Saved outputs/forecast_next_month.csv")


if __name__ == "__main__":
    main()
