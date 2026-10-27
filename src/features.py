"""Turn (feature month -> label month) pairs into model-ready tables."""
import numpy as np
import pandas as pd
from build_graph import build_graph, airport_metrics
from config import WEATHER_CSV, STORM_PRECIP_MM, STORM_WIND_KMH
from gcn import embed

# Features that do NOT use the network structure (used by the baseline model too).
BASE_ROUTE = ["log_flights", "disrupted_rate", "cancel_rate", "mean_arr_delay", "distance"]
BASE_AIRPORT = ["out_disrupted_rate", "in_disrupted_rate", "out_flights", "in_flights"]
WEATHER = ["wx_precip_mean", "wx_precip_max", "wx_wind_mean", "wx_storm_share"]
# Features that DO use the network structure (graph model only).
GRAPH_AIRPORT = ["in_degree", "out_degree", "pagerank", "betweenness", "clustering", "top_origin_share", "is_articulation_point"]


def load_weather():
    if not WEATHER_CSV.exists():
        print(f"WARNING: {WEATHER_CSV} not found. Weather features will be empty. Run fetch_weather.py.")
        return None
    w = pd.read_csv(WEATHER_CSV)
    w["month"] = w.date.str[:7]
    w["storm"] = ((w.precip_mm >= STORM_PRECIP_MM) | (w.wind_max_kmh >= STORM_WIND_KMH)).astype(float)
    return w


def weather_month(w, month):
    if w is None:
        return pd.DataFrame(columns=["airport"] + WEATHER)
    s = w[w.month == month].groupby("airport").agg(
        wx_precip_mean=("precip_mm", "mean"), wx_precip_max=("precip_mm", "max"),
        wx_wind_mean=("wind_max_kmh", "mean"), wx_storm_share=("storm", "mean")).reset_index()
    return s


def build_pair(stats, f_month, l_month, gcn_bundle, weather):
    """Features from f_month, label rate from l_month. Nothing from l_month enters the features."""
    G, _ = build_graph(stats, [f_month])
    metrics = airport_metrics(G)
    emb = embed(gcn_bundle, G, metrics)
    wx = weather_month(weather, f_month)
    feat = stats[stats.month == f_month].copy()
    feat["log_flights"] = np.log1p(feat.flights)
    if l_month is None:                         # forecast mode: the label month has not happened yet
        df = feat.copy(); df["label_rate"] = np.nan; df["label_flights"] = np.nan
    else:
        lab = stats[stats.month == l_month][["Origin", "Dest", "disrupted_rate", "flights"]].rename(
            columns={"disrupted_rate": "label_rate", "flights": "label_flights"})
        df = feat.merge(lab, on=["Origin", "Dest"], how="inner")
    am = metrics.copy(); am["is_articulation_point"] = am.is_articulation_point.astype(float)
    cols = BASE_AIRPORT + GRAPH_AIRPORT
    for side, key in (("o", "Origin"), ("d", "Dest")):
        a = am[["airport"] + cols].rename(columns={c: f"{side}_{c}" for c in cols}).rename(columns={"airport": key})
        e = emb.rename(columns={c: f"{side}_{c}" for c in emb.columns if c != "airport"}).rename(columns={"airport": key})
        x = wx.rename(columns={c: f"{side}_{c}" for c in WEATHER}).rename(columns={"airport": key})
        df = df.merge(a, on=key, how="left").merge(e, on=key, how="left").merge(x, on=key, how="left")
    df["feature_month"], df["label_month"] = f_month, l_month
    return df


def feature_sets(df):
    emb_cols = sorted(c for c in df.columns if "_emb" in c)
    base = BASE_ROUTE + [f"{s}_{c}" for s in "od" for c in BASE_AIRPORT] + [f"{s}_{c}" for s in "od" for c in WEATHER]
    graph = [f"{s}_{c}" for s in "od" for c in GRAPH_AIRPORT] + emb_cols
    base = [c for c in base if c in df.columns and df[c].notna().any()]      # drop empty weather columns
    graph = [c for c in graph if c in df.columns]
    return base, base + graph
