"""Airline Route Delay-Risk dashboard.  Run with:  streamlit run app.py"""
import json
import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from build_graph import airport_metrics, build_graph                       # noqa: E402
from config import AIRPORTS_DAT, OUTPUTS, PROCESSED, RISK_NAMES            # noqa: E402
from simulate import impact, rank_hubs                                      # noqa: E402

st.set_page_config(page_title="Airline Route Delay-Risk", page_icon="✈️", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1 {letter-spacing: -0.5px;}
div[data-testid="stMetric"] {background: #F2F5F9; border-radius: 10px; padding: 8px 14px;}
</style>""", unsafe_allow_html=True)

COLORS = {"low": [46, 160, 90, 170], "medium": [240, 170, 30, 190], "high": [210, 50, 50, 220]}
BADGE = {"low": "🟢 Low", "medium": "🟠 Medium", "high": "🔴 High"}
MODEL_LABEL = {"persistence": "Repeat this month", "xgb_baseline": "XGBoost (no graph)", "xgb_graph": "XGBoost + graph"}
NAMES = {
    "log_flights": "Flight volume", "disrupted_rate": "Disruption rate (latest month)", "cancel_rate": "Cancellation rate",
    "mean_arr_delay": "Average arrival delay", "distance": "Route distance", "out_disrupted_rate": "Outbound disruption rate",
    "in_disrupted_rate": "Inbound disruption rate", "out_flights": "Outbound flights", "in_flights": "Inbound flights",
    "wx_precip_mean": "Average rainfall", "wx_precip_max": "Heaviest rainfall day", "wx_wind_mean": "Average wind",
    "wx_storm_share": "Share of stormy days", "in_degree": "Inbound routes", "out_degree": "Outbound routes",
    "pagerank": "PageRank importance", "betweenness": "Bridge role (betweenness)", "clustering": "Clustering",
    "top_origin_share": "Dependence on top origin", "is_articulation_point": "Network cut-point"}


def pretty(f):
    m = re.match(r"^([od])_(.+)$", f)
    side = ""
    if m and (m.group(2) in NAMES or m.group(2).startswith("emb")):
        side, f = ("Origin " if m.group(1) == "o" else "Destination "), m.group(2)
    if f.startswith("emb"):
        return f"{side}GCN fingerprint {int(f[3:]) + 1}"
    return side + NAMES.get(f, f)


def pretty_drivers(s):
    return re.sub(r"([A-Za-z_]+\d*) \(([+-][0-9.]+)\)", lambda m: f"{pretty(m.group(1))} ({m.group(2)})", str(s))


def month_label(m):
    return pd.Period(m, "M").strftime("%b %Y")


FILES = [OUTPUTS / "metrics.json", OUTPUTS / "predictions_test.csv", PROCESSED / "route_month_stats.csv",
         OUTPUTS / "forecast_next_month.csv"]
STAMP = tuple(p.stat().st_mtime if p.exists() else 0 for p in FILES)


@st.cache_data
def load_all(stamp):
    if any(not p.exists() for p in FILES[:3]):
        return None
    cols = ["id", "name", "city", "country", "iata", "icao", "lat", "lon", "alt", "tz", "dst", "tzdb", "type", "source"]
    ap = pd.read_csv(AIRPORTS_DAT, header=None, names=cols, na_values=["\\N"]).dropna(subset=["iata"]).drop_duplicates("iata")
    d = {"metrics": json.loads(FILES[0].read_text()),
         "comp": pd.read_csv(OUTPUTS / "model_comparison.csv", index_col=0),
         "pred": pd.read_csv(FILES[1]),
         "shap": pd.read_csv(OUTPUTS / "shap_importance.csv", index_col=0),
         "stats": pd.read_csv(FILES[2]),
         "coords": ap.set_index("iata")[["name", "lat", "lon"]],
         "forecast": None, "fmeta": None}
    if FILES[3].exists():
        d["forecast"] = pd.read_csv(FILES[3])
        d["fmeta"] = json.loads((OUTPUTS / "forecast_meta.json").read_text())
    return d


@st.cache_data
def month_graph(month, stamp):
    stats = pd.read_csv(PROCESSED / "route_month_stats.csv")
    G, _ = build_graph(stats, [month])
    return G, airport_metrics(G)


def add_coords(df, coords):
    o = coords.rename(columns={"lat": "o_lat", "lon": "o_lon", "name": "o_name"})
    d = coords.rename(columns={"lat": "d_lat", "lon": "d_lon", "name": "d_name"})
    return df.merge(o, left_on="Origin", right_index=True).merge(d, left_on="Dest", right_index=True)


def deck(layers):
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=39, longitude=-96, zoom=3),
                    tooltip={"text": "{tip}"})


def legend():
    st.markdown("**Risk colour:** 🟢 low &nbsp;&nbsp; 🟠 medium &nbsp;&nbsp; 🔴 high")


def route_map(df, cls_col, coords, key, default_n=300):
    c1, c2 = st.columns([2, 3])
    show = c1.multiselect("Show risk classes", RISK_NAMES, default=RISK_NAMES, key=f"{key}_cls")
    topn = c2.slider("Busiest routes to draw", 50, 1500, default_n, step=50, key=f"{key}_n")
    r = add_coords(df[df[cls_col].isin(show)].nlargest(topn, "flights"), coords)
    if r.empty:
        st.info("No routes match these filters."); return
    r["color"] = r[cls_col].map(COLORS)
    r["tip"] = r.Origin + " to " + r.Dest + " | predicted " + r[cls_col] + " | chance of high risk " + (r.p_high * 100).round(0).astype(int).astype(str) + "%"
    traffic = pd.concat([r.groupby("Origin").flights.sum(), r.groupby("Dest").flights.sum()], axis=1).fillna(0).sum(axis=1)
    pts = coords.loc[traffic.index].rename_axis("iata").reset_index()
    pts["tip"] = pts.iata + " " + pts.name.fillna("")
    labels = pts[pts.iata.isin(traffic.nlargest(20).index)]
    legend()
    st.pydeck_chart(deck([
        pdk.Layer("LineLayer", r, get_source_position=["o_lon", "o_lat"], get_target_position=["d_lon", "d_lat"],
                  get_color="color", get_width=2, pickable=True),
        pdk.Layer("ScatterplotLayer", pts, get_position=["lon", "lat"], get_radius=14000, get_fill_color=[30, 58, 95, 230], pickable=True),
        pdk.Layer("TextLayer", labels, get_position=["lon", "lat"], get_text="iata", get_size=13, get_color=[20, 20, 20, 255])]))


def route_table(df, cls_col, show_actual):
    t = pd.DataFrame({
        "Route": df.Origin + " → " + df.Dest,
        "Flights (latest month)": df.flights,
        "Disruption rate (latest month)": (df.disrupted_rate * 100).round(1),
        "Predicted risk": df[cls_col].map(BADGE),
        "Chance of high risk (%)": (df.p_high * 100).round(0)})
    if show_actual:
        t["Actual risk"] = df.true_class.map(BADGE)
    t["Why (top 3 factors)"] = df.top_drivers.map(pretty_drivers)
    st.dataframe(t, hide_index=True, width="stretch", height=430, column_config={
        "Chance of high risk (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        "Disruption rate (latest month)": st.column_config.NumberColumn(format="%.1f%%"),
        "Why (top 3 factors)": st.column_config.TextColumn(width="large")})


def headline(m, comp):
    f1 = comp.macro_f1
    lines = [f"All three approaches score between {f1.min():.2f} and {f1.max():.2f} macro F1 on the held-out test month "
             f"(random guessing among three classes would give about 0.33)."]
    b = m.get("bootstrap")
    if b:
        d = b["differences"]["xgb_graph minus xgb_baseline"]
        if d["includes_zero"]:
            lines.append(f"Adding network (graph) features did **not clearly improve** results: macro F1 changed by {d['mean']:+.3f} "
                         f"(95% interval {d['low']:+.3f} to {d['high']:+.3f}, which includes zero).")
        else:
            lines.append(f"Adding network (graph) features changed macro F1 by {d['mean']:+.3f} "
                         f"(95% interval {d['low']:+.3f} to {d['high']:+.3f}), which is outside the noise range.")
        p = b["differences"]["xgb_graph minus persistence"]
        if p["includes_zero"]:
            lines.append("The trained model was **not clearly better** than simply assuming next month looks like this month.")
        elif p["mean"] > 0:
            lines.append("The trained model beat the 'repeat this month' baseline by a margin larger than the noise.")
        else:
            lines.append("The 'repeat this month' baseline was **better** than the trained model by a margin larger than the noise.")
    else:
        lines.append("Re-run `python run_all.py` to add confidence intervals that show whether the differences are real.")
    return lines


data = load_all(STAMP)

with st.sidebar:
    st.markdown("## ✈️ Route Delay-Risk")
    st.caption("Graph ML on the US airline route network")
    st.markdown("**Method:** features from one month predict each route's risk class in the next month. Airports are nodes, routes "
                "are edges; a GCN learns airport fingerprints and XGBoost predicts the risk class.")
    st.markdown("**Data:** US BTS on-time performance, OpenFlights airports, Open-Meteo weather.")
    st.caption("For analysis and learning, not operational use.")

st.title("Airline Route Delay-Risk Analyzer")
if data is None:
    st.error("No results found yet. From the project folder run:  python run_all.py   (see README.md)")
    st.stop()

m, comp, pred, coords, stats = data["metrics"], data["comp"], data["pred"], data["coords"], data["stats"]
tabs = st.tabs(["Overview", "Model results", "Route explorer", "Network map", "Forecast", "Hub shutdown"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    lo, hi = m["risk_cutoffs"]
    train_txt = "; ".join(f"{month_label(a)} → {month_label(b)}" for a, b in m["train_pairs"])
    test_txt = "; ".join(f"{month_label(a)} → {month_label(b)}" for a, b in m["test_pairs"])
    st.info(
        "**What does this predict?** For every airline route (an airport pair such as ORD → ASE) it predicts how risky that route "
        "will be **next month**: 🟢 low, 🟠 medium or 🔴 high.\n\n"
        f"**What is 'risk'?** The share of a route's flights that are disrupted (arrive 15+ minutes late, are cancelled or are diverted). "
        f"Low means under {lo:.1%} of flights disrupted, high means over {hi:.1%} (cut-offs set from the training data).\n\n"
        f"**How was it tested?** Trained on {train_txt}, then tested on {test_txt}, outcomes the model never saw. "
        "This is a backtest on the past. The **Forecast** tab scores the month after the latest data.\n\n"
        "**What it does not do:** predict single flights, delay minutes or specific airlines. The 'why' factors show what "
        "influenced a prediction, not proof of cause.")
    best = comp.macro_f1.idxmax()
    c = st.columns(4)
    used = {x for pr in m["train_pairs"] + m["test_pairs"] for x in pr}
    c[0].metric("Flights analysed (months used)", f"{int(stats[stats.month.isin(used)].flights.sum()):,}")
    c[1].metric("Routes scored in test month", f"{m['test_rows']:,}")
    c[2].metric("Best model (macro F1)", f"{comp.macro_f1.max():.2f}", MODEL_LABEL[best], delta_color="off")
    c[3].metric("Predicted high-risk routes", f"{int((pred.pred_xgb_graph == 'high').sum()):,}")
    st.subheader("Key finding")
    for line in headline(m, comp):
        st.markdown(f"- {line}")
    st.caption("Macro F1 averages the F1 score of the three risk classes, so each class counts equally.")

# ------------------------------------------------------------ Model results
with tabs[1]:
    st.subheader("How well does each approach predict next month's risk class?")
    boot = m.get("bootstrap")
    rows = []
    for k in comp.index:
        mean = float(comp.loc[k, "macro_f1"]); l, h = mean, mean
        if boot and k in boot["macro_f1"]:
            _, l, h = boot["macro_f1"][k]
        rows.append({"model": MODEL_LABEL[k], "macro_f1": mean, "low": l, "high": h})
    cdf = pd.DataFrame(rows)
    bars = alt.Chart(cdf).mark_bar(size=46, color="#1F3A5F").encode(
        x=alt.X("model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("macro_f1:Q", title="Macro F1 (test month)", scale=alt.Scale(domain=[0, max(0.7, float(cdf.high.max()) + 0.08)])))
    err = alt.Chart(cdf).mark_rule(color="#C0392B", strokeWidth=3).encode(x=alt.X("model:N", sort=None), y="low:Q", y2="high:Q")
    chance = alt.Chart(pd.DataFrame({"y": [1 / 3]})).mark_rule(strokeDash=[5, 4], color="#888").encode(y="y:Q")
    st.altair_chart(alt.layer(bars, err, chance).properties(height=330), width="stretch")
    st.caption("Red lines are 95% bootstrap intervals (resampling test routes). The dashed line is random guessing (0.33). "
               "'Repeat this month' means predicting that next month's class equals this month's, with no learning.")
    if boot:
        drows = [{"Comparison": k.replace("xgb_graph", "XGBoost + graph").replace("xgb_baseline", "XGBoost (no graph)")
                  .replace("persistence", "Repeat this month").replace(" minus ", " − "),
                  "Difference in macro F1": round(d["mean"], 3), "95% interval": f"{d['low']:+.3f} to {d['high']:+.3f}",
                  "Clear difference?": "No, includes zero" if d["includes_zero"] else "Yes"} for k, d in boot["differences"].items()]
        st.markdown("**Are the differences real or noise?**")
        st.dataframe(pd.DataFrame(drows), hide_index=True, width="stretch")
    st.markdown("**All metrics (test month)**")
    t = comp.rename(index=MODEL_LABEL, columns={"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1",
                                                 "high_precision": "High-risk precision", "high_recall": "High-risk recall"})
    st.dataframe(t, width="stretch")
    st.caption("High-risk precision: of routes flagged high, the share that really were. High-risk recall: of truly high routes, the share caught.")
    pc = m["results"]["xgb_graph"]["per_class"]
    st.markdown("**Per-class scores, XGBoost + graph**")
    st.dataframe(pd.DataFrame(pc).T.rename(columns={"precision": "Precision", "recall": "Recall", "f1": "F1"}).round(3).rename(index=str.capitalize), width="stretch")
    c1, c2 = st.columns(2)
    if (OUTPUTS / "confusion_matrices.png").exists():
        c1.image(str(OUTPUTS / "confusion_matrices.png"), caption="Confusion matrices: rows = actual class, columns = predicted class. The diagonal is correct.")
    imp = data["shap"].head(15).reset_index(); imp.columns = ["feature", "importance"]; imp["feature"] = imp.feature.map(pretty)
    c2.markdown("**What drives the XGBoost + graph model (SHAP)**")
    c2.altair_chart(alt.Chart(imp).mark_bar(color="#1F3A5F").encode(
        x=alt.X("importance:Q", title="Average influence on predictions"), y=alt.Y("feature:N", sort="-x", title=None)).properties(height=420),
        width="stretch")
    st.caption(f"GCN sanity check: link-prediction AUC on a later month's graph = {m['gcn_link_auc_later_month']:.3f} "
               "(1.0 would mean it perfectly separates real routes from random airport pairs).")

# ----------------------------------------------------------- Route explorer
with tabs[2]:
    st.subheader("Predicted vs actual risk by route (test month)")
    st.caption("These routes were predicted from the earlier month and compared with what really happened afterwards.")
    f1, f2, f3 = st.columns(3)
    classes = f1.multiselect("Predicted risk", RISK_NAMES, default=["high"])
    text = f2.text_input("Airport code contains", "", placeholder="e.g. ORD").upper().strip()
    minf = f3.slider("Minimum flights", 0, int(pred.flights.max()), 0)
    v = pred[pred.pred_xgb_graph.isin(classes) & (pred.flights >= minf)]
    if text:
        v = v[v.Origin.str.contains(text) | v.Dest.str.contains(text)]
    v = v.sort_values("p_high", ascending=False)
    st.write(f"**{len(v):,} routes**")
    route_table(v, "pred_xgb_graph", True)
    if len(v):
        st.markdown("**Look at one route**")
        opts = (v.Origin + " → " + v.Dest).head(300).tolist()
        pick = st.selectbox("Route", opts)
        row = v.iloc[opts.index(pick)]
        a, b, c = st.columns(3)
        a.metric("Predicted risk", BADGE[row.pred_xgb_graph]); b.metric("Actual risk", BADGE[row.true_class])
        c.metric("Chance of high risk", f"{row.p_high:.0%}")
        st.markdown("**Main factors behind this prediction** (a + pushes toward the predicted class):")
        for part in pretty_drivers(row.top_drivers).split("), "):
            st.markdown(f"- {part.rstrip(')')})")
        if row.flights < 60:
            st.warning("This route has few flights, so its rates are noisy and the prediction is less reliable.")

# --------------------------------------------------------------- Network map
with tabs[3]:
    st.subheader("Network map: predicted risk (test month)")
    route_map(pred, "pred_xgb_graph", coords, "test")

# ------------------------------------------------------------------ Forecast
with tabs[4]:
    fc, fm = data["forecast"], data["fmeta"]
    st.subheader("Forecast: next month's route risk")
    if fc is None:
        st.info("No forecast yet. Run  python src/forecast.py  (or python run_all.py) and refresh this page.")
    else:
        st.warning(f"This is a **forecast for {month_label(fm['forecast_month'])}**, made from {month_label(fm['feature_month'])} data. "
                   "That month's outcomes are not in the data, so it cannot be checked. "
                   "Use the test-month results in 'Model results' as the guide to how reliable it is.")
        c = st.columns(4)
        c[0].metric("Routes scored", f"{fm['routes_scored']:,}")
        for col, cls in zip(c[1:], ["high", "medium", "low"]):
            col.metric(f"Forecast {cls}", f"{int((fc.pred_class == cls).sum()):,}")
        diff = int((fc.pred_class != fc.pred_persistence).sum())
        st.caption(f"{diff:,} routes get a different class from simply repeating this month's risk.")
        g1, g2 = st.columns(2)
        fcl = g1.multiselect("Forecast risk", RISK_NAMES, default=["high"], key="fc_filter_cls")
        ftext = g2.text_input("Airport code contains", "", key="fc_filter_text", placeholder="e.g. JFK").upper().strip()
        v = fc[fc.pred_class.isin(fcl)]
        if ftext:
            v = v[v.Origin.str.contains(ftext) | v.Dest.str.contains(ftext)]
        st.write(f"**{len(v):,} routes**")
        route_table(v.sort_values("p_high", ascending=False), "pred_class", False)
        st.markdown("**Forecast map**")
        route_map(fc, "pred_class", coords, "fc")

# ------------------------------------------------------------- Hub shutdown
with tabs[5]:
    st.subheader("What if an airport shut down?")
    st.caption("A structural what-if on the route network: remove one airport, then measure lost routes and traffic, airports "
               "cut off, and longer detours. It does not forecast real delays or model rerouting.")
    months = sorted(stats.month.unique())
    month = st.selectbox("Network of month", months, index=len(months) - 1, format_func=month_label)
    G, am = month_graph(month, STAMP)
    names = coords["name"].to_dict()
    hub = st.selectbox("Airport to shut down", am.airport.tolist(), index=0, format_func=lambda c: f"{c} — {names.get(c, '')}")
    res = impact(G, hub)
    s = res["summary"]
    routes_before = G.number_of_edges()
    a, b, c, d = st.columns(4)
    a.metric("Routes in network", f"{routes_before - s['routes_removed']:,}", f"-{s['routes_removed']:,}", delta_color="off")
    b.metric("Share of all flights lost", f"{s['share_of_all_flights_lost']:.1%}")
    c.metric("Airports cut off", s["airports_cut_off"], delta_color="off")
    d.metric("Average hops between airports", f"{s['avg_hops_after']:.2f}", f"{s['avg_hops_after'] - s['avg_hops_before']:+.2f}",
             delta_color="inverse" if abs(s['avg_hops_after'] - s['avg_hops_before']) >= 0.005 else "off")
    st.write(f"Of {s['pairs_tested']} sampled airport pairs, **{s['pairs_with_no_path_after']}** lose every connection.")
    st.markdown("**Most exposed airports** (share of their flights that touched the shut-down airport)")
    tb = res["table"].head(25).copy()
    if len(tb):
        tb = pd.DataFrame({"Airport": tb.airport, "Name": tb.airport.map(names),
                           "Inbound flights lost (%)": (tb.inbound_flights_lost_share * 100).round(1),
                           "Outbound flights lost (%)": (tb.outbound_flights_lost_share * 100).round(1),
                           "Cut off from network": tb.cut_off_from_network.map({True: "Yes", False: "No"})})
        st.dataframe(tb, hide_index=True, width="stretch")
    removed = pd.DataFrame(res["removed_routes"], columns=["Origin", "Dest", "flights"])
    removed = add_coords(removed, coords) if len(removed) else removed
    if len(removed) and hub in coords.index:
        removed["tip"] = removed.Origin + " to " + removed.Dest + " | " + removed.flights.astype(int).astype(str) + " flights"
        hub_pt = coords.loc[[hub]].rename_axis("iata").reset_index(); hub_pt["tip"] = hub
        st.markdown("**Routes lost (red)**")
        st.pydeck_chart(deck([
            pdk.Layer("LineLayer", removed, get_source_position=["o_lon", "o_lat"], get_target_position=["d_lon", "d_lat"],
                      get_color=[210, 50, 50, 150], get_width=2, pickable=True),
            pdk.Layer("ScatterplotLayer", hub_pt, get_position=["lon", "lat"], get_radius=40000, get_fill_color=[210, 50, 50, 240], pickable=True)]))
    st.markdown("---")
    st.markdown("**Which airports hurt the network most if they close?**")
    if st.button("Rank the 15 busiest airports"):
        st.session_state["hub_rank"] = rank_hubs(G, 15)
    if "hub_rank" in st.session_state:
        hr = st.session_state["hub_rank"].copy()
        hr["flights_lost_pct"] = hr.share_of_all_flights_lost * 100
        st.altair_chart(alt.Chart(hr).mark_bar(color="#C0392B").encode(
            x=alt.X("flights_lost_pct:Q", title="Share of all flights lost (%)"), y=alt.Y("airport:N", sort="-x", title=None)).properties(height=380),
            width="stretch")
        st.dataframe(hr.rename(columns={"airport": "Airport", "share_of_all_flights_lost": "Share of flights lost",
                                        "airports_cut_off": "Airports cut off", "extra_hops": "Extra hops",
                                        "pairs_with_no_path": "Pairs with no path"}).drop(columns="flights_lost_pct"),
                     hide_index=True, width="stretch")
