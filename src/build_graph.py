"""Build the airport-route graph for a set of months and compute network metrics."""
import networkx as nx
import numpy as np
import pandas as pd
from config import PROCESSED, TRAIN_MONTHS


def build_graph(stats, months=None):
    """Directed graph: airports are nodes, routes are edges, aggregated over `months`."""
    months = months or TRAIN_MONTHS
    sel = stats[stats.month.isin(months)]
    if sel.empty:
        raise SystemExit(f"No route data for months {months}. Check TRAIN_MONTHS/TEST_MONTHS in config.py.")
    sel = sel.assign(w_dis=sel.disrupted_rate * sel.flights)
    agg = sel.groupby(["Origin", "Dest"]).agg(flights=("flights", "sum"), w_dis=("w_dis", "sum"),
                                              distance=("distance", "median")).reset_index()
    agg["disrupted_rate"] = agg.w_dis / agg.flights
    agg = agg.drop(columns="w_dis")
    G = nx.DiGraph()
    for r in agg.itertuples():
        G.add_edge(r.Origin, r.Dest, flights=float(r.flights), disrupted_rate=float(r.disrupted_rate),
                   distance=float(r.distance), cost=1.0 / float(r.flights))   # busier route = shorter "distance"
    return G, agg


def airport_metrics(G):
    U = G.to_undirected()
    pr = nx.pagerank(G, weight="flights")
    btw = nx.betweenness_centrality(G, weight="cost", normalized=True)
    art = set(nx.articulation_points(U))
    clustering = nx.clustering(U)
    m = pd.DataFrame({"airport": list(G.nodes)})
    m["in_degree"] = m.airport.map(dict(G.in_degree()))
    m["out_degree"] = m.airport.map(dict(G.out_degree()))
    m["in_flights"] = m.airport.map(dict(G.in_degree(weight="flights")))
    m["out_flights"] = m.airport.map(dict(G.out_degree(weight="flights")))
    m["pagerank"] = m.airport.map(pr)
    m["betweenness"] = m.airport.map(btw)
    m["clustering"] = m.airport.map(clustering)
    m["is_articulation_point"] = m.airport.isin(art)
    # dependency: share of an airport's inbound flights coming from its single biggest origin
    top_share = {}
    for n in G.nodes:
        w = [d["flights"] for _, _, d in G.in_edges(n, data=True)]
        top_share[n] = max(w) / sum(w) if w else 0.0
    m["top_origin_share"] = m.airport.map(top_share)
    # airport-level disruption rates (flight-weighted) as outbound and inbound exposure
    out_rate = {n: np.average([d["disrupted_rate"] for _, _, d in G.out_edges(n, data=True)],
                              weights=[d["flights"] for _, _, d in G.out_edges(n, data=True)]) if G.out_degree(n) else np.nan
                for n in G.nodes}
    in_rate = {n: np.average([d["disrupted_rate"] for _, _, d in G.in_edges(n, data=True)],
                             weights=[d["flights"] for _, _, d in G.in_edges(n, data=True)]) if G.in_degree(n) else np.nan
               for n in G.nodes}
    m["out_disrupted_rate"] = m.airport.map(out_rate)
    m["in_disrupted_rate"] = m.airport.map(in_rate)
    return m.sort_values("pagerank", ascending=False).reset_index(drop=True)


def bridge_routes(G):
    U = G.to_undirected()
    return pd.DataFrame(list(nx.bridges(U)), columns=["a", "b"])


if __name__ == "__main__":
    stats = pd.read_csv(PROCESSED / "route_month_stats.csv")
    G, routes = build_graph(stats)
    print(f"train graph ({', '.join(TRAIN_MONTHS)}): {G.number_of_nodes()} airports, {G.number_of_edges()} routes")
    m = airport_metrics(G)
    m.to_csv(PROCESSED / "airport_metrics.csv", index=False)
    routes.to_csv(PROCESSED / "routes_train.csv", index=False)
    br = bridge_routes(G)
    br.to_csv(PROCESSED / "bridge_routes.csv", index=False)
    print("articulation points:", int(m.is_articulation_point.sum()), "| bridge routes:", len(br))
    print(m.head(8)[["airport", "in_degree", "out_degree", "pagerank", "betweenness", "top_origin_share"]].to_string(index=False))
