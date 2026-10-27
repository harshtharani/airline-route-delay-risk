"""Hub-shutdown what-if on the airport network.

This is a STRUCTURAL what-if: it removes an airport from the route graph and measures what happens to
connectivity and traffic. It is not a forecast of real delays.
"""
import networkx as nx
import numpy as np
import pandas as pd
from config import SEED


def _pair_sample(nodes, n_pairs, rng):
    nodes = list(nodes)
    out = set()
    while len(out) < min(n_pairs, len(nodes) * (len(nodes) - 1) // 2):
        a, b = rng.choice(len(nodes), 2, replace=False)
        out.add((nodes[min(a, b)], nodes[max(a, b)]))
    return sorted(out)


def impact(G, hub, n_pairs=400):
    """What changes if `hub` is shut down?"""
    if hub not in G:
        raise ValueError(f"{hub} is not in the graph")
    rng = np.random.default_rng(SEED)
    H = G.copy(); H.remove_node(hub)
    U0, U1 = G.to_undirected(), H.to_undirected()

    total_flights = sum(d["flights"] for _, _, d in G.edges(data=True))
    lost_out = [(hub, v, d["flights"]) for _, v, d in G.out_edges(hub, data=True)]
    lost_in = [(u, hub, d["flights"]) for u, _, d in G.in_edges(hub, data=True)]
    lost_flights = sum(x[2] for x in lost_out + lost_in)

    # airports that lose connectivity to the main network
    lcc0 = max(nx.connected_components(U0), key=len)
    lcc1 = max(nx.connected_components(U1), key=len) if U1.number_of_nodes() else set()
    cut_off = sorted(set(U1.nodes) - lcc1)

    # how much each airport depends on the hub (share of its inbound / outbound flights that touched the hub)
    rows = []
    in_tot = dict(G.in_degree(weight="flights")); out_tot = dict(G.out_degree(weight="flights"))
    lost_in_by = {v: f for _, v, f in lost_out}      # hub -> v : v loses inbound flights
    lost_out_by = {u: f for u, _, f in lost_in}      # u -> hub : u loses outbound flights
    for n in H.nodes:
        li = lost_in_by.get(n, 0.0) / in_tot[n] if in_tot.get(n) else 0.0
        lo = lost_out_by.get(n, 0.0) / out_tot[n] if out_tot.get(n) else 0.0
        if li > 0 or lo > 0 or n in cut_off:
            rows.append({"airport": n, "inbound_flights_lost_share": round(li, 4),
                         "outbound_flights_lost_share": round(lo, 4),
                         "cut_off_from_network": n in cut_off, "exposure": round(max(li, lo), 4)})
    table = pd.DataFrame(rows).sort_values("exposure", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["airport", "inbound_flights_lost_share", "outbound_flights_lost_share", "cut_off_from_network", "exposure"])

    # detour cost: hops between random airport pairs, before vs after
    pool = [n for n in lcc0 if n != hub]
    pairs = _pair_sample(pool, n_pairs, rng)
    before, after, broken = [], [], 0
    for a, b in pairs:
        before.append(nx.shortest_path_length(U0, a, b))
        try:
            after.append(nx.shortest_path_length(U1, a, b))
        except nx.NetworkXNoPath:
            broken += 1
    both = min(len(before), len(after))
    summary = {
        "hub": hub,
        "routes_removed": len(lost_out) + len(lost_in),
        "share_of_all_flights_lost": lost_flights / total_flights if total_flights else 0.0,
        "airports_cut_off": len(cut_off),
        "largest_component_before": len(lcc0), "largest_component_after": len(lcc1),
        "avg_hops_before": float(np.mean(before)) if before else float("nan"),
        "avg_hops_after": float(np.mean(after)) if after else float("nan"),
        "pairs_with_no_path_after": broken, "pairs_tested": len(pairs),
    }
    return {"summary": summary, "table": table, "removed_routes": lost_out + lost_in}


def rank_hubs(G, top_n=15):
    """Run the shutdown for the busiest airports and rank them by the damage."""
    traffic = {n: G.in_degree(n, weight="flights") + G.out_degree(n, weight="flights") for n in G.nodes}
    hubs = sorted(traffic, key=traffic.get, reverse=True)[:top_n]
    rows = []
    for h in hubs:
        s = impact(G, h, n_pairs=150)["summary"]
        rows.append({"airport": h, "share_of_all_flights_lost": round(s["share_of_all_flights_lost"], 4),
                     "airports_cut_off": s["airports_cut_off"],
                     "extra_hops": round(s["avg_hops_after"] - s["avg_hops_before"], 3),
                     "pairs_with_no_path": s["pairs_with_no_path_after"]})
    return pd.DataFrame(rows).sort_values("share_of_all_flights_lost", ascending=False).reset_index(drop=True)
