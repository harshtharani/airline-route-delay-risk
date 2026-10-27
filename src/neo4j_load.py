"""Load the train-period graph into Neo4j. Run after build_graph.py.

Usage:  NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=... python neo4j_load.py
"""
import os
import pandas as pd
from neo4j import GraphDatabase
from config import PROCESSED, OUTPUTS, AIRPORTS_DAT

CYPHER_AIRPORTS = """
UNWIND $rows AS r
MERGE (a:Airport {code: r.airport})
SET a.name = r.name, a.lat = r.lat, a.lon = r.lon,
    a.pagerank = r.pagerank, a.betweenness = r.betweenness,
    a.in_degree = r.in_degree, a.out_degree = r.out_degree,
    a.is_articulation_point = r.is_articulation_point,
    a.top_origin_share = r.top_origin_share
"""
CYPHER_ROUTES = """
UNWIND $rows AS r
MATCH (o:Airport {code: r.Origin}), (d:Airport {code: r.Dest})
MERGE (o)-[x:ROUTE]->(d)
SET x.flights = r.flights, x.disrupted_rate = r.disrupted_rate, x.distance = r.distance
"""


def chunks(rows, n=1000):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def main():
    cols = ["id", "name", "city", "country", "iata", "icao", "lat", "lon", "alt", "tz", "dst", "tzdb", "type", "source"]
    ap = pd.read_csv(AIRPORTS_DAT, header=None, names=cols, na_values=["\\N"])[["iata", "name", "lat", "lon"]].dropna(subset=["iata"])
    m = pd.read_csv(PROCESSED / "airport_metrics.csv").merge(ap, left_on="airport", right_on="iata", how="left")
    m = m.drop(columns="iata").astype(object).where(m.notna(), None)
    routes = pd.read_csv(PROCESSED / "routes_train.csv")
    driver = GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                                  auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]))
    with driver.session() as s:
        s.run("CREATE CONSTRAINT airport_code IF NOT EXISTS FOR (a:Airport) REQUIRE a.code IS UNIQUE")
        for c in chunks(m.to_dict("records")):
            s.run(CYPHER_AIRPORTS, rows=c)
        for c in chunks(routes.to_dict("records")):
            s.run(CYPHER_ROUTES, rows=c)
        # attach the model's predicted risk to each route (run train_eval.py first)
        pred_file = OUTPUTS / "predictions_test.csv"
        if pred_file.exists():
            p = pd.read_csv(pred_file)[["Origin", "Dest", "pred_xgb_graph", "p_high"]].rename(columns={"pred_xgb_graph": "risk_pred"})
            for c in chunks(p.to_dict("records")):
                s.run("UNWIND $rows AS r MATCH (:Airport {code: r.Origin})-[x:ROUTE]->(:Airport {code: r.Dest}) "
                      "SET x.risk_pred = r.risk_pred, x.p_high = r.p_high", rows=c)
            print("Attached predicted risk to routes from", pred_file.name)
        else:
            print("WARNING: predictions file not found at", pred_file, "- run python run_all.py first")
        n = s.run("MATCH (a:Airport) RETURN count(a) AS n").single()["n"]
        r = s.run("MATCH ()-[x:ROUTE]->() RETURN count(x) AS n").single()["n"]
    print(f"Loaded {n} airports and {r} routes into Neo4j")
    driver.close()


if __name__ == "__main__":
    main()
