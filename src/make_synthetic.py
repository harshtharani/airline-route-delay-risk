"""Generate synthetic flights + weather in the real BTS / Open-Meteo formats.

ONLY for testing the pipeline when real data is not available. Never report results from this data.
Hubs have congestion, each route has a persistent delay offset, and storm days raise delays everywhere.
"""
import argparse
import numpy as np
import pandas as pd
from config import AIRPORTS_DAT, RAW_BTS, WEATHER_CSV, TRAIN_MONTHS, TEST_MONTHS


def us_airports(n=70, seed=0):
    cols = ["id", "name", "city", "country", "iata", "icao", "lat", "lon", "alt", "tz", "dst", "tzdb", "type", "source"]
    df = pd.read_csv(AIRPORTS_DAT, header=None, names=cols, na_values=["\\N"])
    df = df[(df.country == "United States") & df.iata.notna() & (df.type == "airport")]
    return df.sample(n, random_state=seed).reset_index(drop=True)


def main(flights_per_month=40000, seed=0):
    rng = np.random.default_rng(seed)
    ap = us_airports()
    codes = ap.iata.tolist()
    hubs = set(codes[:10])
    lat = dict(zip(ap.iata, ap.lat)); lon = dict(zip(ap.iata, ap.lon))
    pairs = []
    for c in codes:
        if c in hubs:
            pairs += [(c, d) for d in codes if d != c]
        else:
            for h in rng.choice(sorted(hubs), 3, replace=False):
                pairs += [(c, h), (h, c)]
    pairs = list(dict.fromkeys(pairs))
    w = np.array([5.0 if (o in hubs and d in hubs) else 1.0 for o, d in pairs]); w /= w.sum()
    route_offset = rng.normal(0, 5, len(pairs))          # persistent route-level risk
    airlines = ["AA", "DL", "UA", "WN", "B6"]
    RAW_BTS.mkdir(parents=True, exist_ok=True)
    weather = []
    for month in TRAIN_MONTHS + TEST_MONTHS:
        idx = rng.choice(len(pairs), flights_per_month, p=w)
        o = [pairs[i][0] for i in idx]; d = [pairs[i][1] for i in idx]
        day = rng.integers(1, 29, flights_per_month)
        storm = rng.random(28) < 0.12
        base = np.array([8.0 if (a in hubs) else 2.0 for a in o]) + np.array([6.0 if (b in hubs) else 1.0 for b in d])
        delay = (rng.gamma(1.5, 6.0, flights_per_month) + base + route_offset[idx] - 6
                 + storm[day - 1] * rng.gamma(2, 12, flights_per_month))
        cancelled = (rng.random(flights_per_month) < 0.015 + 0.05 * storm[day - 1]).astype(int)
        dist = [int(np.hypot(lat[a] - lat[b], lon[a] - lon[b]) * 60) + 100 for a, b in zip(o, d)]
        df = pd.DataFrame({
            "FlightDate": [f"{month}-{x:02d}" for x in day],
            "Reporting_Airline": rng.choice(airlines, flights_per_month),
            "Origin": o, "Dest": d,
            "CRSDepTime": rng.integers(500, 2300, flights_per_month),
            "DepDelay": np.round(delay, 0),
            "ArrDelay": np.round(delay + rng.normal(0, 5, flights_per_month), 0),
            "Cancelled": cancelled, "Diverted": 0, "Distance": dist})
        df.loc[df.Cancelled == 1, ["DepDelay", "ArrDelay"]] = np.nan
        df["ArrDel15"] = (df.ArrDelay >= 15).astype(float).where(df.ArrDelay.notna())
        df.to_csv(RAW_BTS / f"synthetic_{month}.csv", index=False)
        print(month, len(df))
        for a in codes[:40]:                               # weather for 40 airports
            for dd in range(1, 29):
                s = storm[dd - 1]
                weather.append((a, f"{month}-{dd:02d}", round(float(rng.gamma(2, 8) if s else rng.gamma(0.5, 1.5)), 1),
                                round(float(rng.normal(55, 8) if s else rng.normal(25, 8)), 1)))
    pd.DataFrame(weather, columns=["airport", "date", "precip_mm", "wind_max_kmh"]).to_csv(WEATHER_CSV, index=False)
    print("weather rows:", len(weather))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40000)
    main(ap.parse_args().n)
