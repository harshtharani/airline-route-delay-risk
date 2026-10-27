"""Download daily weather for the busiest airports from Open-Meteo (free, no API key).

Run this on your own PC (it needs internet access to api/archive-api.open-meteo.com):
    python fetch_weather.py
Writes data/raw/weather.csv with columns: airport, date, precip_mm, wind_max_kmh
"""
import time
import pandas as pd
import requests
from config import (AIRPORTS_DAT, PROCESSED, WEATHER_CSV, TOP_WEATHER_AIRPORTS, TRAIN_MONTHS, TEST_MONTHS)

URL = "https://archive-api.open-meteo.com/v1/archive"


def airport_coords():
    cols = ["id", "name", "city", "country", "iata", "icao", "lat", "lon", "alt", "tz", "dst", "tzdb", "type", "source"]
    df = pd.read_csv(AIRPORTS_DAT, header=None, names=cols, na_values=["\\N"])
    return df.dropna(subset=["iata"]).drop_duplicates("iata").set_index("iata")[["lat", "lon"]]


def fetch_one(lat, lon, start, end, tries=4):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "daily": "precipitation_sum,wind_speed_10m_max", "timezone": "auto"}
    for k in range(tries):
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()["daily"]
        time.sleep(5 * (k + 1))     # rate limited or temporary error: wait and retry
    r.raise_for_status()


def main():
    stats = pd.read_csv(PROCESSED / "route_month_stats.csv")
    months = sorted(TRAIN_MONTHS + TEST_MONTHS)
    stats = stats[stats.month.isin(months)]
    traffic = pd.concat([stats.groupby("Origin").flights.sum(), stats.groupby("Dest").flights.sum()], axis=1).fillna(0).sum(axis=1)
    top = traffic.sort_values(ascending=False).head(TOP_WEATHER_AIRPORTS).index.tolist()
    coords = airport_coords()
    start = pd.Timestamp(months[0] + "-01").strftime("%Y-%m-%d")
    end = (pd.Timestamp(months[-1] + "-01") + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    frames = []
    for i, code in enumerate(top, 1):
        if code not in coords.index:
            print(f"[{i}/{len(top)}] {code}: no coordinates, skipped"); continue
        try:
            d = fetch_one(coords.loc[code, "lat"], coords.loc[code, "lon"], start, end)
        except Exception as e:
            print(f"[{i}/{len(top)}] {code}: failed ({e}), skipped"); continue
        frames.append(pd.DataFrame({"airport": code, "date": d["time"], "precip_mm": d["precipitation_sum"],
                                    "wind_max_kmh": d["wind_speed_10m_max"]}))
        print(f"[{i}/{len(top)}] {code} ok")
        time.sleep(0.5)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(WEATHER_CSV, index=False)
    print(f"saved {len(out)} rows for {out.airport.nunique()} airports to {WEATHER_CSV}")


if __name__ == "__main__":
    main()
