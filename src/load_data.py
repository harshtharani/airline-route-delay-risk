"""Load and clean BTS On-Time Performance CSVs, then aggregate to monthly route stats."""
import glob
import pandas as pd
from config import RAW_BTS, PROCESSED, DELAY_THRESHOLD_MIN, MIN_FLIGHTS_PER_ROUTE

USECOLS = ["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime",
           "DepDelay", "ArrDelay", "Cancelled", "Diverted", "Distance"]


def load_flights():
    files = sorted(glob.glob(str(RAW_BTS / "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files in {RAW_BTS}. See data/README.md.")
    df = pd.concat((pd.read_csv(f, usecols=lambda c: c in USECOLS) for f in files), ignore_index=True)
    missing = set(USECOLS) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns {missing}. Download the 'Reporting Carrier On-Time Performance' table.")
    df["FlightDate"] = pd.to_datetime(df.FlightDate)
    df["month"] = df.FlightDate.dt.strftime("%Y-%m")
    df["cancelled"] = df.Cancelled.fillna(0).astype(int)
    df["delayed"] = (df.ArrDelay >= DELAY_THRESHOLD_MIN).astype(int)
    # a cancelled or diverted flight counts as disrupted for risk purposes
    df["disrupted"] = ((df.delayed == 1) | (df.cancelled == 1) | (df.Diverted.fillna(0) == 1)).astype(int)
    return df


def route_month_stats(df):
    g = df.groupby(["Origin", "Dest", "month"])
    out = g.agg(flights=("disrupted", "size"),
                disrupted_rate=("disrupted", "mean"),
                cancel_rate=("cancelled", "mean"),
                mean_arr_delay=("ArrDelay", "mean"),
                distance=("Distance", "median")).reset_index()
    return out[out.flights >= MIN_FLIGHTS_PER_ROUTE]


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    flights = load_flights()
    print("flights:", len(flights), "| months:", sorted(flights.month.unique()))
    print("disrupted share: %.3f | cancelled share: %.3f" % (flights.disrupted.mean(), flights.cancelled.mean()))
    stats = route_month_stats(flights)
    stats.to_csv(PROCESSED / "route_month_stats.csv", index=False)
    print("route-month rows:", len(stats))
