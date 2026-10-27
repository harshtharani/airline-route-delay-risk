"""Project settings. Edit the months to match the BTS files you downloaded."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_BTS = ROOT / "data" / "raw" / "bts"          # put BTS monthly CSVs here
AIRPORTS_DAT = ROOT / "data" / "raw" / "airports.dat"
WEATHER_CSV = ROOT / "data" / "raw" / "weather.csv"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

# Months are "YYYY-MM", in time order.
# Forecasting setup: features from month t predict the risk class in month t+1.
#   With TRAIN_MONTHS = [Jan, Feb] and TEST_MONTHS = [Mar]:
#     training pair: features Jan  -> label Feb
#     test pair:     features Feb  -> label Mar   (Mar outcomes are never seen in training)
TRAIN_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"]
TEST_MONTHS = ["2026-08"]

DELAY_THRESHOLD_MIN = 15      # BTS definition of a delayed arrival
MIN_FLIGHTS_PER_ROUTE = 30    # drop very thin routes (noisy rates)

# Risk classes come from the TRAINING label-month disruption rates only (terciles),
# and the same cut-offs are then applied to the test month.
RISK_QUANTILES = (1 / 3, 2 / 3)
RISK_NAMES = ["low", "medium", "high"]

# GCN (unsupervised, link-prediction objective, trained on the first training month's graph)
EMBED_DIM = 8
GCN_HIDDEN = 32
GCN_EPOCHS = 200
GCN_LR = 0.01

# Weather download (Open-Meteo, free, no key). Only the busiest airports to keep it small.
TOP_WEATHER_AIRPORTS = 120
STORM_PRECIP_MM = 10.0        # a "stormy" day: precipitation at or above this
STORM_WIND_KMH = 50.0         # or wind gusts/max at or above this

SEED = 42


def check_consecutive(months):
    """Month pairs are (features month t -> label month t+1), so the months must be consecutive calendar months."""
    import pandas as pd
    for a, b in zip(months[:-1], months[1:]):
        if pd.Period(a, "M") + 1 != pd.Period(b, "M"):
            raise SystemExit(f"TRAIN_MONTHS + TEST_MONTHS must be consecutive calendar months, but {a} is followed by {b}. "
                             "Fix them in src/config.py (and download the missing months).")
