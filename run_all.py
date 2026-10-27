"""Run the whole pipeline in order.   python run_all.py            (real data in data/raw/bts/)

Steps: clean flights -> weather -> graph metrics -> train and evaluate models.
Then:  python src/neo4j_load.py   (optional, needs Neo4j running)   and   streamlit run app.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = [("Cleaning flights and building route-month stats", "load_data.py"),
         ("Downloading weather (Open-Meteo)", "fetch_weather.py"),
         ("Graph metrics for the training months", "build_graph.py"),
         ("Training GCN + XGBoost models and evaluating", "train_eval.py"),
         ("Forecasting next month's route risk", "forecast.py")]

if __name__ == "__main__":
    skip_weather = "--skip-weather" in sys.argv
    for title, script in STEPS:
        if skip_weather and script == "fetch_weather.py":
            print("\n== skipping weather download (using existing data/raw/weather.csv) =="); continue
        print(f"\n== {title} ==", flush=True)
        r = subprocess.run([sys.executable, script], cwd=SRC)
        if r.returncode != 0:
            sys.exit(f"Step failed: {script}. Fix the error above, then run again.")
    print("\nDone. Now run:  streamlit run app.py")
