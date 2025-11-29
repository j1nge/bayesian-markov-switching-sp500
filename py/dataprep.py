import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ROOT = path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "SPX.csv" # Kaggle imported
OUTPUT_PATH = ROOT / "sp500_returns.csv"

def main():

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])

    

