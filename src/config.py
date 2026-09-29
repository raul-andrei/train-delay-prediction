"""Paths, date ranges and constants shared by the whole pipeline."""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"              # downloaded files, untouched
PROCESSED = ROOT / "data" / "processed"  # cleaned Parquet files and features
MODELS = ROOT / "models"                 # trained models
RESULTS = ROOT / "results"               # metrics and tables
FIGURES = ROOT / "figures"               # plots

# Data range
# August 2024 is only "warm-up": the history features look back 30 days,
# so the first arrivals we predict (1 Sep 2024) need August's data behind them.
FIRST_MONTH = "2024-08"
LAST_MONTH = "2026-08"

# Train / validation / test split (by local arrival date)
TRAIN_START = "2024-09-01"   # learn (15 months)
VAL_START = "2025-12-01"     # tune (3 months)
TEST_START = "2026-03-01"    # final test (6 months)
TEST_END = "2026-09-01"      # exclusive: up to and including 31 Aug

# Problem definition
HORIZON_MIN = 60             # predict at T0 = scheduled arrival - 60 minutes
DELAY_THRESHOLD_MIN = 5      # "late" = arrival delay of 5 minutes or more

RANDOM_STATE = 42            # fixed seed so random steps give the same result every run