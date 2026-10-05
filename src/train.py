"""Baselines and models. Everything is chosen on validation. The test set is scored once, at the end."""
import json

import duckdb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROCESSED, RESULTS, RANDOM_STATE

# numeric features; the actual delay (arr_delay) is deliberately not in this list
NUMERIC = [
    "is_intercity", "stop_seq", "n_stops", "stops_remaining", "mins_since_service_start",
    "train_delay_now", "train_max_delay_now", "train_started", "mins_since_train_obs", "train_overdue_min",
    "net_arrivals_1h", "net_late_share_1h", "net_mean_delay_1h", "net_cancelled_1h",
    "stn_arrivals_1h", "stn_late_share_1h", "stn_cancelled_1h",
    "hist_rate_network", "hist_rate_station", "hist_rate_station_hour", "hist_rate_train_number",
    "hist_n_train_number",
    "temperature", "wind_speed", "wind_gust", "precip_mm", "precip_3h", "visibility_code",
    "snow", "thunder", "nl_max_gust",
]
CATEGORICAL = ["station", "hour", "weekday"]

# empty only when the train isn't running yet at T0; "no delay seen yet" -> 0 (train_started says why)
FILL_ZERO = ["train_delay_now", "train_max_delay_now", "mins_since_train_obs"]
FILL_MEDIAN = [c for c in NUMERIC if c not in FILL_ZERO]
LR_SAMPLE = 3_000_000


def load(split):
    cols = ", ".join(f"{c}::float as {c}" for c in NUMERIC)
    return duckdb.sql(f"""
        select row_id, label::tinyint as label, station, hour::tinyint as hour,
               weekday::tinyint as weekday, {cols}
        from '{PROCESSED}/features.parquet'
        where split = '{split}'
    """).df()


def metrics(y, p, probabilities=True):
    m = {"avg_precision": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)}
    if probabilities:
        m["brier"] = brier_score_loss(y, p)
        m["log_loss"] = log_loss(y, p)
    return m


def historical_frozen(train):
    # late rate per station x hour over the training months, smoothed towards the station rate
    g = train.groupby(["station", "hour"]).label.agg(["sum", "count"]).reset_index()
    station_rate = train.groupby("station").label.mean().rename("station_rate").reset_index()
    g = g.merge(station_rate, on="station")
    g["rate"] = (g["sum"] + 20 * g["station_rate"]) / (g["count"] + 20)
    overall = train.label.mean()

    def predict(df):
        out = df[["station", "hour"]].merge(g[["station", "hour", "rate"]], on=["station", "hour"], how="left")
        return out["rate"].fillna(overall).to_numpy()
    return predict


def historical_rolling(df):
    # the same idea, but over the last 28 days (lag 2) instead of fixed training months
    return df.hist_rate_station_hour.to_numpy()


def current_delay_rule(df):
    # what we know about the train's delay at T0; the history rate (always < 1) breaks ties
    known_delay = np.fmax(df.train_delay_now.fillna(0), df.train_overdue_min)
    return (known_delay + df.hist_rate_station_hour).to_numpy()


def logistic_regression(train):
    sample = train.sample(n=min(LR_SAMPLE, len(train)), random_state=RANDOM_STATE)
    pre = ColumnTransformer([
        ("zero", make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler()), FILL_ZERO),
        ("median", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), FILL_MEDIAN),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=1000))
    model.fit(sample, sample.label)
    return model


def main():
    RESULTS.mkdir(exist_ok=True)
    train, val = load("train"), load("val")
    print(f"train {len(train):,} rows, val {len(val):,} rows")

    frozen = historical_frozen(train)
    lr = logistic_regression(train)
    scorers = {
        "historical_frozen": frozen,
        "historical_rolling": historical_rolling,
        "current_delay_rule": current_delay_rule,
        "logistic_regression": lambda d: lr.predict_proba(d)[:, 1],
    }

    results = {}
    for name, score in scorers.items():
        # the rule's scores are not probabilities, so only the ranking metrics make sense for it
        m = metrics(val.label, score(val), probabilities=(name != "current_delay_rule"))
        results[name] = m
        print(f"{name:22s}", "  ".join(f"{k} {v:.4f}" for k, v in m.items()))

    with open(RESULTS / "baselines_val.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()