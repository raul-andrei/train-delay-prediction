"""Baselines and models. Everything is chosen on validation. The test set is scored once, at the end."""
import json

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROCESSED, MODELS, RESULTS, RANDOM_STATE

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

LGBM_FEATURES = NUMERIC + CATEGORICAL
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "cat_smooth": 50,
    "seed": RANDOM_STATE,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
# the small search: mainly how complex each tree may be
SEARCH = [
    {"num_leaves": 63, "min_data_in_leaf": 500},
    {"num_leaves": 127, "min_data_in_leaf": 500},
    {"num_leaves": 255, "min_data_in_leaf": 500},
    {"num_leaves": 127, "min_data_in_leaf": 2000},
]


def load(split):
    cols = ", ".join(f"{c}::float as {c}" for c in NUMERIC)
    df = duckdb.sql(f"""
        select row_id, label::tinyint as label, station, hour::tinyint as hour,
               weekday::tinyint as weekday, {cols}
        from '{PROCESSED}/features.parquet'
        where split = '{split}'
    """).df()
    # same list of station categories for every split, so the codes always mean the same station
    stations = sorted(duckdb.sql(f"select station from '{PROCESSED}/station_map.parquet'").df().station)
    df["station"] = pd.Categorical(df.station, categories=stations)
    return df


def metrics(y, p, probabilities=True):
    m = {"avg_precision": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)}
    if probabilities:
        m["brier"] = brier_score_loss(y, p)
        m["log_loss"] = log_loss(y, p)
    return m


def historical_frozen(train):
    # late rate per station x hour over the training months, smoothed towards the station rate
    g = train.groupby(["station", "hour"], observed=True).label.agg(["sum", "count"]).reset_index()
    station_rate = train.groupby("station", observed=True).label.mean().rename("station_rate").reset_index()
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


def fit_lgbm(train, params, features, val=None, num_rounds=3000):
    categorical = [c for c in ["station"] if c in features]
    dtrain = lgb.Dataset(train[features], train.label, categorical_feature=categorical)
    if val is None:
        # refit: no validation set, so a fixed number of trees
        return lgb.train(params, dtrain, num_boost_round=num_rounds)
    dval = lgb.Dataset(val[features], val.label, reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=[dval], valid_names=["val"],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])


def flag_share_at_best_f1(y, p):
    # share of arrivals flagged at the cut-off with the best F1
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = np.argmax(f1[:-1])
    return float((p >= thresholds[best]).mean())


def precision_recall_at_share(y, p, share):
    flagged = p >= np.quantile(p, 1 - share)
    hits = (flagged & (y == 1)).sum()
    return {"precision": hits / flagged.sum(), "recall": hits / (y == 1).sum()}


def main():
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    train, val = load("train"), load("val")
    print(f"train {len(train):,} rows, val {len(val):,} rows")
    results = {"val": {}, "test": {}}

    # 1. baselines, scored on validation
    frozen = historical_frozen(train)
    lr = logistic_regression(train)
    baselines = {
        "historical_frozen": frozen,
        "historical_rolling": historical_rolling,
        "current_delay_rule": current_delay_rule,
        "logistic_regression": lambda d: lr.predict_proba(d)[:, 1],
    }
    for name, score in baselines.items():
        # the rule's scores are not probabilities, so only the ranking metrics make sense for it
        results["val"][name] = metrics(val.label, score(val), probabilities=(name != "current_delay_rule"))
        print(f"{name:22s}", results["val"][name])

    # 2. LightGBM: small search, early stopping on validation
    search, best = [], None
    for settings in SEARCH:
        gbm = fit_lgbm(train, {**LGBM_PARAMS, **settings}, LGBM_FEATURES, val)
        score = gbm.best_score["val"]["average_precision"]
        search.append({**settings, "best_iteration": gbm.best_iteration, "val_avg_precision": score})
        print(search[-1])
        if best is None or score > best[0]:
            best = (score, settings, gbm)
    _, best_settings, gbm = best
    params = {**LGBM_PARAMS, **best_settings}
    p_val = gbm.predict(val[LGBM_FEATURES], num_iteration=gbm.best_iteration)
    results["val"]["lightgbm"] = metrics(val.label, p_val)

    # 3. ablation: retrain without one feature, on validation only
    ablation = {}
    for dropped in ["train_overdue_min", "station"]:
        features = [f for f in LGBM_FEATURES if f != dropped]
        m = fit_lgbm(train, params, features, val)
        ablation[f"without_{dropped}"] = m.best_score["val"]["average_precision"]
        print(f"without {dropped}: {ablation[f'without_{dropped}']:.4f}")

    # 4. alarm threshold, chosen on validation: flag the same share of arrivals on test
    flag_share = flag_share_at_best_f1(val.label.to_numpy(), p_val)

    # 5. refit on train + validation, with ~20% more trees for ~20% more data
    trainval = pd.concat([train, val], ignore_index=True)
    n_rounds = round(gbm.best_iteration * len(trainval) / len(train))
    del train, val
    gbm_final = fit_lgbm(trainval, params, LGBM_FEATURES, num_rounds=n_rounds)
    frozen_tv = historical_frozen(trainval)
    lr_tv = logistic_regression(trainval)
    del trainval

    # 6. the test set, scored once
    test = load("test")
    final = {
        "historical_frozen": frozen_tv,
        "historical_rolling": historical_rolling,
        "current_delay_rule": current_delay_rule,
        "logistic_regression": lambda d: lr_tv.predict_proba(d)[:, 1],
        "lightgbm_train_only": lambda d: gbm.predict(d[LGBM_FEATURES], num_iteration=gbm.best_iteration),
        "lightgbm": lambda d: gbm_final.predict(d[LGBM_FEATURES]),
    }
    preds = test[["row_id", "label"]].copy()
    y = test.label.to_numpy()
    for name, score in final.items():
        preds[name] = score(test)
        results["test"][name] = metrics(y, preds[name], probabilities=(name != "current_delay_rule"))
        if name.startswith("lightgbm"):
            results["test"][name].update(precision_recall_at_share(y, preds[name].to_numpy(), flag_share))
        print(f"TEST {name:22s}", results["test"][name])

    results["settings"] = {"best": best_settings, "search": search, "ablation_val_avg_precision": ablation,
                           "flag_share": flag_share, "refit_rounds": n_rounds,
                           "late_rate": {"test": float(y.mean())}}
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    preds.to_parquet(RESULTS / "test_predictions.parquet", index=False)
    gbm_final.save_model(str(MODELS / "lgbm.txt"))
    gbm.save_model(str(MODELS / "lgbm_train_only.txt"))


if __name__ == "__main__":
    main()