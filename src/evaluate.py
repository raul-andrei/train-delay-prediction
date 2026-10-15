"""Evaluate the saved test predictions: curves, calibration, SHAP, confidence intervals and errors by segment."""
import json
import warnings

import duckdb
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from src.config import PROCESSED, MODELS, RESULTS, FIGURES, RANDOM_STATE
from src.train import LGBM_FEATURES, load

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402

# one fixed colour per model, the same in every chart
SHOWN = {
    "lightgbm": ("LightGBM", "#2a78d6"),
    "logistic_regression": ("Logistic regression", "#eb6834"),
    "current_delay_rule": ("Current-delay rule", "#1baf7a"),
    "historical_frozen": ("Historical rate, frozen", "#eda100"),
    "historical_rolling": ("Historical rate, rolling", "#e87ba4"),
}
N_BOOT = 500
MIN_SEGMENT_ROWS = 2000

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.grid": True, "grid.color": "#e5e5e3", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8.5,
})


def load_test():
    test = load("test")
    extra = duckdb.sql(f"""
        select row_id, arr_delay, arr_local, train_type
        from '{PROCESSED}/features.parquet' where split = 'test'
    """).df()
    preds = pd.read_parquet(RESULTS / "test_predictions.parquet").drop(columns="label")
    return test.merge(extra, on="row_id").merge(preds, on="row_id")


def plot_curves(df, base_rate):
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (label, color) in SHOWN.items():
        precision, recall, _ = precision_recall_curve(df.label, df[name])
        step = max(1, len(recall) // 2000)  # millions of points look the same as 2000
        ap = average_precision_score(df.label, df[name])
        ax_pr.plot(recall[::step], precision[::step], color=color, label=f"{label} ({ap:.3f})")
        fpr, tpr, _ = roc_curve(df.label, df[name])
        ax_roc.plot(fpr, tpr, color=color, label=label)
    ax_pr.axhline(base_rate, color="#8a8984", linestyle="--", linewidth=1, label=f"Random ({base_rate:.3f})")
    ax_pr.set(xlabel="Recall (share of late trains caught)", ylabel="Precision (share of warnings that are right)",
              title="Precision-recall on test (average precision)", ylim=(0, 1))
    ax_roc.plot([0, 1], [0, 1], color="#8a8984", linestyle="--", linewidth=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC on test")
    ax_pr.legend()
    ax_roc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "pr_roc_curves.png")
    plt.close(fig)


def plot_calibration(df):
    # bins equally wide on a log scale, so the rare high-risk predictions get their own bins
    edges = np.geomspace(1e-3, 1, 19)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in ["lightgbm", "logistic_regression", "historical_frozen"]:
        label, color = SHOWN[name]
        bins = pd.cut(df[name].clip(1e-3, 1), edges, include_lowest=True)
        g = df.groupby(bins, observed=True).agg(pred=(name, "mean"), obs=("label", "mean"), n=("label", "size"))
        g = g[g.n >= 1000]  # skip bins too small to trust
        ax.plot(g.pred, g.obs, marker="o", markersize=5, color=color, label=label)
    ax.plot([1e-3, 1], [1e-3, 1], color="#8a8984", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.set(xscale="log", yscale="log", xlim=(3e-3, 1), ylim=(3e-3, 1),
           xlabel="Predicted probability", ylabel="Observed share late",
           title="Calibration on test (bins with ≥ 1,000 arrivals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png")
    plt.close(fig)


def shap_analysis(df):
    gbm = lgb.Booster(model_file=str(MODELS / "lgbm.txt"))
    sample = df.sample(n=min(20_000, len(df)), random_state=RANDOM_STATE)
    X = sample[LGBM_FEATURES]
    # LightGBM computes SHAP values itself; the last column is the baseline, not a feature
    contrib = gbm.predict(X, pred_contrib=True)[:, :-1]
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=LGBM_FEATURES).sort_values(ascending=False)
    importance.to_csv(RESULTS / "shap_importance.csv", header=["mean_abs_shap"])

    # station has no meaningful "high" or "low", so it's drawn grey instead of coloured by category code
    Xp = X.copy()
    Xp["station"] = np.nan
    plt.figure()
    with warnings.catch_warnings():  # shap warns about the all-empty station column; that's intended
        warnings.simplefilter("ignore", RuntimeWarning)
        shap.summary_plot(contrib, Xp.astype(float), max_display=15, show=False, plot_size=(8, 6))
    plt.title("What drives the predictions (SHAP, 20k test arrivals)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png")
    plt.close()

    # the station effect, shown per station instead
    st = pd.DataFrame({"station": sample.station.astype(str),
                       "shap": contrib[:, LGBM_FEATURES.index("station")]})
    st = st.groupby("station").shap.agg(["mean", "size"])
    st = st[st["size"] >= 30].sort_values("mean")
    shown = pd.concat([st.head(10), st.tail(10)])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(shown["mean"] > 0, "#eb6834", "#2a78d6")
    ax.barh(shown.index, shown["mean"], color=colors, edgecolor="#fcfcfb", linewidth=2)
    ax.axvline(0, color="#8a8984", linewidth=1)
    ax.set(xlabel="Average SHAP value of the station feature",
           title="Stations that raise (orange) or lower (blue) the risk")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_station.png")
    plt.close(fig)
    return importance


class WeightedAP:
    """Average precision with a weight per group (day), fast enough to repeat hundreds of times.
    The rows are sorted once; each bootstrap round only re-weights them."""

    def __init__(self, y, p, groups):
        order = np.argsort(-p, kind="mergesort")
        self.y = y[order].astype(np.float64)
        self.groups = groups[order]
        p_sorted = p[order]
        self.ends = np.r_[np.nonzero(np.diff(p_sorted))[0], len(p_sorted) - 1]  # last row of each tied score

    def __call__(self, group_weights):
        w = group_weights[self.groups]
        tp = np.cumsum(w * self.y)[self.ends]
        flagged = np.cumsum(w)[self.ends]
        precision = np.where(flagged > 0, tp / np.maximum(flagged, 1e-12), 0)
        recall = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def bootstrap(df, models, group_col, n_boot=N_BOOT):
    # resample whole groups (days), because arrivals on the same day share weather and disruptions
    groups, uniques = pd.factorize(df[group_col])
    y = df.label.to_numpy()
    ap = {m: WeightedAP(y, df[m].to_numpy(), groups) for m in models}
    rng = np.random.default_rng(RANDOM_STATE)
    draws = []
    for _ in range(n_boot):
        w = rng.multinomial(len(uniques), np.full(len(uniques), 1 / len(uniques))).astype(np.float64)
        draws.append({m: ap[m](w) for m in models})
    return pd.DataFrame(draws)


def confidence_intervals(df, draws):
    rows = {}
    for m in draws.columns:
        rows[m] = (average_precision_score(df.label, df[m]), *np.percentile(draws[m], [2.5, 97.5]))
    for a, b in [("lightgbm", "logistic_regression"), ("lightgbm", "current_delay_rule"),
                 ("lightgbm", "historical_frozen"), ("lightgbm", "lightgbm_train_only")]:
        if b not in draws:
            continue
        diff = draws[a] - draws[b]
        rows[f"{a} - {b}"] = (rows[a][0] - rows[b][0], *np.percentile(diff, [2.5, 97.5]))
    return pd.DataFrame(rows, index=["avg_precision", "ci_low", "ci_high"]).T


def segment_table(df, col, flagged):
    rows = []
    for key, g in df.groupby(col, observed=True):
        if len(g) < MIN_SEGMENT_ROWS or g.label.nunique() < 2:
            continue
        f = flagged.loc[g.index]
        hits = (f & (g.label == 1)).sum()
        rows.append({col: key, "rows": len(g), "late_rate": g.label.mean(),
                     "avg_precision_lgbm": average_precision_score(g.label, g.lightgbm),
                     "avg_precision_frozen": average_precision_score(g.label, g.historical_frozen),
                     "precision": hits / max(f.sum(), 1), "recall": hits / g.label.sum()})
    return pd.DataFrame(rows)


def error_analysis(df, flag_share):
    df = df.copy()
    flagged = df.lightgbm >= np.quantile(df.lightgbm, 1 - flag_share)
    df["train_at_t0"] = np.where(df.train_started == 1, "running", "not running yet")
    df["overdue_at_t0"] = np.where(df.train_overdue_min > 0, "overdue", "not overdue")
    df["time_of_day"] = pd.cut(df.hour, [-1, 5, 9, 15, 19, 23],
                               labels=["night 0-5", "morning peak 6-9", "day 10-15", "evening peak 16-19", "late 20-23"])
    df["month"] = df.arr_local.dt.strftime("%Y-%m")
    df["gust"] = pd.cut(df.wind_gust, [-1, 10, 15, 20, 100], labels=["< 10 m/s", "10-15", "15-20", ">= 20"])

    tables = {c: segment_table(df, c, flagged)
              for c in ["train_at_t0", "overdue_at_t0", "time_of_day", "train_type", "month", "gust"]}
    late = df[df.label == 1]
    severity = pd.cut(late.arr_delay, [4, 9, 19, 10_000], labels=["5-9 min", "10-19 min", "20+ min"])
    tables["delay_size"] = (flagged.loc[late.index].groupby(severity, observed=True)
                            .agg(late_rows="size", recall="mean").reset_index(names="delay_size"))

    missed = late[~flagged.loc[late.index]]
    summary = {
        "flag_share": flag_share,
        "late_share_not_running": float((late.train_started == 0).mean()),
        "missed_share_not_running": float((missed.train_started == 0).mean()),
    }
    with open(RESULTS / "error_analysis.md", "w") as f:
        f.write(f"Alarm: the riskiest {flag_share:.1%} of arrivals are flagged (share chosen on validation).\n\n")
        f.write(f"{summary['late_share_not_running']:.0%} of late arrivals belong to trains not running yet at T0; "
                f"among the late arrivals the model misses, it is {summary['missed_share_not_running']:.0%}.\n\n")
        for name, t in tables.items():
            f.write(f"## By {name}\n\n{t.to_markdown(index=False, floatfmt='.3f')}\n\n")

    fig, (ax_m, ax_t) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    for ax, t, col in [(ax_m, tables["month"], "month"), (ax_t, tables["time_of_day"], "time_of_day")]:
        x = np.arange(len(t))
        ax.bar(x - 0.2, t.avg_precision_lgbm, width=0.4, color=SHOWN["lightgbm"][1], label="LightGBM",
               edgecolor="#fcfcfb", linewidth=2)
        ax.bar(x + 0.2, t.avg_precision_frozen, width=0.4, color=SHOWN["historical_frozen"][1],
               label="Historical rate, frozen", edgecolor="#fcfcfb", linewidth=2)
        ax.set_xticks(x, t[col].astype(str), rotation=30 if col == "time_of_day" else 0, ha="right" if col == "time_of_day" else "center")
        ax.set_ylabel("Average precision")
    ax_m.set_title("By test month")
    ax_t.set_title("By time of day")
    ax_m.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "error_by_segment.png")
    plt.close(fig)
    return summary, tables


def main():
    FIGURES.mkdir(exist_ok=True)
    with open(RESULTS / "metrics.json") as f:
        metrics = json.load(f)
    df = load_test()
    base_rate = df.label.mean()
    print(f"test {len(df):,} rows, late rate {base_rate:.4f}")

    plot_curves(df, base_rate)
    plot_calibration(df)
    importance = shap_analysis(df)
    print("top features (mean |SHAP|):\n", importance.head(10).round(3).to_string())

    models = list(SHOWN) + ["lightgbm_train_only"]
    df["day"] = df.arr_local.dt.date
    df["week"] = df.arr_local.dt.strftime("%G-%V")
    ci_day = confidence_intervals(df, bootstrap(df, models, "day"))
    ci_week = confidence_intervals(df, bootstrap(df, ["lightgbm", "logistic_regression"], "week"))
    print("95% intervals, resampling days:\n", ci_day.round(4).to_string())
    print("check, resampling weeks:\n", ci_week.round(4).to_string())

    summary, tables = error_analysis(df, metrics["settings"]["flag_share"])
    for name, t in tables.items():
        print(f"\n{name}\n", t.round(3).to_string(index=False))

    out = {"base_rate": base_rate, "ci_by_day": ci_day.to_dict(orient="index"),
           "ci_by_week": ci_week.to_dict(orient="index"), "errors": summary}
    with open(RESULTS / "evaluation.json", "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()