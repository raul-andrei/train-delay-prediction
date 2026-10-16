"""Leakage tests: no feature may change when we change what happens after the prediction moment."""
import numpy as np
import pandas as pd
import pytest

import src.features as features
from src.config import HORIZON_MIN
from src.train import LGBM_FEATURES

# several cutoffs (UTC) at odd minutes, so that hourly data can't hide a look-ahead just past one of them
CUTOFFS = ["2024-09-05 12:50", "2024-09-06 07:58", "2024-09-07 18:07"]
STATIONS = ["AA", "BB", "CC", "DD"]


def make_data(folder, seed=0):
    """A small fake network: trains every 20 minutes on the line AA -> BB -> CC -> DD,
    with random delays and some cancelled stops, plus hourly weather."""
    rng = np.random.default_rng(seed)
    rows, service_id, stop_id = [], 0, 0
    for day in pd.date_range("2024-08-01", "2024-09-10"):
        for slot, start in enumerate(pd.date_range(day + pd.Timedelta("5h"), day + pd.Timedelta("22h"), freq="20min")):
            service_id += 1
            delay = int(rng.choice([0, 0, 0, 1, 3, 6, 12]))
            for seq, (station, arr_min, dep_min) in enumerate(
                    [("AA", None, 0), ("BB", 25, 27), ("CC", 50, 52), ("DD", 75, None)], start=1):
                stop_id += 1
                delay = max(0, delay + int(rng.choice([-1, 0, 0, 1, 2, 5])))
                arr = start + pd.Timedelta(minutes=arr_min) if arr_min is not None else pd.NaT
                dep = start + pd.Timedelta(minutes=dep_min) if dep_min is not None else pd.NaT
                rows.append({
                    "service_id": service_id, "service_date": day.date(), "train_type": "Intercity",
                    "train_number": 1000 + slot, "stop_id": stop_id, "station": station,
                    "arr_sched": arr, "arr_delay": delay if arr_min is not None else None,
                    "arr_cancelled": bool(rng.random() < 0.03) if arr_min is not None else None,
                    "dep_sched": dep, "dep_delay": delay if dep_min is not None else None,
                    "dep_cancelled": False if dep_min is not None else None,
                    "stop_seq": seq, "n_stops": 4, "service_start": start,
                })
    stops = pd.DataFrame(rows)
    stops["arr_delay"] = stops.arr_delay.astype("Int64")
    stops["dep_delay"] = stops.dep_delay.astype("Int64")

    hours = pd.date_range("2024-08-01 01:00", "2024-09-11 00:00", freq="h")
    weather = pd.DataFrame([
        {"knmi_station": k, "obs_time": t, "temperature": rng.normal(15, 5), "wind_speed": rng.gamma(2, 2),
         "wind_gust": rng.gamma(3, 3), "precip_mm": rng.exponential(0.3), "precip_hours": rng.random(),
         "visibility_code": int(rng.integers(40, 80)), "rain": 0, "snow": 0, "thunder": 0}
        for k in [1, 2] for t in hours])
    station_map = pd.DataFrame({"station": STATIONS, "station_name": STATIONS, "lat": [52.0] * 4,
                                "lon": [5.0] * 4, "knmi_station": [1, 1, 2, 2], "knmi_dist_km": [5.0] * 4})
    return stops, weather, station_map


def change_the_future(stops, weather, cutoff, seed=1):
    """Randomly change everything that happens after the cutoff: delays of arrivals and departures
    that happen after it (keeping them after it), and weather hours that end after it."""
    rng = np.random.default_rng(seed)
    stops, weather = stops.copy(), weather.copy()
    for kind in ["arr", "dep"]:
        sched, delay = stops[f"{kind}_sched"], stops[f"{kind}_delay"]
        happens_after = (sched + pd.to_timedelta(delay.astype(float), unit="min")) >= cutoff
        # a new delay that still puts the event after the cutoff
        min_delay = np.ceil(((cutoff - sched) / pd.Timedelta("1min")).clip(lower=0).fillna(0)).astype(int)
        new_delay = np.maximum(rng.integers(0, 30, len(stops)), min_delay)
        stops.loc[happens_after, f"{kind}_delay"] = new_delay[happens_after.to_numpy()]
    later = weather.obs_time > cutoff
    for col in ["temperature", "wind_speed", "wind_gust", "precip_mm"]:
        weather.loc[later, col] = rng.random(later.sum()) * 50
    return stops, weather


def build_features(folder, stops, weather, station_map, monkeypatch):
    folder.mkdir()
    stops.to_parquet(folder / "stops.parquet")
    weather.to_parquet(folder / "weather.parquet")
    station_map.to_parquet(folder / "station_map.parquet")
    monkeypatch.setattr(features, "PROCESSED", folder)
    features.main()
    return pd.read_parquet(folder / "features.parquet")


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_features_do_not_use_the_future(cutoff, tmp_path, monkeypatch):
    cutoff = pd.Timestamp(cutoff)
    stops, weather, station_map = make_data(tmp_path)
    original = build_features(tmp_path / "original", stops, weather, station_map, monkeypatch)
    stops2, weather2 = change_the_future(stops, weather, cutoff)
    changed = build_features(tmp_path / "changed", stops2, weather2, station_map, monkeypatch)

    # targets whose prediction moment T0 is before the cutoff (arr_local is local time, the cutoff is UTC)
    t0_cutoff_local = (cutoff + pd.Timedelta(minutes=HORIZON_MIN)).tz_localize("UTC") \
        .tz_convert("Europe/Amsterdam").tz_localize(None)
    numeric = [c for c in LGBM_FEATURES if c != "station"]
    a = original.set_index("row_id")
    b = changed.set_index("row_id")
    assert a.index.equals(b.index)
    before = a.arr_local < t0_cutoff_local

    # the test must not pass trivially: there are rows on both sides, and the change does reach later rows
    assert before.sum() > 200 and (~before).sum() > 200
    assert a.loc[before, "train_started"].sum() > 0
    assert not np.allclose(a.loc[~before, numeric].astype(float), b.loc[~before, numeric].astype(float),
                           equal_nan=True)

    # the actual check: every feature known at T0 is identical
    pd.testing.assert_frame_equal(a.loc[before, LGBM_FEATURES], b.loc[before, LGBM_FEATURES])


def test_answer_is_not_a_feature():
    # columns that contain (or come from) the answer must never be given to the model
    for col in ["label", "arr_delay", "arr_local", "row_id", "service_id"]:
        assert col not in LGBM_FEATURES