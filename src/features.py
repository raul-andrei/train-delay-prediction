"""One row per scheduled NS arrival, with features that only use information available at
T0 = scheduled arrival - HORIZON_MIN. All timestamps are UTC."""
import duckdb

from src.config import (PROCESSED, TRAIN_START, VAL_START, TEST_START, TEST_END,
                        HORIZON_MIN, DELAY_THRESHOLD_MIN)

HIST_DAYS = 28        # length of the history window
HIST_LAG_DAYS = 2     # skip the last 2 days, see add_history
PRIOR_WEIGHT = 20     # smoothing of history rates towards the network rate
WEATHER_LAG_MIN = 60  # only use weather hours that ended at least this long before T0


def build_base(con):
    con.sql(f"""
        create or replace view stops as select * from '{PROCESSED}/stops.parquet';
        create or replace view station_map as select * from '{PROCESSED}/station_map.parquet';

        -- every NS arrival at a Dutch station, including the warm-up month and cancelled ones
        create or replace table arrivals as
        select s.service_id, s.station, s.train_number, s.train_type,
               s.arr_sched, s.arr_delay, s.arr_cancelled,
               s.arr_sched + to_minutes(s.arr_delay) as arr_actual,
               timezone('Europe/Amsterdam', s.arr_sched::timestamptz) as arr_local,
               s.stop_seq, s.n_stops, s.service_start, m.knmi_station
        from stops s
        join station_map m using (station)
        where s.arr_sched is not null;

        -- the arrivals we predict: not cancelled, inside the train/val/test period
        create or replace table targets as
        select row_number() over (order by arr_sched, service_id, stop_seq) as row_id,
               *,
               arr_sched - to_minutes({HORIZON_MIN}) as t0,
               (arr_delay >= {DELAY_THRESHOLD_MIN})::int as label,
               case when arr_local < '{VAL_START}' then 'train'
                    when arr_local < '{TEST_START}' then 'val'
                    else 'test' end as split
        from arrivals
        where not arr_cancelled
          and arr_local >= '{TRAIN_START}' and arr_local < '{TEST_END}';
    """)

def add_train_state(con):
    con.sql("""
        -- every scheduled arrival/departure of a train, in timetable order,
        -- with the actual time it happened (scheduled + delay)
        create or replace table service_events as
        with ev as (
            select service_id, arr_sched as sched_time, arr_sched + to_minutes(arr_delay) as event_time,
                   arr_delay as delay, stop_seq, 0 as kind
            from stops where arr_sched is not null and not arr_cancelled
            union all
            select service_id, dep_sched, dep_sched + to_minutes(dep_delay), dep_delay, stop_seq, 1
            from stops where dep_sched is not null and not dep_cancelled
        )
        select service_id, sched_time, event_time, delay,
               row_number() over (partition by service_id order by sched_time, stop_seq, kind) as ev_seq,
               max(delay) over (partition by service_id order by sched_time, stop_seq, kind
                                rows between unbounded preceding and current row) as max_delay
        from ev;

        create or replace table f_train as
        with last_seen as (
            -- the last event of this train that had happened before T0
            select t.row_id, t.service_id, t.t0,
                   e.delay, e.max_delay, e.event_time, coalesce(e.ev_seq, 0) as last_seq
            from targets t
            asof left join service_events e
              on t.service_id = e.service_id and t.t0 - interval 1 minute >= e.event_time
        )
        select l.row_id,
               l.delay as train_delay_now,
               l.max_delay as train_max_delay_now,
               (l.event_time is not null)::int as train_started,
               datediff('minute', l.event_time, l.t0) as mins_since_train_obs,
               -- the next event was scheduled before T0 but had not happened yet,
               -- so the train is at least (T0 - scheduled time) late
               greatest(datediff('minute', n.sched_time, l.t0), 0) as train_overdue_min
        from last_seen l
        left join service_events n on n.service_id = l.service_id and n.ev_seq = l.last_seq + 1;
    """)

def add_network_state(con):
    # counts over [T0 - 60 min, T0) as the difference of two running totals
    con.sql(f"""
        create or replace table minute_events as
        select date_trunc('minute', case when arr_cancelled then arr_sched else arr_actual end) as ts,
               station,
               (not arr_cancelled)::int as n_arr,
               (not arr_cancelled and arr_delay >= {DELAY_THRESHOLD_MIN})::int as n_late,
               case when arr_cancelled then 0 else arr_delay end as delay,
               arr_cancelled::int as n_cancel
        from arrivals;

        create or replace table net_cum as
        select ts,
               sum(sum(n_arr)) over w as c_arr, sum(sum(n_late)) over w as c_late,
               sum(sum(delay)) over w as c_delay, sum(sum(n_cancel)) over w as c_cancel
        from minute_events group by ts
        window w as (order by ts rows between unbounded preceding and current row);

        create or replace table station_cum as
        select station, ts,
               sum(sum(n_arr)) over ws as c_arr, sum(sum(n_late)) over ws as c_late,
               sum(sum(n_cancel)) over ws as c_cancel
        from minute_events group by station, ts
        window ws as (partition by station order by ts rows between unbounded preceding and current row);

        create or replace table f_live as
        with a as (
            select t.row_id, t.station, t.t0, n.c_arr, n.c_late, n.c_delay, n.c_cancel
            from targets t asof left join net_cum n on t.t0 - interval 1 minute >= n.ts
        ), b as (
            select a.*, n.c_arr as p_arr, n.c_late as p_late, n.c_delay as p_delay, n.c_cancel as p_cancel
            from a asof left join net_cum n on a.t0 - interval 61 minute >= n.ts
        ), c as (
            select b.*, s.c_arr as sc_arr, s.c_late as sc_late, s.c_cancel as sc_cancel
            from b asof left join station_cum s on b.station = s.station and b.t0 - interval 1 minute >= s.ts
        ), d as (
            select c.*, s.c_arr as sp_arr, s.c_late as sp_late, s.c_cancel as sp_cancel
            from c asof left join station_cum s on c.station = s.station and c.t0 - interval 61 minute >= s.ts
        )
        select row_id,
               coalesce(c_arr, 0) - coalesce(p_arr, 0) as net_arrivals_1h,
               (coalesce(c_late, 0) - coalesce(p_late, 0)) / nullif(coalesce(c_arr, 0) - coalesce(p_arr, 0), 0) as net_late_share_1h,
               (coalesce(c_delay, 0) - coalesce(p_delay, 0)) / nullif(coalesce(c_arr, 0) - coalesce(p_arr, 0), 0) as net_mean_delay_1h,
               coalesce(c_cancel, 0) - coalesce(p_cancel, 0) as net_cancelled_1h,
               coalesce(sc_arr, 0) - coalesce(sp_arr, 0) as stn_arrivals_1h,
               (coalesce(sc_late, 0) - coalesce(sp_late, 0)) / nullif(coalesce(sc_arr, 0) - coalesce(sp_arr, 0), 0) as stn_late_share_1h,
               coalesce(sc_cancel, 0) - coalesce(sp_cancel, 0) as stn_cancelled_1h
        from d;
    """)


def history_feature(con, name, keys):
    # late rate over the days D-29 .. D-2 for the given grouping
    key_cols = ", ".join(keys)
    on_t = " and ".join(f"t.{k} = h.{k}" for k in keys)
    on_a = " and ".join(f"a.{k} = h.{k}" for k in keys)
    con.sql(f"""
        create or replace table h_{name} as
        select {key_cols}, day,
               sum(count(*)) over w as c_n,
               sum(sum((arr_delay >= {DELAY_THRESHOLD_MIN})::int)) over w as c_late
        from (select *, arr_local::date as day, hour(arr_local) as hour, 1 as all_key
              from arrivals where not arr_cancelled)
        group by {key_cols}, day
        window w as (partition by {key_cols} order by day rows between unbounded preceding and current row);

        create or replace table f_{name} as
        with t as (
            select row_id, arr_local::date as day, hour(arr_local) as hour, 1 as all_key, station, train_number
            from targets
        ), a as (
            select t.row_id, t.day, {", ".join("t." + k for k in keys)}, h.c_n, h.c_late
            from t asof left join h_{name} h on {on_t} and t.day - {HIST_LAG_DAYS} >= h.day
        ), b as (
            select a.row_id, a.c_n, a.c_late, h.c_n as p_n, h.c_late as p_late
            from a asof left join h_{name} h on {on_a} and a.day - {HIST_DAYS + HIST_LAG_DAYS} >= h.day
        )
        select row_id,
               coalesce(c_n, 0) - coalesce(p_n, 0) as hist_n_{name},
               coalesce(c_late, 0) - coalesce(p_late, 0) as hist_late_{name}
        from b;
    """)


def add_history(con):
    # lag of 2 days: an arrival just after midnight has its T0 on the previous day,
    # when that day is not over yet, so the previous day can't be used
    history_feature(con, "network", ["all_key"])
    history_feature(con, "station", ["station"])
    history_feature(con, "station_hour", ["station", "hour"])
    history_feature(con, "train_number", ["train_number"])

def add_weather(con):
    con.sql(f"""
        create or replace table weather as
        select *,
               sum(precip_mm) over (partition by knmi_station order by obs_time
                                    range between interval 2 hour preceding and current row) as precip_3h,
               max(wind_gust) over (partition by obs_time) as nl_max_gust
        from '{PROCESSED}/weather.parquet'
        where knmi_station in (select knmi_station from station_map);

        create or replace table f_weather as
        select t.row_id,
               w.temperature, w.wind_speed, w.wind_gust, w.precip_mm, w.precip_3h,
               w.visibility_code, w.snow, w.thunder, w.nl_max_gust
        from targets t
        asof left join weather w
          on t.knmi_station = w.knmi_station and t.t0 - to_minutes({WEATHER_LAG_MIN}) >= w.obs_time;
    """)

def assemble(con):
    rate = lambda n: f"(f_{n}.hist_late_{n} + {PRIOR_WEIGHT} * prior) / (f_{n}.hist_n_{n} + {PRIOR_WEIGHT})"
    con.sql(f"""
        copy (
            with base as (
                select t.*, f_network.hist_late_network / nullif(f_network.hist_n_network, 0) as prior
                from targets t join f_network using (row_id)
            )
            select
                b.row_id, b.split, b.label, b.arr_delay, b.arr_local, b.service_id,
                b.station, b.train_number, b.train_type,
                hour(b.arr_local) as hour,
                isodow(b.arr_local) as weekday,
                (b.train_type = 'Intercity')::int as is_intercity,
                b.stop_seq, b.n_stops,
                b.n_stops - b.stop_seq as stops_remaining,
                datediff('minute', b.service_start, b.arr_sched) as mins_since_service_start,
                ft.train_delay_now, ft.train_max_delay_now, ft.train_started, ft.mins_since_train_obs,
                ft.train_overdue_min,
                fl.net_arrivals_1h, fl.net_late_share_1h, fl.net_mean_delay_1h, fl.net_cancelled_1h,
                fl.stn_arrivals_1h, fl.stn_late_share_1h, fl.stn_cancelled_1h,
                b.prior as hist_rate_network,
                {rate("station")} as hist_rate_station,
                {rate("station_hour")} as hist_rate_station_hour,
                {rate("train_number")} as hist_rate_train_number,
                f_train_number.hist_n_train_number,
                fw.temperature, fw.wind_speed, fw.wind_gust, fw.precip_mm, fw.precip_3h,
                fw.visibility_code, fw.snow, fw.thunder, fw.nl_max_gust
            from base b
            join f_train ft using (row_id)
            join f_live fl using (row_id)
            join f_station using (row_id)
            join f_station_hour using (row_id)
            join f_train_number using (row_id)
            join f_weather fw using (row_id)
            order by b.row_id
        ) to '{PROCESSED}/features.parquet' (format parquet)
    """)


def main():
    db = PROCESSED / "features.duckdb"
    con = duckdb.connect(str(db))
    con.sql("set TimeZone = 'UTC'")
    con.sql("set preserve_insertion_order = false")

    build_base(con)
    add_train_state(con)
    add_network_state(con)
    add_history(con)
    add_weather(con)
    assemble(con)

    print(con.sql(f"""
        select split, count(*) as rows, round(avg(label), 4) as late_rate,
               min(arr_local)::date as first_day, max(arr_local)::date as last_day
        from '{PROCESSED}/features.parquet' group by split order by first_day
    """))
    con.close()
    db.unlink()


if __name__ == "__main__":
    main()