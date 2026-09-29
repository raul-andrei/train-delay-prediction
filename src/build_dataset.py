"""Clean the raw files into Parquet: NS stops, KNMI hourly weather, and a map from train station to weather station."""
import duckdb

from src.config import RAW, PROCESSED

MAX_DELAY_MIN = 600  # larger delays are data errors (schedule off by a day), see the EDA notebook


def build_stops(con):
    con.sql(f"""
        create or replace table stops as
        with raw as (
            select *
            from read_csv('{RAW}/services-*.csv.gz', union_by_name = true)
            where "Service:Company" = 'NS' and "Service:Type" in ('Intercity', 'Sprinter')
        ),
        bad_services as (
            select distinct "Service:RDT-ID" as service_id
            from raw
            where "Stop:Arrival delay" >= {MAX_DELAY_MIN} or "Stop:Departure delay" >= {MAX_DELAY_MIN}
        )
        select
            "Service:RDT-ID" as service_id,
            "Service:Date" as service_date,
            "Service:Type" as train_type,
            "Service:Train number" as train_number,
            "Stop:RDT-ID" as stop_id,
            "Stop:Station code" as station,
            timezone('UTC', "Stop:Arrival time") as arr_sched,
            "Stop:Arrival delay" as arr_delay,
            "Stop:Arrival cancelled" as arr_cancelled,
            timezone('UTC', "Stop:Departure time") as dep_sched,
            "Stop:Departure delay" as dep_delay,
            "Stop:Departure cancelled" as dep_cancelled
        from raw
        where "Service:RDT-ID" not in (select service_id from bad_services)
    """)
    # position of each stop within its trip
    con.sql("""
        create or replace table stops as
        select *,
               row_number() over trip as stop_seq,
               count(*) over (partition by service_id) as n_stops,
               min(dep_sched) over (partition by service_id) as service_start
        from stops
        window trip as (partition by service_id order by coalesce(arr_sched, dep_sched), stop_id)
    """)
    con.sql(f"copy stops to '{PROCESSED}/stops.parquet' (format parquet)")

def knmi_columns(path):
    with open(path) as f:
        for line in f:
            if line.startswith("# STN,"):
                return [c.strip() for c in line.lstrip("# ").split(",")]
    raise ValueError(f"no column header in {path}")


def build_weather(con):
    files = sorted(RAW.glob("knmi-hourly-*.txt"))
    cols = knmi_columns(files[0])
    con.sql(f"""
        create or replace macro num(x) as nullif(trim(x), '')::integer;

        create or replace table weather as
        -- distinct: each monthly KNMI file also contains the first day of the next month
        select distinct
            num(STN) as knmi_station,
            -- HH runs 1..24 and HH = 1 is the hour 00:00-01:00 UTC, so the hour is complete at date + HH hours
            strptime(trim(YYYYMMDD), '%Y%m%d') + to_hours(num(HH)) as obs_time,
            num(T) / 10 as temperature,
            num(FH) / 10 as wind_speed,
            num(FX) / 10 as wind_gust,
            greatest(num(RH), 0) / 10 as precip_mm,  -- RH = -1 means less than 0.05 mm
            num(DR) / 10 as precip_hours,
            num(VV) as visibility_code,
            num(R) as rain,
            num(S) as snow,
            num(O) as thunder
        from read_csv({[str(f) for f in files]}, comment = '#', header = false,
                      all_varchar = true, names = {cols})
        where trim(STN) <> ''
    """)
    con.sql(f"copy weather to '{PROCESSED}/weather.parquet' (format parquet)")


def knmi_locations(path):
    # station table in the file header: "# 260   5.180   52.100   1.90   De Bilt"
    rows, in_table = [], False
    with open(path) as f:
        for line in f:
            if line.startswith("# STN") and "LON" in line:
                in_table = True
                continue
            if in_table:
                parts = line.lstrip("# ").split()
                if not parts or not parts[0].isdigit():
                    break
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return rows


def build_station_map(con):
    locations = knmi_locations(sorted(RAW.glob("knmi-hourly-*.txt"))[-1])
    con.execute("create or replace table knmi_loc (knmi_station int, lon double, lat double)")
    con.executemany("insert into knmi_loc values (?, ?, ?)", locations)

    # only weather stations that measure temperature, rain and gusts almost all the time
    con.sql("""
        create or replace table knmi_ok as
        select knmi_station
        from weather
        group by 1
        having avg((temperature is not null and precip_mm is not null and wind_gust is not null)::int) > 0.95
    """)

    con.sql(f"""
        copy (
            select s.code as station, s.name_long as station_name, s.geo_lat as lat, s.geo_lng as lon,
                   arg_min(k.knmi_station, d.dist_km) as knmi_station,
                   min(d.dist_km) as knmi_dist_km
            from read_csv('{RAW}/stations-2023-09.csv') s
            cross join knmi_loc k
            cross join lateral (
                -- flat-earth distance, fine over a few tens of km
                select 111.2 * sqrt(pow((k.lon - s.geo_lng) * cos(radians(s.geo_lat)), 2)
                                    + pow(k.lat - s.geo_lat, 2)) as dist_km
            ) d
            where s.country = 'NL' and k.knmi_station in (select knmi_station from knmi_ok)
            group by all
        ) to '{PROCESSED}/station_map.parquet' (format parquet)
    """)

def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("set TimeZone = 'UTC'")

    build_stops(con)
    build_weather(con)
    build_station_map(con)

    print(con.sql(f"""
        select count(*) as stop_rows, count(distinct service_id) as services,
               min(arr_sched) as first_arrival, max(arr_sched) as last_arrival
        from '{PROCESSED}/stops.parquet'
    """))
    print(con.sql(f"""
        select count(*) as weather_rows, count(distinct knmi_station) as knmi_stations,
               min(obs_time) as first_hour, max(obs_time) as last_hour
        from '{PROCESSED}/weather.parquet'
    """))
    print(con.sql(f"""
        select count(*) as nl_stations, count(distinct knmi_station) as knmi_used,
               round(avg(knmi_dist_km), 1) as avg_km, round(max(knmi_dist_km), 1) as max_km
        from '{PROCESSED}/station_map.parquet'
    """))


if __name__ == "__main__":
    main()