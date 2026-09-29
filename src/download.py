"""Download the raw data: NS services (Rijden de Treinen), station list, KNMI hourly weather."""
import time
import pandas as pd
import requests
from src.config import RAW, FIRST_MONTH, LAST_MONTH

RDT_SERVICES = "https://opendata.rijdendetreinen.nl/public/services/services-{month}.csv.gz"
# newest version available (Sep 2023); includes foreign stations, filtered to NL later
RDT_STATIONS = "https://opendata.rijdendetreinen.nl/public/stations/stations-2023-09.csv"
KNMI_HOURLY = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
# FH mean wind, FX max gust, T temperature, RH precipitation amount, DR precipitation duration,
# VV visibility, R rain, S snow, O thunder (explained in the header of every KNMI file)
KNMI_VARS = "FH:FX:T:RH:DR:VV:R:S:O"


def months() -> list[str]:
    """All months from FIRST_MONTH to LAST_MONTH as 'YYYY-MM' strings."""
    return list(pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M").strftime("%Y-%m"))


def check_gzip(path) -> None:
    # every gzip file starts with the same two bytes
    with open(path, "rb") as f:
        if f.read(2) != b"\x1f\x8b":
            raise ValueError("not a gzip file")


def check_knmi(path) -> None:
    # a valid KNMI response starts with its '#' comment header; errors come back as HTML
    with open(path) as f:
        if f.read(1) != "#":
            raise ValueError("KNMI returned an error page instead of data")


def download(url, dest, data=None, check=None, retries=3) -> None:
    """Save url to dest. Skip if dest already exists; write to .part first."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip   {dest.name}")
        return
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"fetch  {dest.name}")
            method = "POST" if data else "GET"
            with requests.request(method, url, data=data, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if check:
                check(tmp)
            tmp.rename(dest)
            return
        except (requests.RequestException, ValueError) as e:
            print(f"  attempt {attempt} failed: {e}")
            tmp.unlink(missing_ok=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {dest.name}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    for m in months():
        download(RDT_SERVICES.format(month=m), RAW / f"services-{m}.csv.gz", check=check_gzip)

    download(RDT_STATIONS, RAW / "stations-2023-09.csv")

    # one request per month: the KNMI service rejects much longer ranges
    for m in months():
        p = pd.Period(m, freq="M")
        download(KNMI_HOURLY, RAW / f"knmi-hourly-{m}.txt",
                 data={"stns": "ALL", "vars": KNMI_VARS,
                       "start": p.start_time.strftime("%Y%m%d") + "01",
                       "end": p.end_time.strftime("%Y%m%d") + "24"},
                 check=check_knmi)

    total = sum(f.stat().st_size for f in RAW.iterdir()) / 1e6
    print(f"done: {len(list(RAW.iterdir()))} files, {total:.0f} MB in {RAW}")


if __name__ == "__main__":
    main()