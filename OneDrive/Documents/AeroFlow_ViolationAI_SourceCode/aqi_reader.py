"""
aqi_reader.py  —  AeroFlow AI
Real-time AQI reader using the official Government of India / CPCB API.

Data source:
    https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69
    (Free registration at data.gov.in — hourly updates from CPCB stations)

Strategy:
    1. Try live CPCB API  (requires CPCB_API_KEY env variable)
    2. Fall back to local aqi_dataset.csv if API key is missing or call fails
    3. Cache every result for CACHE_TTL_SECS (default 3600 s = 1 hour)
       CPCB publishes hourly — there is no point re-fetching every frame.

Setup:
    export CPCB_API_KEY="your_key_from_data.gov.in"   # Linux / macOS
    set CPCB_API_KEY=your_key_from_data.gov.in          # Windows CMD

AQI categories (official CPCB / Indian NAQI breakpoints):
    0-50   Good        | 51-100  Satisfactory | 101-200  Moderate
    201-300 Poor       | 301-400 Very Poor    | 401-500  Severe
"""
import os
import time

import requests
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
CPCB_API_KEY  = os.getenv("CPCB_API_KEY", "")
CPCB_ENDPOINT = (
    "https://api.data.gov.in/resource/"
    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
)
CACHE_TTL_SECS = 3600   # 1 hour — matches CPCB update frequency
REQUEST_TIMEOUT = 6     # seconds

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}


def _is_fresh(key: str) -> bool:
    return key in _cache and (time.time() - _cache[key]["ts"]) < CACHE_TTL_SECS


# ── Live API fetch ────────────────────────────────────────────────────────────
def _fetch_live(
    station: str, city: str, pollutant: str = "PM2.5"
) -> float | None:
    """Query the CPCB live API. Returns average pollutant value or None."""
    if not CPCB_API_KEY:
        return None

    params = {
        "api-key":               CPCB_API_KEY,
        "format":                "json",
        "filters[city]":         city,
        "filters[station]":      station,
        "filters[pollutant_id]": pollutant,
        "limit":                 10,
        "fields": "pollutant_avg,pollutant_min,pollutant_max,last_update",
    }

    try:
        resp = requests.get(CPCB_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        records = resp.json().get("records", [])

        values = []
        for rec in records:
            raw = rec.get("pollutant_avg")
            if raw and str(raw).strip().upper() != "NA":
                try:
                    values.append(float(raw))
                except (ValueError, TypeError):
                    pass

        return round(sum(values) / len(values), 2) if values else None

    except Exception as e:
        print(f"[AQI LIVE ERROR] {station}, {city}: {e}")
        return None


# ── CSV fallback ──────────────────────────────────────────────────────────────
def _fetch_csv(
    location: str, city: str,
    pollutant: str = "PM2.5",
    file_path: str = "aqi_dataset.csv",
) -> float | None:
    """
    Fallback: read AQI from a local CPCB CSV export.
    Uses the mean of ALL matching rows (more robust than first-row-only).
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()

        # Detect pollutant column — CPCB CSVs use 'pollutant_id' or 'pollutant_'
        pol_col = next(
            (c for c in df.columns
             if c.startswith("pollutant_")
             and "avg" not in c and "min" not in c and "max" not in c),
            None,
        )
        if pol_col is None:
            print("[AQI CSV WARN] Could not detect pollutant ID column.")
            return None

        mask = (
            df["station"].astype(str).str.contains(location, case=False, na=False)
            & df["city"].astype(str).str.contains(city, case=False, na=False)
            & df[pol_col].astype(str).str.strip().str.upper().eq(pollutant.upper())
        )
        filtered = df[mask]

        if filtered.empty:
            return None

        numeric = pd.to_numeric(filtered["pollutant_avg"], errors="coerce").dropna()
        return round(float(numeric.mean()), 2) if not numeric.empty else None

    except Exception as e:
        print(f"[AQI CSV ERROR] {location}, {city}: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────
def get_aqi(
    location: str,
    city: str,
    pollutant: str = "PM2.5",
    csv_path: str = "aqi_dataset.csv",
) -> float | None:
    """
    Return the PM2.5 (or other) AQI for a location.

    Tries live CPCB API first; falls back to local CSV.
    Results are cached for one hour.

    Args:
        location  : Station name substring (e.g. 'ITO', 'Anand Vihar')
        city      : City name (e.g. 'Delhi')
        pollutant : Pollutant ID string as used by CPCB (default 'PM2.5')
        csv_path  : Path to the local CSV fallback file

    Returns:
        float  AQI sub-index value, or None if unavailable.
    """
    cache_key = f"{location.lower()}|{city.lower()}|{pollutant.upper()}"

    if _is_fresh(cache_key):
        return _cache[cache_key]["value"]

    # 1. Try live CPCB API
    value = _fetch_live(location, city, pollutant)

    # 2. Fall back to local CSV
    if value is None:
        value = _fetch_csv(location, city, pollutant, csv_path)

    _cache[cache_key] = {"value": value, "ts": time.time()}

    source = "LIVE CPCB" if (value is not None and CPCB_API_KEY) else "CSV fallback"
    status = f"{value}" if value is not None else "N/A"
    print(f"  [AQI] {location}, {city} → {status}  ({source})")

    return value


def get_aqi_category(aqi: float | None) -> tuple[str, str]:
    """
    Map a numeric AQI value to its CPCB category name and hex colour.

    Returns:
        (category_name: str, hex_colour: str)
    """
    if aqi is None:
        return "Unknown", "#888888"
    if aqi <= 50:
        return "Good",         "#00B050"
    if aqi <= 100:
        return "Satisfactory", "#92D050"
    if aqi <= 200:
        return "Moderate",     "#FFFF00"
    if aqi <= 300:
        return "Poor",         "#FF9900"
    if aqi <= 400:
        return "Very Poor",    "#FF0000"
    return "Severe",           "#800000"


def clear_cache() -> None:
    """Force fresh fetches on the next call (useful in tests)."""
    global _cache
    _cache = {}


# Backward-compatibility alias for code that still calls get_average_pm()
get_average_pm = get_aqi
