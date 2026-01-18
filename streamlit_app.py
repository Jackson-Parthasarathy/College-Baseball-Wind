import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import requests
from datetime import datetime, date
from zoneinfo import ZoneInfo
import math
import glob
try:
    from timezonefinder import TimezoneFinder
except Exception:
    TimezoneFinder = None

st.set_page_config(page_title="College Baseball Wind — Testing", layout="wide")

@st.cache_data(show_spinner=False)
def load_testing_data(path: str = "stadium_wind_testing.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Testing dataset not found: {path}")
    df = pd.read_csv(p)
    # Normalize expected columns
    rename_map = {
        "Lat": "latitude",
        "Long": "longitude",
        "Azimuth": "Azimuth_deg",
    }
    df = df.rename(columns=rename_map)
    # Coerce to numeric where relevant
    for c in [
        "latitude",
        "longitude",
        "Azimuth_deg",
        "Wind_Speed_10m_ms",
        "Wind_Speed_10m_mph",
        "Wind_Direction_From_deg",
        "Wind_Component_Azimuth_ms",
        "Wind_Component_Azimuth_mph",
        "Component_Along_Azimuth_abs_ms",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Derive handy ranking metric (mph absolute)
    if "Wind_Component_Azimuth_mph" in df.columns:
        df["azimuth_comp_abs_mph"] = df["Wind_Component_Azimuth_mph"].abs()
    elif "Wind_Component_Azimuth_ms" in df.columns:
        df["azimuth_comp_abs_mph"] = (df["Wind_Component_Azimuth_ms"] * 2.23694).abs()
    else:
        df["azimuth_comp_abs_mph"] = np.nan
    # Direction label: blowing toward azimuth vs opposite
    def direction_label(val):
        if pd.isna(val):
            return "unknown"
        return "toward azimuth" if val >= 0 else "opposite azimuth"
    if "Wind_Component_Azimuth_mph" in df.columns:
        df["azimuth_direction"] = df["Wind_Component_Azimuth_mph"].apply(direction_label)
    else:
        df["azimuth_direction"] = np.nan
    return df


@st.cache_data(show_spinner=False)
def load_demo_data(path: str = "stadium_wind_demo_20260213.csv") -> pd.DataFrame:
    """Load precomputed demo dataset for first games (2026-02-13)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Demo dataset not found: {path}")
    df = pd.read_csv(p)
    # Normalize similar to testing
    rename_map = {
        "Lat": "latitude",
        "Long": "longitude",
        "Azimuth": "Azimuth_deg",
    }
    df = df.rename(columns=rename_map)
    for c in [
        "latitude",
        "longitude",
        "Azimuth_deg",
        "Wind_Speed_10m_ms",
        "Wind_Speed_10m_mph",
        "Wind_Direction_From_deg",
        "Wind_Component_Azimuth_ms",
        "Wind_Component_Azimuth_mph",
        "Component_Along_Azimuth_abs_ms",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Wind_Component_Azimuth_mph" in df.columns:
        df["azimuth_comp_abs_mph"] = df["Wind_Component_Azimuth_mph"].abs()
    elif "Wind_Component_Azimuth_ms" in df.columns:
        df["azimuth_comp_abs_mph"] = (df["Wind_Component_Azimuth_ms"] * 2.23694).abs()
    else:
        df["azimuth_comp_abs_mph"] = np.nan
    def direction_label(val):
        if pd.isna(val):
            return "unknown"
        return "toward azimuth" if val >= 0 else "opposite azimuth"
    if "Wind_Component_Azimuth_mph" in df.columns:
        df["azimuth_direction"] = df["Wind_Component_Azimuth_mph"].apply(direction_label)
    else:
        df["azimuth_direction"] = np.nan
    return df


def top5_view(df: pd.DataFrame):
    st.subheader("Top 5 Stadiums by Azimuth-Aligned Wind (Testing — Noon Tomorrow)")
    if df.empty:
        st.info("No rows in testing dataset.")
        return
    # Ensure columns
    needed = ["stadium_final", "Stadium", "latitude", "longitude", "Azimuth_deg",
              "Wind_Speed_10m_mph", "Wind_Component_Azimuth_mph",
              "azimuth_comp_abs_mph", "azimuth_direction", "Forecast_Time_Local", "Timezone"]
    for c in needed:
        if c not in df.columns:
            # Provide fallback if Stadium missing
            if c == "Stadium" and "stadium_final" in df.columns:
                df["Stadium"] = df["stadium_final"]
            else:
                # create empty column to avoid errors
                df[c] = np.nan
    # Sort and select
    df_top = df.dropna(subset=["azimuth_comp_abs_mph"]).sort_values("azimuth_comp_abs_mph", ascending=False).head(5)

    # Display summary table (handle missing optional columns like home/away)
    for c in ["home", "away"]:
        if c not in df_top.columns:
            df_top[c] = np.nan
    base_cols = [
        "Stadium", "home", "away", "Azimuth_deg", "Wind_Speed_10m_mph", "Wind_Component_Azimuth_mph",
        "azimuth_direction", "Forecast_Time_Local", "Timezone"
    ]
    display_cols = [c for c in base_cols if c in df_top.columns]
    st.dataframe(df_top[display_cols], use_container_width=True)

    # Small bar chart: abs azimuth component mph
    chart = alt.Chart(df_top).mark_bar().encode(
        x=alt.X("azimuth_comp_abs_mph:Q", title="|Azimuth Component| (mph)"),
        y=alt.Y("Stadium:N", sort="-x", title="Stadium"),
        color=alt.Color("azimuth_direction:N", title="Direction"),
        tooltip=[
            alt.Tooltip("Stadium", title="Stadium"),
            alt.Tooltip("Wind_Component_Azimuth_mph", title="Azimuth Comp (mph)"),
            alt.Tooltip("Wind_Speed_10m_mph", title="Wind Speed (mph)"),
            alt.Tooltip("Azimuth_deg", title="Azimuth (deg)"),
            alt.Tooltip("Forecast_Time_Local", title="Local Time"),
            alt.Tooltip("Timezone", title="TZ")
        ]
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)


# --- Live mode helpers ---
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
SESSION = requests.Session()
RAD = math.pi / 180.0
TF = TimezoneFinder() if TimezoneFinder is not None else None

# Fallback: infer timezone via Open-Meteo if TimezoneFinder isn't available
_TZ_CACHE: dict[tuple[float, float], str] = {}

def infer_tz_from_openmeteo(lat: float, lon: float) -> str | None:
    try:
        key = (round(float(lat), 4), round(float(lon), 4))
    except Exception:
        return None
    if key in _TZ_CACHE:
        return _TZ_CACHE[key]
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": "wind_speed_10m",  # minimal payload
        "timezone": "auto",
    }
    try:
        r = SESSION.get(OPEN_METEO_URL, params=params, timeout=15)
        r.raise_for_status()
        tzname = (r.json() or {}).get("timezone")
        if isinstance(tzname, str) and tzname:
            _TZ_CACHE[key] = tzname
            return tzname
        return None
    except Exception:
        return None


def wind_components(ws, wd_from_deg, az_deg):
    if ws is None or pd.isna(ws) or wd_from_deg is None or pd.isna(wd_from_deg) or az_deg is None or pd.isna(az_deg):
        return (np.nan, np.nan, np.nan)
    wt_deg = (wd_from_deg + 180.0) % 360.0  # FROM -> TOWARD
    wt = wt_deg * RAD
    az = az_deg * RAD
    w_ex = math.sin(wt)
    w_ny = math.cos(wt)
    a_ex = math.sin(az)
    a_ny = math.cos(az)
    comp_along_az = ws * (w_ex * a_ex + w_ny * a_ny)
    comp_ns = ws * w_ny
    comp_ew = ws * w_ex
    return (comp_along_az, comp_ns, comp_ew)


@st.cache_data(show_spinner=False)
def load_stadium_master(path: str = "Stadium_list_final.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stadium list not found: {path}")
    df = pd.read_csv(p, encoding="cp1252")
    cols = list(df.columns)
    if "Azimuth" not in df.columns:
        raise ValueError("Azimuth column not found in Stadium_list_final.csv")
    az_idx = cols.index("Azimuth")
    lat_col, lon_col = None, None
    for i in range(az_idx + 1, len(cols)):
        name = str(cols[i]).strip().lower()
        if name in ("lat", "latitude") and lat_col is None:
            lat_col = cols[i]
        elif name in ("long", "longitude", "lon") and lon_col is None:
            lon_col = cols[i]
        if lat_col and lon_col:
            break
    if lat_col is None or lon_col is None:
        # Fallback: try anywhere
        for c in df.columns:
            n = str(c).strip().lower()
            if n in ("lat", "latitude") and lat_col is None:
                lat_col = c
            elif n in ("long", "longitude", "lon") and lon_col is None:
                lon_col = c
    if lat_col is None or lon_col is None:
        raise ValueError("Could not find latitude/longitude columns in Stadium_list_final.csv")

    # Clean azimuth
    def parse_azimuth(x):
        if pd.isna(x):
            return np.nan
        s = str(x)
        s = "".join(ch for ch in s if ch.isdigit() or ch == ".")
        try:
            return float(s) % 360.0
        except Exception:
            return np.nan

    df["Azimuth_deg"] = df["Azimuth"].apply(parse_azimuth)
    df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    # Keep essential columns
    keep = [c for c in ["Stadium", "Team", "Azimuth_deg", "latitude", "longitude"] if c in df.columns]
    out = df[keep].dropna(subset=["Azimuth_deg", "latitude", "longitude"]).copy()
    return out


def get_live_hour(lat: float, lon: float):
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "auto",
        "windspeed_unit": "ms",
    }
    r = SESSION.get(OPEN_METEO_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    tzname = data.get("timezone", "UTC")
    times = data["hourly"]["time"]
    speeds = data["hourly"]["wind_speed_10m"]
    dirs_from = data["hourly"]["wind_direction_10m"]
    now_local = datetime.now(ZoneInfo(tzname))
    # pick closest hour to now
    diffs = []
    for i, t in enumerate(times):
        dt = datetime.fromisoformat(t).replace(tzinfo=ZoneInfo(tzname))
        diffs.append((abs((dt - now_local).total_seconds()), i))
    idx = min(diffs)[1]
    return {
        "time": times[idx],
        "timezone": tzname,
        "ws_ms": float(speeds[idx]),
        "wd_from_deg": float(dirs_from[idx]),
    }


def build_live_df(stads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in stads.iterrows():
        lat = r["latitude"]
        lon = r["longitude"]
        az = r["Azimuth_deg"]
        try:
            fc = get_live_hour(lat, lon)
        except Exception:
            # Skip on failures
            continue
        comp_az, comp_ns, comp_ew = wind_components(fc["ws_ms"], fc["wd_from_deg"], az)
        stadium_name = r.get("Stadium") or r.get("Team") or "Unknown"
        rows.append({
            "Stadium": stadium_name,
            "Azimuth_deg": az,
            "latitude": lat,
            "longitude": lon,
            "Forecast_Time_Local": fc["time"],
            "Timezone": fc["timezone"],
            "Wind_Speed_10m_ms": fc["ws_ms"],
            "Wind_Speed_10m_mph": fc["ws_ms"] * 2.23694,
            "Wind_Direction_From_deg": fc["wd_from_deg"],
            "Wind_Component_Azimuth_ms": comp_az,
            "Wind_Component_Azimuth_mph": comp_az * 2.23694,
            "Component_Along_Azimuth_abs_ms": abs(comp_az),
            "azimuth_comp_abs_mph": abs(comp_az) * 2.23694,
            "azimuth_direction": ("toward azimuth" if comp_az >= 0 else "opposite azimuth"),
        })
    return pd.DataFrame(rows)


# --- ESPN games + venue match integration for date-specific forecasts ---
@st.cache_data(show_spinner=False)
def load_games_espx(year: int = 2026, path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = f"espn_{year}_college_baseball_games.csv"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ESPN games CSV not found: {path}")
    df = pd.read_csv(p)
    return df


def norm_name(s: str) -> str:
    s = str(s) if pd.notna(s) else ""
    s = s.strip().lower()
    s = (
        s.replace("—", "-")
         .replace("–", "-")
         .replace("’", "'")
         .replace("“", '"')
         .replace("”", '"')
    )
    s = " ".join(s.split())
    return s


@st.cache_data(show_spinner=False)
def load_match_csv() -> pd.DataFrame:
    # Prefer a venue match report file, fallback to any *stadium*match*.csv
    candidates = []
    candidates += glob.glob("*venue_match_report.csv")
    candidates += glob.glob("*stadium*match*.csv")
    if not candidates:
        raise FileNotFoundError("No stadium match CSV found (expected '*venue_match_report.csv' or '*stadium*match*.csv').")
    # Choose the most recent by modified time
    candidates = sorted(candidates, key=lambda x: Path(x).stat().st_mtime, reverse=True)
    p = Path(candidates[0])
    df = pd.read_csv(p)
    # Normalize columns
    # Ensure espn_venue_norm
    if "espn_venue_norm" not in df.columns:
        if "espn_venue" in df.columns:
            df["espn_venue_norm"] = df["espn_venue"].astype(str).apply(norm_name)
        else:
            raise KeyError("Match CSV missing 'espn_venue' or 'espn_venue_norm'.")
    # Find stadium final/matched column
    stad_col = None
    for cand in ["stadium final", "stadium_final", "matched_stadium", "stadium"]:
        if cand in df.columns:
            stad_col = cand
            break
    if stad_col is None:
        raise KeyError("Match CSV missing stadium mapping column (e.g., 'stadium final', 'matched_stadium').")
    df = df.rename(columns={stad_col: "stadium_final"})
    df["stadium_final_norm"] = df["stadium_final"].astype(str).apply(norm_name)
    return df[["espn_venue_norm", "stadium_final", "stadium_final_norm"]]


def join_games_to_stadiums(games: pd.DataFrame, match_df: pd.DataFrame, stadiums_master: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    # Normalize venue
    g["venue_norm"] = g.get("venue", pd.Series(dtype=str)).astype(str).apply(norm_name)
    # Join with match
    g = g.merge(match_df, left_on="venue_norm", right_on="espn_venue_norm", how="left")
    # Join with stadium master using normalized stadium name
    stadiums_master["stadium_norm"] = stadiums_master.get("Stadium", stadiums_master.get("Team", "")).astype(str).apply(norm_name)
    g = g.merge(stadiums_master, left_on="stadium_final_norm", right_on="stadium_norm", how="left", suffixes=("", "_stad"))
    return g


def parse_event_utc(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%MZ").replace(tzinfo=ZoneInfo("UTC"))
    except Exception:
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return None


def tz_from_latlon(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    # Prefer TimezoneFinder when available
    if TF is not None:
        try:
            tz = TF.timezone_at(lng=float(lon), lat=float(lat))
            if isinstance(tz, str) and tz:
                return tz
        except Exception:
            pass
    # Fallback to Open-Meteo timezone inference
    return infer_tz_from_openmeteo(lat, lon)


def get_fc_nearest_to_local_time(lat: float, lon: float, target_local_dt: datetime) -> dict:
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "timezone": "auto",
        "windspeed_unit": "ms",
    }
    r = SESSION.get(OPEN_METEO_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    tzname = data.get("timezone", "UTC")
    times = data["hourly"]["time"]
    speeds = data["hourly"]["wind_speed_10m"]
    dirs_from = data["hourly"]["wind_direction_10m"]
    # Convert timestamps to timezone-aware using the API's timezone
    diffs = []
    for i, t in enumerate(times):
        dt = datetime.fromisoformat(t).replace(tzinfo=ZoneInfo(tzname))
        diffs.append((abs((dt - target_local_dt).total_seconds()), i))
    idx = min(diffs)[1]
    return {
        "time": times[idx],
        "timezone": tzname,
        "ws_ms": float(speeds[idx]),
        "wd_from_deg": float(dirs_from[idx]),
    }


def build_live_games_for_date(target: date) -> pd.DataFrame:
    # Load ESPN games for the year
    games = load_games_espx(year=target.year)
    # Filter by date (UTC date in event_date)
    games["event_dt_utc"] = games.get("event_date", pd.Series(dtype=str)).apply(parse_event_utc)
    games["event_date_only"] = games["event_dt_utc"].apply(lambda x: x.date() if pd.notna(x) else None)
    games_day = games[games["event_date_only"] == target].copy()
    if games_day.empty:
        return pd.DataFrame()
    # Load match CSV and stadium master
    match_df = load_match_csv()
    stads_master = load_stadium_master()
    # Join to attach stadium_final + coords + azimuth
    gm = join_games_to_stadiums(games_day, match_df, stads_master)
    # Resolve timezone from lat/lon
    gm["tz_name"] = gm.apply(lambda r: tz_from_latlon(r.get("latitude"), r.get("longitude")), axis=1)
    # Convert event time to local
    gm["event_dt_local"] = gm.apply(
        lambda r: r["event_dt_utc"].astimezone(ZoneInfo(r["tz_name"])) if (pd.notna(r["event_dt_utc"]) and pd.notna(r["tz_name"])) else None,
        axis=1
    )
    # Fetch forecast nearest to local game time and compute components
    rows = []
    for _, r in gm.iterrows():
        lat = r.get("latitude")
        lon = r.get("longitude")
        az = r.get("Azimuth_deg")
        local_dt = r.get("event_dt_local")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(az) or local_dt is None:
            continue
        try:
            fc = get_fc_nearest_to_local_time(float(lat), float(lon), local_dt)
        except Exception:
            continue
        comp_az, comp_ns, comp_ew = wind_components(fc["ws_ms"], fc["wd_from_deg"], float(az))
        rows.append({
            "Stadium": r.get("stadium_final") or r.get("Stadium") or r.get("venue"),
            "home": r.get("home"),
            "away": r.get("away"),
            "Azimuth_deg": az,
            "latitude": lat,
            "longitude": lon,
            "Forecast_Time_Local": fc["time"],
            "Timezone": fc["timezone"],
            "Wind_Speed_10m_ms": fc["ws_ms"],
            "Wind_Speed_10m_mph": fc["ws_ms"] * 2.23694,
            "Wind_Direction_From_deg": fc["wd_from_deg"],
            "Wind_Component_Azimuth_ms": comp_az,
            "Wind_Component_Azimuth_mph": comp_az * 2.23694,
            "Component_Along_Azimuth_abs_ms": abs(comp_az),
            "azimuth_comp_abs_mph": abs(comp_az) * 2.23694,
            "azimuth_direction": ("toward azimuth" if comp_az >= 0 else "opposite azimuth"),
        })
    return pd.DataFrame(rows)


st.title("College Baseball Wind")

# Mode toggle
mode = st.sidebar.radio("Mode", ["Testing", "Live", "Demo"], index=0)

if mode == "Testing":
    try:
        df_mode = load_testing_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
elif mode == "Demo":
    try:
        df_mode = load_demo_data()
    except FileNotFoundError as e:
        st.warning(f"{e}\nFalling back to first-day games (2026-02-13) from base data.")
        # Fallback: try ESPN base data filtered to 2026-02-13 and compute winds like testing
        try:
            base_df = load_games_espx(year=2026)
            base_df["event_dt_utc"] = base_df.get("event_date", pd.Series(dtype=str)).apply(parse_event_utc)
            base_df["date_only"] = base_df["event_dt_utc"].apply(lambda x: x.date() if pd.notna(x) else None)
            subset = base_df[base_df["date_only"] == date(2026, 2, 13)].copy()
            if subset.empty:
                st.info("No games found for 2026-02-13.")
                st.stop()
            # Reuse live pipeline to attach stadiums then compute now-hour forecasts; we want noon-tomorrow, so call show_top_wind_games with testing mode
            df_mode = subset
        except Exception as ex:
            st.error(f"Demo fallback failed: {ex}")
            st.stop()
else:
    # Sidebar date picker for Live mode
    selected_date = st.sidebar.date_input("Select date", value=date(2026, 2, 13))
    target_date = selected_date
    with st.spinner(f"Computing winds for games on {target_date.isoformat()}..."):
        try:
            df_mode = build_live_games_for_date(target_date)
        except Exception as e:
            st.error(f"Live mode error: {e}")
            st.stop()
    # If no rows (no games or unmatched venues), show a friendly message
    if df_mode.empty:
        st.info(f"No games found for {target_date.isoformat()} or venues could not be matched.")
        st.stop()

try:
    df_testing = load_testing_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

