import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
import math

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

    # Display summary table
    show_cols = [
        "Stadium", "Azimuth_deg", "Wind_Speed_10m_mph", "Wind_Component_Azimuth_mph",
        "azimuth_direction", "Forecast_Time_Local", "Timezone"
    ]
    st.dataframe(df_top[show_cols], use_container_width=True)

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


st.title("College Baseball Wind")

# Mode toggle
mode = st.sidebar.radio("Mode", ["Testing", "Live"], index=0)

if mode == "Testing":
    try:
        df_mode = load_testing_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
else:
    with st.spinner("Computing live winds per stadium..."):
        try:
            stads = load_stadium_master()
            df_mode = build_live_df(stads)
        except Exception as e:
            st.error(f"Live mode error: {e}")
            st.stop()

try:
    df_testing = load_testing_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Optional filters (light)
col1, col2 = st.columns([1, 1])
with col1:
    team_query = st.text_input("Filter by team/stadium name", value="")
if team_query:
    mask = (
        df_mode.get("stadium_final", pd.Series(dtype=str)).astype(str).str.contains(team_query, case=False, na=False)
        | df_mode.get("Stadium", pd.Series(dtype=str)).astype(str).str.contains(team_query, case=False, na=False)
    )
    df_mode = df_mode[mask]

# Show top 5 view
top5_view(df_mode)

if mode == "Testing":
    st.caption("Testing mode uses local noon forecast per stadium (Open-Meteo).")
else:
    st.caption("Live mode ranks stadiums using the hour closest to now (Open-Meteo).")
# streamlit_app.py
# ----------------
# Division I Baseball Dashboard with map, filters, and optional password.
# Adapted to work with ESPN datasets in this repo.

import os
import datetime as dt
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import requests
import math
import numpy as np
import altair as alt
from zoneinfo import ZoneInfo


# =========================
# -------- SETTINGS -------
# =========================

APP_TITLE = "D1 Baseball Dashboard"
# Default to timezone-enriched ESPN data if present
DATA_PATH = "espn_2025_college_baseball_games_tz_enriched.csv"
DATE_COL = "date"
LAT_COL = "latitude"
LON_COL = "longitude"
HOME_COL = "home_team"
AWAY_COL = "away_team"
CONF_COL = "conference"
VENUE_COL = "venue"
CITY_COL = "city"
STATE_COL = "state"
HOME_SCORE_COL = "home_score"
AWAY_SCORE_COL = "away_score"
ID_COL = "game_id"

# Map defaults (disabled)


# =========================
# ---- AUTHENTICATION -----
# =========================

def check_password() -> bool:
    """
    Simple password gate. Works with:
      - st.secrets["APP_PASSWORD"] (Streamlit Cloud → Settings → Secrets)
      - environment variable APP_PASSWORD
    Set neither to disable password.
    """
    app_pw = st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD"))
    if not app_pw:
        return True  # no password configured

    with st.sidebar:
        st.subheader("🔐 Login")
        pw = st.text_input("Password", type="password")
        if pw == app_pw:
            st.success("Authenticated")
            return True
        elif pw:
            st.error("Incorrect password")
            return False
        else:
            st.info("Enter the password to continue")
            return False
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    """Load and normalize data to the expected schema.

    Supports:
    - ESPN aggregated CSV in this repo (espn_2025_* files)
      Columns: event_id, event_date, home, away, stadium, City, State, lat, lon,
               tz_name, event_dt_local, local_date, local_time
    - Pre-normalized CSV with the expected schema.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported data format. Use .csv or .parquet")

    # If data already conforms, do minimal parsing
    if DATE_COL in df.columns and LAT_COL in df.columns and LON_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.date
        for c in (LAT_COL, LON_COL):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # Otherwise, normalize ESPN schema → expected columns
    df = df.copy()

    # Date
    if "local_date" in df.columns:
        df[DATE_COL] = pd.to_datetime(df["local_date"], errors="coerce").dt.date
    elif "event_date" in df.columns:
        df[DATE_COL] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    else:
        df[DATE_COL] = pd.NaT

    # Coordinates
    if "lat" in df.columns and "lon" in df.columns:
        df[LAT_COL] = pd.to_numeric(df["lat"], errors="coerce")
        df[LON_COL] = pd.to_numeric(df["lon"], errors="coerce")

    # Teams
    if "home" in df.columns:
        df[HOME_COL] = df["home"]
    if "away" in df.columns:
        df[AWAY_COL] = df["away"]

    # Venue and location
    if "stadium" in df.columns:
        df[VENUE_COL] = df["stadium"]
    if "City" in df.columns:
        df[CITY_COL] = df["City"]
    if "State" in df.columns:
        df[STATE_COL] = df["State"]

    # IDs
    if "event_id" in df.columns:
        df[ID_COL] = df["event_id"]

    # Conference and scores may be absent; create empty columns for UI consistency
    if CONF_COL not in df.columns:
        df[CONF_COL] = ""
    if HOME_SCORE_COL not in df.columns:
        df[HOME_SCORE_COL] = pd.NA
    if AWAY_SCORE_COL not in df.columns:
        df[AWAY_SCORE_COL] = pd.NA

    return df


def filter_data(
    df: pd.DataFrame,
    date_range: Tuple[dt.date, dt.date],
    confs: list,
    teams: list,
    min_score_diff: Optional[int] = None,
) -> pd.DataFrame:
    m = pd.Series([True] * len(df))
    if DATE_COL in df.columns and all(date_range):
        start, end = date_range
        m &= (df[DATE_COL] >= start) & (df[DATE_COL] <= end)

    if confs:
        m &= df[CONF_COL].isin(confs)

    if teams:
        # keep games where either team is selected
        m &= (df[HOME_COL].isin(teams)) | (df[AWAY_COL].isin(teams))

    if min_score_diff is not None and {HOME_SCORE_COL, AWAY_SCORE_COL}.issubset(df.columns):
        diff = (df[HOME_SCORE_COL] - df[AWAY_SCORE_COL]).abs()
        m &= diff >= min_score_diff

    out = df[m].copy()
    return out


# =========================
# ----- WIND RANKING -------
# =========================

@st.cache_data(show_spinner=False)
def load_stadiums(path: str = "Stadium_list_final.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp1252")
    cols = list(df.columns)
    if "Azimuth" not in df.columns:
        raise ValueError("Azimuth column not found in Stadium_list_final.csv")
    az_idx = cols.index("Azimuth")
    lat_col = None
    lon_col = None
    for i in range(az_idx + 1, len(cols)):
        cname = cols[i].strip().lower()
        if cname == "lat" and lat_col is None:
            lat_col = cols[i]
        elif cname == "long" and lon_col is None:
            lon_col = cols[i]
        if lat_col and lon_col:
            break
    if lat_col is None or lon_col is None:
        raise ValueError("Could not find Lat/Long columns following Azimuth")

    def parse_azimuth(x):
        if pd.isna(x):
            return np.nan
        s = str(x)
        s = ''.join(ch for ch in s if ch.isdigit() or ch == '.')
        try:
            val = float(s)
            return val % 360.0
        except Exception:
            return np.nan

    df = df.rename(columns={"Team": "team", "Stadium": "stadium"}).copy()
    df["Azimuth_deg"] = df["Azimuth"].apply(parse_azimuth)
    df = df.dropna(subset=["Azimuth_deg", lat_col, lon_col]).copy()
    df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    # Normalize team name for joining
    def norm_team(name: str) -> str:
        if not isinstance(name, str):
            return ""
        return name.strip().lower()
    df["team_norm"] = df["team"].astype(str).apply(norm_team)
    return df[["team", "team_norm", "stadium", "City", "State", "latitude", "longitude", "Azimuth_deg"]]


@st.cache_data(show_spinner=False)
def get_wind_fc(lat: float, lon: float, mode: str = "testing") -> dict:
    """Fetch hourly winds and return a single hour record based on mode.
    mode: "testing" → noon tomorrow (local); "live" → hour closest to now (local).
    """
    OPEN_METEO_URL = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': float(lat),
        'longitude': float(lon),
        'hourly': 'wind_speed_10m,wind_direction_10m',
        'timezone': 'auto',
        'windspeed_unit': 'ms'
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    tzname = data.get('timezone', 'UTC')
    times = data['hourly']['time']
    speeds = data['hourly']['wind_speed_10m']
    dirs_from = data['hourly']['wind_direction_10m']

    idx = None
    if mode == "testing":
        # Local tomorrow at 12:00
        tomorrow_local_date = (dt.datetime.now(ZoneInfo(tzname))).date() + dt.timedelta(days=1)
        for i, t in enumerate(times):
            try:
                parsed = dt.datetime.fromisoformat(t)
            except Exception:
                continue
            if parsed.date() == tomorrow_local_date and parsed.hour == 12:
                idx = i
                break
        if idx is None:
            target = dt.datetime.combine(tomorrow_local_date, dt.time(hour=12))
            diffs = []
            for i, t in enumerate(times):
                try:
                    parsed = dt.datetime.fromisoformat(t)
                except Exception:
                    continue
                diffs.append((abs((parsed - target).total_seconds()), i))
            idx = min(diffs)[1] if diffs else 0
    else:
        # Live: hour closest to now in local stadium time
        now_local = dt.datetime.now(ZoneInfo(tzname)).replace(minute=0, second=0, microsecond=0)
        diffs = []
        for i, t in enumerate(times):
            try:
                parsed = dt.datetime.fromisoformat(t)
            except Exception:
                continue
            diffs.append((abs((parsed - now_local).total_seconds()), i))
        idx = min(diffs)[1] if diffs else 0

    return {'time': times[idx], 'timezone': tzname, 'ws': float(speeds[idx]), 'wd_from': float(dirs_from[idx])}


RAD = math.pi / 180.0
def wind_components(ws: float, wd_from_deg: float, az_deg: float):
    if ws is None or wd_from_deg is None or az_deg is None:
        return (np.nan, np.nan, np.nan)
    wt_deg = (wd_from_deg + 180.0) % 360.0  # toward direction
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


def show_top_wind_games(df: pd.DataFrame, mode: str):
    st.subheader("🌬️ Top 5 wind games")
    st.caption("Mode: Testing → noon tomorrow, Live → closest to now")
    # Expect df columns: date, home_team, away_team, venue, city, state
    if not {HOME_COL, AWAY_COL}.issubset(df.columns):
        st.info("Team columns not found in data.")
        return

    # Load stadiums and join on home team
    try:
        stadiums = load_stadiums()
    except Exception as e:
        st.error(f"Could not load stadium list: {e}")
        return

    def norm_team(name: str) -> str:
        return str(name).strip().lower() if pd.notna(name) else ""

    # Optionally restrict to today's games in Live mode
    df = df.copy()
    if mode == "live" and DATE_COL in df.columns and df[DATE_COL].notna().any():
        today = dt.date.today()
        df_today = df[df[DATE_COL] == today]
        if not df_today.empty:
            df = df_today
    df["home_norm"] = df[HOME_COL].apply(norm_team)
    dfj = df.merge(stadiums, left_on="home_norm", right_on="team_norm", how="left")
    dfj = dfj.dropna(subset=["latitude", "longitude", "Azimuth_deg"]).copy()
    if dfj.empty:
        st.info("No games with matched stadium azimuth/coordinates.")
        return

    # Compute forecast per unique (lat,lon)
    # Cache results to limit requests
    forecasts = {}
    for (lat, lon) in dfj[["latitude", "longitude"]].drop_duplicates().itertuples(index=False):
        try:
            fc = get_wind_fc(lat, lon, mode=mode)
        except Exception as e:
            fc = None
        forecasts[(lat, lon)] = fc

    # Compute components and rank
    rows = []
    for _, r in dfj.iterrows():
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        az = float(r["Azimuth_deg"])
        fc = forecasts.get((lat, lon))
        if not fc:
            continue
        comp_along_az, comp_ns, comp_ew = wind_components(fc['ws'], fc['wd_from'], az)
        rows.append({
            "home": r[HOME_COL],
            "away": r[AWAY_COL],
            "stadium": r.get("stadium", r.get(VENUE_COL, "")),
            "city": r.get("City", r.get(CITY_COL, "")),
            "state": r.get("State", r.get(STATE_COL, "")),
            "ws_ms": fc['ws'],
            "wd_from": fc['wd_from'],
            "comp_az_ms": comp_along_az,
        })

    if not rows:
        st.info("No wind data available.")
        return

    out = pd.DataFrame(rows)
    out["abs_comp"] = out["comp_az_ms"].abs()
    out = out.sort_values("abs_comp", ascending=False).head(5)
    out["direction"] = out["comp_az_ms"].apply(lambda x: "Out" if x > 0 else "In")
    out["label"] = out.apply(lambda r: f"{r['home']} vs {r['away']}\n{r['stadium']} ({r['city']}, {r['state']})", axis=1)

    # Display cards
    for _, r in out.iterrows():
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(f"**{r['home']} (Home)** vs **{r['away']} (Away)**")
            st.caption(f"{r['stadium']} — {r['city']}, {r['state']}")
        with c2:
            st.metric("Wind along azimuth (m/s)", f"{r['comp_az_ms']:.2f}", help="Positive=Out (toward CF), Negative=In")
        # Simple horizontal bar showing direction and magnitude
        bar_df = pd.DataFrame({"comp": [r["comp_az_ms"]], "dir": [r["direction"]], "label": [""]})
        color_scale = alt.Scale(domain=["In", "Out"], range=["#4e79a7", "#e15759"])  # blue vs red
        chart = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("comp", scale=alt.Scale(domain=[-max(1.0, abs(r["comp_az_ms"]) * 1.2), max(1.0, abs(r["comp_az_ms"]) * 1.2)]), title=None),
                color=alt.Color("dir", scale=color_scale, legend=None),
            )
            .properties(height=30)
        )
        st.altair_chart(chart, use_container_width=True)


# =========================
# --------- UI ------------
# =========================

def sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("⚙️ Filters")

    # Date range
    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        min_d = pd.to_datetime(df[DATE_COL]).min()
        max_d = pd.to_datetime(df[DATE_COL]).max()
        if isinstance(min_d, pd.Timestamp):
            min_d = min_d.date()
        if isinstance(max_d, pd.Timestamp):
            max_d = max_d.date()
    else:
        today = dt.date.today()
        min_d, max_d = today - dt.timedelta(days=30), today

    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        help="Filter games by date",
    )
    if isinstance(date_range, dt.date):
        # If user selects only one date, convert to tuple
        date_range = (date_range, date_range)

    # Conferences (may be empty)
    confs = []
    if CONF_COL in df.columns:
        confs = st.sidebar.multiselect(
            "Conference(s)", options=sorted(df[CONF_COL].dropna().unique())
        )

    # Teams
    teams = []
    if HOME_COL in df.columns and AWAY_COL in df.columns:
        all_teams = sorted(
            pd.unique(pd.concat([df[HOME_COL], df[AWAY_COL]], ignore_index=True).dropna())
        )
        teams = st.sidebar.multiselect("Team(s)", options=all_teams)

    # Score diff (likely absent for ESPN schedule-only data)
    min_score_diff = None
    if {HOME_SCORE_COL, AWAY_SCORE_COL}.issubset(df.columns) and df[HOME_SCORE_COL].notna().any():
        min_score_diff = st.sidebar.slider(
            "Minimum score differential",
            min_value=0, max_value=15, value=0
        )

    return date_range, confs, teams, min_score_diff


def kpi_tiles(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    total_games = len(df)
    venues = df[[VENUE_COL, LAT_COL, LON_COL]].dropna().drop_duplicates() if {VENUE_COL, LAT_COL, LON_COL}.issubset(df.columns) else pd.DataFrame()
    unique_venues = len(venues)
    avg_runs = None
    if {HOME_SCORE_COL, AWAY_SCORE_COL}.issubset(df.columns) and total_games > 0 and df[HOME_SCORE_COL].notna().any():
        avg_runs = (df[HOME_SCORE_COL].fillna(0) + df[AWAY_SCORE_COL].fillna(0)).mean()

    c1.metric("Total games", f"{total_games:,}")
    c2.metric("Unique venues", f"{unique_venues:,}")
    if avg_runs is not None:
        c3.metric("Avg total runs", f"{avg_runs:.1f}")
    else:
        c3.metric("Avg total runs", "—")
    if DATE_COL in df.columns and total_games > 0:
        try:
            span = f"{df[DATE_COL].min()} → {df[DATE_COL].max()}"
        except Exception:
            span = "—"
    else:
        span = "—"
    c4.metric("Date span", span)


# Map view removed per request


def games_table(df: pd.DataFrame):
    st.subheader("📋 Games")
    show_cols = [c for c in [
        DATE_COL, HOME_COL, AWAY_COL, CONF_COL, VENUE_COL, CITY_COL, STATE_COL,
        HOME_SCORE_COL, AWAY_SCORE_COL
    ] if c in df.columns]

    st.dataframe(
        df.sort_values(by=[DATE_COL] if DATE_COL in df.columns else df.columns.tolist())
          .reset_index(drop=True)[show_cols],
        use_container_width=True,
        hide_index=True
    )


def team_spotlight(df: pd.DataFrame):
    st.subheader("🏷️ Team spotlight")
    if not ({HOME_COL, AWAY_COL}.issubset(df.columns)):
        st.info("Team columns not found.")
        return

    teams = sorted(pd.unique(pd.concat([df[HOME_COL], df[AWAY_COL]]).dropna()))
    team = st.selectbox("Select a team", teams, index=0 if teams else None)
    if not team:
        return

    tdf = df[(df[HOME_COL] == team) | (df[AWAY_COL] == team)].copy()
    st.write(f"Found **{len(tdf)}** games for **{team}**")

    if {HOME_SCORE_COL, AWAY_SCORE_COL}.issubset(tdf.columns) and len(tdf) > 0 and tdf[HOME_SCORE_COL].notna().any():
        tdf["is_home"] = tdf[HOME_COL] == team
        tdf["team_runs"] = tdf.apply(
            lambda r: r[HOME_SCORE_COL] if r["is_home"] else r[AWAY_SCORE_COL], axis=1
        )
        tdf["opp_runs"] = tdf.apply(
            lambda r: r[AWAY_SCORE_COL] if r["is_home"] else r[HOME_SCORE_COL], axis=1
        )
        wins = (tdf["team_runs"] > tdf["opp_runs"]).sum()
        losses = (tdf["team_runs"] < tdf["opp_runs"]).sum()
        draws = (tdf["team_runs"] == tdf["opp_runs"]).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Wins", int(wins))
        c2.metric("Losses", int(losses))
        c3.metric("Draws", int(draws))

    with st.expander("Show games"):
        cols = [c for c in [DATE_COL, HOME_COL, AWAY_COL, VENUE_COL, CITY_COL, STATE_COL, HOME_SCORE_COL, AWAY_SCORE_COL] if c in tdf.columns]
        st.dataframe(tdf[cols].sort_values(by=[DATE_COL] if DATE_COL in cols else cols), use_container_width=True, hide_index=True)


# =========================
# --------- MAIN ----------
# =========================

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if not check_password():
        st.stop()

    st.title(APP_TITLE)
    st.caption("Interactive D1 games with maps and filters")

    # Detect best available data file
    # Prefer tz-enriched, else fallback to base ESPN CSV
    default_candidates = [
        "espn_2025_college_baseball_games_tz_enriched.csv",
        "espn_2025_college_baseball_games.csv",
        "schedules_2025_d1_baseball_games.csv",
        DATA_PATH,
    ]
    chosen = None
    for p in default_candidates:
        if os.path.exists(p):
            chosen = p
            break
    if not chosen:
        chosen = DATA_PATH

    with st.sidebar.expander("Data Source", expanded=True):
        path_input = st.text_input("Path to data file", value=chosen)

    # Load data
    try:
        df = load_data(path_input)
    except Exception as e:
        st.error(f"Could not load data from '{path_input}': {e}")
        st.stop()

    # Filters
    date_range, confs, teams, min_score_diff = sidebar_filters(df)
    fdf = filter_data(df, date_range, confs, teams, min_score_diff)

    # KPIs
    kpi_tiles(fdf)

    # Tabs (without map)
    t1, t2, t3 = st.tabs(["Wind", "Games", "Teams"])
    with t1:
        mode_choice = st.radio("Mode", ["Testing (Noon Tomorrow)", "Live (Now)"], index=0, help="Switch between testing and live wind selection")
        mode = "testing" if mode_choice.startswith("Testing") else "live"
        show_top_wind_games(fdf, mode)
    with t2:
        games_table(fdf)
    with t3:
        team_spotlight(fdf)

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit. Data © You.")


if __name__ == "__main__":
    main()
