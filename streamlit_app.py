import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import requests
import json
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
import math
import glob
try:
    import folium
    import streamlit.components.v1 as components
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False
try:
    from geopy.distance import distance as geopy_distance
    from geopy import Point as GeopyPoint
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

# Optional timezone finder
try:
    from timezonefinder import TimezoneFinder
except Exception:
    TimezoneFinder = None

# ---------- General settings ----------

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
SESSION = requests.Session()
RAD = math.pi / 180.0
TF = TimezoneFinder() if TimezoneFinder is not None else None

# ---------- Logo helper ----------
def _normalize_name(s: str) -> str:
    return str(s or "").strip().lower()

def find_logo_for_team(team_name: str) -> str | None:
    """Best-effort logo lookup using normalized keys across team_logos.
    Falls back to substring matching if direct lookup fails.
    """
    try:
        lookup = load_team_logos()
    except Exception:
        lookup = {}
    key = _normalize_name(team_name)
    if key in lookup:
        return lookup[key]
    # fallback: substring match
    for k, url in lookup.items():
        if not isinstance(k, str) or not url:
            continue
        if key and (key in k or k in key):
            return url
    return None

# ---------- Team logos ----------
@st.cache_data(show_spinner=False)
def load_team_logos(path: str = "team_logos.csv") -> dict:
    """Load ESPN team logos CSV and return a normalized name→logo URL map."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        logos = pd.read_csv(p)
    except Exception:
        return {}
    # Normalize potential name columns
    for col in ["displayName", "shortDisplayName", "name", "location", "abbreviation", "slug"]:
        if col in logos.columns:
            logos[f"{col}_norm"] = logos[col].fillna("").str.strip().str.lower()
        else:
            logos[f"{col}_norm"] = ""
    lookup = {}
    for _, row in logos.iterrows():
        logo_url = row.get("logo_default") or row.get("logo_dark")
        if not isinstance(logo_url, str) or not logo_url:
            continue
        for key in (
            "displayName_norm",
            "shortDisplayName_norm",
            "name_norm",
            "location_norm",
            "abbreviation_norm",
            "slug_norm",
        ):
            val = row.get(key)
            if isinstance(val, str) and val and val not in lookup:
                lookup[val] = logo_url
    return lookup

def attach_team_logos(df: pd.DataFrame, team_col: str = "Team") -> pd.DataFrame:
    """Attach a 'team_logo_url' column by mapping normalized team names to logo URLs."""
    try:
        lookup = load_team_logos()
    except Exception:
        lookup = {}
    if not lookup or (team_col not in df.columns):
        if "team_logo_url" not in df.columns:
            df["team_logo_url"] = np.nan
        return df
    out = df.copy()
    out["_team_norm"] = out[team_col].astype(str).fillna("").str.strip().str.lower()
    out["team_logo_url"] = out["_team_norm"].map(lookup)
    return out.drop(columns=["_team_norm"])

# ---------- Testing & Demo loaders ----------
@st.cache_data(show_spinner=False)
def load_testing_data(path: str = "stadium_wind_testing.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Testing dataset not found: {path}")
    df = pd.read_csv(p)
    # Normalize
    df = df.rename(columns={"Lat": "latitude", "Long": "longitude", "Azimuth": "Azimuth_deg"})
    for c in [
        "latitude","longitude","Azimuth_deg",
        "Wind_Speed_10m_ms","Wind_Speed_10m_mph",
        "Wind_Direction_From_deg",
        "Wind_Component_Azimuth_ms","Wind_Component_Azimuth_mph",
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

@st.cache_data(show_spinner=False)
def load_demo_data(path: str = "stadium_wind_demo_20260213.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Demo dataset not found: {path}")
    df = pd.read_csv(p)
    df = df.rename(columns={"Lat": "latitude", "Long": "longitude", "Azimuth": "Azimuth_deg"})
    for c in [
        "latitude","longitude","Azimuth_deg",
        "Wind_Speed_10m_ms","Wind_Speed_10m_mph",
        "Wind_Direction_From_deg",
        "Wind_Component_Azimuth_ms","Wind_Component_Azimuth_mph",
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

# ---------- Core math ----------
def wind_components(ws: float, wd_from_deg: float, az_deg: float):
    if ws is None or pd.isna(ws) or wd_from_deg is None or pd.isna(wd_from_deg) or az_deg is None or pd.isna(az_deg):
        return (np.nan, np.nan, np.nan)
    wt_deg = (wd_from_deg + 180.0) % 360.0  # FROM → TOWARD
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

# ---------- Stadium master ----------
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
        # Try anywhere
        for c in df.columns:
            n = str(c).strip().lower()
            if n in ("lat", "latitude") and lat_col is None:
                lat_col = c
            elif n in ("long", "longitude", "lon") and lon_col is None:
                lon_col = c
    if lat_col is None or lon_col is None:
        raise ValueError("Could not find latitude/longitude columns in Stadium_list_final.csv")
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
    keep = [c for c in ["Stadium", "Team", "Azimuth_deg", "latitude", "longitude"] if c in df.columns]
    out = df[keep].dropna(subset=["Azimuth_deg", "latitude", "longitude"]).copy()
    return out

# ---------- Live helpers ----------
def get_live_hour(lat: float, lon: float) -> dict:
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
            continue
        comp_az, comp_ns, comp_ew = wind_components(fc["ws_ms"], fc["wd_from_deg"], az)
        stadium_name = r.get("Stadium") or "Unknown"
        team_name = r.get("Team")
        rows.append({
            "Stadium": stadium_name,
            "Team": team_name,
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
            "azimuth_direction": ("blowing out" if comp_az >= 0 else "blowing in"),
        })
    return pd.DataFrame(rows)

# ---------- Schedule helpers ----------
@st.cache_data(show_spinner=False)
def load_espn_schedule_2026(path: str = "espn_2026_college_baseball_games.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ESPN 2026 schedule not found: {path}")
    df = pd.read_csv(p)
    df["event_dt_utc"] = pd.to_datetime(df.get("event_date"), utc=True, errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_venue_match_report(path: str = "espn_2026_college_baseball_games_venue_match_report.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["espn_venue_norm","matched_stadium","lat","lon","match"])
    return pd.read_csv(p)

def _norm_text(s: str) -> str:
    return str(s or "").strip().lower()

def match_games_to_stadiums(
    games: pd.DataFrame,
    match_report: pd.DataFrame,
    stadiums: pd.DataFrame,
) -> pd.DataFrame:
    out = games.copy()
    out["venue_norm"] = out.get("venue", pd.Series(dtype=str)).astype(str).map(_norm_text)
    report = match_report.copy()
    if "espn_venue_norm" not in report.columns:
        report["espn_venue_norm"] = report.get("espn_venue", pd.Series(dtype=str)).astype(str).map(_norm_text)
    report["matched_stadium"] = report.get("matched_stadium", pd.Series(dtype=str)).astype(str)
    if "match" in report.columns:
        report["match"] = report["match"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        report["match"] = True
    report = report[report["match"] == True]
    report = report[["espn_venue_norm","matched_stadium","lat","lon"]].drop_duplicates()
    out = out.merge(report, left_on="venue_norm", right_on="espn_venue_norm", how="left")
    out["Stadium"] = out["matched_stadium"].fillna(out.get("venue"))
    stadiums_merge = stadiums.rename(columns={
        "Stadium": "Stadium_master",
        "latitude": "latitude_master",
        "longitude": "longitude_master",
    })
    out = out.merge(
        stadiums_merge[["Stadium_master","Team","Azimuth_deg","latitude_master","longitude_master"]],
        left_on="Stadium",
        right_on="Stadium_master",
        how="left",
    )
    out["latitude"] = out.get("latitude_master").fillna(out.get("lat"))
    out["longitude"] = out.get("longitude_master").fillna(out.get("lon"))
    out["Team"] = out.get("Team").fillna(out.get("home"))
    drop_cols = [c for c in ["venue_norm","espn_venue_norm","matched_stadium","Stadium_master","lat","lon","latitude_master","longitude_master"] if c in out.columns]
    out = out.drop(columns=drop_cols)
    return out

def get_timezone_name(lat: float, lon: float) -> str | None:
    if TF is None:
        return None
    try:
        return TF.timezone_at(lng=float(lon), lat=float(lat))
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_hourly_forecast(lat: float, lon: float, local_date: date, tzname: str | None) -> dict:
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "windspeed_unit": "ms",
        "start_date": local_date.isoformat(),
        "end_date": local_date.isoformat(),
    }
    if tzname:
        params["timezone"] = tzname
    else:
        params["timezone"] = "auto"
    r = SESSION.get(OPEN_METEO_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return {
        "timezone": data.get("timezone", tzname or "UTC"),
        "times": data["hourly"]["time"],
        "speeds": data["hourly"]["wind_speed_10m"],
        "dirs_from": data["hourly"]["wind_direction_10m"],
    }

def pick_hourly_forecast(event_dt_utc: datetime, tzname: str, times, speeds, dirs_from) -> dict | None:
    if event_dt_utc is None or pd.isna(event_dt_utc) or not tzname or not times:
        return None
    try:
        event_local = event_dt_utc.astimezone(ZoneInfo(tzname))
    except Exception:
        return None
    diffs = []
    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t).replace(tzinfo=ZoneInfo(tzname))
        except Exception:
            continue
        diffs.append((abs((dt - event_local).total_seconds()), i))
    if not diffs:
        return None
    idx = min(diffs)[1]
    return {
        "time": times[idx],
        "timezone": tzname,
        "ws_ms": float(speeds[idx]),
        "wd_from_deg": float(dirs_from[idx]),
    }

def build_schedule_wind_df(games_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in games_df.iterrows():
        lat = r.get("latitude")
        lon = r.get("longitude")
        az = r.get("Azimuth_deg")
        event_dt_utc = r.get("event_dt_utc")
        forecast_note = ""
        fc_pick = None
        tzname = None
        event_local_str = ""
        if pd.notna(event_dt_utc):
            event_dt_utc = pd.to_datetime(event_dt_utc, utc=True, errors="coerce")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(event_dt_utc):
            forecast_note = "Missing location or game time"
        else:
            tz_guess = get_timezone_name(lat, lon)
            try:
                if tz_guess:
                    local_date = event_dt_utc.astimezone(ZoneInfo(tz_guess)).date()
                else:
                    local_date = event_dt_utc.date()
                fc = get_hourly_forecast(lat, lon, local_date, tz_guess)
                tzname = fc.get("timezone") or tz_guess or "UTC"
                fc_pick = pick_hourly_forecast(event_dt_utc, tzname, fc.get("times", []), fc.get("speeds", []), fc.get("dirs_from", []))
            except Exception:
                forecast_note = "Forecast unavailable"
        if tzname and pd.notna(event_dt_utc):
            try:
                event_local = event_dt_utc.astimezone(ZoneInfo(tzname))
                event_local_str = event_local.strftime("%Y-%m-%d %H:%M")
            except Exception:
                event_local_str = event_dt_utc.strftime("%Y-%m-%d %H:%M UTC")
        elif pd.notna(event_dt_utc):
            event_local_str = event_dt_utc.strftime("%Y-%m-%d %H:%M UTC")

        if fc_pick and pd.notna(az):
            comp_az, comp_ns, comp_ew = wind_components(fc_pick["ws_ms"], fc_pick["wd_from_deg"], az)
            wind_speed_ms = fc_pick["ws_ms"]
            wind_dir_from = fc_pick["wd_from_deg"]
            forecast_time_local = fc_pick["time"]
        else:
            comp_az = np.nan
            wind_speed_ms = np.nan
            wind_dir_from = np.nan
            forecast_time_local = ""
            if not forecast_note and pd.isna(az):
                forecast_note = "Missing stadium azimuth"
            elif not forecast_note:
                forecast_note = "Forecast unavailable"

        rows.append({
            "event_id": r.get("event_id"),
            "event_date_utc": r.get("event_date"),
            "event_time_local": event_local_str,
            "status": r.get("status"),
            "home": r.get("home"),
            "away": r.get("away"),
            "venue": r.get("venue"),
            "Stadium": r.get("Stadium"),
            "Team": r.get("Team"),
            "Azimuth_deg": az,
            "latitude": lat,
            "longitude": lon,
            "Forecast_Time_Local": forecast_time_local,
            "Timezone": tzname,
            "Wind_Speed_10m_ms": wind_speed_ms,
            "Wind_Speed_10m_mph": wind_speed_ms * 2.23694 if pd.notna(wind_speed_ms) else np.nan,
            "Wind_Direction_From_deg": wind_dir_from,
            "Wind_Component_Azimuth_ms": comp_az,
            "Wind_Component_Azimuth_mph": comp_az * 2.23694 if pd.notna(comp_az) else np.nan,
            "Component_Along_Azimuth_abs_ms": abs(comp_az) if pd.notna(comp_az) else np.nan,
            "azimuth_comp_abs_mph": abs(comp_az) * 2.23694 if pd.notna(comp_az) else np.nan,
            "azimuth_direction": ("blowing out" if pd.notna(comp_az) and comp_az >= 0 else ("blowing in" if pd.notna(comp_az) else "")),
            "forecast_note": forecast_note,
        })
    return pd.DataFrame(rows)

# ---------- Testing/Demo view ----------
def _render_top5_map(df_top: pd.DataFrame):
    if not FOLIUM_AVAILABLE:
        st.info("Map rendering is unavailable (install 'folium' to enable).")
        return
    if not GEOPY_AVAILABLE:
        st.info("Geodesic arrows unavailable (install 'geopy' to enable direction arrows).")
    pts = df_top.dropna(subset=["latitude", "longitude"]).copy()
    if pts.empty:
        st.info("No coordinates available to render map.")
        return
    center_lat = float(pts["latitude"].mean())
    center_lon = float(pts["longitude"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="OpenStreetMap")
    max_mag = max(1.0, float(pts["azimuth_comp_abs_mph"].max()))
    for _, r in pts.iterrows():
        mag = float(r.get("azimuth_comp_abs_mph", 0.0) or 0.0)
        ratio = min(1.0, mag / max_mag)
        red = int(255 * ratio)
        green = int(255 * (1.0 - ratio))
        color = f"#{red:02x}{green:02x}50"
        ws_mph = float(r.get("Wind_Speed_10m_mph", 0.0) or 0.0)
        wd_from = float(r.get("Wind_Direction_From_deg", np.nan)) if pd.notna(r.get("Wind_Direction_From_deg")) else np.nan
        wd_toward = (wd_from + 180.0) % 360.0 if not pd.isna(wd_from) else None
        start_lat = float(r["latitude"])
        start_lon = float(r["longitude"])

        logo_url = r.get("team_logo_url")
        logo_tag = (
            f'<img src="{logo_url}" width="40" style="vertical-align:middle;margin-right:6px;" />'
            if isinstance(logo_url, str) and logo_url else ""
        )
        popup_html = (
            f"{logo_tag}<b>{r.get('Stadium','')}</b><br/>"
            f"Team: {r.get('Team','')}<br/>"
            f"Wind: {ws_mph:.1f} mph<br/>"
            f"Toward: {wd_toward if wd_toward is not None else 'N/A'}°<br/>"
            f"Azimuth Comp: {float(r.get('Wind_Component_Azimuth_mph', np.nan)) if pd.notna(r.get('Wind_Component_Azimuth_mph')) else np.nan:.1f} mph<br/>"
            f"Time: {r.get('Forecast_Time_Local','')} ({r.get('Timezone','')})"
        )

        # Marker at stadium
        folium.CircleMarker(
            location=[start_lat, start_lon],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(html=popup_html, max_width=300),
        ).add_to(m)

        # Draw wind direction arrow (toward). Length proportional to speed.
        if GEOPY_AVAILABLE and (wd_toward is not None):
            # Arrow length: 0.4 km per 10 mph, capped 8 km
            length_km = min(8.0, max(0.8, ws_mph * 0.04))
            start_pt = GeopyPoint(start_lat, start_lon)
            end_pt = geopy_distance(kilometers=length_km).destination(start_pt, wd_toward)
            # Main vector line
            folium.PolyLine(
                locations=[[start_lat, start_lon], [end_pt.latitude, end_pt.longitude]],
                color=color,
                weight=4,
                opacity=0.9,
            ).add_to(m)
            # Arrowhead: two short lines at ±25° from end
            for offset in (-25.0, 25.0):
                head_pt = geopy_distance(kilometers=length_km * 0.25).destination(end_pt, (wd_toward + 180.0 + offset) % 360.0)
                folium.PolyLine(
                    locations=[[end_pt.latitude, end_pt.longitude], [head_pt.latitude, head_pt.longitude]],
                    color=color,
                    weight=3,
                    opacity=0.9,
                ).add_to(m)
        components.html(m.get_root().render(), height=500, scrolling=False)


def render_earth_live_map():
    st.subheader("Live Map — Global Wind (earth.nullschool.net)")
    # Controls
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        tz_choice = st.radio("Timezone", ["UTC", "Local"], index=0, help="Set map timestamp as UTC or your browser's local time.")
    with c2:
        zoom = st.slider("Zoom", min_value=500, max_value=5000, value=2500, step=250)
    with c3:
        hour_offset = st.slider("Offset from now (hours)", min_value=-120, max_value=120, value=0, step=3)

    # Stadium dropdown with search and logo
    try:
        master = load_stadium_master()
    except Exception:
        master = pd.DataFrame(columns=["Team","Stadium","latitude","longitude"])  # fallback
    teams_df = master.dropna(subset=["Team","Stadium","latitude","longitude"]).copy()
    teams_df = teams_df.sort_values("Stadium")
    all_stadium_names = teams_df["Stadium"].astype(str).unique().tolist()
    stadium_to_team = teams_df.set_index("Stadium")["Team"].to_dict()

    st.write("Select stadium to center the map")
    search_q = st.text_input("Search stadium or team", value="", placeholder="Type to filter")
    if search_q:
        team_names = [
            s for s in all_stadium_names
            if (search_q.lower() in str(s).lower())
            or (search_q.lower() in str(stadium_to_team.get(s, "")).lower())
        ]
    else:
        team_names = all_stadium_names

    sel_col1, sel_col2, sel_col3 = st.columns([2,1,1])
    with sel_col1:
        selected_stadium = st.selectbox("Stadium", team_names, index=0 if team_names else None)
    logo_url = None
    selected_team = None
    if selected_stadium:
        try:
            logos_lookup = load_team_logos()
            selected_team = stadium_to_team.get(selected_stadium)
            key = str(selected_team).strip().lower()
            logo_url = logos_lookup.get(key)
        except Exception:
            logo_url = None
    with sel_col2:
        if logo_url:
            st.image(logo_url, width=64)
    with sel_col3:
        center_on_team = st.checkbox("Center on stadium", value=True)

    # Default center
    center_lat = 37.67
    center_lon = -122.53
    if selected_stadium and center_on_team and not teams_df.empty:
        row = teams_df[teams_df["Stadium"] == selected_stadium].iloc[0]
        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            center_lat = float(row["latitude"])  # stadium lat
            center_lon = float(row["longitude"])  # stadium lon

    # Allow manual tweak of center after selection
    c4, c5 = st.columns(2)
    with c4:
        center_lat = st.number_input("Center latitude", value=center_lat, format="%.4f")
    with c5:
        center_lon = st.number_input("Center longitude", value=center_lon, format="%.4f")

    # Stadium live wind vane
    if selected_stadium and not teams_df.empty:
        row = teams_df[teams_df["Stadium"] == selected_stadium].iloc[0]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        try:
            fc = get_live_hour(lat, lon)
            ws_mph = fc["ws_ms"] * 2.23694
            wd_from = fc["wd_from_deg"]
            wd_toward = (wd_from + 180.0) % 360.0
            st.markdown("**Live wind at selected stadium**")
            st.caption(f"Speed: {ws_mph:.1f} mph • From: {wd_from:.0f}° • Toward: {wd_toward:.0f}°")
            _render_wind_vane_map(
                latitude=lat,
                longitude=lon,
                wind_speed_mph=ws_mph,
                wind_from_deg=wd_from,
                stadium_name=selected_stadium,
                team_name=selected_team,
            )
        except Exception as e:
            st.info(f"Live wind unavailable: {e}")

    now_utc = datetime.now(timezone.utc)
    target_utc = now_utc + timedelta(hours=int(hour_offset))

    # Display label in chosen TZ and set time suffix
    if tz_choice == "Local":
        local_dt = target_utc.astimezone()
        st.caption(f"Selected time (Local): {local_dt:%Y-%m-%d %H:%M %Z}")
        time_suffix = ""  # local time (Earth will interpret without Z)
    else:
        st.caption(f"Selected time (UTC): {target_utc:%Y-%m-%d %H:%M} UTC")
        time_suffix = "Z"  # explicit UTC

    # Earth URL fragment #YYYY/MM/DD/HH(Z?) with layer + projection
    earth_fragment = target_utc.strftime(f"%Y/%m/%d/%H{time_suffix}")
    earth_url = (
        f"https://earth.nullschool.net/#{earth_fragment}/wind/surface/orthographic={center_lon:.2f},{center_lat:.2f},{int(zoom)}"
    )

    # High contrast via CSS filter to improve visibility
    html = f"""
    <div style=\"width:100%; height:650px; background:#000;\">
      <iframe src=\"{earth_url}\" style=\"width:100%; height:100%; border:0; filter: contrast(1.45) brightness(1.1) saturate(1.25);\" allowfullscreen></iframe>
    </div>
    <div style=\"font-size:12px;margin-top:6px;opacity:0.8;\">URL: {earth_url}</div>
    """
    components.html(html, height=700, scrolling=False)


def top5_view(df: pd.DataFrame, heading_date: str | None = None, show_map: bool = False):
    title = "Top 5 Stadiums by Wind"
    if heading_date:
        title = f"{title} — {heading_date}"
    st.subheader(title)
    if df.empty:
        st.info("No rows to display.")
        return
    needed = [
        "team_logo_url","Stadium","Team","latitude","longitude","Azimuth_deg",
        "Wind_Speed_10m_mph","Wind_Component_Azimuth_mph",
        "azimuth_comp_abs_mph","azimuth_direction","Forecast_Time_Local","Timezone"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    df_top = df.dropna(subset=["azimuth_comp_abs_mph"]).sort_values("azimuth_comp_abs_mph", ascending=False).head(5)
    st.dataframe(
        df_top[[c for c in needed if c in df_top.columns]],
        width='stretch',
        column_config={
            "team_logo_url": st.column_config.ImageColumn("Logo", width=40),
        },
    )
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
            alt.Tooltip("Timezone", title="TZ"),
        ]
    ).properties(height=220)
    st.altair_chart(chart, width='stretch')
    if show_map:
        st.markdown("**Forecast Map (Top 5)**")
        _render_top5_map(df_top)

def _render_stadiums_map(
    df_all: pd.DataFrame,
    center_lat: float | None = None,
    center_lon: float | None = None,
    zoom_start: int = 5,
    auto_center_on_popup: bool = False,
    popup_zoom: int | None = None,
):
    """Render one Folium map with all provided stadiums.
    Scales markers by `azimuth_comp_abs_mph` if available, else by wind speed.
    """
    if not FOLIUM_AVAILABLE:
        st.info("Map rendering is unavailable (install 'folium' to enable).")
        return
    pts = df_all.dropna(subset=["latitude", "longitude"]).copy()
    if pts.empty:
        st.info("No coordinates available to render map.")
        return
    if center_lat is None or center_lon is None:
        center_lat = float(pts["latitude"].mean())
        center_lon = float(pts["longitude"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=int(zoom_start), tiles="OpenStreetMap")
    # Choose metric for marker scaling
    metric_col = "azimuth_comp_abs_mph" if "azimuth_comp_abs_mph" in pts.columns else (
        "Wind_Speed_10m_mph" if "Wind_Speed_10m_mph" in pts.columns else None
    )
    max_metric = 1.0
    if metric_col is not None:
        max_metric = max(1.0, float(pts[metric_col].max()))
    for _, r in pts.iterrows():
        metric_val = float(r.get(metric_col, 0.0) or 0.0) if metric_col is not None else 0.0
        ratio = min(1.0, metric_val / max_metric)
        red = int(255 * ratio)
        green = int(255 * (1.0 - ratio))
        color = f"#{red:02x}{green:02x}50"
        start_lat = float(r["latitude"]) 
        start_lon = float(r["longitude"]) 
        logo_url = r.get("team_logo_url")
        logo_tag = (
            f'<img src="{logo_url}" width="40" style="vertical-align:middle;margin-right:6px;" />'
            if isinstance(logo_url, str) and logo_url else ""
        )
        popup_html = (
            f"{logo_tag}<b>{r.get('Stadium','')}</b><br/>"
            f"Team: {r.get('Team','')}<br/>"
            f"|Azimuth Comp|: {float(r.get('azimuth_comp_abs_mph', np.nan)) if pd.notna(r.get('azimuth_comp_abs_mph')) else np.nan:.1f} mph<br/>"
            f"Direction: {r.get('azimuth_direction','')}<br/>"
            f"From: {float(r.get('Wind_Direction_From_deg', np.nan)) if pd.notna(r.get('Wind_Direction_From_deg')) else np.nan}°<br/>"
            f"Time: {r.get('Forecast_Time_Local','')} ({r.get('Timezone','')})"
        )
        folium.CircleMarker(
            location=[start_lat, start_lon],
            radius=max(5, int(5 + 10 * ratio)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(html=popup_html, max_width=320)
        ).add_to(m)
    if auto_center_on_popup:
        # Auto-center and zoom when a marker popup opens
        popup_zoom_val = int(popup_zoom if popup_zoom is not None else zoom_start)
        script = f"""
        <script>
        (function() {{
            var map = {m.get_name()};
            map.on('popupopen', function(e) {{
                var latlng = e.popup.getLatLng();
                map.setView(latlng, {popup_zoom_val}, {{ animate: true }});
            }});
        }})();
        </script>
        """
        m.get_root().html.add_child(folium.Element(script))
    components.html(m.get_root().render(), height=640, scrolling=False)

def _render_wind_vane_map(
    latitude: float,
    longitude: float,
    wind_speed_mph: float,
    wind_from_deg: float,
    stadium_name: str | None = None,
    team_name: str | None = None,
):
    if not FOLIUM_AVAILABLE:
        st.info("Map rendering is unavailable (install 'folium' to enable).")
        return
    if not GEOPY_AVAILABLE:
        st.info("Geodesic arrows unavailable (install 'geopy' to enable wind vane).")
        return
    start_lat = float(latitude)
    start_lon = float(longitude)
    m = folium.Map(location=[start_lat, start_lon], zoom_start=12, tiles="OpenStreetMap")
    wd_toward = (float(wind_from_deg) + 180.0) % 360.0
    # Arrow length: 0.5 km per 10 mph, capped 10 km
    length_km = min(10.0, max(0.6, float(wind_speed_mph) * 0.05))
    start_pt = GeopyPoint(start_lat, start_lon)
    end_pt = geopy_distance(kilometers=length_km).destination(start_pt, wd_toward)
    color = "#3b82f6"
    popup_html = (
        f"<b>{stadium_name or ''}</b><br/>"
        f"Team: {team_name or ''}<br/>"
        f"Wind: {wind_speed_mph:.1f} mph<br/>"
        f"From: {wind_from_deg:.0f}°<br/>"
        f"Toward: {wd_toward:.0f}°"
    )
    folium.CircleMarker(
        location=[start_lat, start_lon],
        radius=7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=folium.Popup(html=popup_html, max_width=320),
    ).add_to(m)
    # Main vector line (toward)
    folium.PolyLine(
        locations=[[start_lat, start_lon], [end_pt.latitude, end_pt.longitude]],
        color=color,
        weight=5,
        opacity=0.9,
    ).add_to(m)
    # Arrowhead: two short lines at ±25° from end
    for offset in (-25.0, 25.0):
        head_pt = geopy_distance(kilometers=length_km * 0.28).destination(
            end_pt, (wd_toward + 180.0 + offset) % 360.0
        )
        folium.PolyLine(
            locations=[[end_pt.latitude, end_pt.longitude], [head_pt.latitude, head_pt.longitude]],
            color=color,
            weight=4,
            opacity=0.9,
        ).add_to(m)
    # Add small crossbar to hint "vane" at the tail
    tail_left = geopy_distance(kilometers=length_km * 0.12).destination(start_pt, (wd_toward + 90.0) % 360.0)
    tail_right = geopy_distance(kilometers=length_km * 0.12).destination(start_pt, (wd_toward + 270.0) % 360.0)
    folium.PolyLine(
        locations=[[tail_left.latitude, tail_left.longitude], [tail_right.latitude, tail_right.longitude]],
        color=color,
        weight=4,
        opacity=0.9,
    ).add_to(m)
    components.html(m.get_root().render(), height=420, scrolling=False)

def rank_azimuth_view(df: pd.DataFrame, min_abs_mph: float = 12.0):
    """Rank stadiums where |Azimuth Component| ≥ threshold; include direction and map."""
    st.subheader(f"Ranked Stadiums — |Azimuth Component| ≥ {min_abs_mph:.0f} mph")
    if df.empty:
        st.info("No rows to display.")
        return
    needed = [
        "team_logo_url","Stadium","Team","latitude","longitude","Azimuth_deg",
        "Wind_Speed_10m_mph","Wind_Component_Azimuth_mph",
        "azimuth_comp_abs_mph","azimuth_direction","Forecast_Time_Local","Timezone"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    # Ensure |Azimuth Component| exists
    if "azimuth_comp_abs_mph" not in df.columns:
        if "Wind_Component_Azimuth_mph" in df.columns:
            df["azimuth_comp_abs_mph"] = df["Wind_Component_Azimuth_mph"].astype(float).abs()
        else:
            df["azimuth_comp_abs_mph"] = np.nan
    # Ensure direction exists
    if "azimuth_direction" not in df.columns:
        if "Wind_Component_Azimuth_mph" in df.columns:
            df["azimuth_direction"] = np.where(df["Wind_Component_Azimuth_mph"].astype(float) >= 0, "blowing out", "blowing in")
        else:
            df["azimuth_direction"] = ""
    # Filter by |Azimuth Component| threshold
    df_filt = df.dropna(subset=["azimuth_comp_abs_mph"]).copy()
    df_filt = df_filt[df_filt["azimuth_comp_abs_mph"].astype(float) >= float(min_abs_mph)]
    if df_filt.empty:
        st.info("No stadiums meet the wind threshold.")
        return
    # Sort by descending |Azimuth Component|
    df_filt = df_filt.sort_values("azimuth_comp_abs_mph", ascending=False).reset_index(drop=True)
    df_filt.insert(0, "Rank", range(1, len(df_filt) + 1))
    st.dataframe(
        df_filt[["Rank"] + [c for c in needed if c in df_filt.columns]],
        width='stretch',
        column_config={
            "team_logo_url": st.column_config.ImageColumn("Logo", width=40),
            "azimuth_comp_abs_mph": st.column_config.NumberColumn("|Azimuth Component| (mph)", format="%.1f"),
            "Wind_Speed_10m_mph": st.column_config.NumberColumn("Wind Speed (mph)", format="%.1f"),
        },
    )
    st.markdown("**Map — Stadiums Meeting Threshold (by |Azimuth Component|)**")
    _render_stadiums_map(df_filt)

# ---------- Main ----------
def main():
    st.title("College Baseball Wind")
    mode = st.sidebar.radio("Mode", ["Testing", "Live", "Demo", "Schedule 2026", "Live Map"] , index=0)

    if mode == "Live Map":
        render_earth_live_map()
        st.caption("Interactive global wind from earth.nullschool.net with time controls.")
        return
    elif mode == "Schedule 2026":
        try:
            games = load_espn_schedule_2026()
        except Exception as e:
            st.error(str(e))
            st.stop()
        try:
            match_report = load_venue_match_report()
        except Exception:
            match_report = pd.DataFrame()
        try:
            stadiums = load_stadium_master()
        except Exception as e:
            st.error(f"Could not load stadium list: {e}")
            st.stop()

        date_sel = st.date_input("Game date (UTC)", value=date.today())
        games = games[games["event_dt_utc"].dt.date == date_sel].copy()
        if games.empty:
            st.info("No games found for the selected date.")
            st.stop()

        games = match_games_to_stadiums(games, match_report, stadiums)
        games = games.sort_values("event_dt_utc")

        max_games = st.slider("Max games to fetch wind for", min_value=10, max_value=200, value=60, step=10)
        if len(games) > max_games:
            st.info(f"Showing first {max_games} games to limit API calls.")
            games = games.head(max_games)

        with st.spinner("Fetching scheduled winds (hourly forecast)..."):
            df_mode = build_schedule_wind_df(games)
        df_mode = attach_team_logos(df_mode, team_col="Team")
        st.caption("Forecasts are limited to the Open-Meteo window (about 16 days).")
    elif mode == "Testing":
        try:
            df_mode = load_testing_data()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        # Ensure Home team (from stadium master) and derive a common forecast date
        try:
            master = load_stadium_master()
            # Merge to add Team where missing
            if "Team" not in df_mode.columns:
                df_mode = df_mode.merge(master[["Stadium","Team"]], on="Stadium", how="left")
            else:
                # Fill missing Team values from master
                df_mode = df_mode.merge(master[["Stadium","Team"]], on="Stadium", how="left", suffixes=("", "_master"))
                df_mode["Team"] = df_mode["Team"].fillna(df_mode.get("Team_master"))
                if "Team_master" in df_mode.columns:
                    df_mode = df_mode.drop(columns=["Team_master"])
        except Exception:
            pass
        # Compute heading date from Forecast_Time_Local if available (most common date)
        heading_date = None
        if "Forecast_Time_Local" in df_mode.columns:
            dt = pd.to_datetime(df_mode["Forecast_Time_Local"], errors="coerce")
            if dt.notna().any():
                dates = dt.dt.date.dropna()
                if not dates.empty:
                    heading_date = str(dates.mode().iloc[0])
        # Attach team logos for display
        df_mode = attach_team_logos(df_mode)
    elif mode == "Demo":
        try:
            df_mode = load_demo_data()
        except FileNotFoundError as e:
            st.warning(f"{e}\nFalling back to first-day games (2026-02-13) from base ESPN data.")
            try:
                base_csv = "espn_2026_college_baseball_games.csv"
                base_df = pd.read_csv(base_csv)
                base_df["event_dt_utc"] = pd.to_datetime(base_df.get("event_date"), utc=True, errors="coerce")
                base_df["date_only"] = base_df["event_dt_utc"].dt.date
                subset = base_df[base_df["date_only"] == date(2026, 2, 13)].copy()
                if subset.empty:
                    st.info("No games found for 2026-02-13.")
                    st.stop()
                st.info("Demo CSV missing; showing first-day games list only. Run the notebook cell to generate demo CSV.")
                st.dataframe(subset[[c for c in ["home","away","venue","location","event_date"] if c in subset.columns]], width='stretch')
                st.stop()
            except Exception as ex:
                st.error(f"Demo fallback failed: {ex}")
                st.stop()
        # Attach team logos for display
        df_mode = attach_team_logos(df_mode)
    else:
        try:
            stads = load_stadium_master()
        except Exception as e:
            st.error(f"Could not load stadium list: {e}")
            st.stop()
        with st.spinner("Fetching live winds for stadiums..."):
            df_mode = build_live_df(stads)
        if df_mode.empty:
            st.info("No wind data available for stadiums.")
            st.stop()
        # Attach team logos for display
        df_mode = attach_team_logos(df_mode)

    # Optional text filter
    q = st.text_input("Filter by stadium/team", value="")
    if q:
        mask = (
            df_mode.get("Stadium", pd.Series(dtype=str)).astype(str).str.contains(q, case=False, na=False)
            | df_mode.get("Team", pd.Series(dtype=str)).astype(str).str.contains(q, case=False, na=False)
        )
        df_mode = df_mode[mask]

    if mode == "Schedule 2026":
        cols = [
            "team_logo_url","event_time_local","status","home","away","venue","Stadium","Team",
            "Wind_Speed_10m_mph","Wind_Direction_From_deg","Wind_Component_Azimuth_mph",
            "azimuth_comp_abs_mph","azimuth_direction","Forecast_Time_Local","Timezone","forecast_note",
        ]
        st.dataframe(
            df_mode[[c for c in cols if c in df_mode.columns]],
            width='stretch',
            column_config={
                "team_logo_url": st.column_config.ImageColumn("Logo", width=40),
                "Wind_Speed_10m_mph": st.column_config.NumberColumn("Wind Speed (mph)", format="%.1f"),
                "Wind_Component_Azimuth_mph": st.column_config.NumberColumn("Azimuth Comp (mph)", format="%.1f"),
                "azimuth_comp_abs_mph": st.column_config.NumberColumn("|Azimuth Component| (mph)", format="%.1f"),
            },
        )
        show_map = st.checkbox("Show map for scheduled games", value=False)
        if show_map:
            st.markdown("**Map — Scheduled Games**")
            _render_stadiums_map(df_mode)
    elif mode == "Live":
        st.subheader("Live — Stadiums with |Azimuth Component| ≥ 12 mph")
        df_live = df_mode.copy()
        if "azimuth_comp_abs_mph" not in df_live.columns:
            if "Wind_Component_Azimuth_mph" in df_live.columns:
                df_live["azimuth_comp_abs_mph"] = df_live["Wind_Component_Azimuth_mph"].astype(float).abs()
            else:
                df_live["azimuth_comp_abs_mph"] = np.nan
        df_live = df_live.dropna(subset=["azimuth_comp_abs_mph"])
        df_live = df_live[df_live["azimuth_comp_abs_mph"].astype(float) >= 12.0]
        if df_live.empty:
            st.info("No stadiums meet the 12 mph azimuth threshold right now.")
            st.stop()
        df_live = df_live.sort_values("azimuth_comp_abs_mph", ascending=False).reset_index(drop=True)
        cols = [
            "team_logo_url","Stadium","Team","Wind_Speed_10m_mph","Wind_Direction_From_deg",
            "Wind_Component_Azimuth_mph","azimuth_comp_abs_mph","azimuth_direction",
            "Forecast_Time_Local","Timezone",
        ]
        st.dataframe(
            df_live[[c for c in cols if c in df_live.columns]],
            width='stretch',
            column_config={
                "team_logo_url": st.column_config.ImageColumn("Logo", width=40),
                "Wind_Speed_10m_mph": st.column_config.NumberColumn("Wind Speed (mph)", format="%.1f"),
                "Wind_Component_Azimuth_mph": st.column_config.NumberColumn("Azimuth Comp (mph)", format="%.1f"),
                "azimuth_comp_abs_mph": st.column_config.NumberColumn("|Azimuth Component| (mph)", format="%.1f"),
            },
        )

        st.markdown("**Live Map — click a team to zoom**")
        team_choices = df_live["Team"].astype(str).unique().tolist()
        selected_team = st.selectbox("Team", team_choices, index=0 if team_choices else None)
        zoom_on_team = st.checkbox("Zoom to team", value=True)
        zoom_level = st.slider("Map zoom", min_value=3, max_value=13, value=10)
        center_lat = None
        center_lon = None
        if selected_team and zoom_on_team:
            row = df_live[df_live["Team"] == selected_team].iloc[0]
            if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
                center_lat = float(row["latitude"])
                center_lon = float(row["longitude"])
        _render_stadiums_map(
            df_live,
            center_lat=center_lat,
            center_lon=center_lon,
            zoom_start=zoom_level,
            auto_center_on_popup=True,
            popup_zoom=zoom_level,
        )
    elif mode == "Testing":
        # Show ranked stadiums by |Azimuth Component| ≥ 12 mph and one map
        rank_azimuth_view(df_mode, min_abs_mph=12.0)
    else:
        # Default Top 5 view for non-Testing modes
        top5_view(
            df_mode,
            heading_date=heading_date if mode == "Testing" else None,
            show_map=(mode == "Testing"),
        )

    # Caption
    if mode == "Testing":
        st.caption("Testing uses local noon forecast per stadium (Open-Meteo).")
    elif mode == "Demo":
        st.caption("Demo shows first-day (2026-02-13) stadium winds using noon tomorrow (from demo CSV).")
    elif mode == "Schedule 2026":
        st.caption("Schedule uses game-time hourly forecasts by stadium azimuth (Open-Meteo).")
    else:
        st.caption("Live uses current hour forecast across stadiums.")


if __name__ == "__main__":
    main()
