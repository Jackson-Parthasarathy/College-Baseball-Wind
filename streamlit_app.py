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

# ---------- Testing/Demo view ----------
def top5_view(df: pd.DataFrame, heading_date: str | None = None):
    title = "Top 5 Stadiums by Azimuth-Aligned Wind"
    if heading_date:
        title = f"{title} — {heading_date}"
    st.subheader(title)
    if df.empty:
        st.info("No rows to display.")
        return
    needed = [
        "Stadium","Team","latitude","longitude","Azimuth_deg",
        "Wind_Speed_10m_mph","Wind_Component_Azimuth_mph",
        "azimuth_comp_abs_mph","azimuth_direction","Forecast_Time_Local","Timezone"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    df_top = df.dropna(subset=["azimuth_comp_abs_mph"]).sort_values("azimuth_comp_abs_mph", ascending=False).head(5)
    st.dataframe(df_top[[c for c in needed if c in df_top.columns]], use_container_width=True)
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
    st.altair_chart(chart, use_container_width=True)

# ---------- Main ----------
def main():
    st.title("College Baseball Wind")
    mode = st.sidebar.radio("Mode", ["Testing", "Live", "Demo"], index=0)

    if mode == "Testing":
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
                st.dataframe(subset[[c for c in ["home","away","venue","location","event_date"] if c in subset.columns]], use_container_width=True)
                st.stop()
            except Exception as ex:
                st.error(f"Demo fallback failed: {ex}")
                st.stop()
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

    # Optional text filter
    q = st.text_input("Filter by stadium/team", value="")
    if q:
        mask = (
            df_mode.get("Stadium", pd.Series(dtype=str)).astype(str).str.contains(q, case=False, na=False)
            | df_mode.get("Team", pd.Series(dtype=str)).astype(str).str.contains(q, case=False, na=False)
        )
        df_mode = df_mode[mask]

    # Show Top 5
    top5_view(df_mode, heading_date=heading_date if mode == "Testing" else None)

    # Caption
    if mode == "Testing":
        st.caption("Testing uses local noon forecast per stadium (Open-Meteo).")
    elif mode == "Demo":
        st.caption("Demo shows first-day (2026-02-13) stadium winds using noon tomorrow (from demo CSV).")
    else:
        st.caption("Live uses current hour forecast across stadiums.")


if __name__ == "__main__":
    main()
