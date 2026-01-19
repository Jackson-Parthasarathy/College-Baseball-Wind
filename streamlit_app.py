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


def render_live_wind_map():
        st.subheader("Live Wind — Global (Leaflet Velocity)")
        html = """
        <!doctype html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
            <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/gh/onaci/leaflet-velocity/dist/leaflet-velocity.min.css\" />
            <style>
                html, body { height: 100%; margin: 0; }
                #map { width: 100%; height: 600px; }
                .leaflet-control { font-size: 14px; }
            </style>
        </head>
        <body>
            <div id=\"map\"></div>
            <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
            <script src=\"https://cdn.jsdelivr.net/gh/onaci/leaflet-velocity/dist/leaflet-velocity.min.js\"></script>
            <script>
                const map = L.map('map', { center: [20, 0], zoom: 2 });
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);

                fetch('https://cdn.jsdelivr.net/gh/onaci/leaflet-velocity/demo/wind-global.json')
                    .then(r => r.json())
                    .then(data => {
                        const velocityLayer = L.velocityLayer({
                            displayValues: true,
                            displayOptions: {
                                velocityType: 'Global Wind',
                                position: 'bottomleft',
                                emptyString: 'No velocity data',
                                angleConvention: 'bearingCW',
                                speedUnit: 'kt',
                                directionString: 'Direction',
                                speedString: 'Speed'
                            },
                            data: data,
                            minVelocity: 0,
                            maxVelocity: 25,
                            velocityScale: 0.005,
                            opacity: 0.97,
                            paneName: 'overlayPane'
                        });
                        velocityLayer.addTo(map);
                    })
                    .catch(err => {
                        console.error('Failed to load wind data:', err);
                    });
            </script>
        </body>
        </html>
        """
        components.html(html, height=650, scrolling=False)


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

# ---------- Main ----------
def main():
    st.title("College Baseball Wind")
    mode = st.sidebar.radio("Mode", ["Testing", "Live", "Demo", "Live Wind"], index=0)

    if mode == "Live Wind":
        render_live_wind_map()
        st.caption("Global wind visualisation using leaflet-velocity demo data.")
        return
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

    # Show Top 5
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
    else:
        st.caption("Live uses current hour forecast across stadiums.")


if __name__ == "__main__":
    main()
