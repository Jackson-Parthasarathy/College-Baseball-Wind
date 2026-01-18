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
    def main():
        st.set_page_config(page_title="College Baseball Wind", layout="wide")
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
                st.warning(f"{e}\nFalling back to first-day games (2026-02-13) from base ESPN data.")
                try:
                    base_df = load_games_espx(year=2026)
                    base_df["event_dt_utc"] = base_df.get("event_date", pd.Series(dtype=str)).apply(parse_event_utc)
                    base_df["date_only"] = base_df["event_dt_utc"].apply(lambda x: x.date() if pd.notna(x) else None)
                    df_mode = base_df[base_df["date_only"] == date(2026, 2, 13)].copy()
                    if df_mode.empty:
                        st.info("No games found for 2026-02-13.")
                        st.stop()
                except Exception as ex:
                    st.error(f"Demo fallback failed: {ex}")
                    st.stop()
        else:
            # Live mode: sidebar date picker
            selected_date = st.sidebar.date_input("Select date", value=date(2026, 2, 13))
            target_date = selected_date
            with st.spinner(f"Computing winds for games on {target_date.isoformat()}..."):
                try:
                    df_mode = build_live_games_for_date(target_date)
                except Exception as e:
                    st.error(f"Live mode error: {e}")
                    st.stop()
            if df_mode.empty:
                st.info(f"No games found for {target_date.isoformat()} or venues could not be matched.")
                st.stop()

        # Lightweight text filter
        team_query = st.text_input("Filter by team/stadium name", value="")
        if team_query:
            mask = (
                df_mode.get("stadium_final", pd.Series(dtype=str)).astype(str).str.contains(team_query, case=False, na=False)
                | df_mode.get("Stadium", pd.Series(dtype=str)).astype(str).str.contains(team_query, case=False, na=False)
                | df_mode.get("home", pd.Series(dtype=str)).astype(str).str.contains(team_query, case=False, na=False)
                | df_mode.get("away", pd.Series(dtype=str)).astype(str).str.contains(team_query, case=False, na=False)
            )
            df_mode = df_mode[mask]

        # Show Top 5
        if mode == "Demo":
            # Prefer precomputed demo formatting; otherwise reuse testing-style view
            try:
                top5_view(df_mode)
            except Exception:
                show_top_wind_games(df_mode, mode="testing")
        elif mode == "Testing":
            top5_view(df_mode)
        else:
            top5_view(df_mode)

        # Caption
        if mode == "Testing":
            st.caption("Testing uses local noon forecast per stadium (Open-Meteo).")
        elif mode == "Demo":
            st.caption("Demo shows first-day (2026-02-13) stadium winds using noon tomorrow.")
        else:
            st.caption("Live uses the selected date and forecasts nearest each game's local start time.")


    if __name__ == "__main__":
        main()
            return float(s) % 360.0

        except Exception:
