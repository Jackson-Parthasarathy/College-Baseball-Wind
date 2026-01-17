# streamlit_app.py
# ----------------
# Division I Baseball Dashboard with map, filters, and optional password.
# Adapted to work with ESPN datasets in this repo.

import os
import datetime as dt
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import pydeck as pdk


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

# Map defaults
DEFAULT_VIEW_STATE = pdk.ViewState(
    latitude=37.8, longitude=-96.9, zoom=3.8, pitch=30
)

# Colors (RGBA) for map markers
COLOR_HOME = [0, 168, 232, 180]   # light blue
COLOR_AWAY = [255, 111, 97, 180]  # coral


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


# =========================
# ------- DATA I/O --------
# =========================

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


def map_view(df: pd.DataFrame):
    st.subheader("🗺️ Map")

    if not {LAT_COL, LON_COL}.issubset(df.columns):
        st.info("No latitude/longitude columns found to render the map.")
        return

    map_df = df.dropna(subset=[LAT_COL, LON_COL]).copy()
    if map_df.empty:
        st.info("No geocoded rows to display.")
        return

    # Marker layer per venue (aggregate to avoid tons of overlapping points)
    group_cols = [VENUE_COL, CITY_COL, STATE_COL, LAT_COL, LON_COL]
    for c in [VENUE_COL, CITY_COL, STATE_COL]:
        if c not in map_df.columns:
            map_df[c] = ""

    agg = (map_df
           .groupby(group_cols, dropna=True)
           .agg(
               games=(ID_COL, "count") if ID_COL in map_df.columns else (DATE_COL, "count"),
               first_date=(DATE_COL, "min") if DATE_COL in map_df.columns else (DATE_COL, "min"),
               last_date=(DATE_COL, "max") if DATE_COL in map_df.columns else (DATE_COL, "max"),
           )
           .reset_index())

    tooltip_text = [
        f"{row.get(VENUE_COL, '')} — {row.get(CITY_COL, '')}, {row.get(STATE_COL, '')}\n"
        f"Games: {row['games']}\n"
        f"Dates: {row.get('first_date','')} → {row.get('last_date','')}"
        for _, row in agg.iterrows()
    ]
    agg["tooltip"] = tooltip_text

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=agg,
        get_position=[LON_COL, LAT_COL],
        get_fill_color=COLOR_HOME,
        get_radius=12000,
        pickable=True,
        auto_highlight=True,
    )

    r = pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v11",
        initial_view_state=DEFAULT_VIEW_STATE,
        layers=[scatter],
        tooltip={"text": "{tooltip}"},
    )

    st.pydeck_chart(r)

    with st.expander("Show venue table"):
        st.dataframe(
            agg[[VENUE_COL, CITY_COL, STATE_COL, "games", "first_date", "last_date"]],
            use_container_width=True,
            hide_index=True
        )


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

    # Tabs
    t1, t2, t3 = st.tabs(["Map", "Games", "Teams"])
    with t1:
        map_view(fdf)
    with t2:
        games_table(fdf)
    with t3:
        team_spotlight(fdf)

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit and PyDeck. Data © You.")


if __name__ == "__main__":
    main()
