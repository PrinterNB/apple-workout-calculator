from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import folium
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from health_parser import (
    RouteRecord,
    WorkoutRecord,
    match_route_to_workout,
    parse_route_directory,
    parse_workout_export,
    route_speeds_kph,
    route_times,
)


DATA_PARSER_VERSION = 2

st.set_page_config(page_title="Apple Workout Calculator", layout="wide")


@st.cache_data(show_spinner=False)
def load_workouts(export_path: str, file_signature: float | None) -> list[WorkoutRecord]:
    return parse_workout_export(export_path)


@st.cache_data(show_spinner=False)
def load_routes(route_dir: str, file_signature: float | None) -> list[RouteRecord]:
    return parse_route_directory(route_dir)


def file_signature(path: str | Path | None) -> float | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.is_file():
            return p.stat().st_mtime
        latest = 0.0
        for child in p.rglob("*"):
            if child.is_file():
                latest = max(latest, child.stat().st_mtime)
        return latest or p.stat().st_mtime
    except OSError:
        return None


def workouts_to_frame(workouts: list[WorkoutRecord]) -> pd.DataFrame:
    rows = []
    for workout in workouts:
        rows.append(
            {
                "uuid": workout.uuid,
                "activity_type": workout.activity_type,
                "start_date": workout.start_date,
                "end_date": workout.end_date,
                "duration_seconds": workout.duration_seconds,
                "duration_hours": workout.duration_hours,
                "duration_minutes": workout.duration_minutes,
                "total_distance_mi": (workout.total_distance_m / 1609.344) if workout.total_distance_m is not None else None,
                "total_energy_kcal": workout.total_energy_kcal,
                "distance_unit": workout.total_distance_unit,
                "energy_unit": workout.total_energy_unit,
                "source_name": workout.source_name,
                "source_version": workout.source_version,
                "device": workout.device,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
        df["end_date"] = pd.to_datetime(df["end_date"], utc=True)
        for column in ("duration_seconds", "duration_hours", "duration_minutes", "total_distance_mi", "total_energy_kcal"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def apply_filters(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    selected_types: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    if "start_date" in filtered.columns:
        filtered = filtered[filtered["start_date"].dt.date.between(start_date, end_date)]
    if selected_types is not None:
        filtered = filtered[filtered["activity_type"].isin(selected_types)]
    return filtered.sort_values("start_date")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_datetime(value: pd.Timestamp | datetime | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    return value.strftime("%Y-%m-%d %H:%M")


def create_time_grouped_frame(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "hours"])

    frequency_map = {
        "Day": "D",
        "Week": "W-MON",
        "Month": "MS",
    }
    freq = frequency_map[granularity]
    temp = df.copy()
    temp = temp.dropna(subset=["start_date"])
    if temp.empty:
        return pd.DataFrame(columns=["period", "hours"])

    grouped = (
        temp.set_index("start_date")
        .groupby(pd.Grouper(freq=freq))["duration_hours"]
        .sum()
        .reset_index()
        .rename(columns={"start_date": "period", "duration_hours": "hours"})
    )
    return grouped


def create_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["activity_type", "hours"])
    return (
        df.groupby("activity_type", as_index=False)["duration_hours"]
        .sum()
        .sort_values("duration_hours", ascending=False)
        .rename(columns={"duration_hours": "hours"})
    )


def display_metrics(df: pd.DataFrame) -> None:
    total_hours = float(df["duration_hours"].sum()) if not df.empty else 0.0
    workout_count = int(len(df))
    avg_duration_seconds = float(df["duration_seconds"].mean()) if not df.empty else 0.0
    total_distance_mi = df["total_distance_mi"].sum(min_count=1) if not df.empty else None
    total_energy_kcal = df["total_energy_kcal"].sum(min_count=1) if not df.empty else None

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Accumulated Hours", f"{total_hours:.2f}")
    col2.metric("Workout Count", f"{workout_count}")
    col3.metric("Average Duration", format_duration(avg_duration_seconds))
    col4.metric("Total Distance", f"{total_distance_mi:.2f} mi" if pd.notna(total_distance_mi) else "N/A")
    col5.metric("Total Energy", f"{total_energy_kcal:.1f} kcal" if pd.notna(total_energy_kcal) else "N/A")


def render_map(route: RouteRecord) -> None:
    if not route.points:
        st.info("No route points available for this workout.")
        return

    coordinates = [(point.latitude, point.longitude) for point in route.points]
    start_point = route.points[0]
    end_point = route.points[-1]
    center_lat = sum(point.latitude for point in route.points) / len(route.points)
    center_lon = sum(point.longitude for point in route.points) / len(route.points)

    satellite_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=False,
    ).add_to(satellite_map)
    folium.PolyLine(
        coordinates,
        color="#22a7f0",
        weight=5,
        opacity=0.9,
        tooltip="Workout route",
    ).add_to(satellite_map)
    folium.Marker(
        [start_point.latitude, start_point.longitude],
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(satellite_map)
    folium.Marker(
        [end_point.latitude, end_point.longitude],
        tooltip="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(satellite_map)
    satellite_map.fit_bounds(coordinates)
    components.html(satellite_map.get_root().render(), height=600, scrolling=False)


def render_elevation_profile(route: RouteRecord) -> None:
    times = route_times(route)
    elevations = [point.elevation_m for point in route.points]
    speeds = route_speeds_kph(route)

    has_time_series = sum(1 for value in times if value is not None) >= 2
    if not has_time_series:
        st.info("This route does not contain enough timestamps to build an elevation/speed time series.")
        return

    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(times, utc=True, errors="coerce"),
            "elevation_m": elevations,
            "speed_kph": speeds,
        }
    ).dropna(subset=["time"])

    if frame.empty:
        st.info("No timestamped trackpoints were available for the profile.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=frame["elevation_m"],
            mode="lines",
            name="Elevation (m)",
            line=dict(color="#2E86DE", width=2),
        )
    )
    if frame["speed_kph"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["speed_kph"],
                mode="lines",
                name="Speed (kph)",
                yaxis="y2",
                line=dict(color="#E67E22", width=2),
            )
        )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Elevation (m)"),
        yaxis2=dict(title="Speed (kph)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


st.title("Apple Health Workout Explorer")
st.caption("Load an Apple Health export folder containing `export.xml` and, optionally, `workout-routes/`.")

with st.sidebar:
    st.header("Data Source")
    export_folder = st.text_input("Path to apple_health_export folder", value="")

if not export_folder:
    st.info("Enter the path to the Apple Health export folder in the sidebar.")
    st.stop()

export_root = Path(export_folder).expanduser()
if not export_root.exists() or not export_root.is_dir():
    st.error(f"Apple Health export folder not found: {export_folder}")
    st.stop()

export_path = export_root / "export.xml"
route_dir = export_root / "workout-routes"
export_signature = file_signature(export_path)
route_signature = file_signature(route_dir)

if export_signature is None:
    st.error(f"export.xml not found in: {export_root}")
    st.stop()

needs_reload = (
    "workouts" not in st.session_state
    or st.session_state.get("loaded_parser_version") != DATA_PARSER_VERSION
    or st.session_state.get("loaded_export_folder") != str(export_root)
    or st.session_state.get("loaded_export_signature") != export_signature
    or st.session_state.get("loaded_route_signature") != route_signature
)

if needs_reload:
    with st.spinner("Parsing Apple Health export..."):
        st.session_state["workouts"] = load_workouts(export_path, export_signature)
        st.session_state["routes"] = load_routes(route_dir, route_signature) if route_signature is not None else []
        st.session_state["loaded_parser_version"] = DATA_PARSER_VERSION
        st.session_state["loaded_export_folder"] = str(export_root)
        st.session_state["loaded_export_signature"] = export_signature
        st.session_state["loaded_route_signature"] = route_signature

workouts = st.session_state.get("workouts", [])
routes = st.session_state.get("routes", [])

df = workouts_to_frame(workouts)
route_lookup: dict[str, RouteRecord] = {}
if workouts and routes:
    for workout in workouts:
        matched = match_route_to_workout(workout, routes)
        if matched:
            route_lookup[workout.uuid] = matched

if not df.empty:
    df["route_status"] = df["uuid"].map(lambda value: "Route Available" if value in route_lookup else "No Route Data")
    df["route_file"] = df["uuid"].map(lambda value: str(route_lookup[value].file_path.name) if value in route_lookup else "")
else:
    df["route_status"] = pd.Series(dtype="string")
    df["route_file"] = pd.Series(dtype="string")

if df.empty:
    st.warning("No workouts were parsed from the export file.")
    st.stop()

if df["total_distance_mi"].isna().all() and df["total_energy_kcal"].isna().all():
    st.warning(
        "This export contains no readable total distance or energy values on its Workout records. "
        "The app can still show duration and route data."
    )

min_date = df["start_date"].dropna().dt.date.min()
max_date = df["start_date"].dropna().dt.date.max()

with st.sidebar:
    st.header("Filters")
    default_range = (min_date, max_date) if min_date and max_date else (date.today(), date.today())
    selected_dates = st.date_input("Date range", value=default_range)
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        selected_start, selected_end = selected_dates
    else:
        selected_start = selected_end = selected_dates
date_filtered_df = apply_filters(df, selected_start, selected_end)
available_types = sorted(date_filtered_df["activity_type"].fillna("Unknown").unique().tolist())

with st.sidebar:
    selected_activity_types = st.multiselect(
        "Activity types",
        options=available_types,
        default=available_types,
        help="Only activity types present in the selected date range are listed.",
    )

filtered_df = apply_filters(df, selected_start, selected_end, selected_activity_types)

tab1, tab2 = st.tabs(["Workout Accumulator", "Individual Workout Route Inspector"])

with tab1:
    display_metrics(filtered_df)

    c1, c2 = st.columns([1, 1])
    with c1:
        granularity = st.selectbox("Group accumulation by", ["Day", "Week", "Month"])
        time_grouped = create_time_grouped_frame(filtered_df, granularity)
        if time_grouped.empty:
            st.info("No workouts match the current filters.")
        else:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=time_grouped["period"],
                        y=time_grouped["hours"],
                        marker_color="#2E86DE",
                    )
                ]
            )
            fig.update_layout(
                title="Accumulated Workout Time",
                xaxis_title="Period",
                yaxis_title="Hours",
                template="plotly_white",
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        type_breakdown = create_type_breakdown(filtered_df)
        if type_breakdown.empty:
            st.info("No breakdown data available.")
        else:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=type_breakdown["activity_type"],
                        y=type_breakdown["hours"],
                        marker_color="#16A085",
                    )
                ]
            )
            fig.update_layout(
                title="Workout Type Breakdown",
                xaxis_title="Workout Type",
                yaxis_title="Hours",
                template="plotly_white",
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filtered Workouts")
    summary_columns = [
        "start_date",
        "end_date",
        "activity_type",
        "duration_hours",
        "total_distance_mi",
        "total_energy_kcal",
        "route_status",
        "route_file",
    ]
    display_df = filtered_df[summary_columns].copy()
    display_df["start_date"] = display_df["start_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["end_date"] = display_df["end_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["duration_hours"] = pd.to_numeric(display_df["duration_hours"], errors="coerce").round(2)
    display_df["total_distance_mi"] = pd.to_numeric(display_df["total_distance_mi"], errors="coerce").round(2)
    display_df = display_df.rename(columns={"total_distance_mi": "total_distance_miles"})
    display_df["total_energy_kcal"] = pd.to_numeric(display_df["total_energy_kcal"], errors="coerce").round(1)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Workout Selector")
    if filtered_df.empty:
        st.info("No workouts available for inspection with the current filters.")
    else:
        selector_frame = (
            filtered_df.copy()
            .sort_values("start_date", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        selector_frame["label"] = selector_frame.apply(
            lambda row: (
                f"{format_datetime(row['start_date'])} | {row['activity_type']} | {format_duration(row['duration_seconds'])} | "
                f"{str(row['uuid'])[:8]}"
            ),
            axis=1,
        )
        selected_label = st.selectbox("Choose a workout", selector_frame["label"].tolist())
        selected_row = selector_frame.loc[selector_frame["label"] == selected_label].iloc[0]
        selected_uuid = selected_row["uuid"]
        selected_workout = next(workout for workout in workouts if workout.uuid == selected_uuid)
        selected_route = route_lookup.get(selected_uuid)

        detail_cols = st.columns(4)
        detail_cols[0].metric("Start Time (UTC)", format_datetime(selected_row["start_date"]))
        detail_cols[1].metric("End Time (UTC)", format_datetime(selected_row["end_date"]))
        detail_cols[2].metric("Duration", format_duration(selected_row["duration_seconds"]))
        detail_cols[3].metric("Route Status", "Route Available" if selected_route else "No Route Data")

        metric_cols = st.columns(3)
        distance_value = selected_row["total_distance_mi"]
        energy_value = selected_row["total_energy_kcal"]
        metric_cols[0].metric("Distance", f"{distance_value:.2f} mi" if pd.notna(distance_value) else "N/A")
        metric_cols[1].metric("Energy Burned", f"{energy_value:.1f} kcal" if pd.notna(energy_value) else "N/A")
        metric_cols[2].metric("Workout Type", selected_row["activity_type"])

        st.markdown("### Workout Details")
        st.write(
            {
                "UUID": selected_workout.uuid,
                "Source": selected_workout.source_name or "Unknown",
                "Device": selected_workout.device or "Unknown",
                "Route File": selected_route.file_path.name if selected_route else None,
            }
        )

        if selected_route:
            st.markdown("### Route Map")
            render_map(selected_route)

            st.markdown("### Route Elevation / Speed Profile")
            render_elevation_profile(selected_route)
        else:
            st.info("This workout does not have a matched GPS route in the provided route directory.")
