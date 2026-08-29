from __future__ import annotations

import os
import subprocess
from datetime import date, datetime, timedelta
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
    route_distance_meters,
    route_speeds_mph,
    route_times,
)


DATA_PARSER_VERSION = 10

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


@st.cache_data(show_spinner=False)
def folder_size_bytes(folder: str, export_signature: float | None, route_signature: float | None) -> int:
    del export_signature, route_signature
    total = 0
    try:
        for path in Path(folder).rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    except OSError:
        return 0
    return total


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def workouts_to_frame(workouts: list[WorkoutRecord]) -> pd.DataFrame:
    columns = [
        "uuid",
        "activity_type",
        "start_date",
        "end_date",
        "start_date_local",
        "end_date_local",
        "start_date_local_date",
        "duration_seconds",
        "duration_hours",
        "duration_minutes",
        "total_distance_mi",
        "total_energy_kcal",
        "active_energy_kcal",
        "average_heart_rate_bpm",
        "distance_unit",
        "energy_unit",
        "source_name",
        "source_version",
        "device",
    ]
    rows = []
    for workout in workouts:
        rows.append(
            {
                "uuid": workout.uuid,
                "activity_type": workout.activity_type,
                "start_date": workout.start_date,
                "end_date": workout.end_date,
                "start_date_local": workout.start_date,
                "end_date_local": workout.end_date,
                "start_date_local_date": workout.start_date.date() if workout.start_date else None,
                "duration_seconds": workout.duration_seconds,
                "duration_hours": workout.duration_hours,
                "duration_minutes": workout.duration_minutes,
                "total_distance_mi": (workout.total_distance_m / 1609.344) if workout.total_distance_m is not None else None,
                "total_energy_kcal": workout.total_energy_kcal,
                "active_energy_kcal": workout.active_energy_kcal,
                "average_heart_rate_bpm": workout.average_heart_rate_bpm,
                "distance_unit": workout.total_distance_unit,
                "energy_unit": workout.total_energy_unit,
                "source_name": workout.source_name,
                "source_version": workout.source_version,
                "device": workout.device,
            }
        )
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
        df["end_date"] = pd.to_datetime(df["end_date"], utc=True)
        for column in (
            "duration_seconds",
            "duration_hours",
            "duration_minutes",
            "total_distance_mi",
            "total_energy_kcal",
            "active_energy_kcal",
            "average_heart_rate_bpm",
        ):
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
    if "start_date_local_date" in filtered.columns:
        filtered = filtered[filtered["start_date_local_date"].between(start_date, end_date)]
    if selected_types is not None:
        filtered = filtered[filtered["activity_type"].isin(selected_types)]
    return filtered.sort_values("start_date", ascending=False, na_position="last")


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


def format_local_datetime(value: pd.Timestamp | datetime | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    return value.strftime("%Y-%m-%d %I:%M:%S %p")


def create_time_grouped_frame(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "hours"])

    frequency_map = {
        "Day": "D",
        "Week": "W-SUN",
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
    if granularity == "Week":
        grouped["week_start"] = grouped["period"] - pd.Timedelta(days=6)
        grouped["period_label"] = grouped.apply(
            lambda row: (
                f"Week: {row['week_start'].strftime('%B')} {row['week_start'].day}, "
                f"{row['week_start'].year} - {row['period'].strftime('%B')} "
                f"{row['period'].day}, {row['period'].year}"
            ),
            axis=1,
        )
    else:
        grouped["period_label"] = grouped["period"].map(
            lambda value: value.strftime("%B") + f" {value.day}, {value.year}"
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


def create_type_active_calories(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["activity_type", "active_kcal"])
    return (
        df.groupby("activity_type", as_index=False)
        .agg(active_kcal=("active_energy_kcal", lambda values: values.sum(min_count=1)))
        .dropna(subset=["active_kcal"])
        .sort_values("active_kcal", ascending=False)
    )


def create_type_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "activity_type",
                "workout_count",
                "total_distance_mi",
                "total_energy_kcal",
                "active_energy_kcal",
            ]
        )

    return (
        df.groupby("activity_type", as_index=False)
        .agg(
            workout_count=("uuid", "size"),
            total_distance_mi=("total_distance_mi", lambda values: values.sum(min_count=1)),
            total_energy_kcal=("total_energy_kcal", lambda values: values.sum(min_count=1)),
            active_energy_kcal=("active_energy_kcal", lambda values: values.sum(min_count=1)),
        )
        .sort_values("total_distance_mi", ascending=False, na_position="last")
    )


def display_metrics(df: pd.DataFrame) -> None:
    total_hours = float(df["duration_hours"].sum()) if not df.empty else 0.0
    workout_count = int(len(df))
    avg_duration_seconds = float(df["duration_seconds"].mean()) if not df.empty else 0.0
    total_distance_mi = df["total_distance_mi"].sum(min_count=1) if not df.empty else None
    total_energy_kcal = df["total_energy_kcal"].sum(min_count=1) if not df.empty else None
    active_energy_kcal = df["active_energy_kcal"].sum(min_count=1) if not df.empty else None

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Accumulated Hours", f"{total_hours:.2f}")
    col2.metric("Workout Count", f"{workout_count}")
    col3.metric("Average Duration", format_duration(avg_duration_seconds))
    col4.metric("Total Distance", f"{total_distance_mi:.2f} mi" if pd.notna(total_distance_mi) else "N/A")
    col5.metric("Total Calories", f"{total_energy_kcal:.1f} kcal" if pd.notna(total_energy_kcal) else "N/A")
    col6.metric("Active Calories", f"{active_energy_kcal:.1f} kcal" if pd.notna(active_energy_kcal) else "N/A")


def render_map(
    route: RouteRecord,
    time_window: tuple[datetime, datetime] | None = None,
) -> None:
    if not route.points:
        st.info("No route points available for this workout.")
        return

    start_point = route.points[0]
    end_point = route.points[-1]
    speeds_mph = route_speeds_mph(route)

    def segment_visible(index: int) -> bool:
        # index is 1-based: the segment from points[index - 1] to points[index].
        if time_window is None:
            return True
        start_time = route.points[index - 1].timestamp
        end_time = route.points[index].timestamp
        if start_time is None or end_time is None:
            return True
        try:
            segment_start = min(start_time, end_time)
            segment_end = max(start_time, end_time)
        except TypeError:
            return True
        return segment_start <= time_window[1] and segment_end >= time_window[0]

    visible_indices = [index for index in range(1, len(route.points)) if segment_visible(index)]
    if not visible_indices:
        visible_indices = list(range(1, len(route.points)))
    visible_points = [
        point for index in visible_indices for point in (route.points[index - 1], route.points[index])
    ]
    center_lat = sum(point.latitude for point in visible_points) / len(visible_points)
    center_lon = sum(point.longitude for point in visible_points) / len(visible_points)

    valid_speeds = sorted(speed for speed in speeds_mph if speed is not None and speed >= 0)
    if valid_speeds:
        lower_speed = valid_speeds[max(0, int((len(valid_speeds) - 1) * 0.05))]
        upper_speed = valid_speeds[max(0, int((len(valid_speeds) - 1) * 0.95))]
    else:
        lower_speed = 0.0
        upper_speed = 1.0

    def speed_color(speed_mph: Optional[float]) -> str:
        if speed_mph is None or speed_mph < 0:
            return "#9ca3af"
        if upper_speed <= lower_speed:
            position = 0.5
        else:
            position = max(0.0, min(1.0, (speed_mph - lower_speed) / (upper_speed - lower_speed)))
        if position < 0.5:
            blend = position * 2
            red = int(239 + (250 - 239) * blend)
            green = int(68 + (204 - 68) * blend)
            blue = int(68 + (21 - 68) * blend)
        else:
            blend = (position - 0.5) * 2
            red = int(250 + (34 - 250) * blend)
            green = int(204 + (197 - 204) * blend)
            blue = int(21 + (94 - 21) * blend)
        return f"#{red:02x}{green:02x}{blue:02x}"

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
    for index in visible_indices:
        start = route.points[index - 1]
        end = route.points[index]
        speed = speeds_mph[index]
        speed_label = f"{speed:.1f} mph" if speed is not None else "Speed unavailable"
        folium.PolyLine(
            [(start.latitude, start.longitude), (end.latitude, end.longitude)],
            color=speed_color(speed),
            weight=5,
            opacity=0.9,
            tooltip=f"Speed: {speed_label}",
        ).add_to(satellite_map)

    legend_html = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: rgba(255,255,255,.92); padding: 8px 10px;
                border: 1px solid #d1d5db; border-radius: 5px; font-size: 12px;">
      <div style="font-weight: 600; margin-bottom: 4px;">Route speed</div>
      <div style="width: 180px; height: 10px; background: linear-gradient(90deg, #ef4444, #facc15, #22c55e);"></div>
      <div style="width: 180px; display: flex; justify-content: space-between;">
        <span>Slow</span><span>Fast</span>
      </div>
    </div>
    """
    satellite_map.get_root().html.add_child(folium.Element(legend_html))
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
    route_bounds = [
        [min(point.latitude for point in visible_points), min(point.longitude for point in visible_points)],
        [max(point.latitude for point in visible_points), max(point.longitude for point in visible_points)],
    ]
    satellite_map.fit_bounds(
        route_bounds,
        padding_top_left=(30, 30),
        padding_bottom_right=(30, 30),
        max_zoom=16,
    )
    route_key = f"{route.file_path.resolve()}:{len(route.points)}"
    map_html = f"<!-- route-key: {route_key} -->\n{satellite_map.get_root().render()}"
    components.html(map_html, height=600, scrolling=False)


def render_elevation_profile(route: RouteRecord | None, workout: WorkoutRecord) -> None:
    frame = pd.DataFrame(columns=["time", "elevation_m", "speed_mph"])
    if route:
        times = route_times(route)
        if sum(1 for value in times if value is not None) >= 2:
            frame = pd.DataFrame(
                {
                    "time": pd.to_datetime(times, utc=True, errors="coerce"),
                    "elevation_m": [point.elevation_m for point in route.points],
                    "speed_mph": route_speeds_mph(route),
                }
            ).dropna(subset=["time"])

    heart_rate_frame = pd.DataFrame(columns=["time", "heart_rate_bpm"])
    if workout.heart_rate_samples:
        heart_rate_frame = pd.DataFrame(
            workout.heart_rate_samples,
            columns=["time", "heart_rate_bpm"],
        )
        heart_rate_frame["time"] = pd.to_datetime(heart_rate_frame["time"], utc=True, errors="coerce")
        heart_rate_frame = heart_rate_frame.dropna(subset=["time"])

    if frame.empty and heart_rate_frame.empty:
        if route:
            st.info("No timestamped route or heart-rate data were available for this workout.")
        else:
            st.info("No timestamped heart-rate data were available for this workout.")
        return

    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["elevation_m"],
                mode="lines",
                name="Elevation (m)",
                line=dict(color="#2E86DE", width=2),
            )
        )
        if frame["speed_mph"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=frame["time"],
                    y=frame["speed_mph"],
                    mode="lines",
                    name="Speed (mph)",
                    yaxis="y2",
                    line=dict(color="#E67E22", width=2),
                )
            )

    if not heart_rate_frame.empty:
        fig.add_trace(
            go.Scatter(
                x=heart_rate_frame["time"],
                y=heart_rate_frame["heart_rate_bpm"],
                mode="lines",
                name="Heart Rate (bpm)",
                yaxis="y3",
                line=dict(color="#D81B60", width=2),
            )
        )
    elif workout.average_heart_rate_bpm is not None and not frame.empty:
        fig.add_hline(
            y=workout.average_heart_rate_bpm,
            line_dash="dot",
            line_color="#D81B60",
            annotation_text=f"Average HR: {workout.average_heart_rate_bpm:.0f} bpm",
            annotation_position="top left",
            yref="y3",
        )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Elevation (m)"),
        yaxis2=dict(title="Speed (mph)", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(
            title="Heart Rate (bpm)",
            overlaying="y",
            anchor="free",
            side="left",
            position=0.04,
            showgrid=False,
        ),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")


st.title("Apple Health Workout Explorer")
st.caption("Load an Apple Health export folder containing `export.xml` and, optionally, `workout-routes/`.")


def choose_export_folder(initial_dir: Path | None = None) -> str | None:
    """Open a native folder-picker window and return the chosen path (None if cancelled)."""
    initial = str(initial_dir) if initial_dir and initial_dir.is_dir() else None

    # Preferred: tkinter's native "Select Folder" dialog (real Windows file browser).
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
            root.update()
            selected = filedialog.askdirectory(
                title="Select Apple Health export folder",
                initialdir=initial,
            )
        finally:
            root.attributes("-topmost", False)
            root.destroy()
        if selected:
            return selected
    except Exception:
        pass

    # Fallback: Windows Forms folder browser via PowerShell.
    if os.name == "nt":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "if ($dialog.ShowDialog() -eq 'OK') { $dialog.SelectedPath }"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=600,
            )
            selected = result.stdout.strip()
            if selected:
                return selected
        except (OSError, subprocess.SubprocessError):
            pass
    return None


with st.sidebar:
    st.header("Data Source")
    if st.button("Select Folder…", use_container_width=True):
        raw = (st.session_state.get("export_folder_input") or "").strip()
        chosen = choose_export_folder(Path(raw).expanduser() if raw else Path.home())
        if chosen:
            st.session_state["export_folder_input"] = chosen
        else:
            st.caption("No folder selected — you can also type the path below.")
    export_folder = st.text_input(
        "Path to apple_health_export folder",
        value="",
        key="export_folder_input",
    )
    reload_requested = st.button("Reload data")

if not export_folder:
    st.info("Enter the path to the Apple Health export folder in the sidebar.")
    st.stop()

export_root = Path(export_folder).expanduser()
if not export_root.exists() or not export_root.is_dir():
    st.error(f"Apple Health export folder not found: {export_folder}")
    st.stop()

export_path = export_root / "export.xml"
route_dir = export_root / "workout-routes"
folder_key = str(export_root)
path_changed = st.session_state.get("loaded_export_folder") != folder_key
needs_signature_check = (
    reload_requested
    or path_changed
    or "loaded_export_signature" not in st.session_state
    or "loaded_route_signature" not in st.session_state
)

if needs_signature_check:
    export_signature = file_signature(export_path)
    route_signature = file_signature(route_dir)
else:
    export_signature = st.session_state["loaded_export_signature"]
    route_signature = st.session_state["loaded_route_signature"]

if export_signature is None:
    st.error(f"export.xml not found in: {export_root}")
    st.stop()

data_size = folder_size_bytes(folder_key, export_signature, route_signature)
with st.sidebar:
    st.metric("Export Data Size", format_bytes(data_size))

needs_reload = (
    "workouts" not in st.session_state
    or st.session_state.get("loaded_parser_version") != DATA_PARSER_VERSION
    or st.session_state.get("loaded_export_folder") != folder_key
    or st.session_state.get("loaded_export_signature") != export_signature
    or st.session_state.get("loaded_route_signature") != route_signature
    or "route_lookup" not in st.session_state
    or reload_requested
)

if needs_reload:
    with st.spinner("Parsing Apple Health export..."):
        st.session_state["workouts"] = load_workouts(export_path, export_signature)
        st.session_state["routes"] = load_routes(route_dir, route_signature) if route_signature is not None else []
        used_route_paths: set[Path] = set()
        route_lookup: dict[str, RouteRecord] = {}
        for workout in st.session_state["workouts"]:
            available_routes = [route for route in st.session_state["routes"] if route.file_path not in used_route_paths]
            matched = match_route_to_workout(workout, available_routes)
            if matched:
                route_lookup[workout.uuid] = matched
                used_route_paths.add(matched.file_path)
        st.session_state["route_lookup"] = route_lookup
        st.session_state["loaded_parser_version"] = DATA_PARSER_VERSION
        st.session_state["loaded_export_folder"] = folder_key
        st.session_state["loaded_export_signature"] = export_signature
        st.session_state["loaded_route_signature"] = route_signature

workouts = st.session_state.get("workouts", [])
routes = st.session_state.get("routes", [])

df = workouts_to_frame(workouts)
route_lookup = st.session_state.get("route_lookup", {})

if not df.empty:
    df["route_status"] = df["uuid"].map(lambda value: "Route Available" if value in route_lookup else "No Route Data")
    df["route_file"] = df["uuid"].map(lambda value: str(route_lookup[value].file_path.name) if value in route_lookup else "")
    for workout_uuid, route in route_lookup.items():
        row_mask = (df["uuid"] == workout_uuid) & df["total_distance_mi"].isna()
        if row_mask.any():
            df.loc[row_mask, "total_distance_mi"] = route_distance_meters(route) / 1609.344

    pace_mask = (
        df["total_distance_mi"].notna()
        & (df["total_distance_mi"] > 0)
        & df["duration_minutes"].notna()
        & (df["duration_minutes"] > 0)
    )
    df["pace_min_per_mi"] = (df["duration_minutes"] / df["total_distance_mi"]).where(pace_mask)
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

min_date = df["start_date_local_date"].dropna().min()
max_date = df["start_date_local_date"].dropna().max()

with st.sidebar:
    st.header("Filters")
    today = date.today()
    date_preset = st.selectbox("Date range preset", ["All time", "Year to date", "Past year", "Custom"])
    if date_preset == "Year to date":
        selected_start, selected_end = date(today.year, 1, 1), today
        st.caption(f"Showing workouts from {selected_start:%Y-%m-%d} through {selected_end:%Y-%m-%d}.")
    elif date_preset == "Past year":
        selected_start, selected_end = today - timedelta(days=365), today
        st.caption(f"Showing workouts from {selected_start:%Y-%m-%d} through {selected_end:%Y-%m-%d}.")
    elif date_preset == "All time":
        selected_start, selected_end = min_date, max_date
    else:
        default_range = (min_date, max_date) if min_date and max_date else (today, today)
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
                        customdata=time_grouped["period_label"],
                        hovertemplate="%{customdata}<br>Hours: %{y:.2f}<extra></extra>",
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
            st.plotly_chart(fig, width="stretch")

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
                title="Time Per Workout Type",
                xaxis_title="Workout Type",
                yaxis_title="Hours",
                template="plotly_white",
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, width="stretch")

    active_calories = create_type_active_calories(filtered_df)
    if active_calories.empty:
        st.info("No active calorie data available for the current filters.")
    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=active_calories["activity_type"],
                    y=active_calories["active_kcal"],
                    marker_color="#E67E22",
                    hovertemplate="%{x}<br>Active Calories: %{y:.1f} kcal<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Active Calories Per Workout Type",
            xaxis_title="Workout Type",
            yaxis_title="Active Calories (kcal)",
            template="plotly_white",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Distance and Energy by Workout Type")
    type_totals = create_type_totals(filtered_df)
    if type_totals.empty:
        st.info("No workout type totals available.")
    else:
        type_totals_display = type_totals.copy()
        type_totals_display["total_distance_mi"] = type_totals_display["total_distance_mi"].round(2)
        type_totals_display["total_energy_kcal"] = type_totals_display["total_energy_kcal"].round(1)
        type_totals_display["active_energy_kcal"] = type_totals_display["active_energy_kcal"].round(1)
        type_totals_display = type_totals_display.rename(
            columns={
                "activity_type": "Workout Type",
                "workout_count": "Workouts",
                "total_distance_mi": "Total Miles",
                "total_energy_kcal": "Total Calories (kcal)",
                "active_energy_kcal": "Active Calories (kcal)",
            }
        )
        table_height = 45 + (len(type_totals_display) * 35)
        st.dataframe(
            type_totals_display,
            width="stretch",
            hide_index=True,
            height=table_height,
        )

    st.subheader("Filtered Workouts")
    summary_columns = [
        "start_date_local",
        "end_date_local",
        "activity_type",
        "duration_hours",
        "total_distance_mi",
        "pace_min_per_mi",
        "total_energy_kcal",
        "active_energy_kcal",
        "average_heart_rate_bpm",
    ]
    display_df = filtered_df[summary_columns].copy()
    display_df["start_date_local"] = display_df["start_date_local"].map(format_local_datetime)
    display_df["end_date_local"] = display_df["end_date_local"].map(format_local_datetime)
    display_df["duration_hours"] = pd.to_numeric(display_df["duration_hours"], errors="coerce").round(2)
    display_df["total_distance_mi"] = pd.to_numeric(display_df["total_distance_mi"], errors="coerce").round(2)
    display_df["pace_min_per_mi"] = pd.to_numeric(display_df["pace_min_per_mi"], errors="coerce").round(2)
    display_df["total_energy_kcal"] = pd.to_numeric(display_df["total_energy_kcal"], errors="coerce").round(1)
    display_df["active_energy_kcal"] = pd.to_numeric(display_df["active_energy_kcal"], errors="coerce").round(1)
    display_df["average_heart_rate_bpm"] = pd.to_numeric(
        display_df["average_heart_rate_bpm"], errors="coerce"
    ).round(0)
    display_df = display_df.rename(
            columns={
                "start_date_local": "start_date",
                "end_date_local": "end_date",
                "total_distance_mi": "total_distance_miles",
                "pace_min_per_mi": "pace_min_per_mi",
                "average_heart_rate_bpm": "average_heart_rate_bpm",
            }
    )
    st.dataframe(display_df, width="stretch", hide_index=True, height=700)

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
                f"{format_local_datetime(row['start_date_local'])} | {row['activity_type']} | "
                f"{format_duration(row['duration_seconds'])} | "
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
        detail_cols[0].metric("Start Time (Local)", format_local_datetime(selected_row["start_date_local"]))
        detail_cols[1].metric("End Time (Local)", format_local_datetime(selected_row["end_date_local"]))
        detail_cols[2].metric("Duration", format_duration(selected_row["duration_seconds"]))
        detail_cols[3].metric("Route Status", "Route Available" if selected_route else "No Route Data")

        distance_value = selected_row["total_distance_mi"]
        energy_value = selected_row["total_energy_kcal"]
        active_energy_value = selected_row["active_energy_kcal"]
        heart_rate_value = selected_row["average_heart_rate_bpm"]
        pace_value = selected_row["pace_min_per_mi"] if "pace_min_per_mi" in selected_row.index else None
        metric_cols = st.columns(4)
        metric_cols[0].metric("Distance", f"{distance_value:.2f} mi" if pd.notna(distance_value) else "N/A")
        metric_cols[1].metric(
            "Avg Pace",
            f"{pace_value:.2f} min/mi" if pd.notna(pace_value) else "N/A",
        )
        metric_cols[2].metric("Total Calories", f"{energy_value:.1f} kcal" if pd.notna(energy_value) else "N/A")
        metric_cols[3].metric(
            "Active Calories",
            f"{active_energy_value:.1f} kcal" if pd.notna(active_energy_value) else "N/A",
        )

        secondary_metric_cols = st.columns(2)
        secondary_metric_cols[0].metric(
            "Average Heart Rate",
            f"{heart_rate_value:.0f} bpm" if pd.notna(heart_rate_value) else "N/A",
        )
        secondary_metric_cols[1].metric("Workout Type", selected_row["activity_type"])

        st.markdown("### Workout Details")
        st.write(
            {
                "Workout ID": selected_workout.uuid,
                "Source": selected_workout.source_name or "Unknown",
                "Device": selected_workout.device or "Unknown",
                "Route File": selected_route.file_path.name if selected_route else None,
            }
        )

        if selected_route:
            st.markdown("### Route Map")
            route_timestamps = [timestamp for timestamp in route_times(selected_route) if timestamp is not None]
            time_window: tuple[datetime, datetime] | None = None
            if len(route_timestamps) >= 2:
                route_start_time = min(route_timestamps)
                total_route_seconds = int((max(route_timestamps) - route_start_time).total_seconds())
                total_route_minutes = total_route_seconds / 60.0
                if total_route_seconds >= 2:
                    window = st.slider(
                        "Show route between (minutes after start)",
                        min_value=0.0,
                        max_value=total_route_minutes,
                        value=(0.0, total_route_minutes),
                        step=0.1,
                        key=f"route_time_window_minutes_{selected_uuid}",
                        help="Only the path and speed segments recorded inside this time window are drawn on the map.",
                    )
                    time_window = (
                        route_start_time + timedelta(minutes=window[0]),
                        route_start_time + timedelta(minutes=window[1]),
                    )
                    st.caption(
                        f"Map shows the route between {format_duration(window[0] * 60.0)} and "
                        f"{format_duration(window[1] * 60.0)} after the start."
                    )
                else:
                    st.caption("This route is too short to filter by time.")
            else:
                st.caption("This route has no per-point timestamps, so time filtering is unavailable.")
            render_map(selected_route, time_window)
        else:
            st.info("This workout does not have a matched GPS route in the provided route directory.")

        st.markdown("### Elevation / Speed / Heart Rate Over Time")
        render_elevation_profile(selected_route, selected_workout)
