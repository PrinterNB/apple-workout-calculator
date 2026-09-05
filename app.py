from __future__ import annotations

import math
import os
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from health_parser import (
    MetricSample,
    RouteRecord,
    WorkoutRecord,
    _naive_local,
    match_route_to_workout,
    parse_export_all,
    parse_route_directory,
    route_distance_meters,
    route_speeds_mph,
    route_times,
    set_progress_callback,
)


DATA_PARSER_VERSION = 24

st.set_page_config(page_title="Apple Workout Calculator", layout="wide")


@st.cache_data(show_spinner=False)
def load_export(
    export_path: str | Path,
    file_signature: float | None,
    range_start: date | None,
    range_end: date | None,
) -> tuple[list[WorkoutRecord], dict[str, list[MetricSample]], dict[str, int]]:
    """Workouts and health metrics for the selected range.

    export.xml is not date-sorted, so even a narrow range streams the whole
    file; the per-range result is memoized by st.cache_data for the session.
    """
    return parse_export_all(export_path, range_start, range_end)


@st.cache_data(show_spinner=False)
def load_routes(
    route_dir: str | Path,
    file_signature: float | None,
    range_start: date | None,
    range_end: date | None,
) -> list[RouteRecord]:
    return parse_route_directory(route_dir, range_start, range_end)


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


def format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class ParseProgressUI:
    """Progress bar + live ETA caption covering the export processing stages."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.current = -1
        self.completed = 0
        self.stage_started = 0.0
        self.last_report = 0.0
        self.bar: st.delta_generator.DeltaGenerator | None = None
        self.status: st.delta_generator.DeltaGenerator | None = None

    def render(self) -> "ParseProgressUI":
        self.bar = st.progress(0.0)
        self.status = st.empty()
        return self

    def begin_stage(self, index: int) -> None:
        self.current = index
        self.stage_started = time.monotonic()

    def report(self, done: int, total: int) -> None:
        """Called from the parsers via the progress callback (throttled to ~3x/second)."""
        now = time.monotonic()
        if now - self.last_report < 0.35 and done < total:
            return
        self.last_report = now
        label = self.labels[self.current] if 0 <= self.current < len(self.labels) else "Processing"
        text = f"**{label}** — {done:,} / {total:,}"
        elapsed = now - self.stage_started
        if total and 0 < done < total and elapsed > 0:
            text += f" — about {format_eta((total - done) * elapsed / done)} left"
        self.status.markdown(text)
        fraction = min(1.0, done / total) if total else 0.0
        self.bar.progress(min((self.completed + fraction) / len(self.labels), 0.995))

    def finish_stage(self) -> None:
        self.completed = self.current + 1
        self.bar.progress(min(1.0, self.completed / len(self.labels)))

    def dispose(self) -> None:
        if self.bar is not None:
            self.bar.progress(1.0)
            self.bar.empty()
        if self.status is not None:
            self.status.empty()


def coerce_to_date(value: object) -> date:
    """Normalize the loose values st.date_input returns (date, datetime, ISO string)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


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
    start_date: date | None,
    end_date: date | None,
    selected_types: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    if "start_date_local_date" in filtered.columns:
        if start_date is not None:
            filtered = filtered[filtered["start_date_local_date"] >= start_date]
        if end_date is not None:
            filtered = filtered[filtered["start_date_local_date"] <= end_date]
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


def format_date_short(value) -> str:
    """Compact 'Jun 14, 2026' label for a record's date; empty string if none."""
    try:
        if value is None or value is pd.NaT or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime("%b %d, %Y")


def best_single_value(df: pd.DataFrame, column: str, ascending: bool = False):
    """(best value, that row's local date, its activity type) for the extreme of a column, or (None, None, None)."""
    if df.empty or column not in df.columns:
        return None, None, None
    sub = df.dropna(subset=[column])
    if sub.empty:
        return None, None, None
    idx = sub[column].idxmin() if ascending else sub[column].idxmax()
    row = df.loc[idx]
    activity_type = row.get("activity_type")
    if activity_type is None or (isinstance(activity_type, float) and math.isnan(activity_type)):
        activity_type = None
    return row[column], row.get("start_date_local_date"), activity_type


def types_on_day_label(df: pd.DataFrame, day) -> str:
    """'Walking, Running' list of the activity types done on a day, or '' if unknown."""
    if "start_date_local_date" not in df.columns:
        return ""
    day_rows = df[df["start_date_local_date"] == day]
    types = day_rows["activity_type"].dropna().unique()
    return ", ".join(sorted(str(value) for value in types))


def longest_streak_length(dates) -> int:
    """Longest run of calendar days with no gaps, from an iterable of dates/timestamps."""
    days = sorted(dates)
    if not days:
        return 0
    best = current = 1
    for previous, current_day in zip(days, days[1:]):
        current = current + 1 if (current_day - previous).days == 1 else 1
        best = max(best, current)
    return best


def best_streak_days(dates, length: int) -> list:
    """The calendar days forming a best streak of `length` days (ending latest), or []."""
    days = sorted(dates)
    if not days:
        return []
    if length <= 1:
        return [days[-1]]
    current: list = []
    best: list = []
    for day in days:
        if current and (day - current[-1]).days == 1:
            current.append(day)
        else:
            current = [day]
        if len(current) >= length and current[-1] >= (best[-1] if best else day):
            best = current.copy()
    return best


def build_workout_records(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """All-time best single-workout records: (label, value, date) triples."""
    if df.empty:
        return []
    specs = [
        ("Longest Workout", "duration_seconds", False, lambda v: format_duration(v)),
        ("Farthest Workout Distance", "total_distance_mi", False, lambda v: f"{v:.2f} mi"),
        ("Fastest Pace", "pace_min_per_mi", True, lambda v: f"{v:.2f} min/mi"),
        ("Most Workout Calories", "total_energy_kcal", False, lambda v: f"{v:,.0f} kcal"),
        ("Most Active Calories", "active_energy_kcal", False, lambda v: f"{v:,.0f} kcal"),
        ("Highest Avg Heart Rate", "average_heart_rate_bpm", False, lambda v: f"{v:,.0f} bpm"),
    ]
    records = []
    for label, column, ascending, formatter in specs:
        value, record_date, activity_type = best_single_value(df, column, ascending)
        if value is None or pd.isna(value):
            continue
        date_label = format_date_short(record_date)
        if isinstance(activity_type, str) and activity_type:
            date_label += f" ({activity_type})"
        records.append((label, formatter(float(value)), date_label))
    return records


def build_daily_workout_records(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Best single-day aggregates across workouts: (label, value, date) triples."""
    if df.empty or "start_date_local_date" not in df.columns:
        return []
    work = df.dropna(subset=["start_date_local_date"])
    if work.empty:
        return []
    by_day = work.groupby("start_date_local_date")
    specs = [
        ("Most Workouts in a Day", by_day.size(), lambda v: f"{v}"),
        ("Most Workout Hours in a Day", by_day["duration_hours"].sum(min_count=1), lambda v: f"{v:.2f} h"),
        ("Most Workout Distance in a Day", by_day["total_distance_mi"].sum(min_count=1), lambda v: f"{v:.2f} mi"),
        ("Most Workout Calories in a Day", by_day["total_energy_kcal"].sum(min_count=1), lambda v: f"{v:,.0f} kcal"),
    ]
    records = []
    for label, series, formatter in specs:
        series = series.dropna()
        if series.empty:
            continue
        idx = series.idxmax()
        date_label = format_date_short(idx)
        type_label = types_on_day_label(work, idx)
        if type_label:
            date_label += f" ({type_label})"
        records.append((label, formatter(series[idx]), date_label))
    return records


def build_daily_health_records(metrics_frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Best single-day totals from health metrics: (label, value, date) triples."""
    if metrics_frame is None or metrics_frame.empty:
        return []
    specs = [
        ("steps", "Most Steps in a Day", lambda v: f"{v:,.0f} steps"),
        ("walk_run_distance_m", "Most Walk + Run in a Day", lambda v: f"{v * 0.000621371:.2f} mi"),
        ("exercise_minutes", "Most Exercise in a Day", lambda v: f"{v:,.0f} min"),
        ("move_energy_kcal", "Most Move Calories in a Day", lambda v: f"{v:,.0f} kcal"),
        ("total_energy_kcal", "Most Calories Burned in a Day", lambda v: f"{v:,.0f} kcal"),
        ("sleep_hours", "Most Sleep in a Day", lambda v: f"{v:.1f} h"),
        ("time_in_daylight_minutes", "Most Time in Daylight", lambda v: f"{v:,.0f} min"),
        ("stand_hours", "Most Stand Hours in a Day", lambda v: f"{v:.0f} h"),
        ("flights_climbed", "Most Flights Climbed in a Day", lambda v: f"{v:,.0f} flights"),
    ]
    records = []
    for column, label, formatter in specs:
        if column not in metrics_frame.columns:
            continue
        series = metrics_frame[column].dropna()
        if series.empty:
            continue
        idx = series.idxmax()
        records.append((label, formatter(float(series[idx])), format_date_short(idx)))
    return records


def build_body_measurement_records(metrics_frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Best (lowest) tracked body measurements: (label, value, date) triples."""
    if metrics_frame is None or metrics_frame.empty:
        return []
    specs = [
        ("weight_kg", "Lowest Weight", lambda v: f"{v * 2.2046226218:.1f} lb", "min"),
        ("body_fat_pct", "Lowest Body Fat", lambda v: f"{v:.1f}%", "min"),
        ("bmi", "Best (Lowest) BMI", lambda v: f"{v:.1f} BMI", "min"),
        ("resting_hr_bpm", "Lowest Resting Heart Rate", lambda v: f"{v:.0f} bpm", "min"),
        ("walking_hr_bpm", "Lowest Walking Heart Rate", lambda v: f"{v:.0f} bpm", "min"),
        ("vo2_max", "Best (Highest) VO2 Max", lambda v: f"{v:.1f} L/min", "max"),
    ]
    records = []
    for column, label, formatter, direction in specs:
        if column not in metrics_frame.columns:
            continue
        series = metrics_frame[column].dropna()
        if series.empty:
            continue
        idx = series.idxmin() if direction == "min" else series.idxmax()
        records.append((label, formatter(float(series[idx])), format_date_short(idx)))
    return records


def build_running_records(metrics_frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Best daily-average running stats: (label, value, date) triples."""
    if metrics_frame is None or metrics_frame.empty:
        return []
    specs = [
        ("running_power_w", "Highest Running Power", lambda v: f"{v:,.0f} W"),
        ("running_speed_mps", "Fastest Running Speed", lambda v: f"{v * 2.2369362921:.1f} mph"),
        ("running_stride_m", "Longest Running Stride", lambda v: f"{v * 3.280839895:.2f} ft"),
        ("running_cadence_spm", "Highest Running Cadence", lambda v: f"{v:,.0f} steps/min"),
    ]
    records = []
    for column, label, formatter in specs:
        if column not in metrics_frame.columns:
            continue
        series = metrics_frame[column].dropna()
        if series.empty:
            continue
        idx = series.idxmax()
        records.append((label, formatter(float(series[idx])), format_date_short(idx)))
    return records


def build_streak_records(df: pd.DataFrame, metrics_frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Consecutive-day streak records: (label, value, date) triples."""
    records = []
    if not df.empty and "start_date_local_date" in df.columns:
        workout_days = sorted(set(df["start_date_local_date"].dropna()))
        if workout_days:
            length = longest_streak_length(workout_days)
            if length:
                streak = best_streak_days(workout_days, length)
                date_label = format_date_short(streak[-1])
                # List every type done anywhere in the streak, not just the final day.
                streak_types = (
                    ", ".join(
                        sorted(df[df["start_date_local_date"].isin(streak)]["activity_type"].dropna().unique())
                    )
                    if streak
                    else ""
                )
                if streak_types:
                    date_label += f" ({streak_types})"
                records.append(("Longest Workout Streak", f"{length} days", date_label))

    # Health-metric streaks: longest run of consecutive days meeting a daily threshold.
    metric_streak_specs = (
        ("exercise_minutes", 30.0, "Longest Exercise Streak"),
        ("stand_hours", 12.0, "Longest Stand Streak"),
    )
    if metrics_frame is not None and not metrics_frame.empty:
        for column, threshold, label in metric_streak_specs:
            if column not in metrics_frame.columns:
                continue
            # NaN days (no data) compare False, so this is the non-null days at/above the threshold.
            qualified_index = pd.DatetimeIndex(metrics_frame.index)[metrics_frame[column] >= threshold]
            days = sorted(qualified_index.date)
            if not days:
                continue
            length = longest_streak_length(days)
            streak = best_streak_days(days, length)
            records.append((label, f"{length} days", format_date_short(streak[-1])))

    return records


def render_record_grid(records: list[tuple[str, str, str]]) -> None:
    """Render (label, value, date) records as metric tiles, three per row."""
    for row_start in range(0, len(records), 3):
        chunk = records[row_start : row_start + 3]
        cols = st.columns(3)
        for col, (label, value, record_date) in zip(cols, chunk):
            col.metric(label, value)
            if record_date:
                col.caption(record_date)
            else:
                col.empty()


def build_metrics_frame(samples: dict[str, list[MetricSample]]) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    all_days: set = set()
    for column in (
        "weight_kg", "body_fat_pct", "height_m", "resting_hr_bpm", "vo2_max", "sleep_hours",
        "time_in_daylight_minutes",
        "steps", "walk_run_distance_m", "move_energy_kcal", "total_energy_kcal",
        "resting_energy_kcal", "exercise_minutes", "stand_hours", "flights_climbed",
        "walking_hr_bpm",
        "running_power_w", "running_speed_mps", "running_stride_m", "running_cadence_spm",
    ):
        samples_list = samples.get(column) or []
        if not samples_list:
            continue
        by_day: dict = {}
        for sample in samples_list:  # sorted ascending, so later entries win per day
            day = sample.timestamp.date()
            by_day[day] = sample.value
            all_days.add(day)
        frames[column] = pd.Series(by_day)
    if not frames:
        return pd.DataFrame()

    index = pd.to_datetime(sorted(all_days))
    frame = pd.DataFrame(index=index)
    for column in ("weight_kg", "body_fat_pct", "height_m", "resting_hr_bpm", "vo2_max"):
        if column in frames:
            frame[column] = frames[column].reindex(index, method="ffill")
    if "sleep_hours" in frames:
        frame["sleep_hours"] = frames["sleep_hours"].reindex(index)
    if "time_in_daylight_minutes" in frames:
        frame["time_in_daylight_minutes"] = frames["time_in_daylight_minutes"].reindex(index)
    if "steps" in frames:
        frame["steps"] = frames["steps"].reindex(index)
    if "walk_run_distance_m" in frames:
        frame["walk_run_distance_m"] = frames["walk_run_distance_m"].reindex(index)
    if "move_energy_kcal" in frames:
        frame["move_energy_kcal"] = frames["move_energy_kcal"].reindex(index)
    if "total_energy_kcal" in frames:
        frame["total_energy_kcal"] = frames["total_energy_kcal"].reindex(index)
    if "resting_energy_kcal" in frames:
        frame["resting_energy_kcal"] = frames["resting_energy_kcal"].reindex(index)
    if "exercise_minutes" in frames:
        frame["exercise_minutes"] = frames["exercise_minutes"].reindex(index)
    if "stand_hours" in frames:
        frame["stand_hours"] = frames["stand_hours"].reindex(index)
    if "flights_climbed" in frames:
        frame["flights_climbed"] = frames["flights_climbed"].reindex(index)
    if "walking_hr_bpm" in frames:
        frame["walking_hr_bpm"] = frames["walking_hr_bpm"].reindex(index)
    for column in ("running_power_w", "running_speed_mps", "running_stride_m", "running_cadence_spm"):
        if column in frames:
            frame[column] = frames[column].reindex(index)

    if "weight_kg" in frame and "height_m" in frame:
        height = frame["height_m"]
        frame["bmi"] = (frame["weight_kg"] / (height * height)).where(height > 0)
    if "weight_kg" in frame and "body_fat_pct" in frame:
        frame["lbm_kg"] = frame["weight_kg"] * (1.0 - frame["body_fat_pct"] / 100.0)
    return frame


# Day-based metrics that the "Current Measurements" tiles can show as either a
# 7-day average or an average over the whole selected time period.
DAILY_AVG_COLUMNS = (
    "steps", "walk_run_distance_m", "sleep_hours", "resting_hr_bpm", "vo2_max",
    "time_in_daylight_minutes",
    "move_energy_kcal", "total_energy_kcal", "resting_energy_kcal",
    "exercise_minutes", "stand_hours", "flights_climbed", "walking_hr_bpm",
    "running_power_w", "running_speed_mps", "running_stride_m", "running_cadence_spm",
)


def build_metric_summaries(samples: dict[str, list[MetricSample]]) -> dict[str, dict[str, float | None]]:
    """Precompute both average views of every day-based metric at parse time.

    Each entry maps a column to {"7day": ..., "range": ...}: the average of the
    last seven complete days (today is excluded — its total is still in
    progress and would read low) and the average over the whole parsed dataset,
    which is exactly the selected time period since parsing is range-filtered.
    Computing both up front lets the Health Metrics tab flip between them
    instantly without re-processing the export.
    """
    frame = build_metrics_frame(samples)
    today_ts = pd.Timestamp(datetime.now().date())
    week_start = today_ts - timedelta(days=7)
    summaries: dict[str, dict[str, float | None]] = {}
    for column in DAILY_AVG_COLUMNS:
        series = frame[column].dropna() if column in frame.columns else None
        weekly = (
            series[(series.index >= week_start) & (series.index < today_ts)] if series is not None else None
        )
        summaries[column] = {
            "7day": float(weekly.mean()) if weekly is not None and not weekly.empty else None,
            "range": float(series.mean()) if series is not None and not series.empty else None,
        }
    return summaries


def format_height_m(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    total_inches = value * 39.37007874
    feet, inches = divmod(total_inches, 12)
    if round(inches) == 12:
        feet += 1
        inches = 0
    return f"{int(feet)}' {int(round(inches))}\""


def format_kg_as_lb(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 2.2046226218:.1f} lb"


def format_inches_ft_in(total_inches: float) -> str:
    feet, inches = divmod(int(round(total_inches)), 12)
    return f"{feet}' {inches}\""


def height_tick_scale(series_inches: pd.Series) -> tuple[list[float], list[str]]:
    """Whole-inch axis ticks (at most ~9) rendered as ft/in for the height chart."""
    values = series_inches.dropna()
    if values.empty:
        return [], []
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-6:
        low, high = low - 1.0, high + 1.0
    start, end = math.floor(low), math.ceil(high)
    step = 1
    while (end - start) // step > 8:
        step += 1
    tick_values: list[float] = []
    tick_text: list[str] = []
    value = (start // step) * step
    while value <= end:
        tick_values.append(float(value))
        tick_text.append(format_inches_ft_in(value))
        value += step
    return tick_values, tick_text


def height_hover_text(series_inches: pd.Series) -> list[str]:
    return [format_inches_ft_in(float(value)) for value in series_inches]


TREND_WINDOW_DAYS = 15  # centered rolling window for the trend overlay


def smoothed_trendline(series: pd.Series) -> list[float]:
    """Centered rolling mean used to overlay a smoothed trend on noisy series."""
    return (
        series.rolling(window=TREND_WINDOW_DAYS, center=True, min_periods=1).mean().tolist()
    )


def faded_color(hex_color: str, alpha: float) -> str:
    """Returns an rgba() string so plotly can render a hex color at partial opacity."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# One entry per toggleable chart layer. `column` keys build_metrics_frame output;
# new measurements added later only need an entry here plus a parser hook.
# `convert` rescales values for the chart axis (internal storage stays SI);
# `y_title` labels the axis in the units the chart actually shows.
METRIC_LAYERS = [
    {
        "column": "weight_kg",
        "title": "Weight (lb)",
        "color": "#2E86DE",
        "format": format_kg_as_lb,
        "convert": lambda value: value * 2.2046226218,
        "y_title": "Weight (lb)",
    },
    {
        "column": "body_fat_pct",
        "title": "Body Fat (%)",
        "color": "#E67E22",
        "format": lambda value: f"{value:.1f}%" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Body Fat (%)",
    },
    {
        "column": "bmi",
        "title": "BMI",
        "color": "#16A085",
        "format": lambda value: f"{value:.1f}" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "BMI",
    },
    {
        "column": "lbm_kg",
        "title": "Lean Body Mass (lb)",
        "color": "#D81B60",
        "format": format_kg_as_lb,
        "convert": lambda value: value * 2.2046226218,
        "y_title": "Lean Body Mass (lb)",
    },
    {
        "column": "height_m",
        "title": "Height",
        "color": "#8E44AD",
        "format": format_height_m,
        "convert": lambda value: value * 39.37007874,
        "y_title": "Height (ft/in)",
        "tick_func": height_tick_scale,
        "hover_func": height_hover_text,
    },
    {
        "column": "resting_hr_bpm",
        "title": "Resting Heart Rate (bpm)",
        "color": "#EF4444",
        "format": lambda value: f"{value:.0f} bpm" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Resting Heart Rate (bpm)",
    },
    {
        "column": "sleep_hours",
        "title": "Sleep Duration (h)",
        "color": "#2563EB",
        "format": lambda value: f"{value:.1f} h" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Sleep Duration (h)",
    },
    {
        "column": "time_in_daylight_minutes",
        "title": "Time in Daylight (min)",
        "color": "#F4B942",
        "format": lambda value: f"{value:.0f} min" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Time in Daylight (min)",
    },
    {
        "column": "steps",
        "title": "Steps (day)",
        "color": "#10B981",
        "format": lambda value: f"{value:,.0f}" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Steps (day)",
    },
    {
        "column": "walk_run_distance_m",
        "title": "Walking + Running Distance (mi)",
        "color": "#F59E0B",
        "format": lambda value: f"{value * 0.000621371:.1f} mi" if value is not None and pd.notna(value) else "N/A",
        "convert": lambda value: value * 0.000621371,
        "y_title": "Distance (mi)",
    },
    {
        "column": "move_energy_kcal",
        "title": "Move Calories (kcal)",
        "color": "#FF3B30",
        "format": lambda value: f"{value:,.0f} kcal" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Move Calories (kcal)",
    },
    {
        "column": "total_energy_kcal",
        "title": "Total Calories Burned (kcal)",
        "color": "#FF9500",
        "format": lambda value: f"{value:,.0f} kcal" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Total Calories Burned (kcal)",
    },
    {
        "column": "resting_energy_kcal",
        "title": "Resting Calories (kcal)",
        "color": "#999999",
        "format": lambda value: f"{value:,.0f} kcal" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Resting Calories (kcal)",
    },
    {
        "column": "exercise_minutes",
        "title": "Exercise (min)",
        "color": "#64D22D",
        "format": lambda value: f"{value:,.0f} min" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Exercise (min)",
    },
    {
        "column": "stand_hours",
        "title": "Stand (h)",
        "color": "#00B8A9",
        "format": lambda value: f"{value:.1f} h" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Stand (h)",
    },
    {
        "column": "flights_climbed",
        "title": "Flights Climbed (day)",
        "color": "#607D8B",
        "format": lambda value: f"{value:,.0f}" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Flights Climbed (day)",
    },
    {
        "column": "vo2_max",
        "title": "VO2 Max (L/min)",
        "color": "#009688",
        "format": lambda value: f"{value:.1f} L/min" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "VO2 Max (L/min)",
    },
    {
        "column": "walking_hr_bpm",
        "title": "Walking Heart Rate (bpm)",
        "color": "#FF2D55",
        "format": lambda value: f"{value:.0f} bpm" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Walking Heart Rate (bpm)",
    },
    {
        "column": "running_power_w",
        "title": "Running Power (W)",
        "color": "#00C7BE",
        "format": lambda value: f"{value:,.0f} W" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Running Power (W)",
    },
    {
        "column": "running_speed_mps",
        "title": "Running Speed (mph)",
        "color": "#AF52DE",
        "format": lambda value: f"{value * 2.2369362921:.1f} mph" if value is not None and pd.notna(value) else "N/A",
        "convert": lambda value: value * 2.2369362921,
        "y_title": "Running Speed (mph)",
    },
    {
        "column": "running_stride_m",
        "title": "Running Stride Length (ft)",
        "color": "#5856D6",
        "format": lambda value: f"{value * 3.280839895:.2f} ft" if value is not None and pd.notna(value) else "N/A",
        "convert": lambda value: value * 3.280839895,
        "y_title": "Running Stride Length (ft)",
    },
    {
        "column": "running_cadence_spm",
        "title": "Running Cadence (spm)",
        "color": "#00E5FF",
        "format": lambda value: f"{value:,.0f}" if value is not None and pd.notna(value) else "N/A",
        "convert": None,
        "y_title": "Running Cadence (steps/min)",
    },
]


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


def render_running_profile(workout: WorkoutRecord, running_metrics: dict) -> None:
    """Per-workout running power / speed / cadence over time, when the export has them.

    The raw samples were collected during the single export pass and stored in the
    shared metrics dict; they are filtered here to the workout's start/end window so
    no extra parse is required. Non-running workouts simply have no samples in their
    window and fall back to a short note.
    """
    # Metric sample timestamps are stored naive-local (see _naive_local in
    # health_parser), so bring the aware workout window into that same frame.
    window_start = _naive_local(workout.start_date)
    window_end = _naive_local(workout.end_date)
    if window_start is None and window_end is None:
        st.caption("No running power, speed, or cadence samples are associated with this workout.")
        return
    if window_start is None:
        window_start = window_end
    elif window_end is None:
        window_end = window_start
    if window_start > window_end:
        window_start, window_end = window_end, window_start

    def in_window(samples):
        pairs = []
        for sample in samples or []:
            timestamp = _naive_local(sample.timestamp)
            if timestamp is not None and window_start <= timestamp <= window_end:
                pairs.append((timestamp, sample.value))
        pairs.sort(key=lambda item: item[0])
        return pairs

    # (title, filtered pairs, mph factor for m/s samples or None, unit tag)
    series = [
        ("Power (W)", in_window(running_metrics.get("running_power_samples")), None, "watts"),
        (
            "Running Speed (mph)",
            in_window(running_metrics.get("running_speed_samples")),
            2.2369362921,
            "mph",
        ),
        (
            "Running Cadence (steps/min)",
            in_window(running_metrics.get("running_cadence_samples")),
            None,
            "spm",
        ),
    ]
    present = [entry for entry in series if entry[1]]
    if not present:
        st.caption("No running power, speed, or cadence samples are associated with this workout.")
        return

    st.markdown("### Running Power / Speed / Cadence Over Time")

    # At-a-glance averages for whatever this workout actually recorded.
    tiles = []
    for _title, pairs, _factor, unit in present:
        average = sum(value for _, value in pairs) / len(pairs)
        if unit == "mph":
            tiles.append(("Avg Speed (mph)", f"{average * 2.2369362921:.1f}"))
        elif unit == "watts":
            tiles.append(("Avg Power (W)", f"{average:,.0f}"))
        else:
            tiles.append(("Avg Cadence (steps/min)", f"{average:,.0f}"))
    columns = st.columns(len(tiles))
    for index, (label, text) in enumerate(tiles):
        columns[index].metric(label, text)

    fig = make_subplots(
        rows=len(present),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        row_titles=[entry[0] for entry in present],
    )
    for row_index, (_title, pairs, factor, _unit) in enumerate(present, start=1):
        times = [timestamp for timestamp, _ in pairs]
        values = [value * factor for _, value in pairs] if factor else [value for _, value in pairs]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines+markers",
                name=_title,
                line=dict(width=2),
                marker=dict(size=5),
            ),
            row=row_index,
            col=1,
        )
    fig.update_layout(
        height=180 * len(present) + 60,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


st.title("Apple Health Workout Explorer")
st.caption(
    "Load an Apple Health export — point the folder at the one containing `export.xml` "
    "(or a folder that contains it) plus an optional `workout-routes/`."
)


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
        "Path to export folder",
        value="",
        key="export_folder_input",
        help="The apple_health_export folder itself, or the folder that contains it.",
    )

if not export_folder:
    st.info("Enter the path to the Apple Health export folder in the sidebar.")
    st.stop()

export_root = Path(export_folder).expanduser()
if not export_root.exists() or not export_root.is_dir():
    st.error(f"Apple Health export folder not found: {export_folder}")
    st.stop()


def resolve_export_folder(folder: Path) -> Path:
    """Point at the folder that actually contains export.xml.

    Apple Health's export zip unzips to a wrapper folder that holds
    apple_health_export/, so accept either the export folder itself or any
    ancestor of it. The descent is shallow (three levels) because the export
    layout never nests deeper and a full scan of a huge export is wasteful.
    """
    if (folder / "export.xml").is_file():
        return folder
    candidates: list[Path] = []
    stack: list[Path] = [folder]
    for _ in range(3):
        next_stack: list[Path] = []
        for path in stack:
            if (path / "export.xml").is_file():
                candidates.append(path)
                continue
            try:
                children = [child for child in path.iterdir() if child.is_dir()]
            except OSError:
                continue
            next_stack.extend(children)
        stack = next_stack
    if not candidates:
        return folder
    if len(candidates) > 1:
        # Several exports were unzipped side by side; use the newest export.xml.
        return max(candidates, key=lambda path: (path / "export.xml").stat().st_mtime)
    return candidates[0]


export_root = resolve_export_folder(export_root)
if export_root != Path(export_folder).expanduser():
    with st.sidebar:
        st.caption(f"Found the export nested at: {export_root}")
export_path = export_root / "export.xml"
route_dir = export_root / "workout-routes"
folder_key = str(export_root)
path_changed = st.session_state.get("loaded_export_folder") != folder_key

with st.sidebar:
    st.header("Time frame")
    st.caption(
        "Only workouts, routes, and measurements inside this range are parsed, so a "
        "narrower range loads faster. Press Process data after changing the range."
    )
    today = date.today()
    date_preset = st.selectbox(
        "Date range preset",
        ["Year to date", "All time", "Past year", "Custom"],
        key="date_range_preset",
    )
    range_start: date | None
    range_end: date | None
    if date_preset == "Year to date":
        range_start, range_end = date(today.year, 1, 1), today
        st.caption(f"Parsing from {range_start:%Y-%m-%d} through {range_end:%Y-%m-%d}.")
    elif date_preset == "Past year":
        range_start, range_end = today - timedelta(days=365), today
        st.caption(f"Parsing from {range_start:%Y-%m-%d} through {range_end:%Y-%m-%d}.")
    elif date_preset == "All time":
        range_start, range_end = None, None
    else:
        raw_range = st.date_input(
            "Date range",
            value=(today - timedelta(days=365), today),
            key="custom_date_range",
        )
        if isinstance(raw_range, (tuple, list)) and len(raw_range) == 2:
            range_start = coerce_to_date(raw_range[0])
            range_end = coerce_to_date(raw_range[1])
        else:
            single = coerce_to_date(raw_range)
            range_start = range_end = single

    process_requested = st.button(
        "Process data",
        type="primary",
        use_container_width=True,
        help="Parse workouts, routes, and measurements from the export for the selected range. Press again after changing the folder or range.",
    )

needs_signature_check = (
    process_requested
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
    st.error(
        f"export.xml not found in {export_root} or any subfolder. "
        "Point the folder input at the export folder (the one containing apple_health_export/)."
    )
    st.stop()

data_size = folder_size_bytes(folder_key, export_signature, route_signature)
with st.sidebar:
    st.metric("Export Data Size", format_bytes(data_size))

if process_requested:
    progress_labels = ["Parsing export"]
    if route_signature is not None:
        progress_labels.append("Parsing routes")
    progress = ParseProgressUI(progress_labels).render()
    try:
        set_progress_callback(progress.report)
        progress.begin_stage(0)
        parsed_workouts, parsed_metrics, parsed_counts = load_export(
            export_path, export_signature, range_start, range_end
        )
        st.session_state["workouts"] = parsed_workouts
        st.session_state["health_metrics"] = (parsed_metrics, parsed_counts)
        st.session_state["health_metrics_summaries"] = build_metric_summaries(parsed_metrics)
        progress.finish_stage()

        if route_signature is not None:
            progress.begin_stage(1)
            st.session_state["routes"] = load_routes(route_dir, route_signature, range_start, range_end)
            progress.finish_stage()
        else:
            st.session_state["routes"] = []
    finally:
        set_progress_callback(None)
        progress.dispose()

    with st.spinner("Matching routes to workouts..."):
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
        st.session_state["loaded_date_range"] = (range_start, range_end)

if "workouts" not in st.session_state:
    st.info("Select a date range in the sidebar and press **Process data** to load your export.")
    st.stop()

if (
    st.session_state.get("loaded_date_range") != (range_start, range_end)
    or st.session_state.get("loaded_export_folder") != folder_key
    or st.session_state.get("loaded_export_signature") != export_signature
    or st.session_state.get("loaded_route_signature") != route_signature
    or st.session_state.get("loaded_parser_version") != DATA_PARSER_VERSION
):
    st.sidebar.caption("Export or date range changed — press **Process data** to re-parse.")

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
    df["pace_min_per_mi"] = pd.Series(dtype="float64")

if df.empty:
    st.warning(
        "No workouts fall inside the selected time frame — widen the date range in the sidebar and press Process data."
    )
elif df["total_distance_mi"].isna().all() and df["total_energy_kcal"].isna().all():
    st.warning(
        "This export contains no readable total distance or energy values on its Workout records. "
        "The app can still show duration and route data."
    )

if not df.empty:
    min_date = df["start_date_local_date"].dropna().min()
    max_date = df["start_date_local_date"].dropna().max()
else:
    min_date = max_date = None
selected_start: date | None = range_start if range_start is not None else min_date
selected_end: date | None = range_end if range_end is not None else max_date

date_filtered_df = apply_filters(df, selected_start, selected_end)
available_types = sorted(date_filtered_df["activity_type"].fillna("Unknown").unique().tolist())

with st.sidebar:
    st.header("Filters")
    selected_activity_types = st.multiselect(
        "Activity types",
        options=available_types,
        default=available_types,
        help="Only activity types present in the selected date range are listed.",
    )

filtered_df = apply_filters(df, selected_start, selected_end, selected_activity_types)

# on_change="rerun" turns the tabs into a real session-state widget, so the
# selected tab survives any rerun (e.g. flipping the health-metric toggle). A
# plain stateless st.tabs remounts and snaps back to the first tab instead.
# https://github.com/streamlit/streamlit/issues/8239
TAB_NAMES = ["Workout Accumulator", "Individual Workout Route Inspector", "Health Metrics", "Records"]
tab1, tab2, tab3, tab4 = st.tabs(
    TAB_NAMES,
    default=st.session_state.get("main_tabs", TAB_NAMES[0]),
    key="main_tabs",
    on_change="rerun",
)

with tab1:
    display_metrics(filtered_df)

    c1, c2 = st.columns([1, 1])
    with c1:
        granularity = st.selectbox("Group accumulation by", ["Day", "Week", "Month"], index=1)
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

        render_running_profile(
            selected_workout, st.session_state.get("health_metrics", ({}, {}))[0]
        )

with tab3:
    metrics_samples, metrics_counts = st.session_state.get("health_metrics", ({}, {}))
    metrics_frame = build_metrics_frame(metrics_samples)

    if metrics_frame.empty:
        found_counts = {name: count for name, count in metrics_counts.items() if count > 0}
        if not found_counts:
            st.info(
                "No body measurement records (bodymass, bodyfatpercentage, height, "
                "restingheart-rate, vo2max, sleepanalysis, stepcount, walking/running "
                "distance, active/resting energy, exercise, stand hours, or flights "
                "climbed) were found in export.xml."
            )
        else:
            summary = ", ".join(f"{name}: {count}" for name, count in sorted(found_counts.items()))
            st.info(f"Some measurement records were found but none could be read ({summary}).")
        st.stop()

    matched = ", ".join(f"{name}: {count}" for name, count in sorted(metrics_counts.items()) if count > 0)
    if matched:
        st.caption(f"Measurement records matched in export.xml — {matched}")

    st.subheader("Current Measurements")
    view_day = st.toggle(
        "View a specific day",
        value=False,
        key="daily_metrics_view_day",
        help="Show the measurements for one chosen day (within the filtered dataset) "
        "instead of the averages.",
    )
    day_row = None
    tile_labels = {
        "steps": "Steps",
        "walk_run_distance_m": "Walk + Run Distance (mi)",
        "sleep_hours": "Sleep Duration (h)",
        "time_in_daylight_minutes": "Time in Daylight (min)",
        "resting_hr_bpm": "Resting Heart Rate (bpm)",
        "move_energy_kcal": "Move Calories (kcal)",
        "total_energy_kcal": "Total Calories Burned (kcal)",
        "exercise_minutes": "Exercise (min)",
        "stand_hours": "Stand (h)",
        "flights_climbed": "Flights Climbed",
        "walking_hr_bpm": "Walking Heart Rate (bpm)",
        "resting_energy_kcal": "Resting Calories (kcal)",
        "running_power_w": "Running Power (W)",
        "running_speed_mps": "Running Speed (mph)",
        "running_stride_m": "Running Stride Length (ft)",
        "running_cadence_spm": "Running Cadence (steps/min)",
    }
    if view_day:
        # Calendar picker, same widget as the custom range in the sidebar, bounded
        # to the days present in the filtered dataset.
        chosen_day = st.date_input(
            "Day to view",
            value=metrics_frame.index[-1].date(),
            min_value=metrics_frame.index[0].date(),
            max_value=metrics_frame.index[-1].date(),
            key="daily_metrics_chosen_day",
        )
        # Anchor to the most recent day at or before the chosen one so gaps in
        # the data still resolve to the last known value (matches the ffill used
        # to build the frame); day-based columns read N/A for that day itself.
        anchor = metrics_frame.index.asof(pd.Timestamp(chosen_day))
        day_row = metrics_frame.loc[anchor]
        st.caption(f"Showing measurements from {chosen_day:%b %d, %Y}.")
    else:
        # Day-based metrics show one of two views both precomputed at **Process
        # data** time (see build_metric_summaries): the default is the average of
        # the last seven complete days (today's total is still in progress and
        # would read low); the toggle switches to the average over the whole
        # selected period.
        use_filtered_avg = st.toggle(
            "Average over whole selected time period",
            value=False,
            key="daily_metrics_avg_filtered",
            help="Off: day-based metrics show the 7-day average (default). "
            "On: day-based metrics show the average across the entire selected time period.",
        )
        if use_filtered_avg:
            tile_labels = {column: f"{label} (range avg)" for column, label in tile_labels.items()}
            st.caption(
                f"Range averages cover {metrics_frame.index.min():%b %d, %Y} – "
                f"{metrics_frame.index.max():%b %d, %Y} ({len(metrics_frame)} days of data)."
            )
        else:
            tile_labels = {column: f"{label} (7-day avg)" for column, label in tile_labels.items()}
    summaries = st.session_state.get("health_metrics_summaries", {})
    tiles = []
    for layer in METRIC_LAYERS:
        if view_day and day_row is not None:
            value = day_row.get(layer["column"])
            current_value = float(value) if pd.notna(value) else None
        elif layer["column"] in DAILY_AVG_COLUMNS:
            current_value = summaries.get(layer["column"], {}).get("range" if use_filtered_avg else "7day")
        else:
            series = metrics_frame[layer["column"]].dropna() if layer["column"] in metrics_frame.columns else None
            current_value = series.iloc[-1] if series is not None and not series.empty else None
        tile_title = tile_labels.get(layer["column"], layer["title"])
        tiles.append((tile_title, layer["format"](current_value)))
    # Twelve tiles in one row overflow; lay them out six per row.
    TILES_PER_ROW = 6
    for row_start in range(0, len(tiles), TILES_PER_ROW):
        tile_columns = st.columns(TILES_PER_ROW)
        for column_slot, (tile_title, tile_value) in enumerate(tiles[row_start : row_start + TILES_PER_ROW]):
            tile_columns[column_slot].metric(tile_title, tile_value)

    range_frame = metrics_frame
    if selected_start is not None:
        range_frame = range_frame[range_frame.index >= pd.Timestamp(selected_start)]
    if selected_end is not None:
        range_frame = range_frame[range_frame.index <= pd.Timestamp(selected_end)]
    total_walk_run_mi = (
        float(range_frame["walk_run_distance_m"].sum()) * 0.000621371
        if "walk_run_distance_m" in range_frame.columns
        else 0.0
    )
    total_steps = int(range_frame["steps"].sum()) if "steps" in range_frame.columns else 0
    total_sleep_h = float(range_frame["sleep_hours"].sum()) if "sleep_hours" in range_frame.columns else 0.0
    total_daylight_min = (
        float(range_frame["time_in_daylight_minutes"].sum())
        if "time_in_daylight_minutes" in range_frame.columns
        else 0.0
    )
    total_move_kcal = float(range_frame["move_energy_kcal"].sum()) if "move_energy_kcal" in range_frame.columns else 0.0
    total_total_kcal = float(range_frame["total_energy_kcal"].sum()) if "total_energy_kcal" in range_frame.columns else 0.0
    total_exercise_min = float(range_frame["exercise_minutes"].sum()) if "exercise_minutes" in range_frame.columns else 0.0
    total_stand_h = float(range_frame["stand_hours"].sum()) if "stand_hours" in range_frame.columns else 0.0
    total_flights = int(range_frame["flights_climbed"].sum()) if "flights_climbed" in range_frame.columns else 0
    total_start = range_frame.index.min()
    total_end = range_frame.index.max()
    range_label = (
        f"{total_start:%b %d, %Y} - {total_end:%b %d, %Y}"
        if not range_frame.empty
        else "selected time range"
    )
    st.subheader(f"Totals ({range_label})")
    total_columns = st.columns(5)
    total_columns[0].metric("Total Walk + Run Distance", f"{total_walk_run_mi:,.1f} mi")
    total_columns[1].metric("Total Steps", f"{total_steps:,}")
    total_columns[2].metric("Total Sleep", f"{total_sleep_h:,.1f} h")
    total_columns[3].metric("Total Move Calories", f"{total_move_kcal:,.0f} kcal")
    total_columns[4].metric("Total Time in Daylight", f"{total_daylight_min:,.0f} min")
    total_columns_2 = st.columns(4)
    total_columns_2[0].metric("Total Calories Burned", f"{total_total_kcal:,.0f} kcal")
    total_columns_2[1].metric("Total Exercise Minutes", f"{total_exercise_min:,.0f} min")
    total_columns_2[2].metric("Total Stand Hours", f"{total_stand_h:,.1f} h")
    total_columns_2[3].metric("Total Flights Climbed", f"{total_flights:,}")

    st.subheader("Health Metrics Over Time")
    available_layers = [layer for layer in METRIC_LAYERS if layer["column"] in metrics_frame.columns]
    selected_layers = st.multiselect(
        "Metric layers",
        options=[layer["title"] for layer in available_layers],
        default=[layer["title"] for layer in available_layers],
        key="metrics_layers_multiselect",
        help="Check or uncheck individual metrics to show or hide them on the chart.",
    )
    show_trendlines = st.checkbox(
        "Trend lines (15-day smoothed)",
        value=True,
        key="metrics_trendlines",
        help="Draws a smoothed trend (centered 15-day average) as the main line for each displayed metric, with the raw measurements faded in the background.",
    )
    selected_layer_defs = [layer for layer in available_layers if layer["title"] in selected_layers]
    if not selected_layer_defs:
        st.info("Select at least one metric layer to display.")
    else:
        filtered_metrics = metrics_frame
        if selected_start is not None:
            filtered_metrics = filtered_metrics[filtered_metrics.index >= pd.Timestamp(selected_start)]
        if selected_end is not None:
            filtered_metrics = filtered_metrics[filtered_metrics.index <= pd.Timestamp(selected_end)]
        if filtered_metrics.empty:
            st.info("No measurement data matches the selected date range.")
        else:
            from plotly.subplots import make_subplots

            row_count = len(selected_layer_defs)
            # vertical_spacing is a fraction of the FIGURE height per gap, so with 12
            # metrics the default 0.09 leaves 11 * 0.09 = 99% for gaps and squashes
            # every row to a few pixels. Keep total spacing under ~25% instead.
            vertical_spacing = min(0.09, 0.25 / max(row_count - 1, 1))
            fig = make_subplots(
                rows=row_count,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=vertical_spacing,
                row_heights=[1.0 / row_count] * row_count,
            )
            for row, layer in enumerate(selected_layer_defs, start=1):
                series = filtered_metrics[layer["column"]].dropna()
                if series.empty:
                    continue
                convert = layer.get("convert")
                plotted = convert(series) if convert else series
                hover_func = layer.get("hover_func")
                # Raw data stays the main line unless the trend overlay is on, in which case
                # it fades into the background so the trend reads as the primary series.
                raw_line_kwargs = {
                    "color": layer["color"],
                    "width": 2,
                } if not show_trendlines else {
                    "color": faded_color(layer["color"], 0.3),
                    "width": 1,
                }
                trace_extra: dict = {}
                if hover_func is not None:
                    trace_extra["customdata"] = hover_func(plotted)
                    hovertemplate = "%{x|%b %d, %Y}<br>%{customdata}<extra>" + layer["title"] + "</extra>"
                else:
                    hovertemplate = "%{x|%b %d, %Y}<br>%{y:.2f}<extra>" + layer["title"] + "</extra>"
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=plotted.values,
                        mode="lines+markers",
                        name=layer["title"],
                        line=dict(**raw_line_kwargs),
                        marker=dict(
                            size=5 if not show_trendlines else 3,
                            color=layer["color"] if not show_trendlines else faded_color(layer["color"], 0.4),
                        ),
                        hovertemplate=hovertemplate,
                        **trace_extra,
                    ),
                    row=row,
                    col=1,
                )
                axis_kwargs: dict = {"title": layer["y_title"]}
                tick_func = layer.get("tick_func")
                if tick_func is not None:
                    tick_values, tick_text = tick_func(plotted)
                    if tick_values:
                        axis_kwargs.update(tickvals=tick_values, ticktext=tick_text)
                fig.update_yaxes(row=row, col=1, **axis_kwargs)

                if show_trendlines and series.shape[0] >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=smoothed_trendline(plotted),
                            mode="lines",
                            name=layer["title"] + " trend",
                            line=dict(color=layer["color"], width=2),
                            hovertemplate="%{x|%b %d, %Y}<br>Trend: %{y:.2f}<extra>"
                            + layer["title"]
                            + "</extra>",
                        ),
                        row=row,
                        col=1,
                    )
            fig.update_layout(
                template="plotly_white",
                height=240 * row_count + 60,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(visible=False),
            )
            fig.update_xaxes(title="Date", row=row_count, col=1)
            st.plotly_chart(fig, width="stretch")

with tab4:
    st.subheader("Records")
    st.caption(
        "Best single-workout, most-in-a-day, streak, and best-body-measurement records, "
        "computed from the current date range and activity-type filters."
    )

    records_metrics_frame = build_metrics_frame(st.session_state.get("health_metrics", ({}, {}))[0])

    sections = [
        ("Workout Records (best single workout)", build_workout_records(filtered_df)),
        ("Streaks (consecutive days)", build_streak_records(filtered_df, records_metrics_frame)),
        ("Most in a Day (workouts)", build_daily_workout_records(filtered_df)),
        ("Most in a Day (health)", build_daily_health_records(records_metrics_frame)),
        ("Running Records", build_running_records(records_metrics_frame)),
        ("Best Body Measurements", build_body_measurement_records(records_metrics_frame)),
    ]

    shown_any = False
    for title, records in sections:
        if records:
            st.markdown(f"#### {title}")
            render_record_grid(records)
            shown_any = True

    if not shown_any:
        st.info("No data available for the current filters. Widen the date range or activity types in the sidebar.")
