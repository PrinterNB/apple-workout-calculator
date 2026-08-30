from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Optional
import math
import os
import re
import xml.etree.ElementTree as ET

import gpxpy

try:
    from lxml import etree as FAST_XML_ET
except ImportError:
    FAST_XML_ET = None


# Optional progress reporter: set_progress_callback(fn) where fn(done, total)
# is called during long parses so the UI can show live progress and an ETA.
_progress_callback: Optional[Any] = None


def set_progress_callback(callback: Optional[Any]) -> None:
    global _progress_callback
    _progress_callback = callback


def _report_progress(done: int, total: int) -> None:
    if _progress_callback is not None:
        _progress_callback(done, total)


def _count_markers(export_path: Path, needles: list[bytes]) -> list[int]:
    """Fast binary scan counting how often each byte marker occurs in the file."""
    chunk_size = 8 * 1024 * 1024
    counts = [0] * len(needles)
    tails = [b""] * len(needles)
    with open(export_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            for index, needle in enumerate(needles):
                window = tails[index] + chunk
                counts[index] += window.count(needle) - tails[index].count(needle)
                tails[index] = window[-(len(needle) - 1):] if len(needle) > 1 else b""
    return counts


UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@dataclass
class WorkoutRecord:
    uuid: str
    activity_type: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    duration_seconds: float
    total_distance_m: Optional[float]
    total_energy_kcal: Optional[float]
    active_energy_kcal: Optional[float] = None
    basal_energy_kcal: Optional[float] = None
    total_distance_unit: str = ""
    total_energy_unit: str = ""
    source_name: str = ""
    source_version: str = ""
    device: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    heart_rate_samples: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def average_heart_rate_bpm(self) -> Optional[float]:
        if not self.heart_rate_samples:
            return None
        return sum(value for _, value in self.heart_rate_samples) / len(self.heart_rate_samples)

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600.0

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0


@dataclass
class MetricSample:
    timestamp: datetime
    value: float


@dataclass
class RoutePoint:
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class RouteRecord:
    file_path: Path
    workout_uuid: Optional[str]
    route_uuid: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    points: list[RoutePoint]
    source_format: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    @property
    def has_trackpoints(self) -> bool:
        return len(self.points) > 0


def clean_activity_type(value: str) -> str:
    if not value:
        return "Unknown"
    cleaned = value.replace("HKWorkoutActivityType", "")
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned or "Unknown"


# Apple exports record types with a long prefix, e.g.
# HKQuantityTypeIdentifierBodyMass or HKCategoryTypeIdentifierSleepAnalysis.
# The prefix is dropped so callers match on the short form (e.g. "bodymass").
_RECORD_TYPE_PREFIXES = (
    "hkquantitytypeidentifier",
    "hkcategorytypeidentifier",
    "hkworkouttypeidentifier",
    "hkderivedtypeidentifier",
    "hkdiagnostictypeidentifier",
    "hkobjecttypeidentifier",
    "hkcodetypeidentifier",
    "hkcrosstertypeidentifier",
)


def normalize_record_type(raw: Optional[str]) -> str:
    if not raw:
        return ""
    cleaned = re.sub(r"[^a-z0-9]", "", raw.lower())
    for prefix in _RECORD_TYPE_PREFIXES:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):]
    return cleaned


def parse_apple_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    candidates = [
        value,
        value.replace("Z", "+00:00"),
        value.replace(" ", "T", 1).replace(" ", ""),
    ]
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\u2212", "-")
    if not text:
        return None
    if "," in text:
        if "." in text or len(text.rsplit(",", 1)[-1]) == 3:
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", text)
        return float(match.group(0)) if match else None


def _workout_value(elem: ET.Element, names: tuple[str, ...]) -> Optional[str]:
    wanted = {re.sub(r"[^a-z0-9]", "", name.lower()) for name in names}

    for key, value in elem.attrib.items():
        normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
        if normalized_key in wanted:
            return value

    for child in elem.iter():
        normalized_tag = re.sub(r"[^a-z0-9]", "", child.tag.split("}")[-1].lower())
        if normalized_tag not in wanted:
            continue
        if child.text and child.text.strip():
            return child.text.strip()
        for key in ("value", "val", "quantity"):
            if key in child.attrib:
                return child.attrib[key]
    return None


def _workout_statistic(elem: ET.Element, keywords: tuple[str, ...]) -> tuple[Optional[float], str]:
    candidates: list[tuple[int, float, str]] = []
    for child in elem.iter():
        tag = child.tag.split("}")[-1].lower()
        if tag not in {"workoutstatistics", "statistics", "statistic"}:
            continue

        statistic_type = child.attrib.get("type", "").lower()
        if not any(keyword in statistic_type for keyword in keywords):
            continue

        value = parse_numeric(child.attrib.get("sum") or child.attrib.get("value"))
        if value is None:
            continue
        unit = child.attrib.get("unit", "")
        priority = 0 if "activeenergy" in statistic_type else 1
        candidates.append((priority, value, unit))

    if not candidates:
        return None, ""
    _, value, unit = sorted(candidates, key=lambda item: item[0])[0]
    return value, unit


def normalize_distance_meters(distance: Optional[float], unit: Optional[str]) -> Optional[float]:
    if distance is None:
        return None
    unit_name = (unit or "m").strip().lower()
    conversion = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "km": 1000.0,
        "kilometer": 1000.0,
        "kilometers": 1000.0,
        "mi": 1609.344,
        "mile": 1609.344,
        "miles": 1609.344,
        "ft": 0.3048,
        "foot": 0.3048,
        "feet": 0.3048,
        "yd": 0.9144,
        "yard": 0.9144,
        "yards": 0.9144,
    }
    return float(distance) * conversion.get(unit_name, 1.0)


def normalize_energy_kcal(energy: Optional[float], unit: Optional[str]) -> Optional[float]:
    if energy is None:
        return None
    unit_name = (unit or "kcal").strip().lower()
    if unit_name in {"kj", "kilojoule", "kilojoules"}:
        return float(energy) * 0.239005736
    if unit_name in {"j", "joule", "joules"}:
        return float(energy) * 0.000239005736
    return float(energy)


def normalize_mass_kg(mass: Optional[float], unit: Optional[str]) -> Optional[float]:
    if mass is None:
        return None
    unit_name = (unit or "kg").strip().lower()
    conversion = {
        "g": 0.001,
        "gram": 0.001,
        "grams": 0.001,
        "kg": 1.0,
        "kilo": 1.0,
        "kilogram": 1.0,
        "kilograms": 1.0,
        "lb": 0.45359237,
        "lbs": 0.45359237,
        "pound": 0.45359237,
        "pounds": 0.45359237,
        "oz": 0.028349523125,
        "ounce": 0.028349523125,
        "ounces": 0.028349523125,
    }
    return float(mass) * conversion.get(unit_name, 1.0)


def normalize_height_m(height: Optional[float], unit: Optional[str]) -> Optional[float]:
    if height is None:
        return None
    unit_name = (unit or "cm").strip().lower()
    conversion = {
        "cm": 0.01,
        "centimeter": 0.01,
        "centimeters": 0.01,
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "in": 0.0254,
        "inch": 0.0254,
        "inches": 0.0254,
        "ft": 0.3048,
        "feet": 0.3048,
    }
    return float(height) * conversion.get(unit_name, 0.01)


def _naive_local(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.replace(tzinfo=None)


def resolve_daily_steps(
    steps_by_day: dict, watch_sources: set
) -> list[MetricSample]:
    """Collapse per-source step totals into one number per day.

    Every device that counts steps (Watch, iPhone, ...) writes its own stream
    of the same steps, so summing every record double-counts. Prefer a
    watch's count for a day and fall back to the best single non-watch
    source, which matches what the Health app displays. ``steps_by_day`` maps
    day -> {sourceName: total}; ``watch_sources`` names the sources a record's
    device attribute or source name identified as a watch.
    """
    samples: list[MetricSample] = []
    for day, per_source in sorted(steps_by_day.items()):
        watch_totals = [total for source, total in per_source.items() if source in watch_sources]
        daily_total = max(watch_totals) if watch_totals else max(per_source.values())
        samples.append(MetricSample(datetime.combine(day, time.min), daily_total))
    return samples


def is_watch_record(attrib: dict) -> bool:
    """True when a record's device attribute or source names an Apple Watch.

    Newer exports put the counting device in the ``device`` attribute (e.g.
    "name:Apple Watch, ... model:Watch ..."); older ones only have the
    source name. Either signal is enough to identify watch data generically,
    without relying on any user-specific device naming.
    """
    haystack = f"{attrib.get('device') or ''} {attrib.get('sourceName') or ''}".lower()
    return "watch" in haystack


def parse_health_metrics(
    export_path: str | Path,
    range_start: Optional[date] = None,
    range_end: Optional[date] = None,
) -> tuple[dict[str, list[MetricSample]], dict[str, int]]:
    """Stream the export and collect body measurement samples.

    When ``range_start`` / ``range_end`` are given, only records inside that
    inclusive day range are collected, so a narrowed time frame loads faster.

    Returns a dict keyed by canonical column name with one entry per day
    (sleep hours and steps are summed per calendar day), plus a count of how
    many of each recognized body-record type the file actually contains.
    """
    metrics: dict[str, list[MetricSample]] = {
        "weight_kg": [],
        "body_fat_pct": [],
        "height_m": [],
        "resting_hr_bpm": [],
        "sleep_hours": [],
        "steps": [],
        "walk_run_distance_m": [],
        "move_energy_kcal": [],
        "exercise_minutes": [],
        "stand_hours": [],
    }
    record_counts: dict[str, int] = {
        "bodymass": 0,
        "bodyfatpercentage": 0,
        "height": 0,
        "restingheart": 0,
        "sleepanalysis": 0,
        "stepcount": 0,
        "distancewalkingrunning": 0,
        "activeenergyburned": 0,
        "appleexercisetime": 0,
        "applestandtime": 0,
    }
    export_path = Path(export_path)
    if not export_path.exists():
        return metrics, record_counts

    def _day_in_range(day: Optional[date]) -> bool:
        if day is None:
            return range_start is None and range_end is None
        if range_start is not None and day < range_start:
            return False
        if range_end is not None and day > range_end:
            return False
        return True

    def _sleep_in_range(start_dt: Optional[datetime], end_dt: Optional[datetime]) -> bool:
        if range_start is None and range_end is None:
            return True
        if start_dt is None:
            return False
        last_day = (end_dt if (end_dt is not None and end_dt >= start_dt) else start_dt).date()
        if range_start is not None and last_day < range_start:
            return False
        if range_end is not None and start_dt.date() > range_end:
            return False
        return True

    def _distribute_across_days(by_day: dict, start_dt: datetime, end_dt: datetime, value: float) -> None:
        """Split ``value`` across the calendar days the sample spans, by overlap."""
        total_seconds = (end_dt - start_dt).total_seconds()
        day = start_dt.date()
        end_day = end_dt.date()
        while day <= end_day:
            if _day_in_range(day):
                day_start = datetime.combine(day, time.min)
                day_end = day_start + timedelta(days=1)
                overlap = (min(end_dt, day_end) - max(start_dt, day_start)).total_seconds()
                share = value * (overlap / total_seconds) if total_seconds > 0 else value
                if overlap > 0 or total_seconds <= 0:
                    by_day[day] = by_day.get(day, 0.0) + share
            day = day + timedelta(days=1)

    if FAST_XML_ET is not None:
        xml_events = FAST_XML_ET.iterparse(
            str(export_path), events=("end",), recover=True, huge_tree=True
        )
    else:
        xml_events = ET.iterparse(export_path, events=("end",))

    sleep_hours_by_day: dict = {}
    steps_by_day: dict = {}  # date -> {sourceName: steps}
    steps_watch_sources: set = set()
    walk_run_distance_by_day: dict = {}  # date -> {sourceName: meters}
    move_energy_by_day: dict = {}  # date -> {sourceName: kcal}
    exercise_by_day: dict = {}  # date -> {sourceName: minutes}
    stand_by_day: dict = {}  # date -> {sourceName: minutes}

    progress_total: Optional[int] = None
    progress_stride = 1
    if _progress_callback is not None:
        progress_total = _count_markers(export_path, [b"<Record "])[0]
        progress_stride = max(1, progress_total // 200)
    records_seen = 0

    for event, elem in xml_events:
        if elem.tag.split("}")[-1] != "Record":
            continue
        records_seen += 1
        if progress_total is not None and (
            records_seen == progress_total or records_seen % progress_stride == 0
        ):
            _report_progress(records_seen, progress_total)
        record_type = normalize_record_type(elem.attrib.get("type"))
        start = _naive_local(parse_apple_datetime(elem.attrib.get("startDate")))
        end = _naive_local(parse_apple_datetime(elem.attrib.get("endDate")))

        if record_type == "bodymass":
            if not _day_in_range(start.date() if start else None):
                elem.clear()
                continue
            record_counts["bodymass"] += 1
            value = normalize_mass_kg(parse_numeric(elem.attrib.get("value")), elem.attrib.get("unit"))
            if start and value is not None:
                metrics["weight_kg"].append(MetricSample(start, value))
        elif record_type == "bodyfatpercentage":
            if not _day_in_range(start.date() if start else None):
                elem.clear()
                continue
            record_counts["bodyfatpercentage"] += 1
            value = parse_numeric(elem.attrib.get("value"))
            if start and value is not None:
                # Apple exports this as a fraction (0.15 = 15% body fat).
                # Values already above 1 are treated as plain percentages.
                if value <= 1.0:
                    value *= 100.0
                metrics["body_fat_pct"].append(MetricSample(start, value))
        elif record_type == "height":
            if not _day_in_range(start.date() if start else None):
                elem.clear()
                continue
            record_counts["height"] += 1
            value = normalize_height_m(parse_numeric(elem.attrib.get("value")), elem.attrib.get("unit"))
            if start and value is not None:
                metrics["height_m"].append(MetricSample(start, value))
        elif record_type.startswith("restingheart"):
            if not _day_in_range(start.date() if start else None):
                elem.clear()
                continue
            record_counts["restingheart"] += 1
            value = parse_numeric(elem.attrib.get("value"))
            if start and value is not None:
                metrics["resting_hr_bpm"].append(MetricSample(start, value))
        elif record_type == "stepcount":
            if not _day_in_range(start.date() if start else None):
                elem.clear()
                continue
            record_counts["stepcount"] += 1
            value = parse_numeric(elem.attrib.get("value"))
            if start is not None and value:
                end_dt = end if (end is not None and end > start) else start
                distributed: dict = {}
                _distribute_across_days(distributed, start, end_dt, float(value))
                source = elem.attrib.get("sourceName") or "unknown"
                if is_watch_record(elem.attrib):
                    steps_watch_sources.add(source)
                for day, share in distributed.items():
                    per_source = steps_by_day.setdefault(day, {})
                    per_source[source] = per_source.get(source, 0.0) + share
        elif record_type == "distancewalkingrunning":
            if not _day_in_range(start.date() if start else None):
                elem.clear()
                continue
            record_counts["distancewalkingrunning"] += 1
            value = parse_numeric(elem.attrib.get("value"))
            if start and value:
                meters = normalize_distance_meters(float(value), elem.attrib.get("unit"))
                if meters is not None:
                    source = elem.attrib.get("sourceName") or "unknown"
                    by_source = walk_run_distance_by_day.setdefault(start.date(), {})
                    by_source[source] = by_source.get(source, 0.0) + meters
        elif record_type == "activeenergyburned":
            if not _day_in_range(start.date() if start else None):
                elem.clear()
                continue
            record_counts["activeenergyburned"] += 1
            value = parse_numeric(elem.attrib.get("value"))
            if start and value:
                kcal = normalize_energy_kcal(float(value), elem.attrib.get("unit"))
                if kcal is not None:
                    source = elem.attrib.get("sourceName") or "unknown"
                    by_source = move_energy_by_day.setdefault(start.date(), {})
                    by_source[source] = by_source.get(source, 0.0) + kcal
        elif record_type == "appleexercisetime":
            if not _day_in_range(start.date() if start else None):
                elem.clear()
                continue
            record_counts["appleexercisetime"] += 1
            value = parse_numeric(elem.attrib.get("value"))
            if start and value:
                minutes = normalize_duration(float(value), elem.attrib.get("unit")) / 60.0
                source = elem.attrib.get("sourceName") or "unknown"
                by_source = exercise_by_day.setdefault(start.date(), {})
                by_source[source] = by_source.get(source, 0.0) + minutes
        elif record_type == "applestandtime":
            if not _day_in_range(start.date() if start else None):
                elem.clear()
                continue
            record_counts["applestandtime"] += 1
            value = parse_numeric(elem.attrib.get("value"))
            if start and value:
                minutes = normalize_duration(float(value), elem.attrib.get("unit")) / 60.0
                source = elem.attrib.get("sourceName") or "unknown"
                by_source = stand_by_day.setdefault(start.date(), {})
                by_source[source] = by_source.get(source, 0.0) + minutes
        elif record_type.startswith("sleepanalysis"):
            if not _sleep_in_range(start, end):
                elem.clear()
                continue
            record_counts["sleepanalysis"] += 1
            sleep_value = (elem.attrib.get("sleepValue") or "").lower()
            counts = "asleep" in sleep_value or not sleep_value
            duration = 0.0
            if counts and start and end and end >= start:
                duration = (end - start).total_seconds()
            if duration <= 0 and counts:
                duration = parse_numeric(elem.attrib.get("duration")) or 0.0
            if duration > 0 and start is not None:
                end_dt = end if (end is not None and end > start) else start
                _distribute_across_days(sleep_hours_by_day, start, end_dt, duration / 3600.0)
        elem.clear()

    metrics["sleep_hours"] = [
        MetricSample(datetime.combine(day, time.min), hours)
        for day, hours in sorted(sleep_hours_by_day.items())
    ]
    metrics["steps"] = resolve_daily_steps(steps_by_day, steps_watch_sources)
    # Each device (Watch, iPhone) writes its own samples for the same day, so
    # summing every source double counts. Keep the best single-source total per
    # day, matching what the Health app shows.
    metrics["walk_run_distance_m"] = [
        MetricSample(datetime.combine(day, time.min), max(by_source.values()))
        for day, by_source in sorted(walk_run_distance_by_day.items())
    ]
    metrics["move_energy_kcal"] = [
        MetricSample(datetime.combine(day, time.min), max(by_source.values()))
        for day, by_source in sorted(move_energy_by_day.items())
    ]
    metrics["exercise_minutes"] = [
        MetricSample(datetime.combine(day, time.min), max(by_source.values()))
        for day, by_source in sorted(exercise_by_day.items())
    ]
    metrics["stand_hours"] = [
        MetricSample(datetime.combine(day, time.min), max(by_source.values()) / 60.0)
        for day, by_source in sorted(stand_by_day.items())
    ]
    for samples in metrics.values():
        samples.sort(key=lambda sample: sample.timestamp)
    return metrics, record_counts


def _utc_for_comparison(
    value: Optional[datetime], assume_timezone: Optional[Any] = timezone.utc
) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=assume_timezone or timezone.utc)
    return value.astimezone(timezone.utc)


def normalize_duration(duration: Optional[float], unit: Optional[str]) -> float:
    if duration is None:
        return 0.0
    if not unit:
        return float(duration)
    unit = unit.lower()
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return float(duration)
    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return float(duration) * 60.0
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return float(duration) * 3600.0
    return float(duration)


def _extract_metadata_from_element(elem: ET.Element) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in elem.attrib.items():
        metadata[key] = value
    for child in list(elem):
        tag = child.tag.split("}")[-1]
        text = (child.text or "").strip()
        if text:
            metadata[tag] = text
        if child.attrib:
            metadata[tag] = dict(child.attrib) if len(child.attrib) > 1 else next(iter(child.attrib.values()))
    return metadata


def _find_uuid(value: str | None) -> Optional[str]:
    if not value:
        return None
    match = UUID_RE.search(value)
    return match.group(0) if match else None


def parse_workout_export(
    export_path: str | Path,
    range_start: Optional[date] = None,
    range_end: Optional[date] = None,
) -> list[WorkoutRecord]:
    export_path = Path(export_path)
    workouts: list[WorkoutRecord] = []
    heart_rate_samples: list[tuple[datetime, float]] = []
    active_energy_samples: list[tuple[datetime, float]] = []
    basal_energy_samples: list[tuple[datetime, float]] = []

    if not export_path.exists():
        return workouts

    # Samples one day past each bound are kept so workouts starting at the
    # edge of the range still get their full heart-rate/energy history.
    sample_start = range_start - timedelta(days=1) if range_start is not None else None
    sample_end = range_end + timedelta(days=1) if range_end is not None else None

    def _sample_day_in_range(day: date) -> bool:
        if sample_start is not None and day < sample_start:
            return False
        if sample_end is not None and day > sample_end:
            return False
        return True

    if FAST_XML_ET is not None:
        xml_events = FAST_XML_ET.iterparse(
            str(export_path),
            events=("end",),
            recover=True,
            huge_tree=True,
        )
    else:
        xml_events = ET.iterparse(export_path, events=("end",))

    progress_total: Optional[int] = None
    progress_stride = 1
    if _progress_callback is not None:
        record_total, workout_total = _count_markers(export_path, [b"<Record ", b"<Workout "])
        progress_total = record_total + workout_total
        progress_stride = max(1, progress_total // 200)
    records_seen = 0

    def _tick_progress() -> None:
        nonlocal records_seen
        if progress_total is None:
            return
        records_seen += 1
        if records_seen == progress_total or records_seen % progress_stride == 0:
            _report_progress(records_seen, progress_total)

    for event, elem in xml_events:
        tag = elem.tag.split("}")[-1]
        if tag == "Record":
            _tick_progress()
            record_type = elem.attrib.get("type", "").lower()
            if "heartrate" in record_type:
                sample_time = parse_apple_datetime(elem.attrib.get("startDate"))
                sample_value = parse_numeric(elem.attrib.get("value"))
                if sample_time and sample_value is not None and _sample_day_in_range(sample_time.date()):
                    heart_rate_samples.append((sample_time, sample_value))
            elif "activeenergyburned" in record_type:
                sample_time = parse_apple_datetime(elem.attrib.get("startDate"))
                sample_value = normalize_energy_kcal(
                    parse_numeric(elem.attrib.get("value")),
                    elem.attrib.get("unit", "kcal"),
                )
                if sample_time and sample_value is not None and _sample_day_in_range(sample_time.date()):
                    active_energy_samples.append((sample_time, sample_value))
            elif "basalenergyburned" in record_type:
                sample_time = parse_apple_datetime(elem.attrib.get("startDate"))
                sample_value = normalize_energy_kcal(
                    parse_numeric(elem.attrib.get("value")),
                    elem.attrib.get("unit", "kcal"),
                )
                if sample_time and sample_value is not None and _sample_day_in_range(sample_time.date()):
                    basal_energy_samples.append((sample_time, sample_value))
            elem.clear()
            continue

        if tag != "Workout":
            continue
        _tick_progress()

        attrib = dict(elem.attrib)
        start_date = parse_apple_datetime(attrib.get("startDate"))
        if range_start is not None or range_end is not None:
            workout_day = start_date.date() if start_date else None
            if (
                workout_day is None
                or (range_start is not None and workout_day < range_start)
                or (range_end is not None and workout_day > range_end)
            ):
                elem.clear()
                continue
        end_date = parse_apple_datetime(attrib.get("endDate"))
        duration_raw = parse_numeric(attrib.get("duration"))
        duration_seconds = normalize_duration(duration_raw, attrib.get("durationUnit"))
        if duration_seconds == 0.0 and start_date and end_date:
            comparable_start = _utc_for_comparison(start_date)
            comparable_end = _utc_for_comparison(end_date)
            duration_seconds = max(0.0, (comparable_end - comparable_start).total_seconds())

        distance_raw = _workout_value(elem, ("totalDistance", "distance"))
        distance_value = parse_numeric(distance_raw)
        distance_unit = _workout_value(elem, ("totalDistanceUnit", "distanceUnit")) or ""
        if distance_value is None:
            distance_value, distance_unit = _workout_statistic(
                elem,
                ("distance", "walkingrunningdistance", "cyclingdistance", "runningdistance"),
            )
        energy_raw = _workout_value(elem, ("totalEnergyBurned", "totalEnergy", "energyBurned", "energy"))
        energy_value = parse_numeric(energy_raw)
        energy_unit = _workout_value(elem, ("totalEnergyBurnedUnit", "energyUnit")) or "kcal"
        if energy_value is None:
            energy_value, statistic_unit = _workout_statistic(
                elem,
                ("activeenergyburned", "totalenergyburned", "energyburned"),
            )
            energy_unit = statistic_unit or energy_unit
        active_stat_value, active_stat_unit = _workout_statistic(elem, ("activeenergyburned",))
        basal_stat_value, basal_stat_unit = _workout_statistic(elem, ("basalenergyburned",))
        direct_active_energy_kcal = normalize_energy_kcal(energy_value, energy_unit)
        active_energy_kcal = (
            direct_active_energy_kcal
            if direct_active_energy_kcal is not None
            else normalize_energy_kcal(active_stat_value, active_stat_unit)
        )
        basal_energy_kcal = normalize_energy_kcal(basal_stat_value, basal_stat_unit)

        metadata = _extract_metadata_from_element(elem)
        if "MetadataEntry" in metadata and isinstance(metadata["MetadataEntry"], dict):
            metadata.update(metadata["MetadataEntry"])

        workout_uuid = (attrib.get("uuid") or "").strip()
        if not workout_uuid:
            # Some exports, including Connect sources, omit workout UUIDs.
            # Keep each record addressable so route matches do not overwrite one another.
            start_key = start_date.isoformat() if start_date else "unknown-time"
            workout_uuid = f"generated-workout-{len(workouts) + 1:06d}-{start_key}"

        workout = WorkoutRecord(
            uuid=workout_uuid,
            activity_type=clean_activity_type(attrib.get("workoutActivityType", "")),
            start_date=start_date,
            end_date=end_date,
            duration_seconds=duration_seconds,
            total_distance_m=normalize_distance_meters(distance_value, distance_unit),
            total_energy_kcal=active_energy_kcal,
            active_energy_kcal=active_energy_kcal,
            basal_energy_kcal=basal_energy_kcal,
            total_distance_unit=distance_unit,
            total_energy_unit=energy_unit,
            source_name=attrib.get("sourceName", ""),
            source_version=attrib.get("sourceVersion", ""),
            device=attrib.get("device", ""),
            metadata=metadata,
        )
        workouts.append(workout)
        elem.clear()

    heart_rate_samples.sort(key=lambda sample: _utc_for_comparison(sample[0]).timestamp())
    sample_times = [_utc_for_comparison(sample[0]).timestamp() for sample in heart_rate_samples]
    active_energy_samples.sort(key=lambda sample: _utc_for_comparison(sample[0]).timestamp())
    active_energy_times = [_utc_for_comparison(sample[0]).timestamp() for sample in active_energy_samples]
    basal_energy_samples.sort(key=lambda sample: _utc_for_comparison(sample[0]).timestamp())
    basal_energy_times = [_utc_for_comparison(sample[0]).timestamp() for sample in basal_energy_samples]
    for workout in workouts:
        workout_start = _utc_for_comparison(workout.start_date)
        workout_end = _utc_for_comparison(workout.end_date) or workout_start
        if not workout_start or not workout_end:
            continue

        start_timestamp = workout_start.timestamp()
        end_timestamp = workout_end.timestamp()
        start_index = bisect_left(sample_times, start_timestamp)
        end_index = bisect_right(sample_times, end_timestamp)
        workout.heart_rate_samples = [
            heart_rate_samples[index][0:2] for index in range(start_index, end_index)
        ]

        active_start_index = bisect_left(active_energy_times, start_timestamp)
        active_end_index = bisect_right(active_energy_times, end_timestamp)
        if workout.active_energy_kcal is None and active_start_index < active_end_index:
            workout.active_energy_kcal = sum(
                active_energy_samples[index][1]
                for index in range(active_start_index, active_end_index)
            )

        basal_start_index = bisect_left(basal_energy_times, start_timestamp)
        basal_end_index = bisect_right(basal_energy_times, end_timestamp)
        if workout.basal_energy_kcal is None and basal_start_index < basal_end_index:
            workout.basal_energy_kcal = sum(
                basal_energy_samples[index][1]
                for index in range(basal_start_index, basal_end_index)
            )

        if workout.active_energy_kcal is not None and workout.basal_energy_kcal is not None:
            workout.total_energy_kcal = workout.active_energy_kcal + workout.basal_energy_kcal

    empty_date = datetime.min.replace(tzinfo=timezone.utc)
    workouts.sort(key=lambda item: _utc_for_comparison(item.start_date) or empty_date)
    return workouts


def parse_export_all(
    export_path: str | Path,
    range_start: Optional[date] = None,
    range_end: Optional[date] = None,
) -> tuple[
    list[WorkoutRecord],
    dict[str, list[MetricSample]],
    dict[str, int],
]:
    """Stream the export once, collecting workouts and health metrics together.

    This is the fast path used by the app: one pass over ``export.xml``
    replaces the separate :func:`parse_workout_export` and
    :func:`parse_health_metrics` passes, so a large export is only read and
    XML-parsed a single time. Returns the workouts list, the same metrics
    dict and per-type record counts the individual parsers produce.
    """
    workouts: list[WorkoutRecord] = []
    heart_rate_samples: list[tuple[datetime, float]] = []
    active_energy_samples: list[tuple[datetime, float]] = []
    basal_energy_samples: list[tuple[datetime, float]] = []

    metrics: dict[str, list[MetricSample]] = {
        "weight_kg": [],
        "body_fat_pct": [],
        "height_m": [],
        "resting_hr_bpm": [],
        "sleep_hours": [],
        "steps": [],
        "walk_run_distance_m": [],
        "move_energy_kcal": [],
        "exercise_minutes": [],
        "stand_hours": [],
    }
    record_counts: dict[str, int] = {
        "bodymass": 0,
        "bodyfatpercentage": 0,
        "height": 0,
        "restingheart": 0,
        "sleepanalysis": 0,
        "stepcount": 0,
        "distancewalkingrunning": 0,
        "activeenergyburned": 0,
        "appleexercisetime": 0,
        "applestandtime": 0,
    }
    export_path = Path(export_path)
    if not export_path.exists():
        return workouts, metrics, record_counts

    def _day_in_range(day: Optional[date]) -> bool:
        if day is None:
            return range_start is None and range_end is None
        if range_start is not None and day < range_start:
            return False
        if range_end is not None and day > range_end:
            return False
        return True

    # Samples one day past each bound are kept so workouts starting at the
    # edge of the range still get their full heart-rate/energy history.
    sample_start = range_start - timedelta(days=1) if range_start is not None else None
    sample_end = range_end + timedelta(days=1) if range_end is not None else None

    def _sample_day_in_range(day: date) -> bool:
        if sample_start is not None and day < sample_start:
            return False
        if sample_end is not None and day > sample_end:
            return False
        return True

    # A cheap day check on the raw ISO timestamp string ("2026-08-28T07:49:12+00:00")
    # rejects out-of-range records before building datetime objects at all.
    day_gate_start = range_start.isoformat() if range_start is not None else None
    day_gate_end = range_end.isoformat() if range_end is not None else None
    sample_gate_start = sample_start.isoformat() if sample_start is not None else None
    sample_gate_end = sample_end.isoformat() if sample_end is not None else None

    def _day_gate(raw: Optional[str], low: Optional[str], high: Optional[str]) -> bool:
        """True when ``raw``'s ISO date prefix is definitely outside ``[low, high]``."""
        if raw is None or len(raw) < 10 or raw[4] != "-":
            return False
        day = raw[:10]
        return (low is not None and day < low) or (high is not None and day > high)

    def _sleep_in_range(start_dt: Optional[datetime], end_dt: Optional[datetime]) -> bool:
        if range_start is None and range_end is None:
            return True
        if start_dt is None:
            return False
        last_day = (end_dt if (end_dt is not None and end_dt >= start_dt) else start_dt).date()
        if range_start is not None and last_day < range_start:
            return False
        if range_end is not None and start_dt.date() > range_end:
            return False
        return True

    def _distribute_across_days(by_day: dict, start_dt: datetime, end_dt: datetime, value: float) -> None:
        """Split ``value`` across the calendar days the sample spans, by overlap."""
        total_seconds = (end_dt - start_dt).total_seconds()
        day = start_dt.date()
        end_day = end_dt.date()
        while day <= end_day:
            if _day_in_range(day):
                day_start = datetime.combine(day, time.min)
                day_end = day_start + timedelta(days=1)
                overlap = (min(end_dt, day_end) - max(start_dt, day_start)).total_seconds()
                share = value * (overlap / total_seconds) if total_seconds > 0 else value
                if overlap > 0 or total_seconds <= 0:
                    by_day[day] = by_day.get(day, 0.0) + share
            day = day + timedelta(days=1)

    if FAST_XML_ET is not None:
        xml_events = FAST_XML_ET.iterparse(
            str(export_path), events=("end",), recover=True, huge_tree=True
        )
    else:
        xml_events = ET.iterparse(export_path, events=("end",))

    sleep_hours_by_day: dict = {}
    steps_by_day: dict = {}  # date -> {sourceName: steps}
    steps_watch_sources: set = set()
    walk_run_distance_by_day: dict = {}  # date -> {sourceName: meters}
    move_energy_by_day: dict = {}  # date -> {sourceName: kcal}
    exercise_by_day: dict = {}  # date -> {sourceName: minutes}
    stand_by_day: dict = {}  # date -> {sourceName: minutes}
    type_cache: dict[str, str] = {}

    progress_total: Optional[int] = None
    progress_stride = 1
    if _progress_callback is not None:
        record_total, workout_total = _count_markers(export_path, [b"<Record ", b"<Workout "])
        progress_total = record_total + workout_total
        progress_stride = max(1, progress_total // 200)
    records_seen = 0

    def _tick_progress() -> None:
        nonlocal records_seen
        if progress_total is None:
            return
        records_seen += 1
        if records_seen == progress_total or records_seen % progress_stride == 0:
            _report_progress(records_seen, progress_total)

    for event, elem in xml_events:
        tag = elem.tag.split("}")[-1]
        if tag == "Record":
            _tick_progress()
            attrib = elem.attrib
            raw_type = attrib.get("type") or ""
            record_type = type_cache.get(raw_type)
            if record_type is None:
                record_type = type_cache[raw_type] = normalize_record_type(raw_type)

            # Health metrics. Recognized before the substring checks below on
            # purpose: resting-heart-rate type names also contain "heartrate".
            if (
                record_type
                in (
                    "bodymass",
                    "bodyfatpercentage",
                    "height",
                    "stepcount",
                    "distancewalkingrunning",
                    "appleexercisetime",
                    "applestandtime",
                )
                or record_type.startswith(("restingheart", "sleepanalysis"))
            ):
                start_raw = attrib.get("startDate")
                if record_type.startswith("sleepanalysis"):
                    # A sleep that starts before the range can still overlap it;
                    # only reject when it lies entirely past the end bound.
                    gated_out = _day_gate(start_raw, None, day_gate_end)
                else:
                    gated_out = _day_gate(start_raw, day_gate_start, day_gate_end)
                if not gated_out:
                    start = _naive_local(parse_apple_datetime(start_raw))
                    end = _naive_local(parse_apple_datetime(attrib.get("endDate")))
                    if record_type == "bodymass":
                        if _day_in_range(start.date() if start else None):
                            record_counts["bodymass"] += 1
                            value = normalize_mass_kg(parse_numeric(attrib.get("value")), attrib.get("unit"))
                            if start and value is not None:
                                metrics["weight_kg"].append(MetricSample(start, value))
                    elif record_type == "bodyfatpercentage":
                        if _day_in_range(start.date() if start else None):
                            record_counts["bodyfatpercentage"] += 1
                            value = parse_numeric(attrib.get("value"))
                            if start and value is not None:
                                # Apple exports this as a fraction (0.15 = 15% body fat).
                                # Values already above 1 are treated as plain percentages.
                                if value <= 1.0:
                                    value *= 100.0
                                metrics["body_fat_pct"].append(MetricSample(start, value))
                    elif record_type == "height":
                        if _day_in_range(start.date() if start else None):
                            record_counts["height"] += 1
                            value = normalize_height_m(parse_numeric(attrib.get("value")), attrib.get("unit"))
                            if start and value is not None:
                                metrics["height_m"].append(MetricSample(start, value))
                    elif record_type.startswith("restingheart"):
                        if _day_in_range(start.date() if start else None):
                            record_counts["restingheart"] += 1
                            value = parse_numeric(attrib.get("value"))
                            if start and value is not None:
                                metrics["resting_hr_bpm"].append(MetricSample(start, value))
                    elif record_type == "stepcount":
                        if _day_in_range(start.date() if start else None):
                            record_counts["stepcount"] += 1
                            value = parse_numeric(attrib.get("value"))
                            if start is not None and value:
                                end_dt = end if (end is not None and end > start) else start
                                distributed: dict = {}
                                _distribute_across_days(distributed, start, end_dt, float(value))
                                source = attrib.get("sourceName") or "unknown"
                                if is_watch_record(attrib):
                                    steps_watch_sources.add(source)
                                for day, share in distributed.items():
                                    per_source = steps_by_day.setdefault(day, {})
                                    per_source[source] = per_source.get(source, 0.0) + share
                    elif record_type == "distancewalkingrunning":
                        if _day_in_range(start.date() if start else None):
                            record_counts["distancewalkingrunning"] += 1
                            value = parse_numeric(attrib.get("value"))
                            if start and value:
                                meters = normalize_distance_meters(float(value), attrib.get("unit"))
                                if meters is not None:
                                    source = attrib.get("sourceName") or "unknown"
                                    by_source = walk_run_distance_by_day.setdefault(start.date(), {})
                                    by_source[source] = by_source.get(source, 0.0) + meters
                    elif record_type == "appleexercisetime":
                        if _day_in_range(start.date() if start else None):
                            record_counts["appleexercisetime"] += 1
                            value = parse_numeric(attrib.get("value"))
                            if start and value:
                                minutes = normalize_duration(float(value), attrib.get("unit")) / 60.0
                                source = attrib.get("sourceName") or "unknown"
                                by_source = exercise_by_day.setdefault(start.date(), {})
                                by_source[source] = by_source.get(source, 0.0) + minutes
                    elif record_type == "applestandtime":
                        if _day_in_range(start.date() if start else None):
                            record_counts["applestandtime"] += 1
                            value = parse_numeric(attrib.get("value"))
                            if start and value:
                                minutes = normalize_duration(float(value), attrib.get("unit")) / 60.0
                                source = attrib.get("sourceName") or "unknown"
                                by_source = stand_by_day.setdefault(start.date(), {})
                                by_source[source] = by_source.get(source, 0.0) + minutes
                    elif record_type.startswith("sleepanalysis"):
                        if _sleep_in_range(start, end):
                            record_counts["sleepanalysis"] += 1
                            sleep_value = (attrib.get("sleepValue") or "").lower()
                            counts = "asleep" in sleep_value or not sleep_value
                            duration = 0.0
                            if counts and start and end and end >= start:
                                duration = (end - start).total_seconds()
                            if duration <= 0 and counts:
                                duration = parse_numeric(attrib.get("duration")) or 0.0
                            if duration > 0 and start is not None:
                                end_dt = end if (end is not None and end > start) else start
                                _distribute_across_days(sleep_hours_by_day, start, end_dt, duration / 3600.0)
                elem.clear()
                continue

            lower_type = raw_type.lower()
            if (
                "heartrate" in lower_type
                or "activeenergyburned" in lower_type
                or "basalenergyburned" in lower_type
            ):
                sample_raw = attrib.get("startDate")
                if not _day_gate(sample_raw, sample_gate_start, sample_gate_end):
                    sample_time = parse_apple_datetime(sample_raw)
                    sample_value = parse_numeric(attrib.get("value"))
                    if sample_time and sample_value is not None and _sample_day_in_range(sample_time.date()):
                        if "heartrate" in lower_type:
                            heart_rate_samples.append((sample_time, sample_value))
                        elif "activeenergyburned" in lower_type:
                            energy_value = normalize_energy_kcal(sample_value, attrib.get("unit", "kcal"))
                            if energy_value is not None:
                                active_energy_samples.append((sample_time, energy_value))
                                # Per-day Move metric. Samples one day past the
                                # bounds exist only for workout backfill, so the
                                # daily total stays inside the selected range.
                                day = sample_time.date()
                                if _day_in_range(day):
                                    record_counts["activeenergyburned"] += 1
                                    source = attrib.get("sourceName") or "unknown"
                                    by_source = move_energy_by_day.setdefault(day, {})
                                    by_source[source] = by_source.get(source, 0.0) + energy_value
                        else:
                            energy_value = normalize_energy_kcal(sample_value, attrib.get("unit", "kcal"))
                            if energy_value is not None:
                                basal_energy_samples.append((sample_time, energy_value))
            elem.clear()
            continue

        if tag != "Workout":
            continue
        _tick_progress()

        attrib = dict(elem.attrib)
        start_date = parse_apple_datetime(attrib.get("startDate"))
        if range_start is not None or range_end is not None:
            workout_day = start_date.date() if start_date else None
            if (
                workout_day is None
                or (range_start is not None and workout_day < range_start)
                or (range_end is not None and workout_day > range_end)
            ):
                elem.clear()
                continue
        end_date = parse_apple_datetime(attrib.get("endDate"))
        duration_raw = parse_numeric(attrib.get("duration"))
        duration_seconds = normalize_duration(duration_raw, attrib.get("durationUnit"))
        if duration_seconds == 0.0 and start_date and end_date:
            comparable_start = _utc_for_comparison(start_date)
            comparable_end = _utc_for_comparison(end_date)
            duration_seconds = max(0.0, (comparable_end - comparable_start).total_seconds())

        distance_raw = _workout_value(elem, ("totalDistance", "distance"))
        distance_value = parse_numeric(distance_raw)
        distance_unit = _workout_value(elem, ("totalDistanceUnit", "distanceUnit")) or ""
        if distance_value is None:
            distance_value, distance_unit = _workout_statistic(
                elem,
                ("distance", "walkingrunningdistance", "cyclingdistance", "runningdistance"),
            )
        energy_raw = _workout_value(elem, ("totalEnergyBurned", "totalEnergy", "energyBurned", "energy"))
        energy_value = parse_numeric(energy_raw)
        energy_unit = _workout_value(elem, ("totalEnergyBurnedUnit", "energyUnit")) or "kcal"
        if energy_value is None:
            energy_value, statistic_unit = _workout_statistic(
                elem,
                ("activeenergyburned", "totalenergyburned", "energyburned"),
            )
            energy_unit = statistic_unit or energy_unit
        active_stat_value, active_stat_unit = _workout_statistic(elem, ("activeenergyburned",))
        basal_stat_value, basal_stat_unit = _workout_statistic(elem, ("basalenergyburned",))
        direct_active_energy_kcal = normalize_energy_kcal(energy_value, energy_unit)
        active_energy_kcal = (
            direct_active_energy_kcal
            if direct_active_energy_kcal is not None
            else normalize_energy_kcal(active_stat_value, active_stat_unit)
        )
        basal_energy_kcal = normalize_energy_kcal(basal_stat_value, basal_stat_unit)

        metadata = _extract_metadata_from_element(elem)
        if "MetadataEntry" in metadata and isinstance(metadata["MetadataEntry"], dict):
            metadata.update(metadata["MetadataEntry"])

        workout_uuid = (attrib.get("uuid") or "").strip()
        if not workout_uuid:
            # Some exports, including Connect sources, omit workout UUIDs.
            # Keep each record addressable so route matches do not overwrite one another.
            start_key = start_date.isoformat() if start_date else "unknown-time"
            workout_uuid = f"generated-workout-{len(workouts) + 1:06d}-{start_key}"

        workout = WorkoutRecord(
            uuid=workout_uuid,
            activity_type=clean_activity_type(attrib.get("workoutActivityType", "")),
            start_date=start_date,
            end_date=end_date,
            duration_seconds=duration_seconds,
            total_distance_m=normalize_distance_meters(distance_value, distance_unit),
            total_energy_kcal=active_energy_kcal,
            active_energy_kcal=active_energy_kcal,
            basal_energy_kcal=basal_energy_kcal,
            total_distance_unit=distance_unit,
            total_energy_unit=energy_unit,
            source_name=attrib.get("sourceName", ""),
            source_version=attrib.get("sourceVersion", ""),
            device=attrib.get("device", ""),
            metadata=metadata,
        )
        workouts.append(workout)
        elem.clear()

    heart_rate_samples.sort(key=lambda sample: _utc_for_comparison(sample[0]).timestamp())
    sample_times = [_utc_for_comparison(sample[0]).timestamp() for sample in heart_rate_samples]
    active_energy_samples.sort(key=lambda sample: _utc_for_comparison(sample[0]).timestamp())
    active_energy_times = [_utc_for_comparison(sample[0]).timestamp() for sample in active_energy_samples]
    basal_energy_samples.sort(key=lambda sample: _utc_for_comparison(sample[0]).timestamp())
    basal_energy_times = [_utc_for_comparison(sample[0]).timestamp() for sample in basal_energy_samples]
    for workout in workouts:
        workout_start = _utc_for_comparison(workout.start_date)
        workout_end = _utc_for_comparison(workout.end_date) or workout_start
        if not workout_start or not workout_end:
            continue

        start_timestamp = workout_start.timestamp()
        end_timestamp = workout_end.timestamp()
        start_index = bisect_left(sample_times, start_timestamp)
        end_index = bisect_right(sample_times, end_timestamp)
        workout.heart_rate_samples = [
            heart_rate_samples[index][0:2] for index in range(start_index, end_index)
        ]

        active_start_index = bisect_left(active_energy_times, start_timestamp)
        active_end_index = bisect_right(active_energy_times, end_timestamp)
        if workout.active_energy_kcal is None and active_start_index < active_end_index:
            workout.active_energy_kcal = sum(
                active_energy_samples[index][1]
                for index in range(active_start_index, active_end_index)
            )

        basal_start_index = bisect_left(basal_energy_times, start_timestamp)
        basal_end_index = bisect_right(basal_energy_times, end_timestamp)
        if workout.basal_energy_kcal is None and basal_start_index < basal_end_index:
            workout.basal_energy_kcal = sum(
                basal_energy_samples[index][1]
                for index in range(basal_start_index, basal_end_index)
            )

        if workout.active_energy_kcal is not None and workout.basal_energy_kcal is not None:
            workout.total_energy_kcal = workout.active_energy_kcal + workout.basal_energy_kcal

    # Each device (Watch, iPhone) writes its own samples for the same day, so
    # summing every source double counts. Keep the best single-source total per
    # day, matching what the Health app shows.
    metrics["sleep_hours"] = [
        MetricSample(datetime.combine(day, time.min), hours)
        for day, hours in sorted(sleep_hours_by_day.items())
    ]
    metrics["steps"] = resolve_daily_steps(steps_by_day, steps_watch_sources)
    metrics["walk_run_distance_m"] = [
        MetricSample(datetime.combine(day, time.min), max(by_source.values()))
        for day, by_source in sorted(walk_run_distance_by_day.items())
    ]
    metrics["move_energy_kcal"] = [
        MetricSample(datetime.combine(day, time.min), max(by_source.values()))
        for day, by_source in sorted(move_energy_by_day.items())
    ]
    metrics["exercise_minutes"] = [
        MetricSample(datetime.combine(day, time.min), max(by_source.values()))
        for day, by_source in sorted(exercise_by_day.items())
    ]
    metrics["stand_hours"] = [
        MetricSample(datetime.combine(day, time.min), max(by_source.values()) / 60.0)
        for day, by_source in sorted(stand_by_day.items())
    ]
    for samples in metrics.values():
        samples.sort(key=lambda sample: sample.timestamp)

    empty_date = datetime.min.replace(tzinfo=timezone.utc)
    workouts.sort(key=lambda item: _utc_for_comparison(item.start_date) or empty_date)
    return workouts, metrics, record_counts


def _safe_strip(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_time_from_text(text: str) -> Optional[datetime]:
    for pattern in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(text, pattern)
            return dt
        except ValueError:
            continue
    return parse_apple_datetime(text)


def _extract_route_metadata_text(text: str) -> dict[str, Optional[str]]:
    metadata: dict[str, Optional[str]] = {
        "workout_uuid": None,
        "route_uuid": None,
    }
    uuid_matches = UUID_RE.findall(text)
    if uuid_matches:
        metadata["route_uuid"] = uuid_matches[0]
        if len(uuid_matches) > 1:
            metadata["workout_uuid"] = uuid_matches[1]
    return metadata


def _extract_route_time_from_filename(file_path: Path) -> Optional[datetime]:
    match = re.search(
        r"route[_-](\d{4}-\d{2}-\d{2})[_-](\d{1,2})[.](\d{2})(am|pm)",
        file_path.stem.lower(),
    )
    if not match:
        return None

    date_part, hour_text, minute_text, meridiem = match.groups()
    hour = int(hour_text) % 12
    if meridiem == "pm":
        hour += 12
    return datetime.strptime(
        f"{date_part} {hour:02d}:{minute_text}:00",
        "%Y-%m-%d %H:%M:%S",
    )


def _parse_gpx_route(file_path: Path) -> RouteRecord:
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        gpx = gpxpy.parse(handle)

    points: list[RoutePoint] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(
                    RoutePoint(
                        latitude=point.latitude,
                        longitude=point.longitude,
                        elevation_m=point.elevation,
                        timestamp=point.time,
                    )
                )

    for gpx_route in gpx.routes:
        for point in gpx_route.points:
            points.append(
                RoutePoint(
                    latitude=point.latitude,
                    longitude=point.longitude,
                    elevation_m=point.elevation,
                    timestamp=None,
                )
            )

    for point in gpx.waypoints:
        points.append(
            RoutePoint(
                latitude=point.latitude,
                longitude=point.longitude,
                elevation_m=point.elevation,
                timestamp=point.time,
            )
        )

    start_time = next((point.timestamp for point in points if point.timestamp), None)
    start_time = start_time or _extract_route_time_from_filename(file_path)
    end_time = points[-1].timestamp if points else None
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    metadata = _extract_route_metadata_text(raw_text)
    return RouteRecord(
        file_path=file_path,
        workout_uuid=metadata.get("workout_uuid"),
        route_uuid=metadata.get("route_uuid"),
        start_time=start_time,
        end_time=end_time,
        points=points,
        source_format="gpx",
        metadata=metadata,
        raw_text=raw_text,
    )


def _parse_xml_route(file_path: Path) -> RouteRecord:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    root = ET.fromstring(raw_text)
    metadata = _extract_route_metadata_text(raw_text)
    for key, value in root.attrib.items():
        metadata[key] = value
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if "workout" in tag and "uuid" in tag:
            metadata["workout_uuid"] = _safe_strip(elem.text) or metadata.get("workout_uuid")
        if "route" in tag and "uuid" in tag:
            metadata["route_uuid"] = _safe_strip(elem.text) or metadata.get("route_uuid")

    points: list[RoutePoint] = []
    for elem in root.iter():
        attrib = {k.lower(): v for k, v in elem.attrib.items()}
        lat = parse_numeric(attrib.get("lat") or attrib.get("latitude"))
        lon = parse_numeric(attrib.get("lon") or attrib.get("longitude"))
        if lat is None or lon is None:
            continue
        elevation = parse_numeric(
            attrib.get("ele")
            or attrib.get("elevation")
            or elem.findtext(".//{*}ele")
            or elem.findtext(".//{*}elevation")
        )
        timestamp = None
        for child_tag in ("time", "timestamp", "date"):
            child_text = elem.findtext(f".//{{*}}{child_tag}")
            if child_text:
                timestamp = _extract_time_from_text(child_text.strip())
                break
        points.append(RoutePoint(latitude=lat, longitude=lon, elevation_m=elevation, timestamp=timestamp))

    start_time = next((point.timestamp for point in points if point.timestamp), None)
    start_time = start_time or _extract_route_time_from_filename(file_path)
    end_time = next((point.timestamp for point in reversed(points) if point.timestamp), None)
    return RouteRecord(
        file_path=file_path,
        workout_uuid=metadata.get("workout_uuid"),
        route_uuid=metadata.get("route_uuid"),
        start_time=start_time,
        end_time=end_time,
        points=points,
        source_format="xml",
        metadata=metadata,
        raw_text=raw_text,
    )


def parse_route_file(file_path: str | Path) -> Optional[RouteRecord]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".gpx":
            return _parse_gpx_route(path)
        if suffix == ".xml":
            return _parse_xml_route(path)
    except Exception:
        return None
    return None


def parse_route_directory(
    route_dir: str | Path | None,
    range_start: Optional[date] = None,
    range_end: Optional[date] = None,
) -> list[RouteRecord]:
    if not route_dir:
        return []
    base = Path(route_dir)
    if not base.exists() or not base.is_dir():
        return []

    paths = [
        path
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".gpx", ".xml"}
    ]
    if range_start is not None or range_end is not None:
        # Route filenames embed the workout date (route_YYYY-MM-DD_...); files
        # dated outside the range are skipped before GPX parsing. Files without
        # a usable filename date are still parsed.
        kept: list[Path] = []
        for path in paths:
            file_time = _extract_route_time_from_filename(path)
            if file_time is not None:
                if (range_start is not None and file_time.date() < range_start) or (
                    range_end is not None and file_time.date() > range_end
                ):
                    continue
            kept.append(path)
        paths = kept
    def _reporting(files):
        for index, route in enumerate(files, 1):
            if _progress_callback is not None:
                _report_progress(index, len(paths))
            yield route

    if len(paths) < 4:
        parsed_routes = list(_reporting(parse_route_file(path) for path in paths))
    else:
        # Threads avoid Windows multiprocessing re-importing app.py while still
        # allowing lxml-backed parsing to work concurrently without Streamlit
        # session-state or spawn-related failures.
        worker_count = min(len(paths), max(4, os.cpu_count() or 1), 32)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            parsed_routes = list(_reporting(executor.map(parse_route_file, paths)))

    routes: list[RouteRecord] = []
    for route in parsed_routes:
        if route and route.has_trackpoints:
            routes.append(route)
    return routes


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def route_distance_meters(route: RouteRecord) -> float:
    total = 0.0
    for previous, current in zip(route.points, route.points[1:]):
        total += _haversine_meters(previous.latitude, previous.longitude, current.latitude, current.longitude)
    return total


def route_speeds_mph(route: RouteRecord) -> list[Optional[float]]:
    speeds: list[Optional[float]] = [None]
    for previous, current in zip(route.points, route.points[1:]):
        if not previous.timestamp or not current.timestamp:
            speeds.append(None)
            continue
        previous_time = _utc_for_comparison(previous.timestamp)
        current_time = _utc_for_comparison(current.timestamp)
        seconds = (current_time - previous_time).total_seconds()
        if seconds <= 0:
            speeds.append(None)
            continue
        meters = _haversine_meters(previous.latitude, previous.longitude, current.latitude, current.longitude)
        speeds.append((meters / seconds) * 2.236936292)
    return speeds


def route_times(route: RouteRecord) -> list[Optional[datetime]]:
    return [point.timestamp for point in route.points]


def match_route_to_workout(workout: WorkoutRecord, routes: Iterable[RouteRecord]) -> Optional[RouteRecord]:
    routes = list(routes)
    if not routes:
        return None

    workout_uuid = workout.uuid.lower() if workout.uuid else ""
    if workout_uuid:
        for route in routes:
            candidate_text = " ".join(
                str(value).lower()
                for value in (
                    route.workout_uuid,
                    route.route_uuid,
                    route.file_path.name,
                    route.raw_text,
                    route.metadata,
                )
            )
            if workout_uuid in candidate_text:
                return route

    if not workout.start_date:
        return None

    scored: list[tuple[float, RouteRecord]] = []
    for route in routes:
        if not route.start_time:
            continue
        workout_timezone = workout.start_date.tzinfo if workout.start_date else timezone.utc
        route_timezone = workout_timezone if route.start_time and route.start_time.tzinfo is None else timezone.utc
        route_start = _utc_for_comparison(route.start_time, route_timezone)
        workout_start = _utc_for_comparison(workout.start_date)
        route_end_timezone = workout_timezone if route.end_time and route.end_time.tzinfo is None else timezone.utc
        route_end = _utc_for_comparison(route.end_time, route_end_timezone)
        workout_end = _utc_for_comparison(workout.end_date)
        start_gap = abs((route_start - workout_start).total_seconds())
        end_gap = abs((route_end - workout_end).total_seconds()) if route_end and workout_end else 0.0
        scored.append((start_gap + end_gap, route))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0])
    best_gap, best_route = scored[0]
    if best_gap <= max(workout.duration_seconds * 0.75, 4 * 3600.0):
        return best_route
    return None
