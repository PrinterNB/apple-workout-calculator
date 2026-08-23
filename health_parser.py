from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
import math
import re
import xml.etree.ElementTree as ET

import gpxpy


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
    total_distance_unit: str = ""
    total_energy_unit: str = ""
    source_name: str = ""
    source_version: str = ""
    device: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600.0

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0


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


def _utc_for_comparison(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
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


def parse_workout_export(export_path: str | Path) -> list[WorkoutRecord]:
    export_path = Path(export_path)
    workouts: list[WorkoutRecord] = []

    if not export_path.exists():
        return workouts

    for event, elem in ET.iterparse(export_path, events=("end",)):
        if elem.tag.split("}")[-1] != "Workout":
            continue

        attrib = dict(elem.attrib)
        start_date = parse_apple_datetime(attrib.get("startDate"))
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
        energy_raw = _workout_value(elem, ("totalEnergyBurned", "totalEnergy", "energyBurned", "energy"))
        energy_value = parse_numeric(energy_raw)
        energy_unit = _workout_value(elem, ("totalEnergyBurnedUnit", "energyUnit")) or "kcal"

        metadata = _extract_metadata_from_element(elem)
        if "MetadataEntry" in metadata and isinstance(metadata["MetadataEntry"], dict):
            metadata.update(metadata["MetadataEntry"])

        workout = WorkoutRecord(
            uuid=attrib.get("uuid", ""),
            activity_type=clean_activity_type(attrib.get("workoutActivityType", "")),
            start_date=start_date,
            end_date=end_date,
            duration_seconds=duration_seconds,
            total_distance_m=normalize_distance_meters(distance_value, distance_unit),
            total_energy_kcal=normalize_energy_kcal(energy_value, energy_unit),
            total_distance_unit=distance_unit,
            total_energy_unit=energy_unit,
            source_name=attrib.get("sourceName", ""),
            source_version=attrib.get("sourceVersion", ""),
            device=attrib.get("device", ""),
            metadata=metadata,
        )
        workouts.append(workout)
        elem.clear()

    empty_date = datetime.min.replace(tzinfo=timezone.utc)
    workouts.sort(key=lambda item: _utc_for_comparison(item.start_date) or empty_date)
    return workouts


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

    start_time = points[0].timestamp if points else None
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


def parse_route_directory(route_dir: str | Path | None) -> list[RouteRecord]:
    if not route_dir:
        return []
    base = Path(route_dir)
    if not base.exists() or not base.is_dir():
        return []

    routes: list[RouteRecord] = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".gpx", ".xml"}:
            route = parse_route_file(path)
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


def route_speeds_kph(route: RouteRecord) -> list[Optional[float]]:
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
        speeds.append((meters / seconds) * 3.6)
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
        route_start = _utc_for_comparison(route.start_time)
        workout_start = _utc_for_comparison(workout.start_date)
        route_end = _utc_for_comparison(route.end_time)
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
