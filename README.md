# Apple Health Workout Explorer

A Streamlit app for exploring workouts from an Apple Health export. It provides aggregate workout statistics and an individual workout inspector with route maps, elevation, speed, and heart-rate profiles.

## Features

- Streams `export.xml` with `xml.etree.ElementTree.iterparse`, so the full export is not loaded into memory.
- Filters workouts by date range and activity type; activity types are limited to those present in the selected date range.
- Includes All time, Year to date, Past year, and Custom date range presets.
- Summarizes accumulated hours, workout count, average duration, total distance in miles, and total energy in kcal.
- Treats the workout's direct `totalEnergyBurned` value as active calories when available, and derives total calories by adding basal energy when Apple provides it.
- Shows total miles, total calories, and active calories by workout type for the selected date range.
- Charts workout time by day, week, or month.
- Shows a workout-type duration breakdown.
- Shows the filtered workout table newest first.
- Uses the workout's original local time with standard AM/PM formatting.
- Parses `.gpx` and `.xml` files from an optional `workout-routes/` directory.
- Matches routes using workout UUIDs when available, then by route/workout start and end times.
- Displays route polylines with start/end markers on an Esri satellite basemap using Folium.
- Displays elevation and calculated speed in mph over timestamped route points.
- Displays heart rate over time when Apple Health heart-rate samples are available.
- Displays heart rate even when a workout has no matched GPS route.

## Requirements

- Python 3.10 or newer recommended
- An extracted Apple Health `apple_health_export/` folder containing `export.xml`
- Optional `workout-routes/` subdirectory inside that folder

Install the dependencies into your global Python environment:

```bash
python -m pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

In the sidebar, enter the path to the extracted `apple_health_export/` folder. The app automatically finds:

- `apple_health_export/export.xml`
- `apple_health_export/workout-routes/`, when present

The sidebar shows the total size of the selected export folder. The app reloads data when the folder changes or when you click **Reload data**. If the route directory is absent, workout aggregates still work and route status is shown as `No Route Data`.

Use **Reload data** in the sidebar after adding or changing workouts or route files. Filter changes and workout selection reuse the already-loaded data without reparsing the export or rematching routes.

Route files are parsed in parallel when there are several files. If a Windows worker process fails, route parsing automatically falls back to sequential parsing. The main `export.xml` remains streamed sequentially with `iterparse` to keep memory use low; its parsing speed is usually limited by disk I/O and XML size rather than available CPU.

## Apple Health export

On an iPhone, open the Health app, select your profile, choose **Export All Health Data**, and extract the resulting archive. Enter the extracted `apple_health_export/` folder path in the app.

## App tabs

### Workout Accumulator

The accumulator tab applies the sidebar filters and shows:

- Total accumulated workout hours
- Number of workouts
- Average workout duration
- Total distance in miles
- Total calories and active calories
- Accumulated time grouped by day, week, or month
- Duration grouped by workout type
- Distance, total calories, and active calories grouped by workout type
- A newest-first filtered workout summary table

### Individual Workout Route Inspector

Choose a workout from the filtered set to view its local start/end time in AM/PM format, duration, distance, total calories, active calories, average heart rate, workout type, and route status. When a route is matched, the tab also shows:

- Interactive satellite GPS map with the complete route
- Green start marker and red end marker
- Elevation profile
- Calculated speed profile in miles per hour when trackpoint timestamps are available
- Heart-rate profile in beats per minute when heart-rate samples are available

If no GPS route is matched, the profile section can still display heart rate by itself.

## Route matching

Route matching is attempted in this order:

1. Search route metadata, filenames, and raw route contents for the workout UUID.
2. Compare route start/end timestamps with the workout start/end timestamps.
3. Use Apple route filename timestamps such as `route_2026-08-19_2.03pm.gpx` when trackpoint timestamps are unavailable.

The timestamp fallback accepts the closest route when the combined start/end difference is within the workout duration-based tolerance, with a maximum practical tolerance of four hours. Route files without usable trackpoints are ignored.

GPX tracks, route points, and waypoints are parsed with `gpxpy`. XML route files are parsed generically by looking for latitude/longitude attributes and common elevation and timestamp fields. Apple Health XML variants that use a different structure may require additional parser rules in `health_parser.py`.

Workout UUIDs are not present in every Apple Health export. When missing, the app creates a stable internal workout ID so route matches remain separate.

## Project structure

```text
app.py             Streamlit UI, filtering, charts, map, and route profile
health_parser.py   Streaming workout parser, route parsers, and route matching
requirements.txt   Python dependencies
```

## Verification

Run a syntax check with:

```bash
python -m py_compile app.py health_parser.py
```
