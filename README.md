# Apple Health Workout Explorer

A Streamlit app for exploring workouts from an Apple Health export. It provides aggregate workout statistics and an individual workout route inspector with an interactive map and elevation/speed profile.

## Features

- Streams `export.xml` with `xml.etree.ElementTree.iterparse`, so the full export is not loaded into memory.
- Filters workouts by date range and activity type; activity types are limited to those present in the selected date range.
- Summarizes accumulated hours, workout count, average duration, total distance in miles, and total energy in kcal.
- Charts workout time by day, week, or month.
- Shows a workout-type duration breakdown.
- Parses `.gpx` and `.xml` files from an optional `workout-routes/` directory.
- Matches routes using workout UUIDs when available, then by route/workout start and end times.
- Displays route polylines with start/end markers on an Esri satellite basemap using Folium.
- Displays elevation and calculated speed over timestamped route points.

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

The app automatically reloads data when the folder or its files change. If the route directory is absent, workout aggregates still work and route status is shown as `No Route Data`.

## Apple Health export

On an iPhone, open the Health app, select your profile, choose **Export All Health Data**, and extract the resulting archive. Enter the extracted `apple_health_export/` folder path in the app.

## App tabs

### Workout Accumulator

The accumulator tab applies the sidebar filters and shows:

- Total accumulated workout hours
- Number of workouts
- Average workout duration
- Accumulated time grouped by day, week, or month
- Duration grouped by workout type
- A filtered, interactive workout summary table

### Individual Workout Route Inspector

Choose a workout from the filtered set to view its start/end time, duration, distance, energy burned, workout type, and route status. When a route is matched, the tab also shows:

- Interactive satellite GPS map with the complete route
- Green start marker and red end marker
- Elevation profile
- Calculated speed profile when trackpoint timestamps are available

## Route matching

Route matching is attempted in this order:

1. Search route metadata, filenames, and raw route contents for the workout UUID.
2. Compare route start/end timestamps with the workout start/end timestamps.

The timestamp fallback accepts the closest route when the combined start/end difference is within the workout duration-based tolerance, with a maximum practical tolerance of four hours. Route files without usable trackpoints are ignored.

GPX files are parsed with `gpxpy`. XML route files are parsed generically by looking for latitude/longitude attributes and common elevation and timestamp fields. Apple Health XML variants that use a different structure may require additional parser rules in `health_parser.py`.

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
