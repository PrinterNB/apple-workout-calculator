# Apple Health Workout Explorer

A Streamlit app for exploring workouts and body measurements from an Apple Health export. It provides aggregate workout statistics, an individual workout inspector with route maps, elevation, speed, and heart-rate profiles, a health-metrics tab tracking body measurements over time, and a records tab of daily and all-time bests.

## Features

- Streams `export.xml` with `xml.etree.ElementTree.iterparse`, so the full export is not loaded into memory.
- Parses only the selected time frame: after choosing the export folder and a date range (default: Year to date), you press **Process data** and only workouts, routes, and measurements inside that range are parsed — a narrower range loads faster.
- Filters workouts by activity type; activity types are limited to those present in the selected date range.
- Includes All time, Year to date, Past year, and Custom date range presets.
- Summarizes accumulated hours, workout count, average duration, total distance in miles, and total energy in kcal.
- Treats the workout's direct `totalEnergyBurned` value as active calories when available, and derives total calories by adding basal energy when Apple provides it.
- Shows total miles, total calories, and active calories by workout type for the selected date range.
- Charts workout time by day, week, or month (default: week).
- Shows a workout-type duration breakdown ("Time Per Workout Type").
- Charts active calories per workout type for the filtered workouts.
- Shows the filtered workout table newest first.
- Uses the workout's original local time with standard AM/PM formatting.
- Parses `.gpx` and `.xml` files from an optional `workout-routes/` directory.
- Matches routes using workout UUIDs when available, then by route/workout start and end times.
- Displays route polylines with start/end markers on an Esri satellite basemap using Folium.
- Displays elevation and calculated speed in mph over timestamped route points.
- Displays heart rate over time when Apple Health heart-rate samples are available.
- Displays heart rate even when a workout has no matched GPS route.
- Tracks body measurements (weight, body fat percentage, height, resting heart rate, sleep duration, daily steps) plus daily move calories, total calories burned, exercise time, and stand hours, and derived BMI and lean body mass, on a dedicated Health Metrics tab, with toggleable per-metric chart layers that follow the sidebar time frame. Also charts daily walking + running distance in miles from the export's `DistanceWalkingRunning` records for the selected range. Each device (Watch, iPhone) writes its own samples for the same walking, so the per-day value is the single most complete source total rather than the sum of every record — which is why it matches the Health app's own walking/running numbers.

## Requirements

- Python 3.10 or newer recommended
- `lxml` is used for faster streaming XML parsing when installed; the app falls back to Python's standard XML parser if it is unavailable.
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

In the sidebar, enter the path to the extracted `apple_health_export/` folder — or any folder that contains it (for example the unzipped archive root), which the app walks up to three levels deep to locate `export.xml`. The app then works from the folder that actually holds `export.xml`:

- `export.xml`
- `workout-routes/`, when present

Then choose a **Time frame** in the sidebar (Year to date by default, or All time, Past year, or Custom with an explicit date range) and press **Process data**. Only workouts, routes, and health measurements inside that range are parsed, so a narrower range loads faster. Re-parse by pressing **Process data** again after changing the folder, the date range, or the export files themselves — the app shows a reminder when its settings no longer match what is loaded.

The sidebar also shows the total size of the selected export folder. If the route directory is absent, workout aggregates still work and route status is shown as `No Route Data`.

Activity-type filters and workout selection reuse the already-loaded data without reparsing the export or rematching routes.

The main `export.xml` is streamed with `iterparse` in a **single pass** (`parse_export_all` in `health_parser.py`) that collects workouts, heart-rate/energy samples, and health metrics together, so a large export is only read and XML-parsed once. The pass shows a live progress bar with an ETA while it runs. Route files are parsed in parallel when there are several files; if a Windows worker process fails, route parsing automatically falls back to sequential parsing.

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
- Accumulated time grouped by day, week, or month (default: week)
- Time grouped by workout type
- Active calories grouped by workout type
- Distance, total calories, and active calories grouped by workout type
- A newest-first filtered workout summary table

### Individual Workout Route Inspector

Choose a workout from the filtered set to view its local start/end time in AM/PM format, duration, distance, total calories, active calories, average heart rate, workout type, and route status. When a route is matched, the tab also shows:

- Interactive satellite GPS map with the complete route
- Route time-window slider expressed in decimal minutes after the workout start
- Green start marker and red end marker
- Elevation profile
- Calculated speed profile in miles per hour when trackpoint timestamps are available
- Heart-rate profile in beats per minute when heart-rate samples are available

If no GPS route is matched, the profile section can still display heart rate by itself.

### Health Metrics

Shows the latest available value for each tracked metric as rows of stat tiles (day-based metrics — steps, walking + running distance, sleep, resting heart rate, move calories, exercise, and stand — show the average of the last seven complete days rather than today's still-in-progress number; a toggle switches them to the average over the whole selected time period, with the tile labels relabeled `(… 7-day avg)` to `(… range avg)` to match. Both averages are precomputed once when **Process data** runs, so flipping the toggle is instant and never re-parses the export), then totals for the selected time range (walk + run distance in miles, steps, sleep hours, move calories, exercise minutes, stand hours), then a time-series chart below it. Body measurement records are parsed once during the initial export load, alongside workouts and routes, so opening or re-rendering the tab never triggers another pass over `export.xml`. Each metric is its own toggleable layer (a multiselect of check/uncheck options), and the chart respects the sidebar time frame. A **Trend lines** checkbox draws a smoothed trend (centered 15-day rolling average) as the main line for every displayed layer, with the raw measurements faded in the background so the overall direction stays visible through noisy data (sleep, resting heart rate). Metrics:

- **Weight** — from `bodymass` records (grams, kg, lb, and oz are all normalized)
- **Body Fat (%)** — from `bodyfatpercentage` records
- **BMI** — derived from weight and height (Apple's `bodymassindex` records are ignored in favor of the derived value)
- **Lean Body Mass** — derived as weight × (1 − body fat %)
- **Height** — from `height` records (cm, in, ft, and m are all normalized); the chart axis and hover render in feet/inches
- **Resting Heart Rate** — from `restingheart-rate` records
- **Sleep Duration** — asleep time from `sleepanalysis` records summed per calendar day (`inBed` time without an asleep value is excluded)
- **Steps (day)** — `stepcount` records summed per calendar day (a sample that spans midnight is split across the days it covers)
- **Walking + Running Distance (mi)** — `distancewalkingrunning` records per calendar day, in miles. Apple's Watch and iPhone both write samples of the same walking, so the day's value is the largest single-source total, which matches the Health app's own walking/running figure.
- **Move Calories (kcal)** — `activeenergyburned` samples summed per calendar day (samples one day past the range exist only for workout backfill and are included so the daily total stays inside the selected range).
- **Exercise (min)** — `appleexercisetime` records per calendar day; the day's value is the largest single-source total rather than a sum of every record.
- **Total Calories Burned (kcal)** — `activeenergyburned` (move) + `basalenergyburned` (resting) per calendar day; each device's two streams add together, then the best single device's combined total wins the day
- **Stand (h)** — the blue ring: one `applestandhour` record per stood hour, so the daily total is the count of such records; likewise the largest single-source total.

Weight, body fat, height, and resting heart rate are carried forward to later days so the lines stay continuous between measurements; sleep, steps, walking + running distance, move calories, exercise, and stand are only plotted on days with data. New measurements can be added later by collecting them in `parse_health_metrics` in `health_parser.py` and registering a layer in `METRIC_LAYERS` in `app.py`.

### Records

A trophy shelf of daily and all-time records, computed from the current date range and activity-type filters and shown as rows of stat tiles (each tile carries the date the record was set, plus the workout type(s) involved — e.g. `Jan 02, 2026 (Running)` for a single workout, or all the types done that day / across the streak for daily and streak records). Sections:

- **Workout Records (best single workout)** — longest duration, farthest distance, fastest pace, most total calories, most active calories, highest average heart rate
- **Streaks (consecutive days)** — longest run of consecutive days with a workout, and longest run of consecutive days with steps
- **Most in a Day (workouts)** — most workouts, hours, distance, and calories in a single day
- **Most in a Day (health)** — most total steps, total walk + run distance, exercise minutes, move calories, and total calories burned in a single day, plus most sleep and most stand hours
- **Best Body Measurements (lowest)** — lowest weight, lowest body fat, best (lowest) BMI, and lowest resting heart rate

Records follow the same filtering as the other tabs, so narrowing the sidebar (date range or activity types) re-scopes them. New records can be added in the `build_*_records` functions in `app.py`.

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
