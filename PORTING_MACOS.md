# macOS port notes

Branch: `macos-port` (from `main`). Target: macOS 27.x, arm64, CPython 3.13. Windows compatibility preserved.

## Changes made

1. **Line endings normalized to LF** (`app.py`, `health_parser.py`) plus a new `.gitattributes` (`*.py text eol=lf`, `*.gpx text eol=lf`).
   *Why:* the Windows-developed checkout shipped CRLF everywhere. Python treats both as inert, but LF is the first-class macOS convention and Windows tooling reads LF sources without issue. The `.gitattributes` pins the choice across platforms so future checkouts on either OS stay consistent.

2. **`app.py:1340` comment reworded** — the tkinter "Select Folder" dialog comment claimed it was a Windows file browser; it works unchanged on macOS Tk (verified: `import tkinter` + `askdirectory` succeed on this machine's CPython 3.13). Comment-only fix, no behavior change.

3. **`app.py` folder-picker fallback kept as-is** (`choose_export_folder`, ~line 1362): the PowerShell `FolderBrowserDialog` fallback stays guarded behind `if os.name == "nt":`, so it never executes on macOS. On macOS the flow is tkinter → `return None` → sidebar `st.text_input` path (app.py:1393). Deliberately **not deleted** so Windows keeps working; macOS without `_tkcltk` (e.g. Homebrew Python) degrades to typed paths instead of crashing.

4. **README.md:57 corrected** — it documented a "Windows worker process fails → sequential fallback" that no longer exists in code. Verified by audit + reading: route parsing uses `concurrent.futures.ThreadPoolExecutor` (health_parser.py:1912) with a top-level worker (`parse_route_file`) receiving only picklable `Path` args, no `multiprocessing`/`ProcessPoolExecutor` anywhere, hence no spawn-start-method, pickling, or `__main__`-guard issues on macOS; `os.cpu_count()` is valid on arm64. The README now describes the real thread-based behavior.

5. **`.gitignore`**: added `.venv-macos/` so the port's virtual environment never enters the commit set.

6. **`requirements.txt` unchanged** — empirically verified installable on macOS arm64 / Python 3.13 with zero errors (no version pins needed): streamlit 1.67.0, pandas 3.1.1, plotly 7.1.0, folium 0.20.0, gpxpy 1.6.2, lxml 6.1.3 (all arm64 wheels). All third-party imports in `app.py`/`health_parser.py` are covered by these six lines; everything else is stdlib.

7. **Test fixture added**: `testdata/apple_health_export/` — synthetic Apple export (`export.xml`: 5 workouts with `startDate`/`endDate`/`durationUnit`/`totalDistance`/`totalEnergyBurned`, 2 StepCount, 2 BodyMass, 2 SleepAnalysis records, attribute names taken from the parser's own readers) and `workout-routes/` with 4 GPX files named `route_YYYY-MM-DD_H.HHam` (the format `health_parser._extract_route_time_from_filename` expects). 4 routes exercise the `>= 4` `ThreadPoolExecutor` branch and match their workouts by timestamp.

## Verification (all on this Mac, arm64, Python 3.13.0)

| Step | Result |
|---|---|
| `pip install -r requirements.txt` in `.venv-macos` | PASS — exit 0, clean log |
| `python -m py_compile app.py health_parser.py` | PASS |
| Direct-parse test (`parse_export_all`, `parse_route_directory`, `match_route_to_workout` on fixture) | PASS — 5 workouts, record counts 2/2/2, 4/4 routes parsed + matched (swim intentionally has no route) |
| `streamlit run app.py --server.headless true` + `curl http://localhost:8501` | PASS — HTTP 200, clean server log; server killed afterwards |

Note: `python app.py` (bare, outside the Streamlit runtime) is **not** a valid smoke test — `st.stop()` is a no-op there, so execution falls through into empty-frame branches that a real session never reaches. The headless-server check above is the correct verification.

## Assumptions

- No real `apple_health_export/` was found under `~/Downloads` or `~/Documents` (searched 2026-11-07), so the synthetic fixture stands in for real data; it is shaped against the parser's own attribute readers rather than guessed from memory.
- The fixture uses the modern export style (`Record type="HKQuantityTypeIdentifier…"`, `startDate`, `value` attributes) because both parse paths in `health_parser.py` read that style.
- The `Mac` branch was identical to `main` (same commit); the port work lives on `macos-port`.
- tkinter is present in this machine's CPython 3.13 build; if a user's macOS Python lacks Tk, the folder picker returns `None` and the sidebar text-input path takes over (already built into the app).

## Running fresh on macOS

```bash
git checkout macos-port
python3 -m venv .venv-macos
source .venv-macos/bin/activate
pip install -r requirements.txt
streamlit run app.py            # or: streamlit run app.py --server.headless true
```

Point the app at `testdata/apple_health_export/` to see the fixture render end-to-end.
