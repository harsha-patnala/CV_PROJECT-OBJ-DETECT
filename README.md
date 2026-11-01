# Vehicle Intelligence

**AI-powered vehicle detection, tracking, and analytics.**

A Computer Vision web application built with Python and Streamlit, using Ultralytics YOLOv8 for vehicle detection and ByteTrack for multi-object tracking. Detection events are stored in a local SQLite database.

---

## Setup and launch

```bash
cd /path/to/CV_Project_New

# Virtual environment (already present in this folder as venv/)
# /opt/anaconda3/bin/python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

./venv/bin/streamlit run app.py
```

The app opens at `http://localhost:8501`.

Required: Python 3.11+ (tested with Python 3.13). A webcam is required only for Live Camera.

---

## Performance settings

Defaults are **Balanced**: YOLO input 640px, skip every other frame, capture width 960px.

| Setting | Fast | Balanced (default) | Higher Detail |
|---------|------|--------------------|---------------|
| Processing resolution (`imgsz`) | 480 | 640 | 800 |
| Frame skip | 2 | 1 | 0 |
| Max capture width | 640 | 960 | 1280 |

Additional controls in **Settings**:

- **Detection confidence** — default 0.35. Changing this does not reload the model.
- **Processing resolution** — YOLO input size; smaller is faster.
- **Frame skip** — `0` processes every frame. Skipped frames are not inferred; ByteTrack `persist=True` keeps IDs across gaps so unique counting still uses track IDs.
- **YOLO model** — keep `yolov8n.pt` (bundled). Larger models are slower and will download if not already local.

The YOLO model is loaded once via Streamlit `st.cache_resource` keyed on model path and `imgsz`. Colour estimation and optional OCR run once per unique track until a value is obtained.

Processing FPS is shown during video analysis and live camera. It is a measured pipeline rate, not a claim of real-time playback.

---

## Database

**Location:** `data/vehicle_intelligence.db` (created automatically; never wiped on startup).

**Schema (`detections`):**

| Column | Type | Notes |
|--------|------|--------|
| id | INTEGER PK | Internal row id — not shown as SI. No. |
| session_id | TEXT | One processing run (upload or camera) |
| track_id | INTEGER | Tracker ID within the session |
| vehicle_class | TEXT | car, motorcycle, bus, truck |
| confidence | REAL | Maximum observed confidence |
| est_colour | TEXT | Estimated colour, or NULL |
| plate_text | TEXT | OCR text, or NULL |
| wall_time | TEXT | Computer local date/time (`YYYY-MM-DD HH:MM:SS`) |
| video_timestamp | TEXT | Position in the video or camera run (`MM:SS.mmm`) |
| source | TEXT | `Live Cam` or `Up Vid` |
| crossed_line | INTEGER | Counting-line flag |
| line_cross_time | TEXT | Video timestamp of first crossing |

Unique constraint: `(session_id, track_id)`.

### Record-creation policy

One history row per unique tracked vehicle **within a processing session**.

- The first observation of a track ID is committed to SQLite immediately.
- Later frames update colour/plate/confidence on that same row; they do not add rows.
- The same physical vehicle in a **new** session (new upload or new camera start) can be recorded again.
- If tracking is off, a synthetic per-frame id is used — enable tracking for stable unique counts.
- Frame skip does not change this policy; only processed frames can create or update a row.

Interrupted processing leaves already-committed rows in the database.

---

## Detection History

Columns (this order, including CSV export):

1. SI. No. (display numbering, not the database id)
2. Vehicle Type
3. Colour (`Unknown` if estimation failed)
4. Vehicle Number (OCR text, or `Not detected`)
5. Date & Time (computer local time)
6. Source — exactly `Live Cam` or `Up Vid`

Newest records first. Search by plate, type, colour, source, or track id. Filter by type, colour, date range, and source. Paginated at 50 rows.

**Export CSV** downloads the filtered six-column table from saved records.

**Clear History** requires confirmation and is never run automatically.

---

## Number plate OCR

Optional. EasyOCR is **not** installed by default (`requirements.txt` comments it).

```bash
./venv/bin/pip install easyocr
```

Then enable **Number plate OCR** in Settings. First use downloads language weights (~100 MB).

Without EasyOCR, or when a plate cannot be read, Vehicle Number is **Not detected**. YOLO vehicle detection does not read plates. There is no dedicated plate-detector model in this project; OCR is applied to the lower part of the vehicle box and is unreliable.

---

## Camera permissions

Live Camera uses OpenCV device index `0` on the machine running Streamlit (not the browser webcam API).

On macOS: **System Settings → Privacy & Security → Camera** — allow the terminal (or Cursor) that launched Streamlit. Stop Camera and leaving the Live Camera page release the device.

---

## Current performance limitations

- Inference is typically CPU or Apple MPS; speed depends on hardware. Dummy-frame tests are in `tests/test_performance.py`.
- Streamlit replaces still frames; it does not play smooth video.
- Tracking IDs can switch on occlusion or re-entry.
- Colour estimation is approximate (K-Means on HSV).
- Plate OCR without a plate detector is not suitable for identification.

---

## Tests

```bash
./venv/bin/python -m pytest tests/ -v
```

Tests use temporary SQLite files and do not clear `data/vehicle_intelligence.db`.

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| `ModuleNotFoundError: ultralytics` | `./venv/bin/pip install -r requirements.txt` |
| Camera will not open | Grant macOS camera permission; confirm a device exists |
| Slow processing | Use Fast preset, raise frame skip, keep `yolov8n.pt` |
| `streamlit: command not found` | `./venv/bin/streamlit run app.py` |
