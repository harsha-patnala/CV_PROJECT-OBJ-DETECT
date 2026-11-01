# PROJECT_FEATURES.md — Vehicle Intelligence

Status after the in-place enhancement. See README for setup, schema, and the record-creation policy.

---

## Implemented and tested

| Feature | Status | Notes |
|---------|--------|-------|
| Application source / module imports | ✅ Tested | `tests/test_imports.py` |
| SQLite persistence | ✅ Tested | `data/vehicle_intelligence.db`; init, insert, reconnect |
| One row per (session_id, track_id) | ✅ Tested | Duplicate-frame prevention |
| Source labels `Live Cam` / `Up Vid` | ✅ Tested | Mapping + storage |
| Six-column history + CSV order | ✅ Tested | SI. No., Vehicle Type, Colour, Vehicle Number, Date & Time, Source |
| Search and filters | ✅ Tested | SQL filters; search by plate/type/colour/source |
| Clear history (explicit) | ✅ Tested | Confirmation in UI; `clear_all_records` in tests on temp DBs |
| Timestamps | ✅ Tested | Wall-clock vs video-relative |
| OCR missing-dependency handling | ✅ Tested | Returns None; UI shows Not detected |
| Video skip-frame pipeline | ✅ Tested | Fake detector; skip=1 halves inferences |
| Resource cleanup | ✅ Tested | VideoCapture released in `finally` |
| Colour / class recording | ✅ Tested | Session + DB |
| Landing: LIVE CAMERA / UPLOAD VIDEO | ✅ Implemented | Home screen; not covered by UI automation |
| Sidebar: Home, Live Camera, Upload Video, Detection History, Analytics, Settings | ✅ Implemented | Selected state uses Marigold accent |
| Palette Ice Silver / Deep Aubergine / Marigold | ✅ Implemented | CSS in `app.py` |
| Model cache (`st.cache_resource`) | ✅ Implemented | Keyed on model + imgsz, not confidence |
| Quality presets + resolution + skip | ✅ Implemented | Default Balanced |
| Analytics from DB | ✅ Implemented | Totals, class, colour, source, daily counts |
| Live camera / upload recording | ✅ Implemented | Source labels committed at first unique track |

---

## Partially implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Number plate OCR | ⚠️ Partial | Optional EasyOCR on lower vehicle crop; no plate detector |
| MPS acceleration | ⚠️ Partial | Auto-selected if `torch.backends.mps` is available |
| End-to-end camera on this Mac | ⚠️ Unverified here | Needs camera permission and a live session |
| End-to-end uploaded-video YOLO run | ⚠️ Partial | Pipeline tested with a fake detector; real YOLO timed on dummy frames |

---

## Performance (measured)

`tests/test_performance.py` runs a few dummy-frame detections with local `yolov8n.pt` on CPU at imgsz=480. Printed FPS is the measured value for that environment. It is not real-time video playback.

Pipeline changes intended to help speed (not independently A/B tested against the previous build):

- Load YOLO once; do not reload when confidence changes
- Do not re-run colour/OCR after a track already has a value
- Default frame skip = 1 (Balanced)
- Smaller Fast preset (480px, skip 2)
- Camera buffer size 1; UI refresh capped
- Tracker reset does not reload weights

---

## Requires additional setup

| Feature | Notes |
|---------|--------|
| Reliable plate reading | Install EasyOCR *and* a dedicated plate-detection model (not bundled) |
| CUDA | Install a CUDA PyTorch build; detector `device="0"` |
| Browser webcam | Not implemented; OpenCV device 0 only |

---

## Not implemented

Speed estimation, make/model recognition, multi-camera, annotated MP4 download, position heatmaps.

---

## Known limitations

- Tracking IDs can switch on occlusion.
- Colour names are approximate.
- Plate OCR is not identification-grade.
- Streamlit shows sequential stills, not a video player.
- Historical pytest runs that used `DetectionHistory.reset_all()` without a temp path could wipe the project database; tests now isolate temp files.
