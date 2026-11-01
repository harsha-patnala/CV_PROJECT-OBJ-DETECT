"""
app.py — Vehicle Intelligence: Main Streamlit Application.

Colour palette:
  Ice Silver   #DDE3E8  (dominant background)
  Deep Aubergine #3A0F34 (text, headings, primary elements)
  Marigold     #EAAA00  (accents, highlights)
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Page configuration (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Vehicle Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal imports ─────────────────────────────────────────────────────────
from analytics import (
    DetectionHistory,
    SessionAnalytics,
    build_history_display,
    HISTORY_COLUMNS,
)
from database import init_db, DB_PATH
from detector import VehicleDetector, QUALITY_PRESETS, preset_config
from utils import (
    bgr_to_rgb,
    cleanup_temp_file,
    draw_counting_line,
    draw_detection,
    frame_to_timestamp,
    get_box_colour,
    get_video_info,
    resize_frame,
    save_temp_video,
)
from vehicle_attributes import estimate_vehicle_colour
from video_processor import ProcessingConfig, VideoProcessor

logger = logging.getLogger("vehicle_intelligence.app")

# Ensure DB exists on startup
try:
    init_db()
except Exception as _e:
    logger.warning(f"DB init warning: {_e}")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS / DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "ice_silver":      "#DDE3E8",
    "aubergine":       "#3A0F34",
    "marigold":        "#EAAA00",
    "white":           "#FFFFFF",
    "surface":         "#F4F6F8",
    "border":          "#C8CDD3",
    "text_secondary":  "#5A6472",
    "success":         "#2E7D32",
    "error":           "#C62828",
}


def inject_css() -> None:
    st.markdown(f"""
    <style>
    /* ── Root / Body ── */
    html, body, [data-testid="stApp"] {{
        background-color: {PALETTE['ice_silver']};
        color: {PALETTE['aubergine']};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    /* ── Remove default Streamlit chrome ── */
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: {PALETTE['aubergine']};
        min-width: 200px !important;
        max-width: 220px !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        color: {PALETTE['ice_silver']} !important;
        border: none !important;
        text-align: left !important;
        padding: 0.4rem 0.6rem !important;
        font-size: 0.88rem !important;
        width: 100%;
        border-radius: 4px;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(234,170,0,0.15) !important;
        color: {PALETTE['marigold']} !important;
    }}
    .nav-active {{
        background: rgba(234,170,0,0.18);
        color: {PALETTE['marigold']};
        border-left: 3px solid {PALETTE['marigold']};
        padding: 0.4rem 0.6rem;
        font-size: 0.88rem;
        border-radius: 4px;
        margin: 0.15rem 0;
        font-weight: 600;
    }}

    /* ── App header ── */
    .vi-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 1rem;
        background: {PALETTE['aubergine']};
        border-radius: 6px;
        margin-bottom: 1.2rem;
    }}
    .vi-header-brand {{
        color: {PALETTE['ice_silver']};
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }}
    .vi-header-sub {{
        color: {PALETTE['marigold']};
        font-size: 0.72rem;
        letter-spacing: 0.06em;
    }}

    /* ── Page title block ── */
    .vi-title {{
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    .vi-title h1 {{
        color: {PALETTE['aubergine']};
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0 0 0.15rem 0;
    }}
    .vi-title p {{
        color: {PALETTE['text_secondary']};
        font-size: 0.88rem;
        margin: 0;
    }}

    /* ── Primary action cards ── */
    .vi-card {{
        background: {PALETTE['white']};
        border: 1.5px solid {PALETTE['border']};
        border-radius: 8px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.15s, box-shadow 0.15s;
    }}
    .vi-card:hover {{
        border-color: {PALETTE['aubergine']};
        box-shadow: 0 2px 8px rgba(58,15,52,0.10);
    }}
    .vi-card-icon {{
        font-size: 2.0rem;
        margin-bottom: 0.5rem;
    }}
    .vi-card-title {{
        color: {PALETTE['aubergine']};
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }}
    .vi-card-desc {{
        color: {PALETTE['text_secondary']};
        font-size: 0.82rem;
    }}

    /* ── Stat pills ── */
    .vi-stat-row {{
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 0.8rem;
    }}
    .vi-stat {{
        background: {PALETTE['white']};
        border: 1px solid {PALETTE['border']};
        border-radius: 6px;
        padding: 0.4rem 0.85rem;
        font-size: 0.82rem;
        color: {PALETTE['aubergine']};
        font-weight: 500;
    }}
    .vi-stat span {{
        color: {PALETTE['marigold']};
        font-weight: 700;
    }}

    /* ── Section divider ── */
    .vi-divider {{
        border: none;
        border-top: 1px solid {PALETTE['border']};
        margin: 1rem 0;
    }}

    /* ── Accent badge ── */
    .vi-badge {{
        display: inline-block;
        background: {PALETTE['marigold']};
        color: {PALETTE['aubergine']};
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        letter-spacing: 0.04em;
    }}

    /* ── Secondary nav tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: transparent;
        border-bottom: 2px solid {PALETTE['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {PALETTE['text_secondary']};
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.4rem 0.9rem;
        border: none;
        background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {PALETTE['aubergine']};
        border-bottom: 2px solid {PALETTE['aubergine']};
        font-weight: 600;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background-color: {PALETTE['aubergine']};
        color: {PALETTE['ice_silver']};
        border: none;
        border-radius: 5px;
        font-size: 0.88rem;
        font-weight: 500;
        padding: 0.45rem 1.1rem;
    }}
    .stButton > button:hover {{
        background-color: #550f4a;
        color: {PALETTE['white']};
    }}

    /* ── Progress bar tint ── */
    .stProgress > div > div > div {{
        background-color: {PALETTE['aubergine']};
    }}

    /* ── Tables ── */
    .stDataFrame {{ border-radius: 6px; overflow: hidden; }}

    /* ── Info / warning boxes ── */
    .stAlert {{ border-radius: 6px; }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    defaults = {
        "view":             "home",
        "detector":         None,
        "history":          DetectionHistory(),
        "current_session":  None,
        "model_name":       "yolov8n.pt",
        "confidence":       0.35,
        "quality_preset":   "Balanced",
        "imgsz":            640,
        "frame_skip":       1,
        "max_width":        960,
        "use_tracking":     True,
        "estimate_colour":  True,
        "run_plate_ocr":    False,
        "use_counting_line": False,
        "counting_line_y":  0.5,
        "camera_running":   False,
        "processing_active": False,
        "confirm_clear":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_detector(model_name: str, imgsz: int) -> VehicleDetector:
    """Load YOLO once per (model, resolution). Confidence is applied per session."""
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass
    return VehicleDetector(model_path=model_name, confidence=0.35, device=device, imgsz=imgsz)


def get_active_detector() -> VehicleDetector:
    cfg = preset_config(st.session_state.get("quality_preset", "Balanced"))
    imgsz = int(st.session_state.get("imgsz", cfg["imgsz"]))
    detector = load_detector(st.session_state.model_name, imgsz)
    detector.confidence = float(st.session_state.confidence)
    return detector


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.8rem 0.4rem 1rem 0.4rem;">
            <div style="color: #EAAA00; font-size: 1.1rem; font-weight: 700; letter-spacing: 0.05em;">🚗 Vehicle Intelligence</div>
            <div style="color: rgba(221,227,232,0.6); font-size: 0.65rem; letter-spacing: 0.08em; margin-top:2px;">COMPUTER VISION ANALYTICS</div>
        </div>
        """, unsafe_allow_html=True)

        current = st.session_state.get("view", "home")
        nav_items = [
            ("home",      "🏠  Home"),
            ("camera",    "📷  Live Camera"),
            ("upload",    "📁  Upload Video"),
            ("history",   "🗂  Detection History"),
            ("analytics", "📊  Analytics"),
            ("settings",  "⚙️  Settings"),
        ]
        for view_key, label in nav_items:
            if view_key == current:
                st.markdown(f'<div class="nav-active">{label}</div>', unsafe_allow_html=True)
                continue
            if st.button(label, key=f"nav_{view_key}", use_container_width=True):
                if view_key != "camera":
                    st.session_state.camera_running = False
                if view_key != "upload":
                    st.session_state.processing_active = False
                st.session_state.view = view_key
                st.rerun()

        st.markdown(
            '<hr style="border-color: rgba(221,227,232,0.15); margin: 0.8rem 0;">',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="color: rgba(221,227,232,0.4); font-size: 0.62rem; padding: 0 0.2rem;">'
            f'DB: data/vehicle_intelligence.db</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    st.markdown("""
    <div class="vi-header">
        <div>
            <div class="vi-header-brand">🚗 Vehicle Intelligence</div>
            <div class="vi-header-sub">COMPUTER VISION ANALYTICS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME / LANDING VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_home() -> None:
    st.markdown("""
    <div class="vi-title">
        <h1>Vehicle Intelligence</h1>
        <p>Real-time vehicle detection and video analytics.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Primary action cards ──────────────────────────────────────────────────
    col_cam, col_upload = st.columns(2, gap="large")

    with col_cam:
        st.markdown("""
        <div class="vi-card">
            <div class="vi-card-icon">📷</div>
            <div class="vi-card-title">LIVE CAMERA</div>
            <div class="vi-card-desc">Analyse live video from your connected camera with vehicle detection and tracking.</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Open Live Camera", key="btn_camera", use_container_width=True):
            st.session_state.view = "camera"
            st.rerun()

    with col_upload:
        st.markdown("""
        <div class="vi-card">
            <div class="vi-card-icon">📁</div>
            <div class="vi-card-title">UPLOAD VIDEO</div>
            <div class="vi-card-desc">Upload an MP4 or MOV file for vehicle detection, counting, and colour estimation.</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Upload Video", key="btn_upload", use_container_width=True):
            st.session_state.view = "upload"
            st.rerun()

    st.markdown('<hr class="vi-divider">', unsafe_allow_html=True)

    try:
        from database import get_summary_stats
        stats = get_summary_stats()
        if stats["total"] > 0:
            st.caption(f"{stats['total']} recorded vehicle event(s) — see Detection History in the sidebar.")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS VIEW / PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_settings_view() -> None:
    st.markdown("### ⚙️ Settings")
    render_settings_panel()
    st.markdown('<hr class="vi-divider">', unsafe_allow_html=True)
    st.markdown(f"**Database location:** `{DB_PATH}`")
    st.markdown("Records persist across app restarts. Use **Detection History → Clear History** to remove records.")


def render_settings_panel() -> None:
    col1, col2 = st.columns(2)
    with col1:
        preset = st.selectbox(
            "Detection quality",
            list(QUALITY_PRESETS.keys()),
            index=list(QUALITY_PRESETS.keys()).index(
                st.session_state.get("quality_preset", "Balanced")
            ),
            help="Fast skips more frames at 480px. Balanced is the default. Higher Detail uses 800px.",
        )
        if preset != st.session_state.get("quality_preset"):
            cfg = preset_config(preset)
            st.session_state.imgsz = cfg["imgsz"]
            st.session_state.frame_skip = cfg["skip_frames"]
            st.session_state.max_width = cfg["max_width"]
        st.session_state.quality_preset = preset
        cfg = preset_config(preset)

        model = st.selectbox(
            "YOLO model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"].index(
                st.session_state.model_name
            ),
            help="Keep yolov8n.pt unless you already have a larger local weight file. Larger models are slower.",
        )
        st.session_state.model_name = model

        conf = st.slider(
            "Detection confidence",
            0.10, 0.90, float(st.session_state.confidence), 0.05,
        )
        st.session_state.confidence = conf

        st.session_state.imgsz = st.select_slider(
            "Processing resolution",
            options=[480, 640, 800, 960, 1280],
            value=int(st.session_state.get("imgsz", cfg["imgsz"])),
            help="YOLO input size. Smaller is faster.",
        )

    with col2:
        st.session_state.frame_skip = st.slider(
            "Frame skip",
            0, 5, int(st.session_state.get("frame_skip", cfg["skip_frames"])),
            help="0 = every frame. Higher values speed processing; tracking IDs persist across skipped frames.",
        )
        st.session_state.use_tracking = st.checkbox(
            "Enable tracking (ByteTrack)",
            value=st.session_state.use_tracking,
        )
        st.session_state.estimate_colour = st.checkbox(
            "Estimate vehicle colour",
            value=st.session_state.estimate_colour,
        )
        st.session_state.run_plate_ocr = st.checkbox(
            "Number plate OCR (slow — requires EasyOCR)",
            value=st.session_state.run_plate_ocr,
        )
        st.session_state.use_counting_line = st.checkbox(
            "Enable counting line",
            value=st.session_state.use_counting_line,
        )
        if st.session_state.use_counting_line:
            st.session_state.counting_line_y = st.slider(
                "Line position (fraction of height)",
                0.1, 0.9, st.session_state.counting_line_y, 0.05,
            )


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD VIDEO VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_upload_view() -> None:
    st.markdown("### 📁 Upload & Analyse Video")

    uploaded = st.file_uploader(
        "Select a video file (MP4, MOV, AVI, MKV)",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload",
    )

    if uploaded is None:
        st.info("Upload a video file above to begin analysis.", icon="ℹ️")
        with st.expander("⚙️ Analysis Settings", expanded=False):
            render_settings_panel()
        return

    st.markdown(f"**Selected:** `{uploaded.name}` · {uploaded.size / 1_048_576:.1f} MB")

    # Write to temp file once per unique filename
    if "upload_tmp_path" not in st.session_state or st.session_state.get("upload_filename") != uploaded.name:
        if "upload_tmp_path" in st.session_state:
            cleanup_temp_file(st.session_state.upload_tmp_path)
        tmp_path = save_temp_video(uploaded.read(), suffix=Path(uploaded.name).suffix)
        st.session_state.upload_tmp_path = tmp_path
        st.session_state.upload_filename = uploaded.name

    tmp_path = st.session_state.upload_tmp_path

    # Video info
    info = get_video_info(tmp_path)
    if info:
        info_cols = st.columns(4)
        info_cols[0].metric("Width", f"{info['width']}px")
        info_cols[1].metric("Height", f"{info['height']}px")
        info_cols[2].metric("FPS", f"{info['fps']:.1f}")
        info_cols[3].metric("Duration", f"{info['duration_s']:.1f}s")

    with st.expander("⚙️ Analysis Settings", expanded=False):
        render_settings_panel()

    if not st.session_state.processing_active:
        if st.button("▶ Start Analysis", key="start_analysis", use_container_width=True):
            st.session_state.processing_active = True
            st.rerun()
    else:
        if st.button("⏹ Stop Analysis", key="stop_analysis", use_container_width=True):
            st.session_state.processing_active = False
            st.rerun()
        _run_video_analysis(tmp_path)


def _run_video_analysis(video_path: str) -> None:
    """Execute the video processing pipeline and render results."""
    cfg = preset_config(st.session_state.get("quality_preset", "Balanced"))
    try:
        detector = get_active_detector()
    except RuntimeError as e:
        st.error(f"Model loading failed: {e}")
        st.session_state.processing_active = False
        return

    config = ProcessingConfig(
        confidence=st.session_state.confidence,
        use_tracking=st.session_state.use_tracking,
        estimate_colour=st.session_state.estimate_colour,
        run_plate_ocr=st.session_state.run_plate_ocr,
        use_counting_line=st.session_state.use_counting_line,
        counting_line_y_frac=st.session_state.counting_line_y,
        skip_frames=int(st.session_state.get("frame_skip", cfg["skip_frames"])),
        max_width=int(st.session_state.get("max_width", cfg["max_width"])),
    )

    processor = VideoProcessor(detector, config)

    progress_bar = st.progress(0, text="Initialising…")
    frame_display = st.empty()
    stats_placeholder = st.empty()

    last_result = None
    frame_count = 0
    start_time = time.time()

    try:
        for result in processor.process(video_path):
            if not st.session_state.processing_active:
                break

            last_result = result
            frame_count += 1

            if result.total_frames > 0:
                pct = result.frame_index / result.total_frames
                elapsed = time.time() - start_time
                fps_proc = frame_count / elapsed if elapsed > 0 else 0
                progress_bar.progress(
                    min(pct, 1.0),
                    text=f"Frame {result.frame_index}/{result.total_frames} · "
                         f"{result.processing_fps:.1f} proc fps · "
                         f"{result.video_timestamp}",
                )

            # Update display every 3rd rendered frame for performance
            if frame_count % 3 == 0:
                frame_display.image(
                    result.frame_rgb,
                    caption=f"Frame {result.frame_index} · {result.video_timestamp}",
                    use_container_width=True,
                )
                stats_placeholder.markdown(
                    _format_stats_html(result),
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        logger.exception("Video processing error")
        st.error(f"Processing error: {exc}")

    # Register session in history (records already in DB)
    if processor.session and processor.session.unique_vehicles > 0:
        st.session_state.history.add_session(processor.session)
        st.session_state.current_session = processor.session

    progress_bar.progress(1.0, text="✓ Analysis complete")
    st.session_state.processing_active = False

    if last_result:
        frame_display.image(last_result.frame_rgb, use_container_width=True)
        stats_placeholder.markdown(
            _format_stats_html(last_result),
            unsafe_allow_html=True,
        )

    _render_session_results(processor.session)


def _format_stats_html(result) -> str:
    parts = [
        '<div class="vi-stat-row">',
        f'<div class="vi-stat">Unique vehicles: <span>{result.unique_vehicles}</span></div>',
    ]
    for cls, count in sorted(result.class_counts.items()):
        parts.append(f'<div class="vi-stat">{cls.title()}: <span>{count}</span></div>')
    if result.line_crossings > 0:
        parts.append(
            f'<div class="vi-stat">Line crossings: <span>{result.line_crossings}</span></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _render_session_results(session: Optional[SessionAnalytics]) -> None:
    if session is None or session.unique_vehicles == 0:
        st.info("No vehicles detected in this video.")
        return

    st.markdown('<hr class="vi-divider">', unsafe_allow_html=True)
    st.markdown("#### Session Results")

    counts = session.class_counts()
    stat_cols = st.columns(len(counts) + 1)
    stat_cols[0].metric("Unique Vehicles", session.unique_vehicles)
    for i, (cls, cnt) in enumerate(sorted(counts.items()), start=1):
        stat_cols[i].metric(cls.title(), cnt)

    df = session.records_dataframe()
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_data = session.to_csv()
        st.download_button(
            "⬇ Export Session CSV",
            data=csv_data,
            file_name=f"vehicle_session_{session.session_id}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# LIVE CAMERA VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_camera_view() -> None:
    st.markdown("### 📷 Live Camera Detection")

    st.info(
        "**Camera Access:** This mode uses OpenCV to capture frames from the system camera "
        "(device index 0). Works when running **locally**. "
        "Grant camera permissions to your terminal if prompted by macOS.",
        icon="ℹ️",
    )

    with st.expander("⚙️ Detection Settings", expanded=False):
        render_settings_panel()

    col_start, col_stop = st.columns(2)
    with col_start:
        start_pressed = st.button(
            "▶ Start Camera", key="cam_start",
            disabled=st.session_state.camera_running,
            use_container_width=True,
        )
    with col_stop:
        stop_pressed = st.button(
            "⏹ Stop Camera", key="cam_stop",
            disabled=not st.session_state.camera_running,
            use_container_width=True,
        )

    if stop_pressed:
        st.session_state.camera_running = False
        st.rerun()

    if start_pressed:
        st.session_state.camera_running = True

    if st.session_state.camera_running:
        _run_camera_capture()


def _run_camera_capture() -> None:
    """Capture frames from webcam and run detection."""
    cfg = preset_config(st.session_state.get("quality_preset", "Balanced"))
    try:
        detector = get_active_detector()
    except RuntimeError as e:
        st.error(f"Model loading failed: {e}")
        st.session_state.camera_running = False
        return

    cap = cv2.VideoCapture(0)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        st.error(
            "Could not open camera (device 0). "
            "Check that a camera is connected and that macOS has granted camera "
            "permission to the terminal / app running this server."
        )
        st.session_state.camera_running = False
        return

    detector.reset_tracker()
    session_id = f"camera_{uuid.uuid4().hex[:8]}"
    session = SessionAnalytics(session_id=session_id, source="camera")
    st.session_state.current_session = session

    frame_display = st.empty()
    stats_placeholder = st.empty()

    frame_idx = 0
    processed = 0
    start_time = time.time()
    last_display_time = 0.0
    DISPLAY_INTERVAL = 0.12
    skip = int(st.session_state.get("frame_skip", cfg["skip_frames"]))
    max_width = int(st.session_state.get("max_width", cfg["max_width"]))

    try:
        while st.session_state.camera_running:
            ok, frame = cap.read()
            if not ok:
                st.warning("Camera frame read failed. Stopping.")
                break

            frame_idx += 1
            if skip > 0 and (frame_idx - 1) % (skip + 1) != 0:
                continue

            frame = resize_frame(frame, max_width)
            processed += 1
            elapsed = time.time() - start_time
            video_ts = (
                f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                f".{int((elapsed % 1)*1000):03d}"
            )
            proc_fps = processed / elapsed if elapsed > 0 else 0.0

            if st.session_state.use_tracking:
                detections = detector.track(frame)
            else:
                detections = detector.detect(frame)

            annotated = frame.copy()
            for det in detections:
                track_id = det.track_id if det.track_id is not None else frame_idx * 100
                abs_id = abs(track_id)
                colour_box = get_box_colour(abs_id)

                est_colour = session.colour_for(abs_id)
                if st.session_state.estimate_colour and not est_colour:
                    est_colour = estimate_vehicle_colour(
                        frame, det.x1, det.y1, det.x2, det.y2
                    )

                plate_text = session.plate_for(abs_id)
                if st.session_state.run_plate_ocr and not plate_text:
                    from plate_recognition import attempt_plate_ocr
                    plate_text = attempt_plate_ocr(
                        frame, det.x1, det.y1, det.x2, det.y2
                    )

                session.record_detection(
                    track_id=abs_id,
                    vehicle_class=det.class_name,
                    confidence=det.confidence,
                    video_timestamp=video_ts,
                    est_colour=est_colour,
                    plate_text=plate_text,
                )

                label_parts = [det.class_name]
                if det.track_id is not None:
                    label_parts.append(f"#{det.track_id}")
                label_parts.append(f"{det.confidence:.2f}")
                if est_colour:
                    label_parts.append(est_colour)

                draw_detection(
                    annotated,
                    det.x1, det.y1, det.x2, det.y2,
                    " ".join(label_parts),
                    colour=colour_box,
                )

            if st.session_state.use_counting_line:
                line_y = int(frame.shape[0] * st.session_state.counting_line_y)
                draw_counting_line(annotated, line_y)

            now = time.time()
            if now - last_display_time >= DISPLAY_INTERVAL:
                frame_display.image(
                    bgr_to_rgb(annotated),
                    caption=f"Live Cam · {video_ts} · {proc_fps:.1f} proc fps · {session.unique_vehicles} unique",
                    use_container_width=True,
                )

                counts = session.class_counts()
                parts = [
                    '<div class="vi-stat-row">',
                    f'<div class="vi-stat">Unique: <span>{session.unique_vehicles}</span></div>',
                    f'<div class="vi-stat">Proc FPS: <span>{proc_fps:.1f}</span></div>',
                ]
                for cls, cnt in sorted(counts.items()):
                    parts.append(f'<div class="vi-stat">{cls.title()}: <span>{cnt}</span></div>')
                parts.append("</div>")
                stats_placeholder.markdown("".join(parts), unsafe_allow_html=True)
                last_display_time = now

    finally:
        cap.release()
        if session.unique_vehicles > 0:
            st.session_state.history.add_session(session)
        st.session_state.camera_running = False
        st.success("Camera stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_analytics_view() -> None:
    st.markdown("### 📊 Analytics")

    history: DetectionHistory = st.session_state.history
    stats = history.get_summary_stats()

    if stats["total"] == 0:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color: #5A6472;">
            <div style="font-size:2rem;">📭</div>
            <div style="margin-top:0.5rem;">No analysis sessions yet.</div>
            <div style="font-size:0.82rem; margin-top:0.3rem;">
                Process a video or run a camera session to see results here.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    by_class = stats.get("by_class", {})
    by_source = stats.get("by_source", {})

    n_metric_cols = min(len(by_class) + 3, 7)
    m_cols = st.columns(n_metric_cols)
    m_cols[0].metric("Total Events", stats["total"])
    m_cols[1].metric("Live Cam", by_source.get("Live Cam", 0))
    m_cols[2].metric("Up Vid", by_source.get("Up Vid", 0))
    for i, (cls, cnt) in enumerate(sorted(by_class.items())[:4], start=3):
        if i < n_metric_cols:
            m_cols[i].metric(cls.title(), cnt)

    st.markdown('<hr class="vi-divider">', unsafe_allow_html=True)

    # ── Class distribution chart ──────────────────────────────────────────────
    if by_class:
        st.markdown("**Vehicle Class Distribution**")
        chart_df = pd.DataFrame(
            {"Class": list(by_class.keys()), "Count": list(by_class.values())}
        ).sort_values("Count", ascending=False)
        st.bar_chart(chart_df.set_index("Class"), color="#3A0F34")

    # ── Colour distribution chart ─────────────────────────────────────────────
    by_colour = stats.get("by_colour", {})
    if by_colour:
        st.markdown("**Colour Distribution**")
        col_df = pd.DataFrame(
            {"Colour": list(by_colour.keys()), "Count": list(by_colour.values())}
        ).sort_values("Count", ascending=False)
        st.bar_chart(col_df.set_index("Colour"), color="#EAAA00")

    # ── Daily event trend ─────────────────────────────────────────────────────
    daily = stats.get("daily", [])
    if daily:
        st.markdown("**Events per Day (last 30 days)**")
        daily_df = pd.DataFrame(daily).rename(columns={"day": "Date", "cnt": "Events"})
        st.line_chart(daily_df.set_index("Date"))

    # ── Export ────────────────────────────────────────────────────────────────
    csv = history.to_csv()
    st.download_button(
        "⬇ Export All Records CSV",
        data=csv,
        file_name="vehicle_intelligence_history.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION HISTORY VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_history_view() -> None:
    st.markdown("### 🗂 Detection History")

    history: DetectionHistory = st.session_state.history

    search = st.text_input(
        "Search",
        key="hist_search",
        placeholder="Vehicle number, type, colour, or source",
    )

    filter_cols = st.columns(5)

    classes_opts = ["All"] + history.distinct_values("vehicle_class")
    sources_opts = ["All"] + history.distinct_values("source")
    colours_opts = ["All"] + history.distinct_values("est_colour")

    with filter_cols[0]:
        sel_class = st.selectbox("Vehicle Type", classes_opts, key="hist_class")
    with filter_cols[1]:
        sel_source = st.selectbox("Source", sources_opts, key="hist_source")
    with filter_cols[2]:
        sel_colour = st.selectbox("Colour", colours_opts, key="hist_colour")
    with filter_cols[3]:
        date_from = st.date_input("From date", value=None, key="hist_from")
    with filter_cols[4]:
        date_to = st.date_input("To date", value=None, key="hist_to")

    df = history.filtered_records_dataframe(
        vehicle_class=None if sel_class == "All" else sel_class,
        source=None if sel_source == "All" else sel_source,
        colour=None if sel_colour == "All" else sel_colour,
        date_from=date_from.strftime("%Y-%m-%d") if date_from else None,
        date_to=date_to.strftime("%Y-%m-%d") if date_to else None,
        search=search.strip() if search else None,
    )

    if df.empty:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color: #5A6472;">
            <div style="font-size:2rem;">📂</div>
            <div style="margin-top:0.5rem;">No detection records found.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    display_df = build_history_display(df)

    st.markdown(f"**{len(display_df)} record(s)** — newest first")

    # Pagination
    PAGE_SIZE = 50
    total_pages = max(1, (len(display_df) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1, key="hist_page"
    )
    start_idx = (page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_df = display_df.iloc[start_idx:end_idx]

    st.dataframe(
        page_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "SI. No.":       st.column_config.NumberColumn("SI. No.", width="small"),
            "Vehicle Type":  st.column_config.TextColumn("Vehicle Type"),
            "Colour":        st.column_config.TextColumn("Colour"),
            "Vehicle Number": st.column_config.TextColumn("Vehicle Number"),
            "Date & Time":   st.column_config.TextColumn("Date & Time"),
            "Source":        st.column_config.TextColumn("Source", width="small"),
        },
    )

    if total_pages > 1:
        st.caption(f"Page {page} of {total_pages} · {len(display_df)} total records")

    # ── Export & Clear ────────────────────────────────────────────────────────
    col_dl, col_clear = st.columns(2)

    with col_dl:
        csv_export = _build_csv_export(display_df)
        st.download_button(
            "⬇ Export CSV",
            data=csv_export,
            file_name="vehicle_detection_history.csv",
            mime="text/csv",
        )

    with col_clear:
        if not st.session_state.confirm_clear:
            if st.button("🗑 Clear History", key="clear_hist_btn"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("This will permanently delete **all** detection records.")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("✓ Confirm Delete", key="confirm_delete"):
                    history.reset_all()
                    st.session_state.current_session = None
                    st.session_state.confirm_clear = False
                    st.success("All history deleted.")
                    st.rerun()
            with cc2:
                if st.button("✗ Cancel", key="cancel_delete"):
                    st.session_state.confirm_clear = False
                    st.rerun()


def _build_csv_export(display_df: pd.DataFrame) -> str:
    """Build CSV string from display table (same 6 columns, same order)."""
    import io as _io
    buf = _io.StringIO()
    display_df.to_csv(buf, index=False)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    init_session_state()
    render_sidebar()
    render_header()

    view = st.session_state.view

    if view == "home":
        render_home()
    elif view == "camera":
        render_camera_view()
    elif view == "upload":
        render_upload_view()
    elif view == "analytics":
        render_analytics_view()
    elif view == "history":
        render_history_view()
    elif view == "settings":
        render_settings_view()
    else:
        st.session_state.view = "home"
        st.rerun()


if __name__ == "__main__":
    main()
