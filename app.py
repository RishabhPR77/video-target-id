# app.py — Fixed & Enhanced Deployment Version
# -----------------------------------------------
# FIXES:
# ✅ Fix #1: TypeError on st.image (cols[i].image crash) — validate img shape, dtype, safe try/except
# ✅ Fix #2: cosine_sim called with None ref_face in evidence loop — added guard
# ✅ Fix #3: preview_box.image crash during scan — wrapped in try/except
# ✅ Fix #4: fpdf2 pdf.output(dest='S') API change — now uses bytes(pdf.output())
# ✅ Fix #5: f.seek(0) unreliable on UploadedFile — switched to f.getvalue() everywhere
# ✅ Fix #6: load_face_engine unguarded crash — wrapped in try/except with user-facing error
# ✅ Fix #7: Missing threshold slider in UI — added to sidebar
# ✅ Fix #8: cv2.cvtColor on unexpected channel counts — validate before converting
# ✅ Fix #9: pose_embs stack crash when only 1 item — ensured safe stacking
# ✅ Fix #10: index out of range on cols[i] when fewer valid images than cols — track col index separately

import os
import io
import time
import uuid
import shutil
import subprocess
import functools
import zipfile
import tempfile
import gc
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from constants import FACE_THR, CONSEC, COOLDOWN

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

from face_module import init_face_app, get_faces, mean_normalize_stack, cosine_sim
from pose_module import extract_pose_feats_bgr

# ----------------------------
# 1. Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Video Target ID System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------
# 2. Custom CSS (Premium UI)
# ----------------------------
def inject_pro_ui():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif !important;
        }

        .stApp {
            background: #020817;
            background-image:
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(56,189,248,0.08) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,92,246,0.06) 0%, transparent 50%);
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] {
            background: rgba(10, 15, 30, 0.97) !important;
            border-right: 1px solid rgba(56,189,248,0.12) !important;
        }

        /* ---- GLASS CARDS ---- */
        .glass {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(56, 189, 248, 0.1);
            border-radius: 16px;
            padding: 28px;
            box-shadow: 0 4px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
            backdrop-filter: blur(16px);
            margin-bottom: 20px;
        }

        /* ---- TOP BAR ---- */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 28px;
            background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(7,11,22,0.95) 100%);
            border: 1px solid rgba(56,189,248,0.15);
            border-radius: 16px;
            margin-bottom: 28px;
            box-shadow: 0 0 40px rgba(56,189,248,0.05);
        }

        .brand-title {
            font-size: 26px !important;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #f8fafc !important;
            background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }

        .brand-sub {
            font-size: 12px;
            color: #475569;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 2px;
        }

        /* ---- STEP INDICATOR ---- */
        .step-row {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 4px;
            transition: background 0.2s;
        }
        .step-row.active {
            background: rgba(56,189,248,0.08);
            border: 1px solid rgba(56,189,248,0.2);
        }
        .step-row.done {
            background: rgba(34,197,94,0.06);
            border: 1px solid rgba(34,197,94,0.15);
        }
        .step-row.pending {
            opacity: 0.4;
        }

        /* ---- BUTTONS ---- */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            height: 44px !important;
            background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
            border: none !important;
            color: #fff !important;
            transition: all 0.2s ease !important;
            letter-spacing: 0.01em !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 20px rgba(14,165,233,0.35) !important;
        }
        .stButton > button:active {
            transform: translateY(0) !important;
        }

        /* ---- TABS ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: transparent;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            background-color: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px;
            color: #64748b;
            font-weight: 500;
            padding: 0 18px;
            transition: all 0.25s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(56,189,248,0.06);
            border-color: rgba(56,189,248,0.2);
            color: #cbd5e1;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(14,165,233,0.12) 0%, rgba(37,99,235,0.12) 100%) !important;
            border: 1px solid rgba(14,165,233,0.35) !important;
            color: #e2e8f0 !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] { display: none; }

        /* ---- PILLS ---- */
        .pill { padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; display: inline-block; letter-spacing: 0.05em; }
        .pill.ok    { background: rgba(34,197,94,0.1);  color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
        .pill.warn  { background: rgba(234,179,8,0.1);  color: #fbbf24; border: 1px solid rgba(234,179,8,0.25); }
        .pill.info  { background: rgba(14,165,233,0.1); color: #38bdf8; border: 1px solid rgba(14,165,233,0.25); }
        .pill.error { background: rgba(239,68,68,0.1);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

        /* ---- METRIC CARDS ---- */
        [data-testid="stMetric"] {
            background: rgba(15,23,42,0.5);
            border: 1px solid rgba(56,189,248,0.1);
            border-radius: 12px;
            padding: 16px 20px;
        }
        [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700; }

        /* ---- PROGRESS BAR ---- */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #0ea5e9, #8b5cf6) !important;
            border-radius: 4px;
        }

        /* ---- SLIDERS ---- */
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background: #0ea5e9 !important;
            border: 2px solid #38bdf8 !important;
        }

        /* ---- INPUTS ---- */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
            background: rgba(15,23,42,0.8) !important;
            border-color: rgba(56,189,248,0.2) !important;
            color: #e2e8f0 !important;
        }

        /* ---- DATAFRAME ---- */
        .stDataFrame { border-radius: 12px; overflow: hidden; }

        /* ---- FILE UPLOADER ---- */
        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(56,189,248,0.2) !important;
            border-radius: 12px !important;
            background: rgba(14,165,233,0.03) !important;
        }

        /* ---- MATCH CARD ---- */
        .match-card {
            background: rgba(15,23,42,0.5);
            border: 1px solid rgba(56,189,248,0.1);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            transition: border-color 0.2s;
        }
        .match-card:hover {
            border-color: rgba(56,189,248,0.3);
        }

        /* ---- DIVIDER ---- */
        hr { border-color: rgba(255,255,255,0.06) !important; }

        /* Hide Streamlit chrome */
        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }
        header    { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# 3. Helper Functions
# ----------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = sec // 60, sec % 60
    return f"{m:02d}:{s:02d}"


def safe_imdecode(file_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes → BGR uint8 ndarray, or None on failure."""
    try:
        arr = np.frombuffer(file_bytes, np.uint8)
        if arr.size == 0:
            return None
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # Validate result
        if img is None:
            return None
        if img.ndim != 3 or img.shape[2] != 3:
            return None
        if img.shape[0] < 4 or img.shape[1] < 4:
            return None
        return img.astype(np.uint8)
    except Exception:
        return None


def bgr_to_rgb_safe(img: np.ndarray) -> Optional[np.ndarray]:
    """Convert BGR uint8 → RGB uint8 safely, returning None on failure."""
    try:
        if img is None:
            return None
        if img.ndim == 2:           # grayscale → convert to 3-channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:     # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return None
        return np.clip(img, 0, 255).astype(np.uint8)
    except Exception:
        return None


def pick_best_face(faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not faces:
        return None
    return max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))


def sanitize_filename(name: str) -> str:
    """Keep only filesystem-safe characters for generated filenames."""
    safe = ''.join(c if (c.isalnum() or c in ' ._-') else '_' for c in name)
    return safe.strip() or "video"


def draw_match_annotation(frame: np.ndarray, bbox, fused_score: float):
    """In-place green tracking box + confidence label.

    Shared by evidence crops and the annotated video output so the two stay
    visually identical.
    """
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
        label = f"Conf: {fused_score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - 22)
        cv2.rectangle(frame, (x1, ty), (min(w - 1, x1 + tw + 4), y1), (0, 220, 80), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    except Exception:
        pass


def confidence_badge(conf: float):
    """Map a confidence score to (pill_css_class, human_label)."""
    if conf >= 0.75:
        return "ok", "High"
    if conf >= 0.55:
        return "warn", "Medium"
    return "error", "Low"


@functools.lru_cache(maxsize=1)
def _find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg binary for best-effort H.264 transcoding (Issue 16).

    Prefers a system ffmpeg on PATH, then falls back to imageio-ffmpeg's
    bundled binary. Returns None if neither exists — the annotated pipeline
    still works, it just skips the codec upgrade for browser previews.
    """
    try:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    except Exception:
        pass
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return None


def _transcode_h264(src_path: str, dst_path: str, ffmpeg_bin: str, timeout: int = 600) -> bool:
    """Best-effort mp4v → H.264 transcode for browser compatibility."""
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-y", "-i", src_path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", dst_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return (proc.returncode == 0 and os.path.exists(dst_path)
                and os.path.getsize(dst_path) > 1024)
    except Exception:
        return False


def _maybe_transcode(src_path: str, ffmpeg_bin) -> Optional[str]:
    """Transcode src to a sibling *_h264.mp4 if possible; always return a path.

    Best-effort: on any failure or when ffmpeg is unavailable, the original
    mp4v file is kept so the pipeline never breaks over this improvement.
    """
    if not ffmpeg_bin or not src_path or not os.path.exists(src_path):
        return src_path
    dst_path = os.path.splitext(src_path)[0] + "_h264.mp4"
    if _transcode_h264(src_path, dst_path, ffmpeg_bin):
        try:
            os.remove(src_path)
        except Exception:
            pass
        return dst_path
    try:
        if os.path.exists(dst_path):
            os.remove(dst_path)
    except Exception:
        pass
    return src_path


def _write_reel_divider(writer, width, height, fps, video_name, src_start_sec):
    """Write ~1 s of dark labelled frames before a reel clip."""
    try:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, "HIGHLIGHT REEL", (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
        cv2.putText(frame, str(video_name)[:40], (24, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 232, 240), 1)
        cv2.putText(frame, f"Match @ {fmt_time(src_start_sec)}", (24, 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
        n = max(1, int(round(fps)))
        for _ in range(n):
            # ISSUE 17: defensive size guard — divider canvas must match writer
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    except Exception:
        pass


def _write_reel_clip_fallback(cap, start_f, end_f, width, height, writer):
    """Last-resort sequential copy for a reel clip. Never raises.

    Used when the retiming path hits an unexpected error so a single odd clip
    can never abort highlight-reel generation.
    """
    try:
        start_f = max(0, int(start_f))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for _ in range(max(0, int(end_f) - start_f + 1)):
            ok, frm = cap.read()
            if not ok or frm is None:
                break
            if frm.shape[1] != width or frm.shape[0] != height:
                frm = cv2.resize(frm, (width, height))
            writer.write(frm)
    except Exception:
        pass


def _write_reel_clip(cap, start_f, end_f, clip_fps, out_fps, width, height, writer):
    """Write clip frames [start_f..end_f] into the reel at reel timing (Issue 18).

    When the source clip's native frame rate differs from the reel's by more
    than 0.5 fps, the clip is retimed by nearest-source-frame sampling: each
    output frame picks the source frame closest to its virtual playback
    timestamp (src = start_f + round(out_i * clip_fps / out_fps)), using only
    forward reads so decoding stays sequential. Clips at effectively the same
    fps are copied 1:1. Any edge case / error falls back to a plain sequential
    copy — highlight-reel generation must never break over a single clip.
    """
    start_f  = max(0, int(start_f))
    end_f    = max(start_f, int(end_f))
    clip_fps = float(clip_fps)
    out_fps  = float(out_fps)

    try:
        if clip_fps <= 0 or abs(clip_fps - out_fps) <= 0.5:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            for _ in range(end_f - start_f + 1):
                ok, frm = cap.read()
                if not ok or frm is None:
                    break
                if frm.shape[1] != width or frm.shape[0] != height:
                    frm = cv2.resize(frm, (width, height))
                writer.write(frm)
            return

        n_out    = max(1, int(round((end_f - start_f + 1) * out_fps / clip_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        cur      = start_f - 1   # index of the frame held in last_frm
        last_frm = None
        for out_i in range(n_out):
            target = start_f + min(
                end_f - start_f,
                int(round(out_i * clip_fps / out_fps)))
            while cur < target:
                ok, frm = cap.read()
                if not ok or frm is None:
                    last_frm = None
                    break
                last_frm = frm
                cur += 1
            if last_frm is None or cur != target:
                break
            frm = last_frm
            if frm.shape[1] != width or frm.shape[0] != height:
                frm = cv2.resize(frm, (width, height))
            writer.write(frm)
    except Exception:
        _write_reel_clip_fallback(cap, start_f, end_f, width, height, writer)


def build_highlight_reel():
    """Concatenate per-source annotated clips into a highlight reel.

    Annotated videos are now written at real source time (Issue 15), so clip
    windows map 1:1 onto the source timeline. Source clips that don't match the
    reel canvas resolution are resized rather than silently dropped (Issue 17),
    and clips running at a different native frame rate are retimed to the reel's
    fps so every clip plays at the same real-time speed (Issue 18).
    """
    try:
        annotated_videos = st.session_state.get("annotated_videos", {}) or {}
        df = st.session_state.get("timeline_df", pd.DataFrame())
        if not annotated_videos or df is None or len(df) == 0:
            st.session_state.highlight_reel_path = ""
            return

        ffmpeg_bin = _find_ffmpeg()  # ISSUE 16: optional H.264 upgrade
        first_path = next(iter(annotated_videos.values()))
        probe = cv2.VideoCapture(str(first_path))
        if not probe.isOpened():
            st.session_state.highlight_reel_path = ""
            return
        width  = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = probe.get(cv2.CAP_PROP_FPS) or 25.0
        probe.release()

        pad = 2.0  # source-time seconds of context around each clip
        ordered = df.copy().sort_values(["Start (sec)", "Video"])
        clips: list = []
        for _, row in ordered.iterrows():
            path = annotated_videos.get(str(row.get("Video", "")))
            if not path or not os.path.exists(str(path)):
                continue
            src_start = float(row.get("Start (sec)", 0) or 0)
            src_dur   = float(row.get("Duration (sec)", 2.0) or 2.0)
            clips.append((str(path), src_start, src_dur,
                          str(row.get("Video", ""))))

        if not clips:
            st.session_state.highlight_reel_path = ""
            return

        out_path = os.path.join(st.session_state.screens_dir, "highlight_reel.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            st.session_state.highlight_reel_path = ""
            return

        for path, src_start, src_dur, vname in clips:
            _write_reel_divider(writer, width, height, fps, vname, src_start)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            # ISSUE 18: read the clip's OWN frame rate — sources with a truly
            # different native fps must be retimed or they'd play back at the
            # wrong speed against the reel's timeline
            clip_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if clip_fps <= 0:
                clip_fps = fps
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_f = max(0, int((src_start - pad) * clip_fps))
            end_f   = min(total - 1, int((src_start + src_dur + pad) * clip_fps))
            if end_f <= start_f:
                end_f = min(total - 1, start_f + 1)
            try:
                _write_reel_clip(cap, start_f, end_f, clip_fps, fps,
                                 width, height, writer)
            finally:
                cap.release()

        writer.release()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            st.session_state.highlight_reel_path = _maybe_transcode(out_path, ffmpeg_bin)
        else:
            st.session_state.highlight_reel_path = ""
    except Exception:
        st.session_state.highlight_reel_path = ""


def generate_pdf_report(df: pd.DataFrame, case_name: str) -> bytes:
    """Generate a forensic PDF report. Returns bytes."""
    if not FPDF_AVAILABLE:
        return b""
    try:
        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(30, 30, 50)
        pdf.cell(0, 12, "Forensic Video Analysis Report", ln=True, align='C')
        pdf.ln(2)

        pdf.set_font("Arial", size=11)
        pdf.set_text_color(80, 80, 100)
        pdf.cell(0, 8, f"Case: {case_name}", ln=True)
        pdf.cell(0, 8, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)
        pdf.cell(0, 8, f"Total matches: {len(df)}", ln=True)
        pdf.ln(6)

        # Table header
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(30, 58, 138)
        pdf.set_text_color(255, 255, 255)
        for header, w in [("Video", 50), ("Timestamp", 30), ("Confidence", 30), ("Notes", 80)]:
            pdf.cell(w, 10, header, border=1, fill=True)
        pdf.ln()

        # Table rows
        pdf.set_font("Arial", size=9)
        for idx, row in df.iterrows():
            pdf.set_fill_color(245, 247, 255) if idx % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(20, 20, 40)
            vid_name = str(row.get('Video', ''))[:18]
            pdf.cell(50, 9, vid_name, border=1, fill=True)
            pdf.cell(30, 9, str(row.get('Start Time', '')), border=1, fill=True)
            conf = row.get('Best Confidence', 0)
            pdf.cell(30, 9, f"{float(conf):.3f}", border=1, fill=True)
            pdf.cell(80, 9, "Match confirmed via AI scan", border=1, fill=True)
            pdf.ln()

        # FIX #4: fpdf2 returns bytes from output(); legacy fpdf returns str
        raw = pdf.output()
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        return str(raw).encode('latin-1')
    except Exception as e:
        # Fallback: return minimal valid PDF bytes
        st.warning(f"PDF generation error: {e}")
        return b""


@dataclass
class MatchEvent:
    t_sec: float
    face_score: float
    pose_score: float
    fused_score: float
    frame_index: int
    screenshot_path: Optional[str] = None
    video_name: str = ""


def group_events(events: List[MatchEvent], merge_gap_sec: float = 2.0) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["Video", "Start Time", "End Time", "Duration",
                                     "Duration (sec)", "Best Confidence", "Best Face",
                                     "Best Pose", "Start (sec)", "Screenshot"])

    events = sorted(events, key=lambda e: (e.video_name, e.t_sec))
    rows = []
    i = 0
    while i < len(events):
        v = events[i].video_name
        start = events[i].t_sec
        end = events[i].t_sec
        block = [events[i]]
        i += 1
        while i < len(events) and events[i].video_name == v and (events[i].t_sec - end) <= merge_gap_sec:
            end = events[i].t_sec
            block.append(events[i])
            i += 1

        best = max(block, key=lambda e: e.fused_score)
        rows.append({
            "Video":            v,
            "Start (sec)":      float(start),
            "Start Time":       fmt_time(start),
            "End Time":         fmt_time(end),
            "Duration":         f"{end - start:.1f}s",
            "Duration (sec)":   round(float(end - start), 2),
            "Best Confidence":  float(best.fused_score),
            "Best Face":        float(best.face_score),
            "Best Pose":        float(best.pose_score),
            "Screenshot":       best.screenshot_path or "",
        })
    return pd.DataFrame(rows)


def make_zip_of_files(paths: List[str]) -> bytes:
    """Zip existing file paths into an in-memory archive (by basename)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p and os.path.exists(p):
                zf.write(p, arcname=os.path.basename(p))
    return buf.getvalue()


def cleanup_case_artifacts():
    """Best-effort filesystem cleanup before starting a new case.

    Removes only the *current* session's temp videos and evidence screenshots.
    The rmtree on screens_dir also clears the annotated/ subfolder and the
    highlight reel, which live inside it. Never touches other sessions'
    directories (they live under their own session_id). Keeps session_id intact;
    a fresh screens_dir is re-created by the caller/session-init afterwards.
    Failures are swallowed on purpose — a locked/missing file must not crash
    the reset button.
    """
    for p in st.session_state.get('video_files', []):
        try:
            if p and os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass
    screens_dir = st.session_state.get('screens_dir', '')
    if screens_dir:
        try:
            shutil.rmtree(screens_dir, ignore_errors=True)
        except Exception:
            pass


@st.cache_resource(show_spinner=False)
def load_face_engine():
    """Load InsightFace engine, cached across reruns. Returns (engine, error_msg)."""
    try:
        engine = init_face_app()
        return engine, None
    except Exception as e:
        return None, str(e)


# ----------------------------
# 4. Session State Init
# ----------------------------
_DEFAULTS = {
    'step': 1,
    'case_name': "New Investigation",
    'target_files': [],
    'ref_face': None,
    'ref_pose': None,
    'video_files': [],
    'video_names': [],
    'single_video_path': None,
    'raw_events': [],
    'timeline_df': pd.DataFrame(),
    'annotated_videos': {},
    'highlight_reel_path': "",
    'start_time_player': 0,
    'active_video_for_player': "",
    'threshold': 0.55,
    'skip_frames': 5,
    'process_width': "Medium (640px)",
    'face_weight': 0.80,
    'pose_weight': 0.20,
    'consent_ok': False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ISSUE 3: unique session ID so evidence directories never collide across users
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'screens_dir' not in st.session_state:
    st.session_state.screens_dir = ensure_dir(
        os.path.join(tempfile.gettempdir(), "target_id_screens",
                     st.session_state.session_id))


# ----------------------------
# 5. Sidebar
# ----------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎯 Control Panel")

        # Step Progress
        steps = [("Target Setup", "👤"), ("Video Source", "🎥"), ("AI Scan", "🔍"), ("Results", "📊")]
        st.markdown('<div class="glass" style="padding:14px 18px;">', unsafe_allow_html=True)
        for i, (label, icon) in enumerate(steps, 1):
            cur = st.session_state.step
            if i == cur:
                css = "active"
                indicator = f"▶ Step {i}"
            elif i < cur:
                css = "done"
                indicator = f"✓ Step {i}"
            else:
                css = "pending"
                indicator = f"○ Step {i}"
            st.markdown(
                f'<div class="step-row {css}">'
                f'<span style="font-size:16px">{icon}</span>'
                f'<span style="font-size:13px; font-weight:600">{indicator}: {label}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ Detection Settings")

        # FIX #7: Added threshold slider that was missing from UI
        st.session_state.threshold = st.slider(
            "Detection Threshold", 0.30, 0.95,
            float(st.session_state.threshold), 0.05,
            help="Minimum fused score to count as a match"
        )

        # ISSUE 2: cap face_weight min at 0.70 so pose_weight can never exceed 0.30
        st.session_state.face_weight = st.slider(
            "Face Weight", 0.70, 1.0,
            float(st.session_state.face_weight), 0.05,
            help="Minimum 0.70 — pose can nudge but never carry a match on its own"
        )
        st.session_state.pose_weight = round(1.0 - st.session_state.face_weight, 2)
        st.caption(f"Pose Weight: **{st.session_state.pose_weight:.2f}** (auto-capped)")

        st.markdown("---")

        # Reference status
        st.markdown("### 📋 Reference Status")
        face_ok = st.session_state.ref_face is not None
        pose_ok = st.session_state.ref_pose is not None
        st.markdown(
            f'<span class="pill {"ok" if face_ok else "error"}">{"✓" if face_ok else "✗"} Face Embedding</span>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<span class="pill {"ok" if pose_ok else "warn"}">{"✓" if pose_ok else "⚠"} Pose Embedding</span>',
            unsafe_allow_html=True
        )

        if st.session_state.step > 1:
            st.markdown("---")
            if st.button("🔄 New Case", use_container_width=True):
                # ISSUE 7: best-effort cleanup of this session's temp artifacts
                cleanup_case_artifacts()
                for k, v in _DEFAULTS.items():
                    st.session_state[k] = v
                # keep session_id; recreate a fresh evidence dir for the next case
                st.session_state.screens_dir = ensure_dir(
                    os.path.join(tempfile.gettempdir(), "target_id_screens",
                                 st.session_state.session_id))
                st.rerun()


# ----------------------------
# 6. Step Renderers
# ----------------------------
def render_target_step():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 👤 Step 1 — Who to find?")
    st.caption("Upload 1–5 clear reference photos of the target person (front-facing preferred).")

    files = st.file_uploader(
        "Drop reference images here (JPG / PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="ref_uploader"
    )

    # FIX #1 / #10: Robust image preview — track valid col index separately,
    #               validate shape, wrap in try/except, explicit uint8 cast
    if files:
        valid_imgs = []
        for f in files[:5]:
            try:
                img = safe_imdecode(f.getvalue())  # FIX #5: use getvalue() always
                if img is not None:
                    valid_imgs.append((f.name, img))
            except Exception:
                pass

        if valid_imgs:
            st.markdown("#### Preview")
            cols = st.columns(len(valid_imgs))
            for col_idx, (fname, img) in enumerate(valid_imgs):
                try:
                    rgb = bgr_to_rgb_safe(img)      # FIX #8: safe channel conversion
                    if rgb is not None:
                        cols[col_idx].image(
                            rgb,
                            caption=fname[:20],
                            use_container_width=True   # FIX #1: non-deprecated param
                        )
                except Exception as e:
                    cols[col_idx].warning(f"Preview error: {e}")

        st.session_state.target_files = files

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("⚙️ Build Reference", use_container_width=True):
            if not st.session_state.target_files:
                st.error("Upload at least one image first.")
            else:
                # FIX #6: Wrapped engine load in error-checked helper
                with st.spinner("Loading AI engine…"):
                    app, err = load_face_engine()
                if err or app is None:
                    st.error(f"Failed to load face engine: {err}")
                else:
                    with st.spinner("Extracting biometric features…"):
                        face_embs, pose_embs = [], []
                        found_count = 0
                        for f in st.session_state.target_files:
                            try:
                                img = safe_imdecode(f.getvalue())  # FIX #5
                                if img is None:
                                    continue
                                faces = get_faces(img, app)
                                best = pick_best_face(faces)
                                if best:
                                    face_embs.append(best['emb'])
                                    found_count += 1
                                pf = extract_pose_feats_bgr(img)
                                if pf is not None:
                                    pose_embs.append(pf)
                            except Exception:
                                continue

                    if not face_embs:
                        st.error("No faces detected. Try clearer, well-lit front-facing photos.")
                    else:
                        st.session_state.ref_face = mean_normalize_stack(face_embs)
                        # FIX #9: Safe pose stacking — np.vstack handles variable list sizes
                        if pose_embs:
                            try:
                                pose_stack = np.vstack(pose_embs)
                                mean_pose = np.mean(pose_stack, axis=0)
                                norm = np.linalg.norm(mean_pose) + 1e-9
                                st.session_state.ref_pose = (mean_pose / norm).astype(np.float32)
                            except Exception:
                                st.session_state.ref_pose = None
                        else:
                            st.session_state.ref_pose = None

                        st.success(f"✅ Reference built from {found_count} face(s). "
                                   f"Pose: {'✓' if st.session_state.ref_pose is not None else '✗ not found'}")

    with col2:
        if st.button("💾 Save Profile", use_container_width=True):
            if st.session_state.ref_face is None:
                st.warning("Build the reference first.")
            else:
                tmp = io.BytesIO()
                # WARNING — exported .npz contains raw, UNENCRYPTED biometric
                # embeddings. See README → "Data Handling" for storage and
                # deletion obligations.
                np.savez(
                    tmp,
                    ref_face=st.session_state.ref_face,
                    ref_pose=st.session_state.ref_pose
                    if st.session_state.ref_pose is not None else np.array([])
                )
                tmp.seek(0)
                st.download_button(
                    "⬇️ Download .npz",
                    data=tmp.getvalue(),
                    file_name="target_profile.npz",
                    mime="application/octet-stream"
                )
                st.caption("⚠️ Exported file is **unencrypted biometric data** — "
                           "store securely and delete when no longer needed.")

    st.markdown("---")
    st.session_state.consent_ok = st.checkbox(
        "✅  I confirm I am authorised to process this data for lawful purposes.")

    col_nav_1, col_nav_2 = st.columns([4, 1])
    with col_nav_2:
        if st.button("Next →", use_container_width=True):
            if st.session_state.ref_face is None:
                st.error("Click 'Build Reference' first.")
            elif not st.session_state.consent_ok:
                st.error("Tick the authorisation box to continue.")
            else:
                st.session_state.step = 2
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_source_step():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 🎥 Step 2 — Where to look?")
    st.caption("Upload the surveillance footage you want to scan. Multiple files supported.")

    uploaded = st.file_uploader(
        "Upload Video File(s) (MP4 / AVI / MOV / MKV)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        accept_multiple_files=True
    )

    if uploaded:
        st.info(f"**{len(uploaded)}** file(s) ready: {', '.join(u.name for u in uploaded)}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Next →", use_container_width=True):
            if not uploaded and not st.session_state.video_files:
                st.error("Upload at least one video.")
            else:
                if uploaded:
                    paths, names = [], []
                    for up in uploaded:
                        try:
                            # ISSUE 6: keep the real container extension — an
                            # .avi/.mov/.mkv forced into a .mp4 name can make
                            # OpenCV's backend fail or misread the file
                            suffix = os.path.splitext(up.name)[1] or '.mp4'
                            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                            tfile.write(up.getvalue())
                            tfile.close()
                            paths.append(tfile.name)
                            names.append(up.name)
                        except Exception as e:
                            st.warning(f"Could not save {up.name}: {e}")
                    if not paths:
                        st.error("No video files could be saved. Try again.")
                    else:
                        st.session_state.video_files = paths
                        st.session_state.video_names = names
                        st.session_state.single_video_path = paths[0]
                        st.session_state.step = 3
                        st.rerun()
                else:
                    st.session_state.step = 3
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_scan_step():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 🔍 Step 3 — AI Deep Scan")

    with st.expander("ℹ️ Performance guide — click to expand"):
        st.markdown("""
| Video Length | Recommended Skip | Speed | Accuracy |
|:---|:---|:---|:---|
| **Short (< 2 min)** | 0 – 5 | Normal | ⭐⭐⭐⭐⭐ |
| **Medium (2 – 10 min)** | 5 – 15 | Fast | ⭐⭐⭐⭐ |
| **Long (10 – 30 min)** | 15 – 30 | Very Fast | ⭐⭐⭐ |
| **Archive (30+ min)** | 30 – 60 | Turbo | ⭐⭐ |

> Higher skip values are faster but may miss brief appearances.
        """)

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.skip_frames = st.slider("Frame Skipping", 0, 60, st.session_state.skip_frames)
    with c2:
        processing_quality = st.select_slider(
            "Scan Resolution",
            options=["Low (320px)", "Medium (640px)", "High (Native)"],
            value=st.session_state.process_width
        )
        st.session_state.process_width = processing_quality

    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("🚀 Start Analysis", use_container_width=True):
            run_analysis()

    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------
# 7. Analysis Engine
# ----------------------------
def run_analysis():
    # FIX #6: Guard engine load
    with st.spinner("Loading AI engine…"):
        app, err = load_face_engine()
    if err or app is None:
        st.error(f"Cannot start analysis — face engine failed to load: {err}")
        return

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Processing…")

    prog_bar   = st.progress(0.0)
    status_txt = st.empty()
    preview_bx = st.empty()

    all_events: List[MatchEvent] = []
    annotated_videos: Dict[str, str] = {}
    annot_dir = ensure_dir(os.path.join(st.session_state.screens_dir, "annotated"))
    # ISSUE 16: one optional H.264 transcode step for browser-friendly previews
    ffmpeg_bin = _find_ffmpeg()
    # how long (in source-time seconds) the tracking box keeps drawing after the
    # detector last saw the face — avoids flicker between hits
    hold_sec = 1.0

    target_width = 320
    if "640" in st.session_state.process_width:
        target_width = 640
    elif "Native" in st.session_state.process_width:
        target_width = 1280  # capped to avoid OOM on free tiers

    for v_idx, video_path in enumerate(st.session_state.video_files):
        video_name = st.session_state.video_names[v_idx]
        status_txt.markdown(f"**Scanning:** `{video_name}` ({v_idx + 1}/{len(st.session_state.video_files)})")
        prog_bar.progress(0.0)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.warning(f"Cannot open video: {video_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_i      = 0
        seen_last    = -999.0
        consec       = 0          # ISSUE 1: consecutive matching-frame counter

        # ISSUE 13/15: per-source annotated output — lazy writer (needs
        # frame_small dims); every source frame is written at real fps
        writer       = None
        annot_path   = ""
        annot_failed = False
        last_bbox    = None
        last_seen_ts = -999.0
        last_fused   = 0.0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if frame is None or frame.size == 0:
                frame_i += 1
                continue

            h, w = frame.shape[:2]

            # Smart resize — applied to EVERY frame since it feeds the writer
            if w > target_width:
                scale       = target_width / float(w)
                frame_small = cv2.resize(frame, (target_width, int(h * scale)))
            else:
                frame_small = frame.copy()

            t_sec = frame_i / fps

            # ISSUE 13/15: create annotated writer on the first frame
            if writer is None and not annot_failed:
                try:
                    annot_path = os.path.join(
                        annot_dir,
                        f"{sanitize_filename(os.path.splitext(video_name)[0])}_annotated.mp4")
                    writer = cv2.VideoWriter(
                        annot_path, cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (frame_small.shape[1], frame_small.shape[0]))
                    if not writer.isOpened():
                        writer = None
                        annot_failed = True
                except Exception:
                    writer = None
                    annot_failed = True

            # ISSUE 15: write EVERY source frame at real fps so playback speed
            # is correct; the tracking box persists within the hold window
            if writer is not None:
                try:
                    annot_frame = frame_small.copy()
                    if last_bbox is not None and (t_sec - last_seen_ts) <= hold_sec:
                        draw_match_annotation(annot_frame, last_bbox, last_fused)
                    writer.write(annot_frame)
                    del annot_frame
                except Exception:
                    pass

            # ISSUE 15: frame skipping now gates DETECTION only — the video is
            # already written above, so the scan stays fast while the file is
            # smooth and real-time (same detection cadence as before)
            if st.session_state.skip_frames > 0 and (frame_i % (st.session_state.skip_frames + 1) != 0):
                frame_i += 1
                del frame, frame_small
                continue

            # Face scoring
            face_score = 0.0
            best_face  = None
            try:
                faces = get_faces(frame_small, app)
                if faces and st.session_state.ref_face is not None:
                    sims       = [(cosine_sim(f['emb'], st.session_state.ref_face), f) for f in faces]
                    face_score, best_face = max(sims, key=lambda x: x[0])
                    face_score = float(face_score)
            except Exception:
                face_score = 0.0
                best_face  = None

            # Pose scoring — only meaningful once face clears its own threshold
            pose_score = 0.0
            if face_score >= FACE_THR and st.session_state.ref_pose is not None:
                try:
                    pf = extract_pose_feats_bgr(frame_small)
                    if pf is not None:
                        pose_score = cosine_sim(pf, st.session_state.ref_pose)
                except Exception:
                    pose_score = 0.0

            # Fusion (pose contributes only when face gate is passed)
            fused = (st.session_state.face_weight * face_score +
                     st.session_state.pose_weight * pose_score)

            # ISSUE 13: track the latest face passing the face gate so the box
            # can persist between hits (real source time now)
            if (best_face is not None and face_score >= FACE_THR
                    and st.session_state.ref_face is not None):
                last_bbox     = tuple(best_face['bbox'])
                last_seen_ts  = t_sec
                last_fused    = fused

            # ISSUE 1: hit requires BOTH face_gate AND fused threshold
            hit = (face_score >= FACE_THR) and (fused >= st.session_state.threshold)
            if hit:
                consec += 1
            else:
                consec = 0

            if hit and consec >= CONSEC and (t_sec - seen_last) >= COOLDOWN:
                seen_last = t_sec

                # Draw bounding box on evidence frame (shared helper)
                evidence = frame_small.copy()
                if best_face is not None and st.session_state.ref_face is not None:
                    draw_match_annotation(evidence, best_face['bbox'], fused)

                # ISSUE 3: namespace crop filenames with session_id
                crop_path = ""
                try:
                    img_name  = f"match_{st.session_state.session_id}_{v_idx}_{int(t_sec * 100)}.jpg"
                    crop_path = os.path.join(st.session_state.screens_dir, img_name)
                    cv2.imwrite(crop_path, evidence)
                except Exception:
                    crop_path = ""

                all_events.append(MatchEvent(
                    t_sec=t_sec, face_score=face_score, pose_score=pose_score,
                    fused_score=fused, frame_index=frame_i,
                    screenshot_path=crop_path, video_name=video_name
                ))

                try:
                    rgb = bgr_to_rgb_safe(evidence)
                    if rgb is not None:
                        preview_bx.image(
                            rgb,
                            caption=f"Match @ {fmt_time(t_sec)} — Conf: {fused:.2f}",
                            use_container_width=True
                        )
                except Exception:
                    pass

            # Progress update every 10 frames
            if total_frames > 0 and frame_i % 10 == 0:
                prog_bar.progress(min(1.0, frame_i / max(1, total_frames)))

            frame_i += 1
            del frame, frame_small

            if frame_i % 50 == 0:
                gc.collect()

        if writer is not None:
            try:
                writer.release()
            except Exception:
                writer = None
        cap.release()
        gc.collect()

        # ISSUE 13: keep the annotated clip only if it is a real file;
        # ISSUE 16: upgrade mp4v → H.264 for browser previews when possible
        if (annot_path and os.path.exists(annot_path)
                and os.path.getsize(annot_path) > 1024):
            annotated_videos[video_name] = _maybe_transcode(annot_path, ffmpeg_bin)

    prog_bar.progress(1.0)
    time.sleep(0.3)
    prog_bar.empty()
    status_txt.success("✅ Analysis complete!")
    time.sleep(1)
    preview_bx.empty()
    status_txt.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.raw_events       = all_events
    st.session_state.annotated_videos = annotated_videos
    st.session_state.timeline_df      = group_events(all_events)
    build_highlight_reel()
    st.session_state.step             = 4
    st.rerun()


# ----------------------------
# 8. Results Step
# ----------------------------
def _render_match_card(row, i):
    """Render a single match card into the current Streamlit container."""
    conf = float(row.get('Best Confidence', 0) or 0)
    cls, label = confidence_badge(conf)
    st.markdown('<div class="match-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.markdown(
            f"**Match #{i}** &nbsp; "
            f'<span class="pill info">{str(row.get("Video", "") or "")[:20]}</span> '
            f'<span class="pill {cls}">{label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"⏱ **{row.get('Start Time', '')}** → {row.get('End Time', '')}  |  "
            f"{row.get('Duration', '')}")
        st.caption(
            f"Face: {float(row.get('Best Face', 0) or 0):.3f}  |  "
            f"Pose: {float(row.get('Best Pose', 0) or 0):.3f}"
        )
    with c2:
        shot = row.get('Screenshot', '')
        if shot and os.path.exists(str(shot)):
            try:
                img = cv2.imread(str(shot))
                rgb = bgr_to_rgb_safe(img)
                if rgb is not None:
                    st.image(rgb, use_container_width=True)
            except Exception:
                st.warning("Preview unavailable")
        else:
            st.info("No screenshot")
    with c3:
        if st.button("▶ Play", key=f"play_{i}", use_container_width=True):
            st.session_state.start_time_player = int(row.get('Start (sec)', 0) or 0)
            st.session_state.active_video_for_player = str(row.get('Video', ''))
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_results_step():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 📊 Step 4 — Evidence Report")

    df = st.session_state.timeline_df

    if df.empty:
        st.warning("No matches found with the current threshold. Try lowering the Detection Threshold in the sidebar.")
        if st.button("← Try Again"):
            st.session_state.step = 3
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ISSUE 14: richer summary metrics
    total_dur = 0.0
    if 'Duration (sec)' in df.columns:
        try:
            total_dur = float(df['Duration (sec)'].sum())
        except Exception:
            total_dur = 0.0
    best_conf = float(df['Best Confidence'].max())
    avg_conf  = float(df['Best Confidence'].mean())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Matches", len(df))
    m2.metric("Total Sighting Duration", f"{total_dur:.1f}s")
    m3.metric("Videos Scanned", df['Video'].nunique())
    m4.metric("Highest Confidence", f"{best_conf:.1%}")
    m5.metric("Average Confidence", f"{avg_conf:.1%}")

    st.markdown("---")

    tab_video, tab_list, tab_graph, tab_data = st.tabs(
        ["🎬 Tracked Video", "🖼️  Match Details", "📈  Confidence Graph", "📋  Raw Data"])

    annotated_videos = st.session_state.get('annotated_videos', {}) or {}
    reel_path        = st.session_state.get('highlight_reel_path', '')

    # ---- Tracked Video tab ----
    with tab_video:
        st.markdown("#### ▶️ Highlight Reel")
        if reel_path and os.path.exists(reel_path):
            try:
                st.video(reel_path)
            except Exception:
                st.warning("Preview unavailable (browser may not support the video codec).")
            st.caption("If the preview doesn't play, use the download button — "
                       "the file is valid; some browsers don't support this codec.")
            try:
                with open(reel_path, 'rb') as f:
                    reel_bytes = f.read()
                st.download_button("⬇️ Highlight Reel", reel_bytes,
                                   "highlight_reel.mp4", "video/mp4",
                                   key="reel_tab_dl", use_container_width=True)
            except Exception:
                pass
        else:
            st.info("Highlight reel unavailable for this run.")
        st.markdown("---")
        st.markdown("#### 📼 Tracked Videos (per source)")
        if annotated_videos:
            vcols = st.columns(2)
            for vi, (vname, vpath) in enumerate(annotated_videos.items()):
                with vcols[vi % 2]:
                    st.markdown(f"**{vname}**")
                    if os.path.exists(str(vpath)):
                        try:
                            st.video(str(vpath))
                        except Exception:
                            st.warning("Preview unavailable for this annotated video.")
                        st.caption("If the preview doesn't play, use the download button — "
                                   "the file is valid; some browsers don't support this codec.")
                        try:
                            with open(str(vpath), 'rb') as f:
                                st.download_button(
                                    "⬇️ Annotated Video", f.read(),
                                    os.path.basename(str(vpath)), "video/mp4",
                                    key=f"annot_tab_{vname}",
                                    use_container_width=True)
                        except Exception:
                            pass
                    else:
                        st.info("Annotated video missing.")
        else:
            st.info("Annotated videos were not produced for this run.")

    # ---- Match Details tab ----
    with tab_list:
        st.markdown("#### Match Timeline")
        if len(df) > 8:
            for i, row in df.iterrows():
                summary = (f"Match #{i + 1} — {str(row.get('Video', ''))[:24]} | "
                           f"{row.get('Start Time', '')} → {row.get('End Time', '')} "
                           f"({row.get('Duration', '')})")
                with st.expander(summary, expanded=False):
                    _render_match_card(row, i + 1)
        else:
            for i, row in df.iterrows():
                _render_match_card(row, i + 1)

    # ---- Confidence Graph tab ----
    with tab_graph:
        st.markdown("#### Confidence Over Time")
        chart_data = df.copy()
        chart_data["Seconds"] = chart_data["Start (sec)"]

        area = alt.Chart(chart_data).mark_area(
            line={'color': '#38bdf8', 'strokeWidth': 2},
            color=alt.Gradient(
                gradient='linear',
                stops=[
                    alt.GradientStop(color='rgba(56,189,248,0.5)', offset=0),
                    alt.GradientStop(color='rgba(56,189,248,0.02)', offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Seconds', axis=alt.Axis(title='Video Time (sec)', labelColor='#64748b')),
            y=alt.Y('Best Confidence', scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title='Confidence Score', labelColor='#64748b')),
            tooltip=['Start Time', 'Duration', 'Best Confidence', 'Video']
        )

        points = alt.Chart(chart_data).mark_circle(size=90, color='#38bdf8').encode(
            x='Seconds', y='Best Confidence',
            tooltip=['Start Time', 'Best Confidence']
        )

        st.altair_chart((area + points).configure_view(
            strokeOpacity=0
        ).configure(background='transparent').interactive(),
                        use_container_width=True)

    # ---- Raw Data tab ----
    with tab_data:
        st.markdown("#### Raw Data Table")
        st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # ISSUE 14: Downloads split into Reports + Video Evidence rows
    st.markdown("### 📄 Reports")
    r1, r2 = st.columns(2)
    with r1:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ CSV Report", csv_bytes, "report.csv", "text/csv",
                           use_container_width=True)
    with r2:
        if FPDF_AVAILABLE:
            pdf_bytes = generate_pdf_report(df, st.session_state.case_name)
            if pdf_bytes:
                st.download_button("⬇️ PDF Report", pdf_bytes, "forensic_report.pdf",
                                   "application/pdf", use_container_width=True)
            else:
                st.caption("PDF generation failed.")
        else:
            st.caption("fpdf2 not installed — PDF unavailable.")

    st.markdown("### 🎬 Video Evidence")
    v1, v2, v3 = st.columns(3)
    with v1:
        shots = [str(row.get('Screenshot', '')) for _, row in df.iterrows()]
        st.download_button("⬇️ Evidence ZIP", make_zip_of_files(shots),
                           "evidence.zip", "application/zip", use_container_width=True)
    with v2:
        annot_paths = [p for p in annotated_videos.values() if p and os.path.exists(p)]
        if annot_paths:
            st.download_button("⬇️ Annotated Video ZIP", make_zip_of_files(annot_paths),
                               "annotated_videos.zip", "application/zip",
                               key="annot_zip_dl", use_container_width=True)
        else:
            st.caption("No annotated videos available.")
    with v3:
        if reel_path and os.path.exists(reel_path):
            try:
                with open(reel_path, 'rb') as f:
                    reel_bytes = f.read()
                st.download_button("⬇️ Highlight Reel", reel_bytes,
                                   "highlight_reel.mp4", "video/mp4",
                                   key="reel_bottom_dl", use_container_width=True)
            except Exception:
                st.caption("Highlight reel unavailable.")
        else:
            st.caption("No highlight reel available.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Video Player — prefers annotated clip; annotated time == source time now,
    # so the seek position needs no stride mapping (Issue 15)
    if st.session_state.start_time_player > 0:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        active_name = st.session_state.active_video_for_player
        st.markdown(
            f"#### ▶ Playback — `{active_name}` "
            f"@ {fmt_time(st.session_state.start_time_player)}"
        )

        vid_path = None
        seek_sec = st.session_state.start_time_player
        annotated_path = annotated_videos.get(active_name, "")
        if annotated_path and os.path.exists(annotated_path):
            vid_path = annotated_path
        elif active_name in st.session_state.video_names:
            idx = st.session_state.video_names.index(active_name)
            vid_path = st.session_state.video_files[idx]
        elif st.session_state.single_video_path:
            vid_path = st.session_state.single_video_path

        if vid_path and os.path.exists(vid_path):
            try:
                st.video(vid_path, start_time=seek_sec)
            except Exception:
                st.warning("Playback unavailable (codec not supported by browser).")
            st.caption("If the preview doesn't play, use the download button — "
                       "the file is valid; some browsers don't support this codec.")
        else:
            st.error("Video file not found in session — it may have been cleaned up.")

        if st.button("✕ Close Player", use_container_width=False):
            st.session_state.start_time_player = 0
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Start New Analysis", use_container_width=False):
        cleanup_case_artifacts()
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.screens_dir = ensure_dir(
            os.path.join(tempfile.gettempdir(), "target_id_screens",
                         st.session_state.session_id))
        st.rerun()


# ----------------------------
# 9. Entry Point
# ----------------------------
def main_app():
    st.markdown(
        """
        <div class="topbar">
            <div>
                <div class="brand-title">🎯 Video Target ID</div>
                <div class="brand-sub">AI-Powered Forensic Identification System</div>
            </div>
            <div style="display:flex; gap:10px; align-items:center;">
                <span class="pill info">v2.0</span>
                <span class="pill ok">● Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    step_map = {
        1: render_target_step,
        2: render_source_step,
        3: render_scan_step,
        4: render_results_step,
    }
    step_map.get(st.session_state.step, render_target_step)()


inject_pro_ui()
main_app()