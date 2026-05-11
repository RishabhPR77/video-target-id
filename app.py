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
import zipfile
import tempfile
import gc
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

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


def cosine_sim_np(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    try:
        a = a.flatten().astype(np.float32)
        b = b.flatten().astype(np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)
    except Exception:
        return 0.0


def pick_best_face(faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not faces:
        return None
    return max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))


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
                                     "Best Confidence", "Best Face", "Best Pose",
                                     "Start (sec)", "Screenshot"])

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
            "Best Confidence":  float(best.fused_score),
            "Best Face":        float(best.face_score),
            "Best Pose":        float(best.pose_score),
            "Screenshot":       best.screenshot_path or "",
        })
    return pd.DataFrame(rows)


def make_zip_of_screenshots(paths: List[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p and os.path.exists(p):
                zf.write(p, arcname=os.path.basename(p))
    return buf.getvalue()


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
    'start_time_player': 0,
    'active_video_for_player': "",
    'threshold': 0.55,
    'skip_frames': 5,
    'process_width': "Medium (640px)",
    'face_weight': 0.70,
    'pose_weight': 0.30,
    'consent_ok': False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if 'screens_dir' not in st.session_state:
    st.session_state.screens_dir = ensure_dir(
        os.path.join(tempfile.gettempdir(), "target_id_screens"))


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

        st.session_state.face_weight = st.slider(
            "Face Weight", 0.0, 1.0,
            float(st.session_state.face_weight), 0.05,
            help="Weight given to face similarity vs pose"
        )
        st.session_state.pose_weight = round(1.0 - st.session_state.face_weight, 2)
        st.caption(f"Pose Weight: **{st.session_state.pose_weight:.2f}** (auto)")

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
                for k, v in _DEFAULTS.items():
                    st.session_state[k] = v
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
                            use_column_width=True   # FIX #1: use_column_width for max compatibility
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
                            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
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

    target_width = 320
    if "640" in st.session_state.process_width:
        target_width = 640
    elif "Native" in st.session_state.process_width:
        target_width = 1280  # capped to avoid OOM on free tiers

    for v_idx, video_path in enumerate(st.session_state.video_files):
        video_name = st.session_state.video_names[v_idx]
        status_txt.markdown(f"**Scanning:** `{video_name}` ({v_idx + 1}/{len(st.session_state.video_files)})")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.warning(f"Cannot open video: {video_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_i      = 0
        seen_last    = -999.0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            # Skip frames
            if st.session_state.skip_frames > 0 and (frame_i % (st.session_state.skip_frames + 1) != 0):
                frame_i += 1
                continue

            if frame is None or frame.size == 0:
                frame_i += 1
                continue

            h, w = frame.shape[:2]

            # Smart resize
            if w > target_width:
                scale       = target_width / float(w)
                frame_small = cv2.resize(frame, (target_width, int(h * scale)))
            else:
                frame_small = frame.copy()

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

            # Pose scoring
            pose_score = 0.0
            if st.session_state.ref_pose is not None:
                try:
                    pf = extract_pose_feats_bgr(frame_small)
                    if pf is not None:
                        pose_score = cosine_sim_np(pf, st.session_state.ref_pose)
                except Exception:
                    pose_score = 0.0

            # Fusion
            fused = (st.session_state.face_weight * face_score +
                     st.session_state.pose_weight * pose_score)

            t_sec = frame_i / fps

            if fused >= st.session_state.threshold and (t_sec - seen_last) >= 0.5:
                seen_last = t_sec

                # Draw bounding box on evidence frame
                evidence = frame_small.copy()
                # FIX #2: Only draw & call cosine_sim when ref_face is not None
                if best_face is not None and st.session_state.ref_face is not None:
                    try:
                        x1, y1, x2, y2 = map(int, best_face['bbox'])
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(evidence.shape[1] - 1, x2)
                        y2 = min(evidence.shape[0] - 1, y2)
                        cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 220, 80), 2)
                        label = f"Conf: {fused:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(evidence, (x1, y1 - 22), (x1 + tw + 4, y1), (0, 220, 80), -1)
                        cv2.putText(evidence, label, (x1 + 2, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    except Exception:
                        pass

                # Save crop
                crop_path = ""
                try:
                    img_name  = f"match_{v_idx}_{int(t_sec * 100)}.jpg"
                    crop_path = os.path.join(st.session_state.screens_dir, img_name)
                    cv2.imwrite(crop_path, evidence)
                except Exception:
                    crop_path = ""

                all_events.append(MatchEvent(
                    t_sec=t_sec, face_score=face_score, pose_score=pose_score,
                    fused_score=fused, frame_index=frame_i,
                    screenshot_path=crop_path, video_name=video_name
                ))

                # FIX #3: Safe image preview during scan
                try:
                    rgb = bgr_to_rgb_safe(evidence)
                    if rgb is not None:
                        preview_bx.image(
                            rgb,
                            caption=f"Match @ {fmt_time(t_sec)} — Conf: {fused:.2f}",
                            use_column_width=True
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

        cap.release()
        gc.collect()

    prog_bar.progress(1.0)
    time.sleep(0.3)
    prog_bar.empty()
    status_txt.success("✅ Analysis complete!")
    time.sleep(1)
    preview_bx.empty()
    status_txt.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.raw_events   = all_events
    st.session_state.timeline_df  = group_events(all_events)
    st.session_state.step         = 4
    st.rerun()


# ----------------------------
# 8. Results Step
# ----------------------------
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

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Matches", len(df))
    best_conf = df['Best Confidence'].max()
    m2.metric("Highest Confidence", f"{best_conf:.1%}")
    m3.metric("Videos Scanned", df['Video'].nunique())

    st.markdown("---")

    tab_list, tab_graph, tab_data = st.tabs(
        ["🖼️  Match Details", "📈  Confidence Graph", "📋  Raw Data"])

    with tab_list:
        st.markdown("#### Match Timeline")
        for i, row in df.iterrows():
            st.markdown('<div class="match-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                conf = float(row.get('Best Confidence', 0))
                conf_color = "#4ade80" if conf >= 0.75 else ("#fbbf24" if conf >= 0.55 else "#f87171")
                st.markdown(
                    f"**Match #{i + 1}** &nbsp;"
                    f'<span class="pill info">{row["Video"][:20]}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(f"⏱ **{row['Start Time']}** → {row['End Time']}  |  {row['Duration']}")
                st.markdown(
                    f'Confidence: <span style="color:{conf_color}; font-weight:700;">'
                    f'{conf:.3f}</span>',
                    unsafe_allow_html=True
                )
                st.caption(
                    f"Face: {float(row.get('Best Face', 0)):.3f}  |  "
                    f"Pose: {float(row.get('Best Pose', 0)):.3f}"
                )
            with c2:
                shot = row.get('Screenshot', '')
                if shot and os.path.exists(str(shot)):
                    try:
                        img = cv2.imread(str(shot))
                        rgb = bgr_to_rgb_safe(img)
                        if rgb is not None:
                            st.image(rgb, use_column_width=True)
                    except Exception:
                        st.warning("Preview unavailable")
                else:
                    st.info("No screenshot")
            with c3:
                if st.button(f"▶ Play", key=f"play_{i}", use_container_width=True):
                    st.session_state.start_time_player = int(row.get('Start (sec)', 0))
                    st.session_state.active_video_for_player = row['Video']
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

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

    with tab_data:
        st.markdown("#### Raw Data Table")
        st.dataframe(df, use_container_width=True)

    st.markdown("---")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ CSV Report", csv_bytes, "report.csv", "text/csv",
                           use_container_width=True)
    with dl2:
        shots = [str(row.get('Screenshot', '')) for _, row in df.iterrows()]
        st.download_button("⬇️ Evidence ZIP", make_zip_of_screenshots(shots),
                           "evidence.zip", "application/zip", use_container_width=True)
    with dl3:
        if FPDF_AVAILABLE:
            pdf_bytes = generate_pdf_report(df, st.session_state.case_name)
            if pdf_bytes:
                st.download_button("⬇️ PDF Report", pdf_bytes, "forensic_report.pdf",
                                   "application/pdf", use_container_width=True)
            else:
                st.caption("PDF generation failed.")
        else:
            st.caption("fpdf2 not installed — PDF unavailable.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Video Player
    if st.session_state.start_time_player > 0:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(
            f"#### ▶ Playback — `{st.session_state.active_video_for_player}` "
            f"@ {fmt_time(st.session_state.start_time_player)}"
        )

        vid_path = None
        if st.session_state.active_video_for_player in st.session_state.video_names:
            idx      = st.session_state.video_names.index(st.session_state.active_video_for_player)
            vid_path = st.session_state.video_files[idx]
        elif st.session_state.single_video_path:
            vid_path = st.session_state.single_video_path

        if vid_path and os.path.exists(vid_path):
            st.video(vid_path, start_time=st.session_state.start_time_player)
        else:
            st.error("Video file not found in session — it may have been cleaned up.")

        if st.button("✕ Close Player", use_container_width=False):
            st.session_state.start_time_player = 0
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Start New Analysis", use_container_width=False):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
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