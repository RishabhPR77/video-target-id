# app.py — Final Deployment Version (Enhanced Guide)
# ------------------------------------------------------------
# Features:
# ✅ Secure Login/Register System (SQLite)
# ✅ Pro Glassmorphism UI with "Outfit" Font
# ✅ AI Video Analysis (Face + Pose)
# ✅ Auto-Screenshot with Bounding Box Highlighting
# ✅ Cloud-Ready (CPU Optimized)
# ✅ Detailed Performance Guide (4 Tiers)

import os
import io
import time
import base64
import zipfile
import tempfile
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from face_module import init_face_app, get_faces, mean_normalize_stack, cosine_sim
from pose_module import extract_pose_feats_bgr

# ----------------------------
# 1. Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Video Target ID System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------
# 2. Database & Auth Functions
# ----------------------------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False


def create_usertable():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Schema includes name
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT PRIMARY KEY, password TEXT, name TEXT)')
    conn.commit()
    conn.close()


def add_userdata(username, password, name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO userstable(username,password,name) VALUES (?,?,?)', (username, password, name))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success


def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data


# ----------------------------
# 3. Custom CSS (Premium UI)
# ----------------------------
def inject_pro_ui():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
        }

        .stApp {
            background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 100%);
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.95);
            border-right: 1px solid rgba(255,255,255,0.05);
        }

        .glass {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(12px);
            margin-bottom: 20px;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 30px;
            background: linear-gradient(90deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            margin-bottom: 24px;
            backdrop-filter: blur(10px);
        }

        .brand h1 {
            margin: 0;
            font-size: 32px !important;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: -webkit-linear-gradient(45deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .kpi {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .kpi-title { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .kpi-val { font-size: 24px; font-weight: 700; color: #f1f5f9; margin-top: 4px; }

        .stButton>button {
            width: 100%;
            border-radius: 8px !important;
            font-weight: 600;
            height: 45px;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border: none;
            color: white;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }

        .pill {
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }
        .pill.ok { background: rgba(34, 197, 94, 0.1); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.2); }
        .pill.warn { background: rgba(234, 179, 8, 0.1); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.2); }
        .pill.info { background: rgba(59, 130, 246, 0.1); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.2); }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# 4. Helper Functions
# ----------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"


def safe_imdecode(file_bytes: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def cosine_sim_np(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def pick_best_face(faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not faces: return None
    return sorted(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]), reverse=True)[0]


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
        return pd.DataFrame(columns=["Video", "Start Time", "Duration", "Best Confidence", "Screenshot"])

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
            "Video": v,
            "Start (sec)": float(start),
            "Start Time": fmt_time(start),
            "End Time": fmt_time(end),
            "Duration": f"{end - start:.1f}s",
            "Best Confidence": float(best.fused_score),
            "Best Face": float(best.face_score),
            "Best Pose": float(best.pose_score),
            "Screenshot": best.screenshot_path or "",
        })
    return pd.DataFrame(rows)


def make_zip_of_screenshots(paths: List[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p and os.path.exists(p):
                zf.write(p, arcname=os.path.basename(p))
    return buf.getvalue()


@st.cache_resource
def load_face_engine():
    return init_face_app()


# ----------------------------
# 5. Session State Init
# ----------------------------
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ""
if 'realname' not in st.session_state: st.session_state.realname = ""
if 'step' not in st.session_state: st.session_state.step = 1
if 'case_name' not in st.session_state: st.session_state.case_name = "New Investigation"
if 'target_files' not in st.session_state: st.session_state.target_files = []
if 'ref_face' not in st.session_state: st.session_state.ref_face = None
if 'ref_pose' not in st.session_state: st.session_state.ref_pose = None
if 'video_files' not in st.session_state: st.session_state.video_files = []
if 'video_names' not in st.session_state: st.session_state.video_names = []
if 'single_video_path' not in st.session_state: st.session_state.single_video_path = None
if 'raw_events' not in st.session_state: st.session_state.raw_events = []
if 'timeline_df' not in st.session_state: st.session_state.timeline_df = pd.DataFrame()
if 'screens_dir' not in st.session_state:
    st.session_state.screens_dir = ensure_dir(os.path.join(tempfile.gettempdir(), "target_id_screens"))
if 'start_time_player' not in st.session_state: st.session_state.start_time_player = 0
if 'active_video_for_player' not in st.session_state: st.session_state.active_video_for_player = ""

if 'threshold' not in st.session_state: st.session_state.threshold = 0.50
if 'skip_frames' not in st.session_state: st.session_state.skip_frames = 5
if 'face_weight' not in st.session_state: st.session_state.face_weight = 0.7
if 'pose_weight' not in st.session_state: st.session_state.pose_weight = 0.3
if 'consent_ok' not in st.session_state: st.session_state.consent_ok = False


# ----------------------------
# 6. Auth Screens
# ----------------------------
def login_screen():
    st.markdown("<div style='height: 50px'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div class='glass' style='text-align: center;'>
                <h1 style='color: #60a5fa; margin-bottom: 10px; font-size: 32px;'>Video Target ID</h1>
                <p style='color: #94a3b8;'>Secure Forensic Analysis Platform</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        tab_login, tab_reg = st.tabs(["Login", "Create Account"])

        with tab_login:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type='password', key="login_pass")
            if st.button("Log In"):
                create_usertable()
                hashed_pswd = make_hashes(password)
                result = login_user(username, hashed_pswd)
                if result:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.realname = result[0][2]
                    st.success(f"Welcome back, {st.session_state.realname}")
                    st.rerun()
                else:
                    st.error("Incorrect Username/Password")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_reg:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            new_name = st.text_input("Full Name", key="reg_name")
            new_user = st.text_input("New Username", key="reg_user")
            new_password = st.text_input("New Password", type='password', key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type='password', key="reg_conf")

            if st.button("Create Account"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif not new_user or not new_password or not new_name:
                    st.error("Please fill all fields")
                else:
                    create_usertable()
                    if add_userdata(new_user, make_hashes(new_password), new_name):
                        st.success("Account created! Please log in.")
                    else:
                        st.error("Username already exists.")
            st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# 7. Main Application Logic
# ----------------------------
def main_app():
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">
                <h1>Video Target Identification</h1>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <span class="pill info">👤 {st.session_state.realname}</span>
                <span class="pill ok">Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("### Control Panel")
        steps = ["Target", "Source", "Scan", "Results"]
        st.markdown('<div class="glass" style="padding:15px;">', unsafe_allow_html=True)
        for i, s in enumerate(steps, 1):
            icon = "🔷" if i == st.session_state.step else ("✅" if i < st.session_state.step else "⚪")
            st.markdown(f"**{icon} Step {i}: {s}**")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    if st.session_state.step == 1:
        render_target_step()
    elif st.session_state.step == 2:
        render_source_step()
    elif st.session_state.step == 3:
        render_scan_step()
    elif st.session_state.step == 4:
        render_results_step()


def render_target_step():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Step 1: Who to find?")
    st.info("Upload 1-3 clear reference photos of the target person.")

    files = st.file_uploader("Drop images here (JPG/PNG)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True,
                             key="ref_uploader")

    if files:
        st.write("#### Preview")
        cols = st.columns(min(5, len(files)))
        for i, f in enumerate(files[:5]):
            img = safe_imdecode(f.getvalue())
            if img is not None:
                cols[i].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.session_state.target_files = files

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Build Reference"):
            if not st.session_state.target_files:
                st.error("Upload images first.")
            else:
                with st.spinner("Processing biometric data..."):
                    face_embs = []
                    pose_embs = []
                    app = load_face_engine()
                    found_count = 0
                    for f in st.session_state.target_files:
                        f.seek(0)
                        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, 1)
                        if img is None: continue
                        faces = get_faces(img, app)
                        best = pick_best_face(faces)
                        if best:
                            face_embs.append(best['emb'])
                            found_count += 1
                        pf = extract_pose_feats_bgr(img)
                        if pf is not None: pose_embs.append(pf)

                    if not face_embs:
                        st.error("No faces found. Try clearer photos.")
                    else:
                        st.session_state.ref_face = mean_normalize_stack(face_embs)
                        st.session_state.ref_pose = np.mean(np.stack(pose_embs), axis=0) if pose_embs else None
                        st.success(f"Reference Built! (Faces used: {found_count})")

    with col2:
        if st.button("Save Profile"):
            if st.session_state.ref_face is None:
                st.warning("Build the reference first.")
            else:
                tmp = io.BytesIO()
                np.savez(tmp, ref_face=st.session_state.ref_face,
                         ref_pose=st.session_state.ref_pose if st.session_state.ref_pose is not None else np.array([]))
                st.download_button("Download .npz", data=tmp.getvalue(), file_name="target_profile.npz",
                                   mime="application/octet-stream")

    st.markdown("---")
    st.session_state.consent_ok = st.checkbox("I confirm I am authorized to process this data.")

    col_nav_1, col_nav_2 = st.columns([4, 1])
    with col_nav_2:
        if st.button("Next"):
            if st.session_state.ref_face is None:
                st.error("Please click 'Build Reference' first.")
                return
            if not st.session_state.consent_ok:
                st.error("Please check the authorization box.")
                return
            st.session_state.step = 2
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_source_step():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Step 2: Where to look?")
    st.info("Upload the surveillance video file(s) you want to analyze.")

    uploaded = st.file_uploader("Upload Video File (MP4/AVI)", type=['mp4', 'avi', 'mov', 'mkv'],
                                accept_multiple_files=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Next"):
            if not uploaded and not st.session_state.video_files:
                st.error("Please upload a video.")
                return
            if uploaded:
                paths = []
                names = []
                for up in uploaded:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(up.getvalue())
                    tfile.close()
                    paths.append(tfile.name)
                    names.append(up.name)
                st.session_state.video_files = paths
                st.session_state.video_names = names
                st.session_state.single_video_path = paths[0]
            st.session_state.step = 3
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_scan_step():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Step 3: AI Deep Scan")

    # --- UPDATED PERFORMANCE GUIDE TABLE ---
    with st.expander("ℹ️ Performance Guide (Important)"):
        st.markdown("""
        **Adjust Frame Skipping based on your video length:**

        | Video Length | Recommended Skip | Speed | Accuracy |
        | :--- | :--- | :--- | :--- |
        | **Short (< 2 min)** | **0 - 5** | Normal | ⭐⭐⭐⭐⭐ (Max) |
        | **Medium (2 - 10 min)** | **5 - 15** | Fast | ⭐⭐⭐⭐ (High) |
        | **Long (10 - 30 min)** | **15 - 30** | Very Fast | ⭐⭐⭐ (Good) |
        | **Archive (30+ min)** | **30 - 60** | Turbo | ⭐⭐ (Basic) |

        *Tip: Higher skip values make the scan faster but might miss a face that appears for only a split second.*
        """)

    st.session_state.skip_frames = st.slider("Frame Skipping", 0, 60, 5)

    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Start Analysis"):
            run_analysis()
    st.markdown('</div>', unsafe_allow_html=True)


def run_analysis():
    app = load_face_engine()
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("### Processing Video...")

    prog_bar = st.progress(0)
    status_text = st.empty()
    preview_box = st.empty()

    all_events = []
    RESIZE_WIDTH = 320

    for v_idx, video_path in enumerate(st.session_state.video_files):
        video_name = st.session_state.video_names[v_idx]
        status_text.write(f"Scanning: **{video_name}**")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        frame_i = 0
        seen_match_last = -999

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break

            if st.session_state.skip_frames > 0 and (frame_i % (st.session_state.skip_frames + 1) != 0):
                frame_i += 1
                continue

            h, w = frame.shape[:2]
            scale = RESIZE_WIDTH / float(w)
            frame_small = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

            faces = get_faces(frame_small, app)
            face_score = 0.0

            if faces and st.session_state.ref_face is not None:
                try:
                    scores = [cosine_sim(f['emb'], st.session_state.ref_face) for f in faces]
                    face_score = max(scores) if scores else 0.0
                except:
                    face_score = 0.0

            pose_score = 0.0
            if st.session_state.ref_pose is not None:
                p = extract_pose_feats_bgr(frame_small)
                if p is not None:
                    pose_score = cosine_sim_np(p, st.session_state.ref_pose)

            fused = (st.session_state.face_weight * face_score) + (st.session_state.pose_weight * pose_score)

            if fused >= st.session_state.threshold:
                t_sec = frame_i / fps

                if (t_sec - seen_match_last) >= 0.5:
                    seen_match_last = t_sec

                    evidence_frame = frame_small.copy()
                    if faces:
                        best_face = max(faces, key=lambda f: cosine_sim(f['emb'], st.session_state.ref_face))
                        x1, y1, x2, y2 = map(int, best_face['bbox'])
                        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Conf: {fused:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(evidence_frame, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
                        cv2.putText(evidence_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    img_name = f"match_{v_idx}_{int(t_sec * 100)}.jpg"
                    save_path = os.path.join(st.session_state.screens_dir, img_name)
                    cv2.imwrite(save_path, evidence_frame)

                    all_events.append(MatchEvent(
                        t_sec=t_sec, face_score=face_score, pose_score=pose_score,
                        fused_score=fused, frame_index=frame_i,
                        screenshot_path=save_path, video_name=video_name
                    ))

                    preview_box.image(cv2.cvtColor(evidence_frame, cv2.COLOR_BGR2RGB),
                                      caption=f"Match Found: {fmt_time(t_sec)}")

            if total_frames > 0:
                prog_bar.progress(min(1.0, frame_i / total_frames))
            frame_i += 1

        cap.release()

    st.session_state.raw_events = all_events
    st.session_state.timeline_df = group_events(all_events)
    st.session_state.step = 4
    st.rerun()


def render_results_step():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Step 4: Evidence Report")

    df = st.session_state.timeline_df

    if df.empty:
        st.warning("No matches found.")
        if st.button("Start New Case"):
            st.session_state.step = 1
            st.session_state.target_files = []
            st.session_state.video_files = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    m1, m2 = st.columns(2)
    m1.metric("Total Matches", len(df))
    m2.metric("Highest Confidence", f"{df['Best Confidence'].max():.0%}")

    st.markdown("---")

    # --- Improved Graph: Area Chart ---
    st.markdown("#### Confidence Timeline")
    chart_data = df.copy()
    chart_data["Seconds"] = chart_data["Start (sec)"]

    c = alt.Chart(chart_data).mark_area(
        line={'color': '#4ade80'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='#4ade80', offset=0),
                   alt.GradientStop(color='rgba(74, 222, 128, 0.1)', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('Seconds', axis=alt.Axis(title='Video Time (sec)')),
        y=alt.Y('Best Confidence', scale=alt.Scale(domain=[0, 1])),
        tooltip=['Start Time', 'Duration', 'Best Confidence']
    ).interactive()

    points = alt.Chart(chart_data).mark_circle(size=80, color='white').encode(
        x='Seconds',
        y='Best Confidence',
        tooltip=['Start Time', 'Best Confidence']
    )

    st.altair_chart(c + points, use_container_width=True)

    st.markdown("---")

    for i, row in df.iterrows():
        with st.container():
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                st.markdown(f"#### Match #{i + 1}")
                st.write(f"**Time:** {row['Start Time']}")
                st.write(f"**Conf:** {row['Best Confidence']:.2f}")
            with c2:
                if row['Screenshot'] and os.path.exists(row['Screenshot']):
                    st.image(row['Screenshot'], width=300)
            with c3:
                # Explicit Button
                if st.button(f"▶ Jump to {row['Start Time']}", key=f"play_{i}"):
                    st.session_state.start_time_player = int(row['Start (sec)'])
                    st.session_state.active_video_for_player = row['Video']
                    st.rerun()
            st.divider()

    cols = st.columns(2)
    with cols[0]:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "report.csv", "text/csv")
    with cols[1]:
        img_paths = [row['Screenshot'] for _, row in df.iterrows() if row['Screenshot']]
        zip_data = make_zip_of_screenshots(img_paths)
        st.download_button("Download ZIP", zip_data, "evidence.zip", "application/zip")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.start_time_player > 0:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📺 Video Playback")
        vid_path = None
        if st.session_state.active_video_for_player in st.session_state.video_names:
            idx = st.session_state.video_names.index(st.session_state.active_video_for_player)
            vid_path = st.session_state.video_files[idx]
        elif st.session_state.single_video_path:
            vid_path = st.session_state.single_video_path

        if vid_path:
            # Unique key forces player reload
            st.video(vid_path, start_time=st.session_state.start_time_player)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("New Analysis"):
        st.session_state.step = 1
        st.rerun()


# ----------------------------
# 8. Entry Point
# ----------------------------
inject_pro_ui()
create_usertable()

if st.session_state.logged_in:
    main_app()
else:
    login_screen()