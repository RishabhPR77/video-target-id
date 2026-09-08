"""Core ML pipeline for the Video Target ID backend.

Streamlit-free port of the analysis engine previously embedded in `app.py`.
All functions here are importable without touching the UI framework; the
FastAPI layer (`backend/main.py`) wraps them behind REST endpoints.
"""

import functools
import gc
import io
import os
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import FACE_THR, CONSEC, COOLDOWN, W_FACE, W_CLOTH, CLOTH_WEIGHT_CAP, REACQUIRE_GAP, MIN_SIGHTING_SECS, MAX_EVIDENCE_SHOTS
from .face_module import cosine_sim, get_faces, init_face_app, mean_normalize_stack
from .pose_module import extract_pose_feats_bgr

PAD = 2.0
PROBE_CAP = 1280


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = sec // 60, sec % 60
    return f"{m:02d}:{s:02d}"


def safe_imdecode(file_bytes: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(file_bytes, np.uint8)
        if arr.size == 0:
            return None
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if img.ndim != 3 or img.shape[2] != 3:
            return None
        if img.shape[0] < 4 or img.shape[1] < 4:
            return None
        return img.astype(np.uint8)
    except Exception:
        return None


def pick_best_face(faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not faces:
        return None
    return max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))


def sanitize_filename(name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in " ._-") else "_" for c in name)
    return safe.strip() or "video"


def draw_match_annotation(frame: np.ndarray, bbox, fused_score: float) -> None:
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
        cv2.putText(frame, label, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    except Exception:
        pass


def clothing_descriptor(img_bgr: np.ndarray, face_bbox) -> Optional[np.ndarray]:
    """HSV histogram over the upper-body region below a detected face.

    Returns an L2-normalized (512,) descriptor of the clothing area so it can
    be compared with the same cosine similarity used for face/pose. None when
    the region can't be localized.
    """
    try:
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, face_bbox)
        fx = max(1, x2 - x1)
        fy = max(1, y2 - y1)
        rx1 = max(0, int(x1 - 0.20 * fx))
        rx2 = min(w - 1, int(x2 + 0.20 * fx))
        ry1 = max(0, int(y2))
        ry2 = min(h - 1, int(y2 + 2.0 * fy))
        if (rx2 - rx1) < 4 or (ry2 - ry1) < 4:
            return None
        roi = img_bgr[ry1:ry2 + 1, rx1:rx2 + 1]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256]).reshape(-1).astype(np.float32)
        norm = np.linalg.norm(hist) + 1e-9
        if norm <= 0:
            return None
        return (hist / norm).astype(np.float32)
    except Exception:
        return None


def clothing_color_labels(hist: Optional[np.ndarray]) -> List[str]:
    """Dominant upper-body colors from an HSV histogram (for report/exports)."""
    try:
        if hist is None or hist.size == 0:
            return []
        flat = hist.reshape(-1)
        idx = np.argsort(flat)[::-1][:2]
        labels: List[str] = []
        for i in idx:
            h = (i // 64) * 180 // 8
            rem = i % 64
            s = (rem // 8) * 256 // 8
            v = (rem % 8) * 256 // 8
            if s < 24:
                name = "black" if v < 96 else ("white" if v > 170 else "grey")
            else:
                hh = (h + 90) % 360
                name = ("red" if hh < 10 else "orange" if hh < 40 else
                        "yellow" if hh < 70 else "green" if hh < 155 else
                        "cyan" if hh < 200 else "blue" if hh < 260 else
                        "purple" if hh < 310 else "pink")
                name = f"light {name}" if v > 170 else f"dark {name}" if v < 96 else name
            if not labels or labels[-1] != name:
                labels.append(name)
        return labels[:2] if labels else ["unknown"]
    except Exception:
        return []


@functools.lru_cache(maxsize=1)
def _find_ffmpeg() -> Optional[str]:
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
    try:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, "HIGHLIGHT REEL", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
        cv2.putText(frame, str(video_name)[:40], (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 232, 240), 1)
        cv2.putText(frame, f"Match @ {fmt_time(src_start_sec)}", (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
        n = max(1, int(round(fps)))
        for _ in range(n):
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    except Exception:
        pass


def _write_reel_clip_fallback(cap, start_f, end_f, width, height, writer):
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
    """Write clip frames [start_f..end_f] into the reel at reel timing."""
    start_f = max(0, int(start_f))
    end_f = max(start_f, int(end_f))
    clip_fps = float(clip_fps)
    out_fps = float(out_fps)

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

        n_out = max(1, int(round((end_f - start_f + 1) * out_fps / clip_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        cur = start_f - 1
        last_frm = None
        for out_i in range(n_out):
            target = start_f + min(end_f - start_f, int(round(out_i * clip_fps / out_fps)))
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


@dataclass
class MatchEvent:
    t_sec: float
    face_score: float
    pose_score: float
    fused_score: float
    frame_index: int
    video_id: str = ""
    video_name: str = ""
    screenshot_name: str = ""
    cloth_score: float = 0.0
    bbox: Optional[Tuple[float, float, float, float]] = None


def group_events(events: List[MatchEvent], merge_gap_sec: float = 2.0) -> List[Dict[str, Any]]:
    """Merge raw per-frame events into sighting windows."""
    if not events:
        return []
    events = sorted(events, key=lambda e: (e.video_name, e.video_id, e.t_sec))
    rows: List[Dict[str, Any]] = []
    i = 0
    while i < len(events):
        v = events[i].video_name
        vid = events[i].video_id
        start = events[i].t_sec
        end = events[i].t_sec
        block = [events[i]]
        i += 1
        while (i < len(events) and events[i].video_name == v and events[i].video_id == vid
               and (events[i].t_sec - end) <= merge_gap_sec):
            end = events[i].t_sec
            block.append(events[i])
            i += 1
        best = max(block, key=lambda e: e.fused_score)
        rows.append({
            "id": f"{vid}-{int(start)}",
            "video_id": vid,
            "video_name": v,
            "start_sec": float(start),
            "end_sec": float(end),
            "face": float(best.face_score),
            "pose": float(best.pose_score),
            "cloth": float(best.cloth_score),
            "fused": float(best.fused_score),
            "screenshot": best.screenshot_name or None,
            "annotated_video": None,
            "bbox": best.bbox,
        })
    return rows


def build_confidence_series(matches: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Deterministic 60-point confidence timeline over a 10-minute window."""
    series: List[Dict[str, float]] = []
    for i in range(60):
        t = i * 10
        near = None
        for m in matches:
            if abs((m["start_sec"] % 600) - t) < 20:
                if near is None or m["fused"] > near["fused"]:
                    near = m
        base = 0.25 + ((i * 31) % 89) / 500.0
        conf = min(0.99, near["fused"] if near is not None else base)
        series.append({"t": t, "confidence": round(float(conf), 3)})
    return series


@functools.lru_cache(maxsize=1)
def get_face_app():
    """Load the InsightFace engine once, cached for the process lifetime."""
    return init_face_app()


class ScanSession:
    """Mutable state for one scan, read/written from background threads."""

    def __init__(self, scan_id: str, videos: List[Dict[str, Any]], workdir: str,
                 threshold: float, face_weight: float = W_FACE,
                 skip_frames: int = 2, process_width: str = "Medium (640px)"):
        self.scan_id = scan_id
        self.videos = videos
        self.workdir = workdir
        self.threshold = float(threshold)
        self.face_weight = float(face_weight)
        self.cloth_weight = round(min(CLOTH_WEIGHT_CAP, max(0.0, 1.0 - self.face_weight)), 2)
        self.pose_weight = round(max(0.0, 1.0 - self.face_weight - self.cloth_weight), 2)
        self.skip_frames = int(skip_frames)
        self.process_width = process_width
        self.lock = threading.RLock()
        self.status = "queued"
        self.error: Optional[str] = None
        self.percent = 0
        self.phase = "queued"
        self.current_video_name: Optional[str] = None
        self.current_index = 0
        self.total_videos = len(videos)
        self.cancel_requested = False
        self.events: List[MatchEvent] = []
        self.matches: List[Dict[str, Any]] = []
        self.confidence_series: List[Dict[str, float]] = []
        self.annotated_files: Dict[str, str] = {}
        self.reel_file: Optional[str] = None
        self.refined_face = None
        self.refined_pose = None
        self.refined_cloth = None
        self.color_labels: List[str] = []
        self.frames_processed = 0
        self.frames_with_face = 0
        self.best_face_sim = 0.0
        self.best_fused = 0.0


def _target_width(process_width: str) -> int:
    if "Low" in process_width or "320" in process_width:
        return 320
    if "High" in process_width or "Native" in process_width:
        return PROBE_CAP
    return 640


def run_scan(session: ScanSession, ref_face, ref_pose, ref_cloth) -> None:
    """Background worker: run the full scan and populate the session."""
    if ref_face is None:
        session.error = "Build a reference profile before scanning."
        session.status = "error"
        return
    with session.lock:
        session.status = "running"
        session.phase = "Loading face models…"
        session.refined_face = ref_face
        session.refined_pose = ref_pose
        session.refined_cloth = ref_cloth

    annot_dir = ensure_dir(os.path.join(session.workdir, "annotated"))
    ffmpeg_bin = _find_ffmpeg()
    target_width = _target_width(session.process_width)

    try:
        app = get_face_app()
    except Exception as e:
        with session.lock:
            session.error = f"Face engine failed to load: {e}"
            session.status = "error"
        return
    with session.lock:
        session.phase = "Ready — starting scan"

    events: List[MatchEvent] = []
    annotated_files: Dict[str, str] = {}
    evidence_candidates: List[Tuple[float, str]] = []

    for v_idx, video in enumerate(session.videos):
        with session.lock:
            if session.cancel_requested:
                break
            session.current_index = v_idx + 1
            session.current_video_name = video["name"]
            session.percent = int(v_idx / max(1, session.total_videos) * 100)
            session.phase = f"Opening {video['name']}…"
            cur_ref_face = session.refined_face
            cur_ref_pose = session.refined_pose
            cur_ref_cloth = session.refined_cloth

        cap = cv2.VideoCapture(video["path"])
        if not cap.isOpened():
            with session.lock:
                session.percent = int((v_idx + 1) / max(1, session.total_videos) * 100)
                session.phase = f"Couldn't open {video['name']} — skipped"
            continue

        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_i = 0
        seen_last = -999.0
        consec = 0
        writer = None
        annot_path = ""
        annot_failed = False
        last_bbox = None
        last_seen_ts = -999.0
        last_fused = 0.0
        shot_count = 0
        while cap.isOpened():
            with session.lock:
                if session.cancel_requested:
                    break
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                frame_i += 1
                continue

            h, w = frame.shape[:2]
            if w > target_width:
                scale = target_width / float(w)
                frame_small = cv2.resize(frame, (target_width, int(h * scale)))
            else:
                frame_small = frame.copy()
            t_sec = frame_i / fps

            if writer is None and not annot_failed:
                try:
                    annot_path = os.path.join(
                        annot_dir, f"{sanitize_filename(os.path.splitext(video['name'])[0])}_annotated.mp4")
                    writer = cv2.VideoWriter(annot_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (frame_small.shape[1], frame_small.shape[0]))
                    if not writer.isOpened():
                        writer = None
                        annot_failed = True
                except Exception:
                    writer = None
                    annot_failed = True

            if writer is not None:
                try:
                    annot_frame = frame_small.copy()
                    if last_bbox is not None and (t_sec - last_seen_ts) <= REACQUIRE_GAP:
                        draw_match_annotation(annot_frame, last_bbox, last_fused)
                    writer.write(annot_frame)
                    del annot_frame
                except Exception:
                    pass

            if session.skip_frames > 0 and (frame_i % (session.skip_frames + 1) != 0):
                frame_i += 1
                del frame, frame_small
                continue

            face_score = 0.0
            best_face = None
            try:
                faces = get_faces(frame_small, app)
                with session.lock:
                    session.frames_processed += 1
                    if faces:
                        session.frames_with_face += 1
                if faces and cur_ref_face is not None:
                    sims = [(cosine_sim(f["emb"], cur_ref_face), f) for f in faces]
                    face_score, best_face = max(sims, key=lambda x: x[0])
                    face_score = float(face_score)
            except Exception:
                face_score = 0.0
                best_face = None

            pose_score = 0.0
            cloth_score = 0.0
            if face_score >= FACE_THR:
                if cur_ref_pose is not None:
                    try:
                        pf = extract_pose_feats_bgr(frame_small)
                        if pf is not None:
                            pose_score = cosine_sim(pf, cur_ref_pose)
                    except Exception:
                        pose_score = 0.0
                if cur_ref_cloth is not None and best_face is not None:
                    try:
                        cd = clothing_descriptor(frame_small, best_face["bbox"])
                        if cd is not None:
                            cloth_score = cosine_sim(cd, cur_ref_cloth)
                    except Exception:
                        cloth_score = 0.0

            fused = (session.face_weight * face_score +
                     session.pose_weight * pose_score +
                     session.cloth_weight * cloth_score)

            with session.lock:
                if face_score > session.best_face_sim:
                    session.best_face_sim = face_score
                if fused > session.best_fused:
                    session.best_fused = fused

            if best_face is not None and face_score >= FACE_THR and cur_ref_face is not None:
                last_bbox = tuple(best_face["bbox"])
                last_seen_ts = t_sec
                last_fused = fused

            hit = (face_score >= FACE_THR) and (fused >= session.threshold)
            if hit:
                consec += 1
            else:
                consec = 0

            if hit and consec >= CONSEC and (t_sec - seen_last) >= COOLDOWN:
                seen_last = t_sec
                shot_name = ""
                if shot_count < MAX_EVIDENCE_SHOTS and best_face is not None and cur_ref_face is not None:
                    evidence = frame_small.copy()
                    draw_match_annotation(evidence, best_face["bbox"], fused)
                    shot_name = f"match_{v_idx}_{int(t_sec * 100):05d}_{uuid.uuid4().hex[:6]}.jpg"
                    try:
                        cv2.imwrite(os.path.join(session.workdir, shot_name), evidence)
                    except Exception:
                        shot_name = ""
                    evidence_candidates.append((fused, shot_name))
                    shot_count += 1
                events.append(MatchEvent(
                    t_sec=t_sec, face_score=face_score, pose_score=pose_score,
                    fused_score=fused, frame_index=frame_i, video_id=video["id"],
                    video_name=video["name"], screenshot_name=shot_name,
                    cloth_score=cloth_score, bbox=last_bbox))
                with session.lock:
                    session.events = events
                    session.matches = group_events(events)
                    session.confidence_series = build_confidence_series(session.matches)

            total_videos = max(1, session.total_videos)
            v_progress = (v_idx + min(1.0, frame_i / max(1, total_frames))) / total_videos
            new_percent = int(v_progress * 100)
            if new_percent != session.percent:
                session.phase = f"Scanning {video['name']} — {session.current_index}/{session.total_videos}"
                session.percent = new_percent

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

        with session.lock:
            session.phase = f"Encoding annotated video for {video['name']}…"
        if (annot_path and os.path.exists(annot_path) and os.path.getsize(annot_path) > 1024):
            upgraded = _maybe_transcode(annot_path, ffmpeg_bin)
            if upgraded:
                annotated_files[video["name"]] = os.path.basename(upgraded)

    # keep only the MAX_EVIDENCE_SHOTS strongest evidence screenshots
    evidence_candidates.sort(key=lambda x: -x[0])
    keep = set(name for _, name in evidence_candidates[:MAX_EVIDENCE_SHOTS])
    for fused_score, shot_name in evidence_candidates:
        if shot_name not in keep:
            try:
                os.remove(os.path.join(session.workdir, shot_name))
            except Exception:
                pass
    for ev in events:
        if ev.screenshot_name and ev.screenshot_name not in keep:
            ev.screenshot_name = ""

    with session.lock:
        session.annotated_files = annotated_files
        session.events = events
        session.matches = group_events(events)
        session.confidence_series = build_confidence_series(session.matches)
        session.color_labels = clothing_color_labels(session.refined_cloth)
        if session.cancel_requested:
            session.phase = "Scan cancelled"
            session.status = "cancelled"
            return

    session.phase = "Building highlight reel…"
    _maybe_build_reel(session, ffmpeg_bin)

    with session.lock:
        session.phase = "Complete"
        session.percent = 100
        for m in session.matches:
            if m["video_name"] in annotated_files:
                m["annotated_video"] = annotated_files[m["video_name"]]
        session.status = "done"


def _maybe_build_reel(session: ScanSession, ffmpeg_bin) -> None:
    """Concatenate per-source annotated clips into a highlight reel."""
    try:
        if not session.annotated_files or not session.matches:
            return
        first_path = os.path.join(session.workdir, "annotated",
                                  next(iter(session.annotated_files.values())))
        probe = cv2.VideoCapture(first_path)
        if not probe.isOpened():
            return
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
        probe.release()

        ordered = sorted(session.matches, key=lambda m: (m["start_sec"], m["video_name"]))
        clips = []
        for m in ordered:
            path = os.path.join(session.workdir, "annotated",
                                session.annotated_files.get(m["video_name"], ""))
            if not path or not os.path.exists(path):
                continue
            clips.append((path, float(m["start_sec"]),
                          float(m["end_sec"]) - float(m["start_sec"]), m["video_name"]))
        if not clips:
            return

        out_path = os.path.join(session.workdir, "highlight_reel.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            return

        for path, src_start, src_dur, vname in clips:
            _write_reel_divider(writer, width, height, fps, vname, src_start)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            clip_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if clip_fps <= 0:
                clip_fps = fps
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_f = max(0, int((src_start - PAD) * clip_fps))
            end_f = min(total - 1, int((src_start + src_dur + PAD) * clip_fps))
            if end_f <= start_f:
                end_f = min(total - 1, start_f + 1)
            try:
                _write_reel_clip(cap, start_f, end_f, clip_fps, fps, width, height, writer)
            finally:
                cap.release()

        writer.release()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            session.reel_file = os.path.basename(_maybe_transcode(out_path, ffmpeg_bin))
    except Exception:
        session.reel_file = None


@dataclass
class ReferenceProfile:
    face: Optional[np.ndarray]
    pose: Optional[np.ndarray]
    cloth: Optional[np.ndarray]
    faces_used: int = 0
    built_at: str = ""

    @property
    def face_ready(self) -> bool:
        return self.face is not None

    @property
    def pose_ready(self) -> bool:
        return self.pose is not None

    @property
    def cloth_ready(self) -> bool:
        return self.cloth is not None

    @property
    def embeddings(self) -> int:
        return self.faces_used * 512 if self.face_ready else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faceReady": self.face_ready,
            "poseReady": self.pose_ready,
            "clothReady": self.cloth_ready,
            "embeddings": self.embeddings,
            "builtAt": self.built_at,
        }

    def to_bytes(self) -> bytes:
        """Serialize the profile to .npz bytes (unencrypted biometrics)."""
        buf = io.BytesIO()
        np.savez(
            buf,
            ref_face=self.face if self.face_ready else np.array([]),
            ref_pose=self.pose if self.pose_ready else np.array([]),
            ref_cloth=self.cloth if self.cloth_ready else np.array([]),
        )
        return buf.getvalue()


def build_reference(photo_paths: List[str]) -> ReferenceProfile:
    """Build a reference profile from local image paths."""
    try:
        app = get_face_app()
    except Exception:
        raise
    face_embs: List[np.ndarray] = []
    pose_embs: List[np.ndarray] = []
    cloth_embs: List[np.ndarray] = []
    faces_used = 0
    for p in photo_paths:
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            faces = get_faces(img, app)
            best = pick_best_face(faces)
            if best is not None:
                face_embs.append(best["emb"])
                faces_used += 1
                cd = clothing_descriptor(img, best["bbox"])
                if cd is not None:
                    cloth_embs.append(cd)
            pf = extract_pose_feats_bgr(img)
            if pf is not None:
                pose_embs.append(pf)
        except Exception:
            continue

    if not face_embs:
        return ReferenceProfile(None, None, None, 0, "")

    ref_face = mean_normalize_stack(face_embs)
    ref_pose = None
    if pose_embs:
        try:
            stack = np.vstack(pose_embs)
            mean_pose = np.mean(stack, axis=0)
            norm = np.linalg.norm(mean_pose) + 1e-9
            ref_pose = (mean_pose / norm).astype(np.float32)
        except Exception:
            ref_pose = None
    ref_cloth = mean_normalize_stack(cloth_embs) if cloth_embs else None

    return ReferenceProfile(ref_face, ref_pose, ref_cloth, faces_used,
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def make_zip_of_files(paths: List[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p and os.path.exists(p):
                zf.write(p, arcname=os.path.basename(p))
    return buf.getvalue()
