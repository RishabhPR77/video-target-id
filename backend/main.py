"""FastAPI application for Video Target ID.

Run from the repository root:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

The frontend (`frontend/`) points at this service via `VITE_API_BASE`
(defaults to http://localhost:8000). All heavy work runs in background
threads and is polled through GET /api/scan/{id}.
"""

import io
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from . import engine
from .constants import FACE_THR
from .engine import ScanSession, build_reference, ReferenceProfile

# ----------------------------------------------------------------------
# runtime directories (all git-ignored under backend/data/)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
REF_DIR = DATA_DIR / "reference"
SCANS_DIR = DATA_DIR / "scans"

for _d in (DATA_DIR, UPLOADS_DIR, REF_DIR, SCANS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    FPDF_AVAILABLE = False


app = FastAPI(title="Video Target ID API", version="2.4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# in-memory stores
# ----------------------------------------------------------------------
_ref_lock = threading.Lock()
reference: Optional[ReferenceProfile] = None
reference_files: List[str] = []

_uploads_lock = threading.Lock()
uploads: Dict[str, Dict[str, Any]] = {}

_scans_lock = threading.Lock()
scans: Dict[str, ScanSession] = {}


def _probe_video(path: str) -> Dict[str, Any]:
    """Duration/fps probe used when registering uploads."""
    duration = 0.0
    fps = 0.0
    try:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0 and n > 0:
                duration = n / fps
            cap.release()
    except Exception:
        pass
    return {"sizeBytes": os.path.getsize(path), "durationSeconds": round(float(duration), 2)}


def _scan_dir(scan_id: str) -> Path:
    d = SCANS_DIR / scan_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _match_to_server(m: Dict[str, Any], scan_id: str, annotated: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": m["id"],
        "scan_id": scan_id,
        "video_id": m["video_id"],
        "video_name": m["video_name"],
        "start_sec": m["start_sec"],
        "end_sec": m["end_sec"],
        "face": m["face"],
        "pose": m["pose"],
        "cloth": m.get("cloth", 0.0),
        "fused": m["fused"],
        "screenshot": m.get("screenshot"),
        "annotated_video": m.get("annotated_video") or annotated,
    }


# ----------------------------------------------------------------------
# health + reference
# ----------------------------------------------------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "reference_built": reference is not None,
        "reference_cloth_ready": reference.cloth_ready if reference else False,
        "uploaded_videos": len(uploads),
        "active_scans": len(scans),
    }


@app.post("/api/reference")
async def build_reference_endpoint(photos: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Build the reference profile from 1-5 uploaded photos."""
    if not photos:
        raise HTTPException(400, "Provide at least one reference photo.")

    temp_paths: List[str] = []
    for up in photos:
        try:
            data = await up.read()
            if engine.safe_imdecode(data) is None:
                continue
            p = REF_DIR / f"{uuid.uuid4().hex}{Path(up.filename or 'img.jpg').suffix or '.jpg'}"
            p.write_bytes(data)
            temp_paths.append(str(p))
        except Exception:
            continue

    if not temp_paths:
        raise HTTPException(400, "No usable reference photos were provided.")

    try:
        profile = build_reference(temp_paths)
    except Exception as e:
        raise HTTPException(500, f"Reference build failed: {e}")
    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

    if not profile.face_ready:
        raise HTTPException(422, "No faces detected. Try clearer, well-lit front-facing photos.")

    with _ref_lock:
        global reference, reference_files
        reference = profile
        reference_files = temp_paths  # files already removed; kept list empty
        reference_files = []

    return profile.to_dict()


@app.get("/api/reference/profile")
def download_profile() -> Response:
    with _ref_lock:
        if reference is None:
            raise HTTPException(404, "No reference profile built yet.")
        payload = reference.to_bytes()
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="target_profile.npz"'},
    )


@app.get("/api/scan/{scan_id}/refined-profile")
def download_refined_profile(scan_id: str) -> Response:
    """Download the live-refined profile (face+pose+cloth) from a finished scan."""
    with _scans_lock:
        session = scans.get(scan_id)
    if session is None:
        raise HTTPException(404, "Scan not found.")
    _require_done(session)
    with session.lock:
        ref_face = session.refined_face
        ref_pose = session.refined_pose
        ref_cloth = session.refined_cloth
    if ref_face is None:
        raise HTTPException(422, "No refined profile available.")
    import io as _io
    buf = _io.BytesIO()
    np.savez(buf, ref_face=ref_face, ref_pose=ref_pose, ref_cloth=ref_cloth)
    payload = buf.getvalue()
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="refined_profile.npz"'},
    )


# ----------------------------------------------------------------------
# video uploads
# ----------------------------------------------------------------------
@app.post("/api/videos")
async def register_videos(files: List[UploadFile] = File(...)) -> List[Dict[str, Any]]:
    """Upload source footage; probe each clip and return its metadata."""
    results: List[Dict[str, Any]] = []
    for up in files:
        vid = uuid.uuid4().hex
        vdir = UPLOADS_DIR / vid
        vdir.mkdir(parents=True, exist_ok=True)
        suffix = Path(up.filename or "video.mp4").suffix or ".mp4"
        # keep the real container extension — renaming can break OpenCV backends
        target = vdir / f"source{suffix}"
        data = await up.read()
        target.write_bytes(data)

        probe = _probe_video(str(target))
        ext = (up.filename or "").rsplit(".", 1)[-1].upper() if "." in (up.filename or "") else "MP4"
        if ext not in ("MP4", "AVI", "MOV", "MKV"):
            ext = "MP4"

        with _uploads_lock:
            uploads[vid] = {"id": vid, "name": up.filename, "path": str(target)}

        results.append({
            "id": vid,
            "name": up.filename,
            "format": ext,
            "sizeBytes": probe["sizeBytes"],
            "durationSeconds": probe["durationSeconds"],
        })
    return results


# ----------------------------------------------------------------------
# scan lifecycle
# ----------------------------------------------------------------------
@app.post("/api/scan")
def start_scan(payload: Dict[str, Any]) -> Dict[str, str]:
    with _ref_lock:
        if reference is None:
            raise HTTPException(400, "Build a reference profile before scanning.")
        refs = (reference.face, reference.pose, reference.cloth)

    video_ids = payload.get("videos") or []
    if not video_ids:
        raise HTTPException(400, "Select at least one video to scan.")

    with _uploads_lock:
        selected = [dict(uploads[v]) for v in video_ids if v in uploads]
    if not selected:
        raise HTTPException(400, "None of the requested videos are registered.")

    scan_id = uuid.uuid4().hex[:12]
    workdir = str(_scan_dir(scan_id))
    session = ScanSession(
        scan_id=scan_id,
        videos=selected,
        workdir=workdir,
        threshold=float(payload.get("threshold", 0.45)),
        face_weight=float(payload.get("face_weight", 0.80)),
        skip_frames=int(payload.get("skip_frames", 2)),
        process_width=str(payload.get("process_width", "Medium (640px)")),
    )

    with _scans_lock:
        scans[scan_id] = session

    worker = threading.Thread(
        target=engine.run_scan,
        args=(session, refs[0], refs[1], refs[2]),
        daemon=True,
    )
    worker.start()
    return {"scan_id": scan_id}


def _snapshot(session: ScanSession) -> Dict[str, Any]:
    with session.lock:
        done = session.status == "done"
        latest = (sorted(session.matches, key=lambda m: m["fused"], reverse=True)[:1] or [None])[0]
        return {
            "status": session.status,
            "percent": session.percent,
            "phase": session.phase,
            "current_video_name": session.current_video_name,
            "current_index": session.current_index,
            "total_videos": session.total_videos,
            "error": session.error,
            "diagnostics": {
                "frames_processed": session.frames_processed,
                "frames_with_face": session.frames_with_face,
                "best_face_sim": session.best_face_sim,
                "best_fused": session.best_fused,
                "face_thr": FACE_THR,
                "threshold": session.threshold,
            },
            "latest_match": _match_to_server(latest, session.scan_id) if latest else None,
            "matches": [_match_to_server(m, session.scan_id) for m in session.matches],
            "confidence_series": session.confidence_series,
            "annotated": [
                {"id": name, "source_name": name, "matches": sum(
                    1 for m in session.matches if m["video_name"] == name), "video": fname}
                for name, fname in sorted(session.annotated_files.items())
            ] if done else [],
            "reel": session.reel_file if done else None,
            "color_labels": session.color_labels if done else [],
        }


@app.get("/api/scan/{scan_id}")
def scan_status(scan_id: str) -> Dict[str, Any]:
    with _scans_lock:
        session = scans.get(scan_id)
    if session is None:
        raise HTTPException(404, "Scan not found.")
    return _snapshot(session)


@app.post("/api/scan/{scan_id}/cancel")
def cancel_scan(scan_id: str) -> Dict[str, bool]:
    with _scans_lock:
        session = scans.get(scan_id)
    if session is None:
        raise HTTPException(404, "Scan not found.")
    with session.lock:
        session.cancel_requested = True
    return {"ok": True}


@app.delete("/api/scan/{scan_id}")
def delete_scan(scan_id: str) -> Dict[str, bool]:
    with _scans_lock:
        session = scans.pop(scan_id, None)
    if session is None:
        raise HTTPException(404, "Scan not found.")
    with session.lock:
        session.cancel_requested = True
    try:
        import shutil
        shutil.rmtree(session.workdir, ignore_errors=True)
    except Exception:
        pass
    return {"ok": True}


# ----------------------------------------------------------------------
# media
# ----------------------------------------------------------------------
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".zip": "application/zip",
}


@app.get("/api/scan/{scan_id}/media/{file:path}")
def serve_media(scan_id: str, file: str) -> FileResponse:
    with _scans_lock:
        session = scans.get(scan_id)
    if session is None:
        raise HTTPException(404, "Scan not found.")
    root = Path(session.workdir).resolve()
    target = (root / file).resolve()
    if not str(target).startswith(str(root)) or not target.is_file():
        raise HTTPException(404, "File not found.")
    media_type = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(target, media_type=media_type, filename=target.name)


# ----------------------------------------------------------------------
# exports
# ----------------------------------------------------------------------
def _fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _require_done(session: ScanSession) -> None:
    if session is None:
        raise HTTPException(404, "Scan not found.")
    if session.status != "done":
        raise HTTPException(409, "Scan is not finished yet.")
    with session.lock:
        if not session.matches:
            raise HTTPException(422, "No matches to export.")


@app.get("/api/scan/{scan_id}/exports/csv")
def export_csv(scan_id: str) -> Response:
    with _scans_lock:
        session = scans.get(scan_id)
    _require_done(session)
    with session.lock:
        rows = list(session.matches)
    buf = io.StringIO()
    buf.write("Rank,Video,Start (sec),Start Time,End Time,Duration (sec),Face,Pose,Cloth,Fused\n")
    for i, m in enumerate(sorted(rows, key=lambda r: r["fused"], reverse=True), 1):
        dur = m["end_sec"] - m["start_sec"]
        buf.write(f"{i},{engine.sanitize_filename(m['video_name'])},{m['start_sec']:.2f},"
                  f"{_fmt_time(m['start_sec'])},{_fmt_time(m['end_sec'])},{dur:.2f},"
                  f"{m['face']:.3f},{m['pose']:.3f},{m.get('cloth', 0.0):.3f},{m['fused']:.3f}\n")
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="target-id-matches.csv"'},
    )


@app.get("/api/scan/{scan_id}/exports/pdf")
def export_pdf(scan_id: str) -> Response:
    with _scans_lock:
        session = scans.get(scan_id)
    _require_done(session)
    with session.lock:
        rows = list(session.matches)

    if not FPDF_AVAILABLE:
        raise HTTPException(503, "PDF generation unavailable (fpdf2 missing).")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(30, 30, 50)
    pdf.cell(0, 12, "Forensic Video Analysis Report", ln=True, align="C")
    pdf.ln(2)
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(80, 80, 100)
    pdf.cell(0, 8, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)
    pdf.cell(0, 8, f"Total matches: {len(rows)}", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    for header, w in [("Video", 50), ("Timestamp", 30), ("Confidence", 30), ("Notes", 80)]:
        pdf.cell(w, 10, header, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Arial", size=9)
    for idx, m in enumerate(sorted(rows, key=lambda r: r["fused"], reverse=True)):
        pdf.set_fill_color(245, 247, 255) if idx % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(20, 20, 40)
        pdf.cell(50, 9, engine.sanitize_filename(m["video_name"])[:18], border=1, fill=True)
        pdf.cell(30, 9, _fmt_time(m["start_sec"]), border=1, fill=True)
        pdf.cell(30, 9, f"{m['fused']:.3f}", border=1, fill=True)
        pdf.cell(80, 9, f"Match via AI scan (cloth {m.get('cloth', 0.0):.3f})", border=1, fill=True)
        pdf.ln()
    raw = pdf.output()
    payload = bytes(raw) if isinstance(raw, (bytes, bytearray)) else str(raw).encode("latin-1")
    return Response(content=payload, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="target-id-report.pdf"'})


@app.get("/api/scan/{scan_id}/exports/evidence-zip")
def export_evidence_zip(scan_id: str) -> Response:
    with _scans_lock:
        session = scans.get(scan_id)
    _require_done(session)
    paths = []
    with session.lock:
        for m in session.matches:
            s = m.get("screenshot")
            if s:
                p = Path(session.workdir) / s
                if p.is_file():
                    paths.append(str(p))
    if not paths:
        raise HTTPException(422, "No evidence screenshots to export.")
    return zip_response(engine.make_zip_of_files(paths), "target-id-evidence.zip")


@app.get("/api/scan/{scan_id}/exports/annotated-zip")
def export_annotated_zip(scan_id: str) -> Response:
    with _scans_lock:
        session = scans.get(scan_id)
    _require_done(session)
    paths = []
    with session.lock:
        for fname in list(session.annotated_files.values()):
            p = Path(session.workdir) / "annotated" / fname
            if p.is_file():
                paths.append(str(p))
        if session.reel_file:
            p = Path(session.workdir) / session.reel_file
            if p.is_file():
                paths.append(str(p))
    if not paths:
        raise HTTPException(422, "No annotated videos to export.")
    return zip_response(engine.make_zip_of_files(paths), "target-id-annotated.zip")


@app.get("/api/scan/{scan_id}/exports/reel")
def export_reel(scan_id: str) -> FileResponse:
    with _scans_lock:
        session = scans.get(scan_id)
    _require_done(session)
    with session.lock:
        reel = session.reel_file
    if not reel:
        raise HTTPException(422, "No highlight reel was generated.")
    return FileResponse(Path(session.workdir) / reel, media_type="video/mp4",
                        filename="target-id-highlight-reel.mp4")


def zip_response(payload: bytes, filename: str) -> Response:
    return Response(content=payload, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})