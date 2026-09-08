"""Face detection + recognition engine (InsightFace).

Defaults favour accuracy over raw speed so that low-quality, blurry, or
small-face footage still resolves:

- ``FACE_MODEL``: model pack name. ``buffalo_l`` ships the stronger
  ``det_10g`` detector and ResNet50-based ``w600k_r50`` recogniser, which
  generalise much better than the bundled ``buffalo_s`` (MobileFaceNet) pack.
  Falls back to ``buffalo_s`` automatically if the selected pack is unavailable.
- ``FACE_DET_SIZE``: inference input size as ``"W,H"``. Larger sizes help
  find small faces in surveillance-framed footage at a speed cost.
- ``FACE_DET_THRESH``: detection confidence floor. Lower values recover
  blurred / partially-occluded faces at the cost of a few false positives.

Runs fully on the local machine; model packs are auto-downloaded by
InsightFace on first use into ``models_cache/``.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# ----------------------------------------------------------------------
# model configuration
# ----------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_ROOT = str((BACKEND_DIR.parent / "models_cache").resolve())

FACE_MODEL = os.environ.get("FACE_MODEL", "buffalo_l")
FALLBACK_FACE_MODEL = "buffalo_s"
_DEFAULT_DET_SIZE = (896, 896)
_DEFAULT_DET_THRESH = 0.4


def _parse_det_size(raw: str) -> tuple:
    try:
        w, h = (int(x.strip()) for x in raw.split(","))
        return (w, h)
    except Exception:
        return _DEFAULT_DET_SIZE


def _det_size_from_env() -> tuple:
    return _parse_det_size(os.environ.get("FACE_DET_SIZE", "896,896"))


def _det_thresh_from_env() -> float:
    try:
        return float(os.environ.get("FACE_DET_THRESH", str(_DEFAULT_DET_THRESH)))
    except Exception:
        return _DEFAULT_DET_THRESH


def init_face_app(det_size=None, ctx_id=0):
    """Create a prepared FaceAnalysis app, preferring the configured model.

    Falls back to ``buffalo_s`` when the requested pack cannot be downloaded
    or loaded, so the pipeline still works on machines without network access
    or with a stale ``models_cache``.
    """
    det_size = det_size or _det_size_from_env()
    det_thresh = _det_thresh_from_env()
    attempts = [FACE_MODEL]
    if FACE_MODEL != FALLBACK_FACE_MODEL:
        attempts.append(FALLBACK_FACE_MODEL)
    last_err = None
    for name in attempts:
        try:
            app = FaceAnalysis(name=name, root=MODELS_ROOT)
            app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
            return app
        except Exception as e:  # noqa: BLE001 - try the next pack
            last_err = e
            continue
    raise RuntimeError(f"Face engine failed to load ({attempts}): {last_err}")


def _upscale_if_tiny(img_bgr: np.ndarray, min_dim: int = 576) -> tuple:
    """Return (image, scale) — upscale very small frames so faces are large
    enough for detection to find them (common with low-res CCTV clips)."""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest >= min_dim:
        return img_bgr, 1.0
    scale = min(2.0, min_dim / float(longest))
    new_w, new_h = int(w * scale), int(h * scale)
    up = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return up, scale


def get_faces(img_bgr, app):
    """Detect faces and return recognised embeddings.

    Each entry: {"bbox": [x1,y1,x2,y2], "emb": (512,) float32, "det_score": float}.

    Detection is retried on an upscaled copy when the source frame is too
    small for a clear read, scaling results back into original coordinates.
    """
    work, scale = _upscale_if_tiny(img_bgr)
    faces = app.get(work)
    if not faces and scale > 1.0:
        # retry once at full 2x magnification for stubborn small faces
        big = cv2.resize(
            img_bgr, (img_bgr.shape[1] * 2, img_bgr.shape[0] * 2),
            interpolation=cv2.INTER_CUBIC,
        )
        faces = app.get(big)
        scale = 2.0

    outs = []
    for f in faces:
        if getattr(f, "normed_embedding", None) is None:
            # some pipelines require calling get with rec=True; the default
            # buffalo packs compute embeddings automatically
            continue
        bbox = list(map(float, f.bbox))
        if scale > 1.0:
            bbox = [x / scale for x in bbox]
        outs.append({
            "bbox": bbox,
            "emb": f.normed_embedding.astype(np.float32),
            "det_score": float(getattr(f, "det_score", 0.0)),
        })
    return outs


def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))


def mean_normalize_stack(emb_list):
    """Average multiple embeddings then L2 normalize."""
    E = np.vstack(emb_list)
    m = E.mean(axis=0)
    m = m / (norm(m) + 1e-9)
    return m