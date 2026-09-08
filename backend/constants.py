"""Shared constants for Video Target ID pipeline.

These values are used by both the Streamlit app and CLI scripts to
ensure consistent behaviour across interfaces.
"""

# ── Match thresholds ──────────────────────────────────────────────
FACE_THR  = 0.42   # minimum face-similarity before a frame can count as a hit
FUSED_THR = 0.45   # minimum fused (face+pose) score
CONSEC    = 3      # consecutive matching frames required before logging / (re)acquisition
COOLDOWN  = 2.0    # minimum seconds between logged hits

# ── Continuous tracking ───────────────────────────────────────────
# Once the target is locked, its bounding box is followed by motion
# prediction between face detections.  A box continues to be drawn until
# the target has been unseen longer than REACQUIRE_GAP (then we search for
# him again, matching against the live-refined identity).  Evidence
# screenshots are kept only for the strongest sightings — no wall of photos.
REACQUIRE_GAP     = 4.0   # sec without a face association → close the sighting
MIN_SIGHTING_SECS = 0.4   # shorter tracks are discarded
MAX_EVIDENCE_SHOTS = 5    # how many best-evidence screenshots are saved per scan

# ── Default fusion weights ────────────────────────────────────────
# Pose is a single-frame posture heuristic (see pose_module.py) — it must
# never dominate.  Keep W_FACE high so face similarity carries the match.
# W_CLOTH is a weak tertiary signal: an HSV histogram over the upper-body
# region below the face.  Same clothing corroborates a face match but is
# never allowed to carry one on its own.
W_FACE = 0.80
W_POSE = 0.20
W_CLOTH = 0.10

# When a scan uses a custom face_weight, the remaining weight is split
# between pose and clothing with clothing capped at W_CLOTH.
CLOTH_WEIGHT_CAP = W_CLOTH

# ── Authorization gate ────────────────────────────────────────────
AUTH_FLAG = "--i-am-authorized"
AUTH_TEXT = (
    "I confirm I am authorised to process this data for lawful purposes."
)
