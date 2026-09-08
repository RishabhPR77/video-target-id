"""Shared constants for Video Target ID pipeline.

These values are used by both the Streamlit app and CLI scripts to
ensure consistent behaviour across interfaces.
"""

# ── Match thresholds ──────────────────────────────────────────────
FACE_THR  = 0.42   # minimum face-similarity before a frame can count as a hit
FUSED_THR = 0.48   # minimum fused (face+pose) score
CONSEC    = 3      # consecutive matching frames required before logging
COOLDOWN  = 2.0    # minimum seconds between logged hits

# ── Default fusion weights ────────────────────────────────────────
# Pose is a single-frame posture heuristic (see pose_module.py) — it must
# never dominate.  Keep W_FACE high so face similarity carries the match.
W_FACE = 0.80
W_POSE = 0.20

# ── Authorization gate ────────────────────────────────────────────
AUTH_FLAG = "--i-am-authorized"
AUTH_TEXT = (
    "I confirm I am authorised to process this data for lawful purposes."
)
