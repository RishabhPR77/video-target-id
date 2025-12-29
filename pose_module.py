import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Keypoint indices we’ll use (MediaPipe Pose has 33 landmarks)
# We'll craft a compact pose/gait embedding: angles + limb ratios normalized by torso size.
_USED = dict(
    L_SH=np.array([11]), R_SH=np.array([12]),
    L_EL=np.array([13]), R_EL=np.array([14]),
    L_WR=np.array([15]), R_WR=np.array([16]),
    L_HI=np.array([23]), R_HI=np.array([24]),
    L_KN=np.array([25]), R_KN=np.array([26]),
    L_AN=np.array([27]), R_AN=np.array([28]),
    NO=np.array([0])
)

def _v(a, b):
    return b - a

def _angle(u, v):
    un = u / (np.linalg.norm(u) + 1e-9)
    vn = v / (np.linalg.norm(v) + 1e-9)
    return np.arccos(np.clip(np.dot(un, vn), -1.0, 1.0))

def pose_embedding(landmarks32):
    """landmarks32: (33,2) or (33,3) np.array in image coords. Returns (D,) feature."""
    pts = landmarks32[:, :2].astype(np.float32)
    sh_mid = (pts[11] + pts[12]) / 2.0
    hi_mid = (pts[23] + pts[24]) / 2.0
    torso = np.linalg.norm(hi_mid - sh_mid) + 1e-6
    # Normalize to torso size
    P = (pts - hi_mid) / torso

    # Angles (shoulder-elbow-wrist, hip-knee-ankle) left/right
    ang_le = _angle(_v(P[11], P[13]), _v(P[13], P[15]))
    ang_re = _angle(_v(P[12], P[14]), _v(P[14], P[16]))
    ang_lk = _angle(_v(P[23], P[25]), _v(P[25], P[27]))
    ang_rk = _angle(_v(P[24], P[26]), _v(P[26], P[28]))

    # Limb ratios (normalized lengths)
    len_ul_l = np.linalg.norm(P[11]-P[13]) + np.linalg.norm(P[13]-P[15])
    len_ul_r = np.linalg.norm(P[12]-P[14]) + np.linalg.norm(P[14]-P[16])
    len_ll_l = np.linalg.norm(P[23]-P[25]) + np.linalg.norm(P[25]-P[27])
    len_ll_r = np.linalg.norm(P[24]-P[26]) + np.linalg.norm(P[26]-P[28])

    # Shoulder-hip-knee/ankle vertical offsets for posture
    off_sh = float((P[11,1]+P[12,1])/2.0)
    off_hi = float((P[23,1]+P[24,1])/2.0)
    off_kn = float((P[25,1]+P[26,1])/2.0)
    off_an = float((P[27,1]+P[28,1])/2.0)

    feat = np.array([
        ang_le, ang_re, ang_lk, ang_rk,
        len_ul_l, len_ul_r, len_ll_l, len_ll_r,
        off_sh, off_hi, off_kn, off_an
    ], dtype=np.float32)

    # L2 normalize the feature so cosine similarity is meaningful
    n = np.linalg.norm(feat) + 1e-9
    return feat / n

def extract_pose_feats_bgr(img_bgr):
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(img_bgr[:, :, ::-1])  # BGR->RGB
        if not res.pose_landmarks:
            return None
        lm = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark], dtype=np.float32)
        return pose_embedding(lm)
