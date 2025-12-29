import os, cv2, glob, csv, json, numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from face_module import init_face_app, get_faces, cosine_sim
from pose_module import extract_pose_feats_bgr

REF_JSON = "outputs/reference_profile.json"
OUT_CSV = "outputs/detections.csv"

FACE_THR = 0.42
FUSED_THR = 0.48
W_FACE, W_GAIT, W_POST = 0.7, 0.2, 0.1
CONSEC = 3               # require N consecutive frames to mitigate false alarms
FRAME_STRIDE = 3         # analyze every Nth frame for speed

def load_reference():
    with open(REF_JSON, "r") as f:
        ref = json.load(f)
    ref_face = np.array(ref["face"], dtype=np.float32) if ref["face"] is not None else None
    ref_pose = np.array(ref["pose"], dtype=np.float32) if ref["pose"] is not None else None
    return ref_face, ref_pose

def sim_pose(p, q):
    if p is None or q is None: return 0.0
    return float(np.dot(p, q) / (norm(p)*norm(q) + 1e-9))

def run_on_video(video_path, app, ref_face, ref_pose, writer):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx, consec = 0, 0
    last_hit_ts = -999

    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % FRAME_STRIDE != 0:
            idx += 1; continue

        t_sec = idx / fps
        # FACE
        faces = get_faces(frame, app)
        face_score = 0.0
        best_face = None
        if ref_face is not None and faces:
            # pick best face by similarity
            sims = [(cosine_sim(f["emb"], ref_face), f) for f in faces]
            face_score, best_face = max(sims, key=lambda x: x[0])

        # POSE (posture/gait proxy)
        pose = extract_pose_feats_bgr(frame)
        pose_score = sim_pose(pose, ref_pose) if (pose is not None and ref_pose is not None) else 0.0

        fused = W_FACE*face_score + W_GAIT*pose_score + W_POST*pose_score  # simple reuse pose for posture
        hit = (face_score >= FACE_THR) and (fused >= FUSED_THR)

        if hit:
            consec += 1
        else:
            consec = 0

        if hit and consec >= CONSEC and (t_sec - last_hit_ts) > 2.0:
            # save crop for audit
            crop_path = ""
            if best_face is not None:
                x1,y1,x2,y2 = map(int, best_face["bbox"])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
                crop = frame[y1:y2, x1:x2]
                os.makedirs("outputs/crops", exist_ok=True)
                crop_path = os.path.join("outputs/crops", f"{os.path.basename(video_path)}_{int(t_sec*1000)}.jpg")
                if crop.size > 0:
                    cv2.imwrite(crop_path, crop)

            writer.writerow({
                "video": os.path.basename(video_path),
                "timestamp_sec": f"{t_sec:.2f}",
                "face_sim": f"{face_score:.3f}",
                "pose_sim": f"{pose_score:.3f}",
                "fused": f"{fused:.3f}",
                "crop_path": crop_path
            })
            last_hit_ts = t_sec

        idx += 1

    cap.release()

def main():
    os.makedirs("outputs", exist_ok=True)
    ref_face, ref_pose = load_reference()
    app = init_face_app()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video","timestamp_sec","face_sim","pose_sim","fused","crop_path"])
        writer.writeheader()
        for v in tqdm(sorted(glob.glob("data/cctv_videos/*.*"))):
            run_on_video(v, app, ref_face, ref_pose, writer)
    print(f"[OK] Results → {OUT_CSV}")
if __name__ == "__main__":
    main()
