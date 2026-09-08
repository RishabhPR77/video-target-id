import os, sys, cv2, glob, numpy as np, json
from tqdm import tqdm
from face_module import init_face_app, get_faces, mean_normalize_stack
from pose_module import extract_pose_feats_bgr
from constants import AUTH_FLAG, AUTH_TEXT

REF_JSON = "outputs/reference_profile.json"

def build_reference(reference_photos_dir, reference_videos_dir=None):
    os.makedirs("outputs", exist_ok=True)

    app = init_face_app()
    face_embs = []
    pose_embs = []

    # Photos
    for p in sorted(glob.glob(os.path.join(reference_photos_dir, "*.*"))):
        img = cv2.imread(p)
        if img is None: continue
        faces = get_faces(img, app)
        if faces:
            face_embs.append(faces[0]["emb"])

        pf = extract_pose_feats_bgr(img)
        if pf is not None:
            pose_embs.append(pf)

    # Optional: short walking videos
    if reference_videos_dir and os.path.isdir(reference_videos_dir):
        for v in sorted(glob.glob(os.path.join(reference_videos_dir, "*.*"))):
            cap = cv2.VideoCapture(v)
            if not cap.isOpened(): continue
            step = 5
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok: break
                if i % step == 0:
                    faces = get_faces(frame, app)
                    if faces:
                        face_embs.append(faces[0]["emb"])
                    pf = extract_pose_feats_bgr(frame)
                    if pf is not None:
                        pose_embs.append(pf)
                i += 1
            cap.release()

    ref_face = mean_normalize_stack(face_embs) if face_embs else None
    ref_pose = np.mean(np.vstack(pose_embs), axis=0) if pose_embs else None
    if ref_pose is not None:
        ref_pose = ref_pose / (np.linalg.norm(ref_pose) + 1e-9)

    ref = {"face": ref_face.tolist() if ref_face is not None else None,
           "pose": ref_pose.tolist() if ref_pose is not None else None}

    # WARNING — reference_profile.json contains raw, UNENCRYPTED biometric
    # embeddings of identifiable persons. Handle, access-control, and delete
    # per your own legal obligations (see README → "Data Handling").
    with open(REF_JSON, "w") as f:
        json.dump(ref, f)
    print(f"[OK] Saved reference profile → {REF_JSON}")
    print("[NOTICE] This file contains UNENCRYPTED biometric embeddings — "
          "store it securely and delete when no longer needed. "
          "See the README 'Data Handling' section.")

if __name__ == "__main__":
    if AUTH_FLAG not in sys.argv:
        print(
            f"[ERROR] Authorisation required before building a biometric reference.\n"
            f"        {AUTH_TEXT}\n"
            f"        Re-run with the {AUTH_FLAG} flag to confirm.",
            file=sys.stderr,
        )
        sys.exit(1)
    build_reference("data/reference_photos", "data/reference_videos")