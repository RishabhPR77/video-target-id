import os
import cv2
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis

def init_face_app(det_size=(640,640), ctx_id=0):
    app = FaceAnalysis(name="buffalo_s", root=os.path.join(os.getcwd(), "models_cache"))
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

def get_faces(img_bgr, app):
    """Return list of dicts: [{'bbox': [x1,y1,x2,y2], 'emb': (512,), 'kps':...}, ...]"""
    faces = app.get(img_bgr)
    outs = []
    for f in faces:
        if getattr(f, "normed_embedding", None) is None:
            # some pipelines require calling get with rec=True; buffalo_l does embeddings by default
            continue
        outs.append({
            "bbox": list(map(float, f.bbox)),
            "emb": f.normed_embedding.astype(np.float32)
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
