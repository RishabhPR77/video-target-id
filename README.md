# 🎯 Video Target Identification System

An AI-powered forensic video analysis tool that searches CCTV/surveillance footage for a specific person using **face recognition** and **pose/gait analysis** — built with InsightFace, MediaPipe, and Streamlit.

---

## ✨ Features

- **Face Recognition** — InsightFace `buffalo_s` model generates 512-dimensional embeddings from reference photos
- **Pose & Gait Analysis** — MediaPipe extracts body keypoints to build a structural silhouette signature
- **Fused Confidence Score** — Weighted combination of face + pose similarity (adjustable in real time via the sidebar)
- **Multi-video batch scanning** — Upload and scan multiple CCTV files in one session
- **Resolution & speed controls** — Frame skipping and resolution slider to balance speed vs. accuracy
- **Evidence export** — Download a PDF report, CSV log, and ZIP of all screenshot evidence
- **Glassmorphism UI** — Dark-themed, professional Streamlit interface

---

## 🗂️ Project Structure

```
src/
├── app.py                  # Streamlit web app (main entry point)
├── face_module.py          # InsightFace wrapper (init, embedding, cosine sim)
├── pose_module.py          # MediaPipe pose embedding builder
├── build_reference.py      # CLI: build a reference profile from a photo/video folder
├── search_cctv.py          # CLI: batch-scan a folder of CCTV videos
├── requirements.txt
├── README.md
├── .gitignore
│
├── models_cache/           # ← NOT in git (auto-downloaded on first run)
│   └── models/
│       └── buffalo_s/
│
├── data/                   # ← NOT in git (your private input data)
│   ├── reference_photos/
│   ├── reference_videos/   # (optional)
│   └── cctv_videos/
│
└── outputs/                # ← NOT in git (generated at runtime)
    ├── reference_profile.json
    ├── detections.csv
    └── crops/
```

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/video-target-id.git
cd video-target-id/src
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU users:** swap `onnxruntime` for `onnxruntime-gpu` in `requirements.txt` for faster inference.

### 4. Model download (automatic)
The `buffalo_s` face model is downloaded automatically by InsightFace on first run into `models_cache/`. No manual step needed.

---

## 🚀 Usage

### Option A — Streamlit Web App (recommended)

```bash
streamlit run app.py
```

Then follow the 4-step wizard in your browser:

| Step | What you do |
|------|-------------|
| **1 – Target** | Upload 1–3 clear reference photos of the person. Click **Build Reference**. |
| **2 – Source** | Upload one or more CCTV video files (MP4, AVI, MOV, MKV). |
| **3 – Scan** | Set frame skipping & resolution, then click **Start Analysis**. |
| **4 – Results** | View the match timeline, confidence graph, screenshots, and download reports. |

---

### Option B — CLI Scripts (batch / headless)

#### Build a reference profile from photos/videos
```bash
python build_reference.py
```
Reads from `data/reference_photos/` (and optionally `data/reference_videos/`), writes `outputs/reference_profile.json`.

#### Scan a folder of CCTV videos
```bash
python search_cctv.py
```
Reads from `data/cctv_videos/`, writes `outputs/detections.csv` and face crops to `outputs/crops/`.

---

## 🧠 How It Works

```
Reference Photos
      │
      ▼
  InsightFace ──► 512-d face embedding (averaged + L2 normalized)
  MediaPipe   ──► 12-d pose embedding (joint angles + limb ratios)
      │
      ▼
  Reference Profile  (.json / .npz)
      │
      ├─── Per video frame ──────────────────────────────────────┐
      │         InsightFace ──► face_score  (cosine similarity)  │
      │         MediaPipe   ──► pose_score  (cosine similarity)  │
      │                                                          │
      │    fused = face_weight × face_score                      │
      │           + pose_weight × pose_score                     │
      │                                                          │
      │    if fused ≥ threshold  →  log detection + save crop    │
      └──────────────────────────────────────────────────────────┘
```

**Default weights:** Face 0.7 · Pose 0.3 (adjustable in the sidebar at runtime).

---

## 🔧 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FACE_THR` | `0.42` | Minimum face cosine similarity to consider a candidate |
| `FUSED_THR` | `0.48` | Minimum fused score to log a detection |
| `CONSEC` | `3` | Consecutive matching frames required (reduces false alarms) |
| `FRAME_STRIDE` | `3` | Analyze every Nth frame (CLI scripts) |
| Detection threshold (UI) | `0.60` | Fused score threshold in the Streamlit app |

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `insightface` | Face detection & recognition (buffalo_s model) |
| `mediapipe` | Body pose landmark extraction |
| `opencv-python-headless` | Video reading & image processing |
| `streamlit` | Web UI |
| `fpdf2` | PDF report generation |
| `onnxruntime` | InsightFace model inference backend |

---

## ⚠️ Legal & Ethical Notice

This tool is intended for **authorized forensic and security use only**. Processing biometric data without the subject's consent or appropriate legal authority may be illegal in your jurisdiction. The app includes a mandatory authorization checkbox before any analysis begins. **You are solely responsible for ensuring lawful and ethical use.**

---

## 📄 License

MIT License — see `LICENSE` for details.