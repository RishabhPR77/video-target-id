/**
 * ============================================================
 *  API LAYER — talks to the FastAPI backend (backend/main.py).
 *  Every function resolves to the same shapes the UI consumes,
 *  so components never touch fetch() directly.
 * ============================================================
 */

import type {
  AnnotatedVideo,
  ConfidencePoint,
  ExportKind,
  MatchRecord,
  ReferenceStatus,
  ScanProgress,
  ScanResults,
  ScanSettings,
  SourceVideo,
  VideoFormat,
} from "./types";
import type { ReferencePhoto } from "./case-store";

export const DEFAULTS = {
  threshold: 0.45,
  faceWeight: 0.8,
  minReferencePhotos: 1,
  maxReferencePhotos: 5,
  version: "v2.4.1",
};

const envBase = import.meta.env["VITE_API_BASE"] as string | undefined;
const API_BASE = (envBase ?? "http://localhost:8000").replace(/\/$/, "");

const GRADIENTS = [
  "from-primary/40 via-violet/30 to-background",
  "from-violet/40 via-primary/25 to-background",
  "from-primary-bright/40 via-primary/20 to-background",
  "from-success/25 via-primary/30 to-background",
  "from-warning/25 via-violet/30 to-background",
];

const grad = (i: number): string => GRADIENTS[i % GRADIENTS.length] ?? GRADIENTS[0]!;

/** The currently active scan on the backend, used by polls and exports. */
const current: { scanId: string | null } = { scanId: null };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      /* non-JSON error body — keep the generic message */
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

export function detectFormat(fileName: string): VideoFormat {
  const ext = fileName.split(".").pop()?.toUpperCase();
  if (ext === "AVI" || ext === "MOV" || ext === "MKV") return ext;
  return "MP4";
}

function mediaUrl(scanId: string, file: string): string {
  return `${API_BASE}/api/scan/${scanId}/media/${encodeURIComponent(file)}`;
}

/** Map a backend match dict onto the MatchRecord type the UI renders. */
function toMatch(raw: ServerMatch, index: number): MatchRecord {
  return {
    id: raw.id,
    rank: index + 1,
    videoId: raw.video_id,
    videoName: raw.video_name,
    startSeconds: raw.start_sec,
    endSeconds: raw.end_sec,
    faceScore: raw.face,
    poseScore: raw.pose,
    fusedScore: raw.fused,
    clothScore: raw.cloth,
    screenshotGradient: grad(index),
    ...(raw.screenshot ? { screenshotUrl: mediaUrl(raw.scan_id, raw.screenshot) } : {}),
    ...(raw.annotated_video ? { annotatedVideoUrl: mediaUrl(raw.scan_id, raw.annotated_video) } : {}),
  };
}

/** POST /api/reference — uploads the photos and builds the reference profile. */
export async function buildReference(photos: ReferencePhoto[]): Promise<ReferenceStatus> {
  if (photos.length === 0) throw new Error("At least one reference photo is required.");
  const form = new FormData();
  for (const p of photos) {
    const blob = await fetch(p.url).then((r) => r.blob());
    form.append("photos", blob, p.name);
  }
  const res = await fetch(`${API_BASE}/api/reference`, { method: "POST", body: form });
  if (!res.ok) {
    let message = "Reference build failed.";
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  return (await res.json()) as ReferenceStatus;
}

/** GET /api/reference/profile — downloads the portable .npz profile blob. */
export async function saveProfile(): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/reference/profile`);
  if (!res.ok) {
    let message = "Profile download failed.";
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  return await res.blob();
}

/** POST /api/videos — uploads and probes source footage. */
export async function registerVideos(files: File[]): Promise<SourceVideo[]> {
  if (files.length === 0) return [];
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  return await request<SourceVideo[]>("/api/videos", { method: "POST", body: form });
}

/**
 * POST /api/scan + polling — streams progress via onProgress callbacks and
 * resolves the full ScanResults once the backend finishes. Cancel aborts both
 * the polling loop and the backend scan.
 */
export function runScan(
  videos: SourceVideo[],
  settings: ScanSettings,
  onProgress: (p: ScanProgress) => void,
): { promise: Promise<ScanResults>; cancel: () => void } {
  let cancelled = false;
  let running = true;

  const promise = (async () => {
    if (videos.length === 0) throw new Error("No video sources queued for scanning.");

    const scan = await request<{ scan_id: string }>("/api/scan", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        threshold: settings.threshold,
        face_weight: settings.faceWeight,
        videos: videos.map((v) => v.id),
      }),
    });
    current.scanId = scan.scan_id;

    for (;;) {
      if (cancelled) throw new Error("Scan cancelled.");
      const snap = await request<ScanSnapshot>(`/api/scan/${scan.scan_id}`);

      const revealed = (snap.matches ?? []).slice().sort((a, b) => b.fused - a.fused);
      onProgress({
        percent: snap.percent,
        currentVideoName: snap.current_video_name ?? "",
        currentIndex: snap.current_index ?? 0,
        totalVideos: snap.total_videos ?? videos.length,
        latestMatch: revealed.length ? toMatch(revealed[0]!, 0) : null,
        done: snap.status === "done",
        ...(snap.phase ? { phase: snap.phase } : {}),
      });

      if (snap.status === "error") {
        throw new Error(snap.error || "Scan failed on the backend.");
      }
      if (snap.status === "cancelled") {
        throw new Error("Scan cancelled.");
      }
      if (snap.status === "done") {
        running = false;
        return buildResults(snap, scan.scan_id);
      }

      await new Promise((r) => setTimeout(r, 750));
    }
  })();

  return {
    promise,
    cancel: () => {
      cancelled = true;
      if (current.scanId) {
        void fetch(`${API_BASE}/api/scan/${current.scanId}/cancel`, { method: "POST" }).catch(
          () => undefined,
        );
      }
    },
  };
}

function buildResults(snap: ScanSnapshot, scanId: string): ScanResults {
  const matches = (snap.matches ?? []).map((m) => toMatch(m, 0));
  matches.forEach((m, i) => (m.rank = i + 1));

  const durations = matches.reduce((a, m) => a + (m.endSeconds - m.startSeconds), 0);
  const confidences = matches.map((m) => m.fusedScore);

  const annotated: AnnotatedVideo[] = (snap.annotated ?? []).map((a, i) => ({
    id: a.id,
    sourceName: a.source_name,
    matches: a.matches,
    posterGradient: grad(i),
    videoUrl: mediaUrl(scanId, a.video),
  }));

  const reelUrl = snap.reel ? mediaUrl(scanId, snap.reel) : undefined;

  return {
    metrics: {
      totalMatches: matches.length,
      totalSightingSeconds: durations,
      videosScanned: snap.annotated?.length ?? videosFor(snap),
      highestConfidence: confidences.length ? Math.max(...confidences) : 0,
      averageConfidence: confidences.length
        ? confidences.reduce((a, b) => a + b, 0) / confidences.length
        : 0,
    },
    matches,
    confidenceSeries: snap.confidence_series ?? [],
    annotated,
    ...(snap.color_labels ? { colorLabels: snap.color_labels } : {}),
    ...(snap.diagnostics
      ? {
          diagnostics: {
            framesProcessed: snap.diagnostics.frames_processed ?? 0,
            framesWithFace: snap.diagnostics.frames_with_face ?? 0,
            bestFaceSim: snap.diagnostics.best_face_sim ?? 0,
            bestFused: snap.diagnostics.best_fused ?? 0,
            faceThr: snap.diagnostics.face_thr ?? 0,
            threshold: snap.diagnostics.threshold ?? 0,
          },
        }
      : {}),
    refinedReady: !!snap.reel || (snap.matches ?? []).length > 0,
    reel: {
      id: "reel",
      sourceName: "highlight_reel.mp4",
      matches: matches.length,
      posterGradient: grad(1),
      ...(reelUrl ? { videoUrl: reelUrl } : {}),
    },
  };
}

function videosFor(snap: ScanSnapshot): number {
  const names = snap.matches?.map((m) => m.video_name) ?? [];
  return new Set(names).size || (snap.total_videos ?? 0);
}

/** GET /api/scan/{id}/exports/:kind — triggers a browser download. */
export async function requestExport(kind: ExportKind): Promise<{ fileName: string }> {
  const id = current.scanId;
  if (!id) throw new Error("No finished scan to export.");
  const map: Record<ExportKind, string> = {
    csv: "target-id-matches.csv",
    pdf: "target-id-report.pdf",
    "evidence-zip": "target-id-evidence.zip",
    "annotated-zip": "target-id-annotated.zip",
    reel: "target-id-highlight-reel.mp4",
  };
  const fileName = map[kind];
  const a = document.createElement("a");
  a.href = `${API_BASE}/api/scan/${id}/exports/${kind}`;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  return { fileName };
}

/**
 * Download the live-refined profile (.npz) from a finished scan so the next
 * scan on this target starts smarter (sharpened face, actual observed
 * clothing/posture).
 */
export async function requestRefinedProfile(): Promise<{ fileName: string }> {
  const id = current.scanId;
  if (!id) throw new Error("No finished scan to export.");
  const fileName = "refined_profile.npz";
  const a = document.createElement("a");
  a.href = `${API_BASE}/api/scan/${id}/refined-profile`;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  return { fileName };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let v = bytes / 1024;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v.toFixed(1)} ${units[u]}`;
}

export function formatClock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

/* ---------- backend wire shapes (mirrors backend/engine.py) ---------- */

export type ServerMatch = {
  id: string;
  scan_id: string;
  video_id: string;
  video_name: string;
  start_sec: number;
  end_sec: number;
  face: number;
  pose: number;
  cloth: number;
  fused: number;
  screenshot: string | null;
  annotated_video: string | null;
};

export type ScanSnapshot = {
  status: "queued" | "running" | "done" | "error" | "cancelled";
  percent: number;
  phase?: string | null;
  current_video_name?: string | null;
  current_index?: number | null;
  total_videos?: number | null;
  error?: string | null;
  diagnostics?: {
    frames_processed?: number;
    frames_with_face?: number;
    best_face_sim?: number;
    best_fused?: number;
    face_thr?: number;
    threshold?: number;
  };
  matches?: ServerMatch[];
  confidence_series?: ConfidencePoint[];
  annotated?: { id: string; source_name: string; matches: number; video: string }[];
  reel?: string | null;
  color_labels?: string[];
};