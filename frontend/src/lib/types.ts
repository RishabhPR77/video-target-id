/**
 * ============================================================
 *  MOCK DATA CONTRACT — Video Target ID
 * ------------------------------------------------------------
 *  These types describe the shape a real REST API must return.
 *  Components consume ONLY these types via `src/lib/api.ts`.
 *  Nothing here is wired to a backend yet.
 * ============================================================
 */

export type StepId = 1 | 2 | 3 | 4;

export type ReferenceStatus = {
  faceReady: boolean;
  poseReady: boolean;
  clothReady: boolean;
  embeddings: number;
  builtAt: string | null;
};

export type VideoFormat = "MP4" | "AVI" | "MOV" | "MKV";

export type SourceVideo = {
  id: string;
  name: string;
  format: VideoFormat;
  sizeBytes: number;
  durationSeconds: number;
};

export type ScanSettings = {
  threshold: number;
  faceWeight: number;
};

export type MatchRecord = {
  id: string;
  rank: number;
  videoId: string;
  videoName: string;
  startSeconds: number;
  endSeconds: number;
  faceScore: number;
  poseScore: number;
  fusedScore: number;
  /** Secondary corroborating signal: upper-body clothing similarity score. */
  clothScore?: number;
  /** Gradient token used as a fallback when no screenshot URL is available. */
  screenshotGradient: string;
  /** URL of the annotated frame screenshot served by the backend. */
  screenshotUrl?: string;
  /** URL of the source's annotated video (for seekable playback). */
  annotatedVideoUrl?: string;
};

export type ConfidencePoint = {
  t: number;
  confidence: number;
};

export type AnnotatedVideo = {
  id: string;
  sourceName: string;
  matches: number;
  posterGradient: string;
  /** URL of the annotated video file served by the backend. */
  videoUrl?: string;
};

export type ScanMetrics = {
  totalMatches: number;
  totalSightingSeconds: number;
  videosScanned: number;
  highestConfidence: number;
  averageConfidence: number;
};

export type ScanResults = {
  metrics: ScanMetrics;
  matches: MatchRecord[];
  confidenceSeries: ConfidencePoint[];
  annotated: AnnotatedVideo[];
  reel: AnnotatedVideo;
  /** Dominant observed upper-body clothing colors of the target (from the live-refined profile). */
  colorLabels?: string[];
  /** True when a refined profile (smarter future scans) is available to download. */
  refinedReady?: boolean;
  /** Engine diagnostics surfaced when a scan finishes with zero matches. */
  diagnostics?: {
    framesProcessed: number;
    framesWithFace: number;
    bestFaceSim: number;
    bestFused: number;
    faceThr: number;
    threshold: number;
  };
};

export type ScanProgress = {
  percent: number;
  currentVideoName: string;
  currentIndex: number;
  totalVideos: number;
  latestMatch: MatchRecord | null;
  done: boolean;
  /** Human-readable phase text from the backend (e.g. "Scanning clip.mp4 — 1/2"). */
  phase?: string;
};

export type ExportKind = "csv" | "pdf" | "evidence-zip" | "annotated-zip" | "reel";

export type AsyncState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "success"; data: T };
