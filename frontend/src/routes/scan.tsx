import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, ArrowLeft, ArrowRight, Radar, Sparkles } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { AppShell } from "@/components/forensic/app-shell";
import { ActionButton, VideoSurface } from "@/components/forensic/controls";
import {
  EmptyState,
  ErrorBlock,
  GlassCard,
  GlowProgress,
  Pill,
  SectionTitle,
} from "@/components/forensic/primitives";
import { formatClock, runScan } from "@/lib/api";
import { canEnterStepFor, hydrateSnapshot, useCase } from "@/lib/case-store";
import type { ScanProgress } from "@/lib/types";

export const Route = createFileRoute("/scan")({
  beforeLoad: () => {
    if (typeof window !== "undefined" && !canEnterStepFor(hydrateSnapshot(), 3)) {
      throw redirect({ to: "/" });
    }
  },
  head: () => ({
    meta: [
      { title: "AI Scan — Video Target ID" },
      {
        name: "description",
        content: "Live forensic scan progress with real-time match previews and confidence scoring.",
      },
      { property: "og:title", content: "AI Scan — Video Target ID" },
      {
        property: "og:description",
        content: "Watch the AI sweep every queued video for target sightings in real time.",
      },
    ],
  }),
  component: ScanPage,
});

function ScanPage() {
  const state = useCase();
  const navigate = useNavigate();
  const [progress, setProgress] = useState<ScanProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);

  const start = useCallback(() => {
    setError(null);
    setProgress(null);
    setRunning(true);
    const { promise, cancel } = runScan(
      state.videos,
      { threshold: state.threshold, faceWeight: state.faceWeight },
      (p) =>
        setProgress((prev) => {
          if (prev && p.percent < prev.percent) return prev;
          return p;
        }),
    );
    cancelRef.current = cancel;
    promise
      .then((results) => state.set({ results }))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Scan failed."))
      .finally(() => setRunning(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.videos, state.threshold, state.faceWeight]);

  useEffect(() => {
    if (!state.results) start();
    return () => cancelRef.current?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const latest = progress?.latestMatch ?? null;
  const percent = progress?.percent ?? (state.results ? 100 : 0);

  return (
    <AppShell>
      <SectionTitle
        eyebrow="Step 03 / 04"
        title="AI Scan"
        description="Face and pose embeddings are fused per frame and compared against the reference profile."
      />

      <GlassCard className="space-y-5">
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone="info" mono>
            threshold {state.threshold.toFixed(2)}
          </Pill>
          <Pill tone="info" mono>
            face {state.faceWeight.toFixed(2)}
          </Pill>
          <Pill tone="info" mono>
            pose {Math.max(0, 1 - state.faceWeight - 0.1).toFixed(2)}
          </Pill>
          <Pill tone="info" mono>
            cloth {Math.min(0.1, 1 - state.faceWeight).toFixed(2)}
          </Pill>
          <Pill tone="neutral" mono>
            {state.videos.length} sources
          </Pill>
          {running ? (
            <Pill tone="warn">
              <Activity className="size-3" /> Scanning
            </Pill>
          ) : state.results ? (
            <Pill tone="ok">Complete</Pill>
          ) : null}
        </div>

        <GlowProgress
          percent={percent}
          label={
            progress?.phase ??
            (progress
              ? `Scanning ${progress.currentVideoName} (${progress.currentIndex}/${progress.totalVideos})`
              : state.results
                ? "Scan complete"
                : "Initializing scan engine")
          }
        />

        {error ? <ErrorBlock message={error} onRetry={start} /> : null}
      </GlassCard>

      <div className="grid gap-4 lg:grid-cols-2">
        <GlassCard delay={0.06} className="space-y-3">
          <h3 className="flex items-center gap-2 text-sm font-semibold tracking-wide uppercase">
            <Sparkles className="size-4 text-primary-bright" /> Live match preview
          </h3>
          <AnimatePresence mode="wait">
            {latest ? (
              <motion.div
                key={latest.id}
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-3"
              >
                <VideoSurface
                  gradient={latest.screenshotGradient}
                  src={latest.screenshotUrl}
                  caption={`${formatClock(latest.startSeconds)} → ${formatClock(latest.endSeconds)}`}
                />
                <div className="flex flex-wrap items-center gap-2">
                  <Pill tone="ok" mono>
                    fused {latest.fusedScore.toFixed(3)}
                  </Pill>
                  <Pill tone="info" mono>
                    face {latest.faceScore.toFixed(3)}
                  </Pill>
                  <Pill tone="neutral" mono>
                    pose {latest.poseScore.toFixed(3)}
                  </Pill>
                  <Pill tone="neutral" mono>
                    cloth {latest.clothScore?.toFixed(3) ?? "—"}
                  </Pill>
                </div>
                <p className="truncate font-mono text-[11px] text-muted-foreground">
                  {latest.videoName}
                </p>
              </motion.div>
            ) : (
              <EmptyState
                icon={Radar}
                title="No sighting yet"
                description="Matched frames appear here the moment fused confidence crosses your threshold."
              />
            )}
          </AnimatePresence>
        </GlassCard>

        <GlassCard delay={0.12} className="space-y-3">
          <h3 className="text-sm font-semibold tracking-wide uppercase">Queue</h3>
          <ul className="space-y-2">
            {state.videos.map((v, i) => {
              const idx = progress?.currentIndex ?? 0;
              const done = i + 1 < idx || percent === 100;
              return (
                <li
                  key={v.id}
                  className="flex items-center justify-between rounded-xl border border-border bg-muted/10 px-3 py-2 text-sm"
                >
                  <span className="min-w-0 flex-1 truncate">{v.name}</span>
                  <Pill tone={done ? "ok" : i + 1 === idx ? "warn" : "neutral"} mono>
                    {done ? "done" : i + 1 === idx ? "active" : "queued"}
                  </Pill>
                </li>
              );
            })}
          </ul>
        </GlassCard>
      </div>

      <div className="flex flex-wrap justify-between gap-3">
        <ActionButton
          variant="glass"
          onClick={() => {
            cancelRef.current?.();
            state.set({ step: 2 });
            void navigate({ to: "/source" });
          }}
        >
          <ArrowLeft className="size-4" /> Back
        </ActionButton>
        <ActionButton
          variant="action"
          disabled={!state.results}
          onClick={() => {
            state.set({ step: 4 });
            void navigate({ to: "/results" });
          }}
        >
          View Results <ArrowRight className="size-4" />
        </ActionButton>
      </div>
    </AppShell>
  );
}
