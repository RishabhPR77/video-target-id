import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, ArrowRight, Film, Trash2 } from "lucide-react";
import { useState } from "react";

import { AppShell } from "@/components/forensic/app-shell";
import { ActionButton, DropZone } from "@/components/forensic/controls";
import {
  EmptyState,
  ErrorBlock,
  GlassCard,
  Pill,
  SectionTitle,
  SkeletonRow,
} from "@/components/forensic/primitives";
import { formatBytes, formatDuration, registerVideos } from "@/lib/api";
import { canEnterStepFor, hydrateSnapshot, useCase } from "@/lib/case-store";

export const Route = createFileRoute("/source")({
  beforeLoad: () => {
    if (typeof window !== "undefined" && !canEnterStepFor(hydrateSnapshot(), 2)) {
      throw redirect({ to: "/" });
    }
  },
  head: () => ({
    meta: [
      { title: "Video Source — Video Target ID" },
      {
        name: "description",
        content: "Queue MP4, AVI, MOV and MKV surveillance footage for forensic AI scanning.",
      },
      { property: "og:title", content: "Video Source — Video Target ID" },
      {
        property: "og:description",
        content: "Drag and drop multiple video sources into the forensic scan queue.",
      },
    ],
  }),
  component: VideoSourcePage,
});

function VideoSourcePage() {
  const state = useCase();
  const navigate = useNavigate();
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onFiles = async (files: File[]) => {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      const registered = await registerVideos(files);
      const existing = new Set(state.videos.map((v) => v.name));
      state.set({
        videos: [...state.videos, ...registered.filter((v) => !existing.has(v.name))],
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <AppShell>
      <SectionTitle
        eyebrow="Step 02 / 04"
        title="Video Source"
        description="Add every source of footage for this case. Files stay local until the scan is dispatched."
      />

      <GlassCard className="space-y-5">
        <DropZone
          label="Drop video files or click to browse"
          hint="MP4 · AVI · MOV · MKV · multiple files supported"
          accept="video/*"
          onFiles={onFiles}
        />

        {error ? <ErrorBlock message={error} /> : null}

        {uploading ? (
          <div className="grid gap-3 sm:grid-cols-2">
            {[0, 1].map((i) => (
              <div key={i} className="glass-panel space-y-3 p-4">
                <SkeletonRow className="h-4 w-2/3" />
                <SkeletonRow className="h-3 w-1/3" />
              </div>
            ))}
          </div>
        ) : null}

        {!uploading && state.videos.length === 0 ? (
          <EmptyState
            icon={Film}
            title="Scan queue is empty"
            description="Add at least one video file to continue to the AI scan stage."
          />
        ) : null}

        <div className="grid gap-3 sm:grid-cols-2">
          <AnimatePresence>
            {state.videos.map((v, i) => (
              <motion.div
                key={v.id}
                layout
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96 }}
                transition={{ delay: i * 0.04 }}
                className="glass-panel hover-lift flex items-center gap-3 p-4"
              >
                <span className="rounded-xl border border-border bg-primary/10 p-2 text-primary-bright">
                  <Film className="size-4" />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{v.name}</p>
                  <p className="font-mono text-[11px] text-muted-foreground">
                    {formatBytes(v.sizeBytes)} · {formatDuration(v.durationSeconds)}
                  </p>
                </div>
                <Pill tone="info" mono>
                  {v.format}
                </Pill>
                <button
                  aria-label={`Remove ${v.name}`}
                  onClick={() => state.set({ videos: state.videos.filter((x) => x.id !== v.id) })}
                  className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:text-destructive"
                >
                  <Trash2 className="size-4" />
                </button>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </GlassCard>

      <div className="flex flex-wrap justify-between gap-3">
        <ActionButton
          variant="glass"
          onClick={() => {
            state.set({ step: 1 });
            void navigate({ to: "/" });
          }}
        >
          <ArrowLeft className="size-4" /> Back
        </ActionButton>
        <ActionButton
          variant="action"
          disabled={state.videos.length === 0}
          onClick={() => {
            state.set({ step: 3 });
            void navigate({ to: "/scan" });
          }}
        >
          Next: AI Scan <ArrowRight className="size-4" />
        </ActionButton>
      </div>
    </AppShell>
  );
}
