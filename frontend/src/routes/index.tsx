import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, ImagePlus, Save, Settings2, Trash2, ArrowRight } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { AppShell } from "@/components/forensic/app-shell";
import { ActionButton, DropZone } from "@/components/forensic/controls";
import {
  EmptyState,
  GlassCard,
  LoadingBlock,
  ErrorBlock,
  Pill,
  SectionTitle,
} from "@/components/forensic/primitives";
import { Checkbox } from "@/components/ui/checkbox";
import { buildReference, saveProfile, DEFAULTS } from "@/lib/api";
import { useCase, type ReferencePhoto } from "@/lib/case-store";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Target Setup — Video Target ID" },
      {
        name: "description",
        content:
          "Upload reference photos and build a forensic face + pose reference profile for AI video identification.",
      },
      { property: "og:title", content: "Target Setup — Video Target ID" },
      {
        property: "og:description",
        content: "Build a forensic reference profile from 1–5 target photos.",
      },
    ],
  }),
  component: TargetSetupPage,
});

function TargetSetupPage() {
  const state = useCase();
  const navigate = useNavigate();
  const [building, setBuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedFlash, setSavedFlash] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);

  const addPhotos = (files: File[]) => {
    const room = DEFAULTS.maxReferencePhotos - state.photos.length;
    if (room <= 0) {
      toast.error(`Maximum ${DEFAULTS.maxReferencePhotos} reference photos.`);
      return;
    }
    const next: ReferencePhoto[] = files.slice(0, room).map((f, i) => ({
      id: `${Date.now()}-${i}`,
      name: f.name,
      url: URL.createObjectURL(f),
    }));
    state.set({ photos: [...state.photos, ...next] });
  };

  const onBuild = async () => {
    setBuilding(true);
    setError(null);
    try {
      const reference = await buildReference(state.photos);
      state.set({ reference });
      setSavedFlash(true);
      toast.success("Reference profile built");
      setTimeout(() => setSavedFlash(false), 2200);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Reference build failed.");
    } finally {
      setBuilding(false);
    }
  };

  const onSaveProfile = async () => {
    setSavingProfile(true);
    try {
      const blob = await saveProfile();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const ext = blob.type.includes("octet-stream") ? "npz" : "json";
      a.download = `target-reference-profile.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setSavingProfile(false);
    }
  };

  const canContinue = state.reference.faceReady && state.authorized;

  return (
    <AppShell>
      <SectionTitle
        eyebrow="Step 01 / 04"
        title="Target Setup"
        description="Provide 1–5 clear reference photos of the target. Face and pose embeddings are derived from these images."
      />

      <GlassCard className="space-y-5">
        <DropZone
          label="Drop reference photos or click to browse"
          hint={`JPG or PNG · up to ${DEFAULTS.maxReferencePhotos} images · frontal, well-lit shots score highest`}
          accept="image/*"
          onFiles={addPhotos}
        />

        {state.photos.length === 0 ? (
          <EmptyState
            icon={ImagePlus}
            title="No reference photos yet"
            description="At least one photo is required before a reference profile can be built."
          />
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            <AnimatePresence>
              {state.photos.map((p) => (
                <motion.div
                  key={p.id}
                  layout
                  initial={{ opacity: 0, scale: 0.92 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="group relative overflow-hidden rounded-xl border border-border"
                >
                  <img
                    src={p.url}
                    alt={`Reference photo ${p.name}`}
                    className="aspect-square w-full object-cover"
                  />
                  <button
                    aria-label={`Remove ${p.name}`}
                    onClick={() =>
                      state.set({ photos: state.photos.filter((x) => x.id !== p.id) })
                    }
                    className="absolute top-1.5 right-1.5 rounded-lg border border-border bg-background/70 p-1.5 text-destructive opacity-0 backdrop-blur transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}

        {error ? <ErrorBlock message={error} onRetry={onBuild} /> : null}
        {building ? <LoadingBlock label="Extracting face and pose embeddings" /> : null}

        <div className="flex flex-wrap items-center gap-3">
          <ActionButton
            variant="action"
            onClick={onBuild}
            disabled={state.photos.length === 0 || building}
          >
            <Settings2 className="size-4" /> Build Reference
          </ActionButton>

          {state.reference.faceReady ? (
            <ActionButton variant="glass" onClick={onSaveProfile} disabled={savingProfile}>
              <Save className="size-4" /> Save Profile
            </ActionButton>
          ) : null}

          <AnimatePresence>
            {savedFlash ? (
              <motion.span
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2 text-sm text-success"
              >
                <CheckCircle2 className="size-4" /> Reference locked in
              </motion.span>
            ) : null}
          </AnimatePresence>

          {state.reference.faceReady ? (
            <Pill tone="ok" mono>
              {state.reference.embeddings} vectors
            </Pill>
          ) : null}
        </div>
      </GlassCard>

      <GlassCard delay={0.08} className="space-y-4">
        <label className="flex items-start gap-3 text-sm">
          <Checkbox
            checked={state.authorized}
            onCheckedChange={(v) => state.set({ authorized: v === true })}
            aria-label="Authorization confirmation"
          />
          <span className="text-muted-foreground">
            I confirm this analysis is lawfully authorized and the reference subject is covered by an
            active case warrant or consent record.
          </span>
        </label>

        <div className="flex justify-end">
          <ActionButton
            variant="action"
            disabled={!canContinue}
            onClick={() => {
              state.set({ step: 2 });
              void navigate({ to: "/source" });
            }}
          >
            Next: Video Source <ArrowRight className="size-4" />
          </ActionButton>
        </div>
      </GlassCard>
    </AppShell>
  );
}
