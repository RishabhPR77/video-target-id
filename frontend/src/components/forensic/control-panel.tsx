import { Check, Circle, Play, RotateCcw, ScanFace, PersonStanding, SlidersHorizontal } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";

import { ActionButton } from "./controls";
import { GlassCard, Pill } from "./primitives";
import { useCase } from "@/lib/case-store";
import type { StepId } from "@/lib/types";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

const STEPS: { id: StepId; label: string; path: "/" | "/source" | "/scan" | "/results" }[] = [
  { id: 1, label: "Target Setup", path: "/" },
  { id: 2, label: "Video Source", path: "/source" },
  { id: 3, label: "AI Scan", path: "/scan" },
  { id: 4, label: "Results", path: "/results" },
];

export function ControlPanel() {
  const state = useCase();
  const navigate = useNavigate();
  const poseWeight = 1 - state.faceWeight;

  return (
    <aside className="flex w-full flex-col gap-4 lg:w-[300px] lg:shrink-0">
      <GlassCard className="space-y-4">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="size-4 text-primary-bright" />
          <h2 className="text-sm font-semibold tracking-wide uppercase">Control Panel</h2>
        </div>

        <ol className="space-y-1.5">
          {STEPS.map((s) => {
            const done = state.step > s.id;
            const active = state.step === s.id;
            const reachable = state.canEnterStep(s.id);
            return (
              <li key={s.id}>
                <button
                  disabled={!reachable}
                  onClick={() => {
                    state.set({ step: s.id });
                    void navigate({ to: s.path });
                  }}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left text-sm transition-all focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
                    active
                      ? "border-primary-bright/40 bg-primary/10 text-foreground glow-ring"
                      : done
                        ? "border-success/25 bg-success/5 text-foreground"
                        : "border-border bg-muted/10 text-muted-foreground",
                    !reachable && "cursor-not-allowed opacity-50",
                  )}
                >
                  <span
                    className={cn(
                      "flex size-6 shrink-0 items-center justify-center rounded-full border",
                      active
                        ? "border-primary-bright text-primary-bright"
                        : done
                          ? "border-success text-success"
                          : "border-muted-foreground/40 text-muted-foreground",
                    )}
                  >
                    {done ? (
                      <Check className="size-3.5" />
                    ) : active ? (
                      <Play className="size-3" />
                    ) : (
                      <Circle className="size-2.5" />
                    )}
                  </span>
                  <span className="flex-1">{s.label}</span>
                  <span className="font-mono text-[11px] text-muted-foreground">0{s.id}</span>
                </button>
              </li>
            );
          })}
        </ol>
      </GlassCard>

      <GlassCard delay={0.05} className="space-y-6">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label htmlFor="threshold" className="text-xs tracking-wide text-muted-foreground uppercase">
              Detection Threshold
            </label>
            <span className="font-mono text-sm text-primary-bright">
              {state.threshold.toFixed(2)}
            </span>
          </div>
          <Slider
            id="threshold"
            min={0.3}
            max={0.95}
            step={0.01}
            value={[state.threshold]}
            onValueChange={([v]) => state.set({ threshold: v ?? state.threshold })}
          />
          <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
            <span>0.30</span>
            <span>0.95</span>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label htmlFor="face-weight" className="text-xs tracking-wide text-muted-foreground uppercase">
              Face Weight
            </label>
            <span className="font-mono text-sm text-primary-bright">
              {state.faceWeight.toFixed(2)}
            </span>
          </div>
          <Slider
            id="face-weight"
            min={0.7}
            max={1}
            step={0.01}
            value={[state.faceWeight]}
            onValueChange={([v]) => state.set({ faceWeight: v ?? state.faceWeight })}
          />
          <p className="font-mono text-[11px] text-muted-foreground">
            Pose Weight = 1 − face ={" "}
            <span className="text-violet">{poseWeight.toFixed(2)}</span>
          </p>
        </div>
      </GlassCard>

      <GlassCard delay={0.1} className="space-y-3">
        <h3 className="text-xs tracking-wide text-muted-foreground uppercase">Reference Status</h3>
        <div className="flex items-center justify-between rounded-xl border border-border bg-muted/10 px-3 py-2">
          <span className="flex items-center gap-2 text-sm">
            <ScanFace className="size-4 text-primary-bright" /> Face
          </span>
          <Pill tone={state.reference.faceReady ? "ok" : "error"}>
            {state.reference.faceReady ? "✓ Ready" : "✗ Missing"}
          </Pill>
        </div>
        <div className="flex items-center justify-between rounded-xl border border-border bg-muted/10 px-3 py-2">
          <span className="flex items-center gap-2 text-sm">
            <PersonStanding className="size-4 text-violet" /> Pose
          </span>
          <Pill tone={state.reference.poseReady ? "ok" : "warn"}>
            {state.reference.poseReady ? "✓ Ready" : "⚠ Partial"}
          </Pill>
        </div>
        <p className="font-mono text-[11px] text-muted-foreground">
          {state.reference.embeddings} embeddings cached
        </p>
      </GlassCard>

      <ActionButton
        variant="danger"
        onClick={() => {
          state.reset();
          void navigate({ to: "/" });
        }}
      >
        <RotateCcw className="size-4" /> New Case
      </ActionButton>
    </aside>
  );
}
