import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  ArrowDownUp,
  BarChart3,
  Clock,
  FileSpreadsheet,
  FileText,
  Film,
  Gauge,
  Package,
  Play,
  RotateCcw,
  Sigma,
  Target,
  Video,
} from "lucide-react";
import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { toast } from "sonner";

import { AppShell } from "@/components/forensic/app-shell";
import { ActionButton, DownloadButton, VideoSurface } from "@/components/forensic/controls";
import {
  EmptyState,
  GlassCard,
  MetricCard,
  Pill,
  SectionTitle,
} from "@/components/forensic/primitives";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatClock, formatDuration, requestExport, requestRefinedProfile } from "@/lib/api";
import { canEnterStepFor, hydrateSnapshot, useCase } from "@/lib/case-store";
import type { ExportKind, MatchRecord } from "@/lib/types";

export const Route = createFileRoute("/results")({
  beforeLoad: () => {
    if (typeof window !== "undefined" && !canEnterStepFor(hydrateSnapshot(), 4)) {
      throw redirect({ to: "/" });
    }
  },
  head: () => ({
    meta: [
      { title: "Results Dashboard — Video Target ID" },
      {
        name: "description",
        content:
          "Forensic results dashboard: match timeline, confidence graph, annotated videos and evidence exports.",
      },
      { property: "og:title", content: "Results Dashboard — Video Target ID" },
      {
        property: "og:description",
        content: "Review every target sighting with confidence scores and exportable evidence.",
      },
    ],
  }),
  component: ResultsPage,
});

const EXPORTS: { kind: ExportKind; label: string; icon: typeof FileText }[] = [
  { kind: "csv", label: "Export CSV", icon: FileSpreadsheet },
  { kind: "pdf", label: "Export PDF Report", icon: FileText },
  { kind: "evidence-zip", label: "Evidence ZIP", icon: Package },
  { kind: "annotated-zip", label: "Annotated ZIP", icon: Film },
  { kind: "reel", label: "Highlight Reel", icon: Video },
];

function ResultsPage() {
  const state = useCase();
  const navigate = useNavigate();
  const results = state.results;
  const [playing, setPlaying] = useState<MatchRecord | null>(null);
  const [pendingExport, setPendingExport] = useState<ExportKind | null>(null);
  const [query, setQuery] = useState("");
  const [sortDesc, setSortDesc] = useState(true);
  const [downloadRefined, setDownloadRefined] = useState(false);

  const onRefined = async () => {
    setDownloadRefined(true);
    try {
      const { fileName } = await requestRefinedProfile();
      toast.success(`${fileName} ready — next scans start smarter`);
    } catch {
      toast.error("No refined profile available.");
    } finally {
      setDownloadRefined(false);
    }
  };

  const rows = useMemo(() => {
    const list = (results?.matches ?? []).filter(
      (m) =>
        m.videoName.toLowerCase().includes(query.toLowerCase()) ||
        String(m.fusedScore).includes(query),
    );
    return [...list].sort((a, b) =>
      sortDesc ? b.fusedScore - a.fusedScore : a.fusedScore - b.fusedScore,
    );
  }, [results, query, sortDesc]);

  const onExport = async (kind: ExportKind) => {
    setPendingExport(kind);
    try {
      const { fileName } = await requestExport(kind);
      toast.success(`${fileName} is ready`);
    } catch {
      toast.error("Export failed. Try again.");
    } finally {
      setPendingExport(null);
    }
  };

  if (!results) {
    return (
      <AppShell>
        <EmptyState
          icon={Target}
          title="No results in this case"
          description="Run an AI scan to populate the results dashboard."
        />
      </AppShell>
    );
  }

  const m = results.metrics;

  return (
    <AppShell>
      <SectionTitle
        eyebrow="Step 04 / 04"
        title="Results Dashboard"
        description="Every sighting, ranked by fused confidence, with exportable evidence packages."
      />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <MetricCard label="Total Matches" value={m.totalMatches} icon={Target} />
        <MetricCard
          label="Sighting Duration"
          value={m.totalSightingSeconds}
          icon={Clock}
          format={(v) => formatDuration(v)}
          delay={0.05}
        />
        <MetricCard label="Videos Scanned" value={m.videosScanned} icon={Film} delay={0.1} />
        <MetricCard
          label="Highest Confidence"
          value={m.highestConfidence * 100}
          suffix="%"
          icon={Gauge}
          format={(v) => v.toFixed(1)}
          delay={0.15}
        />
        <MetricCard
          label="Average Confidence"
          value={m.averageConfidence * 100}
          suffix="%"
          icon={Sigma}
          format={(v) => v.toFixed(1)}
          delay={0.2}
        />
      </div>

      {results.colorLabels && results.colorLabels.length > 0 ? (
        <GlassCard className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
              Target appearance
            </span>
            {results.colorLabels.map((label) => (
              <Pill key={label} tone="warn">
                {label}
              </Pill>
            ))}
            <span className="text-xs text-muted-foreground">
              best guess from the observed footage
            </span>
          </div>
          <DownloadButton
            label="Save refined profile"
            pending={downloadRefined}
            onClick={onRefined}
          />
        </GlassCard>
      ) : null}

      <Tabs defaultValue="tracked" className="space-y-4">
        <TabsList className="glass-panel h-auto flex-wrap gap-1 p-1">
          <TabsTrigger value="tracked">Tracked Video</TabsTrigger>
          <TabsTrigger value="details">Match Details</TabsTrigger>
          <TabsTrigger value="graph">Confidence Graph</TabsTrigger>
          <TabsTrigger value="raw">Raw Data</TabsTrigger>
        </TabsList>

        <TabsContent value="tracked" className="space-y-4">
          <GlassCard className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-sm font-semibold tracking-wide uppercase">Highlight reel</h3>
              <DownloadButton
                label="Download reel"
                pending={pendingExport === "reel"}
                onClick={() => onExport("reel")}
              />
            </div>
            <VideoSurface
              gradient={results.reel.posterGradient}
              videoSrc={results.reel.videoUrl}
              caption={`${results.reel.sourceName} · ${results.reel.matches} matches`}
            />
          </GlassCard>

          <div className="grid gap-4 lg:grid-cols-2">
            {results.annotated.map((a, i) => (
              <GlassCard key={a.id} delay={i * 0.05} className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="min-w-0 flex-1 truncate text-sm font-medium">{a.sourceName}</p>
                  <Pill tone="info" mono>
                    {a.matches} hits
                  </Pill>
                </div>
                <VideoSurface gradient={a.posterGradient} videoSrc={a.videoUrl} caption="annotated" />
                <DownloadButton
                  label="Download annotated"
                  pending={pendingExport === "annotated-zip"}
                  onClick={() => onExport("annotated-zip")}
                />
              </GlassCard>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="details">
          {results.matches.length === 0 ? (
            <GlassCard className="space-y-4">
              <EmptyState
                icon={Target}
                title="No matches above threshold"
                description="Lower the detection threshold in the control panel and rescan."
              />
              {results.diagnostics ? (
                <div className="grid gap-2 sm:grid-cols-2">
                  <div className="rounded-xl border border-border bg-muted/10 px-3 py-2 text-sm">
                    <span className="text-muted-foreground uppercase">Best face similarity seen</span>
                    <div className="font-mono text-primary-bright">
                      {results.diagnostics.bestFaceSim.toFixed(3)}
                    </div>
                  </div>
                  <div className="rounded-xl border border-border bg-muted/10 px-3 py-2 text-sm">
                    <span className="text-muted-foreground uppercase">Face gate / match threshold</span>
                    <div className="font-mono">
                      {results.diagnostics.faceThr.toFixed(2)} / {results.diagnostics.threshold.toFixed(2)}
                    </div>
                  </div>
                  {results.diagnostics.bestFaceSim >= results.diagnostics.faceThr ? (
                    <div className="sm:col-span-2 rounded-xl border border-success/30 bg-success/10 px-3 py-2 text-sm text-foreground">
                      The engine <strong>did see a qualifying face</strong> ({results.diagnostics.framesWithFace} of{" "}
                      {results.diagnostics.framesProcessed} processed frames) with peak fused{" "}
                      {results.diagnostics.bestFused.toFixed(3)} — raise the face weight or lower the
                      threshold to capture these.
                    </div>
                  ) : (
                    <div className="sm:col-span-2 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-foreground">
                      The best face similarity ({results.diagnostics.bestFaceSim.toFixed(3)}) never cleared the{" "}
                      {results.diagnostics.faceThr.toFixed(2)} face gate — faces were detected but didn't
                      match the reference profile strongly enough. Try brighter reference photos or looser
                      matching (use pose/colour corroboration).
                    </div>
                  )}
                </div>
              ) : null}
            </GlassCard>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {results.matches.map((match, i) => (
                <GlassCard key={match.id} delay={i * 0.03} interactive className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-sm text-primary-bright">
                      #{String(match.rank).padStart(2, "0")}
                    </span>
                    <Pill tone="neutral" mono className="max-w-[60%] truncate">
                      {match.videoName}
                    </Pill>
                  </div>
                  <VideoSurface
                    gradient={match.screenshotGradient}
                    src={match.screenshotUrl}
                    caption={`${formatClock(match.startSeconds)} → ${formatClock(match.endSeconds)}`}
                    onPlay={() => setPlaying(match)}
                  />
                  <dl className="grid grid-cols-3 gap-2 font-mono text-[11px]">
                    {[
                      ["face", match.faceScore, "text-primary-bright"],
                      ["pose", match.poseScore, "text-violet"],
                      ["fused", match.fusedScore, "text-success"],
                    ].map(([label, value, tone]) => (
                      <div
                        key={String(label)}
                        className="rounded-lg border border-border bg-muted/10 px-2 py-1.5"
                      >
                        <dt className="text-muted-foreground uppercase">{label}</dt>
                        <dd className={String(tone)}>{Number(value).toFixed(3)}</dd>
                      </div>
                    ))}
                    {match.clothScore !== undefined ? (
                      <div className="rounded-lg border border-border bg-muted/10 px-2 py-1.5">
                        <dt className="text-muted-foreground uppercase">cloth</dt>
                        <dd className="text-warning">{match.clothScore.toFixed(3)}</dd>
                      </div>
                    ) : null}
                  </dl>
                  <ActionButton variant="glass" size="sm" onClick={() => setPlaying(match)}>
                    <Play className="size-3.5" /> Play
                  </ActionButton>
                </GlassCard>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="graph">
          <GlassCard className="space-y-4">
            <h3 className="flex items-center gap-2 text-sm font-semibold tracking-wide uppercase">
              <BarChart3 className="size-4 text-primary-bright" /> Confidence over timeline
            </h3>
            <div className="h-[340px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={results.confidenceSeries}>
                  <defs>
                    <linearGradient id="confFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--primary-bright)" stopOpacity={0.55} />
                      <stop offset="100%" stopColor="var(--primary)" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="var(--border)" vertical={false} />
                  <XAxis
                    dataKey="t"
                    tickFormatter={(v: number) => formatClock(v)}
                    stroke="var(--muted-foreground)"
                    fontSize={11}
                  />
                  <YAxis
                    domain={[0, 1]}
                    stroke="var(--muted-foreground)"
                    fontSize={11}
                    tickFormatter={(v: number) => v.toFixed(1)}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "var(--popover)",
                      border: "1px solid var(--border)",
                      borderRadius: 12,
                      fontFamily: "var(--font-mono)",
                      fontSize: 12,
                      color: "var(--foreground)",
                    }}
                    labelFormatter={(v: number) => `t = ${formatClock(v)}`}
                    formatter={(v: number) => [v.toFixed(3), "confidence"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="confidence"
                    stroke="var(--primary-bright)"
                    strokeWidth={2}
                    fill="url(#confFill)"
                    animationDuration={1200}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>
        </TabsContent>

        <TabsContent value="raw">
          <GlassCard className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter by video or score…"
                className="max-w-xs"
                aria-label="Filter matches"
              />
              <ActionButton variant="glass" size="sm" onClick={() => setSortDesc((s) => !s)}>
                <ArrowDownUp className="size-4" />
                Confidence {sortDesc ? "↓" : "↑"}
              </ActionButton>
              <Pill tone="neutral" mono>
                {rows.length} rows
              </Pill>
            </div>

            {rows.length === 0 ? (
              <EmptyState
                icon={Target}
                title="Nothing matches that filter"
                description="Clear the filter to see all recorded sightings."
              />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-left font-mono text-xs">
                  <thead className="text-muted-foreground uppercase">
                    <tr className="border-b border-border">
                      <th className="py-2 pr-3">Rank</th>
                      <th className="py-2 pr-3">Video</th>
                      <th className="py-2 pr-3">Start</th>
                      <th className="py-2 pr-3">End</th>
                      <th className="py-2 pr-3">Face</th>
                      <th className="py-2 pr-3">Pose</th>
                      <th className="py-2 pr-3">Cloth</th>
                      <th className="py-2">Fused</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <motion.tr
                        key={r.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="border-b border-border/60 transition-colors hover:bg-accent/20"
                      >
                        <td className="py-2 pr-3 text-primary-bright">{r.rank}</td>
                        <td className="max-w-[220px] truncate py-2 pr-3">{r.videoName}</td>
                        <td className="py-2 pr-3">{formatClock(r.startSeconds)}</td>
                        <td className="py-2 pr-3">{formatClock(r.endSeconds)}</td>
                        <td className="py-2 pr-3">{r.faceScore.toFixed(3)}</td>
                        <td className="py-2 pr-3">{r.poseScore.toFixed(3)}</td>
                        <td className="py-2 pr-3">{r.clothScore?.toFixed(3) ?? "—"}</td>
                        <td className="py-2 text-success">{r.fusedScore.toFixed(3)}</td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </GlassCard>
        </TabsContent>
      </Tabs>

      <GlassCard className="space-y-4">
        <h3 className="text-sm font-semibold tracking-wide uppercase">Reports & Export</h3>
        <div className="flex flex-wrap gap-3">
          {EXPORTS.map(({ kind, label, icon: Icon }) => (
            <ActionButton
              key={kind}
              variant={kind === "pdf" ? "action" : "glass"}
              size="sm"
              disabled={pendingExport === kind}
              onClick={() => onExport(kind)}
            >
              <Icon className="size-4" />
              {pendingExport === kind ? "Preparing…" : label}
            </ActionButton>
          ))}
        </div>
      </GlassCard>

      <div className="flex justify-center pb-6">
        <ActionButton
          variant="danger"
          onClick={() => {
            state.reset();
            void navigate({ to: "/" });
          }}
        >
          <RotateCcw className="size-4" /> Start New Analysis
        </ActionButton>
      </div>

      <Dialog open={playing !== null} onOpenChange={(open) => !open && setPlaying(null)}>
        <DialogContent className="glass-panel max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-base">
              {playing ? `Playback · ${playing.videoName}` : "Playback"}
            </DialogTitle>
          </DialogHeader>
          {playing ? (
            <div className="space-y-3">
              <VideoSurface
                gradient={playing.screenshotGradient}
                src={playing.screenshotUrl}
                videoSrc={playing.annotatedVideoUrl}
                seekTo={playing.startSeconds}
                caption={`seek ${formatClock(playing.startSeconds)}`}
              />
              <div className="flex flex-wrap gap-2">
                <Pill tone="ok" mono>
                  fused {playing.fusedScore.toFixed(3)}
                </Pill>
                <Pill tone="info" mono>
                  face {playing.faceScore.toFixed(3)}
                </Pill>
                <Pill tone="neutral" mono>
                  pose {playing.poseScore.toFixed(3)}
                </Pill>
                {playing.clothScore !== undefined ? (
                  <Pill tone="neutral" mono>
                    cloth {playing.clothScore.toFixed(3)}
                  </Pill>
                ) : null}
                <Pill tone="neutral" mono>
                  {formatClock(playing.startSeconds)} → {formatClock(playing.endSeconds)}
                </Pill>
              </div>
              <p className="text-xs text-muted-foreground">
                Annotated playback seeks to this sighting and plays the surrounding window.
              </p>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}
