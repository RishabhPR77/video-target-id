import { motion } from "framer-motion";
import { AlertTriangle, Loader2, type LucideIcon } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { cn } from "@/lib/utils";

export function GlassCard({
  className,
  children,
  delay = 0,
  interactive = false,
}: {
  className?: string;
  children: ReactNode;
  delay?: number;
  interactive?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: [0.16, 1, 0.3, 1] }}
      className={cn("glass-panel p-5", interactive && "hover-lift", className)}
    >
      {children}
    </motion.div>
  );
}

type PillTone = "ok" | "warn" | "error" | "info" | "neutral";

const toneClasses: Record<PillTone, string> = {
  ok: "border-success/30 bg-success/10 text-success",
  warn: "border-warning/30 bg-warning/10 text-warning",
  error: "border-destructive/30 bg-destructive/10 text-destructive",
  info: "border-primary-bright/30 bg-primary/10 text-primary-bright",
  neutral: "border-border bg-muted/40 text-muted-foreground",
};

export function Pill({
  tone = "neutral",
  children,
  className,
  mono = false,
}: {
  tone?: PillTone;
  children: ReactNode;
  className?: string;
  mono?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium tracking-wide uppercase",
        mono && "font-mono tracking-normal normal-case",
        toneClasses[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function SectionTitle({
  eyebrow,
  title,
  description,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="space-y-1.5">
      {eyebrow ? (
        <p className="font-mono text-[11px] tracking-[0.22em] text-primary-bright uppercase">
          {eyebrow}
        </p>
      ) : null}
      <h2 className="text-2xl font-semibold text-foreground sm:text-3xl">{title}</h2>
      {description ? <p className="text-sm text-muted-foreground">{description}</p> : null}
    </div>
  );
}

export function useCountUp(target: number, duration = 900) {
  const [value, setValue] = useState(0);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setValue(target * eased);
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [target, duration]);

  return value;
}

export function MetricCard({
  label,
  value,
  suffix,
  icon: Icon,
  format,
  delay = 0,
}: {
  label: string;
  value: number;
  suffix?: string;
  icon: LucideIcon;
  format?: (v: number) => string;
  delay?: number;
}) {
  const animated = useCountUp(value);
  return (
    <GlassCard interactive delay={delay} className="p-4">
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{label}</p>
        <span className="rounded-lg border border-border bg-primary/10 p-1.5 text-primary-bright">
          <Icon className="size-4" />
        </span>
      </div>
      <p className="mt-3 font-mono text-2xl font-semibold text-foreground">
        {format ? format(animated) : Math.round(animated)}
        {suffix ? <span className="ml-1 text-sm text-muted-foreground">{suffix}</span> : null}
      </p>
    </GlassCard>
  );
}

export function GlowProgress({ percent, label }: { percent: number; label?: string }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-primary-bright">{percent.toFixed(0)}%</span>
      </div>
      <div className="h-3 w-full overflow-hidden rounded-full border border-border bg-muted/40">
        <motion.div
          className="gradient-action glow-ring h-full rounded-full"
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-border bg-muted/10 px-6 py-12 text-center">
      <span className="rounded-2xl border border-border bg-primary/10 p-3 text-primary-bright">
        <Icon className="size-6" />
      </span>
      <h3 className="text-base font-semibold text-foreground">{title}</h3>
      <p className="max-w-sm text-sm text-muted-foreground">{description}</p>
      {action}
    </div>
  );
}

export function LoadingBlock({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin text-primary-bright" />
      {label}…
    </div>
  );
}

export function ErrorBlock({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-destructive/30 bg-destructive/10 px-6 py-8 text-center">
      <AlertTriangle className="size-5 text-destructive" />
      <p className="text-sm text-foreground">{message}</p>
      {onRetry ? (
        <button
          onClick={onRetry}
          className="rounded-lg border border-border px-3 py-1.5 text-xs text-foreground transition-colors hover:bg-accent"
        >
          Try again
        </button>
      ) : null}
    </div>
  );
}

export function SkeletonRow({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-lg bg-muted/50", className)} />;
}
