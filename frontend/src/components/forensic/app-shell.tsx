import { Radar, ShieldCheck, Wifi } from "lucide-react";
import type { ReactNode } from "react";

import { ControlPanel } from "./control-panel";
import { Pill } from "./primitives";
import { DEFAULTS } from "@/lib/api";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-border bg-background/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <span className="glass-panel flex size-10 items-center justify-center rounded-xl p-0 text-primary-bright">
              <Radar className="size-5" />
            </span>
            <div>
              <h1 className="text-lg leading-tight font-semibold">
                <span className="text-gradient-cyan">Video Target ID</span>
              </h1>
              <p className="text-[11px] tracking-wide text-muted-foreground uppercase">
                AI-Powered Forensic Identification System
              </p>
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Pill tone="info" mono>
              {DEFAULTS.version}
            </Pill>
            <Pill tone="ok">
              <Wifi className="size-3" /> Online
            </Pill>
            <Pill tone="neutral" className="hidden sm:inline-flex">
              <ShieldCheck className="size-3" /> Chain of custody
            </Pill>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1500px] flex-col gap-6 px-4 py-6 sm:px-6 lg:flex-row">
        <ControlPanel />
        <main className="min-w-0 flex-1 space-y-6">{children}</main>
      </div>
    </div>
  );
}
