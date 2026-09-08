import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { DEFAULTS } from "./api";
import type { ReferenceStatus, ScanResults, SourceVideo, StepId } from "./types";

export type ReferencePhoto = { id: string; name: string; url: string };

export type CaseState = {
  step: StepId;
  threshold: number;
  faceWeight: number;
  photos: ReferencePhoto[];
  reference: ReferenceStatus;
  authorized: boolean;
  videos: SourceVideo[];
  results: ScanResults | null;
};

const emptyReference: ReferenceStatus = {
  faceReady: false,
  poseReady: false,
  clothReady: false,
  embeddings: 0,
  builtAt: null,
};

const initialState: CaseState = {
  step: 1,
  threshold: DEFAULTS.threshold,
  faceWeight: DEFAULTS.faceWeight,
  photos: [],
  reference: emptyReference,
  authorized: false,
  videos: [],
  results: null,
};

const STORAGE_KEY = "video-target-id.case";

/**
 * Mirror of the live case state kept outside React so route guards
 * (`beforeLoad`) can read progress before a component mounts. Session
 * storage keeps the wizard intact across reloads.
 */
export const caseSnapshot: { current: CaseState } = { current: initialState };

function persist(state: CaseState) {
  caseSnapshot.current = state;
  if (typeof window === "undefined") return;
  try {
    // Blob URLs from photo previews cannot survive a reload.
    const { photos, ...rest } = state;
    window.sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ ...rest, photoCount: photos.length }),
    );
  } catch {
    /* storage unavailable — in-memory state still works */
  }
}

export function hydrateSnapshot(): CaseState {
  if (typeof window === "undefined") return caseSnapshot.current;
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return caseSnapshot.current;
    const parsed = JSON.parse(raw) as Partial<CaseState> & { photoCount?: number };
    const next: CaseState = {
      ...initialState,
      ...parsed,
      photos: caseSnapshot.current.photos,
    };
    caseSnapshot.current = next;
    return next;
  } catch {
    return caseSnapshot.current;
  }
}

type CaseContextValue = CaseState & {
  set: (patch: Partial<CaseState>) => void;
  reset: () => void;
  canEnterStep: (step: StepId) => boolean;
};

const CaseContext = createContext<CaseContextValue | null>(null);

export function canEnterStepFor(state: CaseState, step: StepId): boolean {
  if (step === 1) return true;
  if (step === 2) return state.reference.faceReady && state.authorized;
  if (step === 3) return canEnterStepFor(state, 2) && state.videos.length > 0;
  return canEnterStepFor(state, 3) && state.results !== null;
}

export function CaseProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<CaseState>(initialState);

  useEffect(() => {
    const restored = hydrateSnapshot();
    if (restored !== initialState) setState(restored);
  }, []);

  const value = useMemo<CaseContextValue>(
    () => ({
      ...state,
      set: (patch) =>
        setState((prev) => {
          const next = { ...prev, ...patch };
          persist(next);
          return next;
        }),
      reset: () => {
        caseSnapshot.current = initialState;
        if (typeof window !== "undefined") window.sessionStorage.removeItem(STORAGE_KEY);
        setState(initialState);
      },
      canEnterStep: (step) => canEnterStepFor(state, step),
    }),
    [state],
  );

  return <CaseContext.Provider value={value}>{children}</CaseContext.Provider>;
}

export function useCase(): CaseContextValue {
  const ctx = useContext(CaseContext);
  if (!ctx) throw new Error("useCase must be used inside <CaseProvider>");
  return ctx;
}
