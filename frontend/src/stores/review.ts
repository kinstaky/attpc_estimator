import { reactive } from "vue";

import {
  nextTrace,
  previousTrace,
  setFilterReviewSession,
  setLabelReviewSession,
} from "../api";
import { useShellStore } from "./shell";
import type { TracePayload } from "../types";

type ReviewSource = "label_set" | "filter_file";
type ReviewFamily = "normal" | "strange";
type VisualMode = "raw" | "cdf" | "curvature";

interface ReviewState {
  source: ReviewSource;
  run: number | null;
  family: ReviewFamily;
  label: string;
  filterFile: string;
  currentTrace: TracePayload | null;
  visualMode: VisualMode;
  loading: boolean;
  error: string;
  statusMessage: string;
}

const state = reactive<ReviewState>({
  source: "label_set",
  run: null,
  family: "normal",
  label: "",
  filterFile: "",
  currentTrace: null,
  visualMode: "cdf",
  loading: false,
  error: "",
  statusMessage: "",
});

function clearTransientUi(): void {
  state.error = "";
  state.statusMessage = "";
}

function ensureDefaults(): void {
  const shell = useShellStore();
  if (state.run === null) {
    state.run = shell.state.selectedRun;
  }
  if (!state.filterFile) {
    state.filterFile = shell.state.bootstrap?.filterFiles?.[0]?.name || "";
  }
}

function setSource(source: ReviewSource): void {
  state.source = source;
  state.currentTrace = null;
  clearTransientUi();
}

function setRun(run: number | string | null): void {
  const shell = useShellStore();
  state.run = run === null || run === "" ? null : Number(run);
  shell.setSelectedRun(state.run);
}

function setFamily(family: ReviewFamily): void {
  state.family = family;
  state.label = "";
}

function setLabel(label: string): void {
  state.label = label || "";
}

function setFilterFile(filterFile: string): void {
  state.filterFile = filterFile || "";
}

function setVisualMode(mode: VisualMode): void {
  if (mode !== "raw" && mode !== "cdf" && mode !== "curvature") {
    return;
  }
  state.visualMode = mode;
}

function toggleVisualMode(): void {
  state.visualMode =
    state.visualMode === "raw"
      ? "cdf"
      : state.visualMode === "cdf"
        ? "curvature"
        : "raw";
}

function applyQuery(query: Record<string, unknown>): void {
  ensureDefaults();
  const source = query.source === "filter_file" ? "filter_file" : "label_set";
  state.source = source;
  if (source === "label_set") {
    if (query.run !== undefined) {
      setRun(Number(query.run));
    }
    state.family = query.family === "strange" ? "strange" : "normal";
    state.label = typeof query.label === "string" ? query.label : "";
    return;
  }
  state.filterFile =
    typeof query.filterFile === "string" ? query.filterFile : state.filterFile;
}

function buildQuery(): Record<string, string | number | undefined> {
  if (state.source === "label_set") {
    return {
      source: "label_set",
      run: state.run ?? undefined,
      family: state.family,
      label: state.label || undefined,
    };
  }
  return {
    source: "filter_file",
    filterFile: state.filterFile || undefined,
  };
}

async function loadReviewSet(): Promise<void> {
  ensureDefaults();
  state.loading = true;
  clearTransientUi();
  try {
    let payload;
    if (state.source === "label_set") {
      if (state.run === null) {
        throw new Error("Select a run before loading labeled review.");
      }
      payload = await setLabelReviewSession(
        state.run,
        state.family,
        state.label || null,
      );
    } else {
      if (!state.filterFile) {
        throw new Error("Select a filter file before loading review.");
      }
      payload = await setFilterReviewSession(state.filterFile);
    }
    state.currentTrace = payload.trace ?? null;
    if (!payload.trace) {
      state.statusMessage = "The selected review set does not contain any traces.";
    }
  } catch (error) {
    state.currentTrace = null;
    state.error = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    state.loading = false;
  }
}

async function nextReviewTrace(): Promise<void> {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentTrace = await nextTrace();
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

async function previousReviewTrace(): Promise<void> {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentTrace = await previousTrace();
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

export function useReviewStore() {
  return {
    state,
    clearTransientUi,
    setSource,
    setRun,
    setFamily,
    setLabel,
    setFilterFile,
    setVisualMode,
    toggleVisualMode,
    applyQuery,
    buildQuery,
    loadReviewSet,
    nextReviewTrace,
    previousReviewTrace,
  };
}
