import { reactive } from "vue";

import {
  getCurrentTrace,
  nextEvent,
  nextTrace,
  previousEvent,
  previousTrace,
  setEventTraceReviewSession,
  setFilterReviewSession,
  setLabelReviewSession,
} from "../api";
import { useShellStore } from "./shell";
import type { ReviewUiState, SessionResponse, TracePayload } from "../types";

type ReviewSource = "label_set" | "filter_file" | "event_trace";
type ReviewFamily = "normal" | "strange";
type VisualMode = "raw" | "cdf" | "curvature";

interface ReviewState {
  source: ReviewSource;
  run: number | null;
  family: ReviewFamily;
  label: string;
  filterFile: string;
  eventId: number | null;
  traceId: number | null;
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
  eventId: null,
  traceId: 0,
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
  ensureDirectSourceDefaults();
}

function ensureDirectSourceDefaults(): void {
  const shell = useShellStore();
  if (state.run === null) {
    return;
  }
  const eventRange = shell.state.bootstrap?.eventRanges?.[String(state.run)];
  if (!eventRange) {
    return;
  }
  if (
    state.eventId === null
    || state.eventId < eventRange.min
    || state.eventId > eventRange.max
  ) {
    state.eventId = eventRange.min;
  }
  if (state.traceId === null || state.traceId < 0) {
    state.traceId = 0;
  }
}

function setSource(source: ReviewSource): void {
  state.source = source;
  state.currentTrace = null;
  clearTransientUi();
  ensureDefaults();
}

function setRun(run: number | string | null): void {
  const shell = useShellStore();
  state.run = run === null || run === "" ? null : Number(run);
  shell.setSelectedRun(state.run);
  ensureDirectSourceDefaults();
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

function setEventId(eventId: number | string | null): void {
  if (eventId === null || eventId === "") {
    state.eventId = null;
    return;
  }
  state.eventId = Number(eventId);
}

function setTraceId(traceId: number | string | null): void {
  if (traceId === null || traceId === "") {
    state.traceId = null;
    return;
  }
  state.traceId = Number(traceId);
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
  const source =
    query.source === "filter_file"
      ? "filter_file"
      : query.source === "event_trace"
        ? "event_trace"
        : "label_set";
  state.source = source;
  if (source === "label_set") {
    if (query.run !== undefined) {
      setRun(Number(query.run));
    }
    state.family = query.family === "strange" ? "strange" : "normal";
    state.label = typeof query.label === "string" ? query.label : "";
    return;
  }
  if (source === "event_trace") {
    if (query.run !== undefined) {
      setRun(Number(query.run));
    }
    if (query.eventId !== undefined) {
      setEventId(Number(query.eventId));
    }
    if (query.traceId !== undefined) {
      setTraceId(Number(query.traceId));
    }
    ensureDirectSourceDefaults();
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
  if (state.source === "event_trace") {
    return {
      source: "event_trace",
      run: state.run ?? undefined,
      eventId: state.eventId ?? undefined,
      traceId: state.traceId ?? undefined,
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
    } else if (state.source === "event_trace") {
      if (state.run === null || state.eventId === null || state.traceId === null) {
        throw new Error("Select a run, event id, and trace id before loading review.");
      }
      payload = await setEventTraceReviewSession(
        state.run,
        state.eventId,
        state.traceId,
      );
    } else {
      if (!state.filterFile) {
        throw new Error("Select a filter file before loading review.");
      }
      payload = await setFilterReviewSession(state.filterFile);
    }
    syncSession(payload);
    state.currentTrace = payload.trace ?? null;
    syncDirectSelectionFromTrace(state.currentTrace);
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

async function restoreCurrentSession(): Promise<void> {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentTrace = await getCurrentTrace();
    syncDirectSelectionFromTrace(state.currentTrace);
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
    syncDirectSelectionFromTrace(state.currentTrace);
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
    syncDirectSelectionFromTrace(state.currentTrace);
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

async function nextReviewEvent(): Promise<void> {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentTrace = await nextEvent();
    syncDirectSelectionFromTrace(state.currentTrace);
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

async function previousReviewEvent(): Promise<void> {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentTrace = await previousEvent();
    syncDirectSelectionFromTrace(state.currentTrace);
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

function syncDirectSelectionFromTrace(trace: TracePayload | null): void {
  if (!trace || state.source !== "event_trace") {
    return;
  }
  state.run = trace.run ?? state.run;
  state.eventId = trace.eventId;
  state.traceId = trace.traceId;
}

function syncSession(payload: SessionResponse): void {
  useShellStore().updateSession(payload.session);
}

function applyUiState(payload: ReviewUiState | null | undefined): void {
  if (!payload) {
    return;
  }
  state.source = payload.source;
  state.run = payload.run;
  state.family = payload.family;
  state.label = payload.label;
  state.filterFile = payload.filterFile;
  state.eventId = payload.eventId;
  state.traceId = payload.traceId;
  state.visualMode = payload.visualMode;
  ensureDefaults();
}

function serializeUiState(): ReviewUiState {
  return {
    source: state.source,
    run: state.run,
    family: state.family,
    label: state.label,
    filterFile: state.filterFile,
    eventId: state.eventId,
    traceId: state.traceId,
    visualMode: state.visualMode,
  };
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
    setEventId,
    setTraceId,
    setVisualMode,
    toggleVisualMode,
    applyUiState,
    serializeUiState,
    applyQuery,
    buildQuery,
    loadReviewSet,
    restoreCurrentSession,
    nextReviewTrace,
    previousReviewTrace,
    nextReviewEvent,
    previousReviewEvent,
  };
}
