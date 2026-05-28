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
import type {
  DetectorName,
  ReviewUiState,
  SessionResponse,
  SiliconSide,
  TracePayload,
} from "../types";

type ReviewSource = "label_set" | "filter_file" | "event_trace";
type ReviewFamily = "normal" | "strange";
type VisualMode = "raw" | "cdf" | "curvature" | "peak";
type DirectFilterItem = "none" | "max";

interface ReviewState {
  source: ReviewSource;
  detector: DetectorName;
  run: number | null;
  family: ReviewFamily;
  label: string;
  filterFile: string;
  eventId: number | null;
  traceId: number | null;
  filterItem: DirectFilterItem;
  filterValue: number | null;
  siSide: SiliconSide;
  siIndex: number | null;
  gaggLayer: 1 | 2;
  gaggIndex: number | null;
  currentTrace: TracePayload | null;
  visualMode: VisualMode;
  loading: boolean;
  error: string;
  statusMessage: string;
}

const DEFAULT_SI_SIDE: SiliconSide = "upstream_front";
const DEFAULT_GAGG_LAYER: 1 | 2 = 1;

const state = reactive<ReviewState>({
  source: "label_set",
  detector: "ATTPC",
  run: null,
  family: "normal",
  label: "",
  filterFile: "",
  eventId: null,
  traceId: 0,
  filterItem: "none",
  filterValue: null,
  siSide: DEFAULT_SI_SIDE,
  siIndex: 0,
  gaggLayer: DEFAULT_GAGG_LAYER,
  gaggIndex: 0,
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

function normalizeDirectDetector(detector: unknown): DetectorName {
  const token = String(detector || "ATTPC").trim().toUpperCase();
  if (token === "IC") {
    return "IC";
  }
  if (token === "SI" || token === "SILICON") {
    return "SI";
  }
  if (token === "GAGG") {
    return "GAGG";
  }
  return "ATTPC";
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
  if (
    eventRange
    && (state.eventId === null || state.eventId < eventRange.min || state.eventId > eventRange.max)
  ) {
    state.eventId = eventRange.min;
  }
  if (state.detector === "IC") {
    state.traceId = 0;
    return;
  }
  if (state.detector === "SI") {
    if (!state.siSide) {
      state.siSide = DEFAULT_SI_SIDE;
    }
    if (state.siIndex === null || state.siIndex < 0) {
      state.siIndex = 0;
    }
  } else if (state.detector === "GAGG") {
    if (state.gaggLayer !== 1 && state.gaggLayer !== 2) {
      state.gaggLayer = DEFAULT_GAGG_LAYER;
    }
    if (state.gaggIndex === null || state.gaggIndex < 0) {
      state.gaggIndex = 0;
    }
  }
  if (state.traceId === null || state.traceId < 0) {
    state.traceId = 0;
  }
}

function setSource(source: ReviewSource): void {
  state.source = source;
  if (source !== "event_trace") {
    state.detector = "ATTPC";
  }
  state.currentTrace = null;
  clearTransientUi();
  ensureDefaults();
}

function setDetector(detector: DetectorName): void {
  state.detector = normalizeDirectDetector(detector);
  if (state.detector === "IC") {
    state.traceId = 0;
  } else if (state.detector === "SI") {
    state.siSide = DEFAULT_SI_SIDE;
    state.siIndex = 0;
  } else if (state.detector === "GAGG") {
    state.gaggLayer = DEFAULT_GAGG_LAYER;
    state.gaggIndex = 0;
  }
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

function setFilterItem(filterItem: DirectFilterItem): void {
  state.filterItem = filterItem === "max" ? "max" : "none";
  if (state.filterItem === "none") {
    state.filterValue = null;
  }
}

function setFilterValue(filterValue: number | string | null): void {
  if (filterValue === null || filterValue === "") {
    state.filterValue = null;
    return;
  }
  state.filterValue = Number(filterValue);
}

function setEventId(eventId: number | string | null): void {
  if (eventId === null || eventId === "") {
    state.eventId = null;
    return;
  }
  state.eventId = Number(eventId);
}

function setTraceId(traceId: number | string | null): void {
  if (state.detector === "IC") {
    state.traceId = 0;
    return;
  }
  if (traceId === null || traceId === "") {
    state.traceId = null;
    return;
  }
  state.traceId = Number(traceId);
}

function setSiSide(side: SiliconSide): void {
  state.siSide = side;
}

function setSiIndex(index: number | string | null): void {
  if (index === null || index === "") {
    state.siIndex = null;
    return;
  }
  state.siIndex = Number(index);
}

function setGaggLayer(layer: number | string | null): void {
  const numeric = Number(layer);
  state.gaggLayer = numeric === 2 ? 2 : 1;
}

function setGaggIndex(index: number | string | null): void {
  if (index === null || index === "") {
    state.gaggIndex = null;
    return;
  }
  state.gaggIndex = Number(index);
}

function setVisualMode(mode: VisualMode): void {
  if (mode !== "raw" && mode !== "cdf" && mode !== "curvature" && mode !== "peak") {
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
        : state.visualMode === "curvature"
          ? "peak"
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
    state.detector = "ATTPC";
    if (query.run !== undefined) {
      setRun(Number(query.run));
    }
    state.family = query.family === "strange" ? "strange" : "normal";
    state.label = typeof query.label === "string" ? query.label : "";
    return;
  }
  if (source === "event_trace") {
    state.detector = normalizeDirectDetector(query.detector);
    if (query.run !== undefined) {
      setRun(Number(query.run));
    }
    if (query.eventId !== undefined) {
      setEventId(Number(query.eventId));
    }
    if (query.traceId !== undefined) {
      setTraceId(Number(query.traceId));
    }
    if (query.filterItem === "max" || query.filterItem === "none") {
      setFilterItem(query.filterItem);
    }
    if (query.filterValue !== undefined) {
      setFilterValue(Number(query.filterValue));
    }
    ensureDirectSourceDefaults();
    return;
  }
  state.detector = "ATTPC";
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
      detector: state.detector,
      run: state.run ?? undefined,
      eventId: state.eventId ?? undefined,
      traceId: state.detector === "IC" ? 0 : state.traceId ?? undefined,
      filterItem: state.filterItem,
      filterValue:
        state.filterItem === "max" ? state.filterValue ?? undefined : undefined,
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
      if (state.run === null || state.eventId === null) {
        throw new Error("Select a run and event id before loading review.");
      }
      if (state.detector === "ATTPC" && state.traceId === null) {
        throw new Error("Select a run, event id, and trace id before loading review.");
      }
      if (state.detector === "SI" && state.siIndex === null) {
        throw new Error("Select a silicon side and index before loading review.");
      }
      if (state.detector === "GAGG" && state.gaggIndex === null) {
        throw new Error("Select a GAGG layer and index before loading review.");
      }
      payload = await setEventTraceReviewSession({
        run: state.run,
        eventId: state.eventId,
        detector: state.detector,
        traceId: state.detector === "IC" ? 0 : state.traceId,
        siSide: state.detector === "SI" ? state.siSide : null,
        siIndex: state.detector === "SI" ? state.siIndex : null,
        gaggLayer: state.detector === "GAGG" ? state.gaggLayer : null,
        gaggIndex: state.detector === "GAGG" ? state.gaggIndex : null,
        filterItem: state.filterItem,
        filterValue: state.filterItem === "max" ? state.filterValue : null,
      });
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
  state.detector = normalizeDirectDetector(trace.detector);
  if (state.detector === "IC") {
    state.traceId = 0;
    return;
  }
  if (trace.traceSelector?.kind === "si") {
    state.siSide = trace.traceSelector.side;
    state.siIndex = trace.traceSelector.index;
    return;
  }
  if (trace.traceSelector?.kind === "gagg") {
    state.gaggLayer = trace.traceSelector.layer;
    state.gaggIndex = trace.traceSelector.index;
  }
}

function syncSession(payload: SessionResponse): void {
  useShellStore().updateSession(payload.session);
}

function applyUiState(payload: ReviewUiState | null | undefined): void {
  if (!payload) {
    return;
  }
  state.source = payload.source;
  state.detector =
    payload.source === "event_trace"
      ? normalizeDirectDetector(payload.detector)
      : "ATTPC";
  state.run = payload.run;
  state.family = payload.family;
  state.label = payload.label;
  state.filterFile = payload.filterFile;
  state.eventId = payload.eventId;
  state.traceId = payload.traceId;
  state.filterItem = payload.filterItem;
  state.filterValue = payload.filterValue;
  state.visualMode = payload.visualMode;
  ensureDefaults();
}

function serializeUiState(): ReviewUiState {
  return {
    source: state.source,
    detector: state.detector,
    run: state.run,
    family: state.family,
    label: state.label,
    filterFile: state.filterFile,
    eventId: state.eventId,
    traceId: state.detector === "IC" ? 0 : state.traceId,
    filterItem: state.filterItem,
    filterValue: state.filterValue,
    visualMode: state.visualMode,
  };
}

export function useReviewStore() {
  return {
    state,
    clearTransientUi,
    setSource,
    setDetector,
    setRun,
    setFamily,
    setLabel,
    setFilterFile,
    setFilterItem,
    setFilterValue,
    setEventId,
    setTraceId,
    setSiSide,
    setSiIndex,
    setGaggLayer,
    setGaggIndex,
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
