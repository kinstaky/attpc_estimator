import { computed, reactive } from "vue";

import {
  getCurrentPointcloudEvent,
  getMappingPads,
  getPointcloudTraces,
  nextPointcloudEvent,
  previousPointcloudEvent,
  setPointcloudBrowseSession,
} from "../api";
import { useShellStore } from "./shell";
import type {
  MappingPad,
  PointcloudEventPayload,
  PointcloudTracePayload,
  PointcloudUiState,
  SessionResponse,
} from "../types";

type LayoutMode = "1x1" | "2x2";
type PointcloudSource = "event_id" | "label_set";

interface PointcloudState {
  loading: boolean;
  error: string;
  statusMessage: string;
  eventPayload: PointcloudEventPayload | null;
  tracePayload: PointcloudTracePayload;
  mappingPads: MappingPad[];
  source: PointcloudSource;
  selectedRun: number | null;
  selectedEventId: number | null;
  selectedLabel: string;
  selectedTraceIds: number[];
  layoutMode: LayoutMode;
  panelTypes: string[];
  xyRange: Record<string, unknown> | null;
  projectedRange: Record<string, unknown> | null;
  camera: Record<string, unknown> | null;
}

const DEFAULT_PANEL_TYPES = ["hits-3d-amplitude", "pads-z", "hits-2d-amplitude", "traces"];

const state = reactive<PointcloudState>({
  loading: false,
  error: "",
  statusMessage: "",
  eventPayload: null,
  tracePayload: { run: 0, eventId: 0, baselineWindowScale: 0, traces: [] },
  mappingPads: [],
  source: "event_id",
  selectedRun: null,
  selectedEventId: null,
  selectedLabel: "",
  selectedTraceIds: [],
  layoutMode: "1x1",
  panelTypes: [...DEFAULT_PANEL_TYPES],
  xyRange: null,
  projectedRange: null,
  camera: null,
});

const sourceOptions = [
  { title: "Event id", value: "event_id" },
  { title: "Label set", value: "label_set" },
];

const layoutOptions = [
  { title: "1x1", value: "1x1" },
  { title: "2x2", value: "2x2" },
];

const plotOptions = [
  { title: "3D xyz · Q", value: "hits-3d-amplitude" },
  { title: "2D xy · z", value: "hits-2d-z" },
  { title: "2D xy · Q", value: "hits-2d-amplitude" },
  { title: "2D x'y' · Q", value: "hits-2d-pca-amplitude" },
  { title: "pads · z", value: "pads-z" },
  { title: "pads · Q", value: "pads-amplitude" },
  { title: "traces", value: "traces" },
];

const pointcloudLabelOptions = [
  { title: "All labeled events", value: "" },
  { title: "0 lines", value: "0" },
  { title: "1 line", value: "1" },
  { title: "2 lines", value: "2" },
  { title: "3 lines", value: "3" },
  { title: "4 lines", value: "4" },
  { title: "5 lines", value: "5" },
  { title: "6+ lines", value: "6+" },
];

const panelCount = computed(() => (state.layoutMode === "1x1" ? 1 : 4));

const pointcloudEventRanges = computed(() => useShellStore().state.bootstrap?.pointcloudEventRanges || {});

function defaultEventIdForRun(run: number | null): number | null {
  if (run === null) {
    return null;
  }
  const range = pointcloudEventRanges.value[String(run)];
  return range ? Number(range.min) : null;
}

function syncSession(payload: SessionResponse): void {
  useShellStore().updateSession(payload.session);
}

async function init(): Promise<void> {
  if (!state.mappingPads.length) {
    state.mappingPads = await getMappingPads();
  }
  if (state.selectedRun === null) {
    state.selectedRun = useShellStore().state.bootstrap?.pointcloudRuns?.[0] ?? null;
  }
  if (state.selectedEventId === null) {
    state.selectedEventId = defaultEventIdForRun(state.selectedRun);
  }
}

function clearTransientUi(): void {
  state.error = "";
  state.statusMessage = "";
}

function resetViewState(): void {
  state.selectedTraceIds = [];
  state.tracePayload = {
    run: state.eventPayload?.run || 0,
    eventId: state.eventPayload?.eventId || 0,
    baselineWindowScale: state.tracePayload.baselineWindowScale || 0,
    traces: [],
  };
  state.xyRange = null;
  state.projectedRange = null;
  state.camera = null;
}

function setSource(source: PointcloudSource): void {
  state.source = source;
  state.eventPayload = null;
  resetViewState();
  clearTransientUi();
}

function setRun(run: number | string | null): void {
  state.selectedRun = run === null || run === "" ? null : Number(run);
  if (state.source === "event_id") {
    state.selectedEventId = defaultEventIdForRun(state.selectedRun);
  }
}

function setEventId(value: number | string | null): void {
  state.selectedEventId = value === null || value === "" ? null : Number(value);
}

function setSelectedLabel(value: string): void {
  state.selectedLabel = value || "";
}

function setLayoutMode(value: LayoutMode): void {
  state.layoutMode = value;
}

function toggleLayoutMode(): void {
  state.layoutMode = state.layoutMode === "1x1" ? "2x2" : "1x1";
}

function setPanelType(index: number, value: string): void {
  state.panelTypes[index] = value;
}

async function loadEvent(): Promise<void> {
  if (state.selectedRun === null) {
    state.error = "Select a run before loading pointcloud browse.";
    return;
  }
  if (state.source === "event_id" && state.selectedEventId === null) {
    state.error = "Select an event id before loading pointcloud browse.";
    return;
  }
  state.loading = true;
  clearTransientUi();
  resetViewState();
  try {
    const payload = await setPointcloudBrowseSession(
      state.selectedRun,
      state.source,
      state.source === "event_id" ? state.selectedEventId : null,
      state.source === "label_set" ? state.selectedLabel || null : null,
    );
    syncSession(payload);
    state.eventPayload = payload.event ?? null;
    if (!state.eventPayload) {
      state.statusMessage = "The selected pointcloud browse set does not contain any events.";
      return;
    }
    state.selectedRun = state.eventPayload.run;
    state.selectedEventId = state.eventPayload.eventId;
  } catch (error) {
    state.eventPayload = null;
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

async function restoreCurrentSession(): Promise<void> {
  state.loading = true;
  clearTransientUi();
  resetViewState();
  try {
    state.eventPayload = await getCurrentPointcloudEvent();
    state.selectedRun = state.eventPayload.run;
    state.selectedEventId = state.eventPayload.eventId;
  } catch (error) {
    state.eventPayload = null;
    state.error = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    state.loading = false;
  }
}

async function loadSelectedTraces(): Promise<void> {
  if (!state.eventPayload || !state.selectedTraceIds.length) {
    state.tracePayload = {
      run: state.eventPayload?.run || 0,
      eventId: state.eventPayload?.eventId || 0,
      baselineWindowScale: state.tracePayload.baselineWindowScale || 0,
      traces: [],
    };
    return;
  }
  try {
    state.tracePayload = await getPointcloudTraces(
      state.eventPayload.run,
      state.eventPayload.eventId,
      state.selectedTraceIds,
    );
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  }
}

function toggleTraceIds(traceIds: number[]): void {
  const next = [...state.selectedTraceIds];
  for (const traceId of traceIds || []) {
    const numeric = Number(traceId);
    const existingIndex = next.indexOf(numeric);
    if (existingIndex >= 0) {
      next.splice(existingIndex, 1);
    } else {
      next.push(numeric);
    }
  }
  state.selectedTraceIds = next.slice(-8);
}

function clearSelection(): void {
  state.selectedTraceIds = [];
}

function sameRange(nextRange: Record<string, unknown> | null, currentRange: Record<string, unknown> | null): boolean {
  if (nextRange === currentRange) {
    return true;
  }
  if (!nextRange || !currentRange) {
    return false;
  }
  const nextX = nextRange.x as number[] | undefined;
  const currentX = currentRange.x as number[] | undefined;
  const nextY = nextRange.y as number[] | undefined;
  const currentY = currentRange.y as number[] | undefined;
  return nextX?.[0] === currentX?.[0]
    && nextX?.[1] === currentX?.[1]
    && nextY?.[0] === currentY?.[0]
    && nextY?.[1] === currentY?.[1];
}

function updateXYRange(range: Record<string, unknown> | null): void {
  if ((range === null && state.xyRange === null) || sameRange(range, state.xyRange)) {
    return;
  }
  state.xyRange = range;
}

function updateProjectedRange(range: Record<string, unknown> | null): void {
  if ((range === null && state.projectedRange === null) || sameRange(range, state.projectedRange)) {
    return;
  }
  state.projectedRange = range;
}

function updateCamera(nextCamera: Record<string, unknown> | null): void {
  state.camera = nextCamera;
}

async function nextEvent(): Promise<void> {
  if (!state.eventPayload) {
    return;
  }
  state.loading = true;
  clearTransientUi();
  resetViewState();
  try {
    state.eventPayload = await nextPointcloudEvent();
    state.selectedRun = state.eventPayload.run;
    state.selectedEventId = state.eventPayload.eventId;
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

async function previousEvent(): Promise<void> {
  if (!state.eventPayload) {
    return;
  }
  state.loading = true;
  clearTransientUi();
  resetViewState();
  try {
    state.eventPayload = await previousPointcloudEvent();
    state.selectedRun = state.eventPayload.run;
    state.selectedEventId = state.eventPayload.eventId;
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

function applyUiState(payload: PointcloudUiState | null | undefined): void {
  if (!payload) {
    return;
  }
  state.source = payload.source;
  state.selectedRun = payload.selectedRun;
  state.selectedEventId = payload.selectedEventId;
  state.selectedLabel = payload.selectedLabel;
  state.layoutMode = payload.layoutMode;
  state.panelTypes = payload.panelTypes.length ? [...payload.panelTypes] : [...DEFAULT_PANEL_TYPES];
  state.selectedTraceIds = [...payload.selectedTraceIds];
}

function serializeUiState(): PointcloudUiState {
  return {
    source: state.source,
    selectedRun: state.selectedRun,
    selectedEventId: state.selectedEventId,
    selectedLabel: state.selectedLabel,
    layoutMode: state.layoutMode,
    panelTypes: [...state.panelTypes],
    selectedTraceIds: [...state.selectedTraceIds],
  };
}

export function usePointcloudStore() {
  return {
    state,
    sourceOptions,
    layoutOptions,
    plotOptions,
    pointcloudLabelOptions,
    panelCount,
    pointcloudEventRanges,
    init,
    clearTransientUi,
    setSource,
    setRun,
    setEventId,
    setSelectedLabel,
    setLayoutMode,
    toggleLayoutMode,
    setPanelType,
    loadEvent,
    restoreCurrentSession,
    loadSelectedTraces,
    toggleTraceIds,
    clearSelection,
    updateXYRange,
    updateProjectedRange,
    updateCamera,
    nextEvent,
    previousEvent,
    applyUiState,
    serializeUiState,
    defaultEventIdForRun,
  };
}
