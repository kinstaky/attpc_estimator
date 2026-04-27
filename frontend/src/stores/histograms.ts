import { computed, reactive } from "vue";

import { createHistogramJob, getHistogram, histogramJobSocketUrl } from "../api";
import { useShellStore } from "./shell";
import type {
  HistogramsUiState,
  HistogramJobMessage,
  HistogramJobProgress,
  HistogramMetric,
  HistogramMode,
  HistogramPhase,
  HistogramPayload,
  HistogramSeries,
  HistogramVariant,
} from "../types";

const DEFAULT_CDF_PROJECTION_BIN = 60;
const LABEL_ORDER_SUFFIX = ":labeled";
type ScaleMode = "linear" | "log";
type CdfRenderMode = "2d" | "projection";

interface HistogramState {
  selectedPhase: HistogramPhase;
  selectedRun: number | null;
  selectedMetric: HistogramMetric;
  selectedMode: HistogramMode;
  selectedBitflipVariant: Extract<HistogramVariant, "baseline" | "value" | "length" | "count">;
  selectedSaturationVariant: Extract<HistogramVariant, "drop" | "length">;
  selectedHistogramFilter: string;
  selectedHistogramVeto: boolean;
  filteredPlotDirty: boolean;
  histogram: HistogramPayload | null;
  cdfScaleMode: ScaleMode;
  amplitudeScaleMode: ScaleMode;
  cdfRenderMode: CdfRenderMode;
  cdfProjectionBin: number;
  loading: boolean;
  progress: HistogramJobProgress | null;
  error: string;
}

const state = reactive<HistogramState>({
  selectedPhase: "phase1",
  selectedRun: null,
  selectedMetric: "cdf",
  selectedMode: "all",
  selectedBitflipVariant: "baseline",
  selectedSaturationVariant: "drop",
  selectedHistogramFilter: "",
  selectedHistogramVeto: false,
  filteredPlotDirty: false,
  histogram: null,
  cdfScaleMode: "linear",
  amplitudeScaleMode: "linear",
  cdfRenderMode: "2d",
  cdfProjectionBin: DEFAULT_CDF_PROJECTION_BIN,
  loading: false,
  progress: null,
  error: "",
});

const labeledSeriesOrder = reactive<Record<string, string[]>>({});
let activeSocket: WebSocket | null = null;
let loadSequence = 0;
let shouldRestoreHistogram = false;

const PHASE_METRICS: Record<HistogramPhase, HistogramMetric[]> = {
  phase1: ["amplitude", "baseline", "bitflip", "cdf", "saturation"],
  phase2: ["line_distance", "line_property", "coplanar"],
};

const MODE_LOCKED_METRICS = new Set<HistogramMetric>(["line_distance", "line_property", "coplanar"]);

function clearTransientUi(): void {
  state.error = "";
}

function closeActiveSocket(): void {
  if (activeSocket === null) {
    return;
  }
  activeSocket.close();
  activeSocket = null;
}

function currentSeriesOrderKey(): string | null {
  if (state.selectedMode !== "labeled") {
    return null;
  }
  return `${state.selectedMetric}${LABEL_ORDER_SUFFIX}`;
}

function syncCurrentSeriesOrder(): void {
  const orderKey = currentSeriesOrderKey();
  const series = state.histogram?.series || [];
  if (!orderKey || !series.length) {
    return;
  }

  const presentKeys = series.map((item) => item.labelKey);
  const existingOrder = labeledSeriesOrder[orderKey] || [];
  const nextOrder = [
    ...existingOrder.filter((key) => presentKeys.includes(key)),
    ...presentKeys.filter((key) => !existingOrder.includes(key)),
  ];

  labeledSeriesOrder[orderKey] = nextOrder;
}

function ensureInitialized(): void {
  const shell = useShellStore();
  if (state.selectedRun === null) {
    state.selectedRun = shell.state.selectedRun;
  }
  if (!state.selectedHistogramFilter) {
    state.selectedHistogramFilter =
      shell.state.bootstrap?.filterFiles?.[0]?.name || "";
  }
}

function currentVariant(): HistogramVariant | "" {
  if (state.selectedMetric === "bitflip") {
    return state.selectedBitflipVariant;
  }
  if (state.selectedMetric === "saturation") {
    return state.selectedSaturationVariant;
  }
  return "";
}

function metricPhase(metric: HistogramMetric): HistogramPhase {
  return metric === "line_distance" || metric === "line_property" || metric === "coplanar" ? "phase2" : "phase1";
}

function getAvailability() {
  const shell = useShellStore();
  if (state.selectedRun === null || !shell.state.bootstrap) {
    return null;
  }
  return shell.state.bootstrap.histogramAvailability?.[String(state.selectedRun)] || null;
}

function ensureModeAvailability(): void {
  const availability = getAvailability();
  if (!availability) {
    return;
  }
  if (MODE_LOCKED_METRICS.has(state.selectedMetric)) {
    state.selectedMode = "all";
    return;
  }
  const metricAvailability = availability?.[state.selectedMetric] || {};
  if (!metricAvailability?.[state.selectedMode]) {
    state.selectedMode =
      (["all", "labeled", "filtered"] as HistogramMode[]).find(
        (mode) => metricAvailability?.[mode],
      ) || "all";
  }
}

async function loadHistogram(forceFiltered = false): Promise<void> {
  ensureInitialized();
  const loadId = ++loadSequence;
  closeActiveSocket();
  if (state.selectedRun === null) {
    state.histogram = null;
    state.progress = null;
    state.filteredPlotDirty = state.selectedMode === "filtered";
    return;
  }
  clearTransientUi();
  state.selectedPhase = metricPhase(state.selectedMetric);
  ensureModeAvailability();
  if (state.selectedMode === "filtered" && !forceFiltered) {
    state.loading = false;
    state.progress = null;
    state.filteredPlotDirty = true;
    if (state.histogram?.mode !== "filtered") {
      state.histogram = null;
    }
    return;
  }
  state.loading = true;
  state.progress = null;
  try {
    if (state.selectedMode === "filtered") {
      state.histogram = await loadFilteredHistogram(loadId);
      state.filteredPlotDirty = false;
    } else {
      state.histogram = await getHistogram(
        state.selectedMetric,
        state.selectedMode,
        state.selectedRun,
        currentVariant(),
        "",
        false,
      );
      state.progress = null;
      state.filteredPlotDirty = false;
    }
    syncCurrentSeriesOrder();
  } catch (error) {
    if (loadId !== loadSequence) {
      return;
    }
    state.histogram = null;
    state.progress = null;
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    if (loadId === loadSequence) {
      state.loading = false;
      if (state.selectedMode !== "filtered") {
        state.progress = null;
      }
    }
  }
}

async function loadFilteredHistogram(loadId: number): Promise<HistogramPayload> {
  if (state.selectedRun === null) {
    throw new Error("run is required");
  }
  if (!state.selectedHistogramFilter) {
    throw new Error("filter file is required");
  }
  const { jobId } = await createHistogramJob(
    state.selectedMetric,
    "filtered",
    state.selectedRun,
    currentVariant(),
    state.selectedHistogramFilter,
    state.selectedHistogramVeto,
  );
  if (loadId !== loadSequence) {
    throw new Error("stale histogram request");
  }

  return await new Promise<HistogramPayload>((resolve, reject) => {
    const socket = new WebSocket(histogramJobSocketUrl(jobId));
    let settled = false;
    activeSocket = socket;

    const finish = (handler: () => void): void => {
      if (settled) {
        return;
      }
      settled = true;
      if (activeSocket === socket) {
        activeSocket = null;
      }
      handler();
    };

    socket.onmessage = (event) => {
      if (loadId !== loadSequence) {
        finish(() => socket.close());
        return;
      }
      const message = JSON.parse(event.data) as HistogramJobMessage;
      if (message.type === "progress") {
        state.progress = {
          current: message.current,
          total: message.total,
          percent: message.percent,
          unit: message.unit,
          message: message.message,
        };
        return;
      }
      if (message.type === "complete") {
        finish(() => {
          socket.close();
          resolve(message.payload);
        });
        return;
      }
      finish(() => {
        socket.close();
        reject(new Error(message.detail));
      });
    };

    socket.onerror = () => {
      finish(() => reject(new Error("histogram progress connection failed")));
    };

    socket.onclose = () => {
      if (settled) {
        return;
      }
      if (loadId !== loadSequence) {
        settled = true;
        return;
      }
      finish(() => reject(new Error("histogram progress connection closed")));
    };
  });
}

async function init(): Promise<void> {
  ensureInitialized();
  const restoreFiltered = shouldRestoreHistogram && state.selectedMode === "filtered";
  shouldRestoreHistogram = false;
  await loadHistogram(restoreFiltered);
}

async function plotFilteredHistogram(): Promise<void> {
  if (state.selectedMode !== "filtered") {
    return;
  }
  await loadHistogram(true);
}

async function setSelectedRun(run: number | string | null): Promise<void> {
  const shell = useShellStore();
  shell.setSelectedRun(run);
  state.selectedRun = run === null ? null : Number(run);
  await loadHistogram();
}

async function setSelectedMetric(metric: HistogramMetric): Promise<void> {
  state.selectedPhase = metricPhase(metric);
  state.selectedMetric = metric;
  if (MODE_LOCKED_METRICS.has(metric)) {
    state.selectedMode = "all";
    state.filteredPlotDirty = false;
  }
  await loadHistogram();
}

async function setSelectedPhase(phase: HistogramPhase): Promise<void> {
  state.selectedPhase = phase;
  const allowedMetrics = PHASE_METRICS[phase];
  if (!allowedMetrics.includes(state.selectedMetric)) {
    state.selectedMetric = allowedMetrics[0];
  }
  if (MODE_LOCKED_METRICS.has(state.selectedMetric)) {
    state.selectedMode = "all";
    state.filteredPlotDirty = false;
  }
  await loadHistogram();
}

async function setSelectedVariant(variant: HistogramVariant): Promise<void> {
  if (
    state.selectedMetric === "bitflip"
    && (variant === "baseline" || variant === "value" || variant === "length" || variant === "count")
  ) {
    state.selectedBitflipVariant = variant;
  } else if (
    state.selectedMetric === "saturation"
    && (variant === "drop" || variant === "length")
  ) {
    state.selectedSaturationVariant = variant;
  } else {
    return;
  }
  await loadHistogram();
}

async function setSelectedMode(mode: HistogramMode): Promise<void> {
  if (MODE_LOCKED_METRICS.has(state.selectedMetric)) {
    state.selectedMode = "all";
    return;
  }
  state.selectedMode = mode;
  if (mode === "filtered" && !state.selectedHistogramFilter) {
    const shell = useShellStore();
    state.selectedHistogramFilter =
      shell.state.bootstrap?.filterFiles?.[0]?.name || "";
  }
  if (mode !== "filtered") {
    state.filteredPlotDirty = false;
  }
  await loadHistogram();
}

async function setSelectedHistogramFilter(name: string): Promise<void> {
  state.selectedHistogramFilter = name;
  await loadHistogram();
}

async function setSelectedHistogramVeto(
  value: boolean | null | undefined,
): Promise<void> {
  state.selectedHistogramVeto = Boolean(value);
  await loadHistogram();
}

function setScaleMode(mode: ScaleMode): void {
  if (mode !== "linear" && mode !== "log") {
    return;
  }
  if (state.selectedMetric === "cdf") {
    state.cdfScaleMode = mode;
    return;
  }
  state.amplitudeScaleMode = mode;
}

function setCdfRenderMode(mode: CdfRenderMode): void {
  if (mode !== "2d" && mode !== "projection") {
    return;
  }
  state.cdfRenderMode = mode;
}

function setCdfProjectionBin(value: number | string): void {
  const parsed = Number.parseInt(String(value), 10);
  if (Number.isNaN(parsed)) {
    return;
  }
  state.cdfProjectionBin = Math.min(150, Math.max(1, parsed));
}

function reorderCurrentSeries(sourceKey: string, targetKey: string): void {
  const orderKey = currentSeriesOrderKey();
  if (!orderKey || sourceKey === targetKey) {
    return;
  }

  syncCurrentSeriesOrder();
  const currentOrder = [...(labeledSeriesOrder[orderKey] || [])];
  const sourceIndex = currentOrder.indexOf(sourceKey);
  const targetIndex = currentOrder.indexOf(targetKey);

  if (sourceIndex < 0 || targetIndex < 0) {
    return;
  }

  const [movedKey] = currentOrder.splice(sourceIndex, 1);
  currentOrder.splice(targetIndex, 0, movedKey);
  labeledSeriesOrder[orderKey] = currentOrder;
}

const scaleMode = computed<ScaleMode>(() =>
  state.selectedMetric === "cdf" ? state.cdfScaleMode : state.amplitudeScaleMode,
);

const orderedSeries = computed<HistogramSeries[]>(() => {
  const series = state.histogram?.series || [];
  const orderKey = currentSeriesOrderKey();
  if (!orderKey || series.length <= 1) {
    return series;
  }

  const order = labeledSeriesOrder[orderKey] || [];
  const positions = new Map(order.map((key, index) => [key, index]));
  return [...series].sort(
    (left, right) =>
      (positions.get(left.labelKey) ?? Number.MAX_SAFE_INTEGER) -
      (positions.get(right.labelKey) ?? Number.MAX_SAFE_INTEGER),
  );
});

export function useHistogramStore() {
  return {
    state,
    scaleMode,
    orderedSeries,
    init,
    getAvailability,
    loadHistogram,
    plotFilteredHistogram,
    setSelectedRun,
    setSelectedPhase,
    setSelectedMetric,
    setSelectedVariant,
    setSelectedMode,
    setSelectedHistogramFilter,
    setSelectedHistogramVeto,
    setScaleMode,
    setCdfRenderMode,
    setCdfProjectionBin,
    reorderCurrentSeries,
    applyUiState,
    serializeUiState,
  };
}

function applyUiState(payload: HistogramsUiState | null | undefined): void {
  if (!payload) {
    return;
  }
  shouldRestoreHistogram = true;
  state.selectedRun = payload.selectedRun;
  state.selectedPhase = payload.selectedPhase;
  state.selectedMetric = payload.selectedMetric;
  state.selectedMode = payload.selectedMode;
  state.selectedBitflipVariant = payload.selectedBitflipVariant;
  state.selectedSaturationVariant = payload.selectedSaturationVariant;
  state.selectedHistogramFilter = payload.selectedHistogramFilter;
  state.selectedHistogramVeto = payload.selectedHistogramVeto;
  state.cdfScaleMode = payload.cdfScaleMode;
  state.amplitudeScaleMode = payload.amplitudeScaleMode;
  state.cdfRenderMode = payload.cdfRenderMode;
  state.cdfProjectionBin = payload.cdfProjectionBin;
  for (const [key, values] of Object.entries(payload.labeledSeriesOrder || {})) {
    labeledSeriesOrder[key] = [...values];
  }
  ensureInitialized();
}

function serializeUiState(): HistogramsUiState {
  return {
    selectedRun: state.selectedRun,
    selectedPhase: state.selectedPhase,
    selectedMetric: state.selectedMetric,
    selectedMode: state.selectedMode,
    selectedBitflipVariant: state.selectedBitflipVariant,
    selectedSaturationVariant: state.selectedSaturationVariant,
    selectedHistogramFilter: state.selectedHistogramFilter,
    selectedHistogramVeto: state.selectedHistogramVeto,
    cdfScaleMode: state.cdfScaleMode,
    amplitudeScaleMode: state.amplitudeScaleMode,
    cdfRenderMode: state.cdfRenderMode,
    cdfProjectionBin: state.cdfProjectionBin,
    labeledSeriesOrder: Object.fromEntries(
      Object.entries(labeledSeriesOrder).map(([key, values]) => [key, [...values]]),
    ),
  };
}
