import { reactive } from "vue";
import {
  getHistogram,
  nextReviewTrace,
  previousReviewTrace,
  selectReviewFilter,
} from "../api";

const DEFAULT_CDF_PROJECTION_BIN = 60;

const state = reactive({
  bootstrap: null,
  page: "histograms",
  selectedRun: null,
  selectedMetric: "cdf",
  selectedMode: "all",
  selectedHistogramFilter: "",
  histogram: null,
  selectedFilter: "",
  currentTrace: null,
  reviewVisualMode: "analysis",
  cdfScaleMode: "linear",
  amplitudeScaleMode: "linear",
  cdfRenderMode: "2d",
  cdfProjectionBin: DEFAULT_CDF_PROJECTION_BIN,
  histogramOrders: {},
  loading: false,
  error: "",
  statusMessage: "",
});

function clearTransientUi() {
  state.error = "";
  state.statusMessage = "";
}

function getAvailability(run = state.selectedRun) {
  if (run === null || !state.bootstrap) {
    return null;
  }
  return state.bootstrap.histogramAvailability?.[String(run)] || null;
}

function ensureModeAvailability() {
  const availability = getAvailability();
  if (!availability) {
    return;
  }
  const metricAvailability = availability?.[state.selectedMetric] || {};
  if (!metricAvailability?.[state.selectedMode]) {
    state.selectedMode = ["all", "labeled", "filtered"].find(
      (mode) => metricAvailability?.[mode],
    ) || "all";
  }
}

function getCurrentScaleMode() {
  return state.selectedMetric === "cdf" ? state.cdfScaleMode : state.amplitudeScaleMode;
}

function getCurrentHistogramOrderKey() {
  if (state.selectedRun === null) {
    return null;
  }
  const cdfKey = state.selectedMetric === "cdf" ? state.cdfRenderMode : "base";
  const filterKey = state.selectedMode === "filtered"
    ? state.selectedHistogramFilter || "no-filter"
    : "base";
  return [
    state.selectedRun,
    state.selectedMetric,
    state.selectedMode,
    cdfKey,
    filterKey,
  ].join(":");
}

function syncHistogramOrder(series) {
  const key = getCurrentHistogramOrderKey();
  if (key === null) {
    return;
  }
  const availableKeys = series.map((item) => item.labelKey);
  const existingOrder = state.histogramOrders[key] || [];
  const nextOrder = existingOrder.filter((labelKey) => availableKeys.includes(labelKey));
  for (const labelKey of availableKeys) {
    if (!nextOrder.includes(labelKey)) {
      nextOrder.push(labelKey);
    }
  }
  state.histogramOrders[key] = nextOrder;
}

function getOrderedHistogramSeries() {
  const series = state.histogram?.series || [];
  if (series.length <= 1) {
    return series;
  }
  const key = getCurrentHistogramOrderKey();
  if (key === null) {
    return series;
  }
  const order = state.histogramOrders[key];
  if (!order || order.length === 0) {
    return series;
  }
  const seriesByKey = new Map(series.map((item) => [item.labelKey, item]));
  return order
    .map((labelKey) => seriesByKey.get(labelKey))
    .filter(Boolean);
}

function reorderHistogramSeries(sourceLabelKey, targetLabelKey) {
  if (
    !sourceLabelKey
    || !targetLabelKey
    || sourceLabelKey === targetLabelKey
  ) {
    return;
  }
  const series = state.histogram?.series || [];
  if (series.length <= 1) {
    return;
  }
  syncHistogramOrder(series);
  const key = getCurrentHistogramOrderKey();
  if (key === null) {
    return;
  }
  const order = [...(state.histogramOrders[key] || [])];
  const sourceIndex = order.indexOf(sourceLabelKey);
  const targetIndex = order.indexOf(targetLabelKey);
  if (sourceIndex === -1 || targetIndex === -1) {
    return;
  }
  order.splice(sourceIndex, 1);
  order.splice(targetIndex, 0, sourceLabelKey);
  state.histogramOrders[key] = order;
}

function setScaleMode(mode) {
  if (mode !== "linear" && mode !== "log") {
    return;
  }
  if (state.selectedMetric === "cdf") {
    state.cdfScaleMode = mode;
    return;
  }
  state.amplitudeScaleMode = mode;
}

function setCdfRenderMode(mode) {
  if (mode !== "2d" && mode !== "projection") {
    return;
  }
  state.cdfRenderMode = mode;
  syncHistogramOrder(state.histogram?.series || []);
}

function setCdfProjectionBin(value) {
  const parsed = Number.parseInt(String(value), 10);
  if (Number.isNaN(parsed)) {
    return;
  }
  state.cdfProjectionBin = Math.min(150, Math.max(1, parsed));
}

async function init(initialBootstrap) {
  state.bootstrap = initialBootstrap;
  state.selectedRun = (initialBootstrap?.runs || [])[0] ?? null;
  state.selectedHistogramFilter = (initialBootstrap?.filterFiles || [])[0]?.name || "";
  state.selectedFilter = "";
  state.currentTrace = null;
  state.histogram = null;
  state.page = "histograms";
  state.selectedMetric = "cdf";
  state.selectedMode = "all";
  state.cdfScaleMode = "linear";
  state.amplitudeScaleMode = "linear";
  state.cdfRenderMode = "2d";
  state.cdfProjectionBin = DEFAULT_CDF_PROJECTION_BIN;
  state.histogramOrders = {};
  clearTransientUi();

  if (state.selectedRun !== null) {
    ensureModeAvailability();
    await loadHistogram();
  }
}

async function loadHistogram() {
  if (state.selectedRun === null) {
    state.histogram = null;
    return;
  }
  state.loading = true;
  clearTransientUi();
  ensureModeAvailability();
  try {
    state.histogram = await getHistogram(
      state.selectedMetric,
      state.selectedMode,
      state.selectedRun,
      state.selectedMode === "filtered" ? state.selectedHistogramFilter : "",
    );
    syncHistogramOrder(state.histogram?.series || []);
  } catch (error) {
    state.histogram = null;
    state.error = error.message;
  } finally {
    state.loading = false;
  }
}

async function setSelectedRun(run) {
  state.selectedRun = run;
  await loadHistogram();
}

async function setSelectedMetric(metric) {
  state.selectedMetric = metric;
  await loadHistogram();
}

async function setSelectedMode(mode) {
  state.selectedMode = mode;
  if (mode === "filtered" && !state.selectedHistogramFilter) {
    state.selectedHistogramFilter = (state.bootstrap?.filterFiles || [])[0]?.name || "";
  }
  await loadHistogram();
}

async function setSelectedHistogramFilter(name) {
  state.selectedHistogramFilter = name;
  await loadHistogram();
}

function setPage(page) {
  state.page = page;
  clearTransientUi();
}

function setReviewVisualMode(mode) {
  if (mode !== "raw" && mode !== "analysis") {
    return;
  }
  state.reviewVisualMode = mode;
}

function toggleReviewVisualMode() {
  state.reviewVisualMode = state.reviewVisualMode === "raw" ? "analysis" : "raw";
}

async function selectFilter(name) {
  if (!name) {
    return;
  }
  state.loading = true;
  clearTransientUi();
  try {
    const payload = await selectReviewFilter(name);
    state.selectedFilter = name;
    state.currentTrace = payload.trace;
    state.page = "review";
    if (!payload.trace) {
      state.statusMessage = "The selected filter file does not contain any traces.";
    }
  } catch (error) {
    state.error = error.message;
  } finally {
    state.loading = false;
  }
}

async function nextTrace() {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentTrace = await nextReviewTrace();
  } catch (error) {
    state.error = error.message;
  } finally {
    state.loading = false;
  }
}

async function previousTrace() {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentTrace = await previousReviewTrace();
  } catch (error) {
    state.error = error.message;
  } finally {
    state.loading = false;
  }
}

function shouldIgnoreKey(event) {
  const tagName = event.target?.tagName?.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

async function handleKeydown(event) {
  if (
    shouldIgnoreKey(event)
    || state.loading
    || state.page !== "review"
    || !state.currentTrace
  ) {
    return;
  }

  const key = event.key.toLowerCase();
  if (key === "f") {
    event.preventDefault();
    toggleReviewVisualMode();
    return;
  }
  if (key === "arrowup" || key === "k") {
    event.preventDefault();
    await previousTrace();
    return;
  }
  if (key === "arrowdown" || key === "j") {
    event.preventDefault();
    await nextTrace();
  }
}

export function useViewerStore() {
  return {
    state,
    init,
    getAvailability,
    getOrderedHistogramSeries,
    getCurrentScaleMode,
    loadHistogram,
    setSelectedRun,
    setSelectedMetric,
    setSelectedMode,
    setSelectedHistogramFilter,
    setScaleMode,
    setCdfRenderMode,
    setCdfProjectionBin,
    setPage,
    setReviewVisualMode,
    reorderHistogramSeries,
    toggleReviewVisualMode,
    selectFilter,
    nextTrace,
    previousTrace,
    handleKeydown,
  };
}
