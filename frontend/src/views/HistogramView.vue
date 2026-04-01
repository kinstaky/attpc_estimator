<template>
  <section class="viewer-panel viewer-panel--histograms">
    <header class="content-navbar">
      <div class="content-navbar-main">
        <div class="content-navbar-title">
        <h1>Accumulated histograms</h1>
        </div>

        <label class="viewer-control content-navbar-control content-navbar-control--run">
          <span class="meta-title">Run</span>
          <select :value="selectedRun" @change="$emit('select-run', Number($event.target.value))">
            <option v-for="run in runs" :key="run" :value="run">Run {{ run }}</option>
          </select>
        </label>
      </div>

      <div class="content-navbar-filters">
        <div class="viewer-control content-navbar-control">
          <span class="meta-title">Metric</span>
          <div class="segmented-control segmented-control--tabs">
            <button
              v-for="metricOption in ['cdf', 'amplitude']"
              :key="metricOption"
              type="button"
              class="segmented-button segmented-button--tab"
              :class="{ active: selectedMetric === metricOption }"
              @click="$emit('select-metric', metricOption)"
            >
              {{ metricOption === 'cdf' ? 'CDF' : 'Amplitude' }}
            </button>
          </div>
        </div>

        <div class="viewer-control content-navbar-control">
          <span class="meta-title">Trace set</span>
          <div class="segmented-control">
            <button
              v-for="modeOption in traceSetOptions"
              :key="modeOption.value"
              type="button"
              class="segmented-button"
              :class="{ active: selectedMode === modeOption.value }"
              :disabled="!availability?.[selectedMetric]?.[modeOption.value]"
              @click="$emit('select-mode', modeOption.value)"
            >
              {{ modeOption.label }}
            </button>
          </div>
        </div>
      </div>

      <div class="content-navbar-filters content-navbar-filters--secondary">
        <label
          v-if="selectedMode === 'filtered'"
          class="viewer-control content-navbar-control content-navbar-control--filter"
        >
          <span class="meta-title">Filter file</span>
          <select
            :value="selectedHistogramFilter || ''"
            @change="$emit('select-histogram-filter', $event.target.value)"
          >
            <option v-for="item in filterFiles" :key="item.name" :value="item.name">
              {{ item.name }}
            </option>
          </select>
        </label>

        <div class="viewer-control content-navbar-control">
          <span class="meta-title">{{ scaleLabel }}</span>
          <div class="segmented-control">
            <button
              v-for="scaleOption in ['linear', 'log']"
              :key="scaleOption"
              type="button"
              class="segmented-button"
              :class="{ active: scaleMode === scaleOption }"
              @click="$emit('select-scale', scaleOption)"
            >
              {{ scaleOption === 'linear' ? 'Linear' : 'Log' }}
            </button>
          </div>
        </div>

        <template v-if="selectedMetric === 'cdf'">
          <div class="viewer-control content-navbar-control">
            <span class="meta-title">CDF display</span>
            <div class="segmented-control">
              <button
                v-for="renderOption in renderOptions"
                :key="renderOption.value"
                type="button"
                class="segmented-button"
                :class="{ active: cdfRenderMode === renderOption.value }"
                @click="$emit('select-cdf-render-mode', renderOption.value)"
              >
                {{ renderOption.label }}
              </button>
            </div>
          </div>

          <label
            v-if="cdfRenderMode === 'projection'"
            class="viewer-control content-navbar-control content-navbar-control--bin"
          >
            <span class="meta-title">Projection bin</span>
            <input
              class="viewer-number-input"
              type="number"
              min="1"
              max="150"
              :value="cdfProjectionBin"
              @input="$emit('select-cdf-projection-bin', Number($event.target.value))"
            />
          </label>
        </template>
      </div>
    </header>

    <div class="status-strip">
      <span v-if="loading">Loading…</span>
      <span v-else-if="error" class="status-error">{{ error }}</span>
      <span v-else-if="statusMessage">{{ statusMessage }}</span>
      <span v-else>&nbsp;</span>
    </div>

    <section
      v-if="histogram && orderedSeries.length"
      class="result-grid"
      :class="resultGridClass"
    >
      <article
        v-for="series in orderedSeries"
        :key="series.labelKey"
        class="result-card"
        :class="[
          resultCardClass,
          { 'result-card--draggable': canDragCards, 'result-card--drag-over': dropTargetLabelKey === series.labelKey },
        ]"
        :draggable="canDragCards"
        @dragstart="onDragStart(series.labelKey)"
        @dragend="onDragEnd"
        @dragover.prevent="onDragOver(series.labelKey)"
        @drop.prevent="onDrop(series.labelKey)"
      >
        <header class="result-card-header">
          <div>
            <p class="progress-kicker">{{ series.labelKey }}</p>
            <h2>{{ series.title }}</h2>
          </div>
          <div class="result-card-meta">
            <strong v-if="series.traceCount !== null && series.traceCount !== undefined">
              {{ series.traceCount }} traces
            </strong>
            <strong v-else>{{ sumCounts(series.histogram) }} peaks</strong>
          </div>
        </header>
        <ResultPlot
          :metric="histogram.metric"
          :series="series"
          :thresholds="histogram.thresholds || []"
          :value-bin-count="histogram.valueBinCount || 0"
          :bin-count="histogram.binCount || 0"
          :scale-mode="scaleMode"
          :cdf-render-mode="cdfRenderMode"
          :cdf-projection-bin="cdfProjectionBin"
        />
      </article>
    </section>

    <section v-else class="empty-card">
      <p class="eyebrow">No data</p>
      <h2>No histogram artifacts are available for this selection.</h2>
      <p class="viewer-copy">
        Generate the matching `cdf` or `amplitude` output in the workspace, then reload this page.
      </p>
    </section>
  </section>
</template>

<script setup>
import { computed, ref } from "vue";

import ResultPlot from "../components/ResultPlot.vue";

const props = defineProps({
  runs: { type: Array, required: true },
  selectedRun: { type: Number, default: null },
  selectedMetric: { type: String, required: true },
  selectedMode: { type: String, required: true },
  filterFiles: { type: Array, required: true },
  selectedHistogramFilter: { type: String, default: "" },
  availability: { type: Object, default: null },
  histogram: { type: Object, default: null },
  orderedSeries: { type: Array, required: true },
  loading: { type: Boolean, default: false },
  error: { type: String, default: "" },
  statusMessage: { type: String, default: "" },
  scaleMode: { type: String, required: true },
  cdfRenderMode: { type: String, required: true },
  cdfProjectionBin: { type: Number, required: true },
});

const emit = defineEmits([
  "select-run",
  "select-metric",
  "select-mode",
  "select-histogram-filter",
  "select-scale",
  "select-cdf-render-mode",
  "select-cdf-projection-bin",
  "reorder-series",
]);

const draggedLabelKey = ref(null);
const dropTargetLabelKey = ref(null);

const renderOptions = [
  { value: "2d", label: "2D histogram" },
  { value: "projection", label: "1D projection" },
];

const traceSetOptions = [
  { value: "all", label: "All traces" },
  { value: "labeled", label: "Labeled" },
  { value: "filtered", label: "From file" },
];

const scaleLabel = computed(() => {
  if (props.selectedMetric === "amplitude") {
    return "Y scale";
  }
  return props.cdfRenderMode === "projection" ? "Y scale" : "Z scale";
});

const resultGridClass = computed(() => ({
  "result-grid--single": props.selectedMode !== "labeled",
  "result-grid--triple": props.selectedMode === "labeled",
}));

const resultCardClass = computed(() => ({
  "result-card--emphasis": props.selectedMode !== "labeled",
  "result-card--cdf": props.selectedMetric === "cdf",
}));

const canDragCards = computed(() => props.orderedSeries.length > 1);

function onDragStart(labelKey) {
  if (!canDragCards.value) {
    return;
  }
  draggedLabelKey.value = labelKey;
}

function onDragEnd() {
  draggedLabelKey.value = null;
  dropTargetLabelKey.value = null;
}

function onDragOver(labelKey) {
  if (!canDragCards.value || draggedLabelKey.value === labelKey) {
    return;
  }
  dropTargetLabelKey.value = labelKey;
}

function onDrop(labelKey) {
  if (!canDragCards.value || !draggedLabelKey.value) {
    return;
  }
  if (draggedLabelKey.value !== labelKey) {
    emit("reorder-series", draggedLabelKey.value, labelKey);
  }
  onDragEnd();
}

function sumCounts(values) {
  return values.reduce((total, value) => total + Number(value || 0), 0);
}
</script>
