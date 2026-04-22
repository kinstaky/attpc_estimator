<template>
  <v-container class="page-container" fluid>
    <div class="page-header">
      <div>
        <p class="page-kicker">Histograms</p>
        <h1>Accumulated trace histograms</h1>
        <p class="page-copy">
          Compare all-trace, labeled, and filtered distributions for the available signal metrics.
        </p>
      </div>
    </div>

    <v-card class="control-card" rounded="xl">
      <v-card-text>
        <v-row dense>
          <v-col cols="12" md="4">
            <v-select
              :items="runOptions"
              item-title="title"
              item-value="value"
              label="Run"
              :model-value="store.state.selectedRun"
              variant="outlined"
              @update:model-value="store.setSelectedRun"
            />
          </v-col>
          <v-col cols="12" md="4">
            <v-select
              :items="phaseOptions"
              item-title="title"
              item-value="value"
              label="Phase"
              :model-value="store.state.selectedPhase"
              variant="outlined"
              @update:model-value="store.setSelectedPhase"
            />
          </v-col>
          <v-col cols="12" md="4">
            <v-select
              :items="metricOptions"
              item-title="title"
              item-value="value"
              label="Metric"
              :model-value="store.state.selectedMetric"
              variant="outlined"
              @update:model-value="store.setSelectedMetric"
            />
          </v-col>
          <v-col
            v-if="variantOptions.length"
            cols="12"
            md="4"
          >
            <v-select
              :items="variantOptions"
              item-title="title"
              item-value="value"
              label="Variant"
              :model-value="selectedVariant"
              variant="outlined"
              @update:model-value="onVariantChange"
            />
          </v-col>
          <v-col
            v-if="store.state.selectedPhase !== 'phase2'"
            cols="12"
            md="4"
          >
            <v-select
              :items="modeOptions"
              item-title="title"
              item-value="value"
              label="Trace set"
              :model-value="store.state.selectedMode"
              variant="outlined"
              @update:model-value="store.setSelectedMode"
            />
          </v-col>
          <v-col v-if="store.state.selectedMode === 'filtered'" cols="12" md="6">
            <v-select
              :items="filterFileOptions"
              item-title="title"
              item-value="value"
              label="Filter file"
              :model-value="store.state.selectedHistogramFilter"
              variant="outlined"
              @update:model-value="store.setSelectedHistogramFilter"
            />
          </v-col>
          <v-col v-if="store.state.selectedMode === 'filtered'" cols="12" md="6">
            <v-switch
              :model-value="store.state.selectedHistogramVeto"
              color="primary"
              hide-details="auto"
              inset
              label="Veto filter file"
              messages="Plot traces not listed in the selected filter file."
              @update:model-value="store.setSelectedHistogramVeto"
            />
          </v-col>
          <v-col cols="12" md="3">
            <v-select
              :items="scaleOptions"
              item-title="title"
              item-value="value"
              :label="scaleLabel"
              :model-value="store.scaleMode.value"
              variant="outlined"
              @update:model-value="store.setScaleMode"
            />
          </v-col>
          <v-col v-if="store.state.selectedMetric === 'cdf'" cols="12" md="3">
            <v-select
              :items="cdfRenderOptions"
              item-title="title"
              item-value="value"
              label="CDF display"
              :model-value="store.state.cdfRenderMode"
              variant="outlined"
              @update:model-value="store.setCdfRenderMode"
            />
          </v-col>
          <v-col
            v-if="store.state.selectedMetric === 'cdf' && store.state.cdfRenderMode === 'projection'"
            cols="12"
            md="3"
          >
            <v-text-field
              label="Projection bin"
              :model-value="store.state.cdfProjectionBin"
              min="1"
              max="150"
              type="number"
              variant="outlined"
              @update:model-value="store.setCdfProjectionBin"
            />
          </v-col>
          <v-col v-if="store.state.selectedMode === 'filtered'" cols="12">
            <v-row dense>
              <v-col cols="12" md="3">
                <v-btn
                  block
                  color="primary"
                  :disabled="!canPlotFiltered"
                  :loading="store.state.loading"
                  @click="store.plotFilteredHistogram"
                >
                  Plot
                </v-btn>
              </v-col>
              <v-col
                v-if="store.state.filteredPlotDirty"
                cols="12"
                md="9"
                class="d-flex align-center"
              >
                <p class="page-copy">Settings changed. Click Plot to refresh.</p>
              </v-col>
            </v-row>
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <v-alert
      v-if="store.state.error"
      class="mt-4"
      color="error"
      icon="mdi-alert-circle-outline"
      rounded="xl"
      variant="tonal"
    >
      {{ store.state.error }}
    </v-alert>

    <div v-if="store.state.loading" class="empty-state">
      <div class="progress-state">
        <v-progress-linear
          :indeterminate="!store.state.progress"
          :model-value="store.state.progress?.percent || 0"
          color="primary"
          height="10"
          rounded
        />
        <p class="page-kicker">
          {{
            store.state.progress
              ? `${store.state.progress.percent}% · ${store.state.progress.current}/${store.state.progress.total} ${store.state.progress.unit}`
              : "Starting…"
          }}
        </p>
        <p v-if="store.state.progress?.message" class="page-copy">
          {{ store.state.progress.message }}
        </p>
      </div>
    </div>

    <LineDistancePlots
      v-else-if="store.state.selectedMetric === 'line_distance' && (store.state.histogram?.plots?.length || 0) > 0"
      class="mt-4"
      :plots="store.state.histogram?.plots || []"
      :scale-mode="store.scaleMode.value"
    />

    <v-row
      v-else-if="store.state.histogram?.series?.length"
      class="mt-4"
      dense
    >
      <v-col
        v-for="series in orderedSeries"
        :key="series.labelKey"
        cols="12"
        :lg="store.state.selectedMode === 'labeled' ? 4 : 12"
      >
        <v-card
          class="result-card-vuetify"
          :class="{
            'result-card-vuetify--draggable': isDraggable,
            'result-card-vuetify--drop-target': dropTargetKey === series.labelKey,
          }"
          rounded="xl"
          :draggable="isDraggable"
          @dragstart="onDragStart(series.labelKey)"
          @dragover.prevent="onDragOver(series.labelKey)"
          @drop.prevent="onDrop(series.labelKey)"
          @dragend="clearDragState"
        >
          <v-card-title class="result-card-title">
            <div>
              <p class="page-kicker">{{ series.labelKey }}</p>
              <h2>{{ series.title }}</h2>
            </div>
            <strong>
              {{ series.traceCount ?? series.histogram.length }}
              {{
                series.traceCount !== null && series.traceCount !== undefined
                  ? store.state.histogram?.metric === "coplanar"
                    ? "events"
                    : "traces"
                  : "bins"
              }}
            </strong>
          </v-card-title>
          <v-card-text>
            <ResultPlot
              :class="{ 'result-plot--all-traces': store.state.selectedMode === 'all' }"
              :metric="store.state.histogram.metric"
              :series="series"
              :thresholds="store.state.histogram.thresholds || []"
              :value-bin-count="store.state.histogram.valueBinCount || 0"
              :bin-count="store.state.histogram.binCount || 0"
              :bin-centers="store.state.histogram.binCenters || []"
              :variant="store.state.histogram.variant || ''"
              :bin-label="store.state.histogram.binLabel || ''"
              :count-label="store.state.histogram.countLabel || ''"
              :scale-mode="store.scaleMode.value"
              :cdf-render-mode="store.state.cdfRenderMode"
              :cdf-projection-bin="store.state.cdfProjectionBin"
            />
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-card v-else class="empty-card mt-4" rounded="xl" variant="tonal">
      <v-card-text>
        <p class="page-kicker">No data</p>
        <h2>No histogram artifacts are available for this selection.</h2>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

import LineDistancePlots from "../components/LineDistancePlots.vue";
import ResultPlot from "../components/ResultPlot.vue";
import { useHistogramStore } from "../stores/histograms";
import { useShellStore } from "../stores/shell";
import type { HistogramAvailabilityEntry, HistogramMetric, HistogramVariant } from "../types";

const shell = useShellStore();
const store = useHistogramStore();
const draggedSeriesKey = ref<string | null>(null);
const dropTargetKey = ref<string | null>(null);

const runOptions = computed(() =>
  (shell.state.bootstrap?.runs || []).map((run) => ({
    title: `Run ${run}`,
    value: Number(run),
  })),
);

const filterFileOptions = computed(() =>
  (shell.state.bootstrap?.filterFiles || []).map((item) => ({
    title: item.name,
    value: item.name,
  })),
);

const phaseOptions = [
  { title: "Phase 1", value: "phase1" },
  { title: "Phase 2", value: "phase2" },
];

const metricOptions = computed(() => {
  if (store.state.selectedPhase === "phase2") {
    return [
      { title: "Line distance", value: "line_distance" },
      { title: "Coplanarity (λ₃/λ₁)", value: "coplanar" },
    ];
  }
  return [
    { title: "Amplitude", value: "amplitude" },
    { title: "Baseline", value: "baseline" },
    { title: "Bitflip", value: "bitflip" },
    { title: "CDF", value: "cdf" },
    { title: "Saturation", value: "saturation" },
  ];
});

const selectedVariant = computed(() => {
  if (store.state.selectedMetric === "bitflip") {
    return store.state.selectedBitflipVariant;
  }
  if (store.state.selectedMetric === "saturation") {
    return store.state.selectedSaturationVariant;
  }
  return "";
});

const variantOptions = computed(() => {
  if (store.state.selectedMetric === "bitflip") {
    return [
      { title: "Baseline", value: "baseline" },
      { title: "Value", value: "value" },
      { title: "Length", value: "length" },
      { title: "Count", value: "count" },
    ];
  }
  if (store.state.selectedMetric === "saturation") {
    return [
      { title: "Drop", value: "drop" },
      { title: "Plateau length", value: "length" },
    ];
  }
  return [];
});

const modeOptions = computed(() => {
  const availability = store.getAvailability() as
    | Record<HistogramMetric, HistogramAvailabilityEntry>
    | null;
  const metricAvailability = availability?.[store.state.selectedMetric];
  return [
    { title: "All traces", value: "all", props: { disabled: !metricAvailability?.all } },
    { title: "Labeled", value: "labeled", props: { disabled: !metricAvailability?.labeled } },
    { title: "From file", value: "filtered", props: { disabled: !metricAvailability?.filtered } },
  ];
});

const scaleOptions = [
  { title: "Linear", value: "linear" },
  { title: "Log", value: "log" },
];

const cdfRenderOptions = [
  { title: "2D histogram", value: "2d" },
  { title: "1D projection", value: "projection" },
];

const scaleLabel = computed(() => {
  if (store.state.selectedMetric === "amplitude") {
    return "Y scale";
  }
  if (store.state.selectedMetric !== "cdf") {
    return "Y scale";
  }
  return store.state.cdfRenderMode === "projection" ? "Y scale" : "Z scale";
});

const orderedSeries = computed(() => store.orderedSeries.value);
const canPlotFiltered = computed(
  () =>
    store.state.selectedMode === "filtered" &&
    store.state.selectedRun !== null &&
    Boolean(store.state.selectedHistogramFilter) &&
    !store.state.loading,
);

const isDraggable = computed(
  () => store.state.selectedMode === "labeled" && orderedSeries.value.length > 1,
);

function onDragStart(labelKey: string): void {
  if (!isDraggable.value) {
    return;
  }
  draggedSeriesKey.value = labelKey;
  dropTargetKey.value = null;
}

function onDragOver(labelKey: string): void {
  if (!isDraggable.value || !draggedSeriesKey.value || draggedSeriesKey.value === labelKey) {
    dropTargetKey.value = null;
    return;
  }
  dropTargetKey.value = labelKey;
}

function onDrop(labelKey: string): void {
  if (!draggedSeriesKey.value || draggedSeriesKey.value === labelKey) {
    clearDragState();
    return;
  }
  store.reorderCurrentSeries(draggedSeriesKey.value, labelKey);
  clearDragState();
}

function clearDragState(): void {
  draggedSeriesKey.value = null;
  dropTargetKey.value = null;
}

function onVariantChange(value: string): void {
  void store.setSelectedVariant(value as HistogramVariant);
}

onMounted(() => {
  void store.init();
});
</script>
