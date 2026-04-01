<template>
  <div class="viewer-shell">
    <AppSidebar
      title="Trace Viewer"
      :items="navItems"
      :active-page="state.page"
      @select="setPage"
    />

    <main class="main-panel">
      <HistogramView
        v-if="state.page === 'histograms'"
        :runs="state.bootstrap?.runs || []"
        :selected-run="state.selectedRun"
        :selected-metric="state.selectedMetric"
        :selected-mode="state.selectedMode"
        :filter-files="state.bootstrap?.filterFiles || []"
        :selected-histogram-filter="state.selectedHistogramFilter"
        :availability="getAvailability()"
        :histogram="state.histogram"
        :ordered-series="getOrderedHistogramSeries()"
        :loading="state.loading"
        :error="state.error"
        :status-message="state.statusMessage"
        :scale-mode="getCurrentScaleMode()"
        :cdf-render-mode="state.cdfRenderMode"
        :cdf-projection-bin="state.cdfProjectionBin"
        @select-run="setSelectedRun"
        @select-metric="setSelectedMetric"
        @select-mode="setSelectedMode"
        @select-histogram-filter="setSelectedHistogramFilter"
        @select-scale="setScaleMode"
        @select-cdf-render-mode="setCdfRenderMode"
        @select-cdf-projection-bin="setCdfProjectionBin"
        @reorder-series="reorderHistogramSeries"
      />

      <TraceReviewView
        v-else
        :filter-files="state.bootstrap?.filterFiles || []"
        :selected-filter="state.selectedFilter"
        :trace="state.currentTrace"
        :visual-mode="state.reviewVisualMode"
        :loading="state.loading"
        :error="state.error"
        :status-message="state.statusMessage"
        @select-filter="selectFilter"
        @set-visual-mode="setReviewVisualMode"
      />
    </main>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted } from "vue";
import AppSidebar from "../components/AppSidebar.vue";
import HistogramView from "../views/HistogramView.vue";
import TraceReviewView from "../views/TraceReviewView.vue";
import { useViewerStore } from "../stores/viewer";

const props = defineProps({
  bootstrap: { type: Object, required: true },
});

const navItems = [
  {
    id: "histograms",
    title: "Histograms",
  },
  {
    id: "review",
    title: "Trace Review",
  },
];

const {
  state,
  init,
  getAvailability,
  getOrderedHistogramSeries,
  getCurrentScaleMode,
  reorderHistogramSeries,
  setPage,
  setSelectedRun,
  setSelectedMetric,
  setSelectedMode,
  setSelectedHistogramFilter,
  setScaleMode,
  setCdfRenderMode,
  setCdfProjectionBin,
  setReviewVisualMode,
  selectFilter,
  handleKeydown,
} = useViewerStore();

onMounted(async () => {
  await init(props.bootstrap);
  window.addEventListener("keydown", handleKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", handleKeydown);
});
</script>
