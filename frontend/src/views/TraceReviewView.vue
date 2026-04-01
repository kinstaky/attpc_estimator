<template>
  <section class="viewer-panel">
    <header class="content-navbar">
      <div class="content-navbar-main">
        <div class="content-navbar-title">
          <p class="eyebrow">Filtered Trace Review</p>
          <h1>Trace review</h1>
          <p class="viewer-copy">
            Load a generated filter file, then step through the selected traces with the same raw and analysis views used in labeling.
          </p>
        </div>
      </div>

      <div class="content-navbar-filters">
        <label class="viewer-control content-navbar-control content-navbar-control--wide">
          <span class="meta-title">Filter file</span>
          <select :value="selectedFilter || ''" @change="$emit('select-filter', $event.target.value)">
            <option disabled value="">Select a filter file</option>
            <option v-for="item in filterFiles" :key="item.name" :value="item.name">{{ item.name }}</option>
          </select>
        </label>

        <div class="viewer-control content-navbar-control viewer-toggle-group">
          <span class="meta-title">Plot mode</span>
          <div class="segmented-control">
            <button
              v-for="modeOption in ['raw', 'analysis']"
              :key="modeOption"
              type="button"
              class="segmented-button"
              :class="{ active: visualMode === modeOption }"
              @click="$emit('set-visual-mode', modeOption)"
            >
              {{ modeOption === 'raw' ? 'Raw' : 'Analysis' }}
            </button>
          </div>
        </div>
      </div>
    </header>

    <div class="status-strip">
      <span v-if="loading">Loading…</span>
      <span v-else-if="error" class="status-error">{{ error }}</span>
      <span v-else-if="statusMessage">{{ statusMessage }}</span>
      <span v-else>&nbsp;</span>
    </div>

    <section v-if="trace" class="label-panel viewer-trace-panel">
      <header class="label-header">
        <div>
          <p class="eyebrow">Run {{ trace.run }} · Event {{ trace.eventId }} · Trace {{ trace.traceId }}</p>
          <h1>{{ currentLabelText(trace.currentLabel) }}</h1>
        </div>
        <div class="label-header-actions">
          <span class="visual-mode-hint">Press F to toggle. Use J/K or arrows to navigate.</span>
        </div>
      </header>

      <TracePlot :trace="trace" :visual-mode="visualMode" />

      <div class="trace-meta">
        <div class="meta-card">
          <span class="meta-title">Trace key</span>
          <strong>{{ trace.run }} / {{ trace.eventId }} / {{ trace.traceId }}</strong>
        </div>
        <div class="meta-card">
          <span class="meta-title">Filter file</span>
          <strong>{{ selectedFilter }}</strong>
        </div>
        <div class="meta-card" v-if="trace.reviewProgress">
          <span class="meta-title">Review progress</span>
          <strong>{{ trace.reviewProgress.current }} / {{ trace.reviewProgress.total }}</strong>
        </div>
      </div>
    </section>

    <section v-else class="empty-card">
      <p class="eyebrow">No trace selected</p>
      <h2>Select a filter file to begin reviewing traces.</h2>
      <p class="viewer-copy">
        The backend discovers `filter_*.npy` files in the workspace and exposes them here automatically.
      </p>
    </section>
  </section>
</template>

<script setup>
import TracePlot from "../components/TracePlot.vue";

defineProps({
  filterFiles: { type: Array, required: true },
  selectedFilter: { type: String, default: "" },
  trace: { type: Object, default: null },
  visualMode: { type: String, required: true },
  loading: { type: Boolean, default: false },
  error: { type: String, default: "" },
  statusMessage: { type: String, default: "" },
});

defineEmits(["select-filter", "set-visual-mode"]);

function currentLabelText(label) {
  if (!label) {
    return "Unlabeled";
  }
  if (label.family === "normal") {
    return label.label === "9" ? "9+ peaks" : `${label.label} peak${label.label === "1" ? "" : "s"}`;
  }
  return `Strange: ${label.label}`;
}
</script>
