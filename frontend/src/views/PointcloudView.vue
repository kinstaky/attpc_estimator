<template>
  <v-container class="page-container" fluid>
    <div class="page-header">
      <div>
        <p class="page-kicker">Browse</p>
        <h1>Pointcloud review</h1>
        <p class="page-copy">
          Inspect phase-1 pointcloud events by direct event id or by saved pointcloud labels, with synced 2D/3D views and trace overlays.
        </p>
      </div>
    </div>

    <v-card class="control-card" rounded="xl">
      <v-card-text>
        <v-row dense>
          <v-col cols="12" md="3">
            <v-select
              :items="sourceOptions"
              item-title="title"
              item-value="value"
              label="Browse source"
              :model-value="store.state.source"
              variant="outlined"
              @update:model-value="store.setSource"
            />
          </v-col>
          <v-col cols="12" md="3">
            <v-select
              :items="runOptions"
              item-title="title"
              item-value="value"
              label="Run"
              :model-value="store.state.selectedRun"
              variant="outlined"
              @update:model-value="store.setRun"
            />
          </v-col>
          <template v-if="store.state.source === 'event_id'">
            <v-col cols="12" md="3">
              <v-text-field
                label="Event id"
                type="number"
                :model-value="store.state.selectedEventId"
                :hint="eventRangeText"
                persistent-hint
                variant="outlined"
                @update:model-value="store.setEventId"
              />
            </v-col>
          </template>
          <template v-else>
            <v-col cols="12" md="3">
              <v-select
                :items="pointcloudLabelOptions"
                item-title="title"
                item-value="value"
                label="Pointcloud label"
                :model-value="store.state.selectedLabel"
                variant="outlined"
                @update:model-value="store.setSelectedLabel"
              />
            </v-col>
          </template>
          <v-col cols="12" md="2" class="d-flex align-end">
            <v-switch
              class="pointcloud-layout-switch"
              inset
              color="primary"
              :label="`Layout ${store.state.layoutMode}`"
              :model-value="store.state.layoutMode === '2x2'"
              @update:model-value="toggleLayout"
            />
          </v-col>
          <v-col cols="12" md="1" class="d-flex align-end justify-end">
            <v-btn
              class="pointcloud-load-button"
              color="primary"
              :loading="store.state.loading"
              @click="store.loadEvent"
            >
              Load
            </v-btn>
          </v-col>
        </v-row>

        <v-row dense class="mt-2">
          <v-col
            v-for="panelIndex in store.panelCount.value"
            :key="panelIndex"
            cols="12"
            md="3"
          >
            <v-select
              :items="plotOptions"
              item-title="title"
              item-value="value"
              :label="`Panel ${panelIndex}`"
              :model-value="store.state.panelTypes[panelIndex - 1]"
              variant="outlined"
              @update:model-value="(value) => store.setPanelType(panelIndex - 1, value)"
            />
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

    <v-alert
      v-else-if="store.state.statusMessage"
      class="mt-4"
      color="secondary"
      icon="mdi-information-outline"
      rounded="xl"
      variant="tonal"
    >
      {{ store.state.statusMessage }}
    </v-alert>

    <v-card v-if="store.state.eventPayload" class="trace-stage-card mt-4" rounded="xl">
      <v-card-title class="trace-stage-title">
        <div>
          <p class="page-kicker">Run {{ store.state.eventPayload.run }} · Event {{ store.state.eventPayload.eventId }}</p>
          <h2>{{ store.state.eventPayload.hits.length }} in-range hits</h2>
        </div>
        <div class="trace-stage-hints">
          <span>{{ sourceSummary }}</span>
          <span>Press F to switch layout, J/K or arrows to navigate.</span>
        </div>
      </v-card-title>

      <v-card-text>
        <div class="trace-action-toolbar">
          <div class="trace-action-group">
            <v-btn variant="text" @click="store.previousEvent">Previous event</v-btn>
            <v-btn variant="text" @click="store.nextEvent">Next event</v-btn>
            <v-btn variant="text" @click="store.clearSelection">Clear selection</v-btn>
          </div>
          <div class="trace-action-group">
            <span class="text-medium-emphasis">FFT window {{ store.state.eventPayload.processing.fftWindowScale }}</span>
          </div>
        </div>

        <div class="pointcloud-grid" :class="gridClass">
          <section
            v-for="panelIndex in store.panelCount.value"
            :key="panelIndex"
            class="pointcloud-panel"
          >
            <PointcloudPlot
              :plot-type="store.state.panelTypes[panelIndex - 1]"
              :layout-mode="store.state.layoutMode"
              :hits="store.state.eventPayload.hits"
              :mapping-pads="store.state.mappingPads"
              :selected-trace-ids="store.state.selectedTraceIds"
              :traces="store.state.panelTypes[panelIndex - 1] === 'traces' ? store.state.tracePayload.traces : []"
              :xy-range="store.state.xyRange"
              :projected-range="store.state.projectedRange"
              :camera="store.state.camera"
              @toggle-traces="store.toggleTraceIds"
              @update-xy-range="store.updateXYRange"
              @update-projected-range="store.updateProjectedRange"
              @update-camera="store.updateCamera"
            />
          </section>
        </div>
      </v-card-text>
    </v-card>

    <v-card v-else class="empty-card mt-4" rounded="xl" variant="tonal">
      <v-card-text>
        <p class="page-kicker">No pointcloud event selected</p>
        <h2>Choose a source and load a pointcloud event.</h2>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, watch } from "vue";

import PointcloudPlot from "../components/PointcloudPlot.vue";
import { usePointcloudStore } from "../stores/pointcloud";
import { useShellStore } from "../stores/shell";

const shell = useShellStore();
const store = usePointcloudStore();
const sourceOptions = store.sourceOptions;
const plotOptions = store.plotOptions;
const pointcloudLabelOptions = store.pointcloudLabelOptions;

const runOptions = computed(() =>
  (shell.state.bootstrap?.pointcloudRuns || []).map((run) => ({
    title: `Run ${run}`,
    value: Number(run),
  })),
);

const pointcloudEventRanges = computed(() => shell.state.bootstrap?.pointcloudEventRanges || {});

const eventRangeText = computed(() => {
  if (store.state.selectedRun === null) {
    return "No run selected";
  }
  const range = pointcloudEventRanges.value[String(store.state.selectedRun)];
  if (!range) {
    return "No pointcloud file for this run";
  }
  return `Available ${range.min} to ${range.max}`;
});

const sourceSummary = computed(() => (
  store.state.source === "label_set"
    ? `Browse labeled pointcloud events${store.state.selectedLabel ? ` with label ${store.state.selectedLabel}` : ""}.`
    : "Browse direct pointcloud events by event id."
));

const gridClass = computed(() => ({
  "pointcloud-grid--single": store.state.layoutMode === "1x1",
  "pointcloud-grid--quad": store.state.layoutMode === "2x2",
}));

function toggleLayout(value) {
  store.setLayoutMode(value ? "2x2" : "1x1");
}

function shouldIgnoreKey(event) {
  const tagName = event.target?.tagName?.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

function onKeydown(event) {
  if (shouldIgnoreKey(event) || store.state.loading || !store.state.eventPayload) {
    return;
  }
  const key = event.key.toLowerCase();
  if (key === "f") {
    event.preventDefault();
    store.toggleLayoutMode();
    return;
  }
  if (key === "j" || key === "arrowdown" || key === "arrowright") {
    event.preventDefault();
    void store.nextEvent();
    return;
  }
  if (key === "k" || key === "arrowup" || key === "arrowleft") {
    event.preventDefault();
    void store.previousEvent();
  }
}

watch(
  () => store.state.selectedTraceIds.slice(),
  () => {
    void store.loadSelectedTraces();
  },
);

onMounted(async () => {
  window.addEventListener("keydown", onKeydown);
  await store.init();
  const session = shell.state.bootstrap?.session;
  if (session?.mode === "pointcloud") {
    if (session.source === "label_set") {
      store.setSource("label_set");
      store.setSelectedLabel(session.label || "");
    } else {
      store.setSource("event_id");
    }
    if (session.run !== null && session.run !== undefined) {
      store.setRun(session.run);
    }
    if (session.eventId !== null && session.eventId !== undefined) {
      store.setEventId(session.eventId);
    }
    try {
      await store.restoreCurrentSession();
    } catch {
      // Store already holds the error.
    }
    return;
  }
  if (store.state.selectedRun !== null) {
    await store.loadEvent();
  }
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKeydown);
});
</script>

<style scoped>
.pointcloud-layout-switch {
  width: 100%;
  margin-inline-end: 12px;
}

.pointcloud-load-button {
  min-width: 96px;
}
</style>
