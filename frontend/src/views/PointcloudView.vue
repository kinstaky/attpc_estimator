<template>
  <v-container class="page-container" fluid>
    <div class="page-header">
      <div>
        <p class="page-kicker">Phase 1</p>
        <h1>Pointcloud viewer</h1>
        <p class="page-copy">
          Inspect first-stage ATTPC hits, sync 2D/3D point views, and overlay baseline-removed traces for selected pads.
        </p>
      </div>
    </div>

    <v-card class="control-card" rounded="xl">
      <v-card-text>
        <v-row dense>
          <v-col cols="12" md="2">
            <v-select
              :items="runOptions"
              item-title="title"
              item-value="value"
              label="Run"
              :model-value="selectedRun"
              variant="outlined"
              @update:model-value="setRun"
            />
          </v-col>
          <v-col cols="12" md="2">
            <v-text-field
              label="Event id"
              type="number"
              :model-value="selectedEventId"
              :hint="eventRangeText"
              persistent-hint
              variant="outlined"
              @update:model-value="setEventId"
            />
          </v-col>
          <v-col cols="12" md="2">
            <v-select
              :items="layoutOptions"
              item-title="title"
              item-value="value"
              label="Layout"
              :model-value="layoutMode"
              variant="outlined"
              @update:model-value="setLayoutMode"
            />
          </v-col>
          <v-col cols="12" md="6" class="d-flex align-center justify-end ga-2">
            <v-btn variant="text" @click="previousEvent">Previous event</v-btn>
            <v-btn variant="text" @click="nextEvent">Next event</v-btn>
            <v-btn color="primary" :loading="loading" @click="loadEvent">Load event</v-btn>
          </v-col>
        </v-row>

        <v-row dense class="mt-2">
          <v-col
            v-for="panelIndex in panelCount"
            :key="panelIndex"
            cols="12"
            md="3"
          >
            <v-select
              :items="plotOptions"
              item-title="title"
              item-value="value"
              :label="`Panel ${panelIndex}`"
              :model-value="panelTypes[panelIndex - 1]"
              variant="outlined"
              @update:model-value="(value) => setPanelType(panelIndex - 1, value)"
            />
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <v-alert
      v-if="error"
      class="mt-4"
      color="error"
      icon="mdi-alert-circle-outline"
      rounded="xl"
      variant="tonal"
    >
      {{ error }}
    </v-alert>

    <v-card v-if="eventPayload" class="trace-stage-card mt-4" rounded="xl">
      <v-card-title class="trace-stage-title">
        <div>
          <p class="page-kicker">Run {{ eventPayload.run }} · Event {{ eventPayload.eventId }}</p>
          <h2>{{ eventPayload.hits.length }} hits</h2>
        </div>
        <div class="trace-stage-hints">
          <span>{{ selectedTraceIds.length }} selected traces</span>
          <span>Selection starts from pad views.</span>
        </div>
      </v-card-title>

      <v-card-text>
        <div class="trace-action-toolbar">
          <div class="trace-action-group">
            <v-btn variant="text" @click="clearSelection">Clear selection</v-btn>
          </div>
          <div class="trace-action-group">
            <span class="text-medium-emphasis">FFT window {{ eventPayload.processing.fftWindowScale }}</span>
          </div>
        </div>

        <div class="pointcloud-grid" :class="gridClass">
          <section
            v-for="panelIndex in panelCount"
            :key="panelIndex"
            class="pointcloud-panel"
          >
            <PointcloudPlot
              :plot-type="panelTypes[panelIndex - 1]"
              :layout-mode="layoutMode"
              :hits="eventPayload.hits"
              :mapping-pads="mappingPads"
              :selected-trace-ids="selectedTraceIds"
              :traces="panelTypes[panelIndex - 1] === 'traces' ? tracePayload.traces : []"
              :xy-range="xyRange"
              :projected-range="projectedRange"
              :camera="camera"
              @toggle-traces="toggleTraceIds"
              @update-xy-range="updateXYRange"
              @update-projected-range="updateProjectedRange"
              @update-camera="updateCamera"
            />
          </section>
        </div>
      </v-card-text>
    </v-card>

    <v-card v-else class="empty-card mt-4" rounded="xl" variant="tonal">
      <v-card-text>
        <p class="page-kicker">No pointcloud event selected</p>
        <h2>Choose a run and event, then load the phase-1 data.</h2>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";

import { getMappingPads, getPointcloudEvent, getPointcloudTraces } from "../api";
import PointcloudPlot from "../components/PointcloudPlot.vue";
import { useShellStore } from "../stores/shell";

const shell = useShellStore();

const loading = ref(false);
const error = ref("");
const eventPayload = ref(null);
const tracePayload = ref({ traces: [] });
const mappingPads = ref([]);
const selectedRun = ref(null);
const selectedEventId = ref(null);
const selectedTraceIds = ref([]);
const layoutMode = ref("1x1");
const panelTypes = ref(["hits-3d-amplitude", "pads-z", "hits-2d-amplitude", "traces"]);
const xyRange = ref(null);
const projectedRange = ref(null);
const camera = ref(null);

const layoutOptions = [
  { title: "1x1", value: "1x1" },
  { title: "1x2", value: "1x2" },
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

const runOptions = computed(() =>
  (shell.state.bootstrap?.pointcloudRuns || []).map((run) => ({
    title: `Run ${run}`,
    value: Number(run),
  })),
);

const pointcloudEventRanges = computed(() => shell.state.bootstrap?.pointcloudEventRanges || {});

const eventRangeText = computed(() => {
  if (selectedRun.value === null) {
    return "No run selected";
  }
  const range = pointcloudEventRanges.value[String(selectedRun.value)];
  if (!range) {
    return "No pointcloud file for this run";
  }
  return `Available ${range.min} to ${range.max}`;
});

const panelCount = computed(() => (layoutMode.value === "1x1" ? 1 : layoutMode.value === "1x2" ? 2 : 4));
const gridClass = computed(() => ({
  "pointcloud-grid--single": layoutMode.value === "1x1",
  "pointcloud-grid--double": layoutMode.value === "1x2",
  "pointcloud-grid--quad": layoutMode.value === "2x2",
}));

function setRun(run) {
  selectedRun.value = run === null || run === "" ? null : Number(run);
  selectedEventId.value = defaultEventIdForRun(selectedRun.value);
}

function setEventId(value) {
  selectedEventId.value = value === null || value === "" ? null : Number(value);
}

function setLayoutMode(value) {
  layoutMode.value = value;
}

function setPanelType(index, value) {
  panelTypes.value[index] = value;
}

function defaultEventIdForRun(run) {
  if (run === null) {
    return null;
  }
  const range = pointcloudEventRanges.value[String(run)];
  return range ? Number(range.min) : null;
}

async function loadEvent() {
  if (selectedRun.value === null || selectedEventId.value === null) {
    error.value = "Select a run and event id before loading.";
    return;
  }
  loading.value = true;
  error.value = "";
  selectedTraceIds.value = [];
  tracePayload.value = { traces: [] };
  xyRange.value = null;
  projectedRange.value = null;
  camera.value = null;
  try {
    eventPayload.value = await getPointcloudEvent(selectedRun.value, selectedEventId.value);
  } catch (err) {
    eventPayload.value = null;
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    loading.value = false;
  }
}

async function loadSelectedTraces() {
  if (!eventPayload.value || !selectedTraceIds.value.length) {
    tracePayload.value = { traces: [] };
    return;
  }
  try {
    tracePayload.value = await getPointcloudTraces(
      eventPayload.value.run,
      eventPayload.value.eventId,
      selectedTraceIds.value,
    );
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err);
  }
}

function toggleTraceIds(traceIds) {
  const next = [...selectedTraceIds.value];
  for (const traceId of traceIds || []) {
    const numeric = Number(traceId);
    const existingIndex = next.indexOf(numeric);
    if (existingIndex >= 0) {
      next.splice(existingIndex, 1);
    } else {
      next.push(numeric);
    }
  }
  selectedTraceIds.value = next.slice(-8);
}

function clearSelection() {
  selectedTraceIds.value = [];
}

function sameRange(nextRange, currentRange) {
  if (nextRange === currentRange) {
    return true;
  }
  if (!nextRange || !currentRange) {
    return false;
  }
  return nextRange.x?.[0] === currentRange.x?.[0]
    && nextRange.x?.[1] === currentRange.x?.[1]
    && nextRange.y?.[0] === currentRange.y?.[0]
    && nextRange.y?.[1] === currentRange.y?.[1];
}

function updateXYRange(range) {
  if (range === null && xyRange.value === null) {
    return;
  }
  if (sameRange(range, xyRange.value)) {
    return;
  }
  xyRange.value = range;
}

function updateProjectedRange(range) {
  if (range === null && projectedRange.value === null) {
    return;
  }
  if (sameRange(range, projectedRange.value)) {
    return;
  }
  projectedRange.value = range;
}

function updateCamera(nextCamera) {
  camera.value = nextCamera;
}

function nextEvent() {
  if (!eventPayload.value) {
    return;
  }
  const range = eventPayload.value.eventIdRange;
  if (eventPayload.value.eventId < range.max) {
    selectedEventId.value = eventPayload.value.eventId + 1;
    void loadEvent();
  }
}

function previousEvent() {
  if (!eventPayload.value) {
    return;
  }
  const range = eventPayload.value.eventIdRange;
  if (eventPayload.value.eventId > range.min) {
    selectedEventId.value = eventPayload.value.eventId - 1;
    void loadEvent();
  }
}

function onKeydown(event) {
  const tagName = event.target?.tagName?.toLowerCase();
  if (
    tagName === "input"
    || tagName === "textarea"
    || tagName === "select"
    || event.target?.isContentEditable
  ) {
    return;
  }
  if (loading.value) {
    return;
  }
  const key = event.key.toLowerCase();
  if (key === "j") {
    event.preventDefault();
    nextEvent();
    return;
  }
  if (key === "k") {
    event.preventDefault();
    previousEvent();
  }
}

watch(
  () => selectedTraceIds.value.slice(),
  () => {
    void loadSelectedTraces();
  },
);

onMounted(async () => {
  window.addEventListener("keydown", onKeydown);
  await shell.init();
  mappingPads.value = await getMappingPads();
  selectedRun.value = shell.state.bootstrap?.pointcloudRuns?.[0] ?? null;
  selectedEventId.value = defaultEventIdForRun(selectedRun.value);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKeydown);
});
</script>
