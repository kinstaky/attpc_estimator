<template>
  <v-container class="page-container" fluid>
    <div class="page-header">
      <div>
        <p class="page-kicker">Review</p>
        <h1>Trace review</h1>
        <p class="page-copy">
          Review labeled traces, filter-file selections, or direct HDF5 event/trace positions.
        </p>
      </div>
    </div>

    <v-card class="control-card" rounded="xl">
      <v-card-text>
        <v-row dense>
          <v-col cols="12" md="3">
            <v-select
              :items="reviewSourceOptions"
              item-title="title"
              item-value="value"
              label="Review source"
              :model-value="review.state.source"
              variant="outlined"
              @update:model-value="review.setSource"
            />
          </v-col>

          <template v-if="review.state.source === 'label_set'">
            <v-col cols="12" md="3">
              <v-select
                :items="runOptions"
                item-title="title"
                item-value="value"
                label="Run"
                :model-value="review.state.run"
                variant="outlined"
                @update:model-value="review.setRun"
              />
            </v-col>
            <v-col cols="12" md="3">
              <v-select
                :items="familyOptions"
                item-title="title"
                item-value="value"
                label="Family"
                :model-value="review.state.family"
                variant="outlined"
                @update:model-value="review.setFamily"
              />
            </v-col>
            <v-col cols="12" md="3">
              <v-select
                :items="labelOptions"
                item-title="title"
                item-value="value"
                label="Label filter"
                :model-value="review.state.label"
                variant="outlined"
                @update:model-value="review.setLabel"
              />
            </v-col>
          </template>
          <template v-else-if="review.state.source === 'event_trace'">
            <v-col v-if="showDetectorSelect" cols="12" md="3">
              <v-select
                :items="detectorOptions"
                item-title="title"
                item-value="value"
                label="Detector"
                :model-value="review.state.detector"
                variant="outlined"
                @update:model-value="review.setDetector"
              />
            </v-col>
            <v-col cols="12" md="3">
              <v-select
                :items="runOptions"
                item-title="title"
                item-value="value"
                label="Run"
                :model-value="review.state.run"
                variant="outlined"
                @update:model-value="review.setRun"
              />
            </v-col>
            <v-col cols="12" md="3">
              <v-text-field
                label="Event id"
                type="number"
                :model-value="review.state.eventId"
                variant="outlined"
                :hint="eventRangeText"
                persistent-hint
                @update:model-value="review.setEventId"
              />
            </v-col>
            <v-col v-if="showTraceIdInput" cols="12" md="3">
              <v-text-field
                label="Trace id"
                type="number"
                :model-value="review.state.traceId"
                variant="outlined"
                :hint="traceRangeText"
                persistent-hint
                @update:model-value="review.setTraceId"
              />
            </v-col>
            <v-col v-else-if="showSiSelector" cols="12" md="3">
              <v-select
                :items="siSideOptions"
                item-title="title"
                item-value="value"
                label="Silicon side"
                :model-value="review.state.siSide"
                variant="outlined"
                @update:model-value="review.setSiSide"
              />
            </v-col>
            <v-col v-else-if="showGaggSelector" cols="12" md="3">
              <v-select
                :items="gaggLayerOptions"
                item-title="title"
                item-value="value"
                label="GAGG layer"
                :model-value="review.state.gaggLayer"
                variant="outlined"
                @update:model-value="review.setGaggLayer"
              />
            </v-col>
            <v-col v-if="showSiSelector" cols="12" md="3">
              <v-text-field
                label="Silicon index"
                type="number"
                :model-value="review.state.siIndex"
                variant="outlined"
                :hint="selectorHintText"
                persistent-hint
                @update:model-value="review.setSiIndex"
              />
            </v-col>
            <v-col v-else-if="showGaggSelector" cols="12" md="3">
              <v-text-field
                label="GAGG index"
                type="number"
                :model-value="review.state.gaggIndex"
                variant="outlined"
                :hint="selectorHintText"
                persistent-hint
                @update:model-value="review.setGaggIndex"
              />
            </v-col>
          </template>

          <v-col v-else cols="12" md="6">
            <v-select
              :items="filterFileOptions"
              item-title="title"
              item-value="value"
              label="Filter file"
              :model-value="review.state.filterFile"
              variant="outlined"
              @update:model-value="review.setFilterFile"
            />
          </v-col>

          <v-col cols="12" md="3">
            <v-select
              :items="visualModeOptions"
              item-title="title"
              item-value="value"
              label="Plot mode"
              :model-value="review.state.visualMode"
              variant="outlined"
              @update:model-value="review.setVisualMode"
            />
          </v-col>
          <v-col cols="12" md="3" class="d-flex align-center">
            <v-btn color="primary" :loading="review.state.loading" @click="submitReview">
              Load review set
            </v-btn>
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <v-card
      v-if="review.state.source === 'event_trace'"
      class="control-card mt-4"
      rounded="xl"
    >
      <v-card-text>
        <v-row dense>
          <v-col cols="12" md="3">
            <v-select
              :items="filterItemOptions"
              item-title="title"
              item-value="value"
              label="Filter item"
              :model-value="review.state.filterItem"
              variant="outlined"
              @update:model-value="review.setFilterItem"
            />
          </v-col>
          <v-col v-if="review.state.filterItem === 'max'" cols="12" md="3">
            <v-text-field
              label="Max threshold"
              type="number"
              :model-value="review.state.filterValue"
              variant="outlined"
              @update:model-value="review.setFilterValue"
            />
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <v-alert
      v-if="review.state.error"
      class="mt-4"
      color="error"
      icon="mdi-alert-circle-outline"
      rounded="xl"
      variant="tonal"
    >
      {{ review.state.error }}
    </v-alert>

    <v-alert
      v-else-if="review.state.statusMessage"
      class="mt-4"
      color="secondary"
      icon="mdi-information-outline"
      rounded="xl"
      variant="tonal"
    >
      {{ review.state.statusMessage }}
    </v-alert>

    <v-card v-if="review.state.currentTrace" class="trace-stage-card mt-4" rounded="xl">
      <v-card-title class="trace-stage-title">
        <div>
          <p class="page-kicker">
            Run {{ review.state.currentTrace.run }} · Event {{ review.state.currentTrace.eventId }} · Trace {{ review.state.currentTrace.traceId }}
          </p>
          <h2>{{ currentLabelText(review.state.currentTrace.currentLabel) }}</h2>
        </div>
        <div class="trace-stage-hints">
          <span>{{ navigationHintText }}</span>
          <span>Press F to cycle the plot view.</span>
        </div>
      </v-card-title>

      <v-card-text>
        <TracePlot
          :trace="review.state.currentTrace"
          :visual-mode="review.state.visualMode"
        />

        <div class="trace-action-toolbar">
          <div class="trace-action-group">
            <v-btn variant="text" @click="review.previousReviewTrace()">Previous</v-btn>
            <v-btn variant="text" @click="review.nextReviewTrace()">Next</v-btn>
          </div>
        </div>

        <v-row class="mt-2" dense>
          <v-col cols="12" md="3">
            <v-card rounded="xl" variant="tonal">
              <v-card-text>
                <p class="page-kicker">Source</p>
                <strong>{{ reviewSourceLabel }}</strong>
              </v-card-text>
            </v-card>
          </v-col>
          <v-col cols="12" md="3">
            <v-card rounded="xl" variant="tonal">
              <v-card-text>
                <p class="page-kicker">Trace key</p>
                <strong>{{ review.state.currentTrace.run }} / {{ review.state.currentTrace.eventId }} / {{ review.state.currentTrace.traceId }}</strong>
              </v-card-text>
            </v-card>
          </v-col>
          <v-col cols="12" md="3" v-if="showDetectorMetaCard">
            <v-card rounded="xl" variant="tonal">
              <v-card-text>
                <p class="page-kicker">Detector meta</p>
                <strong v-for="line in detectorMetaLines" :key="line" class="meta-line">
                  {{ line }}
                </strong>
              </v-card-text>
            </v-card>
          </v-col>
          <v-col cols="12" md="3" v-if="review.state.currentTrace.reviewProgress">
            <v-card rounded="xl" variant="tonal">
              <v-card-text>
                <p class="page-kicker">Review progress</p>
                <strong>
                  {{ review.state.currentTrace.reviewProgress.current }} / {{ review.state.currentTrace.reviewProgress.total }}
                </strong>
              </v-card-text>
            </v-card>
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <v-card v-else class="empty-card mt-4" rounded="xl" variant="tonal">
      <v-card-text>
        <p class="page-kicker">No trace selected</p>
        <h2>Choose a review source and load a trace set.</h2>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<script setup>
import {
  computed,
  onActivated,
  onBeforeUnmount,
  onDeactivated,
  ref,
  watch,
} from "vue";
import { useRoute, useRouter } from "vue-router";

import TracePlot from "../components/TracePlot.vue";
import { useReviewStore } from "../stores/review";
import { useShellStore } from "../stores/shell";

const route = useRoute();
const router = useRouter();
const shell = useShellStore();
const review = useReviewStore();
const isActive = ref(false);
let keydownAttached = false;

const reviewSourceOptions = [
  { title: "Labeled traces", value: "label_set" },
  { title: "Filter file", value: "filter_file" },
  { title: "Direct event/trace", value: "event_trace" },
];

const familyOptions = [
  { title: "Normal", value: "normal" },
  { title: "Strange", value: "strange" },
];

const visualModeOptions = [
  { title: "Raw", value: "raw" },
  { title: "CDF", value: "cdf" },
  { title: "Curvature", value: "curvature" },
  { title: "Peak", value: "peak" },
];

const filterItemOptions = [
  { title: "None", value: "none" },
  { title: "Max", value: "max" },
];

const detectorOptions = [
  { title: "ATTPC", value: "ATTPC" },
  { title: "IC", value: "IC" },
  { title: "Si", value: "SI" },
  { title: "GAGG", value: "GAGG" },
];

const siSideOptions = [
  { title: "Upstream front", value: "upstream_front" },
  { title: "Upstream back", value: "upstream_back" },
  { title: "Downstream front", value: "downstream_front" },
  { title: "Downstream back", value: "downstream_back" },
];

const gaggLayerOptions = [
  { title: "Layer 1", value: 1 },
  { title: "Layer 2", value: 2 },
];

const showDetectorSelect = computed(() => review.state.source === "event_trace");

const showTraceIdInput = computed(() =>
  review.state.source === "event_trace" && review.state.detector === "ATTPC",
);

const showSiSelector = computed(() =>
  review.state.source === "event_trace" && review.state.detector === "SI",
);

const showGaggSelector = computed(() =>
  review.state.source === "event_trace" && review.state.detector === "GAGG",
);

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

const labelOptions = computed(() => {
  if (review.state.family === "normal") {
    return [
      { title: "All normal labels", value: "" },
      { title: "0 peak", value: "0" },
      { title: "1 peak", value: "1" },
      { title: "2 peaks", value: "2" },
      { title: "3 peaks", value: "3" },
      { title: "4+ peaks", value: "4+" },
    ];
  }
  return [
    { title: "All strange labels", value: "" },
    ...((shell.state.bootstrap?.strangeLabels || []).map((item) => ({
      title: item.name,
      value: item.name,
    }))),
  ];
});

const reviewSourceLabel = computed(() => {
  if (review.state.source === "filter_file") {
    return review.state.filterFile || "Filter file";
  }
  if (review.state.source === "event_trace") {
    return `${review.state.detector} direct event/trace`;
  }
  if (!review.state.label) {
    return `Labeled ${review.state.family}`;
  }
  return `${review.state.family}: ${review.state.label}`;
});

const navigationHintText = computed(() =>
  review.state.source === "event_trace"
    ? "Use J / K or up/down for traces, H / L or left/right for events."
    : "Use J / K or arrows to navigate.",
);

const eventRangeText = computed(() => {
  const run = review.state.run;
  if (run === null || run === undefined) {
    return "Select a run to see the event range.";
  }
  const range = shell.state.bootstrap?.eventRanges?.[String(run)];
  if (!range) {
    return "Event range unavailable.";
  }
  return `Event range: ${range.min} .. ${range.max}`;
});

const traceRangeText = computed(() => {
  const trace = review.state.currentTrace;
  if (
    review.state.source !== "event_trace"
    || !trace
    || trace.eventTraceCount === null
    || review.state.eventId !== trace.eventId
  ) {
    return "Load an event to see the trace range.";
  }
  return `Trace range: 0 .. ${trace.eventTraceCount - 1}`;
});

const selectorHintText = computed(() => {
  const selector = review.state.currentTrace?.eventContext?.current?.selector;
  if (review.state.detector === "SI" && selector?.kind === "si") {
    const count = selector.sideCounts?.[review.state.siSide] ?? 0;
    return `Index range for this side: 0 .. ${Math.max(0, count - 1)}`;
  }
  if (review.state.detector === "GAGG" && selector?.kind === "gagg") {
    const count = review.state.gaggLayer === 1
      ? selector.layerCounts?.layer1 ?? 0
      : selector.layerCounts?.layer2 ?? 0;
    return `Index range for this layer: 0 .. ${Math.max(0, count - 1)}`;
  }
  return "Load an event to see the selector range.";
});

const detectorMeta = computed(() => review.state.currentTrace?.traceMetadata ?? null);

const showDetectorMetaCard = computed(() => detectorMeta.value !== null);

const detectorMetaLines = computed(() => {
  const metadata = detectorMeta.value;
  if (!metadata) {
    return [];
  }
  if (metadata.kind === "attpc") {
    return [`Pad ${metadata.padId}`];
  }
  if (metadata.kind === "si") {
    return [
      `Layer ${metadata.layer}`,
      `Side ${metadata.side}`,
      `Strip ${metadata.strip}`,
    ];
  }
  if (metadata.kind === "gagg") {
    return [`Layer ${metadata.layer}`, `Index ${metadata.index}`];
  }
  return [];
});

function normalizeDetector(detector) {
  const token = String(detector || "ATTPC").trim().toUpperCase();
  if (token === "IC" || token === "SI" || token === "GAGG") {
    return token;
  }
  return "ATTPC";
}

function currentLabelText(label) {
  if (!label) {
    return "Unlabeled";
  }
  if (label.family === "normal") {
    return `${label.label} peak${label.label === "1" ? "" : "s"}`;
  }
  return `Strange: ${label.label}`;
}

async function submitReview() {
  await router.replace({ name: "browse-trace", query: review.buildQuery() });
  try {
    await review.loadReviewSet();
    await router.replace({ name: "browse-trace", query: review.buildQuery() });
  } catch {
    // Store already captured the error.
  }
}

function hasReviewQuery(query) {
  return typeof query.source === "string";
}

function reviewQueryMatchesState(query) {
  const source = query.source === "filter_file"
    ? "filter_file"
    : query.source === "event_trace"
      ? "event_trace"
      : "label_set";
  if (review.state.source !== source) {
    return false;
  }
  if (source === "filter_file") {
    const filterFile = typeof query.filterFile === "string" ? query.filterFile : "";
    return review.state.filterFile === filterFile;
  }
  if (source === "event_trace") {
    const queryDetector = normalizeDetector(query.detector);
    const queryRun = query.run === undefined ? null : Number(query.run);
    const queryEventId = query.eventId === undefined ? null : Number(query.eventId);
    const queryTraceId =
      queryDetector === "IC"
        ? 0
        : query.traceId === undefined
          ? null
          : Number(query.traceId);
    const queryFilterItem = query.filterItem === "max" ? "max" : "none";
    const queryFilterValue =
      query.filterValue === undefined ? null : Number(query.filterValue);
    return (
      review.state.detector === queryDetector
      &&
      review.state.run === queryRun
      && review.state.eventId === queryEventId
      && review.state.traceId === queryTraceId
      && review.state.filterItem === queryFilterItem
      && review.state.filterValue === queryFilterValue
    );
  }
  const queryRun = query.run === undefined ? null : Number(query.run);
  const queryFamily = query.family === "strange" ? "strange" : "normal";
  const queryLabel = typeof query.label === "string" ? query.label : "";
  return (
    review.state.run === queryRun
    && review.state.family === queryFamily
    && review.state.label === queryLabel
  );
}

function reviewQueryMatchesSession(query) {
  const session = shell.state.bootstrap?.session;
  if (!session || session.mode !== "review") {
    return false;
  }
  const source = query.source === "filter_file"
    ? "filter_file"
    : query.source === "event_trace"
      ? "event_trace"
      : "label_set";
  if (session.source !== source) {
    return false;
  }
  if (source === "filter_file") {
    return session.filterFile === (typeof query.filterFile === "string" ? query.filterFile : null);
  }
  if (source === "event_trace") {
    const queryDetector = normalizeDetector(query.detector);
    const queryFilterItem = query.filterItem === "max" ? "max" : "none";
    const queryFilterValue =
      query.filterValue === undefined ? null : Number(query.filterValue);
    return (
      session.detector === queryDetector
      &&
      session.run === (query.run === undefined ? null : Number(query.run))
      && session.eventId === (query.eventId === undefined ? null : Number(query.eventId))
      && session.traceId === (
        queryDetector === "IC"
          ? 0
          : query.traceId === undefined
            ? null
            : Number(query.traceId)
      )
      && (session.filterItem ?? "none") === queryFilterItem
      && (session.filterValue ?? null) === queryFilterValue
    );
  }
  return (
    session.run === (query.run === undefined ? null : Number(query.run))
    && session.family === (query.family === "strange" ? "strange" : "normal")
    && session.label === (typeof query.label === "string" ? query.label : null)
  );
}

async function syncFromRoute(query) {
  if (!hasReviewQuery(query)) {
    return;
  }
  const shouldReload = !reviewQueryMatchesState(query) || !review.state.currentTrace;
  review.applyQuery(query);
  if (shouldReload && reviewQueryMatchesSession(query) && !review.state.currentTrace) {
    try {
      await review.restoreCurrentSession();
      return;
    } catch {
      // Store already captured the error.
    }
  }
  if (shouldReload) {
    try {
      await review.loadReviewSet();
    } catch {
      // Store already captured the error.
    }
  }
}

function shouldIgnoreKey(event) {
  const tagName = event.target?.tagName?.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

async function onKeydown(event) {
  if (
    shouldIgnoreKey(event)
    || review.state.loading
    || !review.state.currentTrace
  ) {
    return;
  }
  const key = event.key.toLowerCase();
  if (key === "f") {
    event.preventDefault();
    review.toggleVisualMode();
    return;
  }
  if (key === "arrowup" || key === "k") {
    event.preventDefault();
    await review.previousReviewTrace();
    return;
  }
  if (key === "arrowdown" || key === "j") {
    event.preventDefault();
    await review.nextReviewTrace();
    return;
  }
  if (
    review.state.source === "event_trace"
    && (key === "arrowleft" || key === "h")
  ) {
    event.preventDefault();
    await review.previousReviewEvent();
    return;
  }
  if (
    review.state.source === "event_trace"
    && (key === "arrowright" || key === "l")
  ) {
    event.preventDefault();
    await review.nextReviewEvent();
  }
}

function attachKeydownListener() {
  if (keydownAttached) {
    return;
  }
  window.addEventListener("keydown", onKeydown);
  keydownAttached = true;
}

function detachKeydownListener() {
  if (!keydownAttached) {
    return;
  }
  window.removeEventListener("keydown", onKeydown);
  keydownAttached = false;
}

onActivated(() => {
  isActive.value = true;
  attachKeydownListener();
  void syncFromRoute(route.query);
});

onDeactivated(() => {
  isActive.value = false;
  detachKeydownListener();
});

onBeforeUnmount(() => {
  detachKeydownListener();
});

watch(
  () => route.query,
  (query) => {
    if (!isActive.value || route.name !== "browse-trace") {
      return;
    }
    void syncFromRoute(query);
  },
  { deep: true },
);
</script>

<style scoped>
.meta-line {
  display: block;
}
</style>
