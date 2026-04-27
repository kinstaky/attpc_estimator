<template>
  <div
    class="pointcloud-label-route"
    :class="{ 'pointcloud-label-route--select': isSelectingLabel }"
  >
    <LabelSummaryPanel
      title="Pointcloud"
      kicker="Line count"
      :items="summaryItems"
      side="left"
      :active="isSelectingLabel"
      show-review-all
      @review-all="openReview(null)"
      @review-item="openReview($event)"
    />

    <v-container class="label-workbench" fluid>
      <div class="page-header">
        <div>
          <p class="page-kicker">Label</p>
          <h1>Pointcloud workspace</h1>
          <p class="page-copy">
            Label each pointcloud event by the identified number of lines.
          </p>
        </div>

        <div class="page-header-actions">
          <v-chip prepend-icon="mdi-cube-outline" size="large" variant="tonal">
            Run {{ shell.state.selectedRun ?? "None" }}
          </v-chip>
          <v-chip prepend-icon="mdi-chart-bubble" size="large" variant="tonal">
            {{ visualModeTitle }}
          </v-chip>
          <div class="pointcloud-label-suggested">
            <span class="page-kicker">Suggested</span>
            <strong>{{ store.state.currentEvent?.suggestedLabel ?? "-" }}</strong>
          </div>
        </div>
      </div>

      <v-alert
        v-if="store.state.error"
        class="mb-4"
        color="error"
        icon="mdi-alert-circle-outline"
        rounded="xl"
        variant="tonal"
      >
        {{ store.state.error }}
      </v-alert>

      <v-alert
        v-else-if="store.state.statusMessage"
        class="mb-4"
        color="secondary"
        icon="mdi-information-outline"
        rounded="xl"
        variant="tonal"
      >
        {{ store.state.statusMessage }}
      </v-alert>

      <v-card class="trace-stage-card" :class="{ 'trace-stage-card--dimmed': isSelectingLabel }" rounded="xl">
        <template v-if="store.state.currentEvent">
          <v-card-title class="trace-stage-title">
            <div>
              <p class="page-kicker">
                Event {{ store.state.currentEvent.eventId }}
              </p>
              <h2>{{ store.currentLabelText.value }}</h2>
            </div>
            <div class="trace-stage-hints">
              <span>H / ← then 0-6 to label.</span>
              <span>Space uses merged RANSAC count. F switches layout.</span>
            </div>
          </v-card-title>

          <v-card-text>
            <div class="trace-action-toolbar">
              <div class="trace-action-group">
                <v-btn color="primary" variant="tonal" @click="store.setMode('await_line_count')">
                  Label · H / ←
                </v-btn>
                <v-btn color="primary" variant="text" @click="store.submitSuggestedLabel">
                  Use suggested · Space
                </v-btn>
                <v-btn variant="text" @click="store.toggleVisualMode">
                  Switch view · F
                </v-btn>
              </div>

              <div class="trace-action-group">
                <v-btn variant="text" @click="store.navigate(-1)">Previous</v-btn>
                <v-btn variant="text" @click="store.navigate(1)">Next</v-btn>
              </div>
            </div>

            <div v-if="isSelectingLabel" class="trace-choice-grid">
              <v-btn
                v-for="bucket in labelBuckets"
                :key="bucket"
                color="primary"
                variant="outlined"
                @click="store.submitLabel(bucket)"
              >
                {{ bucket }}
              </v-btn>
            </div>

            <div class="pointcloud-grid mt-4" :class="gridClass">
              <section class="pointcloud-panel">
                <PointcloudPlot
                  :plot-type="leftPlotType"
                  :layout-mode="plotLayoutMode"
                  :hits="store.state.currentEvent.hits"
                  :mapping-pads="[]"
                  :selected-trace-ids="[]"
                  :traces="[]"
                  :camera="camera"
                  @update-camera="updateCamera"
                />
              </section>

              <section v-if="secondPlotType" class="pointcloud-panel">
                <PointcloudPlot
                  :plot-type="secondPlotType"
                  :layout-mode="plotLayoutMode"
                  :hits="store.state.currentEvent.hits"
                  :mapping-pads="[]"
                  :selected-trace-ids="[]"
                  :traces="[]"
                />
              </section>

              <section v-if="thirdPlotType" class="pointcloud-panel">
                <PointcloudPlot
                  :plot-type="thirdPlotType"
                  :layout-mode="plotLayoutMode"
                  :hits="store.state.currentEvent.hits"
                  :mapping-pads="[]"
                  :selected-trace-ids="[]"
                  :traces="[]"
                />
              </section>

              <section v-if="showEmptyPanel" class="pointcloud-panel pointcloud-panel--empty">
                <div class="pointcloud-panel-empty-copy">
                  <span class="page-kicker">Reserved</span>
                  <strong>Panel 4</strong>
                </div>
              </section>
            </div>

            <v-row class="mt-2" dense>
              <v-col cols="12" md="4">
                <v-card rounded="xl" variant="tonal">
                  <v-card-text>
                    <p class="page-kicker">Input mode</p>
                    <strong>{{ inputModeText }}</strong>
                  </v-card-text>
                </v-card>
              </v-col>
              <v-col cols="12" md="4">
                <v-card rounded="xl" variant="tonal">
                  <v-card-text>
                    <p class="page-kicker">Merged line count</p>
                    <strong>{{ store.state.currentEvent.mergedLineCount }}</strong>
                  </v-card-text>
                </v-card>
              </v-col>
              <v-col cols="12" md="4">
                <v-card rounded="xl" variant="tonal">
                  <v-card-text>
                    <p class="page-kicker">Suggested label</p>
                    <strong class="pointcloud-label-suggested-count">{{ store.state.currentEvent.suggestedLabel }}</strong>
                  </v-card-text>
                </v-card>
              </v-col>
            </v-row>
          </v-card-text>
        </template>

        <template v-else>
          <v-card-text class="empty-state">
            <v-progress-circular
              v-if="store.state.loading"
              color="primary"
              indeterminate
            />
            <template v-else>
              <p class="page-kicker">Label</p>
              <h2>No pointcloud event loaded</h2>
            </template>
          </v-card-text>
        </template>
      </v-card>
    </v-container>
  </div>
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

import LabelSummaryPanel from "../components/LabelSummaryPanel.vue";
import PointcloudPlot from "../components/PointcloudPlot.vue";
import { usePointcloudLabelStore } from "../stores/pointcloud_label";
import { useShellStore } from "../stores/shell";

const route = useRoute();
const router = useRouter();
const shell = useShellStore();
const store = usePointcloudLabelStore();

const labelBuckets = ["0", "1", "2", "3", "4", "5", "6+"];
const isActive = ref(false);
const camera = ref(null);
let keydownAttached = false;

const pointcloudRuns = computed(() => shell.state.bootstrap?.pointcloudRuns || []);

const summaryItems = computed(() =>
  (shell.state.bootstrap?.pointcloudSummary || []).map((item) => ({
    key: `pointcloud-${item.bucket}`,
    title: item.title,
    count: item.count,
    value: item.bucket,
  })),
);

const isSelectingLabel = computed(() => store.state.mode === "await_line_count");

const inputModeText = computed(() => (
  isSelectingLabel.value ? "Waiting for line-count selection" : "Browse"
));

const visualModeTitle = computed(() => (
  store.state.visualMode === "basic"
    ? "Basic"
    : "Detailed"
));

const gridClass = computed(() => ({
  "pointcloud-grid--single": store.state.visualMode === "basic",
  "pointcloud-grid--quad": store.state.visualMode !== "basic",
}));

const leftPlotType = computed(() => "hits-3d-amplitude");
const secondPlotType = computed(() => (
  store.state.visualMode === "basic" ? null : "hits-3d-label"
));
const thirdPlotType = computed(() => (
  store.state.visualMode === "basic" ? null : "hits-2d-pca-amplitude"
));
const showEmptyPanel = computed(() => store.state.visualMode !== "basic");
const plotLayoutMode = computed(() => (
  store.state.visualMode === "basic" ? "1x1" : "2x2"
));

function updateCamera(nextCamera) {
  camera.value = nextCamera;
}

function resolveRun() {
  if (pointcloudRuns.value.includes(shell.state.selectedRun)) {
    return shell.state.selectedRun;
  }
  const fallbackRun = pointcloudRuns.value[0] ?? null;
  if (fallbackRun !== null) {
    shell.setSelectedRun(fallbackRun);
  }
  return fallbackRun;
}

async function ensureSession() {
  const run = resolveRun();
  if (run === null || run === undefined) {
    return;
  }
  try {
    const session = shell.state.bootstrap?.session;
    const reviewRequested = route.query.review === "1";
    const reviewLabel = typeof route.query.label === "string" ? route.query.label : null;
    if (reviewRequested) {
      if (
        session?.mode === "pointcloud_label_review"
        && session.run === run
        && (session.label ?? null) === reviewLabel
      ) {
        await store.restoreCurrentSession();
        return;
      }
      await store.enterReviewMode(run, { label: reviewLabel });
      return;
    }
    if (
      session?.mode === "pointcloud_label"
      && session.run === run
    ) {
      await store.restoreCurrentSession();
      return;
    }
    await store.enterLabelMode(run);
  } catch {
    // Store state already carries the error.
  }
}

function openReview(label) {
  router.push({
    name: "label-pointcloud",
    query: {
      review: "1",
      label: label || undefined,
    },
  });
}

function shouldIgnoreKey(event) {
  const tagName = event.target?.tagName?.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

async function onKeydown(event) {
  if (shouldIgnoreKey(event) || store.state.loading) {
    return;
  }
  const key = event.key === " " ? "space" : event.key.toLowerCase();

  if (key === "q" || key === "escape") {
    event.preventDefault();
    if (isSelectingLabel.value) {
      store.cancelSelectionMode();
      return;
    }
    if (store.state.isReviewMode) {
      await store.exitReviewMode();
      await router.replace({ name: "label-pointcloud" });
      return;
    }
    router.push({ name: "home" });
    return;
  }

  if (isSelectingLabel.value) {
    if (["0", "1", "2", "3", "4", "5"].includes(key)) {
      event.preventDefault();
      await store.submitLabel(key);
    } else if (key === "6") {
      event.preventDefault();
      await store.submitLabel("6+");
    }
    return;
  }

  if (key === "f") {
    event.preventDefault();
    store.toggleVisualMode();
    return;
  }
  if (key === "arrowleft" || key === "h") {
    event.preventDefault();
    store.setMode("await_line_count");
    return;
  }
  if (key === "arrowup" || key === "k") {
    event.preventDefault();
    await store.navigate(-1);
    return;
  }
  if (key === "arrowdown" || key === "j") {
    event.preventDefault();
    await store.navigate(1);
    return;
  }
  if (key === "space") {
    event.preventDefault();
    await store.submitSuggestedLabel();
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
  camera.value = null;
  if (
    !store.state.currentEvent
    || store.state.activeRun !== resolveRun()
  ) {
    void ensureSession();
  }
});

onDeactivated(() => {
  isActive.value = false;
  detachKeydownListener();
});

onBeforeUnmount(() => {
  detachKeydownListener();
});

watch(
  () => shell.state.selectedRun,
  () => {
    if (!isActive.value) {
      return;
    }
    camera.value = null;
    void ensureSession();
  },
);

watch(
  () => route.query,
  () => {
    if (!isActive.value) {
      return;
    }
    camera.value = null;
    void ensureSession();
  },
  { deep: true },
);
</script>
