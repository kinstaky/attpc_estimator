<template>
  <div
    class="label-route"
    :class="{
      'label-route--select-normal': isSelectingNormal,
      'label-route--select-strange': isSelectingStrange,
    }"
  >
    <LabelSummaryPanel
      title="Normal"
      kicker="Peak count"
      :items="normalItems"
      side="left"
      :active="isSelectingNormal"
      show-review-all
      @review-all="openReview('normal', null)"
      @review-item="openReview('normal', $event)"
    />

    <v-container class="label-workbench" fluid>
      <div class="page-header">
        <div>
          <p class="page-kicker">Label</p>
          <h1>Trace workspace</h1>
          <p class="page-copy">
            Use the keyboard or the action buttons below to label the current trace.
          </p>
        </div>

        <div class="page-header-actions">
          <v-chip prepend-icon="mdi-waveform" size="large" variant="tonal">
            Run {{ shell.state.selectedRun ?? "None" }}
          </v-chip>
          <v-btn-toggle
            color="primary"
            mandatory
            :model-value="labelStore.state.visualMode"
            @update:model-value="labelStore.setVisualMode"
          >
            <v-btn value="raw">Raw</v-btn>
            <v-btn value="cdf">CDF</v-btn>
            <v-btn value="curvature">Curvature</v-btn>
          </v-btn-toggle>
        </div>
      </div>

      <v-alert
        v-if="labelStore.state.error"
        class="mb-4"
        color="error"
        icon="mdi-alert-circle-outline"
        rounded="xl"
        variant="tonal"
      >
        {{ labelStore.state.error }}
      </v-alert>

      <v-alert
        v-else-if="labelStore.state.statusMessage"
        class="mb-4"
        color="secondary"
        icon="mdi-information-outline"
        rounded="xl"
        variant="tonal"
      >
        {{ labelStore.state.statusMessage }}
      </v-alert>

      <v-card class="trace-stage-card" :class="{ 'trace-stage-card--dimmed': isSelectionMode }" rounded="xl">
        <template v-if="labelStore.state.currentTrace">
          <v-card-title class="trace-stage-title">
            <div>
              <p class="page-kicker">
                Event {{ labelStore.state.currentTrace.eventId }} · Trace {{ labelStore.state.currentTrace.traceId }}
              </p>
              <h2>{{ labelStore.currentLabelText.value }}</h2>
            </div>
            <div class="trace-stage-hints">
              <span>Space from Home enters label mode.</span>
              <span>Q / Esc returns Home.</span>
            </div>
          </v-card-title>

          <v-card-text>
            <TracePlot
              :trace="labelStore.state.currentTrace"
              :visual-mode="labelStore.state.visualMode"
            />

            <div class="trace-action-toolbar">
              <div class="trace-action-group">
                <v-btn
                  color="primary"
                  variant="tonal"
                  @click="labelStore.setMode('await_normal_peak')"
                >
                  Normal · H / ←
                </v-btn>
                <v-btn
                  color="secondary"
                  variant="tonal"
                  :disabled="!strangeLabels.length"
                  @click="labelStore.setMode('await_strange_choice')"
                >
                  Strange · L / →
                </v-btn>
              </div>

              <div class="trace-action-group">
                <v-btn variant="text" @click="labelStore.navigate(-1)">Previous</v-btn>
                <v-btn variant="text" @click="labelStore.navigate(1)">Next</v-btn>
              </div>
            </div>

            <div v-if="labelStore.state.mode === 'await_normal_peak'" class="trace-choice-grid">
              <v-btn
                v-for="bucket in normalBuckets"
                :key="bucket"
                color="primary"
                variant="outlined"
                @click="labelStore.submitNormal(bucket)"
              >
                {{ bucket }}
              </v-btn>
            </div>

            <div
              v-else-if="labelStore.state.mode === 'await_strange_choice'"
              class="trace-choice-grid"
            >
              <v-btn
                v-for="item in strangeLabels"
                :key="item.name"
                color="secondary"
                variant="outlined"
                @click="labelStore.submitStrange(item.name)"
              >
                {{ item.name }}
              </v-btn>
            </div>

            <v-row class="mt-2" dense>
              <v-col cols="12" md="4">
                <v-card rounded="xl" variant="tonal">
                  <v-card-text>
                    <p class="page-kicker">Trace key</p>
                    <strong>{{ labelStore.state.currentTrace.eventId }} / {{ labelStore.state.currentTrace.traceId }}</strong>
                  </v-card-text>
                </v-card>
              </v-col>
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
                    <p class="page-kicker">Current label</p>
                    <strong>{{ labelStore.currentLabelText.value }}</strong>
                  </v-card-text>
                </v-card>
              </v-col>
            </v-row>
          </v-card-text>
        </template>

        <template v-else>
          <v-card-text class="empty-state">
            <v-progress-circular
              v-if="labelStore.state.loading"
              color="primary"
              indeterminate
            />
            <template v-else>
              <p class="page-kicker">Label</p>
              <h2>No trace loaded</h2>
            </template>
          </v-card-text>
        </template>
      </v-card>
    </v-container>

    <LabelSummaryPanel
      title="Strange"
      kicker="Special cases"
      :items="strangeItems"
      side="right"
      :active="isSelectingStrange"
      allow-add
      allow-delete
      show-review-all
      @add="labelStore.openAddDialog"
      @review-all="openReview('strange', null)"
      @review-item="openReview('strange', $event)"
      @delete-item="labelStore.openDeleteDialog"
    />

    <AddStrangeLabelDialog
      v-model="labelStore.state.addDialogOpen"
      :save-label="saveStrangeLabel"
    />
    <DeleteStrangeLabelDialog
      v-model="deleteDialogOpen"
      :delete-label="deleteStrangeLabel"
      :label="labelStore.state.deleteDialogLabel"
    />
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
import { useRouter } from "vue-router";

import AddStrangeLabelDialog from "../components/AddStrangeLabelDialog.vue";
import DeleteStrangeLabelDialog from "../components/DeleteStrangeLabelDialog.vue";
import LabelSummaryPanel from "../components/LabelSummaryPanel.vue";
import TracePlot from "../components/TracePlot.vue";
import { useLabelStore } from "../stores/label";
import { useShellStore } from "../stores/shell";

const router = useRouter();
const shell = useShellStore();
const labelStore = useLabelStore();

const normalBuckets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

const normalItems = computed(() =>
  (shell.state.bootstrap?.normalSummary || []).map((item) => ({
    key: `normal-${item.bucket}`,
    title: item.title,
    count: item.count,
    value: item.bucket >= 4 ? "4+" : String(item.bucket),
  })),
);

const strangeItems = computed(() =>
  (shell.state.bootstrap?.strangeSummary || []).map((item) => ({
    key: `strange-${item.name}`,
    title: item.name,
    count: item.count,
    reviewValue: item.name,
    deleteValue: item,
  })),
);

const strangeLabels = computed(() => shell.state.bootstrap?.strangeLabels || []);
const isActive = ref(false);
let keydownAttached = false;

const deleteDialogOpen = computed({
  get: () => Boolean(labelStore.state.deleteDialogLabel),
  set: (value) => {
    if (!value) {
      labelStore.state.deleteDialogLabel = null;
    }
  },
});

const inputModeText = computed(() => {
  if (labelStore.state.mode === "await_normal_peak") {
    return "Waiting for 0-9 normal peak selection";
  }
  if (labelStore.state.mode === "await_strange_choice") {
    return "Waiting for strange shortcut selection";
  }
  return "Browse";
});

const isSelectingNormal = computed(
  () => labelStore.state.mode === "await_normal_peak",
);

const isSelectingStrange = computed(
  () => labelStore.state.mode === "await_strange_choice",
);

const isSelectionMode = computed(
  () => isSelectingNormal.value || isSelectingStrange.value,
);

async function ensureSession() {
  if (shell.state.selectedRun === null || shell.state.selectedRun === undefined) {
    return;
  }
  try {
    await labelStore.enterLabelMode(shell.state.selectedRun);
  } catch {
    // Store state already carries the error.
  }
}

async function saveStrangeLabel(payload) {
  await labelStore.addStrange(payload.name, payload.shortcutKey);
}

async function deleteStrangeLabel(name) {
  await labelStore.removeStrange(name);
  labelStore.state.deleteDialogLabel = null;
}

function openReview(family, label) {
  router.push({
    name: "review",
    query: {
      source: "label_set",
      run: shell.state.selectedRun ?? undefined,
      family,
      label: label || undefined,
    },
  });
}

function shouldIgnoreKey(event) {
  if (labelStore.state.addDialogOpen || labelStore.state.deleteDialogLabel) {
    return true;
  }
  const tagName = event.target?.tagName?.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

async function onKeydown(event) {
  if (shouldIgnoreKey(event) || labelStore.state.loading) {
    return;
  }

  const key = event.key === " " ? "space" : event.key.toLowerCase();

  if (key === "q" || key === "escape") {
    event.preventDefault();
    if (isSelectionMode.value) {
      labelStore.cancelSelectionMode();
      return;
    }
    router.push({ name: "home" });
    return;
  }

  if (labelStore.state.mode === "await_normal_peak") {
    if (/^[0-9]$/.test(key)) {
      event.preventDefault();
      await labelStore.submitNormal(Number(key));
    }
    return;
  }

  if (labelStore.state.mode === "await_strange_choice") {
    const strangeMatch = strangeLabels.value.find(
      (item) => item.shortcutKey?.toLowerCase() === key,
    );
    if (strangeMatch) {
      event.preventDefault();
      await labelStore.submitStrange(strangeMatch.name);
    }
    return;
  }

  if (labelStore.state.mode === "browse") {
    if (key === "f") {
      event.preventDefault();
      labelStore.toggleVisualMode();
      return;
    }
    if (key === "arrowleft" || key === "h") {
      event.preventDefault();
      labelStore.setMode("await_normal_peak");
      return;
    }
    if (key === "arrowright" || key === "l") {
      event.preventDefault();
      if (!strangeLabels.value.length) {
        labelStore.state.statusMessage = "Add a strange label before using strange mode.";
        return;
      }
      labelStore.setMode("await_strange_choice");
      return;
    }
    if (key === "arrowup" || key === "k") {
      event.preventDefault();
      await labelStore.navigate(-1);
      return;
    }
    if (key === "arrowdown" || key === "j") {
      event.preventDefault();
      await labelStore.navigate(1);
    }
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
  if (
    !labelStore.state.currentTrace
    || labelStore.state.activeRun !== shell.state.selectedRun
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
    void ensureSession();
  },
);
</script>
