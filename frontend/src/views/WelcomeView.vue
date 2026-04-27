<template>
  <v-container class="page-container" fluid>
    <div class="page-header">
      <div>
        <p class="page-kicker">Home</p>
        <h1>Trace labeling overview</h1>
        <p class="page-copy">
          Review labeled-trace summaries, choose the active run, and jump into labeling or review.
        </p>
      </div>

      <v-select
        class="run-select"
        :items="runOptions"
        item-title="title"
        item-value="value"
        label="Active run"
        :model-value="shell.state.selectedRun"
        variant="outlined"
        @update:model-value="shell.setSelectedRun"
      />
    </div>

    <v-row class="mb-6" dense>
      <v-col cols="12" md="4">
        <v-card rounded="xl" variant="tonal">
          <v-card-text>
            <p class="page-kicker">Labeled traces</p>
            <div class="stat-number">{{ labeledCount }}</div>
          </v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="4">
        <v-card rounded="xl" variant="tonal">
          <v-card-text>
            <p class="page-kicker">Normal</p>
            <div class="stat-number">{{ normalTotal }}</div>
          </v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="4">
        <v-card rounded="xl" variant="tonal">
          <v-card-text>
            <p class="page-kicker">Strange</p>
            <div class="stat-number">{{ strangeTotal }}</div>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-row dense>
      <v-col cols="12" lg="6">
        <v-card class="dashboard-card" rounded="xl">
          <v-card-title class="dashboard-title">
            <div>
              <p class="page-kicker">Normal mix</p>
              <h2>Peak-count review</h2>
            </div>
            <v-btn
              color="primary"
              variant="tonal"
              :disabled="normalTotal === 0"
              @click="openReview('normal', null)"
            >
              Review all
            </v-btn>
          </v-card-title>
          <v-card-text class="dashboard-list">
            <button
              v-for="item in normalBreakdown"
              :key="item.filterValue"
              class="dashboard-item"
              :disabled="item.count === 0"
              @click="openReview('normal', item.filterValue)"
            >
              <div>
                <strong>{{ item.label }}</strong>
                <span>{{ item.count }} labeled</span>
              </div>
              <span>{{ formatPercentage(item.percentage) }}</span>
            </button>
          </v-card-text>
        </v-card>
      </v-col>

      <v-col cols="12" lg="6">
        <v-card class="dashboard-card" rounded="xl">
          <v-card-title class="dashboard-title">
            <div>
              <p class="page-kicker">Strange mix</p>
              <h2>Special-case review</h2>
            </div>
            <v-btn
              color="primary"
              variant="tonal"
              :disabled="strangeTotal === 0"
              @click="openReview('strange', null)"
            >
              Review all
            </v-btn>
          </v-card-title>
          <v-card-text class="dashboard-list">
            <button
              v-for="item in strangeBreakdown"
              :key="item.name"
              class="dashboard-item"
              :disabled="item.count === 0"
              @click="openReview('strange', item.name)"
            >
              <div>
                <strong>{{ item.name }}</strong>
                <span>{{ item.count }} labeled</span>
              </div>
              <span>{{ formatPercentage(item.percentage) }}</span>
            </button>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-row class="mt-2" dense>
      <v-col cols="12" md="6">
        <v-card rounded="xl" variant="text">
          <v-card-text>
            <p class="page-kicker">Trace source</p>
            <strong>{{ shell.state.bootstrap?.tracePath }}</strong>
          </v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="6">
        <v-card rounded="xl" variant="text">
          <v-card-text>
            <p class="page-kicker">Database</p>
            <strong>{{ shell.state.bootstrap?.databaseFile }}</strong>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted } from "vue";
import { useRouter } from "vue-router";

import { useShellStore } from "../stores/shell";

const router = useRouter();
const shell = useShellStore();

const runOptions = computed(() =>
  (shell.state.bootstrap?.runs || []).map((run) => ({
    title: `Run ${run}`,
    value: Number(run),
  })),
);

const normalTotal = computed(() =>
  (shell.state.bootstrap?.normalSummary || []).reduce(
    (total, item) => total + Number(item.count || 0),
    0,
  ),
);

const strangeTotal = computed(() =>
  (shell.state.bootstrap?.strangeSummary || []).reduce(
    (total, item) => total + Number(item.count || 0),
    0,
  ),
);

const labeledCount = computed(() => normalTotal.value + strangeTotal.value);

const normalBreakdown = computed(() => {
  const countsByBucket = new Map(
    (shell.state.bootstrap?.normalSummary || []).map((item) => [
      Number(item.bucket),
      Number(item.count || 0),
    ]),
  );
  const total = normalTotal.value;
  const grouped = [
    { label: "0 peak", filterValue: "0", count: countsByBucket.get(0) || 0 },
    { label: "1 peak", filterValue: "1", count: countsByBucket.get(1) || 0 },
    { label: "2 peaks", filterValue: "2", count: countsByBucket.get(2) || 0 },
    { label: "3 peaks", filterValue: "3", count: countsByBucket.get(3) || 0 },
    {
      label: "4+ peaks",
      filterValue: "4+",
      count: [4, 5, 6, 7, 8, 9].reduce(
        (sum, bucket) => sum + (countsByBucket.get(bucket) || 0),
        0,
      ),
    },
  ];
  return grouped.map((item) => ({
    ...item,
    percentage: total > 0 ? (item.count / total) * 100 : 0,
  }));
});

const strangeBreakdown = computed(() => {
  const total = strangeTotal.value;
  return (shell.state.bootstrap?.strangeSummary || []).map((item) => ({
    ...item,
    count: Number(item.count || 0),
    percentage: total > 0 ? (Number(item.count || 0) / total) * 100 : 0,
  }));
});

function formatPercentage(value) {
  const rounded = Math.round(value * 10) / 10;
  return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)}%`;
}

function openReview(family, label) {
  router.push({
    name: "browse-trace",
    query: {
      source: "label_set",
      run: shell.state.selectedRun ?? undefined,
      family,
      label: label || undefined,
    },
  });
}

function onKeydown(event) {
  const tagName = event.target?.tagName?.toLowerCase();
  if (tagName === "input" || tagName === "textarea" || tagName === "select") {
    return;
  }
  if (event.key === " ") {
    event.preventDefault();
    router.push({ name: "label" });
  }
}

onMounted(() => {
  window.addEventListener("keydown", onKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKeydown);
});
</script>
