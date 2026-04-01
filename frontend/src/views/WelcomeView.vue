<template>
  <section class="welcome-panel">
    <p class="eyebrow">Session Overview</p>
    <h1>Label traces</h1>

    <button class="primary-button welcome-start-button" @click="$emit('start')">Start</button>

    <div class="progress-card">
      <div>
        <p class="progress-kicker">Total labeled</p>
        <strong>{{ labeledCount }}</strong>
      </div>
      <div>
        <p class="progress-kicker">Normal</p>
        <strong>{{ normalTotal }}</strong>
      </div>
      <div>
        <p class="progress-kicker">Strange</p>
        <strong>{{ strangeTotal }}</strong>
      </div>
    </div>

    <div class="distribution-grid">
      <section class="distribution-card">
        <button
          type="button"
          class="distribution-header distribution-header-button"
          :disabled="normalTotal === 0"
          @click="emitStartReview('normal', null)"
        >
          <p class="progress-kicker">Normal Mix</p>
          <span class="distribution-note">{{ formatPercentage(normalFamilyPercentage) }} of labeled traces</span>
        </button>
        <div class="distribution-list">
          <button
            v-for="item in normalBreakdown"
            :key="item.filterValue"
            type="button"
            class="distribution-row distribution-row-button"
            :disabled="item.count === 0"
            @click="emitStartReview('normal', item.filterValue)"
          >
            <div>
              <strong class="distribution-label">{{ item.label }}</strong>
              <p class="distribution-meta">{{ item.count }} labeled</p>
            </div>
            <strong class="distribution-value">{{ formatPercentage(item.percentage) }}</strong>
          </button>
        </div>
      </section>

      <section class="distribution-card">
        <button
          type="button"
          class="distribution-header distribution-header-button"
          :disabled="strangeTotal === 0"
          @click="emitStartReview('strange', null)"
        >
          <p class="progress-kicker">Strange Mix</p>
          <span class="distribution-note">{{ formatPercentage(strangeFamilyPercentage) }} of labeled traces</span>
        </button>
        <div v-if="strangeBreakdown.length" class="distribution-list">
          <button
            v-for="item in strangeBreakdown"
            :key="item.id"
            type="button"
            class="distribution-row distribution-row-button"
            :disabled="item.count === 0"
            @click="emitStartReview('strange', item.name)"
          >
            <div>
              <strong class="distribution-label">{{ item.name }}</strong>
              <p class="distribution-meta">{{ item.count }} labeled</p>
            </div>
            <strong class="distribution-value">{{ formatPercentage(item.percentage) }}</strong>
          </button>
        </div>
        <p v-else class="distribution-empty">No strange labels have been created yet.</p>
      </section>
    </div>

    <dl class="meta-list">
      <div>
        <dt>Trace Source</dt>
        <dd>{{ bootstrap.tracePath }}</dd>
      </div>
      <div>
        <dt>Database</dt>
        <dd>{{ bootstrap.databaseFile }}</dd>
      </div>
    </dl>
  </section>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  bootstrap: { type: Object, required: true },
});

const emit = defineEmits(["start", "start-review"]);

const normalTotal = computed(() =>
  (props.bootstrap?.normalSummary || []).reduce((total, item) => total + Number(item.count || 0), 0),
);

const strangeTotal = computed(() =>
  (props.bootstrap?.strangeSummary || []).reduce((total, item) => total + Number(item.count || 0), 0),
);

const labeledCount = computed(() => normalTotal.value + strangeTotal.value);

const normalFamilyPercentage = computed(() =>
  labeledCount.value > 0 ? (normalTotal.value / labeledCount.value) * 100 : 0,
);

const strangeFamilyPercentage = computed(() =>
  labeledCount.value > 0 ? (strangeTotal.value / labeledCount.value) * 100 : 0,
);

const normalBreakdown = computed(() => {
  const countsByBucket = new Map(
    (props.bootstrap?.normalSummary || []).map((item) => [Number(item.bucket), Number(item.count || 0)]),
  );
  const total = normalTotal.value;
  const groupedItems = [
    { label: "0 peak", filterValue: "0", count: countsByBucket.get(0) || 0 },
    { label: "1 peak", filterValue: "1", count: countsByBucket.get(1) || 0 },
    { label: "2 peaks", filterValue: "2", count: countsByBucket.get(2) || 0 },
    { label: "3 peaks", filterValue: "3", count: countsByBucket.get(3) || 0 },
    {
      label: "4+ peaks",
      filterValue: "4+",
      count: [4, 5, 6, 7, 8, 9].reduce((sum, bucket) => sum + (countsByBucket.get(bucket) || 0), 0),
    },
  ];
  return groupedItems.map((item) => ({
    ...item,
    percentage: total > 0 ? (item.count / total) * 100 : 0,
  }));
});

const strangeBreakdown = computed(() => {
  const total = strangeTotal.value;
  return (props.bootstrap?.strangeSummary || []).map((item) => ({
    ...item,
    count: Number(item.count || 0),
    percentage: total > 0 ? (Number(item.count || 0) / total) * 100 : 0,
  }));
});

function formatPercentage(value) {
  const rounded = Math.round(value * 10) / 10;
  return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)}%`;
}

function emitStartReview(family, label) {
  emit("start-review", { family, label });
}
</script>
