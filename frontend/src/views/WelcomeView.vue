<template>
  <section class="welcome-panel">
    <p class="eyebrow">Session Overview</p>
    <h1>Label traces</h1>

    <div class="progress-card">
      <div>
        <p class="progress-kicker">Labeled traces</p>
        <strong>{{ bootstrap.labeledCount ?? "N/A" }}</strong>
      </div>
      <div>
        <p class="progress-kicker">Strange label types</p>
        <strong>{{ bootstrap.strangeSummary?.length || 0 }}</strong>
      </div>
    </div>

    <div class="distribution-grid">
      <section class="distribution-card">
        <div class="distribution-header">
          <p class="progress-kicker">Normal Mix</p>
          <span class="distribution-note">Share of labeled traces</span>
        </div>
        <div class="distribution-list">
          <article
            v-for="item in normalBreakdown"
            :key="item.label"
            class="distribution-row"
          >
            <div>
              <strong class="distribution-label">{{ item.label }}</strong>
              <p class="distribution-meta">{{ item.count }} labeled</p>
            </div>
            <strong class="distribution-value">{{ formatPercentage(item.percentage) }}</strong>
          </article>
        </div>
      </section>

      <section class="distribution-card">
        <div class="distribution-header">
          <p class="progress-kicker">Strange Mix</p>
          <span class="distribution-note">Share of labeled traces</span>
        </div>
        <div v-if="strangeBreakdown.length" class="distribution-list">
          <article
            v-for="item in strangeBreakdown"
            :key="item.id"
            class="distribution-row"
          >
            <div>
              <strong class="distribution-label">{{ item.name }}</strong>
              <p class="distribution-meta">{{ item.count }} labeled</p>
            </div>
            <strong class="distribution-value">{{ formatPercentage(item.percentage) }}</strong>
          </article>
        </div>
        <p v-else class="distribution-empty">No strange labels have been used yet.</p>
      </section>
    </div>

    <div class="button-row">
      <button class="primary-button" @click="$emit('start')">Start</button>
      <button class="ghost-button" @click="$emit('open-review-dialog')">Review</button>
      <button class="ghost-button" @click="$emit('open-add-dialog')">Add</button>
    </div>

    <dl class="meta-list">
      <div>
        <dt>Input</dt>
        <dd>{{ bootstrap.inputFile }}</dd>
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

defineEmits(["start", "open-add-dialog", "open-review-dialog"]);

const labeledCount = computed(() => Number(props.bootstrap?.labeledCount || 0));

const normalBreakdown = computed(() => {
  const countsByBucket = new Map(
    (props.bootstrap?.normalSummary || []).map((item) => [Number(item.bucket), Number(item.count || 0)]),
  );
  const total = labeledCount.value;
  const groups = [
    { label: "0 peak", count: countsByBucket.get(0) || 0 },
    { label: "1 peak", count: countsByBucket.get(1) || 0 },
    { label: "2 peaks", count: countsByBucket.get(2) || 0 },
    {
      label: "3+ peaks",
      count: Array.from({ length: 7 }, (_, offset) => countsByBucket.get(offset + 3) || 0).reduce(
        (sum, value) => sum + value,
        0,
      ),
    },
  ];
  return groups.map((item) => ({
    ...item,
    percentage: total > 0 ? (item.count / total) * 100 : 0,
  }));
});

const strangeBreakdown = computed(() => {
  const total = labeledCount.value;
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
</script>
