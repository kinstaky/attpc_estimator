<template>
  <section class="label-panel" v-if="trace">
    <header class="label-header">
      <div>
        <p class="eyebrow">Event {{ trace.eventId }} · Trace {{ trace.traceId }}</p>
        <h1>{{ currentLabelText }}</h1>
      </div>
      <div class="label-legend">
        <span>Left / h: normal</span>
        <span>Right / l: strange</span>
      </div>
    </header>

    <TracePlot :trace="trace" />

    <div class="trace-meta">
      <div class="meta-card">
        <span class="meta-title">Trace key</span>
        <strong>{{ trace.eventId }} / {{ trace.traceId }}</strong>
      </div>
      <div class="meta-card">
        <span class="meta-title">Input mode</span>
        <strong>{{ modeLabel }}</strong>
      </div>
      <div class="meta-card">
        <span class="meta-title">Session</span>
        <strong>{{ sessionLabel }}</strong>
      </div>
      <div class="meta-card" v-if="trace.reviewProgress">
        <span class="meta-title">Review progress</span>
        <strong>{{ trace.reviewProgress.current }} / {{ trace.reviewProgress.total }}</strong>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";
import TracePlot from "../components/TracePlot.vue";

const props = defineProps({
  trace: { type: Object, required: true },
  currentLabelText: { type: String, required: true },
  mode: { type: String, required: true },
  sessionMode: { type: String, required: true },
  reviewFilter: { type: Object, default: null },
});

const modeLabel = computed(() => {
  if (props.mode === "await_normal_peak") {
    return "Waiting for normal peak selection";
  }
  if (props.mode === "await_strange_choice") {
    return "Waiting for strange label selection";
  }
  return "Browse";
});

const sessionLabel = computed(() => {
  if (props.sessionMode !== "review") {
    return "Label mode";
  }
  if (!props.reviewFilter?.label) {
    return `Review ${props.reviewFilter?.family || ""} labels`.trim();
  }
  if (props.reviewFilter.family === "normal") {
    return `Review ${props.reviewFilter.label === "9" ? "9+ peaks" : `${props.reviewFilter.label} peak${props.reviewFilter.label === "1" ? "" : "s"}`}`;
  }
  return `Review strange: ${props.reviewFilter.label}`;
});
</script>
