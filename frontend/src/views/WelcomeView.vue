<template>
  <section class="welcome-panel">
    <p class="eyebrow">Trace Manual</p>
    <h1>Label traces with the keyboard.</h1>
    <p class="welcome-copy">
      Start a session, move through traces, and keep the side summaries visible while you work.
    </p>

    <div class="progress-card">
      <div>
        <p class="progress-kicker">Progress</p>
        <strong>{{ bootstrap.labeledCount ?? "N/A" }}</strong>
      </div>
      <div>
        <p class="progress-kicker">Strange labels</p>
        <strong>{{ bootstrap.strangeSummary?.length || 0 }}</strong>
      </div>
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
defineProps({
  bootstrap: { type: Object, required: true },
});

defineEmits(["start", "open-add-dialog", "open-review-dialog"]);
</script>
