<template>
  <div class="dialog-backdrop" @click.self="$emit('close')">
    <section class="dialog-panel">
      <header class="dialog-header">
        <div>
          <p class="dialog-kicker">Review Mode</p>
          <h3>Select labeled traces to review</h3>
        </div>
        <button class="ghost-button" @click="$emit('close')">Close</button>
      </header>

      <form class="dialog-form" @submit.prevent="submit">
        <label>
          <span>Label family</span>
          <select v-model="family">
            <option value="normal">Normal</option>
            <option value="strange">Strange</option>
          </select>
        </label>

        <label>
          <span>Label</span>
          <select v-model="selectedLabel">
            <option v-for="option in labelOptions" :key="option.value" :value="option.value">
              {{ option.text }}
            </option>
          </select>
        </label>

        <p class="dialog-help">
          Review mode only browses traces that are already labeled and match the selected filter.
        </p>
        <p class="dialog-error" v-if="localError">{{ localError }}</p>

        <div class="dialog-actions">
          <button class="ghost-button" type="button" @click="$emit('close')">Cancel</button>
          <button class="primary-button" type="submit" :disabled="submitting">Review</button>
        </div>
      </form>
    </section>
  </div>
</template>

<script setup>
import { computed, ref, watch } from "vue";

const props = defineProps({
  normalSummary: { type: Array, default: () => [] },
  strangeLabels: { type: Array, default: () => [] },
  startReview: { type: Function, required: true },
});

defineEmits(["close"]);

const family = ref("normal");
const selectedLabel = ref("__all__");
const localError = ref("");
const submitting = ref(false);

const labelOptions = computed(() => {
  if (family.value === "normal") {
    return [
      { value: "__all__", text: "All labels" },
      ...props.normalSummary.map((item) => ({
        value: String(item.bucket),
        text: item.title,
      })),
    ];
  }
  return [
    { value: "__all__", text: "All labels" },
    ...props.strangeLabels.map((item) => ({
      value: item.name,
      text: item.name,
    })),
  ];
});

watch(labelOptions, (options) => {
  selectedLabel.value = options[0]?.value || "";
}, { immediate: true });

async function submit() {
  localError.value = "";
  if (!selectedLabel.value) {
    localError.value = "Select a label to review.";
    return;
  }
  submitting.value = true;
  try {
    await props.startReview({
      family: family.value,
      label: selectedLabel.value === "__all__" ? null : selectedLabel.value,
    });
  } catch (error) {
    localError.value = error.message;
  } finally {
    submitting.value = false;
  }
}
</script>
