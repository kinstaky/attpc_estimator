<template>
  <div class="dialog-backdrop" @click.self="$emit('close')">
    <section class="dialog-panel">
      <header class="dialog-header">
        <div>
          <p class="dialog-kicker">Strange Label</p>
          <h3>Manage strange labels</h3>
        </div>
        <button class="ghost-button" @click="$emit('close')">Close</button>
      </header>
      <form class="dialog-form" @submit.prevent="submit">
        <label>
          <span>Name</span>
          <input v-model.trim="name" type="text" maxlength="48" autofocus />
        </label>
        <label>
          <span>Shortcut key</span>
          <input v-model.trim="shortcutKey" type="text" maxlength="12" />
        </label>
        <p class="dialog-help">
          Reserved: arrows, h/j/k/l, q, esc, space, 0-9
        </p>
        <p class="dialog-error" v-if="localError">{{ localError }}</p>
        <div class="dialog-actions">
          <button class="ghost-button" type="button" @click="$emit('close')">Cancel</button>
          <button class="primary-button" type="submit" :disabled="submitting">Save</button>
        </div>
      </form>

      <section class="dialog-section">
        <header class="dialog-section-header">
          <div>
            <p class="dialog-kicker">Existing labels</p>
            <h4 class="dialog-section-title">Delete unused or incorrect labels</h4>
          </div>
        </header>
        <div v-if="strangeLabels.length" class="label-admin-list">
          <article
            v-for="label in strangeLabels"
            :key="label.name"
            class="label-admin-card"
          >
            <div>
              <strong>{{ label.name }}</strong>
              <p class="label-admin-meta">key {{ label.shortcutKey === " " ? "space" : label.shortcutKey }}</p>
            </div>
            <button
              class="danger-button"
              type="button"
              :disabled="deletingName === label.name"
              @click="remove(label.name)"
            >
              {{ deletingName === label.name ? "Deleting…" : "Delete" }}
            </button>
          </article>
        </div>
        <p v-else class="distribution-empty">No strange labels have been created yet.</p>
      </section>
    </section>
  </div>
</template>

<script setup>
import { ref } from "vue";

const props = defineProps({
  saveLabel: { type: Function, required: true },
  removeLabel: { type: Function, required: true },
  strangeLabels: { type: Array, default: () => [] },
});

const emit = defineEmits(["close"]);

const name = ref("");
const shortcutKey = ref("");
const localError = ref("");
const submitting = ref(false);
const deletingName = ref("");

async function submit() {
  localError.value = "";
  if (!name.value) {
    localError.value = "Name is required.";
    return;
  }
  if (!shortcutKey.value) {
    localError.value = "Shortcut key is required.";
    return;
  }
  submitting.value = true;
  try {
    await props.saveLabel({ name: name.value, shortcutKey: shortcutKey.value });
    name.value = "";
    shortcutKey.value = "";
  } catch (error) {
    localError.value = error.message;
  } finally {
    submitting.value = false;
  }
}

async function remove(labelName) {
  localError.value = "";
  deletingName.value = labelName;
  try {
    await props.removeLabel(labelName);
  } catch (error) {
    localError.value = error.message;
  } finally {
    deletingName.value = "";
  }
}
</script>
