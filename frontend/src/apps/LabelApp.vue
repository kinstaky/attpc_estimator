<template>
  <div class="app-shell" @keydown.stop>
    <SidebarSummary
      title="Normal"
      kicker="Peak count"
      type="normal"
      side="left"
      :active="state.activeSidebar === 'left'"
      :items="state.bootstrap?.normalSummary || []"
      :show-add-card="true"
      add-label="Add label"
      add-description="Create strange shortcut"
      @add="openAddLabelDialog"
    />

    <main class="main-panel">
      <section class="status-strip">
        <span v-if="state.loading">Loading…</span>
        <span class="status-error" v-else-if="state.error">{{ state.error }}</span>
        <span v-else-if="state.statusMessage">{{ state.statusMessage }}</span>
        <span v-else>&nbsp;</span>
      </section>

      <WelcomeView
        v-if="state.bootstrap && state.page === 'welcome'"
        :bootstrap="state.bootstrap"
        @start="startLabeling"
        @start-review="onStartReview"
      />

      <LabelView
        v-else-if="state.currentTrace && state.page === 'label'"
        :trace="state.currentTrace"
        :mode="state.mode"
        :visual-mode="state.browseVisualMode"
        :session-mode="state.sessionMode"
        :review-filter="state.reviewFilter"
        :current-label-text="currentLabelText()"
        :can-toggle-visual-mode="state.mode === 'label_browse'"
        @set-visual-mode="setBrowseVisualMode"
      />

      <section v-else-if="state.page === 'label'" class="welcome-panel">
        <p class="eyebrow">Trace Manual</p>
        <h1>Loading trace…</h1>
        <p class="welcome-copy">
          Switching mode and fetching the next trace.
        </p>
      </section>

      <section v-else class="welcome-panel">
        <p class="eyebrow">Trace Manual</p>
        <h1>Loading app state…</h1>
      </section>
    </main>

    <SidebarSummary
      title="Strange"
      kicker="Special cases"
      type="strange"
      side="right"
      :active="state.activeSidebar === 'right'"
      :items="state.bootstrap?.strangeSummary || []"
      :show-add-card="true"
      :allow-delete="true"
      add-label="Add label"
      add-description="Create strange shortcut"
      @add="openAddLabelDialog"
      @request-delete="openDeleteLabelDialog"
    />

    <AddLabelDialog
      v-if="state.addLabelDialogOpen"
      :save-label="onSaveLabel"
      @close="closeAddLabelDialog"
    />

    <DeleteLabelDialog
      v-if="state.deleteLabelDialog"
      :label="state.deleteLabelDialog"
      :delete-label="onRemoveLabel"
      @close="closeDeleteLabelDialog"
    />
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted } from "vue";
import AddLabelDialog from "../components/AddLabelDialog.vue";
import DeleteLabelDialog from "../components/DeleteLabelDialog.vue";
import SidebarSummary from "../components/SidebarSummary.vue";
import LabelView from "../views/LabelView.vue";
import WelcomeView from "../views/WelcomeView.vue";
import { useAppStore } from "../stores/app";

const props = defineProps({
  bootstrap: { type: Object, required: true },
});

const {
  state,
  init,
  startLabeling,
  startReview,
  openAddLabelDialog,
  closeAddLabelDialog,
  openDeleteLabelDialog,
  closeDeleteLabelDialog,
  addStrangeLabel,
  removeStrangeLabel,
  handleKeydown,
  currentLabelText,
  setBrowseVisualMode,
} = useAppStore();

async function onSaveLabel(payload) {
  await addStrangeLabel(payload.name, payload.shortcutKey);
}

async function onRemoveLabel(name) {
  await removeStrangeLabel(name);
}

async function onStartReview(payload) {
  await startReview(payload.family, payload.label);
}

onMounted(async () => {
  await init(props.bootstrap);
  window.addEventListener("keydown", handleKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", handleKeydown);
});
</script>
