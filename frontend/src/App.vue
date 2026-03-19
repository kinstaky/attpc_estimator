<template>
  <div class="app-shell" @keydown.stop>
    <SidebarSummary
      title="Normal"
      kicker="Peak count"
      type="normal"
      side="left"
      :active="state.activeSidebar === 'left'"
      :items="state.bootstrap?.normalSummary || []"
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
        @open-add-dialog="openAddLabelDialog"
        @open-review-dialog="openReviewDialog"
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
    />

    <AddLabelDialog
      v-if="state.addLabelDialogOpen"
      :save-label="onSaveLabel"
      :remove-label="onRemoveLabel"
      :strange-labels="state.bootstrap?.strangeLabels || []"
      @close="closeAddLabelDialog"
    />

    <ReviewModeDialog
      v-if="state.reviewDialogOpen"
      :normal-summary="state.bootstrap?.normalSummary || []"
      :strange-labels="state.bootstrap?.strangeLabels || []"
      :start-review="onStartReview"
      @close="closeReviewDialog"
    />
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted } from "vue";
import AddLabelDialog from "./components/AddLabelDialog.vue";
import ReviewModeDialog from "./components/ReviewModeDialog.vue";
import SidebarSummary from "./components/SidebarSummary.vue";
import LabelView from "./views/LabelView.vue";
import WelcomeView from "./views/WelcomeView.vue";
import { useAppStore } from "./stores/app";

const {
  state,
  init,
  startLabeling,
  startReview,
  openAddLabelDialog,
  closeAddLabelDialog,
  openReviewDialog,
  closeReviewDialog,
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
  await init();
  window.addEventListener("keydown", handleKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", handleKeydown);
});
</script>
