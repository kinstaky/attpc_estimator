import { computed, reactive } from "vue";

import { getMappingPads } from "../api";
import type { MappingPad, MappingRenderRule, MappingUiState } from "../types";

interface MappingState {
  loading: boolean;
  error: string;
  pads: MappingPad[];
  selectedLayer: "Pads" | "Si-0" | "Si-1";
  selectedView: "Upstream" | "Downstream";
  rules: MappingRenderRule[];
  dialogOpen: boolean;
  editingIndex: number | null;
}

const state = reactive<MappingState>({
  loading: true,
  error: "",
  pads: [],
  selectedLayer: "Pads",
  selectedView: "Upstream",
  rules: [],
  dialogOpen: false,
  editingIndex: null,
});

const editingRule = computed<MappingRenderRule | null>(() => {
  if (state.editingIndex === null) {
    return null;
  }
  return state.rules[state.editingIndex] ?? null;
});

async function loadPads(): Promise<void> {
  state.loading = true;
  state.error = "";
  try {
    state.pads = await getMappingPads();
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

function setSelectedLayer(value: "Pads" | "Si-0" | "Si-1"): void {
  state.selectedLayer = value;
}

function setSelectedView(value: "Upstream" | "Downstream"): void {
  state.selectedView = value;
}

function openNewRule(): void {
  state.editingIndex = null;
  state.dialogOpen = true;
}

function openEditRule(index: number): void {
  state.editingIndex = index;
  state.dialogOpen = true;
}

function setDialogOpen(value: boolean): void {
  state.dialogOpen = value;
}

function saveRule(payload: { index: number | null; rule: MappingRenderRule }): void {
  if (payload.index === null) {
    state.rules = [...state.rules, payload.rule];
    return;
  }
  state.rules = state.rules.map((rule, index) => (
    index === payload.index ? payload.rule : rule
  ));
}

function deleteRule(index: number): void {
  state.rules = state.rules.filter((_, itemIndex) => itemIndex !== index);
}

function applyUiState(payload: MappingUiState | null | undefined): void {
  if (!payload) {
    return;
  }
  state.selectedLayer = payload.selectedLayer;
  state.selectedView = payload.selectedView;
  state.rules = payload.rules.map((rule) => ({ ...rule }));
}

function serializeUiState(): MappingUiState {
  return {
    selectedLayer: state.selectedLayer,
    selectedView: state.selectedView,
    rules: state.rules.map((rule) => ({ ...rule })),
  };
}

export function useMappingStore() {
  return {
    state,
    editingRule,
    loadPads,
    setSelectedLayer,
    setSelectedView,
    openNewRule,
    openEditRule,
    setDialogOpen,
    saveRule,
    deleteRule,
    applyUiState,
    serializeUiState,
  };
}
