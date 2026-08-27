import { computed, reactive } from "vue";

import { getMappingPads, getMappingSilicon } from "../api";
import type { MappingLayer, MappingPad, MappingRenderRule, MappingStrip, MappingUiState } from "../types";

interface MappingState {
  loading: boolean;
  error: string;
  pads: MappingPad[];
  silicon: Record<"T0D1" | "T0D2", MappingStrip[]>;
  selectedLayer: MappingLayer;
  selectedView: "Upstream" | "Downstream";
  rules: MappingRenderRule[];
  dialogOpen: boolean;
  editingIndex: number | null;
}

const state = reactive<MappingState>({
  loading: true,
  error: "",
  pads: [],
  silicon: { T0D1: [], T0D2: [] },
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
    const [pads, t0d1, t0d2] = await Promise.all([
      getMappingPads(),
      getMappingSilicon("T0D1"),
      getMappingSilicon("T0D2"),
    ]);
    state.pads = pads;
    state.silicon = { T0D1: t0d1, T0D2: t0d2 };
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

function setSelectedLayer(value: MappingLayer): void {
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
  const persistedLayer = payload.selectedLayer as MappingLayer | "Si-0" | "Si-1";
  state.selectedLayer = persistedLayer === "Si-0" ? "T0D1" : persistedLayer === "Si-1" ? "T0D2" : persistedLayer;
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
