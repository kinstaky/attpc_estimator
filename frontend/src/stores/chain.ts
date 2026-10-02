import { reactive } from "vue";

import type { ChainUiState } from "../types";

const state = reactive<ChainUiState>({
  selectedRun: null,
  selectedEventId: null,
  stage: "raw",
  showTheoryLines: true,
  selectedD1Index: null,
  selectedD2Index: null,
  selectedGaggIndex: null,
});

function applyUiState(payload: ChainUiState | null | undefined): void {
  if (!payload) return;
  state.selectedRun = payload.selectedRun;
  state.selectedEventId = payload.selectedEventId;
  state.stage = payload.stage;
  state.showTheoryLines = typeof payload.showTheoryLines === "boolean"
    ? payload.showTheoryLines
    : true;
  state.selectedD1Index = typeof payload.selectedD1Index === "number" ? payload.selectedD1Index : null;
  state.selectedD2Index = typeof payload.selectedD2Index === "number" ? payload.selectedD2Index : null;
  state.selectedGaggIndex = typeof payload.selectedGaggIndex === "number" ? payload.selectedGaggIndex : null;
}

function serializeUiState(): ChainUiState {
  return { ...state };
}

export function useChainStore() {
  return { state, applyUiState, serializeUiState };
}
