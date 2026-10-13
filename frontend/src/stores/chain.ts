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
  ingotDistanceThreshold: 1000,
  selectedD1FrontGuessIndex: null,
  selectedD1BackGuessIndex: null,
  selectedD2FrontGuessIndex: null,
  selectedD2BackGuessIndex: null,
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
  state.ingotDistanceThreshold = typeof payload.ingotDistanceThreshold === "number" && Number.isFinite(payload.ingotDistanceThreshold) && payload.ingotDistanceThreshold >= 0
    ? payload.ingotDistanceThreshold
    : 1000;
  state.selectedD1FrontGuessIndex = typeof payload.selectedD1FrontGuessIndex === "number" ? payload.selectedD1FrontGuessIndex : null;
  state.selectedD1BackGuessIndex = typeof payload.selectedD1BackGuessIndex === "number" ? payload.selectedD1BackGuessIndex : null;
  state.selectedD2FrontGuessIndex = typeof payload.selectedD2FrontGuessIndex === "number" ? payload.selectedD2FrontGuessIndex : null;
  state.selectedD2BackGuessIndex = typeof payload.selectedD2BackGuessIndex === "number" ? payload.selectedD2BackGuessIndex : null;
}

function serializeUiState(): ChainUiState {
  return { ...state };
}

export function useChainStore() {
  return { state, applyUiState, serializeUiState };
}
