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

function validIndex(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : null;
}

function applyUiState(payload: ChainUiState | null | undefined): void {
  if (!payload) return;
  state.selectedRun = validIndex(payload.selectedRun);
  state.selectedEventId = validIndex(payload.selectedEventId);
  state.stage = ["raw", "ingot", "match"].includes(payload.stage) ? payload.stage : "raw";
  state.showTheoryLines = typeof payload.showTheoryLines === "boolean"
    ? payload.showTheoryLines
    : true;
  state.selectedD1Index = validIndex(payload.selectedD1Index);
  state.selectedD2Index = validIndex(payload.selectedD2Index);
  state.selectedGaggIndex = validIndex(payload.selectedGaggIndex);
  state.ingotDistanceThreshold = typeof payload.ingotDistanceThreshold === "number" && Number.isFinite(payload.ingotDistanceThreshold) && payload.ingotDistanceThreshold >= 0
    ? payload.ingotDistanceThreshold
    : 1000;
  state.selectedD1FrontGuessIndex = validIndex(payload.selectedD1FrontGuessIndex);
  state.selectedD1BackGuessIndex = validIndex(payload.selectedD1BackGuessIndex);
  state.selectedD2FrontGuessIndex = validIndex(payload.selectedD2FrontGuessIndex);
  state.selectedD2BackGuessIndex = validIndex(payload.selectedD2BackGuessIndex);
}

function serializeUiState(): ChainUiState {
  return { ...state };
}

export function useChainStore() {
  return { state, applyUiState, serializeUiState };
}
