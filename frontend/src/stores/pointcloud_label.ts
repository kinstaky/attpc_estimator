import { computed, reactive } from "vue";

import {
  getCurrentPointcloudLabelEvent,
  nextPointcloudLabelEvent,
  previousPointcloudLabelEvent,
  savePointcloudLabel,
  setPointcloudLabelSession,
  setPointcloudLabelReviewSession,
} from "../api";
import { useShellStore } from "./shell";
import type {
  PointcloudLabelEventPayload,
  PointcloudLabelUiState,
  SessionResponse,
} from "../types";

type VisualMode = "basic" | "detail";
type LabelMode = "browse" | "await_line_count";

interface PointcloudLabelState {
  currentEvent: PointcloudLabelEventPayload | null;
  activeRun: number | null;
  isReviewMode: boolean;
  reviewLabel: string | null;
  visualMode: VisualMode;
  mode: LabelMode;
  loading: boolean;
  error: string;
  statusMessage: string;
}

const state = reactive<PointcloudLabelState>({
  currentEvent: null,
  activeRun: null,
  isReviewMode: false,
  reviewLabel: null,
  visualMode: "basic",
  mode: "browse",
  loading: false,
  error: "",
  statusMessage: "",
});

function clearTransientUi(): void {
  state.error = "";
  state.statusMessage = "";
}

function setMode(mode: LabelMode): void {
  state.mode = mode;
}

function cancelSelectionMode(): void {
  state.mode = "browse";
}

function setVisualMode(mode: VisualMode): void {
  if (mode !== "basic" && mode !== "detail") {
    return;
  }
  state.visualMode = mode;
}

function toggleVisualMode(): void {
  state.visualMode =
    state.visualMode === "basic"
      ? "detail"
      : "basic";
}

function syncSession(payload: SessionResponse): void {
  useShellStore().updateSession(payload.session);
}

async function enterLabelMode(run: number | null | undefined): Promise<void> {
  if (run === null || run === undefined) {
    throw new Error("Select a pointcloud run before entering pointcloud label mode.");
  }
  if (state.currentEvent && state.activeRun === Number(run)) {
    return;
  }
  state.loading = true;
  clearTransientUi();
  try {
    const payload = await setPointcloudLabelSession(Number(run));
    syncSession(payload);
    state.currentEvent = payload.event ?? null;
    state.activeRun = Number(run);
    state.isReviewMode = false;
    state.reviewLabel = null;
    setMode("browse");
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    state.loading = false;
  }
}

async function enterReviewMode(
  run: number | null | undefined,
  options: {
    label?: string | null;
  } = {},
): Promise<void> {
  if (run === null || run === undefined) {
    throw new Error("Select a pointcloud run before entering pointcloud review mode.");
  }
  state.loading = true;
  clearTransientUi();
  try {
    const label = options.label ?? null;
    const payload = await setPointcloudLabelReviewSession(Number(run), label);
    syncSession(payload);
    state.currentEvent = payload.event ?? null;
    state.activeRun = Number(run);
    state.isReviewMode = true;
    state.reviewLabel = label;
    setMode("browse");
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    state.loading = false;
  }
}

async function restoreCurrentSession(): Promise<void> {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentEvent = await getCurrentPointcloudLabelEvent();
    state.activeRun = state.currentEvent.run;
    const session = useShellStore().state.bootstrap?.session;
    state.isReviewMode = session?.mode === "pointcloud_label_review";
    state.reviewLabel = session?.mode === "pointcloud_label_review" ? session.label ?? null : null;
    setMode("browse");
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    state.loading = false;
  }
}

async function exitReviewMode(): Promise<void> {
  if (!state.isReviewMode) {
    return;
  }
  state.currentEvent = null;
  await enterLabelMode(state.activeRun);
}

async function navigate(delta: number): Promise<void> {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentEvent = delta < 0
      ? await previousPointcloudLabelEvent()
      : await nextPointcloudLabelEvent();
  } catch (error) {
    state.statusMessage = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
    setMode("browse");
  }
}

function normalizeBucket(label: string): string | null {
  if (["0", "1", "2", "3", "4", "5", "6+"].includes(label)) {
    return label;
  }
  return null;
}

async function submitLabel(label: string): Promise<void> {
  if (!state.currentEvent) {
    return;
  }
  const resolvedLabel = normalizeBucket(label);
  if (!resolvedLabel) {
    return;
  }
  state.loading = true;
  clearTransientUi();
  try {
    const payload = await savePointcloudLabel(state.currentEvent.eventId, resolvedLabel);
    const bootstrap = useShellStore().state.bootstrap;
    if (bootstrap) {
      bootstrap.pointcloudSummary = payload.pointcloudSummary;
    }
    state.currentEvent.currentLabel = payload.currentLabel;
    state.currentEvent = await nextPointcloudLabelEvent();
    state.statusMessage = `Saved ${resolvedLabel} line label.`;
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
    setMode("browse");
  }
}

async function submitSuggestedLabel(): Promise<void> {
  if (!state.currentEvent) {
    return;
  }
  await submitLabel(state.currentEvent.suggestedLabel);
}

function applyUiState(payload: PointcloudLabelUiState | null | undefined): void {
  if (!payload) {
    return;
  }
  setVisualMode(payload.visualMode);
}

function serializeUiState(): PointcloudLabelUiState {
  return {
    visualMode: state.visualMode,
  };
}

const currentLabelText = computed(() => {
  if (!state.currentEvent?.currentLabel) {
    return "Unlabeled";
  }
  return `${state.currentEvent.currentLabel} lines`;
});

export function usePointcloudLabelStore() {
  return {
    state,
    currentLabelText,
    clearTransientUi,
    setMode,
    cancelSelectionMode,
    setVisualMode,
    toggleVisualMode,
    enterLabelMode,
    enterReviewMode,
    exitReviewMode,
    restoreCurrentSession,
    navigate,
    submitLabel,
    submitSuggestedLabel,
    applyUiState,
    serializeUiState,
  };
}
