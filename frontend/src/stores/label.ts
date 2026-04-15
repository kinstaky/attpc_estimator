import { computed, reactive } from "vue";

import {
  createStrangeLabel,
  deleteStrangeLabel,
  nextTrace,
  previousTrace,
  saveLabel,
  setLabelSession,
} from "../api";
import { useShellStore } from "./shell";
import type {
  LabelAssignResponse,
  StrangeLabel,
  StrangeSummaryItem,
  TracePayload,
} from "../types";

type LabelMode = "browse" | "await_normal_peak" | "await_strange_choice";
type VisualMode = "raw" | "cdf" | "curvature";

interface LabelState {
  currentTrace: TracePayload | null;
  activeRun: number | null;
  mode: LabelMode;
  visualMode: VisualMode;
  loading: boolean;
  error: string;
  statusMessage: string;
  addDialogOpen: boolean;
  deleteDialogLabel: StrangeSummaryItem | StrangeLabel | null;
}

const state = reactive<LabelState>({
  currentTrace: null,
  activeRun: null,
  mode: "browse",
  visualMode: "raw",
  loading: false,
  error: "",
  statusMessage: "",
  addDialogOpen: false,
  deleteDialogLabel: null,
});

function clearTransientUi(): void {
  state.error = "";
  state.statusMessage = "";
}

function setMode(mode: LabelMode): void {
  state.mode = mode;
}

function cancelSelectionMode(): void {
  if (state.mode === "browse") {
    return;
  }
  state.mode = "browse";
}

function setVisualMode(mode: VisualMode): void {
  if (mode !== "raw" && mode !== "cdf" && mode !== "curvature") {
    return;
  }
  state.visualMode = mode;
}

function toggleVisualMode(): void {
  state.visualMode =
    state.visualMode === "raw"
      ? "cdf"
      : state.visualMode === "cdf"
        ? "curvature"
        : "raw";
}

async function enterLabelMode(run: number | null | undefined): Promise<void> {
  if (run === null || run === undefined) {
    throw new Error("Select a run before entering label mode.");
  }
  if (state.currentTrace && state.activeRun === Number(run)) {
    return;
  }
  state.loading = true;
  clearTransientUi();
  try {
    const payload = await setLabelSession(Number(run));
    state.currentTrace = payload.trace ?? null;
    state.activeRun = Number(run);
    setMode("browse");
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    state.loading = false;
  }
}

async function navigate(delta: number): Promise<void> {
  state.loading = true;
  clearTransientUi();
  try {
    state.currentTrace = delta < 0 ? await previousTrace() : await nextTrace();
  } catch (error) {
    state.statusMessage = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
    setMode("browse");
  }
}

function syncBootstrapSummaries(payload: LabelAssignResponse): void {
  const { state: shellState } = useShellStore();
  if (!shellState.bootstrap) {
    return;
  }
  shellState.bootstrap.normalSummary = payload.normalSummary;
  shellState.bootstrap.strangeSummary = payload.strangeSummary;
}

async function advanceAfterSave(successMessage: string): Promise<void> {
  try {
    state.currentTrace = await nextTrace();
    state.statusMessage = successMessage;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    state.statusMessage = `${successMessage} ${message}`;
  } finally {
    setMode("browse");
  }
}

async function submitNormal(label: number): Promise<void> {
  if (!state.currentTrace) {
    return;
  }
  state.loading = true;
  clearTransientUi();
  try {
    const payload = await saveLabel(
      state.currentTrace.eventId,
      state.currentTrace.traceId,
      "normal",
      String(label),
    );
    syncBootstrapSummaries(payload);
    state.currentTrace.currentLabel = payload.currentLabel;
    await advanceAfterSave(`Saved ${label} peak label.`);
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

async function submitStrange(labelName: string): Promise<void> {
  if (!state.currentTrace) {
    return;
  }
  state.loading = true;
  clearTransientUi();
  try {
    const payload = await saveLabel(
      state.currentTrace.eventId,
      state.currentTrace.traceId,
      "strange",
      labelName,
    );
    syncBootstrapSummaries(payload);
    state.currentTrace.currentLabel = payload.currentLabel;
    await advanceAfterSave("Saved strange label.");
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

function openAddDialog(): void {
  state.addDialogOpen = true;
}

function openDeleteDialog(label: StrangeSummaryItem | StrangeLabel): void {
  state.deleteDialogLabel = label;
}

async function addStrange(name: string, shortcutKey: string): Promise<void> {
  clearTransientUi();
  const { state: shellState } = useShellStore();
  const payload = await createStrangeLabel(name, shortcutKey);
  if (!shellState.bootstrap) {
    return;
  }
  shellState.bootstrap.strangeLabels = [
    ...(shellState.bootstrap.strangeLabels || []),
    payload,
  ];
  shellState.bootstrap.strangeSummary = [
    ...(shellState.bootstrap.strangeSummary || []),
    { ...payload, count: 0 },
  ];
  state.statusMessage = `Added label "${payload.name}".`;
}

async function removeStrange(name: string): Promise<void> {
  clearTransientUi();
  const { state: shellState } = useShellStore();
  const summary = await deleteStrangeLabel(name);
  if (!shellState.bootstrap) {
    return;
  }
  shellState.bootstrap.strangeLabels = (shellState.bootstrap.strangeLabels || []).filter(
    (item) => item.name !== name,
  );
  shellState.bootstrap.strangeSummary = summary.map((item) => ({
    ...item,
    shortcutKey:
      shellState.bootstrap?.strangeLabels.find((label) => label.name === item.name)
        ?.shortcutKey || "",
  }));
  state.statusMessage = `Deleted label "${name}".`;
}

const currentLabelText = computed(() => {
  const label = state.currentTrace?.currentLabel;
  if (!label) {
    return "Unlabeled";
  }
  if (label.family === "normal") {
    return `${label.label} peak${label.label === "1" ? "" : "s"}`;
  }
  return `Strange: ${label.label}`;
});

export function useLabelStore() {
  return {
    state,
    currentLabelText,
    clearTransientUi,
    setMode,
    cancelSelectionMode,
    setVisualMode,
    toggleVisualMode,
    enterLabelMode,
    navigate,
    submitNormal,
    submitStrange,
    openAddDialog,
    openDeleteDialog,
    addStrange,
    removeStrange,
  };
}
