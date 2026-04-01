import { reactive } from "vue";
import {
  createStrangeLabel,
  deleteStrangeLabel,
  getBootstrap,
  getStrangeLabels,
  nextTrace,
  previousTrace,
  saveLabel,
  setTraceMode,
} from "../api";

const state = reactive({
  bootstrap: null,
  currentTrace: null,
  page: "welcome",
  mode: "welcome",
  browseVisualMode: "raw",
  sessionMode: "label",
  reviewFilter: null,
  activeSidebar: null,
  loading: false,
  error: "",
  statusMessage: "",
  addLabelDialogOpen: false,
  deleteLabelDialog: null,
});

function clearTransientUi() {
  state.error = "";
  state.statusMessage = "";
}

function setMode(mode) {
  state.mode = mode;
  state.activeSidebar = mode === "await_normal_peak" ? "left" : mode === "await_strange_choice" ? "right" : null;
}

function setBrowseVisualMode(mode) {
  if (mode !== "raw" && mode !== "analysis") {
    return;
  }
  state.browseVisualMode = mode;
}

function toggleBrowseVisualMode() {
  state.browseVisualMode = state.browseVisualMode === "raw" ? "analysis" : "raw";
}

async function init(initialBootstrap = null) {
  state.loading = true;
  clearTransientUi();
  try {
    const [bootstrap, strangeLabelsPayload] = await Promise.all([
      initialBootstrap ? Promise.resolve(initialBootstrap) : getBootstrap(),
      getStrangeLabels(),
    ]);
    const labeledCount =
      (bootstrap.normalSummary || []).reduce((total, item) => total + (item.count || 0), 0) +
      (bootstrap.strangeSummary || []).reduce((total, item) => total + (item.count || 0), 0);
    state.bootstrap = {
      ...bootstrap,
      labeledCount,
      strangeLabels: strangeLabelsPayload.strangeLabels || [],
    };
  } catch (error) {
    state.error = error.message;
  } finally {
    state.loading = false;
  }
}

async function switchSessionMode(mode, family = null, label = null) {
  const payload = await setTraceMode(mode, family, label);
  state.sessionMode = payload.mode;
  state.reviewFilter = payload.reviewFilter;
}

async function loadNextTrace() {
  state.currentTrace = await nextTrace();
}

async function startLabeling() {
  if (!state.bootstrap) {
    return;
  }
  clearTransientUi();
  const shouldReload = state.sessionMode !== "label" || !state.currentTrace;
  if (shouldReload) {
    state.currentTrace = null;
    state.loading = true;
  }
  try {
    await switchSessionMode("label");
    state.page = "label";
    setMode("label_browse");
    if (shouldReload) {
      await loadNextTrace();
    }
  } catch (error) {
    state.error = error.message;
  } finally {
    if (shouldReload) {
      state.loading = false;
    }
  }
}

async function startReview(family, label) {
  if (!state.bootstrap) {
    return;
  }
  clearTransientUi();
  state.currentTrace = null;
  state.loading = true;
  try {
    await switchSessionMode("review", family, label);
    state.page = "label";
    setMode("label_browse");
    await loadNextTrace();
  } catch (error) {
    state.error = error.message;
    throw error;
  } finally {
    state.loading = false;
  }
}

function goWelcome() {
  state.page = "welcome";
  setMode("welcome");
}

async function navigate(delta) {
  clearTransientUi();
  try {
    state.currentTrace = delta < 0 ? await previousTrace() : await nextTrace();
  } catch (error) {
    state.statusMessage = error.message;
  }
  setMode("label_browse");
}

async function advanceAfterSave(successMessage) {
  try {
    await loadNextTrace();
    state.statusMessage = successMessage;
  } catch (error) {
    state.statusMessage = `${successMessage} ${error.message}`;
  }
  setMode("label_browse");
}

async function submitNormal(peakCount) {
  if (!state.currentTrace) {
    return;
  }
  clearTransientUi();
  try {
    const payload = await saveLabel(
      state.currentTrace.eventId,
      state.currentTrace.traceId,
      "normal",
      String(peakCount),
    );
    syncSummaries(payload);
    state.currentTrace.currentLabel = payload.currentLabel;
    await advanceAfterSave(`Saved ${peakCount === 9 ? "9+" : peakCount} peak label.`);
  } catch (error) {
    state.error = error.message;
  }
}

async function submitStrange(strangeLabelName) {
  if (!state.currentTrace) {
    return;
  }
  clearTransientUi();
  try {
    const payload = await saveLabel(
      state.currentTrace.eventId,
      state.currentTrace.traceId,
      "strange",
      strangeLabelName,
    );
    syncSummaries(payload);
    state.currentTrace.currentLabel = payload.currentLabel;
    await advanceAfterSave("Saved strange label.");
  } catch (error) {
    state.error = error.message;
  }
}

async function addStrangeLabel(name, shortcutKey) {
  clearTransientUi();
  try {
    const payload = await createStrangeLabel(name, shortcutKey);
    if (state.bootstrap) {
      state.bootstrap.strangeLabels = [
        ...(state.bootstrap.strangeLabels || []),
        payload,
      ];
      state.bootstrap.strangeSummary = [
        ...(state.bootstrap.strangeSummary || []),
        { ...payload, count: 0 },
      ];
    }
    state.statusMessage = `Added label "${payload.name}".`;
    state.addLabelDialogOpen = false;
  } catch (error) {
    state.error = error.message;
    throw error;
  }
}

async function removeStrangeLabel(name) {
  clearTransientUi();
  try {
    const payload = await deleteStrangeLabel(name);
    if (state.bootstrap) {
      state.bootstrap.strangeLabels = (state.bootstrap.strangeLabels || []).filter(
        (label) => label.name !== name,
      );
      state.bootstrap.strangeSummary = payload;
    }
    state.statusMessage = `Deleted label "${name}".`;
  } catch (error) {
    state.error = error.message;
    throw error;
  }
}

function openAddLabelDialog() {
  state.addLabelDialogOpen = true;
  clearTransientUi();
}

function closeAddLabelDialog() {
  state.addLabelDialogOpen = false;
}

function openDeleteLabelDialog(label) {
  state.deleteLabelDialog = { ...label };
  clearTransientUi();
}

function closeDeleteLabelDialog() {
  state.deleteLabelDialog = null;
}

function syncSummaries(payload) {
  if (!state.bootstrap) {
    return;
  }
  state.bootstrap.normalSummary = payload.normalSummary;
  state.bootstrap.strangeSummary = payload.strangeSummary;
  state.bootstrap.labeledCount = payload.labeledCount ?? state.bootstrap.labeledCount;
}

function normalizeKey(event) {
  if (event.key === " ") {
    return "space";
  }
  if (event.key === "Esc") {
    return "escape";
  }
  return event.key.toLowerCase();
}

function shouldIgnoreKey(event) {
  if (state.addLabelDialogOpen || state.deleteLabelDialog) {
    return true;
  }
  const tagName = event.target?.tagName?.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

async function handleKeydown(event) {
  if (shouldIgnoreKey(event) || state.loading) {
    return;
  }

  const key = normalizeKey(event);

  if (state.page === "welcome") {
    if (key === "space") {
      event.preventDefault();
      await startLabeling();
    }
    return;
  }

  if (state.page !== "label") {
    return;
  }

  if (state.mode === "label_browse") {
    if (key === "q" || key === "escape") {
      event.preventDefault();
      goWelcome();
      return;
    }
    if (key === "arrowleft" || key === "h") {
      event.preventDefault();
      setMode("await_normal_peak");
      return;
    }
    if (key === "arrowright" || key === "l") {
      event.preventDefault();
      if (!state.bootstrap?.strangeLabels?.length) {
        state.statusMessage = "Add a strange label before using strange mode.";
        return;
      }
      setMode("await_strange_choice");
      return;
    }
    if (key === "arrowup" || key === "k") {
      event.preventDefault();
      await navigate(-1);
      return;
    }
    if (key === "f") {
      event.preventDefault();
      toggleBrowseVisualMode();
      return;
    }
    if (key === "arrowdown" || key === "j") {
      event.preventDefault();
      await navigate(1);
    }
    return;
  }

  if (state.mode === "await_normal_peak") {
    if (key === "q" || key === "escape") {
      event.preventDefault();
      setMode("label_browse");
      return;
    }
    if (key === "space") {
      event.preventDefault();
      await submitNormal(0);
      return;
    }
    if (/^[1-9]$/.test(key)) {
      event.preventDefault();
      await submitNormal(Number.parseInt(key, 10));
    }
    return;
  }

  if (state.mode === "await_strange_choice") {
    if (key === "q" || key === "escape") {
      event.preventDefault();
      setMode("label_browse");
      return;
    }
    const matchingLabel = state.bootstrap?.strangeLabels?.find((label) => label.shortcutKey === key);
    if (matchingLabel) {
      event.preventDefault();
      await submitStrange(matchingLabel.name);
    }
  }
}

function currentLabelText() {
  const label = state.currentTrace?.currentLabel;
  if (!label) {
    return "Unlabeled";
  }
  if (label.family === "normal") {
    return label.label === "9" ? "9+ peaks" : `${label.label} peak${label.label === "1" ? "" : "s"}`;
  }
  return `Strange: ${label.label}`;
}

export function useAppStore() {
  return {
    state,
    init,
    startLabeling,
    startReview,
    goWelcome,
    setMode,
    setBrowseVisualMode,
    toggleBrowseVisualMode,
    navigate,
    submitNormal,
    submitStrange,
    addStrangeLabel,
    removeStrangeLabel,
    openAddLabelDialog,
    closeAddLabelDialog,
    openDeleteLabelDialog,
    closeDeleteLabelDialog,
    handleKeydown,
    currentLabelText,
  };
}
