import { watch } from "vue";

import router from "../router";
import { updateUiState } from "../api";
import type { UiStatePayload } from "../types";
import { useHistogramStore } from "./histograms";
import { useLabelStore } from "./label";
import { useMappingStore } from "./mapping";
import { useReviewStore } from "./review";
import { useShellStore } from "./shell";

let persistenceTimer: ReturnType<typeof window.setTimeout> | null = null;
let persistenceStarted = false;

function shouldRestoreRoute(): boolean {
  return router.currentRoute.value.fullPath === "/";
}

export async function normalizeCurrentRoute(): Promise<void> {
  const current = router.currentRoute.value.fullPath;
  if (
    current.startsWith("/pointcloud")
    || current.startsWith("/label/pointcloud")
    || current.startsWith("/browse/pointcloud")
  ) {
    await router.replace("/");
  }
}

export async function hydrateUiState(payload: UiStatePayload | null | undefined): Promise<void> {
  const shell = useShellStore();
  const label = useLabelStore();
  const review = useReviewStore();
  const histograms = useHistogramStore();
  const mapping = useMappingStore();

  if (!payload) {
    return;
  }

  shell.setSelectedRun(payload.shell.selectedRun);
  label.applyUiState(payload.label);
  review.applyUiState(payload.review);
  histograms.applyUiState(payload.histograms);
  mapping.applyUiState(payload.mapping);

  const routeDisabledInTraceOnly = Boolean(
    payload.route?.startsWith("/pointcloud")
    || payload.route?.startsWith("/label/pointcloud")
    || payload.route?.startsWith("/browse/pointcloud"),
  );
  const normalizedRoute = payload.route?.startsWith("/review")
    ? payload.route.replace("/review", "/browse/trace")
    : routeDisabledInTraceOnly
      ? "/"
      : payload.route;

  if (shouldRestoreRoute() && normalizedRoute && normalizedRoute !== router.currentRoute.value.path) {
    await router.replace(normalizedRoute);
  }

  if (router.currentRoute.value.path === "/histograms") {
    await histograms.loadHistogram();
  }
}

export function snapshotUiState(): UiStatePayload {
  const shell = useShellStore();
  const label = useLabelStore();
  const review = useReviewStore();
  const histograms = useHistogramStore();
  const mapping = useMappingStore();

  return {
    route: router.currentRoute.value.fullPath,
    shell: {
      selectedRun: shell.state.selectedRun,
    },
    label: label.serializeUiState(),
    review: review.serializeUiState(),
    histograms: histograms.serializeUiState(),
    mapping: mapping.serializeUiState(),
  };
}

function queuePersistence(): void {
  if (persistenceTimer !== null) {
    window.clearTimeout(persistenceTimer);
  }
  persistenceTimer = window.setTimeout(() => {
    persistenceTimer = null;
    void updateUiState(snapshotUiState());
  }, 250);
}

export function startUiStatePersistence(): void {
  if (persistenceStarted) {
    return;
  }
  persistenceStarted = true;
  watch(
    snapshotUiState,
    () => {
      queuePersistence();
    },
    { deep: true },
  );
}
