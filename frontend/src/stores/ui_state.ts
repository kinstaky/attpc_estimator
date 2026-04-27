import { watch } from "vue";

import router from "../router";
import { updateUiState } from "../api";
import type { UiStatePayload } from "../types";
import { useHistogramStore } from "./histograms";
import { useLabelStore } from "./label";
import { useMappingStore } from "./mapping";
import { usePointcloudLabelStore } from "./pointcloud_label";
import { usePointcloudStore } from "./pointcloud";
import { useReviewStore } from "./review";
import { useShellStore } from "./shell";

let persistenceTimer: ReturnType<typeof window.setTimeout> | null = null;
let persistenceStarted = false;

function shouldRestoreRoute(): boolean {
  return router.currentRoute.value.fullPath === "/";
}

export async function hydrateUiState(payload: UiStatePayload | null | undefined): Promise<void> {
  const shell = useShellStore();
  const label = useLabelStore();
  const review = useReviewStore();
  const histograms = useHistogramStore();
  const mapping = useMappingStore();
  const pointcloud = usePointcloudStore();
  const pointcloudLabel = usePointcloudLabelStore();

  if (!payload) {
    return;
  }

  shell.setSelectedRun(payload.shell.selectedRun);
  label.applyUiState(payload.label);
  review.applyUiState(payload.review);
  histograms.applyUiState(payload.histograms);
  mapping.applyUiState(payload.mapping);
  pointcloud.applyUiState(payload.pointcloud);
  pointcloudLabel.applyUiState(payload.pointcloudLabel);

  const normalizedRoute = payload.route?.startsWith("/review")
    ? payload.route.replace("/review", "/browse/trace")
    : payload.route?.startsWith("/pointcloud")
      ? payload.route.replace("/pointcloud", "/browse/pointcloud")
      : payload.route;

  if (normalizedRoute?.split("?", 1)[0] === "/histograms") {
    await histograms.init();
  }

  if (shouldRestoreRoute() && normalizedRoute && normalizedRoute !== router.currentRoute.value.path) {
    await router.replace(normalizedRoute);
  }
}

export function snapshotUiState(): UiStatePayload {
  const shell = useShellStore();
  const label = useLabelStore();
  const review = useReviewStore();
  const histograms = useHistogramStore();
  const mapping = useMappingStore();
  const pointcloud = usePointcloudStore();
  const pointcloudLabel = usePointcloudLabelStore();

  return {
    route: router.currentRoute.value.fullPath,
    shell: {
      selectedRun: shell.state.selectedRun,
    },
    label: label.serializeUiState(),
    review: review.serializeUiState(),
    histograms: histograms.serializeUiState(),
    mapping: mapping.serializeUiState(),
    pointcloud: pointcloud.serializeUiState(),
    pointcloudLabel: pointcloudLabel.serializeUiState(),
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
