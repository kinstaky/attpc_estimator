import { reactive } from "vue";

import { getBootstrap } from "../api";
import type { BootstrapPayload, SessionPayload } from "../types";

interface ShellState {
  bootstrap: BootstrapPayload | null;
  selectedRun: number | null;
  loading: boolean;
  error: string;
  initialized: boolean;
}

const state = reactive<ShellState>({
  bootstrap: null,
  selectedRun: null,
  loading: true,
  error: "",
  initialized: false,
});

function resolveInitialRun(bootstrap: BootstrapPayload): number | null {
  const sessionRun = bootstrap.session?.run;
  if (sessionRun !== null && sessionRun !== undefined) {
    return Number(sessionRun);
  }
  const firstRun = bootstrap.runs?.[0];
  return firstRun === undefined ? null : Number(firstRun);
}

function applyBootstrap(bootstrap: BootstrapPayload): void {
  state.bootstrap = bootstrap;
  const availableRuns = (bootstrap.runs || []).map((run) => Number(run));
  if (
    state.selectedRun === null ||
    !availableRuns.includes(Number(state.selectedRun))
  ) {
    state.selectedRun = resolveInitialRun(bootstrap);
  }
  state.initialized = true;
}

async function init(): Promise<void> {
  if (state.initialized && state.bootstrap) {
    return;
  }
  state.loading = true;
  state.error = "";
  try {
    applyBootstrap(await getBootstrap());
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
  }
}

function setSelectedRun(run: number | string | null | undefined): void {
  state.selectedRun =
    run === null || run === undefined || run === "" ? null : Number(run);
}

function updateSession(session: SessionPayload): void {
  if (!state.bootstrap) {
    return;
  }
  state.bootstrap.session = session;
}

export function useShellStore() {
  return {
    state,
    init,
    applyBootstrap,
    setSelectedRun,
    updateSession,
  };
}
