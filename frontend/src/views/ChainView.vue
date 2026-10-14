<template>
  <v-container class="page-container" fluid>
    <div class="page-header"><div><p class="page-kicker">Browse</p><h1>Chain review</h1><p class="page-copy">Review silicon and GAGG processing for one direct event across raw, ingot, and match stages.</p></div></div>
    <v-card class="control-card" rounded="xl"><v-card-text><v-row dense>
      <v-col cols="12" md="3"><v-select :items="runs" item-title="title" item-value="value" label="Run" v-model="run" variant="outlined" /></v-col>
      <v-col cols="12" md="3"><v-text-field label="Event id" type="number" v-model.number="eventId" variant="outlined" /></v-col>
      <v-col cols="12" md="3" class="d-flex align-center"><v-btn color="primary" :loading="loading" @click="load">Load chain</v-btn></v-col>
      <v-col cols="12" md="3" class="d-flex align-center justify-end"><v-btn-toggle v-model="stage" mandatory class="chain-stage-toggle"><v-btn v-for="item in stageNames" :key="item" :value="item">{{ item }}</v-btn></v-btn-toggle></v-col>
    </v-row></v-card-text></v-card>
    <v-alert v-if="error" class="mt-4" color="error" rounded="xl" variant="tonal">{{ error }}</v-alert>
    <v-card v-if="event" class="trace-stage-card mt-4" rounded="xl"><v-card-title class="trace-stage-title"><div><p class="page-kicker">Run {{ event.run }} · Event {{ event.eventId }}</p><h2>{{ stage }} stage</h2></div><div class="trace-stage-hints"><span>{{ shortcutHint }}</span></div></v-card-title><v-card-text>
      <ChainTraceGrid v-if="stage === 'raw'" :traces="event.stages.raw.traces" />
      <ChainIngotReview v-else-if="stage === 'ingot'" :stage="event.stages.ingot" :threshold="chain.state.ingotDistanceThreshold" :selected="ingotSelections" @update-threshold="chain.state.ingotDistanceThreshold = $event" @select-guess="selectIngotGuess" />
      <section v-show="stage === 'match'" class="chain-match-stage" aria-label="matched detector candidates">
        <div class="chain-match-tables">
          <article v-for="detector in detectorOrder" :key="detector" class="chain-match-table-shell" :class="`chain-match-table-shell--${detector}`">
            <header><span>{{ detector.toUpperCase() }}</span><strong>{{ matchRows(detector).length }}</strong></header>
            <v-table class="chain-table chain-match-table"><thead><tr><th v-for="column in matchColumns(detector)" :key="column" class="chain-table__number">{{ columnLabel(column) }}</th></tr></thead><tbody><tr v-for="(row, index) in matchRows(detector)" :key="index" :class="{ 'chain-match-row--selected': selectedIndex(detector) === index }" tabindex="0" @click="selectCandidate(detector, index)" @keydown.enter.prevent="selectCandidate(detector, index)" @keydown.space.prevent="selectCandidate(detector, index)"><td v-for="column in matchColumns(detector)" :key="column" class="chain-table__number">{{ formatValue(column, row[column]) }}</td></tr></tbody></v-table>
            <p v-if="!matchRows(detector).length" class="chain-match-empty">No candidates</p>
          </article>
        </div>
        <v-alert v-if="pidError" class="mt-4" color="warning" rounded="lg" variant="tonal">{{ pidError }}</v-alert>
        <div v-else-if="pidLoading" class="chain-pid-loading">Loading calibrated PID background…</div>
        <ChainMatchPidPlots v-else-if="pidBackground" v-model:show-theory="showTheoryLines" :background="pidBackground" :d1d2-marker="d1d2Marker" :d2gagg-marker="d2gaggMarker" />
      </section>
      <p v-if="stage === 'raw' && !event.stages.raw.traces.length" class="text-medium-emphasis">No raw channels were selected by this event's ingot outputs.</p>
      <p v-if="stage === 'ingot' && !event.stages.ingot.rows.length" class="text-medium-emphasis">No rows are available for this event at this stage.</p>
    </v-card-text></v-card>
  </v-container>
</template>

<script setup lang="ts">
import { computed, onActivated, onBeforeUnmount, onDeactivated, onMounted, ref, watch } from "vue";
import ChainIngotReview, { type IngotGuessTable } from "../components/ChainIngotReview.vue";
import ChainMatchPidPlots from "../components/ChainMatchPidPlots.vue";
import ChainTraceGrid from "../components/ChainTraceGrid.vue";
import {
  getChainMatchPidBackground,
  nextAvailableChainEvent,
  prefetchAvailableChainEvents,
  previousAvailableChainEvent,
  setChainSession,
} from "../api";
import { useChainStore } from "../stores/chain";
import { useShellStore } from "../stores/shell";
import type { ChainEventPayload, ChainIngotGuess, ChainMatchPidBackgroundPayload } from "../types";

type MatchDetector = "d1" | "d2" | "gagg";
type Stage = "raw" | "ingot" | "match";
type Direction = "next" | "previous";
interface Marker { x: number; y: number; }

const detectorOrder: MatchDetector[] = ["d1", "d2", "gagg"];
const ingotTableOrder: IngotGuessTable[] = ["d1Front", "d1Back", "d2Front", "d2Back"];
const stageNames: Stage[] = ["raw", "ingot", "match"];
const matchSelectionFields = {
  d1: "selectedD1Index",
  d2: "selectedD2Index",
  gagg: "selectedGaggIndex",
} as const;
const ingotGuessFields = {
  d1Front: "selectedD1FrontGuessIndex",
  d1Back: "selectedD1BackGuessIndex",
  d2Front: "selectedD2FrontGuessIndex",
  d2Back: "selectedD2BackGuessIndex",
} as const;

const shell = useShellStore();
const chain = useChainStore();
const event = ref<ChainEventPayload | null>(null);
const loading = ref(false);
const error = ref("");
const pidBackground = ref<ChainMatchPidBackgroundPayload | null>(null);
const pidLoading = ref(false);
const pidError = ref("");

const run = computed<number | null>({
  get: () => chain.state.selectedRun ?? shell.state.selectedRun,
  set: (value) => { chain.state.selectedRun = value; },
});
const eventId = computed<number | null>({
  get: () => chain.state.selectedEventId,
  set: (value) => { chain.state.selectedEventId = value; },
});
const stage = computed<Stage>({
  get: () => chain.state.stage,
  set: (value) => { chain.state.stage = value; },
});
const showTheoryLines = computed<boolean>({
  get: () => chain.state.showTheoryLines,
  set: (value) => { chain.state.showTheoryLines = value; },
});
const runs = computed(() => (shell.state.bootstrap?.runs ?? []).map((value) => ({
  title: `Run ${value}`,
  value: Number(value),
})));
const shortcutHint = computed(() => {
  const common = "h / l or Left / Right: event · Shift+H / Shift+L: available event · n / N: stage";
  if (stage.value === "match") return `${common} · 1 / 2 / 3: D1 / D2 / GAGG candidate`;
  if (stage.value === "ingot") return `${common} · 1 / 2 / 3 / 4: D1 front / D1 back / D2 front / D2 back guess`;
  return common;
});

function matchRows(detector: MatchDetector) {
  return event.value?.stages.match.tables[detector].rows ?? [];
}
function matchColumns(detector: MatchDetector): string[] {
  return detector === "gagg" ? ["index", "energy", "time"] : ["frontStrip", "backStrip", "energy", "time"];
}
function columnLabel(column: string): string {
  return column === "frontStrip" ? "front strip" : column === "backStrip" ? "back strip" : column;
}
function selectedIndex(detector: MatchDetector): number | null {
  return chain.state[matchSelectionFields[detector]];
}
function setSelectedIndex(detector: MatchDetector, value: number | null): void {
  chain.state[matchSelectionFields[detector]] = value;
}
const ingotSelections = computed<Record<IngotGuessTable, number | null>>(() => ({
  d1Front: chain.state.selectedD1FrontGuessIndex,
  d1Back: chain.state.selectedD1BackGuessIndex,
  d2Front: chain.state.selectedD2FrontGuessIndex,
  d2Back: chain.state.selectedD2BackGuessIndex,
}));
function ingotGuessRows(table: IngotGuessTable): ChainIngotGuess[] {
  const detector = table.startsWith("d1") ? "d1" : "d2";
  const side = table.endsWith("Front") ? "front" : "back";
  return event.value?.stages.ingot.guesses[detector][side] ?? [];
}
function selectedIngotGuess(table: IngotGuessTable): number | null {
  return chain.state[ingotGuessFields[table]];
}
function setSelectedIngotGuess(table: IngotGuessTable, index: number | null): void {
  chain.state[ingotGuessFields[table]] = index;
}
function normalizedIndex(current: number | null, length: number): number | null {
  if (!length) return null;
  return current !== null && Number.isInteger(current) && current >= 0 && current < length ? current : 0;
}
function normalizeSelections(): void {
  for (const detector of detectorOrder) {
    setSelectedIndex(detector, normalizedIndex(selectedIndex(detector), matchRows(detector).length));
  }
  for (const table of ingotTableOrder) {
    setSelectedIngotGuess(table, normalizedIndex(selectedIngotGuess(table), ingotGuessRows(table).length));
  }
}
function selectCandidate(detector: MatchDetector, index: number): void {
  setSelectedIndex(detector, index);
}
function cycleCandidate(detector: MatchDetector): void {
  const count = matchRows(detector).length;
  if (count) setSelectedIndex(detector, ((selectedIndex(detector) ?? -1) + 1) % count);
}
function selectIngotGuess(table: IngotGuessTable, index: number): void {
  setSelectedIngotGuess(table, index);
}
function cycleIngotGuess(table: IngotGuessTable): void {
  const count = ingotGuessRows(table).length;
  if (count) setSelectedIngotGuess(table, ((selectedIngotGuess(table) ?? -1) + 1) % count);
}
function marker(xDetector: MatchDetector, yDetector: MatchDetector): Marker | null {
  const values = event.value?.stages.match.calibrated;
  const xIndex = selectedIndex(xDetector);
  const yIndex = selectedIndex(yDetector);
  if (!values || xIndex === null || yIndex === null) return null;
  const x = values[xDetector][xIndex];
  const y = values[yDetector][yIndex];
  return x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y) ? null : { x, y };
}
const d1d2Marker = computed(() => marker("d2", "d1"));
const d2gaggMarker = computed(() => marker("gagg", "d2"));

function warmAvailableEvents(): void {
  if (event.value) void prefetchAvailableChainEvents(stage.value).catch(() => undefined);
}
function applyEvent(payload: ChainEventPayload): void {
  event.value = payload;
  chain.state.selectedRun = payload.run;
  chain.state.selectedEventId = payload.eventId;
  normalizeSelections();
  warmAvailableEvents();
  if (stage.value === "match") void loadPidBackground();
}
function message(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}
async function requestEvent(action: () => Promise<ChainEventPayload>, clearOnError = false): Promise<void> {
  if (loading.value) return;
  loading.value = true;
  error.value = "";
  try {
    applyEvent(await action());
  } catch (value) {
    if (clearOnError) event.value = null;
    error.value = message(value);
  } finally {
    loading.value = false;
  }
}
async function load(): Promise<void> {
  if (!Number.isInteger(run.value) || !Number.isInteger(eventId.value) ||
      run.value === null || eventId.value === null || run.value < 0 || eventId.value < 0) {
    error.value = "Select a valid run and nonnegative event id.";
    return;
  }
  const selectedRun = run.value;
  const selectedEventId = eventId.value;
  await requestEvent(async () => {
    const response = await setChainSession(selectedRun, selectedEventId);
    shell.updateSession(response.session);
    return response.event;
  }, true);
}
async function loadPidBackground(): Promise<void> {
  if (pidBackground.value || pidLoading.value) return;
  pidLoading.value = true;
  pidError.value = "";
  try {
    pidBackground.value = await getChainMatchPidBackground();
  } catch (value) {
    pidError.value = message(value);
  } finally {
    pidLoading.value = false;
  }
}
function formatValue(column: string, value: string | number | null | undefined): string | number {
  if (value == null) return "—";
  return (column === "energy" || column === "time") && typeof value === "number"
    ? Math.round(value)
    : value;
}
async function navigate(direction: Direction): Promise<void> {
  if (!event.value) return;
  const targetRun = event.value.run;
  const targetEventId = event.value.eventId + (direction === "next" ? 1 : -1);
  if (targetEventId < 0) {
    error.value = "No previous event before event 0.";
    return;
  }
  await requestEvent(async () => {
    const response = await setChainSession(targetRun, targetEventId);
    shell.updateSession(response.session);
    return response.event;
  });
}
async function navigateAvailable(direction: Direction): Promise<void> {
  if (!event.value) return;
  await requestEvent(() => direction === "next"
    ? nextAvailableChainEvent(stage.value)
    : previousAvailableChainEvent(stage.value));
}
function cycleStage(step: 1 | -1): void {
  const current = stageNames.indexOf(stage.value);
  stage.value = stageNames[(current + step + stageNames.length) % stageNames.length];
}
async function onKeydown(keyEvent: KeyboardEvent): Promise<void> {
  const target = keyEvent.target as HTMLElement | null;
  const tag = target?.tagName.toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select" || target?.isContentEditable ||
      loading.value || !event.value) return;
  const candidate = stage.value === "match" ? detectorOrder[Number(keyEvent.key) - 1] : undefined;
  const guess = stage.value === "ingot" ? ingotTableOrder[Number(keyEvent.key) - 1] : undefined;
  if (!(["n", "N", "h", "H", "l", "L", "ArrowLeft", "ArrowRight"].includes(keyEvent.key) || candidate || guess)) return;
  keyEvent.preventDefault();
  keyEvent.stopImmediatePropagation();
  switch (keyEvent.key) {
    case "n": cycleStage(1); break;
    case "N": cycleStage(-1); break;
    case "H": await navigateAvailable("previous"); break;
    case "L": await navigateAvailable("next"); break;
    case "h":
    case "ArrowLeft": await navigate("previous"); break;
    case "l":
    case "ArrowRight": await navigate("next"); break;
    default:
      if (candidate) cycleCandidate(candidate);
      if (guess) cycleIngotGuess(guess);
  }
}

let mounted = false;
let restoredLoadStarted = false;
let keydownAttached = false;
function restoreSavedEvent(): void {
  if (!mounted || restoredLoadStarted || event.value || run.value === null || eventId.value === null) return;
  restoredLoadStarted = true;
  void load();
}
function attachKeydown(): void {
  if (keydownAttached) return;
  window.addEventListener("keydown", onKeydown, true);
  keydownAttached = true;
}
function detachKeydown(): void {
  if (!keydownAttached) return;
  window.removeEventListener("keydown", onKeydown, true);
  keydownAttached = false;
}
watch(() => [run.value, eventId.value], restoreSavedEvent, { flush: "post" });
watch(stage, (value) => {
  if (value === "match" && event.value) void loadPidBackground();
  warmAvailableEvents();
});
onMounted(() => { mounted = true; attachKeydown(); restoreSavedEvent(); });
onActivated(() => { attachKeydown(); restoreSavedEvent(); });
onDeactivated(detachKeydown);
onBeforeUnmount(detachKeydown);
</script>
