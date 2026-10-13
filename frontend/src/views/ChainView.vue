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
import { getChainMatchPidBackground, nextAvailableChainEvent, prefetchAvailableChainEvents, previousAvailableChainEvent, setChainSession } from "../api";
import { useShellStore } from "../stores/shell";
import { useChainStore } from "../stores/chain";
import type { ChainEventPayload, ChainIngotGuess, ChainMatchPidBackgroundPayload } from "../types";

type MatchDetector = "d1" | "d2" | "gagg";
const detectorOrder: MatchDetector[] = ["d1", "d2", "gagg"];
const shell = useShellStore(); const chain = useChainStore();
const event = ref<ChainEventPayload | null>(null); const loading = ref(false); const error = ref("");
const pidBackground = ref<ChainMatchPidBackgroundPayload | null>(null); const pidLoading = ref(false); const pidError = ref("");
const run = computed<number | null>({ get: () => chain.state.selectedRun ?? shell.state.selectedRun, set: (value) => { chain.state.selectedRun = value; } });
const eventId = computed<number | null>({ get: () => chain.state.selectedEventId, set: (value) => { chain.state.selectedEventId = value; } });
const stage = computed<"raw" | "ingot" | "match">({ get: () => chain.state.stage, set: (value) => { chain.state.stage = value; } });
const showTheoryLines = computed<boolean>({ get: () => chain.state.showTheoryLines, set: (value) => { chain.state.showTheoryLines = value; } });
const stageNames = ["raw", "ingot", "match"] as const;
const runs = computed(() => (shell.state.bootstrap?.runs || []).map((value) => ({ title: `Run ${value}`, value: Number(value) })));
const shortcutHint = computed(() => stage.value === "match" ? "h / l or Left / Right: event · Shift+H / Shift+L: available event · n / N: stage · 1 / 2 / 3: D1 / D2 / GAGG candidate" : stage.value === "ingot" ? "h / l or Left / Right: event · Shift+H / Shift+L: available event · n / N: stage · 1 / 2 / 3 / 4: D1 front / D1 back / D2 front / D2 back guess" : "h / l or Left / Right: event · Shift+H / Shift+L: available event · n / N: stage");
function matchRows(detector: MatchDetector) { return event.value?.stages.match.tables[detector].rows || []; }
function matchColumns(detector: MatchDetector): string[] { return detector === "gagg" ? ["index", "energy", "time"] : ["frontStrip", "backStrip", "energy", "time"]; }
function columnLabel(column: string): string { return column === "frontStrip" ? "front strip" : column === "backStrip" ? "back strip" : column; }
function selectedIndex(detector: MatchDetector): number | null { return detector === "d1" ? chain.state.selectedD1Index : detector === "d2" ? chain.state.selectedD2Index : chain.state.selectedGaggIndex; }
function setSelectedIndex(detector: MatchDetector, value: number | null): void { if (detector === "d1") chain.state.selectedD1Index = value; else if (detector === "d2") chain.state.selectedD2Index = value; else chain.state.selectedGaggIndex = value; }
const ingotGuessFields: Record<IngotGuessTable, keyof typeof chain.state> = { d1Front: "selectedD1FrontGuessIndex", d1Back: "selectedD1BackGuessIndex", d2Front: "selectedD2FrontGuessIndex", d2Back: "selectedD2BackGuessIndex" };
const ingotSelections = computed<Record<IngotGuessTable, number | null>>(() => ({ d1Front: chain.state.selectedD1FrontGuessIndex, d1Back: chain.state.selectedD1BackGuessIndex, d2Front: chain.state.selectedD2FrontGuessIndex, d2Back: chain.state.selectedD2BackGuessIndex }));
function ingotGuessRows(table: IngotGuessTable): ChainIngotGuess[] { const detector = table.startsWith("d1") ? "d1" : "d2"; const side = table.endsWith("Front") ? "front" : "back"; return event.value?.stages.ingot.guesses[detector][side] ?? []; }
function selectedIngotGuess(table: IngotGuessTable): number | null { return chain.state[ingotGuessFields[table]] as number | null; }
function setSelectedIngotGuess(table: IngotGuessTable, index: number | null): void { (chain.state[ingotGuessFields[table]] as number | null) = index; }
function normalizeIngotSelections(): void { for (const table of Object.keys(ingotGuessFields) as IngotGuessTable[]) { const rows = ingotGuessRows(table); const current = selectedIngotGuess(table); setSelectedIngotGuess(table, rows.length ? current === null || current >= rows.length ? 0 : current : null); } }
function normalizeSelections(): void { for (const detector of detectorOrder) { const rows = matchRows(detector); const current = selectedIndex(detector); setSelectedIndex(detector, rows.length ? current === null || current >= rows.length ? 0 : current : null); } normalizeIngotSelections(); }
function selectCandidate(detector: MatchDetector, index: number): void { setSelectedIndex(detector, index); }
function cycleCandidate(detector: MatchDetector): void { const rows = matchRows(detector); if (!rows.length) return; setSelectedIndex(detector, ((selectedIndex(detector) ?? -1) + 1) % rows.length); }
function selectIngotGuess(table: IngotGuessTable, index: number): void { setSelectedIngotGuess(table, index); }
function cycleIngotGuess(table: IngotGuessTable): void { const rows = ingotGuessRows(table); if (!rows.length) return; setSelectedIngotGuess(table, ((selectedIngotGuess(table) ?? -1) + 1) % rows.length); }
const calibrated = computed(() => event.value?.stages.match.calibrated);
const d1d2Marker = computed(() => { const values = calibrated.value; const d1 = selectedIndex("d1"), d2 = selectedIndex("d2"); if (!values || d1 === null || d2 === null) return null; const x = values.d2[d2], y = values.d1[d1]; return x === null || y === null ? null : { x, y }; });
const d2gaggMarker = computed(() => { const values = calibrated.value; const d2 = selectedIndex("d2"), gagg = selectedIndex("gagg"); if (!values || d2 === null || gagg === null) return null; const x = values.gagg[gagg], y = values.d2[d2]; return x === null || y === null ? null : { x, y }; });
function warmAvailableEvents(): void { if (!event.value) return; void prefetchAvailableChainEvents(stage.value).catch(() => undefined); }
function applyEvent(payload: ChainEventPayload): void { event.value = payload; chain.state.selectedRun = payload.run; chain.state.selectedEventId = payload.eventId; normalizeSelections(); warmAvailableEvents(); }
async function load(): Promise<void> { if (run.value === null || eventId.value === null) { error.value = "Select both a run and event id."; return; } loading.value = true; error.value = ""; try { const payload = await setChainSession(run.value, eventId.value); applyEvent(payload.event); shell.updateSession(payload.session); } catch (value) { event.value = null; error.value = value instanceof Error ? value.message : String(value); } finally { loading.value = false; } }
async function loadPidBackground(): Promise<void> { if (pidBackground.value || pidLoading.value) return; pidLoading.value = true; pidError.value = ""; try { pidBackground.value = await getChainMatchPidBackground(); } catch (value) { pidError.value = value instanceof Error ? value.message : String(value); } finally { pidLoading.value = false; } }
function formatValue(column: string, value: string | number | null | undefined): string | number { if (value === null || value === undefined) return "—"; return (column === "energy" || column === "time") && typeof value === "number" ? Math.round(value) : value; }
async function navigate(direction: "next" | "previous"): Promise<void> { if (!event.value || loading.value) return; const targetRun = event.value.run; const targetEventId = event.value.eventId + (direction === "next" ? 1 : -1); loading.value = true; error.value = ""; try { const payload = await setChainSession(targetRun, targetEventId); applyEvent(payload.event); shell.updateSession(payload.session); } catch (value) { error.value = value instanceof Error ? value.message : String(value); } finally { loading.value = false; } }
async function navigateAvailable(direction: "next" | "previous"): Promise<void> { if (!event.value || loading.value) return; loading.value = true; error.value = ""; try { applyEvent(await (direction === "next" ? nextAvailableChainEvent(stage.value) : previousAvailableChainEvent(stage.value))); } catch (value) { error.value = value instanceof Error ? value.message : String(value); } finally { loading.value = false; } }
function cycleStage(step: 1 | -1): void { const current = stageNames.indexOf(stage.value); stage.value = stageNames[(current + step + stageNames.length) % stageNames.length]; }
async function onKeydown(keyEvent: KeyboardEvent): Promise<void> { const tag = (keyEvent.target as HTMLElement | null)?.tagName?.toLowerCase(); if (tag === "input" || tag === "textarea" || tag === "select" || loading.value || !event.value) return; const stageShortcut = stage.value === "match" ? ["1", "2", "3"].includes(keyEvent.key) : stage.value === "ingot" && ["1", "2", "3", "4"].includes(keyEvent.key); const chainKey = keyEvent.key === "n" || keyEvent.key === "N" || keyEvent.key === "h" || keyEvent.key === "l" || keyEvent.key === "H" || keyEvent.key === "L" || keyEvent.key === "ArrowLeft" || keyEvent.key === "ArrowRight" || stageShortcut; if (!chainKey) return; keyEvent.preventDefault(); keyEvent.stopImmediatePropagation(); if (keyEvent.key === "n") { cycleStage(1); } else if (keyEvent.key === "N") { cycleStage(-1); } else if (keyEvent.key === "H") { await navigateAvailable("previous"); } else if (keyEvent.key === "L") { await navigateAvailable("next"); } else if (keyEvent.key === "h" || keyEvent.key === "ArrowLeft") { await navigate("previous"); } else if (keyEvent.key === "l" || keyEvent.key === "ArrowRight") { await navigate("next"); } else if (stage.value === "match") { cycleCandidate(detectorOrder[Number(keyEvent.key) - 1]); } else if (stage.value === "ingot") { cycleIngotGuess((["d1Front", "d1Back", "d2Front", "d2Back"] as IngotGuessTable[])[Number(keyEvent.key) - 1]); } }
let mounted = false; let restoredLoadStarted = false; let keydownAttached = false;
function restoreSavedEvent(): void { if (!mounted || restoredLoadStarted || event.value || run.value === null || eventId.value === null) return; restoredLoadStarted = true; void load(); }
function attachKeydown(): void { if (keydownAttached) return; window.addEventListener("keydown", onKeydown, true); keydownAttached = true; }
function detachKeydown(): void { if (!keydownAttached) return; window.removeEventListener("keydown", onKeydown, true); keydownAttached = false; }
watch(() => [run.value, eventId.value], restoreSavedEvent, { flush: "post" });
watch(stage, (value) => { if (value === "match") void loadPidBackground(); warmAvailableEvents(); }, { immediate: true });
onMounted(() => { mounted = true; attachKeydown(); restoreSavedEvent(); }); onActivated(() => { attachKeydown(); restoreSavedEvent(); }); onDeactivated(detachKeydown); onBeforeUnmount(detachKeydown);
</script>
