<template>
  <section class="chain-ingot-stage" aria-label="ingot event data">
    <section class="chain-table-shell" aria-label="ingot records">
      <div class="chain-table-label"><span>ingot records</span><strong>{{ stage.rows.length }}</strong></div>
      <v-table class="chain-table">
        <thead><tr><th v-for="column in stage.columns" :key="column" :class="{ 'chain-table__number': isNumericColumn(column) }">{{ column }}</th></tr></thead>
        <tbody><tr v-for="(row, index) in stage.rows" :key="index"><td v-for="column in stage.columns" :key="column" :class="{ 'chain-table__number': isNumericColumn(column) }"><span v-if="column === 'detector'" class="chain-detector-tag" :class="detectorClass(row[column])">{{ row[column] }}</span><span v-else-if="column === 'side' && row[column]" class="chain-side-tag" :class="sideClass(row[column])">{{ row[column] }}</span><template v-else>{{ formatValue(column, row[column]) }}</template></td></tr></tbody>
      </v-table>
    </section>

    <section class="chain-ingot-guess-stage" aria-label="silicon strip-combination guesses">
      <header class="chain-ingot-review-heading">
        <div><p class="page-kicker">Silicon combination review</p><h3>Candidate strip-energy guesses</h3></div>
        <v-text-field
          class="chain-ingot-threshold"
          density="compact"
          hide-details
          label="Distance threshold"
          min="0"
          step="1"
          type="number"
          :model-value="threshold"
          @update:model-value="updateThreshold"
        />
      </header>

      <div class="chain-ingot-guess-grid">
        <article v-for="table in guessTables" :key="table.key" class="chain-ingot-guess-table-shell" :class="`chain-ingot-guess-table-shell--${table.detector}`">
          <header><span>{{ table.label }}</span><strong>{{ table.rows.length }}</strong></header>
          <v-table class="chain-table chain-ingot-guess-table">
            <thead><tr><th>strips</th><th>type</th><th class="chain-table__number">time</th><th class="chain-table__number">energy</th><th class="chain-table__number">weight</th></tr></thead>
            <tbody>
              <tr v-for="(guess, index) in table.rows" :key="index" :class="{ 'chain-ingot-guess-row--selected': selected[table.key] === index }" tabindex="0" @click="emit('selectGuess', table.key, index)" @keydown.enter.prevent="emit('selectGuess', table.key, index)" @keydown.space.prevent="emit('selectGuess', table.key, index)">
                <td>{{ formatStrips(guess.strips) }}</td><td><span class="chain-guess-type">{{ guess.type }}</span></td><td class="chain-table__number">{{ formatGuessValue(guess.time) }}</td><td class="chain-table__number">{{ formatGuessValue(guess.energy) }}</td><td class="chain-table__number">{{ formatGuessValue(guess.weight) }}</td>
              </tr>
            </tbody>
          </v-table>
          <p v-if="!table.rows.length" class="chain-match-empty">No guesses</p>
        </article>
      </div>
    </section>

    <section class="chain-ingot-matrix-grid" aria-label="front and back guess pairing matrices">
      <article v-for="detector in detectors" :key="detector" class="chain-ingot-matrix-panel" :class="`chain-ingot-matrix-panel--${detector}`">
        <header><div><p class="page-kicker">{{ detector.toUpperCase() }} pairing</p><h3>Front × back candidate matrix</h3></div><span>Rows: front · Columns: back</span></header>
        <div v-if="hasMatrix(detector)" class="chain-ingot-matrix-pair">
          <MatrixTable title="Distance" :detector="detector" :values="stage.matrices[detector].distances" :selected-front="selected[`${detector}Front`]" :selected-back="selected[`${detector}Back`]" :best="bestPair(detector)" />
          <MatrixTable title="Combined energy" :detector="detector" :values="stage.matrices[detector].energies" :selected-front="selected[`${detector}Front`]" :selected-back="selected[`${detector}Back`]" :best="bestPair(detector)" />
        </div>
        <p v-else class="chain-match-empty">A front and back guess are both required to form a matrix.</p>
      </article>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, defineComponent, h, type PropType } from "vue";
import type { ChainIngotGuess, ChainIngotStage } from "../types";

type Detector = "d1" | "d2";
export type IngotGuessTable = "d1Front" | "d1Back" | "d2Front" | "d2Back";
type SelectionMap = Record<IngotGuessTable, number | null>;
interface Pair { row: number; column: number; }

const props = defineProps<{ stage: ChainIngotStage; threshold: number; selected: SelectionMap; }>();
const emit = defineEmits<{ updateThreshold: [value: number]; selectGuess: [table: IngotGuessTable, index: number]; }>();
const detectors: Detector[] = ["d1", "d2"];
const guessTables = computed(() => [
  { key: "d1Front" as const, detector: "d1" as const, label: "D1 · front", rows: props.stage.guesses.d1.front },
  { key: "d1Back" as const, detector: "d1" as const, label: "D1 · back", rows: props.stage.guesses.d1.back },
  { key: "d2Front" as const, detector: "d2" as const, label: "D2 · front", rows: props.stage.guesses.d2.front },
  { key: "d2Back" as const, detector: "d2" as const, label: "D2 · back", rows: props.stage.guesses.d2.back },
]);
function updateThreshold(value: string | number): void { if (value === "") return; const threshold = Number(value); if (Number.isFinite(threshold) && threshold >= 0) emit("updateThreshold", threshold); }
function isNumericColumn(column: string): boolean { return column !== "detector" && column !== "side"; }
function formatValue(column: string, value: string | number | null | undefined): string | number { if (value === null || value === undefined) return "—"; return (column === "energy" || column === "time") && typeof value === "number" ? Math.round(value) : value; }
function formatGuessValue(value: number | null): string { return value === null ? "—" : Number(value.toFixed(2)).toString(); }
function formatStrips(strips: number[]): string { return strips.length ? strips.join(" · ") : "—"; }
function detectorClass(value: string | number | null | undefined): string { return `chain-detector-tag--${String(value || "").toLowerCase()}`; }
function sideClass(value: string | number | null | undefined): string { return `chain-side-tag--${String(value || "").toLowerCase()}`; }
function hasMatrix(detector: Detector): boolean { const matrix = props.stage.matrices[detector].distances; return matrix.length > 0 && matrix.some((row) => row.length > 0); }
function bestPair(detector: Detector): Pair | null { let best: Pair | null = null; let value = Number.POSITIVE_INFINITY; for (const [row, cells] of props.stage.matrices[detector].distances.entries()) for (const [column, distance] of cells.entries()) if (distance !== null && distance <= props.threshold && distance < value) { value = distance; best = { row, column }; } return best; }

const MatrixTable = defineComponent({
  name: "ChainIngotMatrixTable",
  props: {
    title: { type: String, required: true },
    detector: { type: String as PropType<Detector>, required: true },
    values: { type: Array as PropType<Array<Array<number | null>>>, required: true },
    selectedFront: { type: Number as PropType<number | null>, default: null },
    selectedBack: { type: Number as PropType<number | null>, default: null },
    best: { type: Object as PropType<Pair | null>, default: null },
  },
  setup(matrixProps) {
    const format = (value: number | null): string => value === null ? "—" : Number(value.toFixed(2)).toString();
    return () => h("section", { class: "chain-ingot-matrix-shell" }, [
      h("h4", matrixProps.title),
      h("div", { class: "chain-ingot-matrix-scroll" }, [
        h("table", { class: "chain-ingot-matrix" }, [
          h("thead", [h("tr", [h("th", "F \\ B"), ...(matrixProps.values[0] || []).map((_, column) => h("th", { class: { "chain-ingot-matrix__column--selected": matrixProps.selectedBack === column } }, `B${column + 1}`))])]),
          h("tbody", matrixProps.values.map((row, rowIndex) => h("tr", { class: { "chain-ingot-matrix__row--selected": matrixProps.selectedFront === rowIndex } }, [
            h("th", { class: { "chain-ingot-matrix__row--selected": matrixProps.selectedFront === rowIndex } }, `F${rowIndex + 1}`),
            ...row.map((value, column) => h("td", { class: { "chain-ingot-matrix__cell--row-selected": matrixProps.selectedFront === rowIndex, "chain-ingot-matrix__cell--column-selected": matrixProps.selectedBack === column, "chain-ingot-matrix__cell--selected": matrixProps.selectedFront === rowIndex && matrixProps.selectedBack === column, "chain-ingot-matrix__cell--best": matrixProps.best?.row === rowIndex && matrixProps.best?.column === column } }, format(value))),
          ]))),
        ]),
      ]),
    ]);
  },
});
</script>
