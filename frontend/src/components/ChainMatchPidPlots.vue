<template>
  <section class="chain-pid-section" aria-label="PID plots">
    <div v-if="background?.theoryLines.length" class="chain-theory-toolbar">
      <v-btn
        :aria-pressed="showTheory"
        class="chain-theory-toggle"
        :prepend-icon="showTheory ? 'mdi-eye-off-outline' : 'mdi-eye-outline'"
        size="small"
        variant="tonal"
        @click="emit('update:showTheory', !showTheory)"
      >
        {{ showTheory ? 'Hide theory' : 'Show theory' }}
      </v-btn>
      <div v-if="showTheory" class="chain-theory-legend" aria-label="Theory line legend">
        <span v-for="(line, index) in background.theoryLines" :key="line.particle" class="chain-theory-legend-item">
          <i :style="{ backgroundColor: theoryColor(index) }" aria-hidden="true" />
          {{ line.particle }}
        </span>
      </div>
    </div>
    <div class="chain-pid-grid">
      <div class="chain-pid-plot-shell">
        <div ref="d1d2Root" class="chain-pid-plot" />
        <p v-if="!d1d2Marker" class="chain-pid-empty">Select D1 and D2 candidates to place a marker.</p>
      </div>
      <div class="chain-pid-plot-shell">
        <div ref="d2gaggRoot" class="chain-pid-plot" />
        <p v-if="!d2gaggMarker" class="chain-pid-empty">No calibrated D2–GAGG candidate pair.</p>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch } from "vue";
import { loadPlotly } from "../lib/plotly";
import { plotTheme } from "../lib/plot_theme";
import { useAppearanceStore } from "../stores/appearance";
import type { ChainMatchPidBackgroundPayload, ChainPidHistogram } from "../types";

interface Marker { x: number; y: number; }
const props = defineProps<{ background: ChainMatchPidBackgroundPayload | null; d1d2Marker: Marker | null; d2gaggMarker: Marker | null; showTheory: boolean; }>();
const emit = defineEmits<{ "update:showTheory": [value: boolean]; }>();
const appearance = useAppearanceStore();
const d1d2Root = ref<HTMLElement | null>(null);
const d2gaggRoot = ref<HTMLElement | null>(null);
let plotsReady = false;
// ROOT's familiar spectrum progression, with an intentionally empty zero bin.
const rootBirdColorscale = [
  [0, "rgba(0, 0, 0, 0)"],
  [0.000001, "#071a72"],
  [0.12, "#0759c7"],
  [0.28, "#00a9e6"],
  [0.46, "#00c99b"],
  [0.64, "#9ddd22"],
  [0.80, "#ffe000"],
  [0.92, "#ff8500"],
  [1, "#c51b28"],
] as const;
const rootBirdDarkColorscale = [
  [0, "rgba(0, 0, 0, 0)"],
  [0.000001, "#287dff"],
  [0.12, "#27b6ff"],
  [0.28, "#51d8ff"],
  [0.46, "#1ce3b0"],
  [0.64, "#b4f35e"],
  [0.80, "#fff06a"],
  [0.92, "#ffb35c"],
  [1, "#ff6877"],
] as const;
const theoryColors = ["#6a1b9a", "#c62828", "#1565c0", "#2e7d32", "#e65100", "#00838f", "#ad1457"] as const;
const theoryDarkColors = ["#d5a6ff", "#ff9ca5", "#80c7ff", "#7ae0a2", "#ffc37a", "#6ee7e7", "#f5a4d0"] as const;
type TheoryPlot = "d1d2" | "d2gagg";
function theoryColor(index: number): string {
  const colors = appearance.state.mode === "dark" ? theoryDarkColors : theoryColors;
  return colors[index % colors.length];
}
function theorySeries(plot: TheoryPlot): Record<string, unknown>[] {
  return props.background!.theoryLines.map((line, index) => ({
    type: "scatter",
    mode: "lines",
    x: line[plot].x,
    y: line[plot].y,
    name: line.particle,
    line: { color: theoryColor(index), width: 2.25 },
    visible: props.showTheory,
    hovertemplate: `${line.particle}<br>%{x:.2f}, %{y:.2f}<extra></extra>`,
  }));
}
async function renderPlot(root: HTMLElement | null, title: string, xTitle: string, yTitle: string, histogram: ChainPidHistogram, plot: TheoryPlot, marker: Marker | null): Promise<void> {
  if (!root) return;
  const Plotly = await loadPlotly(); const theme = plotTheme();
  const colorscale = appearance.state.mode === "dark" ? rootBirdDarkColorscale : rootBirdColorscale;
  const series: Record<string, unknown>[] = [
    { type: "heatmap", x: histogram.x, y: histogram.y, z: histogram.z, colorscale, zmin: 0, zsmooth: false, hovertemplate: `${xTitle}: %{x:.1f}<br>${yTitle}: %{y:.1f}<br>count: %{z}<extra></extra>`, colorbar: { title: { text: "count", side: "right" }, thickness: 12 } },
    ...theorySeries(plot),
    { type: "scatter", mode: "markers", x: marker ? [marker.x] : [], y: marker ? [marker.y] : [], name: "Selected", marker: { color: theme.fit, size: 11, line: { color: theme.surface, width: 2 } }, hovertemplate: `${xTitle}: %{x:.2f}<br>${yTitle}: %{y:.2f}<extra>Selected</extra>` },
  ];
  await Plotly.react(root, series, { title: { text: title, x: 0.03, xanchor: "left", font: { size: 15 } }, margin: { t: 48, r: 58, b: 50, l: 58 }, showlegend: false, xaxis: { title: xTitle, zeroline: false }, yaxis: { title: yTitle, zeroline: false } }, { displayModeBar: false, responsive: true });
}
async function renderBackground(): Promise<void> { plotsReady = false; await nextTick(); if (!props.background) return; await Promise.all([renderPlot(d1d2Root.value, "D1 energy vs D2 energy", "D2 energy", "D1 energy", props.background.d1d2, "d1d2", props.d1d2Marker), renderPlot(d2gaggRoot.value, "D2 energy vs GAGG energy", "GAGG energy", "D2 energy", props.background.d2gagg, "d2gagg", props.d2gaggMarker)]); plotsReady = true; await updateMarkers(); }
async function updateMarker(root: HTMLElement | null, marker: Marker | null): Promise<void> { if (!root || !plotsReady || !props.background) return; const Plotly = await loadPlotly(); const theoryCount = props.background.theoryLines.length; await Plotly.restyle(root, { x: [marker ? [marker.x] : []], y: [marker ? [marker.y] : []] }, [theoryCount + 1]); }
async function updateMarkers(): Promise<void> { await Promise.all([updateMarker(d1d2Root.value, props.d1d2Marker), updateMarker(d2gaggRoot.value, props.d2gaggMarker)]); }
async function updateTheoryVisibility(): Promise<void> {
  if (!plotsReady || !props.background || !props.background.theoryLines.length) return;
  const Plotly = await loadPlotly();
  const theoryIndexes = props.background.theoryLines.map((_, index) => index + 1);
  await Promise.all([
    d1d2Root.value ? Plotly.restyle(d1d2Root.value, { visible: props.showTheory }, theoryIndexes) : Promise.resolve(),
    d2gaggRoot.value ? Plotly.restyle(d2gaggRoot.value, { visible: props.showTheory }, theoryIndexes) : Promise.resolve(),
  ]);
}
watch(() => [props.background, appearance.state.mode], () => { void renderBackground(); }, { immediate: true });
watch(() => props.showTheory, () => { void updateTheoryVisibility(); });
watch(() => [props.d1d2Marker, props.d2gaggMarker], () => { void updateMarkers(); });
onBeforeUnmount(async () => { const Plotly = await loadPlotly(); if (d1d2Root.value) Plotly.purge(d1d2Root.value); if (d2gaggRoot.value) Plotly.purge(d2gaggRoot.value); });
</script>
