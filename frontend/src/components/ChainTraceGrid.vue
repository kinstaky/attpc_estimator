<template>
  <div class="chain-trace-grid">
    <article v-for="trace in traces" :key="`${trace.detector}-${trace.side}-${trace.index}`" class="chain-trace-panel">
      <p class="page-kicker">{{ traceTitle(trace) }}</p>
      <div :ref="(element) => setRoot(trace, element)" class="chain-trace-plot" />
    </article>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, watch } from "vue";
import { loadPlotly } from "../lib/plotly";
import { plotTheme } from "../lib/plot_theme";
import type { ChainTrace } from "../types";

const props = defineProps<{ traces: ChainTrace[] }>();
const roots = new Map<string, HTMLElement>();

function key(trace: ChainTrace): string { return `${trace.detector}-${trace.side}-${trace.index}`; }
function traceTitle(trace: ChainTrace): string {
  if (trace.detector === "gagg") return `GAGG ${trace.index}`;
  const detector = trace.side.startsWith("upstream") ? "D1" : "D2";
  const side = trace.side.endsWith("front") ? "front" : "back";
  return `${detector} ${side} · strip ${trace.index}`;
}
function setRoot(trace: ChainTrace, element: unknown): void {
  const traceKey = key(trace);
  if (element instanceof HTMLElement) roots.set(traceKey, element);
  else roots.delete(traceKey);
}

async function render(): Promise<void> {
  await nextTick();
  const Plotly = await loadPlotly();
  const theme = plotTheme();
  for (const trace of props.traces) {
    const root = roots.get(key(trace));
    if (!root) continue;
    const series: Record<string, unknown>[] = [{ x: trace.raw.map((_, i) => i), y: trace.raw, type: "scatter", mode: "lines", name: "Raw", line: { color: theme.raw, width: 1.5 } }];
    if (trace.fit?.fitted) series.push({ x: trace.fit.fitted.map((_, i) => i), y: trace.fit.fitted, type: "scatter", mode: "lines", name: "Free-τ fit", line: { color: theme.fit, width: 2 } });
    Plotly.react(root, series, { margin: { t: 12, r: 12, b: 34, l: 48 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", showlegend: Boolean(trace.fit?.fitted), legend: { orientation: "h", y: 1.16, font: { size: 10 } }, xaxis: { title: "sample", fixedrange: false }, yaxis: { title: "ADC", fixedrange: false } }, { displayModeBar: false, responsive: true });
  }
}

watch(() => props.traces, () => { void render(); }, { deep: true, immediate: true });
onBeforeUnmount(async () => { const Plotly = await loadPlotly(); roots.forEach((root) => Plotly.purge(root)); });
</script>
