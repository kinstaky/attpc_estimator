<template>
  <v-row dense>
    <v-col v-for="plot in plots" :key="plot.key" cols="12"><div ref="roots" class="trace-plot result-plot"></div></v-col>
  </v-row>
</template>
<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { loadPlotly } from "../lib/plotly";
const props = defineProps<{ plots: any[] }>(); const roots = ref<HTMLElement[]>([]);
async function render() { await nextTick(); const Plotly = await loadPlotly(); roots.value.forEach((root, index) => { const plot = props.plots[index]; const z = (plot.histogram || []).map((row: number[]) => row.map((value) => Math.log10(value + 1))); Plotly.react(root, [{ type: "heatmap", x: plot.xBinCenters, y: plot.yBinCenters, z, colorscale: "Viridis", colorbar: { title: "log10(count + 1)" }, hovertemplate: "sample %{x}<br>ADC %{y}<br>log count %{z:.2f}<extra></extra>" }], { title: { text: `${plot.title} (${plot.traceCount || 0} traces)`, x: 0.02, xanchor: "left" }, margin: { t: 48, r: 24, b: 48, l: 56 }, paper_bgcolor: "#fffdf8", plot_bgcolor: "#fffdf8", xaxis: { title: "Sample" }, yaxis: { title: "Raw ADC" } }, { displayModeBar: false, responsive: true }); }); }
onMounted(() => void render()); watch(() => props.plots, () => void render(), { deep: true }); onBeforeUnmount(() => { void loadPlotly().then((Plotly) => roots.value.forEach((root) => Plotly.purge(root))); });
</script>
