<template>
  <v-row dense>
    <v-col v-for="plot in plots" :key="plot.title" cols="12" md="6">
      <div ref="roots" class="trace-plot result-plot"></div>
    </v-col>
  </v-row>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { loadPlotly } from "../lib/plotly";

const props = defineProps({ payload: { type: Object, required: true } });
const roots = ref([]);
const plots = [
  { title: "Model peak vs measured peak", kind: "peak" },
  { title: "Tau distribution", kind: "tau" },
  { title: "Peak time vs t0", kind: "time" },
  { title: "t0 distribution", kind: "t0" },
];

function histogramBars(values, bins = 100) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) return { x: [], y: [], width: 1 };
  let low = Math.min(...finite); let high = Math.max(...finite);
  if (low === high) return { x: [low], y: [finite.length], width: 1 };
  const width = (high - low) / bins;
  const counts = Array(bins).fill(0);
  finite.forEach((value) => { counts[Math.min(bins - 1, Math.floor((value - low) / width))] += 1; });
  return { x: counts.map((_, i) => low + (i + 0.5) * width), y: counts, width };
}

async function render() {
  await nextTick();
  const Plotly = await loadPlotly();
  roots.value.forEach((root, index) => {
    const plot = plots[index];
    const n = props.payload.n || []; const tau = props.payload.tau || []; const A = props.payload.A || [];
    const data = plot.kind === "peak" ? [{ type: "scatter", mode: "markers", x: props.payload.peakAmplitude || [], y: A.map((value, i) => value * Math.pow(n[i] * tau[i], n[i]) * Math.exp(-n[i])), marker: { color: "#174f40" }, name: "fit peak" }]
      : plot.kind === "time" ? [{ type: "scatter", mode: "markers", x: props.payload.peakTime || [], y: props.payload.t0 || [], marker: { color: "#6a4c93" } }]
      : [{ type: "bar", ...histogramBars(props.payload[plot.kind] || []), marker: { color: plot.kind === "tau" ? "#3b5ba9" : "#b46b2f" } }];
    Plotly.react(root, data, {
      title: { text: plot.title, x: 0.02, xanchor: "left" }, barmode: "overlay",
      margin: { t: 48, r: 24, b: 48, l: 56 }, paper_bgcolor: "#fffdf8", plot_bgcolor: "#fffdf8",
      xaxis: { title: plot.kind === "peak" ? "Measured peak amplitude" : plot.kind === "time" ? "Peak time" : undefined, zeroline: false, gridcolor: "#e7dfcf" }, yaxis: { title: plot.kind === "peak" ? "Model peak amplitude" : plot.kind === "time" ? "t0" : "Count", gridcolor: "#e7dfcf" },
      showlegend: false,
    }, { displayModeBar: false, responsive: true });
  });
}
onMounted(() => void render());
watch(() => props.payload, () => void render(), { deep: true });
onBeforeUnmount(() => { void loadPlotly().then((Plotly) => roots.value.forEach((root) => Plotly.purge(root))); });
</script>
