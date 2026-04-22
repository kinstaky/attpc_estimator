<template>
  <div ref="root" class="trace-plot result-plot pointcloud-plot"></div>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import { loadPlotly } from "../lib/plotly";

const HALF_EDGE = 4.5;
const HEIGHT = HALF_EDGE * (3 ** 0.5);
const PAD_GAP_SCALE = 0.92;
const DEFAULT_3D_RANGES = {
  x: [-300, 300],
  y: [0, 1000],
  z: [-300, 300],
};
const DEFAULT_3D_CAMERA = {
  eye: { x: 1.45, y: 1.15, z: 0.58 },
  up: { x: 0, y: 0, z: 1 },
  center: { x: 0, y: 0, z: 0 },
};

const props = defineProps({
  plotType: { type: String, required: true },
  layoutMode: { type: String, default: "2x2" },
  hits: { type: Array, default: () => [] },
  mappingPads: { type: Array, default: () => [] },
  selectedTraceIds: { type: Array, default: () => [] },
  traces: { type: Array, default: () => [] },
  xyRange: { type: Object, default: null },
  projectedRange: { type: Object, default: null },
  camera: { type: Object, default: null },
});

const emit = defineEmits(["toggle-traces", "update-xy-range", "update-projected-range", "update-camera"]);

const root = ref(null);
let resizeObserver = null;
let resizeFrame = 0;

function isPadPlot() {
  return props.plotType === "pads-z" || props.plotType === "pads-amplitude";
}

function isTwoDimensionalHitPlot() {
  return props.plotType === "hits-2d-z"
    || props.plotType === "hits-2d-amplitude"
    || props.plotType === "hits-2d-pca-amplitude"
    || isPadPlot();
}

function isProjectedTwoDimensionalPlot() {
  return props.plotType === "hits-2d-pca-amplitude";
}

function isThreeDimensionalPlot() {
  return props.plotType === "hits-3d-amplitude";
}

function selectedTraceIdSet() {
  return new Set((props.selectedTraceIds || []).map((value) => Number(value)));
}

function baseLayout() {
  return {
    margin: { t: 48, r: 20, b: 48, l: 56 },
    paper_bgcolor: "#fffdf8",
    plot_bgcolor: "#fffdf8",
    font: {
      family: "ui-sans-serif, system-ui, sans-serif",
      color: "#222",
    },
  };
}

function config() {
  return {
    displayModeBar: "hover",
    responsive: true,
  };
}

function threeDimensionalAspectRatio() {
  if (props.layoutMode === "1x2") {
    return { x: 1, y: 1, z: 1 };
  }
  return { x: 1, y: 2, z: 1 };
}

function axisRange(value) {
  return props.xyRange?.[value] || undefined;
}

function projectedAxisRange(value) {
  return props.projectedRange?.[value] || undefined;
}

function twoDimensionalLayout(title, xLabel = "x", yLabel = "y", rangeSource = "xy") {
  const resolvedRange = rangeSource === "projected" ? projectedAxisRange : axisRange;
  const hasRange = rangeSource === "projected" ? props.projectedRange : props.xyRange;
  return {
    ...baseLayout(),
    title: { text: title, x: 0.02, xanchor: "left" },
    xaxis: {
      title: xLabel,
      zeroline: false,
      gridcolor: "#e7dfcf",
      range: resolvedRange("x"),
      autorange: hasRange ? false : true,
    },
    yaxis: {
      title: yLabel,
      zeroline: false,
      gridcolor: "#e7dfcf",
      range: resolvedRange("y"),
      autorange: hasRange ? false : true,
    },
  };
}

function aggregatePads(metricKey) {
  const byPadId = new Map();
  for (const hit of props.hits || []) {
    const padId = Number(hit.padId);
    const entry = byPadId.get(padId) || {
      color: null,
      traceIds: [],
      selected: false,
    };
    const value = Number(hit[metricKey]);
    entry.color = entry.color === null ? value : Math.max(entry.color, value);
    entry.traceIds.push(Number(hit.traceId));
    if (selectedTraceIdSet().has(Number(hit.traceId))) {
      entry.selected = true;
    }
    byPadId.set(padId, entry);
  }

  return (props.mappingPads || []).flatMap((pad) => {
    const aggregate = byPadId.get(Number(pad.pad));
    if (!aggregate?.traceIds?.length) {
      return [];
    }
    return [{
      x: Number(pad.cx),
      y: Number(pad.cy),
      color: aggregate.color,
      traceIds: aggregate.traceIds,
      selected: aggregate.selected,
      direction: Number(pad.direction),
      size: Number(pad.scale),
      padId: Number(pad.pad),
    }];
  });
}

function interpolateColor(a, b, t) {
  return Math.round(a + (b - a) * t);
}

function colorToCss(rgb) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function colorFromScale(value, min, max) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "rgba(199, 188, 170, 0.7)";
  }

  const stops = [
    [48, 18, 59],
    [50, 104, 168],
    [38, 188, 225],
    [127, 225, 106],
    [253, 231, 37],
  ];

  if (max <= min) {
    return colorToCss(stops[stops.length - 1]);
  }

  const normalized = Math.min(Math.max((Number(value) - min) / (max - min), 0), 1);
  const scaled = normalized * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const t = scaled - index;
  const start = stops[index];
  const end = stops[index + 1];

  return colorToCss([
    interpolateColor(start[0], end[0], t),
    interpolateColor(start[1], end[1], t),
    interpolateColor(start[2], end[2], t),
  ]);
}

function padPoints(direction, scale) {
  const multiplier = scale * PAD_GAP_SCALE;
  if (direction === 1) {
    return [
      [-HEIGHT / 3 * multiplier, -HALF_EDGE * multiplier],
      [-HEIGHT / 3 * multiplier, HALF_EDGE * multiplier],
      [(HEIGHT * 2) / 3 * multiplier, 0],
    ];
  }
  return [
    [HEIGHT / 3 * multiplier, -HALF_EDGE * multiplier],
    [HEIGHT / 3 * multiplier, HALF_EDGE * multiplier],
    [(-HEIGHT * 2) / 3 * multiplier, 0],
  ];
}

function createPadPolygonTrace(pad, minColor, maxColor) {
  const vertices = padPoints(pad.direction, pad.size).map(([dx, dy]) => [
    pad.x + dx,
    pad.y + dy,
  ]);
  const x = vertices.map(([vx]) => vx);
  const y = vertices.map(([, vy]) => vy);
  x.push(vertices[0][0]);
  y.push(vertices[0][1]);

  return {
    type: "scatter",
    mode: "lines",
    x,
    y,
    fill: "toself",
    line: {
      color: pad.selected ? "#111111" : "#c7bcaa",
      width: pad.selected ? 2 : 1,
    },
    fillcolor: pad.selected ? "#111111" : colorFromScale(pad.color, minColor, maxColor),
    opacity: pad.selected ? 0.95 : 0.92,
    hoverinfo: "skip",
    showlegend: false,
  };
}

function createPadColorTrace(pads, colorTitle) {
  return {
    type: "scatter",
    mode: "markers",
    x: pads.map((pad) => pad.x),
    y: pads.map((pad) => pad.y),
    marker: {
      size: 0.1,
      opacity: 0,
      color: pads.map((pad) => pad.color),
      colorscale: "Turbo",
      colorbar: { title: colorTitle },
    },
    hoverinfo: "skip",
    showlegend: false,
  };
}

function createPadInteractionTrace(pads, colorTitle) {
  return {
    type: "scatter",
    mode: "markers",
    x: pads.map((pad) => pad.x),
    y: pads.map((pad) => pad.y),
    customdata: pads.map((pad) => ({ traceIds: pad.traceIds, padId: pad.padId })),
    marker: {
      size: pads.map((pad) => Math.max(12, pad.size * 14)),
      opacity: 0.001,
      color: pads.map((pad) => pad.color),
      colorscale: "Turbo",
      showscale: false,
    },
    hovertemplate: "pad %{customdata.padId}<br>x %{x:.2f}<br>y %{y:.2f}<br>value %{marker.color:.2f}<extra></extra>",
    showlegend: false,
  };
}

function next2dRange(eventData) {
  const resolvedAxisRange = isProjectedTwoDimensionalPlot() ? projectedAxisRange : axisRange;
  const currentX = resolvedAxisRange("x");
  const currentY = resolvedAxisRange("y");
  const nextX = [
    eventData?.["xaxis.range[0]"],
    eventData?.["xaxis.range[1]"],
  ];
  const nextY = [
    eventData?.["yaxis.range[0]"],
    eventData?.["yaxis.range[1]"],
  ];

  if (eventData?.["xaxis.autorange"] || eventData?.["yaxis.autorange"]) {
    return null;
  }

  const hasX = nextX.every((value) => value !== undefined);
  const hasY = nextY.every((value) => value !== undefined);
  if (!hasX && !hasY) {
    return undefined;
  }

  return {
    x: hasX ? nextX.map((value) => Number(value)) : currentX,
    y: hasY ? nextY.map((value) => Number(value)) : currentY,
  };
}

async function renderHits3d() {
  const Plotly = await loadPlotly();
  const selected = selectedTraceIdSet();

  await Plotly.react(
    root.value,
    [
      {
        type: "scatter3d",
        mode: "markers",
        x: props.hits.map((hit) => Number(hit.x)),
        y: props.hits.map((hit) => Number(hit.z)),
        z: props.hits.map((hit) => Number(hit.y)),
        customdata: props.hits.map((hit) => ({
          traceId: Number(hit.traceId),
          x: Number(hit.x),
          y: Number(hit.y),
          z: Number(hit.z),
        })),
        marker: {
          size: 2,
          color: props.hits.map((hit) => Number(hit.amplitude)),
          colorscale: "Turbo",
          line: {
            color: "#111111",
            width: props.hits.map((hit) => (selected.has(Number(hit.traceId)) ? 4 : 0)),
          },
          colorbar: { title: "Q" },
        },
        hovertemplate: "trace %{customdata.traceId}<br>x %{customdata.x:.2f}<br>y %{customdata.y:.2f}<br>z %{customdata.z:.2f}<br>Q %{marker.color:.2f}<extra></extra>",
      },
    ],
    {
      ...baseLayout(),
      title: { text: "3D xyz · Q", x: 0.02, xanchor: "left" },
      scene: {
        xaxis: {
          title: { text: "x" },
          backgroundcolor: "#fffdf8",
          gridcolor: "#e7dfcf",
          range: [DEFAULT_3D_RANGES.x[1], DEFAULT_3D_RANGES.x[0]],
          autorange: false,
        },
        yaxis: {
          title: { text: "z" },
          backgroundcolor: "#fffdf8",
          gridcolor: "#e7dfcf",
          range: DEFAULT_3D_RANGES.y,
          autorange: false,
        },
        zaxis: {
          title: { text: "y" },
          backgroundcolor: "#fffdf8",
          gridcolor: "#e7dfcf",
          range: DEFAULT_3D_RANGES.z,
          autorange: false,
        },
        aspectmode: "manual",
        aspectratio: threeDimensionalAspectRatio(),
        camera: props.camera || DEFAULT_3D_CAMERA,
      },
    },
    config(),
  );
}

async function renderHits2d(metricKey, title, colorTitle) {
  const Plotly = await loadPlotly();
  const selected = selectedTraceIdSet();

  await Plotly.react(
    root.value,
    [
      {
        type: "scattergl",
        mode: "markers",
        x: props.hits.map((hit) => Number(hit.x)),
        y: props.hits.map((hit) => Number(hit.y)),
        customdata: props.hits.map((hit) => ({
          traceId: Number(hit.traceId),
          padId: Number(hit.padId),
        })),
        marker: {
          size: 8,
          color: props.hits.map((hit) => Number(hit[metricKey])),
          colorscale: "Turbo",
          colorbar: { title: colorTitle },
          line: {
            color: "#111111",
            width: props.hits.map((hit) => (selected.has(Number(hit.traceId)) ? 2 : 0)),
          },
        },
        hovertemplate: "trace %{customdata.traceId}<br>pad %{customdata.padId}<br>x %{x:.2f}<br>y %{y:.2f}<br>value %{marker.color:.2f}<extra></extra>",
      },
    ],
    twoDimensionalLayout(title),
    config(),
  );
}

async function renderProjectedHits2d() {
  const Plotly = await loadPlotly();
  const selected = selectedTraceIdSet();
  const projectedHits = (props.hits || []).filter(
    (hit) => hit.xPrime !== null && hit.yPrime !== null,
  );

  if (!projectedHits.length) {
    await Plotly.react(
      root.value,
      [],
      {
        ...twoDimensionalLayout("2D x'y' · Q", "x'", "y'", "projected"),
        annotations: [
          {
            text: "Not enough points for PCA reprojection",
            xref: "paper",
            yref: "paper",
            x: 0.5,
            y: 0.5,
            showarrow: false,
            font: { size: 16, color: "#6b6257" },
          },
        ],
      },
      config(),
    );
    return;
  }

  await Plotly.react(
    root.value,
    [
      {
        type: "scattergl",
        mode: "markers",
        x: projectedHits.map((hit) => Number(hit.xPrime)),
        y: projectedHits.map((hit) => Number(hit.yPrime)),
        customdata: projectedHits.map((hit) => ({
          traceId: Number(hit.traceId),
          padId: Number(hit.padId),
          q: Number(hit.amplitude),
        })),
        marker: {
          size: 8,
          color: projectedHits.map((hit) => Number(hit.amplitude)),
          colorscale: "Turbo",
          colorbar: { title: "Q" },
          line: {
            color: "#111111",
            width: projectedHits.map((hit) => (selected.has(Number(hit.traceId)) ? 2 : 0)),
          },
        },
        hovertemplate: "trace %{customdata.traceId}<br>pad %{customdata.padId}<br>x' %{x:.2f}<br>y' %{y:.2f}<br>Q %{customdata.q:.2f}<extra></extra>",
      },
    ],
    twoDimensionalLayout("2D x'y' · Q", "x'", "y'", "projected"),
    config(),
  );
}

async function renderPads(metricKey, title, colorTitle) {
  const Plotly = await loadPlotly();
  const pads = aggregatePads(metricKey);
  const values = pads.map((pad) => Number(pad.color)).filter((value) => !Number.isNaN(value));
  const minColor = values.length ? Math.min(...values) : 0;
  const maxColor = values.length ? Math.max(...values) : 1;
  const traces = pads.map((pad) => createPadPolygonTrace(pad, minColor, maxColor));
  traces.push(createPadColorTrace(pads, colorTitle));
  traces.push(createPadInteractionTrace(pads, colorTitle));

  await Plotly.react(
    root.value,
    traces,
    twoDimensionalLayout(title),
    config(),
  );
}

async function renderTraces() {
  const Plotly = await loadPlotly();
  const traces = [];
  for (const series of props.traces || []) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: series.trace.map((_, index) => index),
      y: series.trace,
      name: `Trace ${series.traceId}`,
      line: { width: 2 },
    });
    traces.push({
      type: "scatter",
      mode: "markers",
      x: (series.peaks || []).map((peak) => Number(peak.timeBucket)),
      y: (series.peaks || []).map((peak) => Number(peak.amplitude)),
      name: `Peaks ${series.traceId}`,
      marker: { size: 8, symbol: "x", color: "#111111" },
      showlegend: false,
    });
  }

  await Plotly.react(
    root.value,
    traces,
    {
      ...baseLayout(),
      title: { text: "traces", x: 0.02, xanchor: "left" },
      xaxis: {
        title: "Time bucket",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      yaxis: {
        title: "Amplitude",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      legend: {
        orientation: "h",
        x: 1,
        xanchor: "right",
        y: 1.14,
      },
    },
    config(),
  );
}

function clearListeners() {
  root.value?.removeAllListeners?.("plotly_click");
  root.value?.removeAllListeners?.("plotly_relayout");
}

async function resizePlot() {
  if (!root.value) {
    return;
  }
  const Plotly = await loadPlotly();
  await nextTick();
  Plotly.Plots.resize(root.value);
}

function scheduleResize() {
  if (resizeFrame) {
    cancelAnimationFrame(resizeFrame);
  }
  resizeFrame = requestAnimationFrame(() => {
    resizeFrame = 0;
    void resizePlot();
  });
}

function bindEvents() {
  clearListeners();
  if (!root.value) {
    return;
  }
  if (isPadPlot()) {
    root.value.on("plotly_click", (event) => {
      const traceIds = event?.points?.[0]?.customdata?.traceIds || [];
      emit("toggle-traces", traceIds);
    });
  }
  root.value.on("plotly_relayout", (eventData) => {
    if (isTwoDimensionalHitPlot()) {
      const range = next2dRange(eventData);
      if (range !== undefined) {
        if (isProjectedTwoDimensionalPlot()) {
          emit("update-projected-range", range);
        } else {
          emit("update-xy-range", range);
        }
      }
    }
    if (isThreeDimensionalPlot() && eventData["scene.camera"]) {
      emit("update-camera", eventData["scene.camera"]);
    }
  });
}

async function renderPlot() {
  if (!root.value) {
    return;
  }
  if (props.plotType === "hits-3d-amplitude") {
    await renderHits3d();
  } else if (props.plotType === "hits-2d-z") {
    await renderHits2d("z", "2D xy · z", "z");
  } else if (props.plotType === "hits-2d-amplitude") {
    await renderHits2d("amplitude", "2D xy · Q", "Q");
  } else if (props.plotType === "hits-2d-pca-amplitude") {
    await renderProjectedHits2d();
  } else if (props.plotType === "pads-z") {
    await renderPads("z", "pads · z", "z");
  } else if (props.plotType === "pads-amplitude") {
    await renderPads("amplitude", "pads · Q", "Q");
  } else {
    await renderTraces();
  }
  bindEvents();
  scheduleResize();
}

onMounted(() => {
  if (typeof ResizeObserver !== "undefined" && root.value) {
    resizeObserver = new ResizeObserver(() => {
      scheduleResize();
    });
    resizeObserver.observe(root.value);
  }
  void renderPlot();
});

watch(
  () => [
    props.plotType,
    props.layoutMode,
    props.hits,
    props.mappingPads,
    props.selectedTraceIds,
    props.traces,
    props.xyRange,
    props.projectedRange,
    props.camera,
  ],
  () => {
    void renderPlot();
  },
  { deep: true },
);

onBeforeUnmount(() => {
  if (resizeFrame) {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = 0;
  }
  resizeObserver?.disconnect?.();
  resizeObserver = null;
  clearListeners();
  void loadPlotly().then((Plotly) => {
    if (root.value) {
      Plotly.purge(root.value);
    }
  });
});
</script>
