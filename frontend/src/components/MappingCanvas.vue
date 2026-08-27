<template>
  <div class="mapping-canvas-shell">
    <div class="mapping-canvas-status">
      <p class="page-kicker">Selection</p>
      <strong v-if="selectedChannel">
        <template v-if="isPad(selectedChannel)">
          Cobo {{ selectedChannel.cobo }} · Asad {{ selectedChannel.asad }} · Aget {{ selectedChannel.aget }} · Ch {{ selectedChannel.channel }} · Pad {{ selectedChannel.pad }}
        </template>
        <template v-else>
          {{ selectedChannel.detector }} · {{ selectedChannel.side }} · Strip {{ selectedChannel.strip }} · Cobo {{ selectedChannel.cobo }} · Asad {{ selectedChannel.asad }} · Aget {{ selectedChannel.aget }} · Ch {{ selectedChannel.channel }}
        </template>
      </strong>
      <span v-else>No channel selected.</span>
    </div>

    <div ref="root" class="mapping-canvas-root"></div>

    <div class="mapping-canvas-footer">
      <p>x: {{ mouseX.toFixed(2) }} mm, y: {{ mouseY.toFixed(2) }} mm</p>
      <span>Drag to pan. Scroll to zoom. Click a channel to inspect it.</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Application, Container, Graphics, Point, Rectangle } from "pixi.js";

import type { MappingChannel, MappingLayer, MappingPad, MappingRenderRule, MappingStrip, MappingViewMode } from "../types";

const props = defineProps<{
  channels: MappingChannel[];
  layer: MappingLayer;
  rules: MappingRenderRule[];
  view: MappingViewMode;
}>();

interface GraphicEntry { index: number; data: MappingChannel; graphic: Graphics; hovered: boolean; }

const DEFAULT_COLOR = 0xc0c0c0;
const SELECTED_COLOR = 0x111111;
const FLOAT_SCALE = 1.3;
const HALF_EDGE = 4.5;
const HEIGHT = HALF_EDGE * (3 ** 0.5);
const HALF_HEIGHT = HEIGHT * 0.5;
const INIT_X_SCALE = 1.3;
const INIT_Y_SCALE = -1.3;
const SILICON_SIZE = 97.22;
const SILICON_PITCH = SILICON_SIZE / 128;
const SILICON_WIDTH = SILICON_PITCH * 0.8;
const SILICON_LENGTH = SILICON_SIZE;

const root = ref<HTMLDivElement | null>(null);
const mouseX = ref(0);
const mouseY = ref(0);
const selectedIndex = ref<number | null>(null);
const selectedChannel = computed<MappingChannel | null>(() => selectedIndex.value === null ? null : props.channels[selectedIndex.value] ?? null);

let app: Application | null = null;
let container: Container | null = null;
let entries: GraphicEntry[] = [];
let zoomScale = 1;
let dragging = false;
let dragged = false;
let lastPointerWasChannel = false;
let dragStart = { x: 0, y: 0 };
let containerStart = { x: 0, y: 0 };
let onMouseDown: ((event: MouseEvent) => void) | null = null;
let onMouseMove: ((event: MouseEvent) => void) | null = null;
let onMouseUp: (() => void) | null = null;
let onMouseLeave: (() => void) | null = null;
let onWheel: ((event: WheelEvent) => void) | null = null;

function isPad(channel: MappingChannel): channel is MappingPad { return "pad" in channel; }
function isStrip(channel: MappingChannel): channel is MappingStrip { return "strip" in channel; }
function matchesRule(channel: MappingChannel, rule: MappingRenderRule): boolean {
  return (rule.cobo === "*" || Number(rule.cobo) === channel.cobo)
    && (rule.asad === "*" || Number(rule.asad) === channel.asad)
    && (rule.aget === "*" || Number(rule.aget) === channel.aget)
    && (rule.channel === "*" || Number(rule.channel) === channel.channel);
}
function resolveFillColor(channel: MappingChannel): number {
  for (const rule of props.rules) {
    if (matchesRule(channel, rule)) {
      const parsed = Number.parseInt(rule.color.replace("#", ""), 16);
      if (!Number.isNaN(parsed)) return parsed;
    }
  }
  return DEFAULT_COLOR;
}
function padPoints(direction: number): number[] {
  return direction === 1
    ? [-HEIGHT / 3, -HALF_EDGE, -HEIGHT / 3, HALF_EDGE, (HEIGHT * 2) / 3, 0]
    : [HEIGHT / 3, -HALF_EDGE, HEIGHT / 3, HALF_EDGE, (-HEIGHT * 2) / 3, 0];
}
function drawSiliconStrip(graphic: Graphics, strip: MappingStrip): void {
  const offset = (63.5 - strip.strip) * SILICON_PITCH;
  const diagonalOffset = offset / Math.SQRT2;
  graphic.rect(-SILICON_LENGTH / 2, -SILICON_WIDTH / 2, SILICON_LENGTH, SILICON_WIDTH);
  graphic.position.set(diagonalOffset, strip.side === "front" ? -diagonalOffset : diagonalOffset);
  graphic.rotation = strip.side === "front" ? Math.PI / 4 : -Math.PI / 4;
}
function resetGraphic(entry: GraphicEntry): void {
  const isSelected = selectedIndex.value === entry.index;
  entry.graphic.clear();
  if (isPad(entry.data)) {
    entry.graphic.poly(padPoints(entry.data.direction));
    entry.graphic.position.set(entry.data.cx, entry.data.cy);
    entry.graphic.rotation = 0;
    entry.graphic.scale.set(entry.data.scale * (isSelected || entry.hovered ? FLOAT_SCALE : 1));
  } else {
    drawSiliconStrip(entry.graphic, entry.data);
    const isEmphasized = isSelected || entry.hovered;
    entry.graphic.scale.set(isEmphasized ? 1.02 : 1, isEmphasized ? 1.1 : 1);
  }
  entry.graphic.fill(isSelected ? SELECTED_COLOR : resolveFillColor(entry.data));
  if (isSelected && entry.graphic.parent) {
    entry.graphic.parent.setChildIndex(entry.graphic, entry.graphic.parent.children.length - 1);
  }
}
function rebuildChannels(): void {
  if (!container) return;
  const scene = container;
  for (const child of scene.removeChildren()) child.destroy();
  entries = [];
  selectedIndex.value = null;
  const channelScene = scene;
  const visibleChannels = props.layer === "Pads"
    ? props.channels
    : props.channels.filter(
      (channel): channel is MappingStrip => isStrip(channel)
        && channel.side === (props.view === "Downstream" ? "front" : "back"),
    );
  visibleChannels.forEach((channel) => {
    const index = props.channels.indexOf(channel);
    const graphic = new Graphics();
    const entry: GraphicEntry = { index, data: channel, graphic, hovered: false };
    graphic.eventMode = "static";
    graphic.cursor = "pointer";
    if (isStrip(channel)) {
      graphic.hitArea = new Rectangle(
        -SILICON_LENGTH / 2,
        -SILICON_PITCH / 2,
        SILICON_LENGTH,
        SILICON_PITCH,
      );
    }
    graphic.on("pointerover", () => { entry.hovered = true; resetGraphic(entry); });
    graphic.on("pointerout", () => { entry.hovered = false; resetGraphic(entry); });
    graphic.on("pointertap", () => { if (!dragged) { lastPointerWasChannel = true; selectedIndex.value = index; redrawAll(); } });
    entries.push(entry);
    resetGraphic(entry);
    channelScene.addChild(graphic);
  });
}
function redrawAll(): void { entries.forEach(resetGraphic); }
function currentRevertX(): number { return props.view === "Upstream" ? -1 : 1; }
function updateContainerScale(): void { if (container) { container.scale.x = INIT_X_SCALE * zoomScale * currentRevertX(); container.scale.y = INIT_Y_SCALE * zoomScale; } }
function eventClipPosition(event: MouseEvent | WheelEvent): { x: number; y: number } {
  if (!app) return { x: 0, y: 0 };
  const rect = app.canvas.getBoundingClientRect();
  return { x: (event.clientX - rect.left) * (app.renderer.width / rect.width), y: (event.clientY - rect.top) * (app.renderer.height / rect.height) };
}
async function mountPixi(): Promise<void> {
  if (!root.value || app) return;
  app = new Application();
  await app.init({ background: "#fffdf8", resizeTo: root.value, antialias: true });
  root.value.appendChild(app.canvas);
  container = new Container(); app.stage.addChild(container); container.position.set(app.screen.width / 2, app.screen.height / 2);
  updateContainerScale(); rebuildChannels();
  onMouseDown = (event) => { dragging = true; dragged = false; dragStart = eventClipPosition(event); containerStart = { x: container?.x ?? 0, y: container?.y ?? 0 }; };
  onMouseMove = (event) => { if (!container) return; const point = eventClipPosition(event); const world = container.toLocal(new Point(point.x, point.y)); mouseX.value = world.x; mouseY.value = world.y; if (dragging) { dragged = true; container.x = containerStart.x + point.x - dragStart.x; container.y = containerStart.y + point.y - dragStart.y; } };
  onMouseUp = () => { dragging = false; if (lastPointerWasChannel) { lastPointerWasChannel = false; return; } if (!dragged) { selectedIndex.value = null; redrawAll(); } };
  onMouseLeave = () => { dragging = false; };
  onWheel = (event) => { if (!container) return; event.preventDefault(); const point = eventClipPosition(event); const before = container.toLocal(new Point(point.x, point.y)); zoomScale = Math.min(Math.max(zoomScale * (event.deltaY < 0 ? 1.1 : 1 / 1.1), 0.8), 12); updateContainerScale(); const after = container.toLocal(new Point(point.x, point.y)); container.x += (after.x - before.x) * INIT_X_SCALE * zoomScale * currentRevertX(); container.y += (after.y - before.y) * INIT_Y_SCALE * zoomScale; };
  app.canvas.addEventListener("mousedown", onMouseDown); app.canvas.addEventListener("mousemove", onMouseMove); app.canvas.addEventListener("mouseup", onMouseUp); app.canvas.addEventListener("mouseleave", onMouseLeave); app.canvas.addEventListener("wheel", onWheel, { passive: false });
}
function cleanupPixi(): void { if (app?.canvas && onMouseDown && onMouseMove && onMouseUp && onMouseLeave && onWheel) { app.canvas.removeEventListener("mousedown", onMouseDown); app.canvas.removeEventListener("mousemove", onMouseMove); app.canvas.removeEventListener("mouseup", onMouseUp); app.canvas.removeEventListener("mouseleave", onMouseLeave); app.canvas.removeEventListener("wheel", onWheel); } entries = []; container = null; app?.destroy(true, { children: true }); app = null; }
watch(() => props.rules, redrawAll);
watch(() => [props.channels, props.layer, props.view], rebuildChannels);
onMounted(() => { void mountPixi(); });
onBeforeUnmount(cleanupPixi);
</script>
