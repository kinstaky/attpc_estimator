import { computed, reactive } from "vue";

import type { AppearanceUiState } from "../types";

export const ACCENT_PRESETS = [
  { id: "cobalt", label: "Cobalt", color: "#1456c4", soft: "#eaf1ff", rgb: "20, 86, 196" },
  { id: "azure", label: "Azure", color: "#0077b6", soft: "#e5f4fb", rgb: "0, 119, 182" },
  { id: "teal", label: "Teal", color: "#087e8b", soft: "#e4f6f7", rgb: "8, 126, 139" },
  { id: "jade", label: "Jade", color: "#16734a", soft: "#e8f5ed", rgb: "22, 115, 74" },
  { id: "amber", label: "Amber", color: "#a86400", soft: "#fff3db", rgb: "168, 100, 0" },
  { id: "orange", label: "Orange", color: "#c2410c", soft: "#fff0e8", rgb: "194, 65, 12" },
  { id: "crimson", label: "Crimson", color: "#b42318", soft: "#feeceb", rgb: "180, 35, 24" },
  { id: "raspberry", label: "Raspberry", color: "#be185d", soft: "#fce8f1", rgb: "190, 24, 93" },
  { id: "violet", label: "Violet", color: "#6d28d9", soft: "#f1eafd", rgb: "109, 40, 217" },
  { id: "indigo", label: "Indigo", color: "#4338ca", soft: "#ecebff", rgb: "67, 56, 202" },
] as const;

export type AppearanceMode = AppearanceUiState["mode"];
export type AccentId = AppearanceUiState["accent"];

const state = reactive<AppearanceUiState>({ mode: "light", accent: "cobalt" });

function selectedPreset() {
  return ACCENT_PRESETS.find((preset) => preset.id === state.accent) ?? ACCENT_PRESETS[0];
}

function applyToDocument(): void {
  if (typeof document === "undefined") {
    return;
  }
  const root = document.documentElement;
  const preset = selectedPreset();
  root.dataset.appearanceMode = state.mode;
  root.style.setProperty("--signal", preset.color);
  root.style.setProperty("--signal-soft", preset.soft);
}

function setMode(mode: AppearanceMode): void {
  state.mode = mode;
  applyToDocument();
}

function setAccent(accent: AccentId): void {
  state.accent = accent;
  applyToDocument();
}

function applyUiState(payload: AppearanceUiState | null | undefined): void {
  state.mode = payload?.mode === "dark" ? "dark" : "light";
  state.accent = ACCENT_PRESETS.some((preset) => preset.id === payload?.accent)
    ? payload!.accent
    : "cobalt";
  applyToDocument();
}

const plotAppearanceKey = computed(() => `${state.mode}:${state.accent}`);
const vuetifyThemeVars = computed(() => {
  const dark = state.mode === "dark";
  return {
    "--v-theme-primary": selectedPreset().rgb,
    "--v-theme-accent": selectedPreset().rgb,
    "--v-theme-info": selectedPreset().rgb,
    "--v-theme-background": dark ? "23, 25, 29" : "247, 247, 245",
    "--v-theme-surface": dark ? "32, 35, 41" : "255, 255, 255",
    "--v-theme-surface-variant": dark ? "41, 45, 52" : "241, 242, 244",
    "--v-theme-secondary": dark ? "176, 183, 195" : "71, 85, 105",
    "--v-theme-on-surface": dark ? "242, 244, 247" : "23, 25, 29",
    "--v-theme-on-background": dark ? "242, 244, 247" : "23, 25, 29",
  };
});

export function useAppearanceStore() {
  return {
    state,
    presets: ACCENT_PRESETS,
    plotAppearanceKey,
    vuetifyThemeVars,
    setMode,
    setAccent,
    applyUiState,
    applyToDocument,
  };
}
