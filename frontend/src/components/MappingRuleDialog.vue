<template>
  <v-dialog :model-value="modelValue" max-width="560" @update:model-value="onDialogToggle">
    <v-card rounded="xl">
      <v-card-title class="dialog-title">
        {{ index === null ? "Add render rule" : "Edit render rule" }}
      </v-card-title>
      <v-card-text>
        <p class="dialog-copy">
          Match detector channels by hardware identifiers. Use digits for an exact match or <code>*</code> as a wildcard.
        </p>

        <v-row dense>
          <v-col cols="12" sm="6">
            <v-text-field
              v-model="draft.cobo"
              label="Cobo"
              variant="outlined"
              :error-messages="errors.cobo"
            />
          </v-col>
          <v-col cols="12" sm="6">
            <v-text-field
              v-model="draft.asad"
              label="Asad"
              variant="outlined"
              :error-messages="errors.asad"
            />
          </v-col>
          <v-col cols="12" sm="6">
            <v-text-field
              v-model="draft.aget"
              label="Aget"
              variant="outlined"
              :error-messages="errors.aget"
            />
          </v-col>
          <v-col cols="12" sm="6">
            <v-text-field
              v-model="draft.channel"
              label="Channel"
              variant="outlined"
              :error-messages="errors.channel"
            />
          </v-col>
        </v-row>

        <v-row dense>
          <v-col cols="12" sm="8">
            <v-text-field
              v-model="draft.color"
              label="Color"
              variant="outlined"
              :error-messages="errors.color"
            />
          </v-col>
          <v-col cols="12" sm="4" class="mapping-rule-color-field">
            <label class="mapping-rule-color-picker-label">
              <span class="page-kicker">Picker</span>
              <input
                v-model="pickerColor"
                class="mapping-rule-color-picker"
                type="color"
              />
            </label>
          </v-col>
        </v-row>

        <div class="mapping-rule-swatches">
          <button
            v-for="color in swatches"
            :key="color"
            class="mapping-rule-swatches-button"
            :aria-label="`Choose ${color}`"
            :style="{ backgroundColor: color }"
            type="button"
            @click="draft.color = color"
          />
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="close">Cancel</v-btn>
        <v-btn color="primary" variant="tonal" @click="save">Apply</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import { computed, reactive, watch } from "vue";

import type { MappingRenderRule } from "../types";

const DEFAULT_COLOR = "#E91E63";
const FORBIDDEN_CHANNELS = new Set([11, 22, 45, 56]);

const swatches = [
  "#E91E63",
  "#FF6D00",
  "#FDD835",
  "#4CAF50",
  "#00BCD4",
  "#2196F3",
  "#7E57C2",
  "#9C27B0",
  "#F77976",
  "#173D31",
];

const props = defineProps<{
  modelValue: boolean;
  index: number | null;
  initialRule: MappingRenderRule | null;
}>();

const emit = defineEmits<{
  "update:modelValue": [value: boolean];
  save: [payload: { index: number | null; rule: MappingRenderRule }];
}>();

const draft = reactive<MappingRenderRule>({
  cobo: "",
  asad: "",
  aget: "",
  channel: "",
  color: DEFAULT_COLOR,
});

function normalizeColor(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return DEFAULT_COLOR;
  }
  return trimmed.startsWith("#") ? trimmed.toUpperCase() : `#${trimmed.toUpperCase()}`;
}

function applyInitialRule(): void {
  const source = props.initialRule ?? {
    cobo: "",
    asad: "",
    aget: "",
    channel: "",
    color: DEFAULT_COLOR,
  };
  draft.cobo = source.cobo;
  draft.asad = source.asad;
  draft.aget = source.aget;
  draft.channel = source.channel;
  draft.color = normalizeColor(source.color);
}

watch(
  () => props.modelValue,
  (open) => {
    if (open) {
      applyInitialRule();
    }
  },
  { immediate: true },
);

const pickerColor = computed({
  get: () => normalizeColor(draft.color),
  set: (value: string) => {
    draft.color = normalizeColor(value);
  },
});

function validateDigitOrWildcard(
  label: string,
  value: string,
  minimum: number,
  maximum: number,
): string[] {
  const trimmed = value.trim();
  if (!trimmed) {
    return [`${label} is required.`];
  }
  if (trimmed === "*") {
    return [];
  }
  if (!/^\d+$/.test(trimmed)) {
    return [`${label} must be digits or *.`];
  }
  const numeric = Number(trimmed);
  if (numeric < minimum || numeric > maximum) {
    return [`${label} must be between ${minimum} and ${maximum}.`];
  }
  return [];
}

const errors = computed(() => {
  const channelErrors = validateDigitOrWildcard("Channel", draft.channel, 0, 67);
  if (channelErrors.length === 0 && draft.channel.trim() !== "*") {
    const numeric = Number(draft.channel.trim());
    if (FORBIDDEN_CHANNELS.has(numeric)) {
      channelErrors.push("Channel cannot be 11, 22, 45, or 56.");
    }
  }

  const colorErrors = /^#?[0-9A-Fa-f]{6}$/.test(draft.color.trim())
    ? []
    : ["Color must be a 6-digit hex value."];

  return {
    cobo: validateDigitOrWildcard("Cobo", draft.cobo, 0, 10),
    asad: validateDigitOrWildcard("Asad", draft.asad, 0, 3),
    aget: validateDigitOrWildcard("Aget", draft.aget, 0, 3),
    channel: channelErrors,
    color: colorErrors,
  };
});

function close(): void {
  emit("update:modelValue", false);
}

function onDialogToggle(value: boolean): void {
  emit("update:modelValue", value);
}

function save(): void {
  const currentErrors = errors.value;
  if (Object.values(currentErrors).some((messages) => messages.length > 0)) {
    return;
  }
  emit("save", {
    index: props.index,
    rule: {
      cobo: draft.cobo.trim(),
      asad: draft.asad.trim(),
      aget: draft.aget.trim(),
      channel: draft.channel.trim(),
      color: normalizeColor(draft.color),
    },
  });
  close();
}
</script>
