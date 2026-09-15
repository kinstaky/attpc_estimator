<template>
  <v-container class="page-container settings-page" fluid>
    <div class="page-header">
      <div>
        <p class="page-kicker">Preferences</p>
        <h1>Settings</h1>
        <p class="page-copy">Adjust the workspace appearance for comfortable analysis and clear captures.</p>
      </div>
    </div>

    <section class="settings-section" aria-labelledby="appearance-heading">
      <div class="settings-section-heading">
        <p class="page-kicker">Appearance</p>
        <h2 id="appearance-heading">Workspace theme</h2>
      </div>

      <div class="settings-row">
        <div>
          <strong>Brightness</strong>
          <p>Use a bright canvas for screenshots or a dark canvas for low-light work.</p>
        </div>
        <v-switch
          :model-value="appearance.state.mode === 'dark'"
          color="primary"
          hide-details
          inset
          label="Dark mode"
          @update:model-value="appearance.setMode($event ? 'dark' : 'light')"
        />
      </div>

      <div class="settings-row settings-row--stacked">
        <div>
          <strong>Theme color</strong>
          <p>Choose the accent used for active controls, selection states, and focus cues.</p>
        </div>
        <div class="accent-picker" role="radiogroup" aria-label="Theme color">
          <button
            v-for="preset in appearance.presets"
            :key="preset.id"
            class="accent-swatch"
            :class="{ 'accent-swatch--selected': appearance.state.accent === preset.id }"
            type="button"
            role="radio"
            :aria-checked="appearance.state.accent === preset.id"
            :aria-label="preset.label"
            :title="preset.label"
            :style="{ '--accent-swatch': preset.color }"
            @click="appearance.setAccent(preset.id)"
          >
            <span class="accent-swatch-color" aria-hidden="true"></span>
            <span>{{ preset.label }}</span>
          </button>
        </div>
      </div>
    </section>
  </v-container>
</template>

<script setup lang="ts">
import { useAppearanceStore } from "../stores/appearance";

const appearance = useAppearanceStore();
</script>
