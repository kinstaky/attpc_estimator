<template>
  <v-app :style="vuetifyThemeVars">
    <section v-if="state.loading" class="app-loading-state">
      <p class="page-kicker">Application</p>
      <h1>Loading app state…</h1>
    </section>

    <section v-else-if="state.error" class="app-loading-state">
      <p class="page-kicker">Application</p>
      <h1>Failed to load the backend bootstrap.</h1>
      <p class="page-copy">{{ state.error }}</p>
    </section>

    <section v-else-if="state.bootstrap?.appType !== 'merged'" class="app-loading-state">
      <p class="page-kicker">Application</p>
      <h1>Unsupported app type.</h1>
    </section>

    <div v-else class="app-layout">
      <MainNavRail />

      <v-main class="app-main">
        <router-view v-slot="{ Component }">
          <keep-alive>
            <component :is="Component" :key="$route.fullPath + plotAppearanceKey" />
          </keep-alive>
        </router-view>
      </v-main>
    </div>
  </v-app>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import MainNavRail from "./components/MainNavRail.vue";
import { useAppearanceStore } from "./stores/appearance";
import { useShellStore } from "./stores/shell";
import {
  hydrateUiState,
  startUiStatePersistence,
} from "./stores/ui_state";

const { state, init } = useShellStore();
const { applyToDocument, plotAppearanceKey, vuetifyThemeVars } = useAppearanceStore();

onMounted(() => {
  void (async () => {
    await init();
    if (state.bootstrap?.uiState) {
      await hydrateUiState(state.bootstrap.uiState);
    }
    applyToDocument();
    startUiStatePersistence();
  })();
});
</script>
