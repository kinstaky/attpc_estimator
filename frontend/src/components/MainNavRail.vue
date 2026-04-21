<template>
  <v-navigation-drawer
    v-model:rail="isRail"
    class="main-nav-rail"
    color="surface"
    rail
    expand-on-hover
    permanent
    width="220"
    rail-width="72"
  >
    <div class="main-nav-brand">
      <span class="main-nav-brand-mark">AT</span>
      <div
        class="main-nav-brand-copy"
        :class="{ 'main-nav-brand-copy--visible': !isRail }"
      >
        <strong>AT-TPC</strong>
        <span>Estimator</span>
      </div>
    </div>

    <v-list class="main-nav-list" density="comfortable" nav>
      <v-list-item
        v-for="item in items"
        :key="item.to"
        :to="item.to"
        class="main-nav-item"
        rounded="xl"
      >
        <template #prepend>
          <v-icon :icon="item.icon" />
        </template>
        <v-list-item-title
          class="main-nav-item-title"
          :class="{ 'main-nav-item-title--visible': !isRail }"
        >
          {{ item.title }}
        </v-list-item-title>
      </v-list-item>
    </v-list>

    <template #append>
      <div class="main-nav-footer">
        <span class="main-nav-footer-mark">AT</span>
        <div
          class="main-nav-footer-copy"
          :class="{ 'main-nav-footer-copy--visible': !isRail }"
        >
          <span class="main-nav-footer-label">Run</span>
          <strong>{{ selectedRunLabel }}</strong>
        </div>
      </div>
    </template>
  </v-navigation-drawer>
</template>

<script setup>
import { computed, ref } from "vue";

import { useShellStore } from "../stores/shell";

const { state } = useShellStore();
const isRail = ref(true);

const items = [
  { to: "/", title: "Home", icon: "mdi-home-outline" },
  { to: "/label", title: "Label", icon: "mdi-pencil-box-outline" },
  { to: "/histograms", title: "Histograms", icon: "mdi-chart-box-outline" },
  { to: "/mapping", title: "Mapping", icon: "mdi-vector-polygon" },
  { to: "/review", title: "Review", icon: "mdi-file-search-outline" },
  { to: "/pointcloud", title: "Pointcloud", icon: "mdi-cube-outline" },
];

const selectedRunLabel = computed(() => {
  if (state.selectedRun === null || state.selectedRun === undefined) {
    return "None";
  }
  return `Run ${state.selectedRun}`;
});
</script>
