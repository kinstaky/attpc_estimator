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
      <template v-for="item in items" :key="item.title">
        <div v-if="item.children" class="main-nav-group">
          <v-list-item
            class="main-nav-item main-nav-item--static"
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

          <div
            class="main-nav-submenu"
            :class="{ 'main-nav-submenu--visible': !isRail }"
          >
            <v-list-item
              v-for="child in item.children"
              :key="child.to"
              :to="child.to"
              class="main-nav-item main-nav-item--child"
              rounded="xl"
            >
              <template #prepend>
                <v-icon :icon="child.icon" />
              </template>
              <v-list-item-title class="main-nav-item-title main-nav-item-title--visible">
                {{ child.title }}
              </v-list-item-title>
            </v-list-item>
          </div>
        </div>

        <v-list-item
          v-else
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
      </template>
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
  {
    title: "Label",
    icon: "mdi-pencil-box-outline",
    children: [
      { to: "/label/trace", title: "Trace", icon: "mdi-waveform" },
      { to: "/label/pointcloud", title: "Pointcloud", icon: "mdi-chart-bubble" },
    ],
  },
  {
    title: "Browse",
    icon: "mdi-file-search-outline",
    children: [
      { to: "/browse/trace", title: "Trace", icon: "mdi-waveform" },
      { to: "/browse/pointcloud", title: "Pointcloud", icon: "mdi-cube-outline" },
    ],
  },
  { to: "/histograms", title: "Histograms", icon: "mdi-chart-box-outline" },
  { to: "/mapping", title: "Mapping", icon: "mdi-vector-polygon" },
];

const selectedRunLabel = computed(() => {
  if (state.selectedRun === null || state.selectedRun === undefined) {
    return "None";
  }
  return `Run ${state.selectedRun}`;
});
</script>
