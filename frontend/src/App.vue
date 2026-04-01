<template>
  <section v-if="loading" class="welcome-panel root-panel">
    <p class="eyebrow">Application</p>
    <h1>Loading app state…</h1>
  </section>

  <section v-else-if="error" class="welcome-panel root-panel">
    <p class="eyebrow">Application</p>
    <h1>Failed to load the backend bootstrap.</h1>
    <p class="welcome-copy">{{ error }}</p>
  </section>

  <LabelApp v-else-if="bootstrap?.appType === 'label'" :bootstrap="bootstrap" />
  <ViewerApp v-else-if="bootstrap?.appType === 'viewer'" :bootstrap="bootstrap" />

  <section v-else class="welcome-panel root-panel">
    <p class="eyebrow">Application</p>
    <h1>Unsupported app type.</h1>
  </section>
</template>

<script setup>
import { onMounted, ref } from "vue";
import LabelApp from "./apps/LabelApp.vue";
import ViewerApp from "./apps/ViewerApp.vue";
import { getBootstrap } from "./api";

const loading = ref(true);
const error = ref("");
const bootstrap = ref(null);

onMounted(async () => {
  try {
    bootstrap.value = await getBootstrap();
  } catch (caughtError) {
    error.value = caughtError.message;
  } finally {
    loading.value = false;
  }
});
</script>
