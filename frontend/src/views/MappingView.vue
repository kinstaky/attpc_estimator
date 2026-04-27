<template>
  <v-container class="page-container" fluid>
    <div class="page-header">
      <div>
        <p class="page-kicker">Mapping</p>
        <h1>Detector pad mapping</h1>
        <p class="page-copy">
          Inspect the packaged detector geometry, recolor pads by hardware rules, and flip between upstream and downstream views.
        </p>
      </div>
    </div>

    <v-row dense>
      <v-col cols="12" lg="3">
        <v-card class="control-card mapping-sidebar-card" rounded="xl">
          <v-card-title class="result-card-title">
            <div>
              <p class="page-kicker">Rules</p>
              <h2>Color matching</h2>
            </div>
            <v-btn color="primary" variant="tonal" @click="store.openNewRule">
              New rule
            </v-btn>
          </v-card-title>
          <v-card-text>
            <p class="page-copy mapping-rule-copy">
              Rules are checked from top to bottom. The first matching rule decides the pad color.
            </p>

            <div v-if="store.state.rules.length" class="mapping-rule-list">
              <article
                v-for="(rule, index) in store.state.rules"
                :key="`${rule.cobo}-${rule.asad}-${rule.aget}-${rule.channel}-${index}`"
                class="mapping-rule-item"
              >
                <span
                  class="mapping-rule-swatch"
                  :style="{ backgroundColor: rule.color }"
                />
                <div class="mapping-rule-content">
                  <strong>{{ rule.color }}</strong>
                  <span>Cobo {{ rule.cobo }} · Asad {{ rule.asad }} · Aget {{ rule.aget }} · Ch {{ rule.channel }}</span>
                </div>
                <div class="mapping-rule-actions">
                  <v-btn
                    icon="mdi-pencil-outline"
                    size="small"
                    variant="text"
                    @click="store.openEditRule(index)"
                  />
                  <v-btn
                    icon="mdi-delete-outline"
                    size="small"
                    variant="text"
                    @click="store.deleteRule(index)"
                  />
                </div>
              </article>
            </div>
            <div v-else class="mapping-rule-empty">
              <p class="page-kicker">No rules</p>
              <p class="page-copy">Add a rule to highlight hardware channels with custom colors.</p>
            </div>
          </v-card-text>
        </v-card>
      </v-col>

      <v-col cols="12" lg="9">
        <v-card class="trace-stage-card mapping-stage-card" rounded="xl">
          <v-card-title class="result-card-title">
            <div>
              <p class="page-kicker">Pads</p>
              <h2>{{ store.state.selectedLayer === "Pads" ? "Detector footprint" : `${store.state.selectedLayer} unavailable` }}</h2>
            </div>
            <v-chip variant="tonal">{{ store.state.pads.length }} pads</v-chip>
          </v-card-title>
          <v-card-text>
            <div class="mapping-stage-toolbar">
              <div class="mapping-stage-toolbar-group">
                <p class="page-kicker">Layer</p>
                <v-btn-toggle
                  :model-value="store.state.selectedLayer"
                  class="mapping-toggle"
                  color="primary"
                  divided
                  mandatory
                  @update:model-value="store.setSelectedLayer"
                >
                  <v-btn
                    v-for="layer in layers"
                    :key="layer"
                    :value="layer"
                    class="text-none"
                  >
                    {{ layer }}
                  </v-btn>
                </v-btn-toggle>
              </div>

              <div class="mapping-stage-toolbar-group">
                <p class="page-kicker">View</p>
                <v-btn-toggle
                  :model-value="store.state.selectedView"
                  class="mapping-toggle"
                  color="primary"
                  divided
                  mandatory
                  @update:model-value="store.setSelectedView"
                >
                  <v-btn
                    v-for="viewMode in views"
                    :key="viewMode"
                    :value="viewMode"
                    class="text-none"
                  >
                    {{ viewMode }}
                  </v-btn>
                </v-btn-toggle>
              </div>
            </div>

            <v-alert
              v-if="store.state.error"
              class="mb-4 mt-4"
              color="error"
              icon="mdi-alert-circle-outline"
              rounded="xl"
              variant="tonal"
            >
              {{ store.state.error }}
            </v-alert>

            <div v-else-if="store.state.loading" class="empty-state">
              <v-progress-circular color="primary" indeterminate />
            </div>

            <v-alert
              v-else-if="store.state.selectedLayer !== 'Pads'"
              class="mt-4"
              color="warning"
              icon="mdi-layers-outline"
              rounded="xl"
              variant="tonal"
            >
              {{ store.state.selectedLayer }} is not wired to packaged geometry yet. Switch back to Pads to inspect the detector mapping.
            </v-alert>

            <MappingCanvas
              v-else
              :pads="store.state.pads"
              :rules="store.state.rules"
              :view="store.state.selectedView"
            />
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <MappingRuleDialog
      :model-value="store.state.dialogOpen"
      :index="store.state.editingIndex"
      :initial-rule="store.editingRule.value"
      @update:model-value="store.setDialogOpen"
      @save="store.saveRule"
    />
  </v-container>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import MappingCanvas from "../components/MappingCanvas.vue";
import MappingRuleDialog from "../components/MappingRuleDialog.vue";
import { useMappingStore } from "../stores/mapping";

const store = useMappingStore();
const layers = ["Pads", "Si-0", "Si-1"];
const views = ["Upstream", "Downstream"];

onMounted(() => {
  void store.loadPads();
});
</script>
