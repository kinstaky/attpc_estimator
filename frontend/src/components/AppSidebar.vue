<template>
  <aside class="nav-sidebar">
    <div class="nav-sidebar-header">
      <h1>{{ title }}</h1>
    </div>

    <nav class="nav-sidebar-links">
      <button
        v-for="item in items"
        :key="item.id"
        type="button"
        class="nav-sidebar-link"
        :class="{ active: activePage === item.id }"
        @click="$emit('select', item.id)"
      >
        <span class="nav-sidebar-link-title">{{ item.title }}</span>
      </button>
    </nav>
  </aside>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  items: { type: Array, required: true },
  activePage: { type: String, required: true },
});

defineEmits(["select"]);
</script>
