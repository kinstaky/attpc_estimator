import { createApp } from "vue";
import { Buffer } from "buffer";

import App from "./App.vue";
import vuetify from "./plugins/vuetify";
import router from "./router";
import "./styles.css";

const globalWithBuffer = globalThis as typeof globalThis & {
  Buffer?: typeof Buffer;
};

if (!globalWithBuffer.Buffer) {
  globalWithBuffer.Buffer = Buffer;
}

createApp(App).use(router).use(vuetify).mount("#app");
