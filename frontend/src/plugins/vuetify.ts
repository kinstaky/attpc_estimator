import "@mdi/font/css/materialdesignicons.css";
import "vuetify/styles";

import { createVuetify } from "vuetify";
import { aliases, mdi } from "vuetify/iconsets/mdi";
import {
  VAlert,
  VApp,
  VBtn,
  VBtnToggle,
  VCard,
  VCardActions,
  VCardText,
  VCardTitle,
  VChip,
  VCol,
  VContainer,
  VDialog,
  VIcon,
  VLayout,
  VList,
  VListItem,
  VMain,
  VNavigationDrawer,
  VProgressCircular,
  VProgressLinear,
  VRow,
  VSelect,
  VSpacer,
  VSwitch,
  VTextField,
} from "vuetify/components";
import { Ripple } from "vuetify/directives";

const vuetify = createVuetify({
  components: {
    VAlert,
    VApp,
    VBtn,
    VBtnToggle,
    VCard,
    VCardActions,
    VCardText,
    VCardTitle,
    VChip,
    VCol,
    VContainer,
    VDialog,
    VIcon,
    VLayout,
    VList,
    VListItem,
    VMain,
    VNavigationDrawer,
    VProgressCircular,
    VProgressLinear,
    VRow,
    VSelect,
    VSpacer,
    VSwitch,
    VTextField,
  },
  directives: {
    Ripple,
  },
  icons: {
    defaultSet: "mdi",
    aliases,
    sets: { mdi },
  },
  theme: {
    defaultTheme: "estimator",
    themes: {
      estimator: {
        dark: false,
        colors: {
          background: "#f7f7f5",
          surface: "#ffffff",
          surfaceVariant: "#f1f2f4",
          primary: "#1456c4",
          secondary: "#475569",
          accent: "#1456c4",
          info: "#1456c4",
          success: "#16734a",
          warning: "#a86400",
          error: "#b42318",
          "on-primary": "#ffffff",
          "on-surface": "#17191d",
        },
      },
    },
  },
});

export default vuetify;
