import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    name: "home",
    component: () => import("./views/WelcomeView.vue"),
  },
  {
    path: "/label",
    name: "label",
    redirect: { name: "label-trace" },
  },
  {
    path: "/label/trace",
    name: "label-trace",
    component: () => import("./views/LabelView.vue"),
  },
  {
    path: "/browse",
    name: "browse",
    redirect: { name: "browse-trace" },
  },
  {
    path: "/browse/trace",
    name: "browse-trace",
    component: () => import("./views/TraceReviewView.vue"),
  },
  {
    path: "/mapping",
    name: "mapping",
    component: () => import("./views/MappingView.vue"),
  },
  {
    path: "/histograms",
    name: "histograms",
    component: () => import("./views/HistogramView.vue"),
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
