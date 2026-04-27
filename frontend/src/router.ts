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
    path: "/label/pointcloud",
    name: "label-pointcloud",
    component: () => import("./views/PointcloudLabelView.vue"),
  },
  {
    path: "/histograms",
    name: "histograms",
    component: () => import("./views/HistogramView.vue"),
  },
  {
    path: "/mapping",
    name: "mapping",
    component: () => import("./views/MappingView.vue"),
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
    path: "/browse/pointcloud",
    name: "browse-pointcloud",
    component: () => import("./views/PointcloudView.vue"),
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
