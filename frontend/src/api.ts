import type {
  BootstrapPayload,
  HistogramJobCreateResponse,
  HistogramMetric,
  HistogramPayload,
  HistogramVariant,
  LabelAssignResponse,
  MappingPad,
  PointcloudEventPayload,
  PointcloudLabelEventPayload,
  PointcloudTracePayload,
  UiStatePayload,
  SessionPayload,
  SessionResponse,
  StrangeLabel,
  TracePayload,
} from "./types";

interface ErrorBody {
  detail?: string;
}

interface LabelAssignRequest {
  eventId: number;
  traceId: number;
  family: "normal" | "strange";
  label: string;
}

interface CreateStrangeLabelRequest {
  name: string;
  shortcutKey: string;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as ErrorBody;
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // Keep the generic message when the body is not JSON.
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return null as T;
  }

  return (await response.json()) as T;
}

export function getBootstrap(): Promise<BootstrapPayload> {
  return request<BootstrapPayload>("/api/bootstrap");
}

export function getMappingPads(): Promise<MappingPad[]> {
  return request<MappingPad[]>("/api/mapping/pads");
}

export function setSession(payload: SessionPayload): Promise<SessionResponse> {
  return request<SessionResponse>("/api/session", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getCurrentTrace(): Promise<TracePayload> {
  return request<TracePayload>("/api/traces/current");
}

export function getStrangeLabels(): Promise<{ strangeLabels: StrangeLabel[] }> {
  return request<{ strangeLabels: StrangeLabel[] }>("/api/labels/strange");
}

export function nextTrace(): Promise<TracePayload> {
  return request<TracePayload>("/api/traces/next", {
    method: "POST",
  });
}

export function nextEvent(): Promise<TracePayload> {
  return request<TracePayload>("/api/traces/next-event", {
    method: "POST",
  });
}

export function setLabelSession(run: number): Promise<SessionResponse> {
  return setSession({ mode: "label", run });
}

export function setLabelReviewRelabelSession(
  run: number,
  family: "normal" | "strange",
  label: string | null = null,
): Promise<SessionResponse> {
  return setSession({
    mode: "label_review",
    run,
    source: "label_set",
    family,
    label,
  });
}

export function setPointcloudLabelSession(run: number): Promise<SessionResponse> {
  return setSession({ mode: "pointcloud_label", run });
}

export function setPointcloudLabelReviewSession(
  run: number,
  label: string | null = null,
): Promise<SessionResponse> {
  return setSession({
    mode: "pointcloud_label_review",
    run,
    source: "label_set",
    label,
  });
}

export function setPointcloudBrowseSession(
  run: number,
  source: "event_id" | "label_set",
  eventId: number | null = null,
  label: string | null = null,
): Promise<SessionResponse> {
  return setSession({
    mode: "pointcloud",
    run,
    source,
    eventId,
    label,
  });
}

export function setLabelReviewSession(
  run: number,
  family: "normal" | "strange",
  label: string | null = null,
): Promise<SessionResponse> {
  return setSession({
    mode: "review",
    run,
    source: "label_set",
    family,
    label,
  });
}

export function setFilterReviewSession(filterFile: string): Promise<SessionResponse> {
  return setSession({
    mode: "review",
    run: null,
    source: "filter_file",
    filterFile,
  });
}

export function setEventTraceReviewSession(
  run: number,
  eventId: number,
  traceId: number,
): Promise<SessionResponse> {
  return setSession({
    mode: "review",
    run,
    source: "event_trace",
    eventId,
    traceId,
  });
}

export function previousTrace(): Promise<TracePayload> {
  return request<TracePayload>("/api/traces/previous", {
    method: "POST",
  });
}

export function previousEvent(): Promise<TracePayload> {
  return request<TracePayload>("/api/traces/previous-event", {
    method: "POST",
  });
}

export function saveLabel(
  eventId: number,
  traceId: number,
  family: "normal" | "strange",
  label: string,
): Promise<LabelAssignResponse> {
  const payload: LabelAssignRequest = { eventId, traceId, family, label };
  return request<LabelAssignResponse>("/api/labels/assign", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createStrangeLabel(
  name: string,
  shortcutKey: string,
): Promise<StrangeLabel> {
  const payload: CreateStrangeLabelRequest = { name, shortcutKey };
  return request<StrangeLabel>("/api/labels/strange", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function deleteStrangeLabel(name: string): Promise<Array<{ name: string; count: number }>> {
  return request<Array<{ name: string; count: number }>>(
    `/api/labels/strange/${encodeURIComponent(name)}`,
    {
      method: "DELETE",
    },
  );
}

export function getHistogram(
  metric: HistogramMetric,
  mode: "all" | "labeled" | "filtered",
  run: number,
  variant: HistogramVariant | "" = "",
  filterFile = "",
  veto = false,
): Promise<HistogramPayload> {
  const params = new URLSearchParams({
    metric,
    mode,
    run: String(run),
  });
  if (variant) {
    params.set("variant", variant);
  }
  if (filterFile) {
    params.set("filterFile", filterFile);
  }
  if (veto) {
    params.set("veto", "true");
  }
  return request<HistogramPayload>(`/api/histograms?${params.toString()}`);
}

export function createHistogramJob(
  metric: HistogramMetric,
  mode: "filtered",
  run: number,
  variant: HistogramVariant | "" = "",
  filterFile: string,
  veto = false,
): Promise<HistogramJobCreateResponse> {
  return request<HistogramJobCreateResponse>("/api/histograms/jobs", {
    method: "POST",
    body: JSON.stringify({
      metric,
      mode,
      run,
      variant: variant || undefined,
      filterFile,
      veto,
    }),
  });
}

export function histogramJobSocketUrl(jobId: string): string {
  const url = new URL(`/api/histograms/jobs/${jobId}`, window.location.href);
  url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export function getPointcloudEvent(
  run: number,
  eventId: number,
): Promise<PointcloudEventPayload> {
  const params = new URLSearchParams({
    run: String(run),
    eventId: String(eventId),
  });
  return request<PointcloudEventPayload>(`/api/pointcloud/event?${params.toString()}`);
}

export function getCurrentPointcloudEvent(): Promise<PointcloudEventPayload> {
  return request<PointcloudEventPayload>("/api/pointcloud/current");
}

export function nextPointcloudEvent(): Promise<PointcloudEventPayload> {
  return request<PointcloudEventPayload>("/api/pointcloud/next", {
    method: "POST",
  });
}

export function previousPointcloudEvent(): Promise<PointcloudEventPayload> {
  return request<PointcloudEventPayload>("/api/pointcloud/previous", {
    method: "POST",
  });
}

export function getPointcloudTraces(
  run: number,
  eventId: number,
  traceIds: number[],
): Promise<PointcloudTracePayload> {
  return request<PointcloudTracePayload>("/api/pointcloud/traces", {
    method: "POST",
    body: JSON.stringify({
      run,
      eventId,
      traceIds,
    }),
  });
}

export function getCurrentPointcloudLabelEvent(): Promise<PointcloudLabelEventPayload> {
  return request<PointcloudLabelEventPayload>("/api/pointcloud-label/current");
}

export function nextPointcloudLabelEvent(): Promise<PointcloudLabelEventPayload> {
  return request<PointcloudLabelEventPayload>("/api/pointcloud-label/next", {
    method: "POST",
  });
}

export function previousPointcloudLabelEvent(): Promise<PointcloudLabelEventPayload> {
  return request<PointcloudLabelEventPayload>("/api/pointcloud-label/previous", {
    method: "POST",
  });
}

export function savePointcloudLabel(
  eventId: number,
  label: string,
): Promise<{ pointcloudSummary: Array<{ bucket: string; title: string; count: number }>; currentLabel: string }> {
  return request<{ pointcloudSummary: Array<{ bucket: string; title: string; count: number }>; currentLabel: string }>(
    "/api/pointcloud-label/assign",
    {
      method: "POST",
      body: JSON.stringify({ eventId, label }),
    },
  );
}

export function updateUiState(payload: UiStatePayload): Promise<{ uiState: UiStatePayload }> {
  return request<{ uiState: UiStatePayload }>("/api/ui-state", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
