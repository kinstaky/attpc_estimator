import type {
  BootstrapPayload,
  HistogramJobCreateResponse,
  HistogramMetric,
  HistogramPayload,
  HistogramVariant,
  LabelAssignResponse,
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

export function setSession(payload: SessionPayload): Promise<SessionResponse> {
  return request<SessionResponse>("/api/session", {
    method: "POST",
    body: JSON.stringify(payload),
  });
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
