async function request(path, options = {}) {
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
      const body = await response.json();
      if (body?.detail) {
        detail = body.detail;
      }
    } catch {
      // Ignore JSON parsing failures and keep the generic message.
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

export function getBootstrap() {
  return request("/api/bootstrap");
}

export function getStrangeLabels() {
  return request("/api/label/strange");
}

export function nextTrace() {
  return request("/api/trace/next", {
    method: "POST",
  });
}

export function setTraceMode(mode, family = null, label = null) {
  return request("/api/trace/mode", {
    method: "POST",
    body: JSON.stringify({ mode, family, label }),
  });
}

export function previousTrace() {
  return request("/api/trace/previous", {
    method: "POST",
  });
}

export function saveLabel(eventId, traceId, family, label) {
  return request("/api/trace/label", {
    method: "POST",
    body: JSON.stringify({ eventId, traceId, family, label }),
  });
}

export function createStrangeLabel(name, shortcutKey) {
  return request("/api/label/strange", {
    method: "POST",
    body: JSON.stringify({ name, shortcutKey }),
  });
}

export function deleteStrangeLabel(name) {
  return request(`/api/label/strange/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export function getHistogram(metric, mode, run, filterFile = "") {
  const params = new URLSearchParams({
    metric,
    mode,
    run: String(run),
  });
  if (filterFile) {
    params.set("filterFile", filterFile);
  }
  return request(`/api/histogram?${params.toString()}`);
}

export function selectReviewFilter(name) {
  return request("/api/review/filter", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function nextReviewTrace() {
  return request("/api/review/next", {
    method: "POST",
  });
}

export function previousReviewTrace() {
  return request("/api/review/previous", {
    method: "POST",
  });
}
