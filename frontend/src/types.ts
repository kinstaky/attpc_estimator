export interface SessionPayload {
  mode: "label" | "review";
  run: number | null;
  source?: "label_set" | "filter_file" | "event_trace" | null;
  family?: "normal" | "strange" | null;
  label?: string | null;
  filterFile?: string | null;
  eventId?: number | null;
  traceId?: number | null;
}

export interface EventIdRange {
  min: number;
  max: number;
}

export interface StrangeLabel {
  name: string;
  shortcutKey: string;
}

export interface NormalSummaryItem {
  bucket: number;
  title: string;
  count: number;
}

export interface StrangeSummaryItem {
  name: string;
  shortcutKey?: string;
  count: number;
}

export interface ReviewProgress {
  current: number;
  total: number;
}

export interface CurrentLabel {
  family: "normal" | "strange";
  label: string;
}

export interface BitflipAnalysis {
  xIndices: number[];
  firstDerivative: number[];
  secondDerivative: number[];
  structures: BitflipStructure[];
}

export interface BitflipStructure {
  startBaselineIndex: number;
  endBaselineIndex: number;
}

export interface TracePayload {
  run?: number;
  eventId: number;
  traceId: number;
  raw: number[];
  trace: number[];
  transformed: number[];
  bitflipAnalysis: BitflipAnalysis;
  currentLabel: CurrentLabel | null;
  reviewProgress: ReviewProgress | null;
  eventTraceCount: number | null;
  eventIdRange: EventIdRange | null;
}

export interface FilterFileItem {
  name: string;
}

export interface HistogramAvailabilityEntry {
  all: boolean;
  labeled: boolean;
  filtered: boolean;
}

export type HistogramMetric = "cdf" | "amplitude" | "baseline" | "bitflip" | "saturation";
export type HistogramMode = "all" | "labeled" | "filtered";
export type HistogramVariant =
  | "baseline"
  | "value"
  | "drop"
  | "length"
  | "count";

export interface BootstrapPayload {
  appType: "merged";
  workspace: string;
  tracePath: string;
  databaseFile: string;
  runs: number[];
  eventRanges: Record<string, EventIdRange>;
  filterFiles: FilterFileItem[];
  histogramAvailability: Record<
    string,
    Record<HistogramMetric, HistogramAvailabilityEntry>
  >;
  normalSummary: NormalSummaryItem[];
  strangeSummary: StrangeSummaryItem[];
  strangeLabels: StrangeLabel[];
  session: SessionPayload;
}

export interface HistogramSeries {
  labelKey: string;
  title: string;
  traceCount: number | null;
  histogram: number[] | number[][];
}

export interface HistogramPayload {
  metric: HistogramMetric;
  mode: HistogramMode;
  run: number;
  variant?: HistogramVariant | null;
  filterFile?: string | null;
  veto?: boolean;
  thresholds?: number[];
  valueBinCount?: number;
  binCount?: number;
  binCenters?: number[];
  binLabel?: string;
  countLabel?: string;
  series: HistogramSeries[];
}

export interface HistogramJobProgress {
  current: number;
  total: number;
  percent: number;
  unit: string;
  message: string;
}

export interface HistogramJobCreateResponse {
  jobId: string;
}

export interface HistogramJobProgressMessage extends HistogramJobProgress {
  type: "progress";
}

export interface HistogramJobCompleteMessage {
  type: "complete";
  payload: HistogramPayload;
}

export interface HistogramJobErrorMessage {
  type: "error";
  detail: string;
}

export type HistogramJobMessage =
  | HistogramJobProgressMessage
  | HistogramJobCompleteMessage
  | HistogramJobErrorMessage;

export interface LabelAssignResponse {
  labeledCount?: number;
  normalSummary: NormalSummaryItem[];
  strangeSummary: StrangeSummaryItem[];
  currentLabel: CurrentLabel;
}

export interface SessionResponse {
  session: SessionPayload;
  trace?: TracePayload | null;
  traceCount?: number;
}
