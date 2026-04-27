export interface SessionPayload {
  mode: "label" | "label_review" | "review" | "pointcloud_label" | "pointcloud_label_review" | "pointcloud";
  run: number | null;
  source?: "label_set" | "filter_file" | "event_trace" | "event_id" | null;
  family?: "normal" | "strange" | null;
  label?: string | null;
  filterFile?: string | null;
  eventId?: number | null;
  traceId?: number | null;
}

export interface ShellUiState {
  selectedRun: number | null;
}

export interface LabelUiState {
  visualMode: "raw" | "cdf" | "curvature";
}

export interface ReviewUiState {
  source: "label_set" | "filter_file" | "event_trace";
  run: number | null;
  family: "normal" | "strange";
  label: string;
  filterFile: string;
  eventId: number | null;
  traceId: number | null;
  visualMode: "raw" | "cdf" | "curvature";
}

export interface HistogramsUiState {
  selectedRun: number | null;
  selectedPhase: "phase1" | "phase2";
  selectedMetric:
    | "cdf"
    | "amplitude"
    | "baseline"
    | "bitflip"
    | "saturation"
    | "line_distance"
    | "line_property"
    | "coplanar";
  selectedMode: "all" | "labeled" | "filtered";
  selectedBitflipVariant: "baseline" | "value" | "length" | "count";
  selectedSaturationVariant: "drop" | "length";
  selectedHistogramFilter: string;
  selectedHistogramVeto: boolean;
  cdfScaleMode: "linear" | "log";
  amplitudeScaleMode: "linear" | "log";
  cdfRenderMode: "2d" | "projection";
  cdfProjectionBin: number;
  labeledSeriesOrder: Record<string, string[]>;
}

export interface MappingUiState {
  selectedLayer: "Pads" | "Si-0" | "Si-1";
  selectedView: "Upstream" | "Downstream";
  rules: MappingRenderRule[];
}

export interface PointcloudUiState {
  source: "event_id" | "label_set";
  selectedRun: number | null;
  selectedEventId: number | null;
  selectedLabel: string;
  layoutMode: "1x1" | "2x2";
  panelTypes: string[];
  selectedTraceIds: number[];
}

export interface PointcloudLabelUiState {
  visualMode: "basic" | "detail";
}

export interface UiStatePayload {
  route: string;
  shell: ShellUiState;
  label: LabelUiState;
  review: ReviewUiState;
  histograms: HistogramsUiState;
  mapping: MappingUiState;
  pointcloud: PointcloudUiState;
  pointcloudLabel: PointcloudLabelUiState;
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

export interface PointcloudSummaryItem {
  bucket: string;
  title: string;
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

export type HistogramPhase = "phase1" | "phase2";
export type HistogramMetric =
  | "cdf"
  | "amplitude"
  | "baseline"
  | "bitflip"
  | "saturation"
  | "line_distance"
  | "line_property"
  | "coplanar";
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
  pointcloudRuns: number[];
  pointcloudEventRanges: Record<string, EventIdRange>;
  filterFiles: FilterFileItem[];
  histogramAvailability: Record<
    string,
    Record<HistogramMetric, HistogramAvailabilityEntry>
  >;
  normalSummary: NormalSummaryItem[];
  pointcloudSummary: PointcloudSummaryItem[];
  strangeSummary: StrangeSummaryItem[];
  strangeLabels: StrangeLabel[];
  session: SessionPayload;
  uiState: UiStatePayload;
}

export interface HistogramSeries {
  labelKey: string;
  title: string;
  traceCount: number | null;
  histogram: number[] | number[][];
}

export interface HistogramPlotSeries {
  labelKey: string;
  title: string;
  histogram: number[];
}

export interface HistogramPlot {
  key: string;
  render: "bar" | "heatmap" | "grouped_bar";
  title: string;
  histogram?: number[] | number[][];
  binCenters?: number[];
  binLabel?: string;
  countLabel?: string;
  xBinCenters?: number[];
  yBinCenters?: number[];
  xLabel?: string;
  yLabel?: string;
  series?: HistogramPlotSeries[];
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
  plots?: HistogramPlot[];
  summary?: Record<string, number>;
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

export type MappingLayer = "Pads" | "Si-0" | "Si-1";
export type MappingViewMode = "Upstream" | "Downstream";

export interface MappingPad {
  pad: number;
  scale: number;
  direction: number;
  cobo: number;
  asad: number;
  aget: number;
  channel: number;
  cx: number;
  cy: number;
}

export interface MappingRenderRule {
  cobo: string;
  asad: string;
  aget: string;
  channel: string;
  color: string;
}

export interface LabelAssignResponse {
  labeledCount?: number;
  normalSummary: NormalSummaryItem[];
  strangeSummary: StrangeSummaryItem[];
  currentLabel: CurrentLabel;
}

export interface SessionResponse {
  session: SessionPayload;
  trace?: TracePayload | null;
  event?: PointcloudLabelEventPayload | null;
  traceCount?: number;
}

export interface PointcloudProcessing {
  fftWindowScale: number;
  micromegasTimeBucket: number;
  windowTimeBucket: number;
  detectorLength: number;
}

export interface PointcloudHit {
  traceId: number;
  x: number;
  y: number;
  z: number;
  xPrime: number | null;
  yPrime: number | null;
  amplitude: number;
  integral: number;
  padId: number;
  timeBucket: number;
  scale: number;
  mergedLabel: number;
}

export interface PointcloudEventPayload {
  run: number;
  eventId: number;
  eventIdRange: EventIdRange;
  hits: PointcloudHit[];
  processing: PointcloudProcessing;
}

export interface PointcloudLabelEventPayload extends PointcloudEventPayload {
  mergedLineCount: number;
  suggestedLabel: string;
  currentLabel: string | null;
}

export interface PointcloudPeak {
  timeBucket: number;
  amplitude: number;
  integral: number;
  z: number;
  padId: number;
}

export interface PointcloudTraceSeries {
  traceId: number;
  raw: number[];
  trace: number[];
  peaks: PointcloudPeak[];
}

export interface PointcloudTracePayload {
  run: number;
  eventId: number;
  baselineWindowScale: number;
  traces: PointcloudTraceSeries[];
}
