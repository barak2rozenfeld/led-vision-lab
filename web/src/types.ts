export type LedState = "ON" | "OFF";

export type MethodId = "hsv" | "template" | "knn" | "cnn" | "yolo";

/** Normalised to the image: 0..1 on both axes. */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Mirrors the Python `DetectionResult` returned by `POST /api/v1/detect`. */
export interface DetectionResult {
  state: LedState;
  confidence: number;
  runtime_ms: number;
  method: MethodId;
  bbox?: BoundingBox | null;
  debug?: Record<string, unknown>;
}

export type ResultSource = "api" | "demo";

export interface DetectionResponse {
  result: DetectionResult;
  source: ResultSource;
}

export type MethodStatus = "ready" | "not_trained" | "planned";

/** `GET /api/v1/methods` */
export interface ApiMethod {
  id: MethodId;
  name: string;
  family: string;
  requires_training: boolean;
  status: MethodStatus;
}

export interface Provenance {
  generated_at: string;
  git_commit: string | null;
  git_dirty?: boolean | null;
  machine: string;
  python: string;
}

/** `GET /api/v1/dataset/stats` — written by scripts/compute_dataset_stats.py */
export interface DatasetStats {
  dataset: string;
  scenes: number;
  images: number;
  images_by_size: Record<string, number>;
  config: { seed?: number; total?: number; export_sizes?: number[] };
  counts: Record<string, Record<string, number>>;
  cross_tabs: Record<string, Record<string, number>>;
  provenance: Provenance;
}

export interface BinaryMetrics {
  n: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  false_on: number;
  false_off: number;
  confusion: { tp: number; tn: number; fp: number; fn: number };
}

export interface MethodBenchmark {
  overall: BinaryMetrics;
  latency: { mean_ms: number; p50_ms: number; p95_ms: number; max_ms: number };
  by_difficulty: Record<string, BinaryMetrics>;
  by_background_family: Record<string, BinaryMetrics>;
  distractor_scenes: { count: number; predicted_on: number };
}

/** `GET /api/v1/benchmark` — written by scripts/run_benchmark.py */
export interface BenchmarkReport {
  evaluated_images: number;
  missing_images: number;
  methods: Partial<Record<MethodId, MethodBenchmark>>;
  dataset: string;
  split: string;
  image_size: number;
  provenance: Provenance;
}

/** Where a number on screen came from: the live API or the snapshot committed to the repo. */
export type DataOrigin = "api" | "snapshot";

export interface SampleEntry {
  file: string;
  id: string;
  state: "on" | "off";
  background: string;
  led_color: string;
  led_shape: string;
  led_lens_type: string;
  led_brightness: string;
  difficulty: "easy" | "medium" | "hard" | "extreme";
  ambient_type: string;
  glare_type: string;
  occlusion_type: string;
  size: string;
}

export interface Sample extends SampleEntry {
  url: string;
}
