import type { MethodId } from "../types";

export interface MethodInfo {
  id: MethodId;
  name: string;
  family: string;
  training: boolean;
  phase: string;
  blurb: string;
  accent: string;
}

export const METHODS: MethodInfo[] = [
  {
    id: "hsv",
    name: "HSV Threshold",
    family: "Classical CV",
    training: false,
    phase: "Baseline",
    blurb: "Colour-space mask and brightness score. No training, near-zero latency.",
    accent: "#38bdf8",
  },
  {
    id: "template",
    name: "Template Diff",
    family: "Classical CV",
    training: false,
    phase: "Baseline",
    blurb: "Compares the ROI against ON/OFF reference images.",
    accent: "#a78bfa",
  },
  {
    id: "knn",
    name: "KNN Classifier",
    family: "Traditional ML",
    training: true,
    phase: "ML",
    blurb: "Hand-crafted colour and brightness features with nearest neighbours.",
    accent: "#fbbf24",
  },
  {
    id: "cnn",
    name: "Small CNN",
    family: "Deep Learning",
    training: true,
    phase: "DL",
    blurb: "Compact convolutional classifier trained on the synthetic dataset.",
    accent: "#fb7185",
  },
  {
    id: "yolo",
    name: "YOLOv8-nano",
    family: "Deep Learning (fine-tuned)",
    training: true,
    phase: "DL",
    blurb: "Pre-trained detector fine-tuned to localise LEDs and classify their state.",
    accent: "#34f5a0",
  },
];

export const METHOD_BY_ID = Object.fromEntries(METHODS.map((m) => [m.id, m])) as Record<
  MethodId,
  MethodInfo
>;

/**
 * Illustrative targets only — shown until the evaluation pipeline produces measured
 * numbers. Keep `BENCHMARK_IS_PLACEHOLDER` true until then.
 */
export const BENCHMARK_IS_PLACEHOLDER = true;

export interface BenchmarkRow {
  method: MethodId;
  accuracy: number;
  f1: number;
  latencyMs: number;
  byDifficulty: Record<"easy" | "medium" | "hard" | "extreme", number>;
}

export const BENCHMARK: BenchmarkRow[] = [
  { method: "hsv", accuracy: 0.84, f1: 0.82, latencyMs: 2, byDifficulty: { easy: 0.97, medium: 0.88, hard: 0.72, extreme: 0.55 } },
  { method: "template", accuracy: 0.81, f1: 0.79, latencyMs: 3, byDifficulty: { easy: 0.95, medium: 0.85, hard: 0.68, extreme: 0.52 } },
  { method: "knn", accuracy: 0.89, f1: 0.88, latencyMs: 7, byDifficulty: { easy: 0.98, medium: 0.92, hard: 0.81, extreme: 0.64 } },
  { method: "cnn", accuracy: 0.95, f1: 0.95, latencyMs: 15, byDifficulty: { easy: 0.99, medium: 0.97, hard: 0.92, extreme: 0.81 } },
  { method: "yolo", accuracy: 0.97, f1: 0.96, latencyMs: 23, byDifficulty: { easy: 0.995, medium: 0.98, hard: 0.95, extreme: 0.86 } },
];
