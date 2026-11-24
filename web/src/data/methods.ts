import type { MethodId } from "../types";
import type { NotebookId } from "./notebooks";

export interface MethodInfo {
  id: MethodId;
  name: string;
  family: string;
  training: boolean;
  phase: string;
  blurb: string;
  accent: string;
  notebook?: NotebookId;
}

export const METHODS: MethodInfo[] = [
  {
    id: "hsv",
    name: "HSV Threshold",
    family: "Classical CV",
    training: false,
    phase: "Baseline",
    blurb: "Brightness above the local background, a colour bonus and a glow check. No training.",
    accent: "#38bdf8",
    notebook: "hsv",
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
