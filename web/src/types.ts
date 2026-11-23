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
