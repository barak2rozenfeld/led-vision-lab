import type { DetectionResult, LedState, MethodId } from "../types";

/**
 * Demo-mode stand-in for the backend. It does NOT look at the image content:
 * results are pseudo-random but deterministic per (image bytes, method), so the
 * UI can be exercised before the Python detectors and API exist.
 */

const TYPICAL_RUNTIME_MS: Record<MethodId, number> = {
  hsv: 2.1,
  template: 3.4,
  knn: 6.8,
  cnn: 14.5,
  yolo: 23.0,
};

function fnv1a(bytes: Uint8Array, seed: number): number {
  let hash = 0x811c9dc5 ^ seed;
  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i];
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const METHOD_SEED: Record<MethodId, number> = { hsv: 1, template: 2, knn: 3, cnn: 4, yolo: 5 };

export async function mockDetect(image: Blob, method: MethodId): Promise<DetectionResult> {
  const head = new Uint8Array(await image.slice(0, 65536).arrayBuffer());
  const imageHash = fnv1a(head, image.size);
  const rand = mulberry32(imageHash);
  const sceneOn = rand() > 0.5;

  // Each method agrees with the "scene" most of the time, like real detectors would.
  const methodRand = mulberry32(imageHash ^ (METHOD_SEED[method] * 0x9e3779b9));
  const agrees = methodRand() < 0.85;
  const state: LedState = sceneOn === agrees ? "ON" : "OFF";

  const size = 0.12 + rand() * 0.18;
  const runtime = TYPICAL_RUNTIME_MS[method] * (0.8 + methodRand() * 0.5);

  await new Promise((resolve) => setTimeout(resolve, 250 + runtime * 8));

  return {
    state,
    confidence: 0.62 + methodRand() * 0.37,
    runtime_ms: Math.round(runtime * 10) / 10,
    method,
    bbox: {
      x: 0.08 + rand() * (0.84 - size),
      y: 0.08 + rand() * (0.84 - size),
      width: size,
      height: size,
    },
    debug: { mode: "demo" },
  };
}
