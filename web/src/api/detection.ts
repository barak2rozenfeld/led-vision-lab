import type {
  ApiMethod,
  DetectionResponse,
  DetectionResult,
  MethodId,
  ResultSource,
} from "../types";
import { mockDetect } from "./mock";

const API_BASE: string = import.meta.env.VITE_API_BASE ?? "/api/v1";

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp"];

export class DetectionError extends Error {}

/** Client-side guard so obviously bad input never reaches the backend. */
export function validateImage(file: File): void {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    throw new DetectionError(
      `Unsupported file type "${file.type || "unknown"}". Use JPG, PNG, BMP or WebP.`,
    );
  }
  if (file.size === 0) throw new DetectionError("The file is empty.");
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new DetectionError(`Image is ${(file.size / 1048576).toFixed(1)} MB; the limit is 10 MB.`);
  }
}

export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal });
    return res.ok;
  } catch {
    return false;
  }
}

function isDetectionResult(value: unknown): value is DetectionResult {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    (v.state === "ON" || v.state === "OFF") &&
    typeof v.confidence === "number" &&
    typeof v.runtime_ms === "number" &&
    typeof v.method === "string"
  );
}

/** FastAPI errors are `{"detail": "..."}`; fall back to the raw body. */
async function errorMessage(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  try {
    const detail: unknown = JSON.parse(text).detail;
    if (typeof detail === "string") return detail;
  } catch {
    // not JSON
  }
  return `API returned ${res.status}${text ? `: ${text}` : ""}`;
}

async function postImage(path: string, image: Blob, fields: Record<string, string> = {}): Promise<unknown> {
  const body = new FormData();
  body.append("image", image, "upload");
  for (const [k, v] of Object.entries(fields)) body.append(k, v);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { method: "POST", body });
  } catch {
    throw new DetectionError("Could not reach the detection API. Is the server running?");
  }
  if (!res.ok) throw new DetectionError(await errorMessage(res));
  return res.json();
}

async function apiDetect(image: Blob, method: MethodId): Promise<DetectionResult> {
  const json = await postImage("/detect", image, { method });
  if (!isDetectionResult(json)) {
    throw new DetectionError("API response did not match the DetectionResult schema.");
  }
  return json;
}

async function apiCompare(image: Blob): Promise<DetectionResult[]> {
  const json = (await postImage("/compare", image)) as { results?: unknown };
  if (!Array.isArray(json.results) || !json.results.every(isDetectionResult)) {
    throw new DetectionError("API response did not match the compare schema.");
  }
  return json.results;
}

export async function detect(
  image: Blob,
  method: MethodId,
  source: ResultSource,
): Promise<DetectionResponse> {
  const result = source === "api" ? await apiDetect(image, method) : await mockDetect(image, method);
  return { result, source };
}

/** In API mode the server runs every *ready* method; in demo mode all methods are simulated. */
export async function compareAll(
  image: Blob,
  methods: readonly MethodId[],
  source: ResultSource,
): Promise<DetectionResponse[]> {
  if (source === "api") {
    return (await apiCompare(image)).map((result) => ({ result, source }));
  }
  return Promise.all(methods.map((m) => detect(image, m, source)));
}

export async function fetchMethods(signal?: AbortSignal): Promise<ApiMethod[]> {
  const res = await fetch(`${API_BASE}/methods`, { signal });
  if (!res.ok) throw new DetectionError(await errorMessage(res));
  return (await res.json()) as ApiMethod[];
}

/** Returns null when the server has no such artifact yet (404). */
export async function fetchArtifact<T>(path: "/dataset/stats" | "/benchmark", signal?: AbortSignal): Promise<T | null> {
  const res = await fetch(`${API_BASE}${path}`, { signal });
  if (res.status === 404) return null;
  if (!res.ok) throw new DetectionError(await errorMessage(res));
  return (await res.json()) as T;
}
