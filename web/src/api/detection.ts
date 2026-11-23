import type { DetectionResponse, DetectionResult, MethodId, ResultSource } from "../types";
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

async function apiDetect(image: Blob, method: MethodId): Promise<DetectionResult> {
  const body = new FormData();
  body.append("image", image, "upload");
  body.append("method", method);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/detect`, { method: "POST", body });
  } catch {
    throw new DetectionError("Could not reach the detection API. Is the server running?");
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new DetectionError(`API returned ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  const json: unknown = await res.json();
  if (!isDetectionResult(json)) {
    throw new DetectionError("API response did not match the DetectionResult schema.");
  }
  return json;
}

export async function detect(
  image: Blob,
  method: MethodId,
  source: ResultSource,
): Promise<DetectionResponse> {
  const result = source === "api" ? await apiDetect(image, method) : await mockDetect(image, method);
  return { result, source };
}

export async function compareAll(
  image: Blob,
  methods: readonly MethodId[],
  source: ResultSource,
): Promise<DetectionResponse[]> {
  return Promise.all(methods.map((m) => detect(image, m, source)));
}
