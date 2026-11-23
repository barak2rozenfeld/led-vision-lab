import { useEffect, useState } from "react";
import { checkHealth } from "../api/detection";
import type { ResultSource } from "../types";

const POLL_MS = 15_000;

/** "api" when `GET /api/v1/health` answers, otherwise "demo" (mock results). */
export function useApiStatus(): { source: ResultSource; checking: boolean } {
  const [source, setSource] = useState<ResultSource>("demo");
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const probe = async () => {
      const ok = await checkHealth(controller.signal);
      if (cancelled) return;
      setSource(ok ? "api" : "demo");
      setChecking(false);
    };

    void probe();
    const timer = window.setInterval(probe, POLL_MS);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  return { source, checking };
}
