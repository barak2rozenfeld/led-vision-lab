import { useEffect, useState } from "react";
import { fetchArtifact, fetchMethods } from "../api/detection";
import type {
  BenchmarkReport,
  DataOrigin,
  DatasetStats,
  MethodId,
  MethodStatus,
  ResultSource,
} from "../types";

// Reports committed to the repo, bundled so the page shows measured numbers even without the API.
// `glob` (not a plain import) so a missing file yields `undefined` instead of a build error.
const STATS_SNAPSHOT = Object.values(
  import.meta.glob<DatasetStats>("../../../artifacts/dataset/stats.json", { eager: true, import: "default" }),
)[0];
const BENCHMARK_SNAPSHOT = Object.values(
  import.meta.glob<BenchmarkReport>("../../../artifacts/benchmarks/summary.json", {
    eager: true,
    import: "default",
  }),
)[0];

export interface Sourced<T> {
  data: T;
  origin: DataOrigin;
}

export interface BackendData {
  /** null in demo mode: every method is simulated, so none is "not trained". */
  methodStatus: Partial<Record<MethodId, MethodStatus>> | null;
  stats: Sourced<DatasetStats> | null;
  benchmark: Sourced<BenchmarkReport> | null;
}

const snapshot = <T,>(data: T | undefined): Sourced<T> | null =>
  data ? { data, origin: "snapshot" } : null;

export function useBackendData(source: ResultSource): BackendData {
  const [state, setState] = useState<BackendData>({
    methodStatus: null,
    stats: snapshot(STATS_SNAPSHOT),
    benchmark: snapshot(BENCHMARK_SNAPSHOT),
  });

  useEffect(() => {
    if (source !== "api") {
      setState({ methodStatus: null, stats: snapshot(STATS_SNAPSHOT), benchmark: snapshot(BENCHMARK_SNAPSHOT) });
      return;
    }
    const controller = new AbortController();
    const live = <T,>(data: T | null, fallback: T | undefined): Sourced<T> | null =>
      data ? { data, origin: "api" } : snapshot(fallback);

    Promise.all([
      fetchMethods(controller.signal),
      fetchArtifact<DatasetStats>("/dataset/stats", controller.signal),
      fetchArtifact<BenchmarkReport>("/benchmark", controller.signal),
    ])
      .then(([methods, stats, benchmark]) => {
        setState({
          methodStatus: Object.fromEntries(methods.map((m) => [m.id, m.status])),
          stats: live(stats, STATS_SNAPSHOT),
          benchmark: live(benchmark, BENCHMARK_SNAPSHOT),
        });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        console.error("Failed to load backend data", err);
      });
    return () => controller.abort();
  }, [source]);

  return state;
}
