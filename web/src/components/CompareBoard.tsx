import { motion } from "framer-motion";
import { Cpu, FlaskConical, Layers3 } from "lucide-react";
import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DIFFICULTY_ORDER } from "../data/dataset";
import { METHODS, METHOD_BY_ID } from "../data/methods";
import { prettyLabel } from "../data/samples";
import type { Sourced } from "../hooks/useBackendData";
import type { BenchmarkReport, MethodBenchmark, MethodId, MethodStatus } from "../types";
import { ColabLink, ProvenanceNote, Section } from "./ui";

const tooltipStyle = {
  contentStyle: { background: "#0d151e", border: "1px solid #ffffff1a", borderRadius: 12 },
  labelStyle: { color: "#94a3b8" },
  itemStyle: { color: "#e6edf3" },
  cursor: { fill: "#ffffff08" },
};

const pct = (v: number) => Math.round(v * 1000) / 10;

function breakdown(
  measured: [MethodId, MethodBenchmark][],
  key: "by_difficulty" | "by_background_family",
  order?: readonly string[],
) {
  const groups = order ?? [...new Set(measured.flatMap(([, b]) => Object.keys(b[key])))].sort();
  return groups.map((g) => ({
    group: prettyLabel(g),
    ...Object.fromEntries(measured.map(([id, b]) => [id, b[key][g] ? pct(b[key][g].accuracy) : null])),
  }));
}

const STATUS_LABEL: Record<MethodStatus | "measured", string> = {
  measured: "Measured",
  ready: "Ready · not benchmarked",
  not_trained: "Not trained yet",
  planned: "Planned",
};

export function CompareBoard({
  benchmark,
  methodStatus,
}: {
  benchmark: Sourced<BenchmarkReport> | null;
  methodStatus: Partial<Record<MethodId, MethodStatus>> | null;
}) {
  const report = benchmark?.data;
  const measured = (report ? Object.entries(report.methods) : []) as [MethodId, MethodBenchmark][];
  const ranked = [...measured].sort((a, b) => b[1].overall.f1 - a[1].overall.f1);
  const statusOf = (id: MethodId): MethodStatus | "measured" =>
    report?.methods[id] ? "measured" : (methodStatus?.[id] ?? "planned");

  return (
    <Section
      id="compare"
      eyebrow="04 · Method comparison"
      title={
        <>
          Five detectors. <span className="text-amber">One benchmark.</span>
        </>
      }
      subtitle="Every method implements the same LedDetector interface and is scored by the same benchmark runner on a held-out test split it never saw during calibration or training."
    >
      <div className="mb-8 grid gap-4 md:grid-cols-5">
        {METHODS.map((m, i) => {
          const status = statusOf(m.id);
          return (
            <motion.div
              key={m.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
              whileHover={{ y: -4 }}
              className={`glass relative overflow-hidden rounded-2xl p-5 ${status === "measured" ? "" : "opacity-70"}`}
            >
              <div
                className="absolute -top-10 -right-10 h-28 w-28 rounded-full blur-2xl"
                style={{ backgroundColor: m.accent, opacity: 0.25 }}
              />
              <p className="font-mono text-[10px] tracking-widest uppercase" style={{ color: m.accent }}>
                {m.family}
              </p>
              <p className="mt-2 text-lg font-bold text-white">{m.name}</p>
              <p className="mt-2 text-sm text-slate-400">{m.blurb}</p>
              <p
                className={`mt-4 inline-block rounded-full border px-2 py-0.5 font-mono text-[10px] ${
                  status === "measured" ? "border-on/40 text-on" : "border-white/10 text-slate-500"
                }`}
              >
                {STATUS_LABEL[status]}
              </p>
              {m.notebook && (
                <div className="mt-2">
                  <ColabLink notebook={m.notebook} compact />
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {!report ? (
        <div className="flex items-start gap-3 rounded-2xl border border-amber/30 bg-amber/5 p-4 text-sm text-amber">
          <FlaskConical size={18} className="mt-0.5 shrink-0" />
          No benchmark report yet. Run{" "}
          <code className="font-mono">uv run python scripts/run_benchmark.py --dataset &lt;dir&gt; --split test</code>
        </div>
      ) : (
        <>
          <div className="mb-6">
            <ProvenanceNote
              origin={benchmark.origin}
              provenance={report.provenance}
              facts={[
                `${report.split} split`,
                `${report.evaluated_images.toLocaleString("en-US")} images`,
                report.dataset,
                `${report.image_size} px`,
              ]}
              command={`uv run python scripts/run_benchmark.py --dataset data/${report.dataset} --split ${report.split} --size ${report.image_size}`}
              notebook="hsv"
            />
          </div>

          <div className="mb-8 grid gap-5 lg:grid-cols-2">
            <BreakdownChart
              title="Accuracy by difficulty"
              icon={<Cpu size={16} className="text-on" />}
              data={breakdown(measured, "by_difficulty", DIFFICULTY_ORDER)}
              methods={measured.map(([id]) => id)}
            />
            <BreakdownChart
              title="Accuracy by surface"
              icon={<Layers3 size={16} className="text-signal" />}
              data={breakdown(measured, "by_background_family")}
              methods={measured.map(([id]) => id)}
            />
          </div>

          <div className="glass overflow-x-auto rounded-3xl">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-white/5 text-[11px] tracking-widest text-slate-500 uppercase">
                <tr>
                  {["Rank", "Method", "Accuracy", "F1", "Precision", "Recall", "False ON", "False OFF", "Latency p50", "Distractors → ON"].map(
                    (h) => (
                      <th key={h} className="px-5 py-4 whitespace-nowrap">
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody className="font-mono">
                {ranked.map(([id, b], i) => (
                  <tr key={id} className="border-b border-white/5 hover:bg-white/[0.02]">
                    <td className="px-5 py-4 text-slate-500">#{i + 1}</td>
                    <td className="px-5 py-4 font-sans font-semibold" style={{ color: METHOD_BY_ID[id].accent }}>
                      {METHOD_BY_ID[id].name}
                    </td>
                    <td className="px-5 py-4 text-white">{pct(b.overall.accuracy)}%</td>
                    <td className="px-5 py-4 text-white">{b.overall.f1.toFixed(3)}</td>
                    <td className="px-5 py-4 text-slate-300">{b.overall.precision.toFixed(3)}</td>
                    <td className="px-5 py-4 text-slate-300">{b.overall.recall.toFixed(3)}</td>
                    <td className="px-5 py-4 text-off">{b.overall.false_on}</td>
                    <td className="px-5 py-4 text-amber">{b.overall.false_off}</td>
                    <td className="px-5 py-4 text-slate-300">{b.latency.p50_ms.toFixed(2)} ms</td>
                    <td className="px-5 py-4 text-slate-300">
                      {b.distractor_scenes.predicted_on}/{b.distractor_scenes.count}
                    </td>
                  </tr>
                ))}
                {METHODS.filter((m) => !report.methods[m.id]).map((m) => (
                  <tr key={m.id} className="border-b border-white/5 text-slate-600 last:border-0">
                    <td className="px-5 py-4">–</td>
                    <td className="px-5 py-4 font-sans">{m.name}</td>
                    <td colSpan={8} className="px-5 py-4 font-sans text-xs">
                      {STATUS_LABEL[statusOf(m.id)]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Section>
  );
}

function BreakdownChart({
  title,
  icon,
  data,
  methods,
}: {
  title: string;
  icon: ReactNode;
  data: Record<string, string | number | null>[];
  methods: MethodId[];
}) {
  return (
    <div className="glass rounded-3xl p-6">
      <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
        {icon} {title}
      </div>
      <div className="h-72">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#ffffff10" vertical={false} />
            <XAxis dataKey="group" stroke="#64748b" fontSize={12} />
            <YAxis domain={[0, 100]} unit="%" stroke="#64748b" fontSize={12} />
            <Tooltip {...tooltipStyle} formatter={(v) => `${v} %`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {methods.map((id) => (
              <Bar
                key={id}
                dataKey={id}
                name={METHOD_BY_ID[id].name}
                fill={METHOD_BY_ID[id].accent}
                radius={[6, 6, 0, 0]}
                maxBarSize={48}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
