import { motion } from "framer-motion";
import { Cpu, FlaskConical, Gauge } from "lucide-react";
import {
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { BENCHMARK, BENCHMARK_IS_PLACEHOLDER, METHODS, METHOD_BY_ID } from "../data/methods";
import { Section } from "./ui";

const tooltipStyle = {
  contentStyle: { background: "#0d151e", border: "1px solid #ffffff1a", borderRadius: 12 },
  labelStyle: { color: "#94a3b8" },
  itemStyle: { color: "#e6edf3" },
};

const DIFFICULTY_SERIES = (["easy", "medium", "hard", "extreme"] as const).map((d) => ({
  difficulty: d[0].toUpperCase() + d.slice(1),
  ...Object.fromEntries(BENCHMARK.map((row) => [row.method, Math.round(row.byDifficulty[d] * 1000) / 10])),
}));

export function CompareBoard() {
  const ranked = [...BENCHMARK].sort((a, b) => b.f1 - a.f1);
  return (
    <Section
      id="compare"
      eyebrow="04 · Method comparison"
      title={
        <>
          Five detectors. <span className="text-amber">One benchmark.</span>
        </>
      }
      subtitle="Every method implements the same LedDetector interface and is scored on the same held-out test split: accuracy, F1, false ON / false OFF and latency, broken down by difficulty."
    >
      {BENCHMARK_IS_PLACEHOLDER && (
        <div className="mb-8 flex items-start gap-3 rounded-2xl border border-amber/30 bg-amber/5 p-4 text-sm text-amber">
          <FlaskConical size={18} className="mt-0.5 shrink-0" />
          These numbers are illustrative targets, not measurements. They will be replaced by the
          evaluation results once the detectors are trained and benchmarked.
        </div>
      )}

      <div className="mb-8 grid gap-4 md:grid-cols-5">
        {METHODS.map((m, i) => (
          <motion.div
            key={m.id}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.08 }}
            whileHover={{ y: -4 }}
            className="glass relative overflow-hidden rounded-2xl p-5"
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
            <p className="mt-4 text-xs text-slate-500">
              {m.training ? "Requires training" : "No training"}
            </p>
          </motion.div>
        ))}
      </div>

      <div className="mb-8 grid gap-5 lg:grid-cols-2">
        <div className="glass rounded-3xl p-6">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
            <Gauge size={16} className="text-signal" /> Accuracy vs latency
          </div>
          <div className="h-72">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                <CartesianGrid stroke="#ffffff10" />
                <XAxis
                  type="number"
                  dataKey="latencyMs"
                  name="Latency"
                  unit=" ms"
                  stroke="#64748b"
                  fontSize={12}
                  label={{ value: "latency (ms)", position: "insideBottom", offset: -10, fill: "#64748b", fontSize: 12 }}
                />
                <YAxis
                  type="number"
                  dataKey="accuracy"
                  name="Accuracy"
                  domain={[0.75, 1]}
                  tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                  stroke="#64748b"
                  fontSize={12}
                />
                <ZAxis type="number" dataKey="z" range={[220, 220]} />
                <Tooltip {...tooltipStyle} cursor={{ stroke: "#ffffff20" }} />
                {BENCHMARK.map((row) => (
                  <Scatter
                    key={row.method}
                    name={METHOD_BY_ID[row.method].name}
                    data={[{ ...row, z: 1, label: METHOD_BY_ID[row.method].name }]}
                    fill={METHOD_BY_ID[row.method].accent}
                  >
                    <LabelList dataKey="label" position="top" offset={12} fill="#cbd5e1" fontSize={11} />
                  </Scatter>
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass rounded-3xl p-6">
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
            <Cpu size={16} className="text-on" /> Accuracy by difficulty
          </div>
          <div className="h-72">
            <ResponsiveContainer>
              <LineChart data={DIFFICULTY_SERIES} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#ffffff10" />
                <XAxis dataKey="difficulty" stroke="#64748b" fontSize={12} />
                <YAxis domain={[50, 100]} unit="%" stroke="#64748b" fontSize={12} />
                <Tooltip {...tooltipStyle} />
                {METHODS.map((m) => (
                  <Line
                    key={m.id}
                    type="monotone"
                    dataKey={m.id}
                    name={m.name}
                    stroke={m.accent}
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: m.accent }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="glass overflow-x-auto rounded-3xl">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-white/5 text-[11px] tracking-widest text-slate-500 uppercase">
            <tr>
              <th className="px-6 py-4">Rank</th>
              <th className="px-6 py-4">Method</th>
              <th className="px-6 py-4">Accuracy</th>
              <th className="px-6 py-4">F1</th>
              <th className="px-6 py-4">Latency</th>
              <th className="px-6 py-4">Extreme cases</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((row, i) => {
              const m = METHOD_BY_ID[row.method];
              return (
                <tr key={row.method} className="border-b border-white/5 last:border-0 hover:bg-white/[0.02]">
                  <td className="px-6 py-4 font-mono text-slate-500">#{i + 1}</td>
                  <td className="px-6 py-4 font-semibold" style={{ color: m.accent }}>
                    {m.name}
                  </td>
                  <td className="px-6 py-4 font-mono text-white">{(row.accuracy * 100).toFixed(1)}%</td>
                  <td className="px-6 py-4 font-mono text-white">{row.f1.toFixed(2)}</td>
                  <td className="px-6 py-4 font-mono text-slate-300">{row.latencyMs} ms</td>
                  <td className="px-6 py-4">
                    <div className="h-2 w-32 overflow-hidden rounded-full bg-white/5">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${row.byDifficulty.extreme * 100}%`, backgroundColor: m.accent }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}
