import { AnimatePresence, motion } from "framer-motion";
import { ScanLine, X } from "lucide-react";
import { useMemo, useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { DIFFICULTY_COLOR, DIFFICULTY_ORDER, FAMILY_COLOR } from "../data/dataset";
import { BACKGROUND_FAMILY, SAMPLES, prettyLabel } from "../data/samples";
import type { Sourced } from "../hooks/useBackendData";
import type { DatasetStats, Sample } from "../types";
import { ProvenanceNote, Section, StateBadge } from "./ui";

const share = (counts: Record<string, number>, key: string) => {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return total ? Math.round(((counts[key] ?? 0) / total) * 1000) / 10 : 0;
};

const FAMILIES = ["All", "PCB", "Enclosure", "Plastic", "Metal", "Studio"] as const;
const STATES = ["All", "on", "off"] as const;
const DIFFICULTIES = ["All", "easy", "medium"] as const;

export function DatasetExplorer({
  stats,
  onDetect,
}: {
  stats: Sourced<DatasetStats> | null;
  onDetect: (sample: Sample) => void;
}) {
  const [family, setFamily] = useState<(typeof FAMILIES)[number]>("All");
  const [state, setState] = useState<(typeof STATES)[number]>("All");
  const [difficulty, setDifficulty] = useState<(typeof DIFFICULTIES)[number]>("All");
  const [open, setOpen] = useState<Sample | null>(null);

  const filtered = useMemo(
    () =>
      SAMPLES.filter(
        (s) =>
          (family === "All" || BACKGROUND_FAMILY[s.background] === family) &&
          (state === "All" || s.state === state) &&
          (difficulty === "All" || s.difficulty === difficulty),
      ),
    [family, state, difficulty],
  );

  const s = stats?.data;
  const { backgrounds, pcbShare, difficulties, maxDifficulty } = useMemo(() => {
    if (!s) return { backgrounds: [], pcbShare: 0, difficulties: [], maxDifficulty: 1 };
    const familyOrder = Object.keys(FAMILY_COLOR);
    const bg = Object.keys(s.counts.background)
      .map((name) => ({
        name: prettyLabel(name),
        value: share(s.counts.background, name),
        family: BACKGROUND_FAMILY[name] ?? "Studio",
      }))
      .sort((a, b) => familyOrder.indexOf(a.family) - familyOrder.indexOf(b.family) || b.value - a.value);
    const diff = DIFFICULTY_ORDER.map((name) => ({
      name,
      count: s.counts.difficulty[name] ?? 0,
      value: share(s.counts.difficulty, name),
      color: DIFFICULTY_COLOR[name],
    }));
    return {
      backgrounds: bg,
      pcbShare: bg.filter((b) => b.family === "PCB").reduce((a, b) => a + b.value, 0),
      difficulties: diff,
      maxDifficulty: Math.max(1, ...diff.map((d) => d.value)),
    };
  }, [s]);

  return (
    <Section
      id="dataset"
      eyebrow="03 · Synthetic dataset"
      title={
        <>
          Thousands of LEDs, <span className="text-on">none of them real.</span>
        </>
      }
      subtitle="Procedurally rendered LEDs on PCBs, enclosures and metal, with glare, occlusion, motion blur, sensor noise and look-alike distractors. Every image is reproducible from its seed and described by 40 metadata fields. The same generator scales to a million images."
    >
      {!s ? (
        <p className="mb-10 rounded-2xl border border-amber/30 bg-amber/5 p-4 text-sm text-amber">
          No dataset statistics yet. Run{" "}
          <code className="font-mono">uv run python scripts/compute_dataset_stats.py --dataset &lt;dir&gt;</code>
        </p>
      ) : (
      <>
      <div className="mb-5 grid gap-5 lg:grid-cols-3">
        <div className="glass rounded-3xl p-6 lg:col-span-1">
          <p className="mb-1 text-sm font-semibold text-white">Background mix</p>
          <p className="mb-2 text-xs text-slate-500">
            {backgrounds.length} styles · {pcbShare.toFixed(0)} % PCB
          </p>
          <div className="h-56">
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={backgrounds}
                  dataKey="value"
                  nameKey="name"
                  innerRadius="58%"
                  outerRadius="92%"
                  paddingAngle={2}
                  stroke="none"
                >
                  {backgrounds.map((entry, i) => (
                    <Cell
                      key={entry.name}
                      fill={FAMILY_COLOR[entry.family]}
                      fillOpacity={1 - (i % 4) * 0.18}
                    />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ background: "#0d151e", border: "1px solid #ffffff1a", borderRadius: 12 }}
                  itemStyle={{ color: "#e6edf3" }}
                  formatter={(v) => `${v} %`}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 flex flex-wrap justify-center gap-x-4 gap-y-1">
            {Object.entries(FAMILY_COLOR).map(([name, color]) => (
              <span key={name} className="inline-flex items-center gap-1.5 text-xs text-slate-400">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                {name}
              </span>
            ))}
          </div>
        </div>

        <div className="glass rounded-3xl p-6">
          <p className="mb-5 text-sm font-semibold text-white">Difficulty levels</p>
          <div className="flex flex-col gap-4">
            {difficulties.map((d, i) => (
              <div key={d.name}>
                <div className="mb-1.5 flex justify-between text-xs">
                  <span className="text-slate-300">{prettyLabel(d.name)}</span>
                  <span className="font-mono text-slate-500">
                    {d.count.toLocaleString("en-US")} · {d.value} %
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white/5">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: d.color, boxShadow: `0 0 12px ${d.color}` }}
                    initial={{ width: 0 }}
                    whileInView={{ width: `${(d.value / maxDifficulty) * 100}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 1, delay: i * 0.1 }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass grid grid-cols-2 gap-3 rounded-3xl p-6">
          {[
            ["Scenes", s.scenes.toLocaleString("en-US")],
            ["Images", s.images.toLocaleString("en-US")],
            ["Sizes (px)", Object.keys(s.images_by_size).join(" / ")],
            ["ON / OFF", `${share(s.counts.state, "on")} / ${share(s.counts.state, "off")}`],
            ["Split", ["train", "val", "test"].map((k) => Math.round(share(s.counts.split, k))).join(" / ")],
            ["Distractor scenes", `${share(s.counts.false_positive, "True")} %`],
          ].map(([label, value]) => (
            <div key={label} className="rounded-2xl bg-white/[0.03] p-3">
              <div className="font-mono text-sm text-white">{value}</div>
              <div className="text-[10px] tracking-widest text-slate-500 uppercase">{label}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="mb-10">
        <ProvenanceNote
          origin={stats.origin}
          provenance={s.provenance}
          facts={[s.dataset, `seed ${s.config.seed ?? "?"}`, "counted from metadata.csv"]}
          command={`uv run python scripts/compute_dataset_stats.py --dataset data/${s.dataset}`}
        />
      </div>
      </>
      )}

      <div className="mb-6 flex flex-wrap gap-6">
        <FilterGroup label="Surface" options={FAMILIES} value={family} onChange={setFamily} />
        <FilterGroup label="State" options={STATES} value={state} onChange={setState} />
        <FilterGroup label="Difficulty" options={DIFFICULTIES} value={difficulty} onChange={setDifficulty} />
      </div>

      <motion.div layout className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        <AnimatePresence mode="popLayout">
          {filtered.map((sample) => (
            <motion.button
              layout
              key={sample.file}
              type="button"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              onClick={() => setOpen(sample)}
              className="group glass relative overflow-hidden rounded-2xl text-left"
            >
              <img
                src={sample.url}
                alt={`${sample.background} ${sample.state}`}
                loading="lazy"
                className="aspect-square w-full object-cover transition duration-500 group-hover:scale-110"
              />
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 to-transparent p-2.5 pt-8">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-xs text-slate-200">{prettyLabel(sample.background)}</span>
                  <StateBadge state={sample.state === "on" ? "ON" : "OFF"} />
                </div>
              </div>
            </motion.button>
          ))}
        </AnimatePresence>
      </motion.div>
      {filtered.length === 0 && (
        <p className="py-12 text-center text-slate-500">No samples match these filters.</p>
      )}

      <AnimatePresence>
        {open && (
          <SampleModal
            sample={open}
            onClose={() => setOpen(null)}
            onDetect={() => {
              onDetect(open);
              setOpen(null);
              document.getElementById("detect")?.scrollIntoView({ behavior: "smooth" });
            }}
          />
        )}
      </AnimatePresence>
    </Section>
  );
}

function FilterGroup<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div>
      <p className="mb-2 text-[10px] tracking-widest text-slate-500 uppercase">{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              value === opt ? "bg-white text-ink-950" : "bg-white/5 text-slate-400 hover:text-white"
            }`}
          >
            {opt === "on" || opt === "off" ? opt.toUpperCase() : prettyLabel(opt)}
          </button>
        ))}
      </div>
    </div>
  );
}

function SampleModal({
  sample,
  onClose,
  onDetect,
}: {
  sample: Sample;
  onClose: () => void;
  onDetect: () => void;
}) {
  const rows: [string, string][] = [
    ["Scene id", sample.id],
    ["Background", prettyLabel(sample.background)],
    ["LED colour", prettyLabel(sample.led_color)],
    ["Shape", prettyLabel(sample.led_shape)],
    ["Lens", prettyLabel(sample.led_lens_type)],
    ["Brightness", sample.led_brightness],
    ["Difficulty", prettyLabel(sample.difficulty)],
    ["Ambient", prettyLabel(sample.ambient_type)],
    ["Glare", prettyLabel(sample.glare_type)],
    ["Occlusion", prettyLabel(sample.occlusion_type)],
    ["Resolution", `${sample.size} × ${sample.size}`],
  ];
  return (
    <motion.div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.92, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.92, y: 20 }}
        onClick={(e) => e.stopPropagation()}
        className="glass grid w-full max-w-3xl overflow-hidden rounded-3xl bg-ink-900/90 md:grid-cols-2"
      >
        <img src={sample.url} alt="" className="aspect-square w-full object-cover [image-rendering:pixelated]" />
        <div className="flex flex-col p-6">
          <div className="mb-4 flex items-start justify-between">
            <StateBadge state={sample.state === "on" ? "ON" : "OFF"} large />
            <button type="button" onClick={onClose} aria-label="Close" className="text-slate-500 hover:text-white">
              <X size={20} />
            </button>
          </div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
            {rows.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-slate-500">{k}</dt>
                <dd className="truncate font-mono text-slate-200">{v}</dd>
              </div>
            ))}
          </dl>
          <button
            type="button"
            onClick={onDetect}
            className="mt-6 inline-flex items-center justify-center gap-2 rounded-xl bg-on px-5 py-3 font-semibold text-ink-950 transition hover:brightness-110"
          >
            <ScanLine size={18} /> Run detection on this image
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}
