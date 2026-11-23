import { motion } from "framer-motion";
import { Database } from "lucide-react";
import type { ReactNode } from "react";
import type { DataOrigin, LedState, Provenance, ResultSource } from "../types";

/** Says where the numbers in a panel came from and how to regenerate them. */
export function ProvenanceNote({
  origin,
  provenance,
  facts,
  command,
}: {
  origin: DataOrigin;
  provenance: Provenance;
  facts: string[];
  command: string;
}) {
  const date = new Date(provenance.generated_at).toLocaleString("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  });
  return (
    <details className="glass group rounded-2xl px-4 py-3 text-xs text-slate-400">
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-x-3 gap-y-1">
        <Database size={14} className="text-signal" />
        <span className="font-semibold text-slate-200">Where these numbers come from</span>
        <span
          className={`rounded-full border px-2 py-0.5 font-mono text-[10px] ${
            origin === "api" ? "border-on/40 text-on" : "border-signal/40 text-signal"
          }`}
        >
          {origin === "api" ? "live from API" : "committed snapshot"}
        </span>
        <span className="font-mono">{facts.join(" · ")}</span>
        <span className="ml-auto text-slate-500 group-open:hidden">details ▾</span>
      </summary>
      <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono">
        <dt className="text-slate-500">generated</dt>
        <dd>{date}</dd>
        <dt className="text-slate-500">git commit</dt>
        <dd>
          {provenance.git_commit?.slice(0, 10) ?? "unknown"}
          {provenance.git_dirty && (
            <span className="ml-2 text-amber">+ uncommitted changes (not exactly reproducible)</span>
          )}
        </dd>
        <dt className="text-slate-500">machine</dt>
        <dd>
          {provenance.machine} · Python {provenance.python}
        </dd>
        <dt className="text-slate-500">reproduce</dt>
        <dd className="break-all text-slate-300">{command}</dd>
      </dl>
    </details>
  );
}

export function Section({
  id,
  eyebrow,
  title,
  subtitle,
  children,
}: {
  id: string;
  eyebrow: string;
  title: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} className="relative mx-auto max-w-7xl scroll-mt-24 px-5 py-24 md:px-8">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="mb-12 max-w-3xl"
      >
        <p className="mb-3 font-mono text-xs tracking-[0.3em] text-on uppercase">{eyebrow}</p>
        <h2 className="text-3xl font-bold tracking-tight text-white md:text-5xl">{title}</h2>
        {subtitle && <p className="mt-4 text-lg text-slate-400">{subtitle}</p>}
      </motion.div>
      {children}
    </section>
  );
}

/** A CSS-only LED: glows when on, dark glass when off. */
export function GlowLed({
  on,
  size = 56,
  color = "var(--color-on)",
}: {
  on: boolean;
  size?: number;
  color?: string;
}) {
  return (
    <motion.div
      animate={{
        boxShadow: on
          ? `0 0 ${size * 0.4}px ${color}, 0 0 ${size * 1.4}px color-mix(in oklab, ${color} 45%, transparent)`
          : "0 0 0px transparent",
      }}
      transition={{ duration: 0.35 }}
      className="relative rounded-full"
      style={{
        width: size,
        height: size,
        background: on
          ? `radial-gradient(circle at 40% 35%, #ffffff 0%, ${color} 35%, color-mix(in oklab, ${color} 60%, black) 100%)`
          : "radial-gradient(circle at 40% 35%, #3a4651 0%, #141b22 55%, #070b0f 100%)",
      }}
    >
      <span
        className="absolute top-[14%] left-[22%] rounded-full bg-white/50 blur-[1px]"
        style={{ width: size * 0.22, height: size * 0.14 }}
      />
    </motion.div>
  );
}

export function StateBadge({ state, large = false }: { state: LedState; large?: boolean }) {
  const on = state === "ON";
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border font-mono font-semibold tracking-widest ${
        large ? "px-4 py-1.5 text-sm" : "px-2.5 py-0.5 text-[11px]"
      } ${on ? "border-on/40 bg-on/10 text-on" : "border-off/40 bg-off/10 text-off"}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${on ? "bg-on" : "bg-off"}`} />
      {state}
    </span>
  );
}

export function SourcePill({ source, checking }: { source: ResultSource; checking: boolean }) {
  if (checking) {
    return <span className="font-mono text-[11px] text-slate-500">checking API…</span>;
  }
  const live = source === "api";
  return (
    <span
      title={
        live
          ? "Connected to the FastAPI detection service"
          : "No API found: results are simulated and do not analyse the image"
      }
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-[11px] tracking-wider ${
        live ? "border-on/40 text-on" : "border-amber/40 text-amber"
      }`}
    >
      <span className="relative flex h-2 w-2">
        <span
          className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${
            live ? "bg-on" : "bg-amber"
          }`}
        />
        <span className={`relative inline-flex h-2 w-2 rounded-full ${live ? "bg-on" : "bg-amber"}`} />
      </span>
      {live ? "LIVE API" : "DEMO MODE"}
    </span>
  );
}

export function GitHubMark({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 0 0 7.86 10.92c.58.1.79-.25.79-.56v-2c-3.2.7-3.87-1.37-3.87-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.7 5.39-5.27 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
    </svg>
  );
}

export function ConfidenceRing({ value, on }: { value: number; on: boolean }) {
  const r = 42;
  const c = 2 * Math.PI * r;
  const color = on ? "var(--color-on)" : "var(--color-off)";
  return (
    <div className="relative h-28 w-28">
      <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
        <circle cx="50" cy="50" r={r} fill="none" stroke="rgb(255 255 255 / 0.08)" strokeWidth="8" />
        <motion.circle
          cx="50"
          cy="50"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - value) }}
          transition={{ duration: 0.9, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-2xl font-semibold text-white">{Math.round(value * 100)}%</span>
        <span className="text-[10px] tracking-widest text-slate-500 uppercase">confidence</span>
      </div>
    </div>
  );
}
