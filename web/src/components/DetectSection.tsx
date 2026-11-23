import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, ImageUp, Layers, Loader2, Timer } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { DetectionError, compareAll, detect, validateImage } from "../api/detection";
import { METHODS, METHOD_BY_ID } from "../data/methods";
import { SAMPLES, prettyLabel, sampleToBlob } from "../data/samples";
import type { DetectionResponse, MethodId, ResultSource, Sample } from "../types";
import { ConfidenceRing, Section, StateBadge } from "./ui";

type MethodChoice = MethodId | "all";

interface LoadedImage {
  blob: Blob;
  url: string;
  name: string;
  groundTruth?: "on" | "off";
}

const STRIP = SAMPLES.filter((s) => s.background.includes("pcb")).concat(
  SAMPLES.filter((s) => !s.background.includes("pcb")),
).slice(0, 10);

export function DetectSection({
  source,
  requestedSample,
}: {
  source: ResultSource;
  requestedSample: Sample | null;
}) {
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [method, setMethod] = useState<MethodChoice>("yolo");
  const methodRef = useRef<MethodChoice>("yolo");
  const [results, setResults] = useState<DetectionResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const runId = useRef(0);

  useEffect(() => () => {
    if (image) URL.revokeObjectURL(image.url);
  }, [image]);

  const run = useCallback(
    async (img: LoadedImage, choice: MethodChoice) => {
      const id = ++runId.current;
      setLoading(true);
      setError(null);
      setResults([]);
      try {
        const out =
          choice === "all"
            ? await compareAll(img.blob, METHODS.map((m) => m.id), source)
            : [await detect(img.blob, choice, source)];
        if (id === runId.current) setResults(out);
      } catch (err) {
        if (id !== runId.current) return;
        setError(err instanceof DetectionError ? err.message : "Detection failed unexpectedly.");
      } finally {
        if (id === runId.current) setLoading(false);
      }
    },
    [source],
  );

  const loadBlob = useCallback(
    (blob: Blob, name: string, groundTruth?: "on" | "off") => {
      const next = { blob, url: URL.createObjectURL(blob), name, groundTruth };
      setImage(next);
      // Read the ref: the user may change detector while a sample is still loading.
      void run(next, methodRef.current);
    },
    [run],
  );

  const onFile = (file: File | undefined) => {
    if (!file) return;
    try {
      validateImage(file);
    } catch (err) {
      setResults([]);
      setError(err instanceof Error ? err.message : String(err));
      return;
    }
    loadBlob(file, file.name);
  };

  const onSample = useCallback(
    async (sample: Sample) => {
      try {
        loadBlob(await sampleToBlob(sample), sample.file, sample.state);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [loadBlob],
  );

  useEffect(() => {
    if (requestedSample) void onSample(requestedSample);
    // Only react to a new request from the dataset explorer.
  }, [requestedSample]);

  const chooseMethod = (choice: MethodChoice) => {
    setMethod(choice);
    methodRef.current = choice;
    if (image) void run(image, choice);
  };

  const primary = results[0]?.result;

  return (
    <Section
      id="detect"
      eyebrow="01 · Live detection"
      title={
        <>
          Drop an image. <span className="text-on">Get a verdict.</span>
        </>
      }
      subtitle="Upload a photo of an LED, or pick one of the synthetic samples, and choose a detector, or run all five side by side."
    >
      <div className="grid gap-6 lg:grid-cols-[1fr_1.15fr]">
        {/* ------------------------------------------------ input column */}
        <div className="flex flex-col gap-5">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              onFile(e.dataTransfer.files[0]);
            }}
            className={`glass group flex min-h-52 flex-col items-center justify-center gap-3 rounded-3xl border-2 border-dashed p-8 transition ${
              dragging ? "border-on bg-on/5" : "border-white/10 hover:border-on/50"
            }`}
          >
            <div className="rounded-2xl bg-on/10 p-4 text-on transition group-hover:scale-110">
              <ImageUp size={30} />
            </div>
            <p className="font-semibold text-white">Drag & drop an image, or click to browse</p>
            <p className="text-sm text-slate-500">JPG, PNG, BMP or WebP · up to 10 MB</p>
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/bmp,image/webp"
              className="hidden"
              onChange={(e) => {
                onFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </button>

          <div>
            <p className="mb-3 text-sm font-medium text-slate-400">Or try a synthetic sample</p>
            <div className="grid grid-cols-5 gap-2">
              {STRIP.map((sample) => (
                <button
                  key={sample.file}
                  type="button"
                  onClick={() => void onSample(sample)}
                  title={`${prettyLabel(sample.background)} · ${sample.state.toUpperCase()}`}
                  className={`relative aspect-square overflow-hidden rounded-xl border transition hover:scale-105 ${
                    image?.name === sample.file ? "border-on" : "border-white/10"
                  }`}
                >
                  <img src={sample.url} alt="" className="h-full w-full object-cover" />
                </button>
              ))}
            </div>
          </div>

          <div>
            <p className="mb-3 text-sm font-medium text-slate-400">Detector</p>
            <div className="flex flex-wrap gap-2">
              {METHODS.map((m) => (
                <MethodChip
                  key={m.id}
                  active={method === m.id}
                  accent={m.accent}
                  onClick={() => chooseMethod(m.id)}
                >
                  {m.name}
                </MethodChip>
              ))}
              <MethodChip active={method === "all"} accent="#ffffff" onClick={() => chooseMethod("all")}>
                <Layers size={14} /> Compare all
              </MethodChip>
            </div>
          </div>
        </div>

        {/* ------------------------------------------------ output column */}
        <div className="glass relative flex flex-col overflow-hidden rounded-3xl">
          <div className="relative aspect-[4/3] w-full overflow-hidden bg-ink-900">
            {image ? (
              <img src={image.url} alt="Selected input" className="h-full w-full object-contain" />
            ) : (
              <div className="grid-bg flex h-full items-center justify-center">
                <p className="font-mono text-sm text-slate-600">awaiting input_</p>
              </div>
            )}

            {loading && (
              <div className="pointer-events-none absolute inset-0 overflow-hidden">
                <div className="animate-scan h-1/3 w-full bg-gradient-to-b from-transparent via-on/25 to-transparent" />
              </div>
            )}

            <AnimatePresence>
              {primary?.bbox && !loading && image && (
                <motion.div
                  key={`${primary.method}-${primary.bbox.x}`}
                  initial={{ opacity: 0, scale: 1.4 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ type: "spring", stiffness: 220, damping: 18 }}
                  className={`absolute rounded-md border-2 ${
                    primary.state === "ON" ? "border-on glow-on" : "border-off glow-off"
                  }`}
                  style={bboxStyle(primary.bbox)}
                >
                  <span
                    className={`absolute -top-6 left-0 rounded px-1.5 py-0.5 font-mono text-[10px] font-bold ${
                      primary.state === "ON" ? "bg-on text-ink-950" : "bg-off text-white"
                    }`}
                  >
                    LED {primary.state} {Math.round(primary.confidence * 100)}%
                  </span>
                </motion.div>
              )}
            </AnimatePresence>

            {image?.groundTruth && (
              <span className="absolute top-3 right-3 rounded-full bg-black/60 px-3 py-1 font-mono text-[11px] text-slate-300 backdrop-blur">
                ground truth: {image.groundTruth.toUpperCase()}
              </span>
            )}
          </div>

          <div className="flex-1 p-6">
            {error ? (
              <div className="flex items-start gap-3 rounded-2xl border border-off/30 bg-off/10 p-4 text-sm text-off">
                <AlertTriangle size={18} className="mt-0.5 shrink-0" />
                {error}
              </div>
            ) : loading ? (
              <div className="flex items-center gap-3 text-slate-400">
                <Loader2 size={18} className="animate-spin text-on" />
                Analysing image…
              </div>
            ) : results.length === 1 && primary ? (
              <SingleResult response={results[0]} />
            ) : results.length > 1 ? (
              <CompareResults responses={results} />
            ) : (
              <p className="text-slate-500">Results will appear here.</p>
            )}
            {results[0]?.source === "demo" && !loading && (
              <p className="mt-4 text-xs text-amber/80">
                Demo mode: these results are simulated and do not analyse the image. Start the
                FastAPI service to get real detections.
              </p>
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}

function bboxStyle(b: { x: number; y: number; width: number; height: number }) {
  return {
    left: `${b.x * 100}%`,
    top: `${b.y * 100}%`,
    width: `${b.width * 100}%`,
    height: `${b.height * 100}%`,
  };
}

function MethodChip({
  active,
  accent,
  onClick,
  children,
}: {
  active: boolean;
  accent: string;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-xl border px-3.5 py-2 text-sm font-medium transition ${
        active ? "text-ink-950" : "border-white/10 text-slate-300 hover:border-white/25 hover:text-white"
      }`}
      style={active ? { backgroundColor: accent, borderColor: accent } : undefined}
    >
      {children}
    </button>
  );
}

function SingleResult({ response }: { response: DetectionResponse }) {
  const { result } = response;
  const info = METHOD_BY_ID[result.method];
  const on = result.state === "ON";
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-wrap items-center gap-8"
    >
      <ConfidenceRing value={result.confidence} on={on} />
      <div className="flex-1">
        <p className="text-xs tracking-widest text-slate-500 uppercase">LED state</p>
        <p className={`text-6xl font-extrabold tracking-tight ${on ? "text-on" : "text-off"}`}>
          {result.state}
        </p>
        <div className="mt-3 flex flex-wrap gap-4 font-mono text-xs text-slate-400">
          <span className="inline-flex items-center gap-1.5">
            <Timer size={14} /> {result.runtime_ms.toFixed(1)} ms
          </span>
          <span style={{ color: info?.accent }}>{info?.name ?? result.method}</span>
          <span>{info?.family}</span>
        </div>
      </div>
    </motion.div>
  );
}

function CompareResults({ responses }: { responses: DetectionResponse[] }) {
  const onVotes = responses.filter((r) => r.result.state === "ON").length;
  const consensus = onVotes * 2 >= responses.length ? "ON" : "OFF";
  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <p className="text-sm text-slate-400">
          Consensus <StateBadge state={consensus} /> · {Math.max(onVotes, responses.length - onVotes)}/
          {responses.length} agree
        </p>
      </div>
      <div className="flex flex-col gap-2.5">
        {responses.map(({ result }, i) => {
          const info = METHOD_BY_ID[result.method];
          return (
            <motion.div
              key={result.method}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.07 }}
              className="grid grid-cols-[8.5rem_4rem_1fr_4.5rem] items-center gap-3 text-sm"
            >
              <span className="truncate font-medium" style={{ color: info.accent }}>
                {info.name}
              </span>
              <StateBadge state={result.state} />
              <div className="h-2 overflow-hidden rounded-full bg-white/5">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: info.accent }}
                  initial={{ width: 0 }}
                  animate={{ width: `${result.confidence * 100}%` }}
                  transition={{ duration: 0.8, delay: i * 0.07 }}
                />
              </div>
              <span className="text-right font-mono text-xs text-slate-500">
                {result.runtime_ms.toFixed(1)} ms
              </span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
