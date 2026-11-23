import { motion } from "framer-motion";
import { AlertTriangle, Camera, CameraOff } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { DetectionError, detect } from "../api/detection";
import { METHODS, METHOD_BY_ID } from "../data/methods";
import type { DetectionResult, MethodId, MethodStatus, ResultSource } from "../types";
import { GlowLed, Section } from "./ui";

const FRAME_GAP_MS = 400;

function cameraErrorMessage(err: unknown): string {
  if (!window.isSecureContext) return "The camera needs HTTPS or localhost.";
  if (err instanceof DOMException) {
    if (err.name === "NotAllowedError") return "Camera permission was denied.";
    if (err.name === "NotFoundError") return "No camera was found on this device.";
    if (err.name === "NotReadableError") return "The camera is already in use by another app.";
  }
  return "Could not start the camera.";
}

export function WebcamSection({
  source,
  methodStatus,
}: {
  source: ResultSource;
  methodStatus: Partial<Record<MethodId, MethodStatus>> | null;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const liveRef = useRef(false);
  const loopId = useRef(0);
  const [live, setLive] = useState(false);
  const [method, setMethod] = useState<MethodId>("hsv");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [frames, setFrames] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    liveRef.current = false;
    setLive(false);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => stop, [stop]);

  const captureFrame = (): Promise<Blob | null> => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0) return Promise.resolve(null);
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
  };

  const loop = useCallback(
    async (activeMethod: MethodId) => {
      const id = ++loopId.current;
      const current = () => liveRef.current && id === loopId.current;
      while (current()) {
        const frame = await captureFrame();
        if (frame) {
          try {
            const { result: r } = await detect(frame, activeMethod, source);
            if (!current()) break;
            setResult(r);
            setFrames((n) => n + 1);
          } catch (err) {
            setError(err instanceof DetectionError ? err.message : "Detection failed.");
            stop();
            break;
          }
        }
        await new Promise((resolve) => setTimeout(resolve, FRAME_GAP_MS));
      }
    },
    [source, stop],
  );

  const start = async () => {
    setError(null);
    setResult(null);
    setFrames(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      liveRef.current = true;
      setLive(true);
      void loop(method);
    } catch (err) {
      setError(cameraErrorMessage(err));
      stop();
    }
  };

  const changeMethod = (m: MethodId) => {
    setMethod(m);
    // Starting a new loop bumps loopId, which makes the old loop exit.
    if (liveRef.current) void loop(m);
  };

  const on = result?.state === "ON";

  return (
    <Section
      id="webcam"
      eyebrow="02 · Webcam live mode"
      title={
        <>
          Point a camera at a board. <span className="text-signal">Watch it think.</span>
        </>
      }
      subtitle="Frames are captured in the browser and sent to the same detection service as uploads, so this is how a test station would see the device under test."
    >
      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        <div className="glass relative aspect-video overflow-hidden rounded-3xl bg-ink-900">
          <video ref={videoRef} muted playsInline className={`h-full w-full object-cover ${live ? "" : "hidden"}`} />
          <canvas ref={canvasRef} className="hidden" />
          {!live && (
            <div className="grid-bg flex h-full flex-col items-center justify-center gap-4">
              <CameraOff size={36} className="text-slate-600" />
              <p className="font-mono text-sm text-slate-500">camera offline</p>
            </div>
          )}
          {live && (
            <>
              <div className="pointer-events-none absolute inset-6 rounded-2xl border border-white/10">
                {["top-0 left-0 border-t-2 border-l-2", "top-0 right-0 border-t-2 border-r-2", "bottom-0 left-0 border-b-2 border-l-2", "right-0 bottom-0 border-r-2 border-b-2"].map(
                  (pos) => (
                    <span key={pos} className={`absolute h-6 w-6 border-signal ${pos}`} />
                  ),
                )}
              </div>
              <div className="absolute top-4 left-4 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 font-mono text-[11px] text-off backdrop-blur">
                <span className="h-2 w-2 animate-pulse rounded-full bg-off" /> REC
              </div>
              {result && (
                <motion.div
                  key={result.state}
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className={`absolute right-4 bottom-4 rounded-2xl px-5 py-3 font-mono backdrop-blur ${
                    on ? "bg-on/20 text-on glow-on" : "bg-off/20 text-off glow-off"
                  }`}
                >
                  <div className="text-3xl font-extrabold">{result.state}</div>
                  <div className="text-xs opacity-80">{Math.round(result.confidence * 100)}% confidence</div>
                </motion.div>
              )}
            </>
          )}
        </div>

        <div className="glass flex flex-col gap-6 rounded-3xl p-6">
          <div className="flex items-center gap-5">
            <GlowLed on={live && on} size={64} color={live && result && !on ? "var(--color-off)" : undefined} />
            <div>
              <p className="text-xs tracking-widest text-slate-500 uppercase">Live state</p>
              <p className={`text-4xl font-extrabold ${!result ? "text-slate-600" : on ? "text-on" : "text-off"}`}>
                {result?.state ?? "—"}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 font-mono">
            <Stat label="latency" value={result ? `${result.runtime_ms.toFixed(1)}ms` : "—"} />
            <Stat label="frames" value={String(frames)} />
            <Stat label="method" value={METHOD_BY_ID[method].name.split(" ")[0]} />
          </div>

          <div>
            <p className="mb-2 text-sm text-slate-400">Detector</p>
            <select
              value={method}
              onChange={(e) => changeMethod(e.target.value as MethodId)}
              className="w-full rounded-xl border border-white/10 bg-ink-800 px-3 py-2.5 text-sm text-white outline-none focus:border-signal"
            >
              {METHODS.map((m) => {
                const status = methodStatus?.[m.id] ?? "ready";
                return (
                  <option key={m.id} value={m.id} disabled={status !== "ready"}>
                    {m.name} · {status === "ready" ? m.family : status.replace("_", " ")}
                  </option>
                );
              })}
            </select>
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-xl border border-off/30 bg-off/10 p-3 text-sm text-off">
              <AlertTriangle size={16} className="mt-0.5 shrink-0" /> {error}
            </div>
          )}

          <button
            type="button"
            onClick={live ? stop : () => void start()}
            className={`mt-auto inline-flex items-center justify-center gap-2 rounded-xl px-5 py-3 font-semibold transition ${
              live ? "bg-off/90 text-white hover:bg-off" : "bg-signal text-ink-950 hover:brightness-110"
            }`}
          >
            {live ? <CameraOff size={18} /> : <Camera size={18} />}
            {live ? "Stop camera" : "Start camera"}
          </button>
          {source === "demo" && (
            <p className="text-xs text-amber/80">Demo mode: live results are simulated.</p>
          )}
        </div>
      </div>
    </Section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-white/[0.03] px-3 py-2.5">
      <div className="truncate text-sm text-white">{value}</div>
      <div className="text-[10px] tracking-widest text-slate-500 uppercase">{label}</div>
    </div>
  );
}
