import { animate, motion, useInView, useMotionValue, useTransform } from "framer-motion";
import { ArrowRight, Database } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { DATASET_STATS } from "../data/dataset";
import { METHODS } from "../data/methods";
import { GlowLed } from "./ui";

const COLS = 22;
const ROWS = 9;
const PALETTE = ["#34f5a0", "#38bdf8", "#fbbf24", "#ff4d6d", "#a78bfa", "#ffffff"];

function LedMatrix() {
  const cells = useMemo(
    () =>
      Array.from({ length: COLS * ROWS }, (_, i) => {
        const x = i % COLS;
        const y = Math.floor(i / COLS);
        const wave = Math.hypot(x - COLS / 2, (y - ROWS / 2) * 2);
        return { i, delay: wave * 0.12, color: PALETTE[(x * 7 + y * 3) % PALETTE.length] };
      }),
    [],
  );
  return (
    <div
      className="pointer-events-none absolute inset-x-0 top-24 mx-auto grid max-w-6xl gap-4 opacity-40 [mask-image:radial-gradient(ellipse_at_center,black_20%,transparent_70%)]"
      style={{ gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))` }}
    >
      {cells.map((cell) => (
        <motion.span
          key={cell.i}
          className="mx-auto h-2 w-2 rounded-full"
          style={{ backgroundColor: cell.color }}
          animate={{ opacity: [0.08, 1, 0.08], scale: [0.7, 1.15, 0.7] }}
          transition={{ duration: 3.2, repeat: Infinity, delay: cell.delay, ease: "easeInOut" }}
        />
      ))}
    </div>
  );
}

function CountUp({ to, suffix = "" }: { to: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const value = useMotionValue(0);
  const text = useTransform(value, (v) => Math.round(v).toLocaleString("en-US") + suffix);
  useEffect(() => {
    if (!inView) return;
    const controls = animate(value, to, { duration: 1.8, ease: "easeOut" });
    return () => controls.stop();
  }, [inView, to, value]);
  return <motion.span ref={ref}>{text}</motion.span>;
}

function HeroLed() {
  const [on, setOn] = useState(true);
  useEffect(() => {
    const id = window.setInterval(() => setOn((v) => !v), 1800);
    return () => window.clearInterval(id);
  }, []);
  return (
    <div className="flex flex-col items-center gap-5">
      <div className="relative">
        <motion.div
          className="absolute inset-[-60px] rounded-full bg-on/20 blur-3xl"
          animate={{ opacity: on ? 1 : 0 }}
          transition={{ duration: 0.4 }}
        />
        <GlowLed on={on} size={120} />
      </div>
      <div className="flex h-8 items-center gap-3 font-mono text-sm">
        <span className="text-slate-500">state =</span>
        <motion.span
          key={String(on)}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className={on ? "text-on" : "text-off"}
        >
          {on ? '"ON"' : '"OFF"'}
        </motion.span>
      </div>
    </div>
  );
}

const STATS = [
  { label: "synthetic images", to: DATASET_STATS.images },
  { label: "background styles", to: DATASET_STATS.backgrounds },
  { label: "LED colours", to: DATASET_STATS.colors },
  { label: "detection methods", to: METHODS.length },
];

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden pt-36 pb-24">
      <div className="grid-bg absolute inset-0" />
      <div className="absolute top-[-20%] left-1/2 h-[600px] w-[900px] -translate-x-1/2 rounded-full bg-on/10 blur-[140px]" />
      <LedMatrix />

      <div className="relative mx-auto flex max-w-7xl flex-col items-center px-5 text-center md:px-8">
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass mb-8 rounded-full px-4 py-1.5 font-mono text-xs tracking-[0.25em] text-slate-300 uppercase"
        >
          Computer vision · hardware test automation
        </motion.p>

        <HeroLed />

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.7 }}
          className="mt-10 max-w-4xl text-5xl leading-[1.05] font-extrabold tracking-tight md:text-7xl"
        >
          <span className="text-gradient">Is the LED on?</span>
          <br />
          <span className="text-white">Let the camera decide.</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.7 }}
          className="mt-6 max-w-2xl text-lg text-slate-400"
        >
          From a simple HSV threshold to a fine-tuned YOLO detector, LED Vision Lab compares five
          ways to verify LED indicators on real hardware, trained on a million synthetic PCB images.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45, duration: 0.7 }}
          className="mt-10 flex flex-wrap justify-center gap-3"
        >
          <a
            href="#detect"
            className="group inline-flex items-center gap-2 rounded-xl bg-on px-6 py-3 font-semibold text-ink-950 transition hover:brightness-110 glow-on"
          >
            Try detection
            <ArrowRight size={18} className="transition group-hover:translate-x-1" />
          </a>
          <a
            href="#dataset"
            className="glass inline-flex items-center gap-2 rounded-xl px-6 py-3 font-semibold text-white transition hover:bg-white/10"
          >
            <Database size={18} />
            Explore the dataset
          </a>
        </motion.div>

        <div className="mt-20 grid w-full max-w-4xl grid-cols-2 gap-4 md:grid-cols-4">
          {STATS.map((stat) => (
            <div key={stat.label} className="glass rounded-2xl px-5 py-6">
              <div className="font-mono text-2xl font-semibold text-white md:text-3xl">
                <CountUp to={stat.to} />
              </div>
              <div className="mt-1 text-sm text-slate-500">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
