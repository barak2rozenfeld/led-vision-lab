import type { ResultSource } from "../types";
import { GitHubMark, GlowLed, SourcePill } from "./ui";

const LINKS = [
  { href: "#detect", label: "Detect" },
  { href: "#webcam", label: "Webcam" },
  { href: "#dataset", label: "Dataset" },
  { href: "#compare", label: "Methods" },
];

export function Nav({ source, checking }: { source: ResultSource; checking: boolean }) {
  return (
    <header className="fixed inset-x-0 top-0 z-50">
      <nav className="glass mx-auto mt-3 flex max-w-7xl items-center justify-between rounded-2xl px-4 py-2.5 md:mx-8 xl:mx-auto">
        <a href="#top" className="flex items-center gap-3">
          <GlowLed on size={18} />
          <span className="font-semibold tracking-tight text-white">
            LED Vision <span className="text-on">Lab</span>
          </span>
        </a>
        <div className="hidden items-center gap-1 md:flex">
          {LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="rounded-lg px-3 py-1.5 text-sm text-slate-400 transition hover:bg-white/5 hover:text-white"
            >
              {link.label}
            </a>
          ))}
        </div>
        <div className="flex items-center gap-3">
          <SourcePill source={source} checking={checking} />
          <a
            href="https://github.com/barak2rozenfeld/led-vision-lab"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            className="rounded-lg p-1.5 text-slate-400 transition hover:bg-white/5 hover:text-white"
          >
            <GitHubMark size={18} />
          </a>
        </div>
      </nav>
    </header>
  );
}
