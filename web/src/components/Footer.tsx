import { GitHubMark, GlowLed } from "./ui";

export function Footer() {
  return (
    <footer className="border-t border-white/5">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-5 py-10 text-sm text-slate-500 md:flex-row md:px-8">
        <div className="flex items-center gap-3">
          <GlowLed on size={14} />
          <span>LED Vision Lab · final project</span>
        </div>
        <p className="font-mono text-xs">Python · OpenCV · PyTorch · FastAPI · React</p>
        <a
          href="https://github.com/barak2rozenfeld/led-vision-lab"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 hover:text-white"
        >
          <GitHubMark size={16} /> Source on GitHub
        </a>
      </div>
    </footer>
  );
}
