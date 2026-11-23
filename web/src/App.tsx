import { useState } from "react";
import { CompareBoard } from "./components/CompareBoard";
import { DatasetExplorer } from "./components/DatasetExplorer";
import { DetectSection } from "./components/DetectSection";
import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { Nav } from "./components/Nav";
import { WebcamSection } from "./components/WebcamSection";
import { useApiStatus } from "./hooks/useApiStatus";
import type { Sample } from "./types";

export default function App() {
  const { source, checking } = useApiStatus();
  // A fresh object each time so picking the same sample twice still triggers detection.
  const [requested, setRequested] = useState<Sample | null>(null);

  return (
    <div className="min-h-screen overflow-x-hidden bg-ink-950">
      <Nav source={source} checking={checking} />
      <main>
        <Hero />
        <DetectSection source={source} requestedSample={requested} />
        <WebcamSection source={source} />
        <DatasetExplorer onDetect={(sample) => setRequested({ ...sample })} />
        <CompareBoard />
      </main>
      <Footer />
    </div>
  );
}
