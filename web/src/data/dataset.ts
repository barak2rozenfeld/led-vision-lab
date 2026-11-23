/** Composition of configs/dataset/pcb_1m.yaml (keep in sync with that file). */

export const DATASET_STATS = {
  images: 1_000_000,
  scenes: 250_000,
  sizes: [64, 128, 224, 256],
  backgrounds: 15,
  colors: 14,
  shapes: 6,
};

export const BACKGROUND_WEIGHTS: { name: string; value: number; family: string }[] = [
  { name: "PCB (green)", value: 22, family: "PCB" },
  { name: "Realistic PCB", value: 22, family: "PCB" },
  { name: "Blue PCB", value: 12, family: "PCB" },
  { name: "Aluminum PCB", value: 10, family: "PCB" },
  { name: "Enclosures", value: 12, family: "Enclosure" },
  { name: "Plastic", value: 9, family: "Plastic" },
  { name: "Metal", value: 7, family: "Metal" },
  { name: "Studio / plain", value: 6, family: "Studio" },
];

export const DIFFICULTY_WEIGHTS = [
  { name: "Easy", value: 25, color: "#34f5a0" },
  { name: "Medium", value: 35, color: "#38bdf8" },
  { name: "Hard", value: 25, color: "#fbbf24" },
  { name: "Extreme", value: 15, color: "#ff4d6d" },
];

export const FAMILY_COLOR: Record<string, string> = {
  PCB: "#34f5a0",
  Enclosure: "#38bdf8",
  Plastic: "#a78bfa",
  Metal: "#94a3b8",
  Studio: "#fbbf24",
};
