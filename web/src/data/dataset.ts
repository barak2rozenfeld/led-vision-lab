/** Colours for dataset charts. The numbers themselves come from DatasetStats. */

export const FAMILY_COLOR: Record<string, string> = {
  PCB: "#34f5a0",
  Enclosure: "#38bdf8",
  Plastic: "#a78bfa",
  Metal: "#94a3b8",
  Studio: "#fbbf24",
};

export const DIFFICULTY_COLOR: Record<string, string> = {
  easy: "#34f5a0",
  medium: "#38bdf8",
  hard: "#fbbf24",
  extreme: "#ff4d6d",
};

export const DIFFICULTY_ORDER = ["easy", "medium", "hard", "extreme"] as const;
