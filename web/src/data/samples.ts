import manifest from "@samples/samples.json";
import type { Sample, SampleEntry } from "../types";

const urls = import.meta.glob<string>("@samples/*.jpg", {
  eager: true,
  query: "?url",
  import: "default",
});

const urlByFile = new Map(
  Object.entries(urls).map(([key, url]) => [key.split("/").pop() ?? key, url]),
);

export const SAMPLES: Sample[] = (manifest as SampleEntry[])
  .filter((entry) => urlByFile.has(entry.file))
  .map((entry) => ({ ...entry, url: urlByFile.get(entry.file)! }));

export const BACKGROUND_FAMILY: Record<string, string> = {
  pcb: "PCB",
  blue_pcb: "PCB",
  realistic_pcb: "PCB",
  aluminum_pcb: "PCB",
  black_plastic: "Plastic",
  gray_plastic: "Plastic",
  white_plastic: "Plastic",
  dark_enclosure: "Enclosure",
  light_enclosure: "Enclosure",
  metal: "Metal",
  brushed_metal: "Metal",
  white_studio: "Studio",
  textured: "Studio",
  gradient: "Studio",
  plain: "Studio",
};

export function prettyLabel(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export async function sampleToBlob(sample: Sample): Promise<Blob> {
  const res = await fetch(sample.url);
  if (!res.ok) throw new Error(`Could not load sample ${sample.file}`);
  return res.blob();
}
