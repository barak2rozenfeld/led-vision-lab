const COLAB_BASE =
  "https://colab.research.google.com/github/barak2rozenfeld/led-vision-lab/blob/master/notebooks";

export const NOTEBOOKS = {
  dataset: { file: "01_dataset.ipynb", title: "Dataset notebook" },
  hsv: { file: "02_hsv_baseline.ipynb", title: "HSV baseline notebook" },
} as const;

export type NotebookId = keyof typeof NOTEBOOKS;

export function colabUrl(id: NotebookId): string {
  return `${COLAB_BASE}/${NOTEBOOKS[id].file}`;
}
