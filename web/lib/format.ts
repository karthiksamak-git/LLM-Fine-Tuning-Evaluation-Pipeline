export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

export function pp(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(digits)} pp`;
}

export function ci(low: number | null | undefined, high: number | null | undefined): string {
  if (low === null || low === undefined || high === null || high === undefined) return "";
  return `[${(low * 100).toFixed(1)}%, ${(high * 100).toFixed(1)}%]`;
}

export function pValue(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (value < 0.0001) return "< 0.0001";
  return value.toFixed(4);
}

export function number(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export const RUN_LABELS: Record<string, string> = {
  base_zero_shot: "Base (zero-shot)",
  base_few_shot: "Base (3-shot)",
  finetuned_adapter: "Fine-tuned (LoRA)",
  finetuned_merged: "Fine-tuned (merged)",
};

export const RUN_ORDER = [
  "base_zero_shot",
  "base_few_shot",
  "finetuned_adapter",
  "finetuned_merged",
];

/** Colorblind-safe palette (Okabe-Ito). */
export const PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"];
