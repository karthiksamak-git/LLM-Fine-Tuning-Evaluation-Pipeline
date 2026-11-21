"use client";

import type { ReactNode } from "react";

import type { ArtifactState } from "@/lib/useArtifact";

import { ErrorState, Loading } from "./ui";

export function ArtifactBoundary<T>({
  state,
  artifact,
  children,
}: {
  state: ArtifactState<T>;
  artifact: string;
  children: (data: T) => ReactNode;
}) {
  if (state.status === "loading") {
    return <Loading label={`Loading ${artifact}…`} />;
  }
  if (state.status === "error") {
    return <ErrorState message={state.error} artifact={artifact} />;
  }
  return <>{children(state.data)}</>;
}

export function HardwareBadges({
  hardware,
  profile,
}: {
  hardware: {
    device?: string | null;
    gpu_name?: string | null;
    vram_gb?: number | null;
    ram_total_gb?: number | null;
    cpu_count_logical?: number | null;
    precision_label?: string | null;
  };
  profile: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="inline-flex items-center rounded-full bg-brand-600 px-2.5 py-0.5 font-semibold text-white">
        profile: {profile}
      </span>
      <span className="inline-flex items-center rounded-full border border-slate-300 px-2.5 py-0.5 dark:border-slate-700">
        device: {hardware.device ?? "unknown"}
      </span>
      {hardware.gpu_name ? (
        <span className="inline-flex items-center rounded-full border border-slate-300 px-2.5 py-0.5 dark:border-slate-700">
          gpu: {hardware.gpu_name} ({hardware.vram_gb} GB)
        </span>
      ) : (
        <span className="inline-flex items-center rounded-full border border-slate-300 px-2.5 py-0.5 dark:border-slate-700">
          no GPU
        </span>
      )}
      {hardware.cpu_count_logical ? (
        <span className="inline-flex items-center rounded-full border border-slate-300 px-2.5 py-0.5 dark:border-slate-700">
          {hardware.cpu_count_logical} vCPU · {hardware.ram_total_gb} GB RAM
        </span>
      ) : null}
      {hardware.precision_label ? (
        <span className="inline-flex items-center rounded-full border border-slate-300 px-2.5 py-0.5 dark:border-slate-700">
          precision: {hardware.precision_label}
        </span>
      ) : null}
    </div>
  );
}
