"use client";

import { ArtifactBoundary, HardwareBadges } from "@/components/ArtifactBoundary";
import { GroupedBars } from "@/components/charts";
import { Badge, Card, PageHeader, Stat } from "@/components/ui";
import { number, pct } from "@/lib/format";
import { useArtifact } from "@/lib/useArtifact";

const LATENCY_LABELS: Record<string, string> = {
  base: "Base model",
  adapter_unmerged: "Adapter (unmerged)",
  merged: "Merged model",
};

export default function EfficiencyPage() {
  const efficiency = useArtifact("efficiency.json");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Efficiency"
        description="Parameter counts, adapter size on disk, inference latency with warmup and repeats, and — only where it actually applies — a mixed-precision comparison. Skipped measurements carry their reason instead of a simulated number."
      >
        <ArtifactBoundary state={efficiency} artifact="efficiency.json">
          {(data) => <HardwareBadges hardware={data.hardware} profile={data.profile} />}
        </ArtifactBoundary>
      </PageHeader>

      <ArtifactBoundary state={efficiency} artifact="efficiency.json">
        {(data) => {
          const sizes = data.sizes ?? {};
          const latency = data.latency ?? {};
          const bench = data.mixed_precision_benchmark;
          const latencyEntries = Object.entries(latency);

          return (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Stat
                  label="Parameters trained"
                  value={pct(
                    sizes.adapter_lora_percent !== undefined
                      ? (sizes.adapter_lora_percent as number) / 100
                      : null,
                    3
                  )}
                  hint={`${number(sizes.adapter_lora as number)} LoRA params of ${number(
                    sizes.adapter_total as number
                  )}`}
                />
                <Stat
                  label="Adapter size"
                  value={`${sizes.adapter_mb ?? "?"} MB`}
                  hint="adapter + tokenizer files on disk"
                />
                <Stat
                  label="Base model"
                  value={String(sizes.base_model ?? data.model ?? "n/a")}
                  hint="fp32 on CPU"
                />
                <Stat
                  label="Precision"
                  value={data.precision?.label ?? "n/a"}
                  hint={data.precision?.reason}
                />
              </div>

              {latencyEntries.length ? (
                <Card
                  title="Inference latency"
                  subtitle="Single fixed prompt, greedy decoding, 2 warmup runs, repeats documented in the artifact."
                >
                  <GroupedBars
                    data={latencyEntries.map(([key, value]) => ({
                      label: LATENCY_LABELS[key] ?? key,
                      "p50 latency (ms)": value.p50_ms ?? 0,
                      "p95 latency (ms)": value.p95_ms ?? 0,
                    }))}
                    series={[
                      { key: "p50 latency (ms)", label: "p50" },
                      { key: "p95 latency (ms)", label: "p95" },
                    ]}
                    percent={false}
                    caption="Median and p95 generation latency per run."
                    ariaLabel="Bar chart of p50 and p95 inference latency for the base, adapter and merged models"
                    height={280}
                  />

                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>
                          <th className="th">Model</th>
                          <th className="th">p50</th>
                          <th className="th">p95</th>
                          <th className="th">mean</th>
                          <th className="th">tokens/s</th>
                        </tr>
                      </thead>
                      <tbody>
                        {latencyEntries.map(([key, value]) => (
                          <tr key={key}>
                            <td className="td">{LATENCY_LABELS[key] ?? key}</td>
                            <td className="td">{value.p50_ms ?? "?"} ms</td>
                            <td className="td">{value.p95_ms ?? "?"} ms</td>
                            <td className="td">{value.mean_ms ?? "?"} ms</td>
                            <td className="td">{value.tokens_per_second ?? "n/a"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              ) : (
                <Card title="Latency">
                  <p className="prose-note">No latency measurements in this artifact.</p>
                </Card>
              )}

              <Card title="Mixed precision: fp32 vs bf16/fp16">
                {bench?.skipped ? (
                  <div className="flex flex-wrap items-center gap-2 text-sm">
                    <Badge tone="warn">skipped</Badge>
                    <span className="prose-note">{bench.reason}</span>
                  </div>
                ) : bench ? (
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>
                          <th className="th">Precision</th>
                          <th className="th">Seconds</th>
                          <th className="th">Steps/s</th>
                          <th className="th">Peak MB</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(
                          [
                            { label: "fp32", row: bench.fp32 as Record<string, unknown> | undefined },
                            {
                              label: String(
                                (bench as Record<string, unknown>).precision ?? "mixed"
                              ),
                              row: bench.mixed as Record<string, unknown> | undefined,
                            },
                          ] as Array<{ label: string; row: Record<string, unknown> | undefined }>
                        ).map((entry) => (
                          <tr key={entry.label}>
                            <td className="td">{entry.label}</td>
                            <td className="td">{String(entry.row?.seconds ?? "?")}</td>
                            <td className="td">{String(entry.row?.steps_per_second ?? "?")}</td>
                            <td className="td">{String(entry.row?.peak_memory_mb ?? "?")}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="prose-note">No benchmark recorded.</p>
                )}
              </Card>

              {data.training && Object.keys(data.training).length ? (
                <Card title="Training cost">
                  <pre className="overflow-x-auto rounded-lg bg-slate-100 p-3 text-xs dark:bg-slate-950">
                    {JSON.stringify(data.training, null, 2)}
                  </pre>
                </Card>
              ) : null}
            </>
          );
        }}
      </ArtifactBoundary>
    </div>
  );
}
