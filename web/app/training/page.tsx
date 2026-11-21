"use client";

import { ArtifactBoundary, HardwareBadges } from "@/components/ArtifactBoundary";
import { Lines } from "@/components/charts";
import { Card, PageHeader, Stat } from "@/components/ui";
import { number, pct } from "@/lib/format";
import { useArtifact } from "@/lib/useArtifact";

export default function TrainingPage() {
  const training = useArtifact("training.json");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Training"
        description="Loss and learning-rate trajectories from the real run, plus the exact hyperparameters, LoRA configuration, throughput and precision policy. Nothing here is simulated."
      >
        <ArtifactBoundary state={training} artifact="training.json">
          {(data) => <HardwareBadges hardware={data.hardware} profile={data.profile} />}
        </ArtifactBoundary>
      </PageHeader>

      <ArtifactBoundary state={training} artifact="training.json">
        {(data) => {
          const trainCurve = data.curves?.train ?? [];
          const evalCurve = data.curves?.eval ?? [];
          const byStep = new Map<number, Record<string, number | null>>();
          for (const point of trainCurve) {
            if (point.step === null || point.step === undefined) continue;
            byStep.set(point.step, {
              step: point.step,
              "train loss": point.loss ?? null,
              "learning rate": point.learning_rate ?? null,
            });
          }
          for (const point of evalCurve) {
            if (point.step === null || point.step === undefined) continue;
            const existing = byStep.get(point.step) ?? { step: point.step };
            existing["validation loss"] = point.eval_loss ?? null;
            byStep.set(point.step, existing);
          }
          const chart = [...byStep.values()].sort((a, b) => Number(a.step) - Number(b.step));

          return (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Stat label="Optimizer steps" value={number(data.throughput?.steps as number)} />
                <Stat
                  label="Wall clock"
                  value={`${number(data.throughput?.wall_clock_s as number, 1)} s`}
                  hint={`${data.throughput?.epochs_completed ?? "?"} epochs`}
                />
                <Stat
                  label="Trainable parameters"
                  value={pct(
                    data.lora?.trainable_percent !== undefined
                      ? (data.lora.trainable_percent as number) / 100
                      : null,
                    3
                  )}
                  hint={`${number(data.lora?.trainable_parameters as number)} of ${number(
                    data.lora?.total_parameters as number
                  )}`}
                />
                <Stat
                  label="Precision"
                  value={data.precision?.label ?? "n/a"}
                  hint={`device: ${data.device ?? "unknown"}`}
                />
              </div>

              <Card
                title="Loss curves"
                subtitle="Train loss is logged every N steps; validation loss every eval interval. Best checkpoint is selected by validation loss."
              >
                {chart.length ? (
                  <Lines
                    data={chart}
                    xKey="step"
                    series={[
                      { key: "train loss", label: "train loss" },
                      { key: "validation loss", label: "validation loss" },
                    ]}
                    caption="Training and validation loss against optimizer step."
                    ariaLabel="Line chart of training loss and validation loss over optimizer steps"
                    height={320}
                  />
                ) : (
                  <p className="prose-note">No curve data in this artifact.</p>
                )}
              </Card>

              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="LoRA configuration">
                  <dl className="grid gap-2 text-sm">
                    <Row label="rank (r)" value={data.lora?.lora_r} />
                    <Row label="alpha" value={data.lora?.lora_alpha} />
                    <Row label="dropout" value={data.lora?.lora_dropout} />
                    <Row
                      label="target modules"
                      value={(data.lora?.target_modules ?? []).join(", ") || "n/a"}
                      mono
                    />
                    <Row label="task type" value={data.lora?.task_type ?? "CAUSAL_LM"} mono />
                  </dl>
                </Card>

                <Card title="Hyperparameters">
                  <dl className="grid gap-2 text-sm">
                    {Object.entries(data.hyperparameters ?? {}).map(([key, value]) => (
                      <Row
                        key={key}
                        label={
                          key === "num_train_epochs" &&
                          Number(data.hyperparameters?.max_steps ?? 0) > 0
                            ? "num train epochs (config; max steps overrides it)"
                            : key.replace(/_/g, " ")
                        }
                        value={String(value)}
                      />
                    ))}
                  </dl>
                </Card>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Throughput and memory">
                  <dl className="grid gap-2 text-sm">
                    <Row
                      label="train tokens/s (estimated)"
                      value={number(
                        (data.throughput?.train_tokens_per_second as number) ?? null,
                        1
                      )}
                    />
                    <Row label="effective batch size" value={data.throughput?.effective_batch_size} />
                    <Row label="peak RSS" value={`${String(data.memory?.peak_rss_mb ?? "?")} MB`} />
                    <Row
                      label="CUDA peak allocated"
                      value={
                        data.memory?.cuda_peak_allocated_mb
                          ? `${data.memory.cuda_peak_allocated_mb} MB`
                          : "n/a (no CUDA)"
                      }
                    />
                  </dl>
                </Card>

                <Card title="Mixed precision policy">
                  <p className="text-sm font-medium">{data.precision?.label}</p>
                  <p className="prose-note mt-2">{data.precision?.reason}</p>
                  <p className="prose-note mt-3 text-xs">
                    CUDA with bf16 → bf16 autocast. CUDA without bf16 (e.g. T4) → fp16 autocast with
                    GradScaler and LoRA params cast to fp32. CPU/MPS → fp32, recorded as
                    &quot;mixed precision: not used&quot;.
                  </p>
                </Card>
              </div>

              <Card title="Tokenization at train time">
                <pre className="overflow-x-auto rounded-lg bg-slate-100 p-3 text-xs dark:bg-slate-950">
                  {JSON.stringify(
                    { train: data.train_dataset, val: data.val_dataset },
                    null,
                    2
                  )}
                </pre>
              </Card>
            </>
          );
        }}
      </ArtifactBoundary>
    </div>
  );
}

function Row({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: unknown;
  mono?: boolean;
}) {
  const display =
    value === null || value === undefined
      ? "n/a"
      : typeof value === "string" || typeof value === "number" || typeof value === "boolean"
        ? String(value)
        : JSON.stringify(value);
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-100 pb-1.5 dark:border-slate-800/60">
      <dt className="text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className={`text-right ${mono ? "font-mono text-xs" : ""}`}>{display}</dd>
    </div>
  );
}
