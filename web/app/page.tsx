"use client";

import Link from "next/link";

import { ArtifactBoundary, HardwareBadges } from "@/components/ArtifactBoundary";
import { Badge, Card, PageHeader, Stat } from "@/components/ui";
import { RUN_LABELS, pValue, pct } from "@/lib/format";
import { useArtifact } from "@/lib/useArtifact";

export default function OverviewPage() {
  const summary = useArtifact("summary.json");
  const comparison = useArtifact("comparison.json");

  return (
    <div className="space-y-6">
      <PageHeader
        title="LoRA fine-tuning and evaluation for text-to-SQL"
        description="Given CREATE TABLE statements and a natural-language question, generate a SQL query. The base model is compared with its own zero-shot and 3-shot baselines, then fine-tuned with LoRA and evaluated on the same held-out test split with bootstrap confidence intervals and paired McNemar tests."
      >
        <ArtifactBoundary state={summary} artifact="summary.json">
          {(data) => <HardwareBadges hardware={data.hardware} profile={data.profile} />}
        </ArtifactBoundary>
      </PageHeader>

      <ArtifactBoundary state={summary} artifact="summary.json">
        {(data) => (
          <>
            {data.profile_notice ? (
              <div
                role="note"
                className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200"
              >
                <strong className="font-semibold">Hardware honesty.</strong> {data.profile_notice}
              </div>
            ) : null}

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Stat
                label="Test examples"
                value={data.headline.test_n ?? "n/a"}
                hint={typeof data.dataset === "string" ? data.dataset : undefined}
              />
              <Stat
                label="Base zero-shot"
                value={pct(data.headline.execution_match?.base_zero_shot)}
                hint="execution match"
              />
              <Stat
                label="Best baseline"
                value={pct(
                  data.headline.execution_match?.[data.headline.best_baseline ?? "base_few_shot"]
                )}
                hint={RUN_LABELS[data.headline.best_baseline ?? ""] ?? data.headline.best_baseline}
              />
              <Stat
                label="Fine-tuned (LoRA)"
                value={pct(data.headline.execution_match?.finetuned_adapter)}
                hint={
                  data.headline.gain_over_best_baseline ? (
                    <span>
                      {data.headline.gain_over_best_baseline.absolute !== null &&
                      data.headline.gain_over_best_baseline.absolute !== undefined
                        ? `${data.headline.gain_over_best_baseline.absolute > 0 ? "+" : ""}${(
                            data.headline.gain_over_best_baseline.absolute * 100
                          ).toFixed(2)} pp vs best baseline`
                        : "vs best baseline"}{" "}
                      · McNemar p = {pValue(data.headline.gain_over_best_baseline.p_value)}
                    </span>
                  ) : undefined
                }
              />
            </div>

            <div className="grid gap-4 lg:grid-cols-3">
              <Card title="Task and models" className="lg:col-span-2">
                <dl className="grid gap-3 text-sm sm:grid-cols-2">
                  <div>
                    <dt className="text-slate-500 dark:text-slate-400">Task</dt>
                    <dd className="font-medium">text-to-SQL (SQLite dialect)</dd>
                  </div>
                  <div>
                    <dt className="text-slate-500 dark:text-slate-400">Dataset</dt>
                    <dd className="font-mono text-xs">
                      {typeof data.dataset === "string" ? data.dataset : "b-mc2/sql-create-context"}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500 dark:text-slate-400">Base model</dt>
                    <dd className="font-mono text-xs">{data.model}</dd>
                  </div>
                  <div>
                    <dt className="text-slate-500 dark:text-slate-400">Method</dt>
                    <dd>LoRA (PEFT) · completion-only loss masking</dd>
                  </div>
                  <div>
                    <dt className="text-slate-500 dark:text-slate-400">Train / val / test</dt>
                    <dd>
                      {data.splits?.train} / {data.splits?.val} / {data.splits?.test}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500 dark:text-slate-400">Training</dt>
                    <dd>
                      {String(data.training?.steps ?? "n/a")} steps ·{" "}
                      {String(data.training?.wall_clock_s ?? "n/a")} s ·{" "}
                      {data.training?.trainable_percent
                        ? `${Number(data.training.trainable_percent).toFixed(2)}% params trained`
                        : "n/a"}
                    </dd>
                  </div>
                </dl>
              </Card>

              <Card title="What the pipeline does">
                <ol className="list-decimal space-y-1.5 pl-4 text-sm">
                  {(data.pipeline_steps ?? []).map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </Card>
            </div>

            <ArtifactBoundary state={comparison} artifact="comparison.json">
              {(comp) => (
                <Card
                  title="Headline comparison"
                  subtitle="Execution match on the test split. Intervals are 95% bootstrap CIs; p-values are paired McNemar tests against the fine-tuned adapter."
                >
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>
                          <th className="th">Run</th>
                          <th className="th">Execution match</th>
                          <th className="th">Exact match</th>
                          <th className="th">Valid SQL</th>
                          <th className="th">vs fine-tuned</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(comp.runs).map(([label, run]) => {
                          const key = `finetuned_adapter__vs__${label}`;
                          const cmp = comp.comparisons?.[key];
                          return (
                            <tr key={label}>
                              <td className="td font-medium">{RUN_LABELS[label] ?? label}</td>
                              <td className="td">
                                {pct(run.metrics.execution_match.value)}{" "}
                                <span className="text-xs text-slate-500">
                                  [{pct(run.metrics.execution_match.ci_low)},{" "}
                                  {pct(run.metrics.execution_match.ci_high)}]
                                </span>
                              </td>
                              <td className="td">{pct(run.metrics.exact_match.value)}</td>
                              <td className="td">{pct(run.metrics.is_valid.value)}</td>
                              <td className="td text-xs">
                                {cmp?.execution_match ? (
                                  <>
                                    p = {pValue(cmp.execution_match.mcnemar?.p_value)}
                                    {cmp.execution_match.mcnemar?.p_value !== undefined &&
                                    cmp.execution_match.mcnemar?.p_value !== null &&
                                    cmp.execution_match.mcnemar.p_value < 0.05 ? (
                                      <Badge tone="good">significant</Badge>
                                    ) : (
                                      <Badge tone="neutral">n.s.</Badge>
                                    )}
                                  </>
                                ) : (
                                  <span className="text-slate-400">—</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <p className="prose-note mt-3 text-xs">
                    Execution match runs gold and predicted SQL on synthetic seeded SQLite tables,
                    so equivalences are approximate. See{" "}
                    <Link className="underline" href="/results">
                      Results
                    </Link>{" "}
                    for the full breakdown and{" "}
                    <Link className="underline" href="/data">
                      Data
                    </Link>{" "}
                    for the decontamination report.
                  </p>
                </Card>
              )}
            </ArtifactBoundary>

            {data.limitations ? (
              <Card title="Limitations (read before quoting any number)">
                <ul className="list-disc space-y-1.5 pl-4 text-sm">
                  {data.limitations.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </Card>
            ) : null}
          </>
        )}
      </ArtifactBoundary>
    </div>
  );
}
