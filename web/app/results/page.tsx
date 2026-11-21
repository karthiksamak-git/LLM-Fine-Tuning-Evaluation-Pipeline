"use client";

import { ArtifactBoundary, HardwareBadges } from "@/components/ArtifactBoundary";
import { GroupedBars } from "@/components/charts";
import { Badge, Card, PageHeader, Stat } from "@/components/ui";
import { RUN_LABELS, RUN_ORDER, pValue, pct } from "@/lib/format";
import { useArtifact } from "@/lib/useArtifact";

const METRICS = [
  { key: "execution_match", label: "Execution match" },
  { key: "exact_match", label: "Exact match" },
  { key: "is_valid", label: "Valid SQL" },
] as const;

export default function ResultsPage() {
  const comparison = useArtifact("comparison.json");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Before / after results"
        description="All four runs are scored on the same held-out test split with identical greedy decoding settings. Intervals are 95% percentile bootstrap CIs over the test set."
      >
        <ArtifactBoundary state={comparison} artifact="comparison.json">
          {(data) => <HardwareBadges hardware={data.hardware} profile={data.profile} />}
        </ArtifactBoundary>
      </PageHeader>

      <ArtifactBoundary state={comparison} artifact="comparison.json">
        {(data) => {
          const labels = RUN_ORDER.filter((label) => data.runs[label]);
          const chartData = labels.map((label) => {
            const row: Record<string, string | number> = {
              label: RUN_LABELS[label] ?? label,
            };
            for (const metric of METRICS) {
              row[metric.label] = data.runs[label].metrics[metric.key].value;
            }
            return row;
          });

          return (
            <>
              <Card
                title="Metric comparison"
                subtitle={`n = ${data.test_set?.n ?? "n/a"} test examples · higher is better`}
              >
                <GroupedBars
                  data={chartData}
                  series={METRICS.map((metric) => ({ key: metric.label, label: metric.label }))}
                  caption="Execution match, exact match and SQL validity per run on the test split."
                  ariaLabel="Bar chart comparing execution match, exact match and valid SQL across the base zero-shot, base three-shot, fine-tuned LoRA and merged runs"
                  height={340}
                />
              </Card>

              <div className="grid gap-4 sm:grid-cols-3">
                {METRICS.map((metric) => {
                  const best = labels.reduce((acc, label) =>
                    data.runs[label].metrics[metric.key].value > acc.value
                      ? { label, value: data.runs[label].metrics[metric.key].value }
                      : acc,
                    { label: labels[0], value: data.runs[labels[0]].metrics[metric.key].value }
                  );
                  return (
                    <Stat
                      key={metric.key}
                      label={`Best ${metric.label.toLowerCase()}`}
                      value={pct(best.value)}
                      hint={RUN_LABELS[best.label] ?? best.label}
                    />
                  );
                })}
              </div>

              <Card title="Metrics with 95% confidence intervals">
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr>
                        <th className="th">Run</th>
                        {METRICS.map((metric) => (
                          <th key={metric.key} className="th">
                            {metric.label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {labels.map((label) => (
                        <tr key={label}>
                          <td className="td font-medium">{RUN_LABELS[label] ?? label}</td>
                          {METRICS.map((metric) => {
                            const value = data.runs[label].metrics[metric.key];
                            return (
                              <td key={metric.key} className="td">
                                {pct(value.value)}{" "}
                                <span className="text-xs text-slate-500">
                                  [{pct(value.ci_low)}, {pct(value.ci_high)}]
                                </span>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              <Card
                title="Paired significance (McNemar)"
                subtitle="Same test items, same decoding. b = items the fine-tune fixed, c = items it broke."
              >
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr>
                        <th className="th">Comparison</th>
                        <th className="th">Δ execution match</th>
                        <th className="th">Δ relative</th>
                        <th className="th">b / c</th>
                        <th className="th">p-value</th>
                        <th className="th">Verdict</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(data.comparisons ?? {}).map(([name, entry]) => {
                        const metric = entry.execution_match;
                        const p = metric?.mcnemar?.p_value;
                        return (
                          <tr key={name}>
                            <td className="td font-mono text-xs">{name.replace(/__/g, " ")}</td>
                            <td className="td">
                              {metric?.absolute_difference !== null &&
                              metric?.absolute_difference !== undefined
                                ? `${metric.absolute_difference > 0 ? "+" : ""}${(
                                    metric.absolute_difference * 100
                                  ).toFixed(2)} pp`
                                : "n/a"}
                            </td>
                            <td className="td">
                              {metric?.relative_improvement !== null &&
                              metric?.relative_improvement !== undefined
                                ? `${(metric.relative_improvement * 100).toFixed(1)}%`
                                : "n/a"}
                            </td>
                            <td className="td">
                              {metric?.mcnemar?.b ?? "?"} / {metric?.mcnemar?.c ?? "?"}
                            </td>
                            <td className="td">{pValue(p)}</td>
                            <td className="td">
                              {p !== undefined && p !== null && p < 0.05 ? (
                                <Badge tone="good">significant</Badge>
                              ) : (
                                <Badge tone="neutral">not significant</Badge>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="prose-note mt-3 text-xs">
                  A non-significant result is reported as-is: on a small test split and a small
                  model, a fine-tune can easily fail to beat a strong few-shot baseline.
                </p>
              </Card>

              <Card
                title="Breakdown by gold-query complexity"
                subtitle="simple: no joins/aggregations/subqueries · medium: exactly one · complex: two or more"
              >
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr>
                        <th className="th">Bucket</th>
                        <th className="th">Run</th>
                        <th className="th">n</th>
                        <th className="th">Execution match</th>
                        <th className="th">Exact match</th>
                      </tr>
                    </thead>
                    <tbody>
                      {labels.flatMap((label) => {
                        const buckets = data.runs[label].by_bucket ?? {};
                        return Object.entries(buckets).map(([bucket, payload]) => (
                          <tr key={`${label}-${bucket}`}>
                            <td className="td capitalize">{bucket}</td>
                            <td className="td">{RUN_LABELS[label] ?? label}</td>
                            <td className="td">{payload.n}</td>
                            <td className="td">
                              {pct(payload.execution_match?.value)}{" "}
                              <span className="text-xs text-slate-500">
                                [{pct(payload.execution_match?.ci_low)},{" "}
                                {pct(payload.execution_match?.ci_high)}]
                              </span>
                            </td>
                            <td className="td">{pct(payload.exact_match?.value)}</td>
                          </tr>
                        ));
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>

              {data.seeds?.note ? (
                <Card title="Seed variance">
                  <p className="prose-note">{data.seeds.note}</p>
                </Card>
              ) : null}
            </>
          );
        }}
      </ArtifactBoundary>
    </div>
  );
}
