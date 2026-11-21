"use client";

import { useState } from "react";

import { ArtifactBoundary, HardwareBadges } from "@/components/ArtifactBoundary";
import { GroupedBars } from "@/components/charts";
import { Badge, Card, PageHeader } from "@/components/ui";
import { RUN_LABELS, RUN_ORDER } from "@/lib/format";
import { useArtifact } from "@/lib/useArtifact";

const CATEGORIES = ["all", "fixed", "regressed", "still_wrong"] as const;
type Category = (typeof CATEGORIES)[number];

const CATEGORY_LABELS: Record<Category, string> = {
  all: "All",
  fixed: "Fixed by fine-tuning",
  regressed: "Regressed",
  still_wrong: "Still wrong",
};

export default function ErrorsPage() {
  const analysis = useArtifact("error_analysis.json");
  const [category, setCategory] = useState<Category>("all");
  const [query, setQuery] = useState("");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Error analysis"
        description="A heuristic failure taxonomy derived from SQL structure, plus a side-by-side explorer of real test items: schema, question, gold query, and what each model produced."
      >
        <ArtifactBoundary state={analysis} artifact="error_analysis.json">
          {(data) => <HardwareBadges hardware={data.hardware} profile={data.profile} />}
        </ArtifactBoundary>
      </PageHeader>

      <ArtifactBoundary state={analysis} artifact="error_analysis.json">
        {(data) => {
          const taxonomy = data.taxonomy ?? {};
          const examples = data.examples ?? [];
          const labels = RUN_ORDER.filter((label) => taxonomy[label]);
          const categories = [
            ...new Set(labels.flatMap((label) => Object.keys(taxonomy[label] ?? {}))),
          ];
          const chartData = categories.map((cat) => {
            const row: Record<string, string | number> = { label: cat };
            for (const label of labels) {
              row[label] = taxonomy[label]?.[cat] ?? 0;
            }
            return row;
          });

          const filtered = examples.filter((example) => {
            if (category !== "all" && example.category !== category) return false;
            const haystack = `${example.question} ${example.gold} ${example.schema}`.toLowerCase();
            return haystack.includes(query.trim().toLowerCase());
          });

          return (
            <>
              <div className="grid gap-4 sm:grid-cols-4">
                {Object.entries(data.outcomes ?? {}).map(([key, value]) => (
                  <div key={key} className="card">
                    <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      {key.replace(/_/g, " ")}
                    </p>
                    <p className="stat-value mt-1">{value}</p>
                  </div>
                ))}
              </div>

              <Card
                title="Failure taxonomy"
                subtitle="Counts on the test split. 'correct' means execution match; the rest are heuristic failure labels."
              >
                <GroupedBars
                  data={chartData}
                  series={labels.map((label) => ({
                    key: label,
                    label: RUN_LABELS[label] ?? label,
                  }))}
                  percent={false}
                  caption="Failure categories per run on the test split."
                  ariaLabel="Bar chart of failure taxonomy counts for the base and fine-tuned models"
                  height={340}
                />
                {data.method ? <p className="prose-note mt-3 text-xs">{data.method}</p> : null}
              </Card>

              <Card
                title="Example explorer"
                subtitle={`${examples.length} curated examples. Filter by outcome and search the question, schema or gold SQL.`}
              >
                <div className="mb-4 flex flex-wrap items-center gap-2">
                  {CATEGORIES.map((option) => (
                    <button
                      key={option}
                      type="button"
                      onClick={() => setCategory(option)}
                      aria-pressed={category === option}
                      className={`rounded-full border px-3 py-1 text-xs font-medium ${
                        category === option
                          ? "border-brand-600 bg-brand-600 text-white"
                          : "border-slate-300 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                      }`}
                    >
                      {CATEGORY_LABELS[option]}
                    </button>
                  ))}
                  <label className="ml-auto flex items-center gap-2 text-xs">
                    <span className="sr-only">Search examples</span>
                    <input
                      type="search"
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      placeholder="Search question, schema or SQL…"
                      className="w-64 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900"
                    />
                  </label>
                </div>

                <p className="prose-note mb-3 text-xs" aria-live="polite">
                  Showing {filtered.length} of {examples.length} examples.
                </p>

                {filtered.length === 0 ? (
                  <p className="prose-note">No examples match this filter.</p>
                ) : (
                  <ul className="space-y-4">
                    {filtered.map((example) => (
                      <li key={example.id} className="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
                        <div className="mb-2 flex flex-wrap items-center gap-2">
                          <Badge
                            tone={
                              example.category === "fixed"
                                ? "good"
                                : example.category === "regressed"
                                  ? "bad"
                                  : "warn"
                            }
                          >
                            {CATEGORY_LABELS[example.category as Category]}
                          </Badge>
                          <span className="font-mono text-xs text-slate-500">{example.id}</span>
                        </div>

                        <p className="text-sm font-medium">{example.question}</p>

                        <div className="mt-3 grid gap-3 lg:grid-cols-2">
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                              Schema
                            </p>
                            <pre className="mt-1 overflow-x-auto rounded-lg bg-slate-100 p-2 text-xs dark:bg-slate-950">
                              {example.schema}
                            </pre>
                          </div>
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                              Gold SQL
                            </p>
                            <pre className="mt-1 overflow-x-auto rounded-lg bg-emerald-50 p-2 text-xs dark:bg-emerald-950/40">
                              {example.gold}
                            </pre>
                          </div>
                        </div>

                        <div className="mt-3 space-y-2">
                          {Object.entries(example.models).map(([label, output]) => (
                            <div key={label} className="rounded-lg border border-slate-100 p-2 dark:border-slate-800/60">
                              <div className="mb-1 flex items-center gap-2 text-xs">
                                <span className="font-medium">{RUN_LABELS[label] ?? label}</span>
                                {output.execution_match ? (
                                  <Badge tone="good">pass</Badge>
                                ) : (
                                  <Badge tone="bad">fail</Badge>
                                )}
                                {output.failure_type ? (
                                  <span className="text-slate-500">{output.failure_type}</span>
                                ) : null}
                              </div>
                              <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs">
                                {output.prediction || "(empty output)"}
                              </pre>
                            </div>
                          ))}
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            </>
          );
        }}
      </ArtifactBoundary>
    </div>
  );
}
