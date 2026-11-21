"use client";

import { ArtifactBoundary, HardwareBadges } from "@/components/ArtifactBoundary";
import { Badge, Card, PageHeader, Stat } from "@/components/ui";
import { number } from "@/lib/format";
import { useArtifact } from "@/lib/useArtifact";

type SplitStats = {
  count?: number;
  complexity?: Record<string, number>;
  question_chars?: Record<string, number>;
  schema_chars?: Record<string, number>;
  answer_chars?: Record<string, number>;
};

export default function DataPage() {
  const card = useArtifact("data_card.json");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dataset card"
        description="Cleaning, deduplication, the fixed split, the decontamination proof and the exact prompt template. The test split is created once and never used for training or selection."
      >
        <ArtifactBoundary state={card} artifact="data_card.json">
          {(data) => <HardwareBadges hardware={data.hardware} profile={data.profile} />}
        </ArtifactBoundary>
      </PageHeader>

      <ArtifactBoundary state={card} artifact="data_card.json">
        {(data) => {
          const splits = (data.splits ?? {}) as Record<string, SplitStats>;
          const cleaning = (data.cleaning ?? {}) as Record<string, unknown>;
          const decontam = (data.decontamination ?? {}) as {
            disjoint?: boolean;
            method?: string;
            pairs?: Record<string, { exact_overlap?: number; near_duplicate_pairs?: number }>;
          };
          const split = (data.split ?? {}) as Record<string, unknown>;
          const template = data.prompt_template ?? {};
          const tokenStats = (data.token_stats ?? {}) as Record<string, unknown>;

          return (
            <>
              <div className="grid gap-4 sm:grid-cols-3">
                {(["train", "val", "test"] as const).map((name) => (
                  <Stat
                    key={name}
                    label={`${name} split`}
                    value={number(splits[name]?.count)}
                    hint={
                      splits[name]?.complexity
                        ? Object.entries(splits[name].complexity ?? {})
                            .map(([bucket, count]) => `${bucket}: ${count}`)
                            .join(" · ")
                        : undefined
                    }
                  />
                ))}
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Cleaning and splits">
                  <dl className="grid gap-2 text-sm">
                    <Row label="raw rows" value={number(cleaning.raw_rows as number)} />
                    <Row label="kept" value={number(cleaning.kept as number)} />
                    <Row
                      label="exact duplicates removed"
                      value={number(cleaning.exact_duplicates_removed as number)}
                    />
                    <Row
                      label="dropped"
                      value={JSON.stringify(cleaning.dropped ?? {}, null, 0)}
                      mono
                    />
                    <Row label="split seed" value={String(split.seed ?? "n/a")} />
                  </dl>
                  {split.policy ? <p className="prose-note mt-3 text-xs">{String(split.policy)}</p> : null}
                </Card>

                <Card title="Decontamination">
                  <div className="flex items-center gap-2">
                    <Badge tone={decontam.disjoint ? "good" : "bad"}>
                      {decontam.disjoint ? "splits are disjoint" : "overlap detected"}
                    </Badge>
                  </div>
                  <p className="prose-note mt-2 text-xs">{decontam.method}</p>
                  {decontam.pairs ? (
                    <table className="mt-3 w-full border-collapse text-sm">
                      <thead>
                        <tr>
                          <th className="th">Pair</th>
                          <th className="th">Exact overlap</th>
                          <th className="th">Near duplicates</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(decontam.pairs).map(([pair, payload]) => (
                          <tr key={pair}>
                            <td className="td font-mono text-xs">{pair.replace("__", " / ")}</td>
                            <td className="td">{payload.exact_overlap ?? 0}</td>
                            <td className="td">{payload.near_duplicate_pairs ?? 0}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : null}
                </Card>
              </div>

              <Card title="Prompt template" subtitle="Rendered through the tokenizer's chat template.">
                <pre className="overflow-x-auto whitespace-pre-wrap rounded-lg bg-slate-100 p-3 text-xs dark:bg-slate-950">
{`system:
${template.system ?? ""}

user:
${template.user ?? ""}

assistant:
${template.assistant ?? ""}`}
                </pre>
                <p className="prose-note mt-2 text-xs">{template.rendering}</p>
              </Card>

              <Card title="Token statistics and truncation report">
                <pre className="overflow-x-auto rounded-lg bg-slate-100 p-3 text-xs dark:bg-slate-950">
                  {JSON.stringify(tokenStats, null, 2)}
                </pre>
                <p className="prose-note mt-2 text-xs">
                  The pipeline raises if the truncation fraction reaches 1% on any split; the
                  reported value is therefore the real, gate-checked number.
                </p>
              </Card>

              <Card title="Attribution">
                <p className="prose-note text-sm">
                  {(data.dataset as unknown as { name?: string })?.name ?? "b-mc2/sql-create-context"}{" "}
                  — CC-BY-4.0. Prompt/answer pairs derived from the Spider and WikiSQL text-to-SQL
                  corpora.
                </p>
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
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-100 pb-1.5 dark:border-slate-800/60">
      <dt className="text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className={`text-right ${mono ? "font-mono text-xs" : ""}`}>{value}</dd>
    </div>
  );
}
