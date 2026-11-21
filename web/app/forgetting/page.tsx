"use client";

import { ArtifactBoundary, HardwareBadges } from "@/components/ArtifactBoundary";
import { GroupedBars } from "@/components/charts";
import { Badge, Card, PageHeader } from "@/components/ui";
import { pct } from "@/lib/format";
import { useArtifact } from "@/lib/useArtifact";

export default function ForgettingPage() {
  const forgetting = useArtifact("forgetting.json");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Forgetting checks"
        description="Fine-tuning can damage general ability. Two independent probes are measured before and after training on a fixed slice: WikiText-2 perplexity (token-level NLL) and ARC-Easy accuracy by mean log-likelihood over answer options."
      >
        <ArtifactBoundary state={forgetting} artifact="forgetting.json">
          {(data) => <HardwareBadges hardware={data.hardware} profile={data.profile} />}
        </ArtifactBoundary>
      </PageHeader>

      <ArtifactBoundary state={forgetting} artifact="forgetting.json">
        {(data) => {
          const delta = data.delta;
          if (!delta) {
            return (
              <Card title="No forgetting data">
                <p className="prose-note">
                  Run <code className="font-mono">mlfte baseline</code> and{" "}
                  <code className="font-mono">mlfte eval</code> to produce before/after checks.
                </p>
              </Card>
            );
          }
          const ppl = delta.wikitext_perplexity;
          const arc = delta.arc_easy_accuracy;

          return (
            <>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="WikiText-2 perplexity (lower is better)">
                  <GroupedBars
                    data={[
                      { label: "before", perplexity: ppl.before },
                      { label: "after", perplexity: ppl.after },
                    ]}
                    series={[{ key: "perplexity", label: "perplexity" }]}
                    percent={false}
                    caption="Perplexity on a fixed WikiText-2 test slice before and after LoRA fine-tuning."
                    ariaLabel="Bar chart of WikiText-2 perplexity before and after fine-tuning"
                    height={240}
                  />
                  <p className="prose-note mt-2 text-sm">
                    Δ {ppl.delta.toFixed(3)}{" "}
                    <Badge tone={ppl.direction === "degraded" ? "bad" : "good"}>
                      {ppl.direction}
                    </Badge>
                  </p>
                </Card>

                <Card title={`ARC-Easy accuracy (n = ${arc.n ?? "?"})`}>
                  <GroupedBars
                    data={[
                      { label: "before", accuracy: arc.before },
                      { label: "after", accuracy: arc.after },
                    ]}
                    series={[{ key: "accuracy", label: "accuracy" }]}
                    caption="ARC-Easy accuracy before and after LoRA fine-tuning."
                    ariaLabel="Bar chart of ARC-Easy accuracy before and after fine-tuning"
                    height={240}
                  />
                  <p className="prose-note mt-2 text-sm">
                    Δ {(arc.delta * 100).toFixed(2)} pp{" "}
                    <Badge tone={arc.direction === "degraded" ? "bad" : "good"}>
                      {arc.direction}
                    </Badge>
                  </p>
                </Card>
              </div>

              <Card title="Numbers">
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr>
                        <th className="th">Check</th>
                        <th className="th">Before</th>
                        <th className="th">After</th>
                        <th className="th">Change</th>
                        <th className="th">Direction</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="td">WikiText-2 perplexity</td>
                        <td className="td">{ppl.before.toFixed(3)}</td>
                        <td className="td">{ppl.after.toFixed(3)}</td>
                        <td className="td">{ppl.delta.toFixed(3)}</td>
                        <td className="td">{ppl.direction}</td>
                      </tr>
                      <tr>
                        <td className="td">ARC-Easy accuracy</td>
                        <td className="td">{pct(arc.before)}</td>
                        <td className="td">{pct(arc.after)}</td>
                        <td className="td">{(arc.delta * 100).toFixed(2)} pp</td>
                        <td className="td">{arc.direction}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <p className="prose-note mt-3 text-xs">
                  Degradation is reported honestly when present. These probes are small slices, so
                  a few points of movement are within noise; no confidence intervals are claimed for
                  the difference here.
                </p>
              </Card>
            </>
          );
        }}
      </ArtifactBoundary>
    </div>
  );
}
