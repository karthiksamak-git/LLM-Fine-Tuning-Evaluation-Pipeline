"use client";

import { ArtifactBoundary, HardwareBadges } from "@/components/ArtifactBoundary";
import { Badge, Card, PageHeader } from "@/components/ui";
import { number } from "@/lib/format";
import { useArtifact } from "@/lib/useArtifact";

export default function ModelCardPage() {
  const modelCard = useArtifact("model_card.json");
  const env = useArtifact("env.json");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model card"
        description="What was trained, how, with which seeds and library versions — plus the known limitations and the merge-parity check that proves the exported weights are equivalent to the trained adapter."
      >
        <ArtifactBoundary state={modelCard} artifact="model_card.json">
          {(data) => <HardwareBadges hardware={data.hardware} profile={data.profile} />}
        </ArtifactBoundary>
      </PageHeader>

      <ArtifactBoundary state={modelCard} artifact="model_card.json">
        {(data) => {
          const lora = (data.lora ?? {}) as Record<string, unknown>;
          const parity = (data.merge_parity ?? {}) as Record<string, unknown>;
          const hyper = (data.hyperparameters ?? {}) as Record<string, unknown>;

          return (
            <>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Base model and adapter">
                  <dl className="grid gap-2 text-sm">
                    <Row label="base model" value={data.base_model ?? "n/a"} mono />
                    <Row label="LoRA rank" value={String(lora.r ?? "n/a")} />
                    <Row label="alpha" value={String(lora.alpha ?? "n/a")} />
                    <Row label="dropout" value={String(lora.dropout ?? "n/a")} />
                    <Row
                      label="target modules"
                      value={Array.isArray(lora.target_modules) ? lora.target_modules.join(", ") : "n/a"}
                      mono
                    />
                    <Row
                      label="trainable parameters"
                      value={`${number(lora.trainable_parameters as number)} / ${number(
                        lora.total_parameters as number
                      )}`}
                    />
                    <Row
                      label="trainable %"
                      value={
                        lora.trainable_percent !== undefined
                          ? `${Number(lora.trainable_percent).toFixed(3)}%`
                          : "n/a"
                      }
                    />
                  </dl>
                </Card>

                <Card title="Reproducibility">
                  <dl className="grid gap-2 text-sm">
                    <Row label="seeds" value={(data.seeds ?? []).join(", ") || "n/a"} />
                    <Row label="git commit" value={data.git_commit.slice(0, 12)} mono />
                    <Row
                      label="data fingerprint"
                      value={(data.data_fingerprint ?? "n/a").slice(0, 12)}
                      mono
                    />
                    <Row
                      label="precision"
                      value={String((data.precision as Record<string, unknown>)?.label ?? "n/a")}
                    />
                    <Row label="device" value={String(data.device ?? "n/a")} />
                    <Row
                      label="Hub"
                      value={
                        data.hub_published && data.hub_url ? (
                          <a className="underline" href={data.hub_url} rel="noreferrer noopener">
                            published
                          </a>
                        ) : (
                          "not published (HF_TOKEN not set)"
                        )
                      }
                    />
                  </dl>
                </Card>
              </div>

              <Card title="Hyperparameters">
                <div className="grid gap-2 sm:grid-cols-2">
                  {Object.entries(hyper).map(([key, value]) => (
                    <Row key={key} label={key.replace(/_/g, " ")} value={String(value)} />
                  ))}
                </div>
              </Card>

              <Card
                title="Merge parity"
                subtitle="Logit difference between the trained adapter and the merged model on a fixed batch. The export fails loudly if it exceeds the documented tolerance."
              >
                {Object.keys(parity).length ? (
                  <>
                    <div className="mb-2 flex items-center gap-2">
                      <Badge tone={parity.passed ? "good" : "bad"}>
                        {parity.passed ? "passed" : "failed"}
                      </Badge>
                    </div>
                    <dl className="grid gap-2 text-sm sm:grid-cols-2">
                      <Row
                        label="max abs logit diff"
                        value={String(parity.max_abs_logit_diff ?? "n/a")}
                        mono
                      />
                      <Row
                        label="mean abs logit diff"
                        value={String(parity.mean_abs_logit_diff ?? "n/a")}
                        mono
                      />
                      <Row label="tolerance" value={String(parity.tolerance_max_abs ?? "n/a")} mono />
                      <Row label="batch shape" value={JSON.stringify(parity.batch_shape ?? [])} mono />
                    </dl>
                  </>
                ) : (
                  <p className="prose-note">No parity record.</p>
                )}
              </Card>

              <Card title="Limitations">
                <ul className="list-disc space-y-1.5 pl-4 text-sm">
                  {(data.limitations ?? []).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </Card>

              <ArtifactBoundary state={env} artifact="env.json">
                {(envData) => (
                  <Card title="Environment">
                    <pre className="overflow-x-auto rounded-lg bg-slate-100 p-3 text-xs dark:bg-slate-950">
                      {JSON.stringify(
                        {
                          packages: (envData as unknown as { package_versions?: unknown })
                            .package_versions,
                          env: envData.env,
                        },
                        null,
                        2
                      )}
                    </pre>
                  </Card>
                )}
              </ArtifactBoundary>
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
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-100 pb-1.5 dark:border-slate-800/60">
      <dt className="text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className={`text-right ${mono ? "font-mono text-xs" : ""}`}>{value}</dd>
    </div>
  );
}
