"use client";

import { ArtifactBoundary, HardwareBadges } from "@/components/ArtifactBoundary";
import { GroupedBars } from "@/components/charts";
import { Card, PageHeader } from "@/components/ui";
import { pct } from "@/lib/format";
import { useArtifact } from "@/lib/useArtifact";

const GRID_ORDER = ["rank", "learning_rate", "target_modules", "train_size", "loss_masking"];

const GRID_TITLES: Record<string, string> = {
  rank: "LoRA rank (r)",
  learning_rate: "Learning rate",
  target_modules: "Target modules",
  train_size: "Training-set size",
  loss_masking: "Loss masking",
};

export default function AblationsPage() {
  const ablations = useArtifact("ablations.json");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Ablations"
        description="Each point retrains from scratch with one setting changed, then is scored on the validation split. Selection uses validation only — the test split is never used to pick a config."
      >
        <ArtifactBoundary state={ablations} artifact="ablations.json">
          {(data) => <HardwareBadges hardware={data.hardware} profile={data.profile} />}
        </ArtifactBoundary>
      </PageHeader>

      <ArtifactBoundary state={ablations} artifact="ablations.json">
        {(data) => {
          const points = data.points ?? [];
          if (!points.length) {
            return (
              <Card title="No ablation points">
                <p className="prose-note">
                  This artifact contains no points (the ablation step may have been skipped or
                  failed). Re-run <code className="font-mono">mlfte ablate</code>.
                </p>
              </Card>
            );
          }
          const grids = GRID_ORDER.filter((grid) => points.some((point) => point.grid === grid));
          const budget = data.budget as Record<string, unknown> | undefined;

          return (
            <>
              {budget ? (
                <Card title="Budget">
                  <p className="prose-note">
                    {String(budget.note ?? "")} Training steps per point:{" "}
                    <strong>{String(budget.max_steps ?? "?")}</strong>; validation examples:{" "}
                    <strong>{String(budget.val_examples ?? "?")}</strong>.
                  </p>
                </Card>
              ) : null}

              <div className="grid gap-4 lg:grid-cols-2">
                {grids.map((grid) => {
                  const rows = points.filter((point) => point.grid === grid);
                  const chartData = rows.map((point) => ({
                    label: String(point.value),
                    "validation execution match": point.val.execution_match?.value ?? 0,
                  }));
                  return (
                    <Card key={grid} title={GRID_TITLES[grid] ?? grid}>
                      <GroupedBars
                        data={chartData}
                        series={[
                          {
                            key: "validation execution match",
                            label: "val execution match",
                          },
                        ]}
                        caption={`Validation execution match for ${GRID_TITLES[grid] ?? grid}.`}
                        ariaLabel={`Bar chart of validation execution match for the ${grid} ablation`}
                        height={260}
                      />
                    </Card>
                  );
                })}
              </div>

              <Card title="All ablation points">
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr>
                        <th className="th">Grid</th>
                        <th className="th">Value</th>
                        <th className="th">Steps</th>
                        <th className="th">Train examples</th>
                        <th className="th">Val execution match</th>
                        <th className="th">Val exact match</th>
                        <th className="th">Best val loss</th>
                      </tr>
                    </thead>
                    <tbody>
                      {points.map((point) => (
                        <tr key={`${point.grid}-${point.value}`}>
                          <td className="td">{GRID_TITLES[point.grid] ?? point.grid}</td>
                          <td className="td font-mono text-xs">{String(point.value)}</td>
                          <td className="td">{point.steps ?? "?"}</td>
                          <td className="td">{point.train_examples ?? "?"}</td>
                          <td className="td">{pct(point.val.execution_match?.value)}</td>
                          <td className="td">{pct(point.val.exact_match?.value)}</td>
                          <td className="td">
                            {point.eval_loss !== null && point.eval_loss !== undefined
                              ? point.eval_loss.toFixed(4)
                              : "n/a"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              {data.selection?.rationale ? (
                <Card title="Selection">
                  <p className="prose-note">{data.selection.rationale}</p>
                </Card>
              ) : null}
            </>
          );
        }}
      </ArtifactBoundary>
    </div>
  );
}
