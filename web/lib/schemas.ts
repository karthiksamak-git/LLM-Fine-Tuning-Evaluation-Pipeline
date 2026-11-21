import { z } from "zod";

/**
 * Zod schemas for every artifact written by `mlfte export`.
 * They are intentionally permissive about extra keys (`.passthrough()`), strict about the
 * fields the UI actually renders, so a schema drift surfaces as a visible error state.
 */

export const metricSchema = z
  .object({
    value: z.number(),
    ci_low: z.number().nullable().optional(),
    ci_high: z.number().nullable().optional(),
    n: z.number().optional(),
  })
  .passthrough();

export const hardwareSchema = z
  .object({
    device: z.string().nullable().optional(),
    gpu_name: z.string().nullable().optional(),
    vram_gb: z.number().nullable().optional(),
    ram_total_gb: z.number().nullable().optional(),
    cpu_count_logical: z.number().nullable().optional(),
    precision_label: z.string().nullable().optional(),
  })
  .passthrough();

export const commonSchema = z
  .object({
    schema_version: z.literal(1),
    profile: z.enum(["cpu", "gpu", "smoke"]),
    model: z.string().optional(),
    dataset: z.union([z.string(), z.record(z.unknown())]).optional(),
    hardware: hardwareSchema,
    git_commit: z.string(),
    data_fingerprint: z.string().nullable().optional(),
    generated_at: z.string().optional(),
  })
  .passthrough();

export const summarySchema = commonSchema.extend({
  headline: z
    .object({
      primary_metric: z.string().optional(),
      test_n: z.number().nullable().optional(),
      execution_match: z.record(z.number()).optional(),
      exact_match: z.record(z.number()).optional(),
      best_baseline: z.string().optional(),
      gain_over_best_baseline: z
        .object({
          absolute: z.number().nullable().optional(),
          relative: z.number().nullable().optional(),
          p_value: z.number().nullable().optional(),
          significant_at_0_05: z.boolean().optional(),
        })
        .optional(),
    })
    .passthrough(),
  profile_notice: z.string().nullable().optional(),
  training: z.record(z.unknown()).optional(),
  splits: z.record(z.number()).optional(),
  pipeline_steps: z.array(z.string()).optional(),
  limitations: z.array(z.string()).optional(),
});

const bucketSchema = z
  .object({
    n: z.number(),
    exact_match: metricSchema.optional(),
    is_valid: metricSchema.optional(),
    execution_match: metricSchema.optional(),
  })
  .passthrough();

const runSchema = z
  .object({
    metrics: z
      .object({
        exact_match: metricSchema,
        is_valid: metricSchema,
        execution_match: metricSchema,
      })
      .passthrough(),
    by_bucket: z.record(bucketSchema).optional(),
    n: z.number().optional(),
    few_shot_k: z.number().optional(),
    merged: z.boolean().optional(),
  })
  .passthrough();

const mcnemarSchema = z
  .object({
    b: z.number().optional(),
    c: z.number().optional(),
    p_value: z.number().nullable(),
    method: z.string().optional(),
  })
  .passthrough();

const comparisonEntrySchema = z
  .object({
    n_paired: z.number().optional(),
    execution_match: z
      .object({
        baseline: z.record(z.unknown()).optional(),
        model: z.record(z.unknown()).optional(),
        absolute_difference: z.number().nullable().optional(),
        relative_improvement: z.number().nullable().optional(),
        mcnemar: mcnemarSchema.optional(),
        cohens_h: z.number().optional(),
      })
      .passthrough()
      .optional(),
    exact_match: z.record(z.unknown()).optional(),
    is_valid: z.record(z.unknown()).optional(),
  })
  .passthrough();

export const comparisonSchema = commonSchema.extend({
  runs: z.record(runSchema),
  comparisons: z.record(comparisonEntrySchema).optional(),
  test_set: z
    .object({ n: z.number().optional(), data_fingerprint: z.string().optional() })
    .passthrough()
    .optional(),
  seeds: z
    .object({
      n: z.number().optional(),
      note: z.string().optional(),
      execution_match: z
        .object({ mean: z.number().optional(), std: z.number().optional() })
        .optional(),
    })
    .passthrough()
    .optional(),
});

export const trainingSchema = commonSchema.extend({
  seed: z.number().optional(),
  curves: z
    .object({
      train: z.array(
        z
          .object({
            step: z.number().nullable().optional(),
            loss: z.number().nullable().optional(),
            learning_rate: z.number().nullable().optional(),
            grad_norm: z.number().nullable().optional(),
          })
          .passthrough()
      ),
      eval: z.array(
        z
          .object({ step: z.number().nullable().optional(), eval_loss: z.number().nullable() })
          .passthrough()
      ),
    })
    .optional(),
  hyperparameters: z.record(z.unknown()).optional(),
  lora: z
    .object({
      lora_r: z.number().optional(),
      lora_alpha: z.number().optional(),
      lora_dropout: z.number().optional(),
      task_type: z.string().optional(),
      target_modules: z.array(z.string()).optional(),
      trainable_parameters: z.number().optional(),
      total_parameters: z.number().optional(),
      trainable_percent: z.number().optional(),
    })
    .passthrough()
    .optional(),
  throughput: z.record(z.unknown()).optional(),
  memory: z.record(z.unknown()).optional(),
  precision: z
    .object({ label: z.string().optional(), reason: z.string().optional() })
    .passthrough()
    .optional(),
  device: z.string().nullable().optional(),
});

export const ablationsSchema = commonSchema.extend({
  points: z.array(
    z
      .object({
        grid: z.string(),
        value: z.union([z.string(), z.number()]),
        steps: z.number().optional(),
        train_examples: z.number().optional(),
        val: z
          .object({
            n: z.number().optional(),
            exact_match: metricSchema.optional(),
            is_valid: metricSchema.optional(),
            execution_match: metricSchema.optional(),
          })
          .passthrough(),
        eval_loss: z.number().nullable().optional(),
      })
      .passthrough()
  ).default([]),
  selection: z
    .object({ rationale: z.string().optional(), best_per_grid: z.record(z.unknown()).optional() })
    .passthrough()
    .optional(),
  budget: z.record(z.unknown()).optional(),
});

const latencySchema = z
  .object({
    p50_ms: z.number().optional(),
    p95_ms: z.number().optional(),
    mean_ms: z.number().optional(),
    tokens_per_second: z.number().nullable().optional(),
  })
  .passthrough();

export const efficiencySchema = commonSchema.extend({
  sizes: z
    .object({
      adapter_mb: z.number().optional(),
      adapter_trainable_percent: z.number().optional(),
      adapter_lora: z.number().optional(),
      adapter_lora_percent: z.number().optional(),
      adapter_total: z.number().optional(),
      base_trainable_percent: z.number().optional(),
    })
    .passthrough()
    .optional(),
  latency: z.record(latencySchema).optional(),
  memory: z.record(z.unknown()).optional(),
  training: z.record(z.unknown()).optional(),
  mixed_precision_benchmark: z
    .object({
      skipped: z.boolean().optional(),
      reason: z.string().optional(),
      fp32: z.record(z.unknown()).optional(),
      mixed: z.record(z.unknown()).optional(),
    })
    .passthrough()
    .optional(),
  precision: z
    .object({ label: z.string().optional(), reason: z.string().optional() })
    .passthrough()
    .optional(),
});

export const forgettingSchema = commonSchema.extend({
  before: z.record(z.unknown()).optional(),
  after: z.record(z.unknown()).optional(),
  delta: z
    .object({
      wikitext_perplexity: z.object({
        before: z.number(),
        after: z.number(),
        delta: z.number(),
        direction: z.string(),
      }),
      arc_easy_accuracy: z.object({
        before: z.number(),
        after: z.number(),
        delta: z.number(),
        direction: z.string(),
        n: z.number().optional(),
      }),
    })
    .passthrough()
    .optional(),
});

const modelOutputSchema = z
  .object({
    prediction: z.string().nullable().optional(),
    execution_match: z.boolean().nullable().optional(),
    exact_match: z.boolean().nullable().optional(),
    is_valid: z.boolean().nullable().optional(),
    failure_type: z.string().nullable().optional(),
  })
  .passthrough();

export const exampleSchema = z
  .object({
    id: z.string(),
    category: z.enum(["fixed", "regressed", "still_wrong"]),
    question: z.string(),
    schema: z.string(),
    gold: z.string(),
    models: z.record(modelOutputSchema),
  })
  .passthrough();

export const errorAnalysisSchema = commonSchema.extend({
  taxonomy: z.record(z.record(z.number())).optional(),
  outcomes: z.record(z.number()).optional(),
  examples: z.array(exampleSchema).optional(),
  method: z.string().optional(),
});

export const examplesSchema = commonSchema.extend({
  count: z.number(),
  examples: z.array(exampleSchema),
});

export const dataCardSchema = commonSchema.extend({
  splits: z.record(z.record(z.unknown())).optional(),
  cleaning: z.record(z.unknown()).optional(),
  split: z.record(z.unknown()).optional(),
  decontamination: z
    .object({ disjoint: z.boolean().optional(), method: z.string().optional() })
    .passthrough()
    .optional(),
  total_complexity: z.record(z.number()).optional(),
  prompt_template: z.record(z.string()).optional(),
  token_stats: z.record(z.unknown()).optional(),
});

export const modelCardSchema = commonSchema.extend({
  base_model: z.string().optional(),
  lora: z.record(z.unknown()).optional(),
  hyperparameters: z.record(z.unknown()).optional(),
  seeds: z.array(z.number()).optional(),
  precision: z.record(z.unknown()).optional(),
  device: z.string().nullable().optional(),
  merge_parity: z.record(z.unknown()).optional(),
  hub_url: z.string().nullable().optional(),
  hub_published: z.boolean().optional(),
  limitations: z.array(z.string()).optional(),
});

export const envSchema = commonSchema.extend({
  env: z.record(z.unknown()).optional(),
  config: z.record(z.unknown()).optional(),
});

export const schemas = {
  "summary.json": summarySchema,
  "comparison.json": comparisonSchema,
  "training.json": trainingSchema,
  "ablations.json": ablationsSchema,
  "efficiency.json": efficiencySchema,
  "forgetting.json": forgettingSchema,
  "error_analysis.json": errorAnalysisSchema,
  "examples.json": examplesSchema,
  "data_card.json": dataCardSchema,
  "model_card.json": modelCardSchema,
  "env.json": envSchema,
} as const;

export type ArtifactName = keyof typeof schemas;
export type Summary = z.infer<typeof summarySchema>;
export type Comparison = z.infer<typeof comparisonSchema>;
export type Training = z.infer<typeof trainingSchema>;
export type Ablations = z.infer<typeof ablationsSchema>;
export type Efficiency = z.infer<typeof efficiencySchema>;
export type Forgetting = z.infer<typeof forgettingSchema>;
export type ErrorAnalysis = z.infer<typeof errorAnalysisSchema>;
export type Examples = z.infer<typeof examplesSchema>;
export type DataCard = z.infer<typeof dataCardSchema>;
export type ModelCard = z.infer<typeof modelCardSchema>;

export const ARTIFACT_NAMES = Object.keys(schemas) as ArtifactName[];

export type ArtifactData = {
  "summary.json": Summary;
  "comparison.json": Comparison;
  "training.json": Training;
  "ablations.json": Ablations;
  "efficiency.json": Efficiency;
  "forgetting.json": Forgetting;
  "error_analysis.json": ErrorAnalysis;
  "examples.json": Examples;
  "data_card.json": DataCard;
  "model_card.json": ModelCard;
  "env.json": z.infer<typeof envSchema>;
};

export function parseArtifact<T extends ArtifactName>(name: T, payload: unknown): ArtifactData[T] {
  const result = schemas[name].safeParse(payload);
  if (!result.success) {
    const issues = result.error.issues
      .slice(0, 3)
      .map((issue) => `${issue.path.join(".") || "(root)"}: ${issue.message}`)
      .join("; ");
    throw new Error(`${name} does not match the expected schema — ${issues}`);
  }
  return result.data as ArtifactData[T];
}
