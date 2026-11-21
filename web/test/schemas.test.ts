import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  ARTIFACT_NAMES,
  exampleSchema,
  parseArtifact,
  schemas,
  summarySchema,
} from "../lib/schemas";

const DATA_DIR = join(process.cwd(), "public", "data");

function load(name: string): unknown {
  return JSON.parse(readFileSync(join(DATA_DIR, name), "utf8"));
}

describe("data artifacts", () => {
  it("ships every artifact the dashboard expects", () => {
    const files = readdirSync(DATA_DIR);
    for (const name of ARTIFACT_NAMES) {
      expect(files, `missing ${name}`).toContain(name);
    }
  });

  for (const name of ARTIFACT_NAMES) {
    it(`${name} matches its zod schema`, () => {
      const payload = load(name);
      const result = schemas[name].safeParse(payload);
      if (!result.success) {
        throw new Error(
          `${name}: ${result.error.issues
            .map((issue) => `${issue.path.join(".")} ${issue.message}`)
            .join("; ")}`
        );
      }
      expect(result.success).toBe(true);
    });
  }

  it("summary carries profile, hardware and a headline", () => {
    const summary = parseArtifact("summary.json", load("summary.json"));
    expect(["cpu", "gpu", "smoke"]).toContain(summary.profile);
    expect(summary.hardware).toBeTypeOf("object");
    expect(summary.headline.primary_metric).toBeTypeOf("string");
  });

  it("every curated example has the fields the explorer renders", () => {
    const examples = parseArtifact("examples.json", load("examples.json"));
    expect(examples.examples.length).toBeGreaterThan(0);
    for (const example of examples.examples) {
      const parsed = exampleSchema.parse(example);
      expect(parsed.question.length).toBeGreaterThan(0);
      expect(parsed.gold.length).toBeGreaterThan(0);
      expect(Object.keys(parsed.models).length).toBeGreaterThan(0);
    }
  });

  it("reports a schema violation instead of silently accepting it", () => {
    const broken = { ...(load("summary.json") as Record<string, unknown>), schema_version: 999 };
    expect(() => parseArtifact("summary.json", broken)).toThrow(/schema_version/);
    expect(() => parseArtifact("summary.json", { profile: "cpu" })).toThrow();
    expect(summarySchema.safeParse({}).success).toBe(false);
  });
});
