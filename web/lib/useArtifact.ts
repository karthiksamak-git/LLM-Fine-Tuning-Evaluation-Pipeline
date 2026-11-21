"use client";

import { useEffect, useState } from "react";

import { parseArtifact, type ArtifactData, type ArtifactName } from "./schemas";

export type ArtifactState<T> =
  | { status: "loading"; data: null; error: null }
  | { status: "ready"; data: T; error: null }
  | { status: "error"; data: null; error: string };

/**
 * Fetch one static artifact from /data and validate it with zod.
 * A schema failure becomes a visible error state instead of a broken page.
 */
export function useArtifact<T extends ArtifactName>(name: T): ArtifactState<ArtifactData[T]> {
  const [state, setState] = useState<ArtifactState<ArtifactData[T]>>({
    status: "loading",
    data: null,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const response = await fetch(`/data/${name}`, { cache: "no-cache" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} while loading /data/${name}`);
        }
        const payload: unknown = await response.json();
        const parsed = parseArtifact(name, payload);
        if (!cancelled) {
          setState({ status: "ready", data: parsed, error: null });
        }
      } catch (error) {
        if (!cancelled) {
          setState({
            status: "error",
            data: null,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [name]);

  return state;
}
