import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {title ? (
        <header className="mb-3">
          <h2 className="text-base font-semibold">{title}</h2>
          {subtitle ? <p className="prose-note mt-1">{subtitle}</p> : null}
        </header>
      ) : null}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <div className="card">
      <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
      <p className="stat-value mt-1">{value}</p>
      {hint ? <p className="prose-note mt-1 text-xs">{hint}</p> : null}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad" | "info";
  title?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200",
    good: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200",
    warn: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200",
    bad: "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-200",
    info: "bg-brand-100 text-brand-700 dark:bg-brand-700/30 dark:text-brand-100",
  };
  return (
    <span
      title={title}
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function PageHeader({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children?: ReactNode;
}) {
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {description ? <p className="prose-note mt-2 max-w-3xl">{description}</p> : null}
      {children ? <div className="mt-3 flex flex-wrap gap-2">{children}</div> : null}
    </div>
  );
}

export function Table({ head, children }: { head: string[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left">
        <thead>
          <tr>
            {head.map((cell) => (
              <th key={cell} scope="col" className="th">
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Loading({ label = "Loading results…" }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="card animate-pulse text-sm text-slate-500 dark:text-slate-400"
    >
      {label}
    </div>
  );
}

export function ErrorState({ message, artifact }: { message: string; artifact: string }) {
  return (
    <div role="alert" className="card border-rose-300 bg-rose-50 dark:border-rose-900 dark:bg-rose-950/40">
      <h2 className="text-base font-semibold text-rose-800 dark:text-rose-200">
        Could not load {artifact}
      </h2>
      <p className="mt-2 text-sm text-rose-700 dark:text-rose-300">{message}</p>
      <p className="mt-3 text-xs text-rose-600 dark:text-rose-400">
        The data files are validated against zod schemas. Regenerate them with{" "}
        <code className="font-mono">python -m mlfte.cli export --profile &lt;profile&gt;</code>, or check
        that <code className="font-mono">web/public/data/{artifact}</code> exists.
      </p>
    </div>
  );
}
