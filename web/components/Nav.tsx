"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { ThemeToggle } from "./ThemeToggle";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/results", label: "Results" },
  { href: "/training", label: "Training" },
  { href: "/ablations", label: "Ablations" },
  { href: "/forgetting", label: "Forgetting" },
  { href: "/efficiency", label: "Efficiency" },
  { href: "/errors", label: "Errors" },
  { href: "/data", label: "Data" },
  { href: "/model-card", label: "Model card" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-ink-900/90">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="font-semibold tracking-tight">
          llm-finetune-eval
          <span className="ml-2 rounded-md bg-brand-100 px-2 py-0.5 text-xs font-medium text-brand-700 dark:bg-brand-700/30 dark:text-brand-100">
            text-to-SQL
          </span>
        </Link>
        <nav aria-label="Main" className="flex flex-1 flex-wrap gap-x-4 gap-y-1 text-sm">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={
                  active
                    ? "font-semibold text-brand-600 dark:text-brand-300"
                    : "text-slate-600 hover:text-brand-600 dark:text-slate-300 dark:hover:text-brand-300"
                }
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
        <ThemeToggle />
      </div>
    </header>
  );
}
