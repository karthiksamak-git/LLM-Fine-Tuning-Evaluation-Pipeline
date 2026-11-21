"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { PALETTE } from "@/lib/format";

function Figure({
  caption,
  ariaLabel,
  children,
  height = 300,
}: {
  caption: string;
  ariaLabel: string;
  children: React.ReactElement;
  height?: number;
}) {
  return (
    <figure className="w-full" role="group" aria-label={ariaLabel}>
      <div style={{ width: "100%", height }} aria-hidden="true">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-2 text-xs text-slate-500 dark:text-slate-400">{caption}</figcaption>
    </figure>
  );
}

export type GroupedSeries = { key: string; label: string };

export function GroupedBars({
  data,
  series,
  xKey = "label",
  height = 320,
  caption,
  ariaLabel,
  percent = true,
}: {
  data: Array<Record<string, string | number>>;
  series: GroupedSeries[];
  xKey?: string;
  height?: number;
  caption: string;
  ariaLabel: string;
  percent?: boolean;
}) {
  return (
    <Figure caption={caption} ariaLabel={ariaLabel} height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
        <XAxis dataKey={xKey} tick={{ fontSize: 11 }} interval={0} />
        <YAxis
          tick={{ fontSize: 11 }}
          domain={percent ? [0, 1] : undefined}
          tickFormatter={(value: number) => (percent ? `${Math.round(value * 100)}%` : String(value))}
        />
        <Tooltip
          formatter={(value: number) => (percent ? `${(value * 100).toFixed(1)}%` : value)}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map((entry, index) => (
          <Bar
            key={entry.key}
            dataKey={entry.key}
            name={entry.label}
            fill={PALETTE[index % PALETTE.length]}
            radius={[3, 3, 0, 0]}
          />
        ))}
      </BarChart>
    </Figure>
  );
}

export function HorizontalBars({
  data,
  height = 320,
  caption,
  ariaLabel,
  color = PALETTE[0],
}: {
  data: Array<{ label: string; value: number }>;
  height?: number;
  caption: string;
  ariaLabel: string;
  color?: string;
}) {
  return (
    <Figure caption={caption} ariaLabel={ariaLabel} height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 40 }}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
        <XAxis type="number" tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="label" tick={{ fontSize: 10 }} width={130} />
        <Tooltip />
        <Bar dataKey="value" name="count" radius={[0, 3, 3, 0]}>
          {data.map((entry, index) => (
            <Cell key={entry.label} fill={index === 0 ? color : PALETTE[(index + 1) % PALETTE.length]} />
          ))}
        </Bar>
      </BarChart>
    </Figure>
  );
}

export function Lines({
  data,
  series,
  xKey,
  height = 300,
  caption,
  ariaLabel,
}: {
  data: Array<Record<string, number | null>>;
  series: GroupedSeries[];
  xKey: string;
  height?: number;
  caption: string;
  ariaLabel: string;
}) {
  return (
    <Figure caption={caption} ariaLabel={ariaLabel} height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
        <XAxis dataKey={xKey} tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip formatter={(value: number) => (typeof value === "number" ? value.toFixed(4) : value)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map((entry, index) => (
          <Line
            key={entry.key}
            type="monotone"
            dataKey={entry.key}
            name={entry.label}
            stroke={PALETTE[index % PALETTE.length]}
            dot={false}
            strokeWidth={2}
            connectNulls
          />
        ))}
      </LineChart>
    </Figure>
  );
}
