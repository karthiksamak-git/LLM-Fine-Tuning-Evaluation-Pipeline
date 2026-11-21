import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Everything under public/data is static JSON produced by `mlfte export`.
  // No server-side inference and no secrets exist in this app.
  outputFileTracingIncludes: {},
};

export default nextConfig;
