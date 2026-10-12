import type { NextConfig } from "next";

const config: NextConfig = {
  output: "standalone",
  transpilePackages: ["@wasifyai-art/contracts"],
};
export default config;
