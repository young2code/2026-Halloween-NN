import { defineConfig } from "vitest/config";

export default defineConfig({
  root: ".",
  base: "./",
  publicDir: "public",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
  },
  test: {
    include: ["app/**/*.test.ts"],
  },
});

