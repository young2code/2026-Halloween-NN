import { chromium } from "playwright-core";
import path from "node:path";

const chrome = process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const baseUrl = process.env.PREVIEW_URL || "http://127.0.0.1:4173/";
const browser = await chromium.launch({
  executablePath: chrome,
  headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan"],
});

async function exercise(forceWasm) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  if (forceWasm) {
    await context.addInitScript(() => Object.defineProperty(navigator, "gpu", { value: undefined }));
  }
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator("#runtime-pill.ready").waitFor({ timeout: 30_000 });
  await page.locator("#generation-status").filter({ hasText: "Carved locally" }).waitFor({ timeout: 30_000 });
  const vocabularyTerms = await page.locator("#vocabulary-list button").count();
  if (vocabularyTerms !== 100) throw new Error(`Expected 100 visible vocabulary terms, found ${vocabularyTerms}.`);
  const runtime = (await page.locator("#runtime-pill").innerText()).trim();
  const firstStatus = (await page.locator("#generation-status").innerText()).trim();
  await page.locator("#prompt").fill("tall green spooky pumpkin with star eyes fangs and blue glow");
  await page.locator("details:not(.vocabulary-guide) summary").click();
  await page.locator("#seed").fill("4242");
  await page.locator("#carve").click();
  await page.locator("#generation-status").filter({ hasText: "seed 4242" }).waitFor({ timeout: 30_000 });
  await page.locator("#ascii-output").waitFor({ state: "visible" });
  const asciiRows = await page.locator(".ascii-row").count();
  if ((await page.locator("#ascii-toggle").innerText()).trim() !== "SHOW IMAGE") {
    throw new Error("ASCII portrait was not the default result view.");
  }
  if (!forceWasm) {
    await page.screenshot({ path: path.resolve("training/runs/browser-preview.png"), fullPage: true });
  }
  await page.locator("#ascii-toggle").click();
  await page.locator("#pumpkin-canvas").waitFor({ state: "visible" });
  const canvasSignal = await page.locator("#pumpkin-canvas").evaluate((canvas) => {
    const context = canvas.getContext("2d");
    return [...context.getImageData(0, 0, 8, 8).data].reduce((sum, value) => sum + value, 0);
  });
  await context.close();
  const actionableErrors = errors.filter((error) => !error.includes("Unknown CPU vendor"));
  if (actionableErrors.length) throw new Error(`${forceWasm ? "WASM" : "default"} browser errors:\n${actionableErrors.join("\n")}`);
  if (asciiRows < 10 || canvasSignal === 0) throw new Error("Generated output did not render correctly.");
  return { runtime, firstStatus, asciiRows, canvasSignal };
}

try {
  const normal = await exercise(false);
  const wasm = await exercise(true);
  console.log(JSON.stringify({ normal, wasm }, null, 2));
} finally {
  await browser.close();
}

