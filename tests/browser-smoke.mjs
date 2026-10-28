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
  if ((await page.locator("#surprise").innerText()).trim().toUpperCase() !== "SURPRISE ME") {
    throw new Error("The surprise button does not have a visible text label.");
  }
  const diceIconSize = await page.locator("#surprise svg").evaluate((icon) => icon.getBoundingClientRect().width);
  if (diceIconSize < 24) throw new Error(`The surprise dice icon is too small: ${diceIconSize}px.`);
  const clickTime = await page.evaluate(() => Date.now() >>> 0);
  await page.locator("#surprise").click();
  const surpriseSeed = Number(await page.locator("#seed").inputValue());
  const seedDistance = Math.min(
    Math.abs(surpriseSeed - clickTime),
    2 ** 32 - Math.abs(surpriseSeed - clickTime),
  );
  if (seedDistance > 5_000) throw new Error(`Surprise seed ${surpriseSeed} was not based on click time ${clickTime}.`);
  await page.locator("#generation-status").filter({ hasText: `seed ${surpriseSeed}` }).waitFor({ timeout: 30_000 });
  if (await page.locator("#surprise").isDisabled()) throw new Error("The surprise button stayed disabled after auto-carving.");
  const surpriseWords = (await page.locator("#prompt").inputValue()).match(/[a-z]+(?:-[a-z]+)*/gi) ?? [];
  if (surpriseWords.length < 12) throw new Error(`Surprise prompt was too simple: ${surpriseWords.join(" ")}`);
  if (!(await page.locator("#prompt-note").innerText()).includes("All prompt words recognized")) {
    throw new Error("Surprise prompt contained words outside the supported vocabulary.");
  }
  if ((await page.locator("#chips").count()) !== 0) throw new Error("The duplicate compact vocabulary widget still exists.");
  const selectedTerms = page.locator("#vocabulary-list button[aria-pressed='true']");
  if ((await selectedTerms.count()) !== 7) throw new Error("A surprise prompt did not select one vocabulary term per group.");
  const replacement = page.locator("#vocabulary-list button[data-group='shape'][aria-pressed='false']").first();
  const replacementTerm = (await replacement.innerText()).trim();
  await replacement.click();
  if ((await selectedTerms.count()) !== 7 || !(await page.locator("#prompt").inputValue()).includes(replacementTerm)) {
    throw new Error("The primary vocabulary widget did not replace and select the chosen term.");
  }
  await page.locator("#prompt").fill("tall green spooky pumpkin with star eyes fangs and blue glow");
  await page.locator("details:not(.vocabulary-guide) summary").click();
  await page.locator("#seed").fill("4242");
  await page.locator("#carve").click();
  await page.locator("#generation-status").filter({ hasText: "seed 4242" }).waitFor({ timeout: 30_000 });
  await page.locator("#ascii-output").waitFor({ state: "visible" });
  await page.locator("#pumpkin-canvas").waitFor({ state: "visible" });
  const asciiRows = await page.locator(".ascii-row").count();
  const outputViews = await page.locator(".output-view").count();
  if (outputViews !== 2) throw new Error(`Expected two simultaneous output views, found ${outputViews}.`);
  const frames = await page.locator(".output-view .canvas-frame").evaluateAll((elements) =>
    elements.map((element) => {
      const { x, width, height } = element.getBoundingClientRect();
      return { x, width, height };
    }),
  );
  if (
    frames.length !== 2
    || Math.abs(frames[0].width - frames[1].width) > 1
    || Math.abs(frames[0].height - frames[1].height) > 1
    || frames[1].x <= frames[0].x + frames[0].width
  ) {
    throw new Error(`Output frames were not equal-sized and side by side: ${JSON.stringify(frames)}`);
  }
  const widestAsciiRow = await page.locator(".ascii-row").evaluateAll((rows) =>
    Math.max(...rows.map((row) => row.getBoundingClientRect().width)),
  );
  if (widestAsciiRow < frames[1].width * 0.75) {
    throw new Error(`ASCII portrait is too small: ${widestAsciiRow}px inside a ${frames[1].width}px frame.`);
  }
  if (!forceWasm) {
    await page.screenshot({ path: path.resolve("training/runs/browser-preview.png"), fullPage: true });
  }
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

