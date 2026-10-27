import { mkdir } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright-core";

const chrome = process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const baseUrl = process.env.PREVIEW_URL || "http://127.0.0.1:4173/";
const outputDirectory = path.resolve("docs");

await mkdir(outputDirectory, { recursive: true });

const browser = await chromium.launch({ executablePath: chrome, headless: true });
const page = await browser.newPage({
  viewport: { width: 1440, height: 1000 },
  deviceScaleFactor: 1,
});

try {
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator("#runtime-pill.ready").waitFor({ timeout: 30_000 });
  await page.locator("#generation-status").filter({ hasText: "Carved locally" }).waitFor({ timeout: 30_000 });

  await page.locator("#prompt").fill("tall green spooky pumpkin with star eyes fangs and blue glow");
  await page.locator("details:not(.vocabulary-guide) summary").click();
  await page.locator("#seed").fill("4242");
  await page.locator("#carve").click();
  await page.locator("#generation-status").filter({ hasText: "seed 4242" }).waitFor({ timeout: 30_000 });

  await page.locator("#ascii-output").screenshot({
    path: path.join(outputDirectory, "sample-output-ascii.png"),
  });

  await page.locator("#ascii-toggle").click();
  await page.locator("#pumpkin-canvas").screenshot({
    path: path.join(outputDirectory, "sample-output-image.png"),
  });
} finally {
  await browser.close();
}

console.log("Captured the fixed-seed README samples in docs/.");
