/**
 * Record the TakeoffLens demo GIF.
 *
 *   docker compose up -d
 *   node docs/record_demo.mjs
 *
 * Drives a real browser through the real app: upload page -> upload a plan -> wait for
 * processing -> viewer -> OCR overlay -> hover rooms -> switch approach -> export CSV.
 * Nothing is mocked and no API call is made to Anthropic: the interactive path runs
 * PIPELINE_APPROACHES, which defaults to `rules` alone, and the `vlm` and `hybrid` rooms the
 * approach toggle switches between are replayed from the already-paid-for evaluation cache by
 * docs/seed_demo_sources.py.
 *
 * Writes docs/demo.raw.webm plus docs/demo.phases.json, which records when each phase started
 * so the render step can keep interactions at real speed and compress only the OCR wait.
 * docs/make_demo_gif.mjs does the rendering.
 */

import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const OUT_DIR = resolve(HERE, "recordings");

const WEB = process.env.DEMO_WEB_URL ?? "http://localhost:3000";
const PLAN_PNG =
  process.env.DEMO_PLAN_PNG ??
  "C:/Users/Newtation/Desktop/datasets/cubicasa5k/cubicasa5k/high_quality_architectural/1191/F1_scaled.png";
const SOURCE_PLAN = process.env.DEMO_SOURCE_PLAN ?? "1191";

// 1280x800 keeps the viewer's two-column layout intact; anything narrower collapses the
// rooms table under the canvas and the demo stops showing the thing it is meant to show.
const SIZE = { width: 1280, height: 800 };

const started = Date.now();
const phases = [];
function mark(name) {
  const at = (Date.now() - started) / 1000;
  phases.push({ name, at });
  console.log(`  ${at.toFixed(1)}s  ${name}`);
}

/** Replay the cached vlm/hybrid rooms for this plan. No API calls. */
function seedCachedSources(planId) {
  // exec, not run: a second api container loading PaddleOCR against the shared model
  // volume left the running container's OCR crashing natively on every orientation, which
  // produced three silent recordings with zero tokens and no rules rooms.
  execFileSync(
    "docker",
    [
      "compose", "exec", "-T", "api",
      "python", "/docs/seed_demo_sources.py",
      "--plan-id", planId, "--source-plan", SOURCE_PLAN,
    ],
    { cwd: ROOT, stdio: "inherit" },
  );
}

const main = async () => {
  rmSync(OUT_DIR, { recursive: true, force: true });
  mkdirSync(OUT_DIR, { recursive: true });

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: SIZE,
    recordVideo: { dir: OUT_DIR, size: SIZE },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") console.log(`  [browser] ${m.text()}`);
  });
  page.on("pageerror", (e) => console.log(`  [pageerror] ${e.message}`));

  // Slow the cursor down so a viewer can follow it. Playwright moves instantly otherwise and
  // hovers read as teleporting highlights.
  const hover = async (locator) => {
    await locator.scrollIntoViewIfNeeded();
    const box = await locator.boundingBox();
    if (box) await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 24 });
  };

  mark("start");
  await page.goto(WEB, { waitUntil: "networkidle" });
  await page.waitForTimeout(1200);
  mark("upload-page");

  // Upload through the real file input rather than faking a drop event.
  const planIdPromise = page
    .waitForResponse((r) => r.url().endsWith("/plans") && r.request().method() === "POST")
    .then((r) => r.json())
    .then((body) => body.id);

  await page.setInputFiles('input[type="file"]', PLAN_PNG);
  const planId = await planIdPromise;
  mark("uploaded");
  console.log(`  plan ${planId}`);

  await page.waitForURL(/\/plans\//, { timeout: 300_000 });

  // Ask the API directly rather than inferring readiness from the DOM: the viewer renders a
  // spinner, an error or a table depending on status, and waiting on "table" alone turns any
  // other state into an unexplained timeout.
  const api = process.env.DEMO_API_URL ?? "http://localhost:8000";
  const deadline = Date.now() + 300_000;
  let status = "";
  while (Date.now() < deadline) {
    const body = await fetch(`${api}/plans/${planId}?include_tokens=false`).then((r) => r.json());
    status = body.status;
    if (status === "done" || status === "failed") break;
    await page.waitForTimeout(1000);
  }
  if (status !== "done") throw new Error(`plan ${planId} ended as ${status}`);

  // Seed only now that OCR has finished. Seeding earlier starves it: the seed runs in a
  // second api container, which loads PaddleOCR on import, and the two competing for memory
  // left the real OCR returning zero tokens on three consecutive recordings.
  seedCachedSources(planId);
  await page.reload({ waitUntil: "networkidle" });

  try {
    await page.waitForSelector("table", { timeout: 60_000 });
  } catch (error) {
    await page.screenshot({ path: resolve(HERE, "demo.failure.png"), fullPage: true });
    const text = (await page.locator("body").innerText()).slice(0, 600);
    console.log(`
  viewer never showed a table. url=${page.url()}
  body: ${text}`);
    throw error;
  }

  await page.waitForTimeout(1500);
  mark("viewer");

  // OCR overlay on.
  const ocrToggle = page.getByRole("checkbox");
  await hover(ocrToggle);
  await ocrToggle.check();
  await page.waitForTimeout(1800);
  mark("ocr-boxes-on");

  await ocrToggle.uncheck();
  await page.waitForTimeout(700);

  // Hover three rows so their boxes light up on the plan.
  const rows = page.locator("tbody tr");
  const n = Math.min(3, await rows.count());
  for (let i = 0; i < n; i += 1) {
    await hover(rows.nth(i));
    await page.waitForTimeout(900);
  }
  mark("hover-rooms");

  // Switch approaches. Built from the sources present in the data, so these exist only
  // because the cached results were seeded above.
  for (const label of ["VLM", "Hybrid", "Rules"]) {
    const button = page.getByRole("button", { name: label, exact: true });
    if ((await button.count()) === 0) continue;
    await hover(button.first());
    await button.first().click();
    await page.waitForTimeout(1100);
  }
  mark("approach-switch");

  // Export. Caught rather than followed, so the browser does not navigate away.
  const csv = page.getByRole("link", { name: /Export CSV/i });
  await hover(csv);
  const download = page.waitForEvent("download", { timeout: 15_000 }).catch(() => null);
  await csv.click();
  await download;
  await page.waitForTimeout(1400);
  mark("export");

  // The video is only finalized on context close, but saveAs needs the browser still open -
  // closing it first makes saveAs fail with "Target page, context or browser has been closed".
  const video = page.video();
  await context.close();

  const raw = resolve(HERE, "demo.raw.webm");
  if (video) await video.saveAs(raw);
  await browser.close();
  rmSync(OUT_DIR, { recursive: true, force: true });

  writeFileSync(
    resolve(HERE, "demo.phases.json"),
    JSON.stringify({ planId, phases }, null, 2),
    "utf-8",
  );
  console.log(`\nwrote ${raw}`);
  console.log("next: node docs/make_demo_gif.mjs");
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
