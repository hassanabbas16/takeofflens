/**
 * Render docs/demo.raw.webm into docs/demo.gif.
 *
 *   node docs/make_demo_gif.mjs
 *
 * Two things matter here.
 *
 * **The processing wait is compressed, not cut.** OCR takes tens of seconds on CPU across
 * four orientations, which cannot fit in a 20-second GIF. Rather than cutting to the finished
 * viewer and implying the tool is instant, the wait is kept and sped up, so the spinner and
 * the status text are still visible. Every other phase plays at real speed. The phase
 * boundaries come from docs/demo.phases.json, written by the recording script.
 *
 * **Two-pass palette.** palettegen builds a palette from the actual frames instead of a fixed
 * 256-colour table; a single-pass encode bands badly on UI screenshots.
 */

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, statSync, unlinkSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const RAW = resolve(HERE, "demo.raw.webm");
const PHASES = resolve(HERE, "demo.phases.json");
const PALETTE = resolve(HERE, "demo.palette.png");
const TRIMMED = resolve(HERE, "demo.trimmed.mp4");
const GIF = resolve(HERE, "demo.gif");

const FPS = 12;
const WIDTH = 900;
const WAIT_SPEEDUP = 45; // the OCR wait only
const MAX_SECONDS = 20;
const MAX_BYTES = 8 * 1024 * 1024;

const ffmpeg = (args) => execFileSync("ffmpeg", ["-y", "-hide_banner", "-loglevel", "error", ...args], { stdio: "inherit" });
const mb = (p) => (statSync(p).size / 1024 / 1024).toFixed(2);

if (!existsSync(RAW)) {
  console.error(`missing ${RAW} - run: node docs/record_demo.mjs`);
  process.exit(1);
}

const { phases } = JSON.parse(readFileSync(PHASES, "utf-8"));
const at = (name) => phases.find((p) => p.name === name)?.at;

const uploaded = at("uploaded");
const viewer = at("viewer");
const end = phases[phases.length - 1].at + 0.5;

// Keep a couple of seconds of the real wait at normal speed so it reads as a wait, then
// compress the rest of it.
const waitStart = uploaded + 0.8;
const waitEnd = viewer - 0.4;

const finalSeconds = waitStart + (waitEnd - waitStart) / WAIT_SPEEDUP + (end - waitEnd);
console.log(`raw ${mb(RAW)} MB, ${end.toFixed(1)}s -> ${finalSeconds.toFixed(1)}s rendered`);
console.log(`  wait ${waitStart.toFixed(1)}s -> ${waitEnd.toFixed(1)}s compressed ${WAIT_SPEEDUP}x`);

// Three segments: before the wait at 1x, the wait sped up, everything after at 1x.
const filter = [
  `[0:v]trim=0:${waitStart},setpts=PTS-STARTPTS[a]`,
  `[0:v]trim=${waitStart}:${waitEnd},setpts=(PTS-STARTPTS)/${WAIT_SPEEDUP}[b]`,
  `[0:v]trim=${waitEnd}:${end},setpts=PTS-STARTPTS[c]`,
  `[a][b][c]concat=n=3:v=1:a=0[v]`,
].join(";");

ffmpeg(["-i", RAW, "-filter_complex", filter, "-map", "[v]", "-an", "-r", String(FPS), TRIMMED]);
console.log(`trimmed ${mb(TRIMMED)} MB`);

const scale = `fps=${FPS},scale=${WIDTH}:-1:flags=lanczos`;
ffmpeg(["-i", TRIMMED, "-vf", `${scale},palettegen=max_colors=128`, PALETTE]);
ffmpeg([
  "-i", TRIMMED, "-i", PALETTE,
  "-lavfi", `${scale} [x]; [x][1:v] paletteuse=dither=sierra2_4a`,
  "-loop", "0", GIF,
]);
console.log(`gif ${mb(GIF)} MB`);

// gifsicle squeezes a UI capture hard; keep whichever file is smaller.
try {
  const { default: gifsicle } = await import("gifsicle");
  const optimized = resolve(HERE, "demo.opt.gif");
  execFileSync(gifsicle, ["-O3", "--lossy=40", "--colors", "128", GIF, "-o", optimized], { stdio: "inherit" });
  if (statSync(optimized).size < statSync(GIF).size) {
    unlinkSync(GIF);
    execFileSync("cmd", ["/c", "move", "/y", optimized, GIF], { stdio: "ignore" });
    console.log(`gifsicle -> ${mb(GIF)} MB`);
  } else {
    unlinkSync(optimized);
    console.log("gifsicle output was larger; kept the ffmpeg gif");
  }
} catch {
  console.log("gifsicle not available; kept the ffmpeg gif");
}

for (const tmp of [PALETTE, TRIMMED]) if (existsSync(tmp)) unlinkSync(tmp);

const bytes = statSync(GIF).size;
console.log(`\n${GIF}  ${mb(GIF)} MB`);
if (bytes > MAX_BYTES) {
  console.error(`OVER BUDGET (${MAX_BYTES / 1024 / 1024} MB). Drop FPS, then WIDTH, then raise --lossy.`);
  process.exit(1);
}
console.log(`within the 8 MB budget, ${finalSeconds.toFixed(1)}s long`);
if (finalSeconds > MAX_SECONDS) {
  console.error(`OVER ${MAX_SECONDS}s. Raise WAIT_SPEEDUP or tighten the dwells in record_demo.mjs.`);
  process.exit(1);
}
