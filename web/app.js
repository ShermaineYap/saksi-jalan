/* Saksi Jalan: all inference runs in the browser with onnxruntime-web. Nothing is uploaded. */
"use strict";

const CLIP_URL = "models/clip_visual_w8.onnx";
const DETR_URL = "models/detr_q.onnx";
const PROBE_URL = "models/probe.json";
const THRESHOLD = 0.70;           // chosen from the measured refusal curve
const DET_MIN_SCORE = 0.7;        // detector confidence for a car
const DET_MIN_AREA = 0.01;        // ignore cars smaller than 1% of the photo
const COCO_VEHICLE = new Set([3, 8]); // car, truck
const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073], CLIP_STD = [0.26862954, 0.26130258, 0.27577711];
const IMNET_MEAN = [0.485, 0.456, 0.406], IMNET_STD = [0.229, 0.224, 0.225];
const SAMPLES = ["bezza_1", "myvi_4", "saga_7", "persona_3", "iriz_3", "axia_7", "bezza_2", "axia_3", "myvi_7"];
const COLOURS = {
  white: ["white", "putih"], black: ["black", "hitam"], grey: ["grey", "kelabu"], silver: ["silver", "perak"],
  red: ["red", "merah"], orange: ["orange", "jingga"], yellow: ["yellow", "kuning"], green: ["green", "hijau"],
  blue: ["blue", "biru"], brown: ["brown", "coklat"], purple: ["purple", "ungu"],
};

ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

const $ = (id) => document.getElementById(id);
const state = { img: null, cars: [], chosen: 0, verdict: null };
let models = null;

// ------------------------------------------------------------------ loading
async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Couldn't download ${url} (${res.status})`);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader(); const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    chunks.push(value); got += value.length; onBytes(got, total);
  }
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  return buf;
}

async function loadModels() {
  if (models) return models;
  showProgress(true);
  const sizes = { clip: 0, detr: 0 }, totals = { clip: 0, detr: 0 };
  const tick = () => {
    const g = sizes.clip + sizes.detr, t = totals.clip + totals.detr;
    setStatus(`Downloading the AI models to your device (first time only)… ${(g / 1e6).toFixed(0)}${t ? " of " + (t / 1e6).toFixed(0) : ""} MB`);
    if (t) $("bar").style.width = `${(100 * g / t).toFixed(1)}%`;
  };
  const [clipBuf, detrBuf, probe] = await Promise.all([
    fetchWithProgress(CLIP_URL, (g, t) => { sizes.clip = g; totals.clip = t; tick(); }),
    fetchWithProgress(DETR_URL, (g, t) => { sizes.detr = g; totals.detr = t; tick(); }),
    fetch(PROBE_URL).then((r) => r.json()),
  ]);
  setStatus("Starting the models…");
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const [clip, detr] = await Promise.all([ort.InferenceSession.create(clipBuf, opts), ort.InferenceSession.create(detrBuf, opts)]);
  showProgress(false);
  models = { clip, detr, probe };
  return models;
}

// ------------------------------------------------------------------ image helpers
function toCanvas(img, w, h, sx = 0, sy = 0, sw = img.naturalWidth || img.width, sh = img.naturalHeight || img.height) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(img, sx, sy, sw, sh, 0, 0, w, h);
  return c;
}

function chw(canvas, mean, std) {
  const { width: w, height: h } = canvas;
  const d = canvas.getContext("2d").getImageData(0, 0, w, h).data;
  const out = new Float32Array(3 * w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4)
    for (let c = 0; c < 3; c++) out[c * w * h + i] = (d[p + c] / 255 - mean[c]) / std[c];
  return out;
}

const softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };

// ------------------------------------------------------------------ detection
async function findCars(img) {
  const W = img.naturalWidth, H = img.naturalHeight;
  const s = Math.min(512 / Math.min(W, H), 900 / Math.max(W, H));
  const w = Math.round(W * s), h = Math.round(H * s);
  const x = new ort.Tensor("float32", chw(toCanvas(img, w, h), IMNET_MEAN, IMNET_STD), [1, 3, h, w]);
  const mask = new ort.Tensor("int64", new BigInt64Array(64 * 64).fill(1n), [1, 64, 64]);
  const out = await models.detr.run({ pixel_values: x, pixel_mask: mask });
  const L = out.logits.data, B = out.pred_boxes.data, Q = out.logits.dims[1], K = out.logits.dims[2];
  const cars = [];
  for (let q = 0; q < Q; q++) {
    const p = softmax(Array.from(L.slice(q * K, q * K + K)));
    let best = 0, cls = -1; for (let k = 0; k < K - 1; k++) if (p[k] > best) { best = p[k]; cls = k; }
    if (!COCO_VEHICLE.has(cls) || best < DET_MIN_SCORE) continue;
    const [cx, cy, bw, bh] = B.slice(q * 4, q * 4 + 4);
    const box = [Math.max(0, (cx - bw / 2) * W), Math.max(0, (cy - bh / 2) * H), Math.min(W, (cx + bw / 2) * W), Math.min(H, (cy + bh / 2) * H)];
    if ((box[2] - box[0]) * (box[3] - box[1]) < DET_MIN_AREA * W * H) continue;
    cars.push({ box, score: best });
  }
  return cars.sort((a, b) => area(b.box) - area(a.box));
}
const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);

// ------------------------------------------------------------------ recognition
function cropCanvas(img, box, pad = 0.05) {
  const W = img.naturalWidth, H = img.naturalHeight;
  const dx = (box[2] - box[0]) * pad, dy = (box[3] - box[1]) * pad;
  const x0 = Math.max(0, box[0] - dx), y0 = Math.max(0, box[1] - dy), x1 = Math.min(W, box[2] + dx), y1 = Math.min(H, box[3] + dy);
  return toCanvas(img, Math.round(x1 - x0), Math.round(y1 - y0), x0, y0, x1 - x0, y1 - y0);
}

async function classify(crop) {
  // CLIP preprocessing: shortest side to 224, centre crop 224x224
  const s = 224 / Math.min(crop.width, crop.height);
  const rw = Math.max(224, Math.round(crop.width * s)), rh = Math.max(224, Math.round(crop.height * s));
  const resized = toCanvas(crop, rw, rh);
  const c224 = toCanvas(resized, 224, 224, Math.floor((rw - 224) / 2), Math.floor((rh - 224) / 2), 224, 224);
  const x = new ort.Tensor("float32", chw(c224, CLIP_MEAN, CLIP_STD), [1, 3, 224, 224]);
  const e = (await models.clip.run({ pixels: x })).embedding.data;
  const { W, b, classes } = models.probe;
  const z = W.map((row, i) => row.reduce((acc, w, j) => acc + w * e[j], b[i]));
  const p = softmax(z);
  return classes.map((c, i) => [c, p[i]]).sort((a, b) => b[1] - a[1]);
}

// Colour: same rules as app/saksi_core.py
function rgbToHls(r, g, b) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  if (mx === mn) return [0, l, 0];
  const d = mx - mn, s = l <= 0.5 ? d / (mx + mn) : d / (2 - mx - mn);
  let h;
  if (mx === r) h = ((g - b) / d) % 6; else if (mx === g) h = (b - r) / d + 2; else h = (r - g) / d + 4;
  h /= 6; if (h < 0) h += 1;
  return [h, l, s];
}
function nameColour(R, G, B) {
  const [h, l, s] = rgbToHls(R / 255, G / 255, B / 255);
  if (l > 0.84 || (l > 0.78 && s < 0.18)) return "white";
  if (l < 0.16) return "black";
  if (s < 0.18) return l > 0.5 ? "silver" : "grey";
  const deg = h * 360;
  if (deg < 15 || deg >= 340) return l < 0.30 ? "brown" : "red";
  if (deg < 40) return l < 0.40 ? "brown" : "orange";
  if (deg < 70) return "yellow";
  if (deg < 170) return "green";
  if (deg < 255) return "blue";
  return deg < 300 ? "purple" : "red";
}
function dominantColour(crop) {
  const c = toCanvas(crop, 96, 96); const d = c.getContext("2d").getImageData(12, 38, 72, 40).data;
  const counts = {};
  for (let i = 0; i < d.length / 4; i += 3) { const n = nameColour(d[4 * i], d[4 * i + 1], d[4 * i + 2]); counts[n] = (counts[n] || 0) + 1; }
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

// ------------------------------------------------------------------ statement (same wording as saksi_core.py)
function statement(v, when, where, plate, dir, extra) {
  when = when.trim() || "[time not given]"; where = where.trim() || "[place not given]"; plate = plate.trim().toUpperCase();
  let carEn, carMs;
  if (!v) { carEn = "a vehicle I could not identify"; carMs = "sebuah kenderaan yang tidak dapat saya kenal pasti"; }
  else if (v.confident) { carEn = `a ${v.colourEn} ${v.best}`; carMs = `sebuah ${v.best} berwarna ${v.colourMs}`; }
  else { const alts = v.top3.map((t) => t[0]).join(", "); carEn = `a ${v.colourEn} car, possibly one of: ${alts}`; carMs = `sebuah kereta berwarna ${v.colourMs}, mungkin salah satu daripada: ${alts}`; }
  const en = [`On ${when}, at ${where}, I saw ${carEn}.`], ms = [`Pada ${when}, di ${where}, saya telah melihat ${carMs}.`];
  if (plate) { en.push(`The registration number I could read was: ${plate}.`); ms.push(`Nombor pendaftaran yang dapat saya baca ialah: ${plate}.`); }
  else { en.push("I could not read the registration number."); ms.push("Saya tidak dapat membaca nombor pendaftaran."); }
  if (dir.trim()) { en.push(`The vehicle left heading ${dir.trim()}.`); ms.push(`Kenderaan itu beredar menuju ke arah ${dir.trim()}.`); }
  if (extra.trim()) { en.push(extra.trim()); ms.push(extra.trim()); }
  return [
    en.join(" ") + "\n\nThe vehicle description was suggested by Saksi Jalan, an AI tool, from a photo. It is an aid to my statement, not proof.",
    ms.join(" ") + "\n\nKeterangan kenderaan ini dicadangkan oleh Saksi Jalan, sebuah alat AI, berdasarkan gambar. Ia hanya membantu keterangan saya, bukan bukti.",
  ];
}

// ------------------------------------------------------------------ UI
function setStatus(t) { $("status").textContent = t; }
function showProgress(on) { $("progress").hidden = !on; }
function activate(id) { document.querySelectorAll(".step").forEach((s) => s.classList.toggle("active", s.id === id)); }

function draw() {
  const img = state.img, c = $("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext("2d"); g.drawImage(img, 0, 0);
  const lw = Math.max(3, img.naturalWidth / 220);
  g.font = `800 ${Math.max(18, img.naturalWidth / 28)}px Overpass, sans-serif`;
  state.cars.forEach((car, i) => {
    const [x0, y0, x1, y1] = car.box, on = i === state.chosen;
    g.lineWidth = on ? lw * 2 : lw; g.strokeStyle = on ? "#f2c230" : "#ffffff";
    g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const label = String(i + 1), pad = lw * 2, tw = g.measureText(label).width;
    g.fillStyle = on ? "#f2c230" : "#1d4f9e"; g.fillRect(x0, y0, tw + pad * 2, parseInt(g.font.split(" ")[1]) + pad);
    g.fillStyle = on ? "#2f3237" : "#ffffff"; g.fillText(label, x0 + pad, y0 + parseInt(g.font.split(" ")[1]));
  });
}

function renderVerdict(v) {
  const el = $("verdict"); el.hidden = false;
  if (!v) {
    el.className = "verdict none";
    el.innerHTML = `<h3>No car found</h3><p class="why">I couldn't find a car in this photo, so I won't guess. Try a photo where the car is bigger or clearer.</p>`;
    return;
  }
  const pct = (p) => `${Math.round(p * 100)}%`;
  const rows = v.top3.map(([n, p]) => `<li><span>${n}</span><span class="track"><span class="fill" style="width:${(p * 100).toFixed(1)}%"></span></span><span class="pct">${pct(p)}</span></li>`).join("");
  el.className = `verdict ${v.confident ? "sure" : "unsure"}`;
  el.innerHTML = v.confident
    ? `<h3>Most likely a ${v.colourEn} ${v.best}</h3><p class="ms" lang="ms">${v.best} berwarna ${v.colourMs}</p><ul class="bars" aria-label="Top 3 guesses">${rows}</ul>`
    : `<h3>Not sure which model</h3><p class="why">It's less than 70% sure, so it won't name one model. The colour looks <strong>${v.colourEn}</strong> <span lang="ms">(${v.colourMs})</span>. Its top 3 guesses:</p><ul class="bars" aria-label="Top 3 guesses">${rows}</ul>`;
  for (const n of v.notes) el.insertAdjacentHTML("beforeend", `<p class="note">${n}</p>`);
}

async function analyseChosen() {
  const car = state.cars[state.chosen];
  const crop = cropCanvas(state.img, car.box);
  const top = await classify(crop);
  const colour = dominantColour(crop);
  const v = { top3: top.slice(0, 3), best: top[0][0], confident: top[0][1] >= THRESHOLD, colourEn: COLOURS[colour][0], colourMs: COLOURS[colour][1], notes: [] };
  if (crop.width < 120 || crop.height < 80) v.notes.push("The car is very small in this photo, so this answer is less reliable.");
  state.verdict = v; draw(); renderVerdict(v);
  $("pick-hint").hidden = state.cars.length < 2;
  setStatus(state.cars.length > 1 ? `Found ${state.cars.length} cars. Describing car ${state.chosen + 1}, the biggest unless you pick another.` : "Found 1 car.");
}

async function analyse(img) {
  state.img = img; state.cars = []; state.chosen = 0; state.verdict = null;
  $("step-car").hidden = false; $("verdict").hidden = true; $("pick-hint").hidden = true; activate("step-car");
  $("step-car").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  const c = $("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight; c.getContext("2d").drawImage(img, 0, 0);
  try {
    await loadModels();
    setStatus("Looking for cars…");
    const t0 = performance.now();
    state.cars = await findCars(img);
    if (!state.cars.length) { setStatus(""); renderVerdict(null); }
    else await analyseChosen();
    window.saksiLastMs = performance.now() - t0;
    $("step-statement").hidden = false;
    if (!$("f-when").value) { const d = new Date(); $("f-when").value = `${d.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" })}, ${d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`; }
  } catch (err) {
    showProgress(false);
    setStatus(`Something went wrong: ${err.message}. Reload the page and try again.`);
    console.error(err);
  }
}

function loadImage(src) {
  return new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => fail(new Error("That file isn't an image this browser can open.")); i.src = src; });
}
async function useFile(file) {
  if (!file || !file.type.startsWith("image/")) { setStatus("Please choose an image file."); $("step-car").hidden = false; return; }
  const url = URL.createObjectURL(file);
  try { await analyse(await loadImage(url)); } catch (e) { $("step-car").hidden = false; setStatus(e.message); }
}

// wiring
$("file").addEventListener("change", (e) => useFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => useFile(e.dataTransfer.files[0]));

for (const s of SAMPLES) {
  const b = document.createElement("button"); b.type = "button"; b.setAttribute("aria-label", `Try sample photo ${s.replace("_", " ")}`);
  b.innerHTML = `<img src="examples/${s}.jpg" alt="" loading="lazy">`;
  b.addEventListener("click", async () => analyse(await loadImage(`examples/${s}.jpg`)));
  $("samples").appendChild(b);
}

$("canvas").addEventListener("click", async (e) => {
  if (state.cars.length < 2) return;
  const r = e.target.getBoundingClientRect(), sx = state.img.naturalWidth / r.width;
  const x = (e.clientX - r.left) * sx, y = (e.clientY - r.top) * sx;
  const hit = state.cars.map((c, i) => [c, i]).filter(([c]) => x >= c.box[0] && x <= c.box[2] && y >= c.box[1] && y <= c.box[3]).sort((a, b) => area(a[0].box) - area(b[0].box))[0];
  if (hit && hit[1] !== state.chosen) { state.chosen = hit[1]; await analyseChosen(); }
});

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const [en, ms] = statement(state.verdict, $("f-when").value, $("f-where").value, $("f-plate").value, $("f-dir").value, $("f-extra").value);
  $("en").textContent = en; $("ms").textContent = ms; $("out").hidden = false; activate("step-statement");
});
document.querySelectorAll(".copy").forEach((b) => b.addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($(b.dataset.target).textContent); const t = b.textContent; b.textContent = b.dataset.target === "ms" ? "Disalin" : "Copied"; setTimeout(() => (b.textContent = t), 1500); }
  catch { b.textContent = "Select the text to copy"; }
}));
activate("step-photo");
window.saksi = { state, statement, nameColour };
