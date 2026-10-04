/**
 * app.js — the browser workbench.
 *
 * It imports the *same* modules the Node tests and CLI use (`/src/…` is served
 * straight off disk), so what you see here is the library under test, not a
 * port of it. Every number on the page is computed in this tab from the complex
 * field: design the element, propagate it to the screen, measure the result.
 */

import { ProjectorSystem, makeTarget } from '../src/projector.js';
import { designDOE } from '../src/doe.js';
import { planeToRGB, planeToGray } from '../src/color.js';
import { encodePNG } from '../src/png.js';
import { summarizeMetrics, rmsError } from '../src/metrics.js';
import { ComplexField } from '../src/field.js';

const $ = (id) => document.getElementById(id);
const num = (id) => Number($(id).value);
const el = {
  mode: $('mode'),
  algorithm: $('algorithm'),
  target: $('target'),
  text: $('text'),
  textRow: $('textRow'),
  n: $('n'),
  pitch: $('pitch'),
  lambda: $('lambda'),
  distance: $('distance'),
  iterations: $('iterations'),
  levels: $('levels'),
  source: $('source'),
  run: $('run'),
  stop: $('stop'),
  progress: $('progress'),
  status: $('status'),
  dz: $('dz'),
  dzLabel: $('dzLabel'),
  metrics: $('metrics'),
  geometry: $('geometry'),
  log: $('log'),
  targetInfo: $('targetInfo'),
  screenScale: $('screenScale'),
  stackInfo: $('stackInfo'),
};

let system = null;
let design = null;
let running = false;
let cancelRequested = false;
let lastProjection = null;

function log(msg) {
  const line = `${new Date().toLocaleTimeString()}  ${msg}`;
  el.log.textContent = `${line}\n${el.log.textContent}`.slice(0, 4000);
}

function readConfig() {
  const targetKind = el.target.value;
  const target = { kind: targetKind };
  if (targetKind === 'text') target.text = el.text.value.toUpperCase() || 'DOE';
  const cfg = {
    n: Math.max(64, Math.min(512, Math.round(num('n') / 64) * 64)),
    pitch: num('pitch') * 1e-6,
    lambda: num('lambda') * 1e-9,
    distance: num('distance') * 1e-3,
    mode: el.mode.value,
    algorithm: el.algorithm.value,
    iterations: Math.max(1, Math.round(num('iterations'))),
    levels: Math.max(0, Math.round(num('levels'))),
    source: el.source.value,
    seed: 12345,
    aperture: 0.95,
    apodization: 2,
    target,
  };
  return cfg;
}

/* ------------------------------------------------------------------ *
 * canvas helpers
 * ------------------------------------------------------------------ */

function drawPlane(canvas, plane, opts = {}) {
  const n = opts.n ?? plane.length;
  const size = Math.round(Math.sqrt(plane.length));
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const rgb = planeToRGB(plane, { width: size, height: size, ...opts });
  const img = ctx.createImageData(size, size);
  img.data.set(rgb);
  for (let k = 3; k < img.data.length; k += 4) img.data[k] = 255;
  ctx.putImageData(img, 0, 0);
}

function drawStrip(canvas, frames, { cols = frames.length, map = 'inferno' } = {}) {
  const size = Math.round(Math.sqrt(frames[0].length));
  const gap = 3;
  const W = size * cols + gap * (cols - 1);
  const merged = new Float64Array(W * size);
  let mx = 0;
  for (const f of frames) for (const v of f) mx = Math.max(mx, v);
  frames.forEach((f, idx) => {
    for (let j = 0; j < size; j++) {
      for (let i = 0; i < size; i++) {
        merged[i + idx * (size + gap) + j * W] = mx > 0 ? f[i + j * size] / mx : 0;
      }
    }
  });
  canvas.width = W;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const rgb = planeToRGB(merged, { map, min: 0, max: 1 });
  const img = ctx.createImageData(W, size);
  img.data.set(rgb);
  for (let k = 3; k < img.data.length; k += 4) img.data[k] = 255;
  ctx.putImageData(img, 0, 0);
}

function metricCard(label, value, sub = '', cls = '') {
  return `<div class="metric ${cls}"><b>${value}</b><span>${label}</span>${sub ? `<span>${sub}</span>` : ''}</div>`;
}

function renderMetrics(m) {
  const pct = (v) => `${(v * 100).toFixed(2)} %`;
  const good = (ok) => (ok ? 'good' : '');
  el.metrics.innerHTML = [
    metricCard('RMSE vs target', pct(m.rmse), `${(20 * Math.log10(1 / Math.max(m.rmse, 1e-9))).toFixed(1)} dB`, good(m.rmse < 0.1)),
    metricCard('correlation', m.correlation.toFixed(4), 'Pearson, in the ROI', good(m.correlation > 0.95)),
    metricCard('SSIM', m.ssim.toFixed(4), 'structural similarity', good(m.ssim > 0.8)),
    metricCard('efficiency', pct(m.efficiency), 'power in the image / incident', good(m.efficiency > 0.5)),
    // Uniformity means different things for different targets: for a spot array
    // it is the standard min/max over the *spots*; for an image target the
    // useful figure is flatness (1 - std/mean) over the lit part of the image,
    // because a dim glyph edge is not a defect.
    m.spotUniformity !== null && m.spotUniformity !== undefined
      ? metricCard('spot uniformity', pct(m.spotUniformity), '2·min/(max+min) over the spots', good(m.spotUniformity > 0.8))
      : metricCard('flatness', pct(m.flatness), '1 − std/mean over lit pixels', good(m.flatness > 0.6)),
    metricCard('zero-order leak', pct(m.zeroOrder), 'undiffracted power at the element', good(m.zeroOrder < 0.02)),
    metricCard('peak SNR', `${m.snr.toFixed(2)} dB`, 'relative to the target power', good(m.snr > 20)),
    metricCard('spot radius', `${(m.rmsRadius * 1e6).toFixed(1)} µm`, 'rms, second moment'),
  ];
}

function renderGeometry(g) {
  el.geometry.innerHTML = [
    metricCard('aperture', `${g.apertureMm.toFixed(3)} mm`, `${g.gridN}² pixels at ${g.pitchUm.toFixed(1)} µm`),
    metricCard('screen window', `${g.screenWindowMm.toFixed(3)} mm`, g.mode === 'far-field' ? 'λz/dx (grows with z)' : 'N·dx (fixed)'),
    metricCard('screen pitch', `${g.screenPitchUm.toFixed(3)} µm`, 'sampling at the screen'),
    metricCard('diffraction spot', `${g.diffractionSpotUm.toFixed(2)} µm`, '1.22 λ z / D'),
    metricCard('resolvable spots', `${Math.round(g.resolvableSpotsPerAxis)} / axis`, `${g.addressablePixels.toLocaleString()} addressable`),
    metricCard('max deflection', `${g.maxDeflectionDeg.toFixed(2)}°`, 'λ/2p grating limit'),
    metricCard('f-number', `${g.fNumber.toFixed(2)}`, `throw ratio ${g.throwRatio.toFixed(2)}`),
    metricCard(
      'Fresnel limit',
      `${g.fresnelCriticalMm.toFixed(1)} mm`,
      g.undersampledChirp ? 'chirp undersampled at z' : 'chirp resolved at z',
    ),
  ];
}

/* ------------------------------------------------------------------ *
 * the pipeline
 * ------------------------------------------------------------------ */

function makeTargetFor(cfg) {
  const n = cfg.n;
  return makeTarget(cfg.target.kind, n, n, cfg.target);
}

function showTarget(cfg) {
  const t = makeTargetFor(cfg);
  drawPlane($('cTarget'), t.irradiance, { map: 'viridis' });
  el.targetInfo.textContent = t.name;
  return t;
}

async function run() {
  if (running) return;
  const cfg = readConfig();
  // keep the inputs canonical with what the library will actually use
  el.n.value = cfg.n;
  const t0 = performance.now();
  system = new ProjectorSystem(cfg);
  showTarget(cfg);
  renderGeometry(system.geometry());
  const total = cfg.iterations;
  running = true;
  cancelRequested = false;
  el.run.disabled = true;
  el.stop.disabled = false;
  el.status.textContent = 'designing…';

  const batches = Math.max(1, Math.ceil(total / 5));
  const per = Math.ceil(total / batches);
  let done = 0;
  let result = null;
  // iterate in small batches so the page stays responsive and the progress bar
  // actually moves (the whole design is plain JS in this tab)
  for (let b = 0; b < batches; b++) {
    const it = Math.min(per, total - done);
    if (it <= 0) break;
    // design() is deterministic and always restarts from the same seeded phase,
    // so asking for `done + it` iterations is exactly "continue where we left
    // off" while keeping the page responsive between batches.
    result = system.design({ iterations: done + it });
    design = result;
    done += it;
    el.progress.style.width = `${(100 * done) / total}%`;
    el.status.textContent = `designing… ${done}/${total} iterations (rmse ${(result.error[result.error.length - 1] * 100).toFixed(1)} %)`;
    await new Promise((r) => setTimeout(r, 0));
    if (cancelRequested) break;
  }
  el.progress.style.width = '100%';
  if (!design) {
    running = false;
    el.run.disabled = false;
    el.stop.disabled = true;
    return;
  }
  drawPlane($('cPhase'), design.phase, { map: 'phase', min: -Math.PI, max: Math.PI });
  log(
    `designed ${cfg.n}² ${cfg.mode} element, ${done} iterations of ${design.algorithm}, ` +
      `error ${(design.error[0] * 100).toFixed(1)} % → ${(design.error[design.error.length - 1] * 100).toFixed(1)} % ` +
      `in ${((performance.now() - t0) / 1000).toFixed(1)} s`,
  );
  await project(false);
  await drawFocusStack();
  running = false;
  el.run.disabled = false;
  el.stop.disabled = true;
  el.status.textContent = `ready — ${((performance.now() - t0) / 1000).toFixed(1)} s total`;
}

/** Simulate at the current slider position and refresh the screen panels. */
async function project(live = true) {
  if (!system || !design) return;
  const cfg = system.cfg;
  const frac = num('dz') / 100;
  const z = system.z * frac;
  el.dzLabel.textContent = `${(z * 1e3).toFixed(1)} mm${frac === 1 ? ' (design distance)' : ''}`;
  if (system.cfg.mode === 'far-field' && !(z > 0)) return;
  const field = system.simulate({ z });
  const m = system.screenMetrics(field);
  lastProjection = { field, metrics: m, z };
  drawPlane($('cScreen'), field.intensity(), { map: 'inferno', percentile: 0.999 });
  // error map: |projected - target| on the same normalisation
  const irr = field.intensity();
  const tgt = Float64Array.from(system.target.irradiance);
  const dx = field.dx;
  const dy = field.dy;
  let pt = 0;
  let tt = 0;
  for (let k = 0; k < irr.length; k++) {
    pt += irr[k] * system.target.roi[k];
    tt += tgt[k];
  }
  const s = tt > 0 ? pt / tt : 0;
  for (let k = 0; k < tgt.length; k++) tgt[k] *= s;
  const err = new Float64Array(irr.length);
  for (let k = 0; k < err.length; k++) err[k] = Math.abs(irr[k] - tgt[k]);
  drawPlane($('cError'), err, { map: 'turbo', percentile: 0.999 });
  renderMetrics(m);
  el.screenScale.textContent =
    `window ${(field.nx * field.dx * 1e3).toFixed(2)} mm · ${(field.dx * 1e6).toFixed(2)} µm/px`;
  if (!live) log(`projected at ${(z * 1e3).toFixed(1)} mm: ${summarizeMetrics(m).replace(/\n/g, ' | ')}`);
}

/** Focus stack along the axis — the picture of what "true projection" means. */
async function drawFocusStack() {
  if (!system) return;
  const frames = [];
  const labels = [];
  const cols = 7;
  for (let i = 0; i < cols; i++) {
    const frac = 0.55 + (0.9 * i) / (cols - 1);
    const z = system.z * frac;
    const field = system.simulate({ z, pad: 1 });
    frames.push(field.intensity());
    labels.push(`${(z * 1e3).toFixed(0)}mm`);
  }
  drawStrip($('cStack'), frames);
  el.stackInfo.textContent = labels.join(' · ');
  await new Promise((r) => setTimeout(r, 0));
}

/* ------------------------------------------------------------------ *
 * wiring
 * ------------------------------------------------------------------ */

el.textRow.style.display = el.target.value === 'text' ? 'block' : 'none';
el.target.addEventListener('change', () => {
  el.textRow.style.display = el.target.value === 'text' ? 'block' : 'none';
  if (system) showTarget(readConfig());
});
el.mode.addEventListener('change', () => {
  const far = el.mode.value === 'far-field';
  el.distance.disabled = false;
  el.status.textContent = far
    ? 'far field: the element is a Fourier CGH — the screen shows the lens focal-plane transform'
    : 'image plane: a real image forms at z';
});
el.run.addEventListener('click', () => run().catch((e) => log(`error: ${e.message}`)));
el.stop.addEventListener('click', () => {
  cancelRequested = true;
  log('stopping after the current batch…');
});
el.dz.addEventListener('input', () => project(true).catch((e) => log(`error: ${e.message}`)));

document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) el.run.click();
});

// export the current projection as a PNG (encoded in the page, no server round trip)
window.exportPNG = async () => {
  if (!lastProjection) return;
  const { field } = lastProjection;
  const gray = planeToGray(field.intensity(), { norm: true });
  const png = encodePNG({ width: field.nx, height: field.ny, gray });
  const url = URL.createObjectURL(new Blob([png], { type: 'image/png' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'projection.png';
  a.click();
  URL.revokeObjectURL(url);
};

renderGeometry(new ProjectorSystem(readConfig()).geometry());
showTarget(readConfig());
log('workbench loaded — the library modules came from /src, the same files the test suite runs');
log('press “Design & project” (or ctrl/cmd-enter) to run the pipeline');
run().catch((e) => log(`error: ${e.message}`));
