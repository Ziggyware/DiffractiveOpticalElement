// Renders side-by-side previews (SEM mask | replay field) of every scene from
// the CPU reference model, so the optics can be eyeballed without a browser.
//   node tools/preview.mjs [outdir]
import fs from 'node:fs';
import path from 'node:path';
import { encodePNG } from './png.mjs';
import {
  S_SCREEN, TAU, SCENES, designPhase, quantise, replay, sampleReplay,
  spectrumRGB, heightOf, semShade, gsSolve, gsTarget, holoPhaseAt, LAM_R, LAM_G, LAM_B,
  N_INDEX,
} from './doe.mjs';

const outDir = process.argv[2] || '/home/user/previews';
const PANEL = 256;

export function defaultParams(N = 512) {
  return {
    N, p: 600e-9, lamDesign: 532e-9,
    lamR: LAM_R, lamG: LAM_G, lamB: LAM_B,
    levels: 6,
    gratingPeriodPx: 8,
    zflF: 2e-3,
    axiconAlpha: 0.18,
    vortexL: 3,
    dammannSpacing: 16,
    dammannSpots: 5,
    dammannRot: 0,
    text: 'DOE',
    holoWindow: 0.7, holoOffX: 0.16, holoOffY: 0.10,
    holoEfficiency: 0.8, holoM: N / 2, holoField: null,
    w: [0, 0, 0, 0, 0, 0],
  };
}

export function buildPhase(prm) {
  const N = prm.N;
  const phi = new Float32Array(N * N);
  const raw = new Float32Array(prm.levels && prm.levels >= 2 ? N * N : 0);
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const v = quantise(designPhase(i, j, prm), prm.levels);
      phi[j * N + i] = v;
      if (raw.length) raw[j * N + i] = v;
    }
  prm.phaseFn = (i, j) => phi[j * N + i];
  return phi;
}

// --- SEM panel -------------------------------------------------------------
function wrapHalf(x) { return x - Math.round(x); }

export function renderSEM(prm, phi, zoom = 1, cx = 0.5, cy = 0.5) {
  const N = prm.N, img = new Uint8Array(PANEL * PANEL * 3);
  const span = 1 / zoom;                       // fraction of the mask in view
  const levels = prm.levels >= 2 ? prm.levels : 0;
  const sample = (u, v) => {
    // level index in [0,1) -> no 2*pi wrap artefacts
    const x = u * N - 0.5, y = v * N - 0.5;
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = x - x0, fy = y - y0;
    let acc = 0;
    for (let b = 0; b < 4; b++) {
      const xi = Math.min(N - 1, Math.max(0, x0 + (b & 1)));
      const yi = Math.min(N - 1, Math.max(0, y0 + (b >> 1)));
      const ph = phi[yi * N + xi];
      const lv = levels >= 2 ? Math.round((ph / TAU) * levels) % levels / levels
        : ph / TAU;
      acc += lv * ((b & 1) ? fx : 1 - fx) * ((b >> 1) ? fy : 1 - fy);
    }
    return acc;
  };
  const dPix = 1 / (N * zoom);                 // metres per screen pixel / (p*N)
  const hUnit = prm.lamDesign / (N_INDEX - 1); // metres per unit phase fraction
  for (let py = 0; py < PANEL; py++) {
    for (let px = 0; px < PANEL; px++) {
      const u = cx + (px + 0.5) / PANEL * span - span / 2;
      const v = cy + (py + 0.5) / PANEL * span - span / 2;
      const du = 1 / (N * zoom) * 0.5, dv = du;
      const h0 = sample(u, v);
      // wrap-aware central differences, in phase-fraction per mask pixel
      const gx = 0.5 * (wrapHalf(sample(u + du, v) - h0) + wrapHalf(h0 - sample(u - du, v))) / (du * prm.N);
      const gy = 0.5 * (wrapHalf(sample(u, v + dv) - h0) + wrapHalf(h0 - sample(u, v - dv))) / (dv * prm.N);
      // dimensionless physical slope: (fraction per pixel) * height unit / pitch
      const dzx = gx * hUnit / prm.p, dzy = gy * hUnit / prm.p;
      let s = semShade(dzx, dzy, (py * 7919 + px * 104729) | 0);
      const o = (py * PANEL + px) * 3;
      const g = Math.round(255 * Math.pow(s, 1 / 1.6));
      img[o] = g; img[o + 1] = g; img[o + 2] = g;
    }
  }
  return img;
}

// --- replay panel ----------------------------------------------------------
export function renderReplay(prm, I3, exposure = null) {
  const N = prm.N, img = new Uint8Array(PANEL * PANEL * 3);
  const cols = [spectrumRGB(prm.lamR * 1e9), spectrumRGB(prm.lamG * 1e9), spectrumRGB(prm.lamB * 1e9)];
  const lams = [prm.lamR, prm.lamG, prm.lamB];
  const SS = 2;                       // supersample each panel pixel
  const lin = new Float32Array(PANEL * PANEL * 3);
  let peak = 0;
  for (let py = 0; py < PANEL; py++) {
    for (let px = 0; px < PANEL; px++) {
      const acc = [0, 0, 0];
      let n = 0;
      for (let sy2 = 0; sy2 < SS; sy2++)
        for (let sx2 = 0; sx2 < SS; sx2++) {
          const sx = (2 * (px + (sx2 + 0.5) / SS) / PANEL - 1) * S_SCREEN;
          const sy = (2 * (py + (sy2 + 0.5) / SS) / PANEL - 1) * S_SCREEN;
          for (let c = 0; c < 3; c++) acc[c] += sampleReplay(I3[c], N, lams[c], prm.p, sx, sy);
          n++;
        }
      const o = (py * PANEL + px) * 3;
      let sum = 0;
      for (let c = 0; c < 3; c++) { const v = acc[c] / n; lin[o + c] = v; sum += v; }
      if (sum > peak) peak = sum;
    }
  }
  if (exposure === null) exposure = peak;
  // log tone mapping: the replay spans ~5 decades (zero order .. speckle)
  const DECADES = 3.5;
  for (let i = 0; i < PANEL * PANEL; i++) {
    const o = i * 3;
    let r = 0, g = 0, b = 0;
    for (let c = 0; c < 3; c++) {
      const d = Math.min(1, Math.max(0,
        Math.log10(1 + Math.max(0, lin[o + c]) / ((exposure || 1e-30) * 1e-3)) / DECADES));
      r += d * cols[c][0]; g += d * cols[c][1]; b += d * cols[c][2];
    }
    img[o] = Math.round(255 * Math.pow(Math.min(1, r), 1 / 2.2));
    img[o + 1] = Math.round(255 * Math.pow(Math.min(1, g), 1 / 2.2));
    img[o + 2] = Math.round(255 * Math.pow(Math.min(1, b), 1 / 2.2));
  }
  return { img, peak };
}

function sideBySide(a, b) {
  const w = PANEL * 2 + 4, h = PANEL;
  const out = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    out.set(a.subarray(y * PANEL * 3, (y + 1) * PANEL * 3), (y * w) * 3);
    for (let x = 0; x < 4; x++) {
      const o = (y * w + PANEL + x) * 3;
      out[o] = 30; out[o + 1] = 30; out[o + 2] = 34;
    }
    out.set(b.subarray(y * PANEL * 3, (y + 1) * PANEL * 3), (y * w + PANEL + 4) * 3);
  }
  return { out, w, h };
}

export function shot(prm, name, opts = {}) {
  const phi = buildPhase(prm);
  const I3 = replay(prm);
  const sem = renderSEM(prm, phi, opts.zoom || 1);
  const rep = renderReplay(prm, I3, opts.exposure);
  const { out, w, h } = sideBySide(sem, rep.img);
  fs.writeFileSync(path.join(outDir, name + '.png'), encodePNG(out, w, h));
  return { phi, I3, peak: rep.peak };
}

function main() {
  fs.mkdirSync(outDir, { recursive: true });
  const base = defaultParams(512);
  const scenes = [
    ['1-grating', { w: [1, 0, 0, 0, 0, 0] }],
    ['2-zfl-f2mm', { w: [0, 1, 0, 0, 0, 0] }],
    ['2-zfl-f6mm', { w: [0, 1, 0, 0, 0, 0], zflF: 6e-3 }],
    ['3-axicon', { w: [0, 0, 1, 0, 0, 0] }],
    ['4-vortex-l3', { w: [0, 0, 0, 1, 0, 0] }],
    ['4-vortex-l8', { w: [0, 0, 0, 1, 0, 0], vortexL: 8 }],
    ['5-dammann-25', { w: [0, 0, 0, 0, 1, 0] }],
    ['5-dammann-4x4', { w: [0, 0, 0, 0, 1, 0], dammannSpots: 4 }],
  ];
  for (const [name, over] of scenes) {
    const prm = { ...base, ...over, w: over.w || base.w };
    console.log('rendering', name);
    shot(prm, name);
  }
  // hologram: solve GS, then preview
  const prm = { ...base, w: [0, 0, 0, 0, 0, 1], text: 'DOE' };
  const M = prm.holoM;
  console.log('solving GS at M=' + M);
  const t0 = Date.now();
  const gs = gsSolve(prm, M, 12, null);
  console.log('  rms =', gs.rms.toFixed(4), 'in', Date.now() - t0, 'ms');
  prm.holoField = gs.field;
  shot(prm, '6-holo-DOE', { zoom: 2 });
  // binarised hologram: conjugate twin appears (real-valued mask)
  const prm2 = { ...prm, levels: 2 };
  shot(prm2, '6-holo-DOE-2level', { zoom: 2 });
  // off-design illumination: zero order blows out, channels slide apart
  const prm3 = { ...base, w: [1, 0, 0, 0, 0, 0], lamDesign: 700e-9 };
  shot(prm3, '1-grating-offdesign700');
  // large pitch: cutoff circles become visible
  const prm4 = { ...base, w: [1, 0, 0, 0, 0, 0], p: 1200e-9 };
  shot(prm4, '1-grating-p1200');
  console.log('done ->', outDir);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
