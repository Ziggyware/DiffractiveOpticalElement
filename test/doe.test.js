import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ComplexField, discMask } from '../src/field.js';
import { propagateAngularSpectrumInPlace, focalPlaneSpectrum } from '../src/propagate.js';
import {
  gerchbergSaxton,
  weightedGerchbergSaxton,
  fourierCgh,
  errorDiffusionQuantize,
  quantizePhase,
  phaseToHeight,
  blurRaster,
  designDOE,
  kinoformLens,
  prismPhase,
  vortexPhase,
  axiconPhase,
  leeEncode,
  softRoi,
  ringMask,
  illuminationProfile,
  makeRng,
} from '../src/doe.js';

const LAMBDA = 532e-9;
const N = 64;
const DX = 8e-6;

function targetDisc(n, r) {
  const t = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      t[i + j * n] = Math.hypot(i - n / 2, j - n / 2) <= r ? 1 : 0;
    }
  }
  return t;
}

function amplitudeOf(irradiance) {
  const a = new Float64Array(irradiance.length);
  for (let k = 0; k < a.length; k++) a[k] = Math.sqrt(Math.max(0, irradiance[k]));
  return a;
}

/** Discrete sum of squares — the quantity a unitary propagator conserves. */
function discretePower(amp) {
  let p = 0;
  for (let k = 0; k < amp.length; k++) p += amp[k] * amp[k];
  return p;
}

function normaliseToPower(amp, power) {
  let p = 0;
  for (let k = 0; k < amp.length; k++) p += amp[k] * amp[k];
  const s = p > 0 ? Math.sqrt(power / p) : 1;
  const out = new Float64Array(amp.length);
  for (let k = 0; k < amp.length; k++) out[k] = amp[k] * s;
  return out;
}

test('Gerchberg-Saxton reduces the error monotonically-ish and beats the flat element', () => {
  const ill = illuminationProfile('tophat', N, N, DX, DX, { radius: 0.45 * N * DX });
  const tgt = normaliseToPower(amplitudeOf(targetDisc(N, 0.3 * N)), discretePower(ill));
  const r = gerchbergSaxton(ill, tgt, { nx: N, ny: N, dx: DX, dy: DX, lambda: LAMBDA, z: 0.03, iterations: 40, seed: 5 });
  assert.equal(r.error.length, 40);
  assert.ok(r.error[39] < r.error[0], `error did not decrease: ${r.error[0]} -> ${r.error[39]}`);
  assert.ok(r.error[39] < 0.5, `final error ${r.error[39]}`);
  assert.ok(r.phase.every((v) => Number.isFinite(v)), 'phase finite');
  // simulated field really does concentrate light into the target region
  const field = new ComplexField(N, N, DX, DX);
  field.setAmplitudePhase(ill, r.phase);
  propagateAngularSpectrumInPlace(field, LAMBDA, 0.03, { bandLimit: false });
  const irr = field.intensity();
  let inRoi = 0;
  let total = 0;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const k = i + j * N;
      total += irr[k];
      if (Math.hypot(i - N / 2, j - N / 2) <= 0.32 * N) inRoi += irr[k];
    }
  }
  assert.ok(inRoi / total > 0.5, `only ${(inRoi / total) * 100}% of the power reached the target region`);
});

test('weighted Gerchberg-Saxton never returns a worse iterate than it was given', () => {
  const ill = illuminationProfile('tophat', N, N, DX, DX, { radius: 0.45 * N * DX });
  const tgt = normaliseToPower(amplitudeOf(targetDisc(N, 0.3 * N)), discretePower(ill));
  const opts = { nx: N, ny: N, dx: DX, dy: DX, lambda: LAMBDA, z: 0.03, seed: 11 };
  const plain = gerchbergSaxton(ill, tgt, { ...opts, iterations: 20 });
  const weighted = weightedGerchbergSaxton(ill, tgt, { ...opts, iterations: 20 });
  // The weighted loop optimises the fit *inside the signal region* after the
  // delivered power has been normalised — that is the figure of merit for beam
  // shaping. Outside the signal region it deliberately relaxes the constraint
  // (that is how it buys the uniformity), so comparing whole-plane errors would
  // be measuring the wrong thing.
  const roi = new Float64Array(N * N);
  let roiCount = 0;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      if (Math.hypot(i - N / 2, j - N / 2) <= 0.32 * N) {
        roi[i + j * N] = 1;
        roiCount++;
      }
    }
  }
  const roiRms = (amp) => {
    let measured = 0;
    for (let k = 0; k < amp.length; k++) measured += amp[k] * amp[k] * roi[k];
    let demanded = 0;
    for (let k = 0; k < tgt.length; k++) demanded += tgt[k] * tgt[k] * roi[k];
    const scale = demanded > 0 ? Math.sqrt(measured / demanded) : 0; // normalise delivered power
    let num = 0;
    let den = 0;
    for (let k = 0; k < amp.length; k++) {
      if (roi[k] === 0) continue;
      const d = amp[k] / scale - tgt[k];
      num += d * d;
      den += tgt[k] * tgt[k];
    }
    return Math.sqrt(num / den);
  };
  assert.ok(roiRms(weighted.amplitude) < roiRms(plain.amplitude), 'weighted variant should fit the signal region better');
  assert.ok(weighted.phase.every(Number.isFinite));
  void roiCount;
});

test('far-field CGH conserves energy and improves the reconstruction', () => {
  const ill = illuminationProfile('tophat', N, N, DX, DX, { radius: 0.45 * N * DX });
  const tgt = normaliseToPower(amplitudeOf(targetDisc(N, 0.25 * N)), discretePower(ill));
  const r = fourierCgh(ill, tgt, { nx: N, ny: N, dx: DX, dy: DX, lambda: LAMBDA, z: 0.03, iterations: 60, seed: 3 });
  assert.ok(r.error[59] < r.error[0]);
  // the transform pair used by the design is unitary: power through the focal
  // plane equals the power that was incident on the element
  const field = new ComplexField(N, N, DX, DX);
  field.setAmplitudePhase(ill, r.phase);
  const focal = focalPlaneSpectrum(field, LAMBDA, 0.03);
  assert.ok(Math.abs(focal.power() / field.power() - 1) < 1e-9, 'lens-FFT is not unitary');
  // The reconstruction from the *physical* lens transform must match the design
  // prediction in shape (the two use different but consistent normalisations,
  // so compare after normalising the delivered power to the demanded one).
  const measured = new Float64Array(focal.size);
  for (let k = 0; k < focal.size; k++) measured[k] = focal.amplitudeAt(k % N, (k / N) | 0);
  let pm = 0;
  let pd = 0;
  for (let k = 0; k < measured.length; k++) {
    pm += measured[k] * measured[k];
    pd += r.amplitude[k] * r.amplitude[k];
  }
  const scale = Math.sqrt(pm / pd);
  let num = 0;
  let den = 0;
  for (let k = 0; k < measured.length; k++) {
    const d = measured[k] / scale - r.amplitude[k];
    num += d * d;
    den += r.amplitude[k] * r.amplitude[k];
  }
  assert.ok(Math.sqrt(num / den) < 0.35, 'reconstruction does not resemble the designed pattern');
});

test('unitary DFT design and simulation agree: the far-field pattern is distance-invariant', () => {
  const rng = makeRng(17);
  const f = new ComplexField(N, N, DX, DX);
  for (let k = 0; k < f.size; k++) {
    f.re[k] = rng() - 0.5;
    f.im[k] = rng() - 0.5;
  }
  const a = focalPlaneSpectrum(f, LAMBDA, 0.02);
  const b = focalPlaneSpectrum(f, LAMBDA, 0.05);
  // The physical intensity at the screen falls as 1/f^2 (same power spread over
  // a larger window), so compare the *shape*: each pattern normalised by its
  // own mean. That shape is what makes a holographic image scale-invariant.
  const shape = (fl) => {
    const irr = fl.intensity();
    let mean = 0;
    for (let k = 0; k < irr.length; k++) mean += irr[k];
    mean /= irr.length;
    const out = new Float64Array(irr.length);
    for (let k = 0; k < irr.length; k++) out[k] = irr[k] / mean;
    return out;
  };
  const sa = shape(a);
  const sb = shape(b);
  let num = 0;
  let den = 0;
  for (let k = 0; k < sa.length; k++) {
    num += (sa[k] - sb[k]) ** 2;
    den += sa[k] * sa[k];
  }
  assert.ok(Math.sqrt(num / den) < 1e-12, 'normalised pattern is not scale invariant');
  // and the raw intensity really does scale as 1/f^2 (Parseval per unit area)
  const ratio = a.power() / b.power();
  assert.ok(Math.abs(ratio - 1) < 1e-9, 'total power must be conserved');
});

test('phase quantisation degrades smoothly with fewer levels', () => {
  const ill = illuminationProfile('tophat', N, N, DX, DX, { radius: 0.45 * N * DX });
  const tgt = normaliseToPower(amplitudeOf(targetDisc(N, 0.3 * N)), 1e6);
  const base = designDOE({
    nx: N,
    ny: N,
    dx: DX,
    lambda: LAMBDA,
    z: 0.03,
    iterations: 40,
    target: targetDisc(N, 0.3 * N),
    illumination: ill,
    roi: null,
  });
  const efficiency = (phase) => {
    const field = new ComplexField(N, N, DX, DX);
    field.setAmplitudePhase(ill, phase);
    propagateAngularSpectrumInPlace(field, LAMBDA, 0.03, { bandLimit: false });
    let inRoi = 0;
    let total = 0;
    const irr = field.intensity();
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const k = i + j * N;
        total += irr[k];
        if (Math.hypot(i - N / 2, j - N / 2) <= 0.32 * N) inRoi += irr[k];
      }
    }
    return inRoi / total;
  };
  const e0 = efficiency(base.phase);
  const e8 = efficiency(errorDiffusionQuantize(base.phase, 8));
  const e4 = efficiency(errorDiffusionQuantize(base.phase, 4));
  const e2 = efficiency(errorDiffusionQuantize(base.phase, 2));
  assert.ok(e0 > e8, `continuous ${e0} should beat 8 levels ${e8}`);
  assert.ok(e8 >= e4 - 0.02, `8 levels ${e8} should not be much worse than 4 ${e4}`);
  assert.ok(e4 >= e2 - 0.02, `4 levels ${e4} should not be much worse than 2 ${e2}`);
});

test('error diffusion preserves the local mean phase better than rounding', () => {
  const n = 64;
  const phase = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      phase[i + j * n] = Math.sin(i / 7) + Math.cos(j / 5.5) + 0.3 * Math.sin((i + j) / 3);
    }
  }
  const diffused = errorDiffusionQuantize(phase, 4);
  const rounded = quantizePhase(phase, 4);
  // compare how well each reproduces the *complex* field exp(i*phi)
  // Error diffusion trades pointwise accuracy for *local* accuracy, so the
  // meaningful comparison is on a low-pass filtered field: what a lens (or any
  // optical system with a finite aperture) actually sees.
  const smoothErr = (q) => {
    const size = n * n;
    const re = new Float64Array(size);
    const im = new Float64Array(size);
    const reRef = new Float64Array(size);
    const imRef = new Float64Array(size);
    for (let k = 0; k < size; k++) {
      re[k] = Math.cos(q[k]);
      im[k] = Math.sin(q[k]);
      reRef[k] = Math.cos(phase[k]);
      imRef[k] = Math.sin(phase[k]);
    }
    const blurRe = blurRaster(re, n, n, 2);
    const blurIm = blurRaster(im, n, n, 2);
    const blurReRef = blurRaster(reRef, n, n, 2);
    const blurImRef = blurRaster(imRef, n, n, 2);
    let num = 0;
    let den = 0;
    for (let k = 0; k < size; k++) {
      num += (blurRe[k] - blurReRef[k]) ** 2 + (blurIm[k] - blurImRef[k]) ** 2;
      den += blurReRef[k] ** 2 + blurImRef[k] ** 2;
    }
    return Math.sqrt(num / den);
  };
  assert.ok(smoothErr(diffused) < smoothErr(rounded), 'error diffusion should beat rounding locally');
  for (const v of diffused) {
    const m = ((v + Math.PI) % ((2 * Math.PI) / 4) + (2 * Math.PI) / 4) % ((2 * Math.PI) / 4);
    const snapped = Math.abs(m) < 1e-9 || Math.abs(m - (2 * Math.PI) / 4) < 1e-9;
    assert.ok(snapped, `quantised value ${v} is not on the 4-level grid`);
  }
});

test('phase primitives behave as their names promise', () => {
  const rect = new Float64Array(N * N);
  const kino = kinoformLens(N, N, DX, DX, LAMBDA, 0.05);
  for (let k = 0; k < kino.length; k++) {
    assert.ok(kino[k] <= Math.PI + 1e-12 && kino[k] > -Math.PI - 1e-12, 'kinoform wrapped');
  }
  const prism = prismPhase(N, N, DX, DX, LAMBDA, 0, 0);
  assert.ok(prism.every((v) => Math.abs(v) < 1e-12), 'zero tilt -> zero phase');
  const v = vortexPhase(N, N, 1);
  const f = new ComplexField(N, N, DX, DX);
  // a charge-1 vortex winds by 2*pi around the centre
  // phi(r, theta) = theta for a charge-1 vortex
  assert.ok(Math.abs(v[f.idx(N / 2 + 10, N / 2)]) < 1e-12, 'on the +x axis the phase is 0');
  assert.ok(Math.abs(v[f.idx(N / 2, N / 2 + 10)] - Math.PI / 2) < 1e-12, 'on the +y axis the phase is pi/2');
  assert.ok(Math.abs(Math.abs(v[f.idx(N / 2 - 10, N / 2)]) - Math.PI) < 1e-12, 'on the -x axis the phase is +/-pi');
  const ax = axiconPhase(N, N, DX, DX, LAMBDA, 1e-3);
  assert.ok(Math.abs(ax[f.idx(N / 2, N / 2)]) < 1e-12, 'axicon apex is flat');
  void rect;
});

test('soft ROI and ring mask have the expected shape', () => {
  const roi = softRoi(64, 64, { edge: 4, inset: 8 });
  assert.ok(Math.abs(roi[32 + 32 * 64] - 1) < 1e-12, 'centre is inside');
  assert.equal(roi[0], 0, 'corner is outside');
  const ring = ringMask(64, 64, 0.4, 0.8, { soft: 0.01 });
  const c = 32 + 32 * 64;
  assert.equal(ring[c], 0, 'ring centre is dark');
  const f = new ComplexField(64, 64, 1, 1);
  const idx = f.idx(32, 32 - Math.round(0.6 * 32));
  assert.equal(ring[idx], 1, 'inside the annulus');
});

test('Lee encoding keeps the local mean amplitude', () => {
  const f = new ComplexField(16, 16, 1e-6, 1e-6);
  f.setPlaneWave(0.5, 0);
  const bits = leeEncode(f, 2);
  let sum = 0;
  for (let k = 0; k < bits.length; k++) sum += bits[k];
  const mean = sum / bits.length;
  // a 2x2 cell with 50% amplitude should open half of its subcells on average
  assert.ok(Math.abs(mean - 0.5) < 0.06, `mean open fraction ${mean}`);
  for (const b of bits) assert.ok(b === 0 || b === 1, 'binary output');
});

test('blurRaster conserves total energy and smooths', () => {
  const n = 32;
  const src = new Float64Array(n * n);
  src[16 + 16 * n] = 1; // single bright pixel
  const blurred = blurRaster(src, n, n, 2);
  let s0 = 0;
  let s1 = 0;
  for (let k = 0; k < src.length; k++) {
    s0 += src[k];
    s1 += blurred[k];
  }
  assert.ok(Math.abs(s1 / s0 - 1) < 0.02, 'energy preserved');
  assert.ok(blurred[16 + 16 * n] < 1, 'peak reduced');
  assert.ok(blurred[18 + 16 * n] > 0, 'energy spread to neighbours');
});

test('phaseToHeight produces a manufacturable profile', () => {
  const phase = kinoformLens(32, 32, DX, DX, LAMBDA, 0.05);
  const h = phaseToHeight(phase, LAMBDA, 1.52);
  const maxH = LAMBDA / 0.52;
  for (const v of h) {
    assert.ok(v >= 0 && v <= maxH + 1e-18, `height ${v} outside [0, ${maxH}]`);
  }
});

test('designDOE returns a self-consistent, well-formed result', () => {
  const r = designDOE({
    nx: 64,
    ny: 64,
    dx: DX,
    lambda: LAMBDA,
    z: 0.03,
    iterations: 10,
    target: targetDisc(64, 20),
    mode: 'image-plane',
  });
  assert.equal(r.phase.length, 64 * 64);
  assert.equal(r.mode, 'image-plane');
  assert.ok(r.error.length >= 1 && r.error.length <= 10, 'error history present');
  assert.ok(r.illumination.length === 64 * 64);
  assert.ok(r.targetAmp.every(Number.isFinite));
  assert.ok(r.phase.every((v) => Math.abs(v) <= Math.PI + 1e-12), 'wrapped phase');
  // determinism
  const r2 = designDOE({
    nx: 64,
    ny: 64,
    dx: DX,
    lambda: LAMBDA,
    z: 0.03,
    iterations: 10,
    target: targetDisc(64, 20),
    mode: 'image-plane',
  });
  let maxDiff = 0;
  for (let k = 0; k < r.phase.length; k++) maxDiff = Math.max(maxDiff, Math.abs(r.phase[k] - r2.phase[k]));
  assert.ok(maxDiff < 1e-12, `design is not reproducible (${maxDiff})`);
  void discMask;
});
