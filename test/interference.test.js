import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ComplexField } from '../src/field.js';
import {
  multipleBeamInterference,
  fringePeriod,
  nBeamInterference,
  modalIntensity,
  partiallyCoherentModes,
  recordHologram,
  reconstructHologram,
  unwrapPhase2d,
  zernikePhase,
  lgMode,
  hgMode,
  speckleField,
  diffractionOrders,
  fringeAnalysis,
} from '../src/interference.js';
import { propagateAngularSpectrumInPlace } from '../src/propagate.js';
import { fft2d, fftshift2d, ifftshift2d } from '../src/fft.js';
import { visibility } from '../src/metrics.js';

const N = 128;
const DX = 8e-6;
const LAMBDA = 532e-9;

test('two-beam interference fringes have the analytic period and visibility', () => {
  // A plane wave and its mirror image about the z axis, half-angle theta.
  const theta = 0.02; // 20 mrad between the beams
  const beams = [
    { thetaX: Math.sin(theta / 2), amplitude: 1 },
    { thetaX: -Math.sin(theta / 2), amplitude: 1 },
  ];
  const { intensity, visibility: vis } = multipleBeamInterference(N, N, DX, DX, LAMBDA, beams);
  const expectedPeriod = fringePeriod(LAMBDA, theta);
  // measure the fringe period along the centre row
  const row = [];
  for (let i = 0; i < N; i++) row.push(intensity[i + (N / 2) * N]);
  let maxima = 0;
  for (let i = 1; i < N - 1; i++) {
    if (row[i] > row[i - 1] && row[i] >= row[i + 1]) maxima++;
  }
  const measuredPeriod = (N * DX) / Math.max(1, maxima);
  assert.ok(
    Math.abs(measuredPeriod / expectedPeriod - 1) < 0.08,
    `measured period ${(measuredPeriod * 1e6).toFixed(2)} um vs analytic ${(expectedPeriod * 1e6).toFixed(2)} um`,
  );
  // equal amplitudes interfere with unit visibility
  assert.ok(Math.abs(vis - 1) < 0.05, `visibility ${vis}`);
  assert.ok(Math.abs(visibility(intensity) - 1) < 0.05, 'visibility helper agrees');
});

test('unequal beam amplitudes reduce the visibility analytically', () => {
  const beams = [
    { thetaX: 0.01, amplitude: 3 },
    { thetaX: -0.01, amplitude: 1 },
  ];
  const { visibility: analytic, intensity } = multipleBeamInterference(N, N, DX, DX, LAMBDA, beams);
  // V = 2 sqrt(I1 I2)/(I1+I2) = 2*3/(9+1) = 0.6
  assert.ok(Math.abs(analytic - 0.6) < 1e-3, `analytic visibility ${analytic}`);
  assert.ok(Math.abs(visibility(intensity) - 0.6) < 0.05, `measured visibility ${visibility(intensity)}`);
});

test('partial coherence: incoherent modes wash the fringes out', () => {
  // Two mutually incoherent plane waves travelling at +/- theta: the fringe
  // pattern of the coherent case disappears because the two intensities add.
  const a = new ComplexField(N, N, DX, DX);
  const b = new ComplexField(N, N, DX, DX);
  const k = (2 * Math.PI) / LAMBDA;
  const theta = 0.01;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const x = a.x(i);
      a.re[i + j * N] = Math.cos(k * Math.sin(theta) * x);
      a.im[i + j * N] = Math.sin(k * Math.sin(theta) * x);
      b.re[i + j * N] = Math.cos(-k * Math.sin(theta) * x);
      b.im[i + j * N] = Math.sin(-k * Math.sin(theta) * x);
    }
  }
  const coherent = ComplexField.sum([a, b]);
  const coherentVis = visibility(coherent.intensity());
  const incoherent = modalIntensity([a, b]);
  const incohVis = visibility(incoherent.intensity);
  assert.ok(coherentVis > 0.9, `coherent visibility ${coherentVis}`);
  assert.ok(incohVis < 0.05, `incoherent visibility ${incohVis}`);
});

test('partially coherent modes reduce speckle contrast', () => {
  const modes = partiallyCoherentModes(N, N, DX, DX, { modes: 6, sigma: 4 * DX, waist: 0.3 * N * DX, seed: 5 });
  assert.equal(modes.length, 6);
  const single = modes[0];
  const singleField = single.clone();
  propagateAngularSpectrumInPlace(singleField, LAMBDA, 0.05, { bandLimit: false });
  const { intensity } = modalIntensity(
    modes.map((m) => {
      const f = m.clone();
      propagateAngularSpectrumInPlace(f, LAMBDA, 0.05, { bandLimit: false });
      return f;
    }),
  );
  const singleI = singleField.intensity();
  const contrastOf = (irr) => {
    let sum = 0;
    let sumSq = 0;
    for (let k = 0; k < irr.length; k++) {
      sum += irr[k];
      sumSq += irr[k] * irr[k];
    }
    const mean = sum / irr.length;
    return Math.sqrt(Math.max(0, sumSq / irr.length - mean * mean)) / mean;
  };
  assert.ok(
    contrastOf(intensity) < contrastOf(singleI),
    `mode-averaged contrast ${contrastOf(intensity)} should beat single mode ${contrastOf(singleI)}`,
  );
});

test('hologram recording and reconstruction recover the object', () => {
  // Off-axis (Leith–Upatnieks) geometry: the object wave interferes with a
  // tilted plane reference so that the real image, the twin image and the
  // undiffracted zero order separate in the spatial-frequency domain. That
  // separation is the whole point of off-axis holography, so the test uses it:
  // the recorded plate is numerically illuminated with the reference, and the
  // image is recovered by band-passing the reconstruction around the carrier.
  const carrierCycles = 16; // fringes across the window
  const sinTheta = (LAMBDA * carrierCycles) / (N * DX); // sin of the reference tilt
  const k = (2 * Math.PI) / LAMBDA;
  const object = new ComplexField(N, N, DX, DX);
  const reference = new ComplexField(N, N, DX, DX);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const x = object.x(i);
      const y = object.y(j);
      // band-limited "object": two smooth blobs on a weak background
      const blob1 = Math.exp(-((x - 120e-6) ** 2 + (y + 60e-6) ** 2) / (120e-6) ** 2);
      const blob2 = 0.8 * Math.exp(-((x + 180e-6) ** 2 + (y - 120e-6) ** 2) / (70e-6) ** 2);
      object.re[i + j * N] = 0.05 + Math.max(blob1, blob2);
      reference.re[i + j * N] = Math.cos(k * sinTheta * x);
      reference.im[i + j * N] = Math.sin(k * sinTheta * x);
    }
  }
  const { plate, recorded } = recordHologram(object, reference, { encode: 'raw' });
  // A photographic plate records intensity: real, non-negative, peak-normalised.
  let min = Infinity;
  let max = 0;
  for (const v of recorded) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  assert.ok(min >= 0, 'recorded intensity is non-negative');
  assert.ok(Math.abs(max - 1) < 1e-12, 'recording is peak-normalised');
  assert.ok(plate.im.every((v) => v === 0), 'amplitude recording has no imaginary part');
  assert.ok(visibility(recorded) > 0.3, 'recording shows fringes');

  // Reconstruct: illuminate with the reference and propagate to the real image.
  const z = 0.005;
  const { real } = reconstructHologram(plate, reference, LAMBDA, z);
  const irr = real.intensity();
  let mean = 0;
  for (const v of irr) mean += v;
  mean /= irr.length;
  const re = new Float64Array(irr.length);
  const im = new Float64Array(irr.length);
  for (let p = 0; p < irr.length; p++) re[p] = irr[p] - mean;
  fft2d(re, im, N, N);
  fftshift2d(re, im, N, N);
  // Locate the carrier sideband: the reference runs along x, so the image
  // information sits in a narrow band of vertical frequencies around fy = 0.
  let best = -1;
  let bestI = 0;
  for (let i = 0; i < N; i++) {
    const fx = i - N / 2;
    if (Math.abs(fx) < 6 || Math.abs(fx) > 26) continue;
    let energy = 0;
    for (let j = 0; j < N; j++) {
      const fy = j - N / 2;
      if (Math.abs(fy) > 3) continue;
      const p = i + j * N;
      energy += re[p] * re[p] + im[p] * im[p];
    }
    if (energy > best) {
      best = energy;
      bestI = i;
    }
  }
  assert.ok(best > 0, 'found a carrier sideband');
  const fc = bestI - N / 2;
  // Keep only that sideband, which isolates the reconstructed image from the
  // zero order and from the twin image.
  const band = 6;
  for (let j = 0; j < N; j++) {
    const fy = j - N / 2;
    for (let i = 0; i < N; i++) {
      const p = i + j * N;
      const fx = i - N / 2;
      if (Math.abs(fx - fc) <= band && Math.abs(fy) <= band) continue;
      re[p] = 0;
      im[p] = 0;
    }
  }
  ifftshift2d(re, im, N, N);
  fft2d(re, im, N, N, true);
  // The recovered envelope should track the object amplitude. Compare only the
  // object's own field of view: outside it there is nothing to recover, so it
  // would just dilute the correlation.
  const env = new Float64Array(irr.length);
  let sumO = 0;
  let sumE = 0;
  let count = 0;
  for (let j = 0; j < N; j++) {
    const y = object.y(j);
    for (let i = 0; i < N; i++) {
      const x = object.x(i);
      const p = i + j * N;
      env[p] = Math.hypot(re[p], im[p]);
      if (Math.abs(x) < 400e-6 && Math.abs(y) < 400e-6) {
        sumO += object.re[p];
        sumE += env[p];
        count++;
      }
    }
  }
  const meanO = sumO / count;
  const meanE = sumE / count;
  let num = 0;
  let den = 0;
  let varO = 0;
  for (let j = 0; j < N; j++) {
    const y = object.y(j);
    for (let i = 0; i < N; i++) {
      const x = object.x(i);
      if (Math.abs(x) >= 400e-6 || Math.abs(y) >= 400e-6) continue;
      const p = i + j * N;
      const dE = env[p] - meanE;
      const dO = object.re[p] - meanO;
      num += dE * dO;
      den += dE * dE;
      varO += dO * dO;
    }
  }
  const corr = num / Math.sqrt(den * varO);
  assert.ok(corr > 0.85, `recovered image correlates with the object at ${corr.toFixed(4)}`);
  // and the brightest part of the recovered envelope must be inside the object
  let peakIdx = 0;
  for (let p = 0; p < env.length; p++) if (env[p] > env[peakIdx]) peakIdx = p;
  assert.ok(object.re[peakIdx] > 0.5, 'envelope peak lands on the object');
});

test('phase unwrapping recovers a smooth phase ramp', () => {
  const phase = new Float64Array(N * N);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      phase[i + j * N] = 0.3 * i + 0.2 * j; // exceeds 2*pi many times
    }
  }
  const wrapped = new Float64Array(phase.length);
  for (let k = 0; k < phase.length; k++) {
    let v = phase[k] % (2 * Math.PI);
    if (v > Math.PI) v -= 2 * Math.PI;
    else if (v <= -Math.PI) v += 2 * Math.PI;
    wrapped[k] = v;
  }
  const unwrapped = unwrapPhase2d(wrapped, N, N);
  let maxErr = 0;
  for (let k = 0; k < phase.length; k++) {
    // unwrapping is defined up to a global 2*pi multiple
    const d = unwrapped[k] - phase[k];
    maxErr = Math.max(maxErr, Math.abs(d - 2 * Math.PI * Math.round(d / (2 * Math.PI))));
  }
  assert.ok(maxErr < 1e-9, `unwrap error ${maxErr}`);
});

test('mode bases have the expected symmetries', () => {
  // Zernike defocus (n=2, m=0) is 2*r^2 - 1, so it is -1 on axis and +1 at the
  // rim, and it is a pure radial function.
  const defocus = zernikePhase(64, 64, 2, 0);
  const f = new ComplexField(64, 64, 1, 1);
  const c = f.idx(32, 32);
  assert.ok(defocus.some((v) => v !== 0), 'defocus is non-trivial');
  assert.ok(Math.abs(defocus[c] + 1) < 1e-12, 'defocus is -1 on axis');
  const r24 = f.idx(32, 32 - 16); // r = 0.5
  assert.ok(Math.abs(defocus[r24] - (2 * 0.25 - 1)) < 1e-12, 'defocus follows 2r^2-1');
  assert.ok(defocus[f.idx(4, 32)] === defocus[f.idx(32, 4)], 'radially symmetric');
  // a charge-1 Laguerre-Gauss mode has a phase vortex of 2*pi
  const lg = lgMode(64, 64, 1e-5, 1e-5, { p: 0, l: 1, w0: 200e-6 });
  const amp = Float64Array.from(lg.amplitude());
  assert.ok(Math.abs(amp[c]) < 1e-9, 'vortex has a null on axis');
  const hg = hgMode(64, 64, 1e-5, 1e-5, { m: 1, n: 0, w0: 200e-6 });
  const vals = hg.re;
  let asym = 0;
  for (let i = 0; i < 32; i++) asym += vals[f.idx(32 + i, 32)] + vals[f.idx(32 - i, 32)];
  assert.ok(Math.abs(asym) < 1e-18, 'HG10 is antisymmetric in x');
  assert.ok(hg.re.every(Number.isFinite));
  void hg.re[0];
});

test('speckle field is fully modulated and reproducible', () => {
  const a = speckleField(N, N, DX, DX, { sigma: DX, seed: 9 });
  const b = speckleField(N, N, DX, DX, { sigma: DX, seed: 9 });
  let same = true;
  for (let k = 0; k < a.size; k++) {
    if (Math.abs(a.re[k] - b.re[k]) > 1e-15 || Math.abs(a.im[k] - b.im[k]) > 1e-15) same = false;
  }
  assert.ok(same, 'same seed -> same speckle');
  for (let k = 0; k < a.size; k++) {
    assert.ok(Math.abs(Math.hypot(a.re[k], a.im[k]) - 1) < 1e-12, 'unit modulus before propagation');
  }
  const propagated = a.clone();
  propagateAngularSpectrumInPlace(propagated, LAMBDA, 0.1, { bandLimit: false });
  const irr = propagated.intensity();
  // fully developed speckle has an exponential intensity distribution: the
  // contrast (std/mean) approaches 1
  let sum = 0;
  let sumSq = 0;
  for (const v of irr) {
    sum += v;
    sumSq += v * v;
  }
  const mean = sum / irr.length;
  const contrast = Math.sqrt(Math.max(0, sumSq / irr.length - mean * mean)) / mean;
  // Fully developed speckle: the intensity is exponentially distributed and the
  // contrast (std/mean) is 1. This is the single most distinctive signature of
  // a coherent imaging system, and it is reproduced to within a few percent.
  assert.ok(Math.abs(contrast - 1) < 0.05, `speckle contrast ${contrast} (expected ~1)`);
  // a coarser phase screen has fewer independent scatterers -> less contrast
  const coarse = speckleField(N, N, DX, DX, { sigma: 8 * DX, seed: 9 });
  propagateAngularSpectrumInPlace(coarse, LAMBDA, 0.03, { bandLimit: false });
  const coarseI = coarse.intensity();
  let cs = 0;
  let cq = 0;
  for (const v of coarseI) {
    cs += v;
    cq += v * v;
  }
  const cm = cs / coarseI.length;
  assert.ok(Math.sqrt(cq / coarseI.length - cm * cm) / cm < contrast, 'coarse screen has lower contrast');
});

test('diffraction order synthesis and fringe analysis agree', () => {
  const orderField = diffractionOrders(N, N, DX, DX, LAMBDA, 0.03, [
    { mx: 0, my: 0, amplitude: 1 },
    { mx: 4, my: 0, amplitude: 1 },
  ]);
  const { visibility: vis, peak } = fringeAnalysis(orderField.intensity(), N, N, DX, DX);
  assert.ok(vis > 0.9, `visibility ${vis}`);
  // Orders 0 and 4 beat against each other: the intensity carries the
  // difference frequency 4/(N*dx), so the fringe period is (N*dx)/4.
  // An intensity is real, so its spectrum is Hermitian: the +/- fringe orders
  // are exact mirror images and the reported sign is arbitrary, only the
  // frequency magnitude is physical.
  assert.ok(
    Math.abs(Math.abs(peak.fx) - 4 / (N * DX)) < 1.1 / (N * DX),
    `fringe frequency |${peak.fx}| cyc/m, expected ${4 / (N * DX)}`,
  );
  assert.ok(Math.abs(peak.period - (N * DX) / 4) < 1.05 * DX, `fringe period ${peak.period * 1e6} um`);
  const nBeam = nBeamInterference(N, N, DX, DX, LAMBDA, { n: 6, theta: 0.01 });
  assert.ok(visibility(nBeam.intensity) > 0.3, 'N-beam pattern has structure');
});
