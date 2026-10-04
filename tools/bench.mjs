#!/usr/bin/env node
/**
 * bench.mjs — throughput and physics benchmarks for the library.
 *
 *   node tools/bench.mjs [--n 256] [--json]
 *
 * Two things are reported side by side:
 *   1. speed  — how long the transforms, propagators, design loop and metrics
 *      take at the grid sizes the workbench uses;
 *   2. fidelity — the conservation and round-trip errors that prove the fast
 *      paths (FFT, angular spectrum, Fresnel, lens transform) compute the
 *      physics they claim to.
 */

import { ComplexField } from '../src/field.js';
import { fft2d, dft2dNaive } from '../src/fft.js';
import {
  propagateAngularSpectrumInPlace,
  propagateFresnelFFT,
  propagateFresnelFFTInverse,
  focalPlaneSpectrum,
  angularSpectrumTransferFunction,
} from '../src/propagate.js';
import { ProjectorSystem, convergenceSweep } from '../src/projector.js';
import { ssim, rmsError, regionStats } from '../src/metrics.js';
import { designDOE, makeRng, illuminationProfile } from '../src/doe.js';

const args = process.argv.slice(2);
const getArg = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : dflt;
};
const JSON_OUT = args.includes('--json');
const LAMBDA = 532e-9;
const DX = 8e-6;

const results = { timings: {}, physics: {} };

function time(label, fn, { warmup = 1, reps = 5 } = {}) {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) fn();
  const dt = (performance.now() - t0) / reps;
  results.timings[label] = dt;
  if (!JSON_OUT) console.log(`  ${label.padEnd(46)} ${dt.toFixed(2).padStart(9)} ms`);
  return dt;
}

function randField(n, dx, seed = 3) {
  const f = new ComplexField(n, n, dx, dx);
  const rng = makeRng(seed);
  for (let k = 0; k < f.size; k++) {
    f.re[k] = rng() - 0.5;
    f.im[k] = rng() - 0.5;
  }
  return f;
}

function relErr(a, b) {
  let num = 0;
  let den = 0;
  for (let k = 0; k < a.re.length; k++) {
    num += (a.re[k] - b.re[k]) ** 2 + (a.im[k] - b.im[k]) ** 2;
    den += a.re[k] ** 2 + a.im[k] ** 2;
  }
  return Math.sqrt(num / den);
}

function record(key, value) {
  results.physics[key] = value;
  if (!JSON_OUT) console.log(`  ${key.padEnd(46)} ${typeof value === 'number' ? value.toExponential(3) : value}`);
}

const N = getArg('n', 256);
const nSmall = getArg('small', 64);

console.log('=================================================================');
console.log(` DiffractiveOpticalElement benchmarks   (node ${process.version})`);
console.log(` grid ${N}x${N}, pitch ${(DX * 1e6).toFixed(1)} um, lambda ${(LAMBDA * 1e9).toFixed(0)} nm`);
console.log('=================================================================\n');

console.log('-- transforms ------------------------------------------------');
{
  const f = randField(N, DX);
  const n2 = N * N;
  time(`fft2d forward ${N}x${N}`, () => fft2d(f.re, f.im, N, N, false), { reps: 10 });
  // naive DFT on a small grid for reference (the oracle: O(n^4))
  time(`dft2dNaive forward ${nSmall}x${nSmall} (oracle)`, () => {
    const g = randField(nSmall, DX, 5);
    dft2dNaive(g.re, g.im, nSmall, nSmall, false);
  }, { reps: 2 });
}

console.log('\n-- propagators -----------------------------------------------');
{
  const zc = (N * DX * DX) / LAMBDA;
  const src = randField(N, DX);
  time('angular spectrum (band-limited)', () => {
    const f = src.clone();
    propagateAngularSpectrumInPlace(f, LAMBDA, 0.03, { bandLimit: true });
  });
  time('angular spectrum (exact)', () => {
    const f = src.clone();
    propagateAngularSpectrumInPlace(f, LAMBDA, 0.03, { bandLimit: false });
  });
  time('single-FFT Fresnel (forward)', () => {
    const f = src.clone();
    propagateFresnelFFT(f, LAMBDA, zc, { dx: DX, dy: DX });
  });
  time('single-FFT Fresnel (inverse)', () => {
    const f = src.clone();
    propagateFresnelFFTInverse(f, LAMBDA, zc, { dx: DX, dy: DX });
  });
  time('lens focal-plane transform', () => focalPlaneSpectrum(src, LAMBDA, 0.03));
  time('transfer function build', () => angularSpectrumTransferFunction(N, N, DX, DX, LAMBDA, 0.03));

  // fidelity
  const g = src.clone();
  propagateAngularSpectrumInPlace(g, LAMBDA, 0.03, { bandLimit: false });
  propagateAngularSpectrumInPlace(g, LAMBDA, -0.03, { bandLimit: false });
  record('ASM round-trip relative error', relErr(src, g));

  const h = src.clone();
  propagateFresnelFFT(h, LAMBDA, zc, { dx: DX, dy: DX });
  const pFwd = h.power();
  const h2 = h.clone();
  propagateFresnelFFTInverse(h2, LAMBDA, zc, { dx: DX, dy: DX });
  record('Fresnel round-trip relative error', relErr(src, h2));
  record('Fresnel power drift (fwd/back)', Math.abs(h2.power() / pFwd - 1) + 0);

  const lens = focalPlaneSpectrum(src, LAMBDA, 0.03);
  record('lens transform power ratio', lens.power() / src.power());

  // ASM vs a brute-force Fresnel integral is checked in the test suite; here we
  // report the grid-bandwidth number that decides whether a distance is safe
  record('critical distance N*p^2/lambda [mm]', (zc * 1e3).toFixed(3));
}

console.log('\n-- design and simulation -------------------------------------');
{
  const cfg = { n: N, pitch: DX, lambda: LAMBDA, distance: 30e-3, iterations: 60, target: { kind: 'text', text: 'DOE' } };
  const sys = new ProjectorSystem({ ...cfg, mode: 'image-plane' });
  time('design, image-plane (60 iterations)', () => sys.design());
  time('simulate + metrics (image-plane)', () => sys.screenMetrics(sys.simulate()));
  const m = sys.screenMetrics(sys.simulate());
  console.log(
    `    -> rmse ${(m.rmse * 100).toFixed(2)} %, correlation ${m.correlation.toFixed(4)}, ` +
      `efficiency ${(m.efficiency * 100).toFixed(1)} %, ssim ${m.ssim.toFixed(4)}`,
  );

  const far = new ProjectorSystem({ ...cfg, mode: 'far-field' });
  time('design, far-field (60 iterations)', () => far.design());
  time('simulate + metrics (far-field)', () => far.screenMetrics(far.simulate()));
  const mf = far.screenMetrics(far.simulate());
  console.log(
    `    -> rmse ${(mf.rmse * 100).toFixed(2)} %, correlation ${mf.correlation.toFixed(4)}, ` +
      `efficiency ${(mf.efficiency * 100).toFixed(1)} %, ssim ${mf.ssim.toFixed(4)}`,
  );

  time('defocus curve (11 points, pad 2)', () => sys.defocusCurve({ points: 11, span: 0.6, pad: 2 }));
}

console.log('\n-- metrics ---------------------------------------------------');
{
  const n2 = N * N;
  const a = new Float64Array(n2);
  const b = new Float64Array(n2);
  const rng = makeRng(11);
  for (let k = 0; k < n2; k++) {
    const x = (k % N) - N / 2;
    a[k] = Math.exp(-(x * x) / 500) + 0.01 * rng();
    b[k] = a[k] + 0.02 * (rng() - 0.5);
  }
  const mask = new Float64Array(n2).fill(1);
  time('rmsError', () => rmsError(a, b, mask), { reps: 20 });
  time('ssim (8x8 windows)', () => ssim(a, b, N, N), { reps: 5 });
  time('regionStats', () => regionStats(a, mask), { reps: 20 });
}

console.log('\n-- end-to-end ------------------------------------------------');
{
  const sweepCfg = { n: 128, pitch: DX, lambda: LAMBDA, distance: 30e-3, mode: 'image-plane', target: { kind: 'text', text: 'DOE' } };
  const t0 = performance.now();
  const sweep = convergenceSweep(sweepCfg, { checkpoints: [1, 5, 20, 80] });
  const dt = performance.now() - t0;
  console.log(`  full convergence sweep (128x128, 1+5+20+80 iterations)  ${dt.toFixed(0)} ms`);
  console.log(`    -> rmse ${sweep.map((s) => `${s.iterations}:${(s.rmse * 100).toFixed(1)}%`).join('  ')}`);
  results.timings['convergence sweep 128'] = dt;
}

console.log('\n-- summary ---------------------------------------------------');
const designMs = results.timings['design, image-plane (60 iterations)'];
const simMs = results.timings['simulate + metrics (image-plane)'];
if (designMs !== undefined) {
  const iterations = 60;
  console.log(`  image-plane design: ${(designMs / iterations).toFixed(2)} ms/iteration at ${N}x${N}`);
  console.log(`  interactive replay: ${(1000 / simMs).toFixed(1)} projections/s (simulate + metrics)`);
}
console.log('  all propagators are O(N^2 log N) in the grid, O(1) in the wavelength.');

if (JSON_OUT) console.log(JSON.stringify(results, null, 2));
