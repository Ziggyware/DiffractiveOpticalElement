import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  maskedSum,
  powerOf,
  regionStats,
  rmsError,
  correlation,
  psnr,
  zeroOrderFraction,
  rmsRadius,
  ssim,
  visibility,
  summarizeMetrics,
} from '../src/metrics.js';
import { ComplexField } from '../src/field.js';

const n = 64;
const size = n * n;
const bump = new Float64Array(size);
for (let k = 0; k < size; k++) {
  const x = (k % n) - n / 2;
  const y = ((k / n) | 0) - n / 2;
  bump[k] = Math.exp(-(x * x + y * y) / 200);
}
const flat = new Float64Array(size).fill(1);
const scaled = Float64Array.from(bump, (v) => 2 * v);

test('maskedSum and powerOf respect the mask and the cell area', () => {
  const mask = new Float64Array(size);
  for (let k = 0; k < size; k++) mask[k] = k < size / 2 ? 1 : 0;
  assert.ok(Math.abs(maskedSum(flat, null) - size) < 1e-12);
  assert.ok(Math.abs(maskedSum(flat, mask) - size / 2) < 1e-12);
  const dx = 8e-6;
  assert.ok(Math.abs(powerOf(flat, dx, dx) - size * dx * dx) < 1e-20);
});

test('regionStats reports the standard projector uniformity figures', () => {
  const mask = new Float64Array(size).fill(1);
  const uniform = regionStats(flat, mask);
  assert.ok(Math.abs(uniform.uniformity - 1) < 1e-12, 'a flat region is perfectly uniform');
  assert.ok(Math.abs(uniform.cv) < 1e-12);
  assert.ok(Math.abs(uniform.flatness - 1) < 1e-12);
  // a region with a 10% hot spot: uniformity = 2*min/(max+min) = 2*1/2.1
  const hot = Float64Array.from(flat);
  hot[5] = 1.1;
  const st = regionStats(hot, mask);
  assert.ok(Math.abs(st.uniformity - (2 * 1) / 2.1) < 1e-12, `uniformity ${st.uniformity}`);
  assert.ok(st.max === 1.1 && st.min === 1);
  assert.ok(st.flatness < 1 && st.flatness > 0.99);
  // pixels at or below the (absolute) threshold are excluded, which is how the
  // dark background is kept out of the min/max. The default threshold of 0
  // therefore already drops exactly-zero pixels; a dim pixel still counts unless
  // the caller raises the threshold.
  const withZero = Float64Array.from(flat);
  withZero[0] = 0;
  assert.equal(regionStats(withZero, mask).uniformity, 1, 'a zero pixel is excluded by default');
  const withDim = Float64Array.from(flat);
  withDim[1] = 0.5;
  assert.ok(
    Math.abs(regionStats(withDim, mask).uniformity - (2 * 0.5) / 1.5) < 1e-12,
    'a 0.5 pixel scores 2*min/(max+min)',
  );
  assert.equal(regionStats(withDim, mask, { threshold: 0.5 }).uniformity, 1, 'a raised threshold drops dim pixels');
  // empty region
  const empty = regionStats(flat, new Float64Array(size));
  assert.equal(empty.n, 0);
  assert.equal(empty.uniformity, 0);
});

test('rmsError, correlation and psnr behave like their definitions', () => {
  assert.equal(rmsError(bump, bump), 0);
  assert.ok(Math.abs(rmsError(scaled, bump) - 1) < 1e-12, 'a 2x error is 100%');
  assert.ok(Math.abs(rmsError(Float64Array.from(bump, (v) => 1.1 * v), bump) - 0.1) < 1e-12);
  assert.ok(Math.abs(correlation(bump, scaled) - 1) < 1e-12, 'correlation is scale invariant');
  assert.ok(correlation(bump, Float64Array.from(bump, (v) => -v)) === -1);
  assert.equal(psnr(bump, bump), Infinity);
  const noisy = Float64Array.from(bump);
  noisy[100] += 0.1;
  assert.ok(Number.isFinite(psnr(bump, noisy)));
  assert.ok(psnr(bump, noisy) > 20);
  // an error of exactly 10% of the peak gives 20 dB (the peak comes from the
  // target/reference argument, so the reference goes second)
  const off = Float64Array.from(bump, (v) => v + 0.1 * Math.max(...bump));
  assert.ok(Math.abs(psnr(off, bump) - 20) < 0.01, `psnr ${psnr(off, bump)}`);
});

test('SSIM responds to image quality and is calibrated to the data range', () => {
  assert.ok(Math.abs(ssim(bump, bump, n, n) - 1) < 1e-12, 'identical planes score 1');
  const flatSsim = ssim(bump, flat, n, n);
  assert.ok(flatSsim < 0.2, `structureless comparison scores ${flatSsim}`);
  // noise of 1% of the peak barely moves SSIM
  const noisy = Float64Array.from(bump);
  let s = 7;
  const rng = () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648;
  };
  for (let k = 0; k < size; k++) noisy[k] += 0.01 * (rng() - 0.5);
  assert.ok(ssim(bump, noisy, n, n) > 0.98, `small noise scores ${ssim(bump, noisy, n, n)}`);
  // and it degrades monotonically as the noise grows
  const prev = [];
  for (const amp of [0.05, 0.2, 0.5, 1]) {
    const bad = Float64Array.from(bump);
    for (let k = 0; k < size; k++) bad[k] += amp * (rng() - 0.5);
    prev.push(ssim(bump, bad, n, n));
  }
  for (let i = 1; i < prev.length; i++) {
    assert.ok(prev[i] < prev[i - 1], `ssim must fall with noise (${prev.join(', ')})`);
  }
  // the data range matters: passing an explicit range must change the score the
  // same way as rescaling the planes
  const a = ssim(bump, scaled, n, n);
  const b = ssim(bump, scaled, n, n, { dataRange: 1 });
  assert.ok(a > b, 'a smaller assumed range makes the luminance mismatch hurt more');
});

test('SSIM can weight the windows by target energy (sparse images)', () => {
  // A sparse target: a small bright square in an otherwise dark plane, plus a
  // reconstruction that nails the square but leaves speckle everywhere else.
  const tgt = new Float64Array(size);
  const rec = new Float64Array(size);
  for (let k = 0; k < size; k++) {
    const x = (k % n) - n / 2;
    const y = ((k / n) | 0) - n / 2;
    const inside = Math.abs(x) < 6 && Math.abs(y) < 6;
    tgt[k] = inside ? 1 : 0;
    rec[k] = inside ? 1 : 0.2 * ((k % 7) - 3) * 0.1;
  }
  const unweighted = ssim(rec, tgt, n, n);
  const weighted = ssim(rec, tgt, n, n, { weight: 'target' });
  assert.ok(weighted > unweighted, `target weighting must help the signal region (${weighted} vs ${unweighted})`);
  assert.ok(weighted > 0.8, `the signal region is nearly perfect, scored ${weighted}`);
  // both must be 1 for a perfect reconstruction
  assert.ok(Math.abs(ssim(tgt, tgt, n, n, { weight: 'target' }) - 1) < 1e-12);
});

test('zeroOrderFraction measures DC leakage at the element', () => {
  const f = new ComplexField(32, 32, 8e-6, 8e-6);
  f.setPlaneWave(1, 0); // a flat element dumps everything into DC
  assert.ok(Math.abs(zeroOrderFraction(f) - 1) < 1e-12, 'flat phase -> 100% DC');
  // a pure grating with equal split has (almost) no DC
  const k = (2 * Math.PI) / 532e-9;
  const lam = 532e-9;
  const f2 = new ComplexField(32, 32, 8e-6, 8e-6);
  for (let j = 0; j < 32; j++) {
    for (let i = 0; i < 32; i++) {
      const ph = k * (lam / (4 * 8e-6)) * f2.x(i); // +-1 orders only
      f2.re[i + j * 32] = Math.cos(ph);
      f2.im[i + j * 32] = Math.sin(ph);
    }
  }
  assert.ok(zeroOrderFraction(f2) < 0.01, `grating DC ${zeroOrderFraction(f2)}`);
});

test('rmsRadius is the second-moment spot size in metres', () => {
  const dx = 10e-6;
  const plane = new Float64Array(size);
  // single illuminated pixel at the centre
  plane[n / 2 + (n / 2) * n] = 1;
  const r1 = rmsRadius(plane, n, dx);
  assert.ok(Math.abs(r1.r) < 1e-12, 'a centred delta has zero radius');
  // two pixels symmetric about the centre: rms radius = the offset
  const off = 3 * dx;
  plane.fill(0);
  plane[n / 2 + 3 + (n / 2) * n] = 1;
  plane[n / 2 - 3 + (n / 2) * n] = 1;
  const r2 = rmsRadius(plane, n, dx);
  assert.ok(Math.abs(r2.x - off) < 1e-15, `x radius ${r2.x} vs ${off}`);
  assert.ok(Math.abs(r2.y) < 1e-15);
  // anisotropic pitch is honoured
  const r3 = rmsRadius(plane, n, dx, null, null, 2 * dx);
  assert.ok(Math.abs(r3.x - off) < 1e-15);
  assert.equal(r3.y, 0);
});

test('visibility and summarizeMetrics', () => {
  const fringes = new Float64Array(size);
  for (let k = 0; k < size; k++) fringes[k] = 1 + 0.5 * Math.cos((2 * Math.PI * (k % n)) / 8);
  const v = visibility(fringes);
  // (max-min)/(max+min) = 1/2 for a perfect sinusoid of contrast 1/2
  assert.ok(Math.abs(v - 0.5) < 0.01, `visibility ${v}`);
  assert.equal(visibility(flat), 0);
  const txt = summarizeMetrics({
    efficiency: 0.7,
    uniformity: 0.9,
    flatness: 0.8,
    contrast: 0.2,
    rmse: 0.02,
    ssim: 0.99,
    snr: 34,
    zeroOrder: 0.001,
  });
  assert.ok(txt.includes('efficiency'));
  assert.ok(txt.includes('70.00 %'));
  assert.ok(txt.includes('SSIM'));
  assert.ok(txt.split('\n').length === 8);
});
