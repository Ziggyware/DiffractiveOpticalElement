import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ComplexField, discMask, rectMask, superGaussian } from '../src/field.js';

test('ComplexField geometry, power and normalisation', () => {
  const n = 32;
  const dx = 10e-6;
  const f = new ComplexField(n, n, dx, dx);
  f.setPlaneWave(2, 0);
  assert.equal(f.size, n * n);
  assert.ok(Math.abs(f.extentX - n * dx) < 1e-18);
  assert.ok(Math.abs(f.power() - 4 * (n * dx) ** 2) < 1e-20, 'uniform plane wave power');
  assert.ok(Math.abs(f.peakIntensity() - 4) < 1e-12);
  f.normalizePeak(1);
  assert.ok(Math.abs(f.peakIntensity() - 1) < 1e-12);
  f.normalizePower(1);
  assert.ok(Math.abs(f.power() - 1) < 1e-12);
});

test('amplitude, intensity and phase are consistent', () => {
  const f = new ComplexField(8, 8, 1e-6, 1e-6);
  f.setAmplitudePhase(3, Math.PI / 3);
  const irr = f.intensity();
  for (let k = 0; k < f.size; k++) {
    assert.ok(Math.abs(irr[k] - 9) < 1e-12);
    assert.ok(Math.abs(f.phase()[k] - Math.PI / 3) < 1e-12);
    assert.ok(Math.abs(f.amplitude()[k] - 3) < 1e-12);
  }
});

test('coherent and incoherent sums of two equal fields differ by the interference term', () => {
  const a = new ComplexField(16, 16, 1e-6, 1e-6);
  a.setPlaneWave(1, 0);
  const b = new ComplexField(16, 16, 1e-6, 1e-6);
  b.setPlaneWave(1, 0);
  const coherent = ComplexField.sum([a, b]);
  const incoh = ComplexField.sumIncoherent([a, b]);
  // in phase: |1+1|^2 = 4, incoherent: 1+1 = 2
  assert.ok(Math.abs(coherent.peakIntensity() - 4) < 1e-12);
  assert.ok(Math.abs(incoh.peakIntensity() - 2) < 1e-12);
  // pi apart: fully destructive coherent sum
  const c = new ComplexField(16, 16, 1e-6, 1e-6);
  c.setPlaneWave(1, Math.PI);
  const anti = ComplexField.sum([a, c]);
  assert.ok(anti.peakIntensity() < 1e-24, `destructive sum ${anti.peakIntensity()}`);
});

test('setAmplitudePhase / multiplyPhase / multiplyConjPhase are adjoint pairs', () => {
  const f = new ComplexField(8, 8, 1e-6, 1e-6);
  const p = new Float64Array(f.size);
  for (let k = 0; k < p.length; k++) p[k] = 0.1 * k;
  f.setPlaneWave(1, 0);
  const g = f.clone().multiplyPhase(p);
  g.multiplyConjPhase(p);
  let err = 0;
  for (let k = 0; k < f.size; k++) {
    err = Math.max(err, Math.abs(g.re[k] - f.re[k]), Math.abs(g.im[k] - f.im[k]));
  }
  assert.ok(err < 1e-12, `adjoint round trip ${err}`);
});

test('discMask and rectMask produce the expected areas', () => {
  const n = 128;
  const dx = 8e-6;
  const r = 200e-6;
  const disc = discMask(n, n, dx, dx, r, { soft: 0 });
  let count = 0;
  for (let k = 0; k < disc.length; k++) count += disc[k];
  const area = count * dx * dx;
  const analytic = Math.PI * r * r;
  assert.ok(Math.abs(area / analytic - 1) < 0.02, `disc area ${area} vs ${analytic}`);
  const rect = rectMask(n, n, dx, dx, 400e-6, 200e-6, { soft: 0 });
  let rc = 0;
  for (let k = 0; k < rect.length; k++) rc += rect[k];
  assert.ok(Math.abs((rc * dx * dx) / (400e-6 * 200e-6) - 1) < 0.04, 'rect area');
});

test('super-Gaussian approaches a flat top for large order', () => {
  const n = 128;
  const dx = 8e-6;
  const r = 300e-6;
  const a = superGaussian(n, n, dx, dx, r, 12);
  const f = new ComplexField(n, n, dx, dx);
  // inside 80% of the radius the profile is essentially flat
  const idx = f.idx(n / 2 + Math.round(0.8 * r / dx), n / 2);
  assert.ok(a[idx] > 0.9, `inside value ${a[idx]}`);
  assert.ok(a[f.idx(n / 2, n / 2)] > a[idx], 'profile decreases with radius');
  // beyond 120% it is essentially zero
  const idxOut = f.idx(n / 2 + Math.round(1.2 * r / dx), n / 2);
  assert.ok(a[idxOut] < 0.1, `outside value ${a[idxOut]}`);
});

test('resample pads and crops while preserving the field centre', () => {
  const n = 32;
  const dx = 8e-6;
  const f = new ComplexField(n, n, dx, dx);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = f.x(i);
      const y = f.y(j);
      f.re[i + j * n] = Math.exp(-(x * x + y * y) / (60e-6 * 60e-6));
    }
  }
  const padded = f.resample(n * 2, n * 2);
  assert.equal(padded.nx, n * 2);
  assert.ok(Math.abs(padded.re[padded.idx(n, n)] - f.re[f.idx(n / 2, n / 2)]) < 1e-15, 'centre preserved');
  const cropped = padded.resample(n, n);
  let err = 0;
  for (let k = 0; k < f.size; k++) err = Math.max(err, Math.abs(cropped.re[k] - f.re[k]));
  assert.ok(err < 1e-12, `pad+crop round trip ${err}`);
});

test('sampleImage maps a raster onto the grid', () => {
  const img = new Float64Array([1, 0, 0, 1]); // 2x2 checker
  const f = new ComplexField(32, 32, 1e-6, 1e-6);
  f.sampleImage(img, 2, 2, 1, 0);
  // The 2x2 checker covers the whole window, so each quadrant of the grid
  // carries one image pixel (sampled away from the interpolated seams).
  const q = 4;
  assert.ok(f.re[f.idx(q, q)] > 0.9, 'top-left bright');
  assert.ok(f.re[f.idx(32 - q, q)] < 0.1, 'top-right dark');
  assert.ok(f.re[f.idx(q, 32 - q)] < 0.1, 'bottom-left dark');
  assert.ok(f.re[f.idx(32 - q, 32 - q)] > 0.9, 'bottom-right bright');
  // the seam between image pixels falls halfway across the window
  assert.ok(Math.abs(f.re[f.idx(15, 4)] - 1) < 0.2, 'left half of a seam is bright');
  assert.ok(f.re[f.idx(17, 4)] < 0.2, 'right half of a seam is dark');
  // a smaller image is inset and its surroundings get the fill value
  const g = new ComplexField(32, 32, 1e-6, 1e-6);
  g.sampleImage(new Float64Array([1]), 1, 1, 0.5, 0);
  assert.ok(g.re[g.idx(16, 16)] > 0.99, 'centre of an inset image');
  assert.ok(g.re[g.idx(1, 1)] === 0, 'corner outside the inset image is fill');
});

test('spectrum of a plane wave is a delta at DC', () => {
  const n = 32;
  const f = new ComplexField(n, n, 8e-6, 8e-6);
  f.setPlaneWave(1, 0.3);
  const s = f.spectrum();
  const dc = Math.hypot(s.re[n / 2 + (n / 2) * n], s.im[n / 2 + (n / 2) * n]);
  assert.ok(Math.abs(dc - n * n) < 1e-8, `DC magnitude ${dc}`);
  // everything else is ~0
  let other = 0;
  for (let k = 0; k < s.re.length; k++) {
    if (k === n / 2 + (n / 2) * n) continue;
    other = Math.max(other, Math.hypot(s.re[k], s.im[k]));
  }
  assert.ok(other < 1e-9, `off-DC ${other}`);
});
