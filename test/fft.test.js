import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fft1d, fft2d, dft2dNaive, fftshift1d, ifftshift1d, nextPow2, isPow2, clearTwiddleCache } from '../src/fft.js';

function randomArray(n, seed = 1) {
  let s = seed;
  const rnd = () => {
    s = (1103515245 * s + 12345) % 2147483648;
    return s / 2147483648 - 0.5;
  };
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = rnd();
  return a;
}

test('fft1d matches the naive DFT', () => {
  const n = 16;
  const re = randomArray(n, 3);
  const im = randomArray(n, 7);
  const expectedRe = new Float64Array(n);
  const expectedIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let i = 0; i < n; i++) {
      const a = (-2 * Math.PI * i * k) / n;
      sr += re[i] * Math.cos(a) - im[i] * Math.sin(a);
      si += re[i] * Math.sin(a) + im[i] * Math.cos(a);
    }
    expectedRe[k] = sr;
    expectedIm[k] = si;
  }
  const r = Float64Array.from(re);
  const i2 = Float64Array.from(im);
  fft1d(r, i2, n);
  for (let k = 0; k < n; k++) {
    assert.ok(Math.abs(r[k] - expectedRe[k]) < 1e-10, `re[${k}]`);
    assert.ok(Math.abs(i2[k] - expectedIm[k]) < 1e-10, `im[${k}]`);
  }
});

test('fft2d matches the naive 2D DFT and is invertible', () => {
  const nx = 8;
  const ny = 8;
  const re = randomArray(nx * ny, 11);
  const im = randomArray(nx * ny, 13);
  const naive = dft2dNaive(re, im, nx, ny, false);
  const r = Float64Array.from(re);
  const i2 = Float64Array.from(im);
  fft2d(r, i2, nx, ny, false);
  let maxErr = 0;
  for (let k = 0; k < r.length; k++) {
    maxErr = Math.max(maxErr, Math.abs(r[k] - naive.re[k]), Math.abs(i2[k] - naive.im[k]));
  }
  assert.ok(maxErr < 1e-10, `max err ${maxErr}`);
  fft2d(r, i2, nx, ny, true);
  let roundTrip = 0;
  for (let k = 0; k < r.length; k++) {
    roundTrip = Math.max(roundTrip, Math.abs(r[k] - re[k]), Math.abs(i2[k] - im[k]));
  }
  assert.ok(roundTrip < 1e-12, `round trip ${roundTrip}`);
});

test('fftshift is its own inverse and moves DC to the centre', () => {
  const n = 8;
  const a = new Float64Array(n);
  a[0] = 1; // delta at index 0 -> flat spectrum
  const b = Float64Array.from(a);
  fftshift1d(b, n, true);
  assert.equal(b[n / 2], 1);
  ifftshift1d(b, n);
  assert.equal(b[0], 1);
});

test('transform of a delta is flat and Parseval holds', () => {
  const n = 32;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re[0] = 1;
  fft1d(re, im, n);
  for (let k = 0; k < n; k++) {
    assert.ok(Math.abs(Math.hypot(re[k], im[k]) - 1) < 1e-12);
  }
  const x = randomArray(n, 17);
  const y = randomArray(n, 19);
  const xr = Float64Array.from(x);
  const xi = Float64Array.from(y);
  fft1d(xr, xi, n);
  let time = 0;
  let freq = 0;
  for (let k = 0; k < n; k++) {
    time += x[k] * x[k] + y[k] * y[k];
    freq += xr[k] * xr[k] + xi[k] * xi[k];
  }
  assert.ok(Math.abs(time - freq / n) / time < 1e-12, 'Parseval');
});

test('utilities', () => {
  assert.equal(nextPow2(1), 1);
  assert.equal(nextPow2(5), 8);
  assert.equal(nextPow2(1024), 1024);
  assert.ok(isPow2(256));
  assert.ok(!isPow2(100));
  clearTwiddleCache();
});
