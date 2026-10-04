/**
 * fft.js — in-place power-of-two complex FFTs over split (real, imaginary) planes.
 *
 * Conventions
 * -----------
 * Forward transform (unscaled DFT):
 *
 *      X[k] = sum_n x[n] * exp(-2*pi*i*n*k/N)
 *
 * Inverse transform (unitary-ish, scaled by 1/N):
 *
 *      x[k] = (1/N) * sum_n X[n] * exp(+2*pi*i*n*k/N)
 *
 * Complex fields are stored as two Float64Array planes of length nx*ny with
 * row-major indexing `idx = x + y*nx`. Keeping the planes split (rather than
 * interleaved) makes the row/column passes cache friendly and avoids per-sample
 * object allocation, which matters because DOE synthesis runs hundreds of
 * transforms per design.
 *
 * Everything here requires power-of-two lengths. That is a deliberate choice:
 * radix-2 keeps the kernel branch-free and exactly symmetric between the
 * forward and inverse directions, so round-trip tests are bit-tight.
 */

/** Twiddle tables e^{-2*pi*i*k/n} for k in [0, n/2), cached per length. */
const twiddleCache = new Map();
const TWIDDLE_CACHE_LIMIT = 64;

function twiddles(n) {
  let t = twiddleCache.get(n);
  if (t === undefined) {
    const half = n >>> 1;
    const cos = new Float64Array(half);
    const sin = new Float64Array(half);
    for (let k = 0; k < half; k++) {
      const a = (-2 * Math.PI * k) / n;
      cos[k] = Math.cos(a);
      sin[k] = Math.sin(a);
    }
    t = { cos, sin };
    if (twiddleCache.size >= TWIDDLE_CACHE_LIMIT) {
      // Bounded cache: drop the oldest entry.
      twiddleCache.delete(twiddleCache.keys().next().value);
    }
    twiddleCache.set(n, t);
  }
  return t;
}

/**
 * Clear the twiddle cache (useful in tests / long-lived workers that sweep
 * across many different transform sizes).
 */
export function clearTwiddleCache() {
  twiddleCache.clear();
}

/** Smallest power of two >= n. */
export function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** True when n is a power of two (and positive). */
export function isPow2(n) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

/**
 * 1D in-place complex FFT over a strided slice.
 *
 * Decimation-in-time, iterative, with an initial bit-reversal permutation.
 * Striding lets the 2D transform reuse this kernel for columns without a
 * transpose (each column pass walks memory with stride nx, which for
 * power-of-two nx is a fixed set of cache lines — acceptable and much simpler
 * than a blocked transpose).
 *
 * @param {Float64Array} re real plane
 * @param {Float64Array} im imaginary plane
 * @param {number} n transform length (power of two)
 * @param {number} stride element stride between consecutive samples
 * @param {number} offset index of the first sample
 * @param {boolean} inverse true for the conjugate kernel (+i)
 */
export function fft1d(re, im, n, stride = 1, offset = 0, inverse = false) {
  if (n <= 1) return;
  if (!isPow2(n)) throw new Error(`fft1d: length must be a power of two, got ${n}`);

  // --- bit-reversal permutation -------------------------------------------
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >>> 1;
    for (; j & bit; bit >>>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const ia = offset + i * stride;
      const ja = offset + j * stride;
      const tr = re[ia];
      re[ia] = re[ja];
      re[ja] = tr;
      const ti = im[ia];
      im[ia] = im[ja];
      im[ja] = ti;
    }
  }

  // --- butterflies ---------------------------------------------------------
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >>> 1;
    const t = twiddles(len);
    const { cos, sin } = t;
    for (let base = 0; base < n; base += len) {
      const b0 = offset + base * stride;
      for (let k = 0; k < half; k++) {
        const wr = cos[k];
        const wi = inverse ? -sin[k] : sin[k];
        const p = b0 + k * stride;
        const q = p + half * stride;
        const xr = re[q];
        const xi = im[q];
        const tr = xr * wr - xi * wi;
        const ti = xr * wi + xi * wr;
        re[q] = re[p] - tr;
        im[q] = im[p] - ti;
        re[p] += tr;
        im[p] += ti;
      }
    }
  }

  if (inverse) {
    const s = 1 / n;
    for (let i = 0; i < n; i++) {
      const p = offset + i * stride;
      re[p] *= s;
      im[p] *= s;
    }
  }
}

/**
 * 2D in-place complex FFT of an nx-by-ny grid (both powers of two).
 *
 * Forward is UNSCALED, inverse carries the full 1/(nx*ny) — so
 * fft2d(re, im, nx, ny, true) is the exact inverse of
 * fft2d(re, im, nx, ny, false), which the tests verify against a naive DFT.
 */
export function fft2d(re, im, nx, ny, inverse = false) {
  if (re.length !== nx * ny || im.length !== nx * ny) {
    throw new Error(`fft2d: plane length ${re.length} does not match ${nx}x${ny}`);
  }
  for (let y = 0; y < ny; y++) fft1d(re, im, nx, 1, y * nx, inverse);
  for (let x = 0; x < nx; x++) fft1d(re, im, ny, nx, x, inverse);
}

/** Out-of-place 2D FFT into new planes. */
export function fft2dOut(re, im, nx, ny, inverse = false) {
  const or = Float64Array.from(re);
  const oi = Float64Array.from(im);
  fft2d(or, oi, nx, ny, inverse);
  return { re: or, im: oi };
}

/**
 * Swap the quadrants along one axis so that the zero frequency moves to the
 * array centre. `shift(true)` performs a forward fftshift, `shift(false)` the
 * inverse (they differ only for odd lengths; we keep the distinction so the
 * code reads the way the maths does).
 */
export function fftshift1d(arr, n, shift = true, stride = 1, offset = 0) {
  const m = n >>> 1;
  const d = shift ? m : n - m;
  for (let i = 0; i < m; i++) {
    const a = offset + i * stride;
    const b = offset + ((i + d) % n) * stride;
    const t = arr[a];
    arr[a] = arr[b];
    arr[b] = t;
  }
  return arr;
}

export function ifftshift1d(arr, n, stride = 1, offset = 0) {
  return fftshift1d(arr, n, false, stride, offset);
}

/** Quadrant swap of a 2D grid, applied to both planes. */
export function fftshift2d(re, im, nx, ny, shift = true) {
  for (let y = 0; y < ny; y++) {
    const row = y * nx;
    fftshift1d(re, nx, shift, 1, row);
    if (im !== null) fftshift1d(im, nx, shift, 1, row);
  }
  for (let x = 0; x < nx; x++) {
    fftshift1d(re, ny, shift, nx, x);
    if (im !== null) fftshift1d(im, ny, shift, nx, x);
  }
}

export function ifftshift2d(re, im, nx, ny) {
  fftshift2d(re, im, nx, ny, false);
}

/**
 * Naive O(N^2) 2D DFT — only used as a reference oracle by the test suite.
 * Signs follow the forward convention above; useful for small grids.
 */
export function dft2dNaive(re, im, nx, ny, inverse = false) {
  const or = new Float64Array(nx * ny);
  const oi = new Float64Array(nx * ny);
  const sign = inverse ? 1 : -1;
  const scale = inverse ? 1 / (nx * ny) : 1;
  for (let ky = 0; ky < ny; ky++) {
    for (let kx = 0; kx < nx; kx++) {
      let sr = 0;
      let si = 0;
      for (let y = 0; y < ny; y++) {
        const ay = (sign * 2 * Math.PI * ky * y) / ny;
        const cy = Math.cos(ay);
        const sy = Math.sin(ay);
        for (let x = 0; x < nx; x++) {
          const ax = (sign * 2 * Math.PI * kx * x) / nx;
          const c = Math.cos(ax) * cy - Math.sin(ax) * sy;
          const s = Math.sin(ax) * cy + Math.cos(ax) * sy;
          const vr = re[x + y * nx];
          const vi = im[x + y * nx];
          sr += vr * c - vi * s;
          si += vr * s + vi * c;
        }
      }
      const o = kx + ky * nx;
      or[o] = sr * scale;
      oi[o] = si * scale;
    }
  }
  return { re: or, im: oi };
}

/**
 * Real-input forward transform helper: packs a purely real plane and returns
 * the Hermitian half-spectrum-free full complex spectrum (simple version used
 * by visualization code).
 */
export function fft2dReal(real, nx, ny) {
  const re = Float64Array.from(real);
  const im = new Float64Array(nx * ny);
  fft2d(re, im, nx, ny, false);
  return { re, im };
}
