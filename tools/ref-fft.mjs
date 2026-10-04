// Reference implementation of the exact FFT used by index.html's GPU pipeline.
//
// Stockham radix-2, out-of-place, ping-pong buffers.
//
// ---------------------------------------------------------------------------
// Derivation (this is the algorithm transcribed into the GLSL 1-D pass kernel)
// ---------------------------------------------------------------------------
// Decimation-in-frequency, N = 2^t. After stage s the working array holds
// 2^s sub-DFTs of length M = N/2^s, interleaved with stride 2^s: element k of
// sub-DFT sigma lives at index  k*2^s + sigma.
// Sub-DFT sigma (length M, input g) splits into the two length-M/2 sub-DFTs
//      P[k] = DF T(g[0..M/2-1] + g[M/2..M-1])[k]
//      Q[k] = DFT((g[0..M/2-1] - g[M/2..M-1]) * exp(-2*pi*i*k/M))[k]
// so, in storage,  src[k*2^s+sigma]  and  src[k*2^s+sigma+N/2]  combine into
// dst[2*(k*2^s+sigma)] and dst[2*(k*2^s+sigma)+1].  Writing the OUTPUT index as
// o = 2*i + b with i = k*2^s + sigma gives  k = o >> (s+1)  and the sources are
// i and i + N/2 -- always exactly N/2 apart, so each shader invocation needs
// two texelFetch calls 0.5N apart along the axis.
// Output ordering after the last stage is bit-reversed (the classic Stockham
// property); the final pass of each axis undoes it by evaluating the fragment
// at output index o = bitReverse(p) since bit reversal is an involution.
//
// Forward transform multiplies by 1/N per 1-D pass (1/N^2 for the 2-D
// transform) so that for a unit-modulus phase mask  sum_u I(u) == 1  and the
// zero-order fraction is directly  I(0,0) / sum_u I(u).
// ---------------------------------------------------------------------------

export function bitrev(x, bits) {
  let r = 0;
  for (let i = 0; i < bits; i++) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

// One pass of the 1-D Stockham radix-2 FFT applied to every row (axis=0, along
// x) or every column (axis=1, along y) of an N x N complex array.
// src/dst are Float64Array(2*N*N), interleaved re/im, index = (y*N+x)*2.
export function fft1dPass(src, dst, N, s, axis, sign, norm, permute) {
  const half = N >> 1;
  const M = N >> s; // sub-DFT length of the *source* array
  const t = Math.log2(N) | 0;
  for (let q = 0; q < N; q++) {
    for (let p = 0; p < N; p++) {   // p = physical index of the output texel
      const o = permute ? bitrev(p, t) : p; // logical output index
      const i = o >> 1;                    // source pair index
      const k = o >> (s + 1);              // index inside the source sub-DFT
      let a0, a1, da;
      if (axis === 0) {
        a0 = (q * N + i) * 2; a1 = (q * N + i + half) * 2; da = (q * N + p) * 2;
      } else {
        a0 = (i * N + q) * 2; a1 = ((i + half) * N + q) * 2; da = (p * N + q) * 2;
      }
      const c0r = src[a0], c0i = src[a0 + 1];
      const c1r = src[a1], c1i = src[a1 + 1];
      if ((o & 1) === 0) {
        dst[da] = (c0r + c1r) * norm; dst[da + 1] = (c0i + c1i) * norm;
      } else {
        const ang = (sign * 2 * Math.PI * k) / M;
        const wr = Math.cos(ang), wi = Math.sin(ang);
        const dr = (c0r - c1r) * norm, di = (c0i - c1i) * norm;
        dst[da] = dr * wr - di * wi;
        dst[da + 1] = dr * wi + di * wr;
      }
    }
  }
}

// Full 2-D transform. Returns the buffer holding the result.
export function fft2d(a, b, N, opts = {}) {
  if (a === b) throw new Error('fft2d: source and destination must differ (the transform ping-pongs in place)');
  const inverse = !!opts.inverse;
  const sign = inverse ? +1 : -1;
  const t = Math.log2(N) | 0;
  let cur = a, nxt = b;
  for (let axis = 0; axis < 2; axis++)
    for (let s = 0; s < t; s++) {
      // forward: 1/N once per axis (on its last pass) => 1/N^2 for the 2-D
      // transform; inverse: no scaling, so inv(fwd(f)) == f.
      const norm = !inverse && s === t - 1 ? 1 / N : 1;
      fft1dPass(cur, nxt, N, s, axis, sign, norm, s === t - 1);
      const tmp = cur; cur = nxt; nxt = tmp;
    }
  return cur;
}

// Naive DFT for validation, same normalisation convention (1/N^2 forward).
export function dft2d(a, N, sign = -1, norm = 1 / (N * N)) {
  const out = new Float64Array(2 * N * N);
  for (let v = 0; v < N; v++)
    for (let u = 0; u < N; u++) {
      let sr = 0, si = 0;
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++) {
          const ang = (sign * 2 * Math.PI * (u * x + v * y)) / N;
          const c = Math.cos(ang), sn = Math.sin(ang);
          const fr = a[(y * N + x) * 2], fi = a[(y * N + x) * 2 + 1];
          sr += fr * c - fi * sn;
          si += fr * sn + fi * c;
        }
      out[(v * N + u) * 2] = sr * norm;
      out[(v * N + u) * 2 + 1] = si * norm;
    }
  return out;
}

export const ihash = (i) => {
  // Seeded integer hash, identical bit-for-bit in GLSL ES 3.0 uint maths.
  let x = i >>> 0;
  x = (x ^ 61) ^ (x >>> 16);
  x = (x + (x << 3)) >>> 0;
  x = x ^ (x >>> 4);
  x = Math.imul(x, 0x27d4eb2d) >>> 0;
  x = x ^ (x >>> 15);
  return x >>> 0;
};
