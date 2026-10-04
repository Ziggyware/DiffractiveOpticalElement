// Design tool: numerically optimises the transition points of a 1-D binary
// (0/pi) Dammann-style phase cell so that the 2K+1 lowest diffraction orders
// carry equal energy.  The resulting transition tables are pasted into
// doe.mjs / index.html (DAMMANN) so the runtime needs no solver.
//
// Cell coefficients of a binary phase cell with transitions T (phase 0 on the
// first segment, pi on the second, ...):
//   c_m = sum_s (-1)^s (e^{-2i.pi.m.b_s} - e^{-2i.pi.m.a_s}) / (-2i.pi.m)
function coeffs(T, M) {
  const seg = [];
  let a = 0, s = 0;
  for (const t of T) { seg.push([a, t, s]); a = t; s ^= 1; }
  seg.push([a, 1, s]);
  const out = [];
  for (let m = 0; m <= M; m++) {
    let re = 0, im = 0;
    for (const [sa, sb, sg] of seg) {
      if (m === 0) { re += (sg ? -1 : 1) * (sb - sa); continue; }
      const w = (sg ? -1 : 1) / (2 * Math.PI * m);
      // e^{-2i.pi m b} - e^{-2i.pi m a} rotated by (-i): use direct formula
      const cb = Math.cos(2 * Math.PI * m * sb), sb2 = Math.sin(2 * Math.PI * m * sb);
      const ca = Math.cos(2 * Math.PI * m * sa), sa2 = Math.sin(2 * Math.PI * m * sa);
      const dr = cb - ca, di = -(sb2 - sa2);          // e^{-2i.pi m b} - e^{-2i.pi m a}
      const qr = di, qi = -dr;                        // divide by -2i.pi.m  -> *i/(2.pi.m)
      re += w * qr; im += w * qi;
    }
    out.push(Math.hypot(re, im));
  }
  return out;
}

function cost(T, K) {
  const a = coeffs(T, K);
  // sign pattern is irrelevant: intensity of order m is |c_m|^2, and symmetric
  // transitions make c_m real, so |c_m| = |c_-m|
  const I = a.map((v) => v * v);
  const mean = I.reduce((s, v) => s + v, 0) / I.length;
  let c = 0;
  for (const v of I) c += (v - mean) ** 2;
  // second term: push as much energy as possible into the designed orders
  // (the 2-D element is the outer product of two 1-D cells, so its efficiency
  // is the square of the 1-D efficiency)
  const eff = I.reduce((s, v) => s + v, 0) * 2 - I[0];
  return c / (mean * mean) + 0.05 * (1 - eff);
}

function optimise(K, restarts = 400) {
  const nPar = K;                       // symmetry: T[2K-1-i] = 1 - T[i]
  let best = null;
  let seed = 12345;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  for (let r = 0; r < restarts; r++) {
    const p = []; // half-cell transitions, kept < 0.5
    for (let i = 0; i < nPar; i++) p.push(0.01 + 0.48 * rnd());
    p.sort((x, y) => x - y);
    let step = 0.08;
    let cur = cost(expand(p), K);
    for (let it = 0; it < 4000; it++) {
      const q = p.slice();
      const i = (rnd() * nPar) | 0;
      q[i] += (rnd() * 2 - 1) * step;
      q.sort((x, y) => x - y);
      if (q.some((v, k) => v < 0.005 || v > 0.495 || (k && q[k] - q[k - 1] < 0.005))) continue;
      const c = cost(expand(q), K);
      if (c < cur) { cur = c; for (let k = 0; k < nPar; k++) p[k] = q[k]; }
      step = Math.max(0.0004, step * 0.9995);
    }
    if (!best || cur < best.c) best = { c: cur, T: expand(p) };
  }
  return best;
}

function expand(p) {
  // mirror the half-cell about 0.5
  const T = [];
  for (const v of p) T.push(v);
  for (let i = p.length - 1; i >= 0; i--) T.push(1 - p[i]);
  return T;
}

const table = {};
for (const [spots, K] of [[3, 1], [5, 2], [7, 3], [9, 4]]) {
  const b = optimise(K);
  const a = coeffs(b.T, K);
  const I = a.map((v) => v * v);
  const tot = I.reduce((s, v) => s + v, 0) + I.slice(1).reduce((s, v) => s + v, 0);
  table[spots] = b.T.map((v) => +v.toFixed(4));
  console.log(
    `${spots} spots (K=${K}): cost=${b.c.toFixed(6)}  T=[${b.T.map((v) => v.toFixed(4)).join(', ')}]`
  );
  console.log('   order intensities (0..K): ' + I.map((v) => v.toFixed(4)).join(' ') +
    '   total in designed orders: ' + tot.toFixed(3));
}
console.log(JSON.stringify(table));
