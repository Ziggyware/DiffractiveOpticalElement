// ===========================================================================
//  doe.mjs -- canonical scalar-diffraction model used by the WebGL2 piece.
//  Everything in index.html's GLSL is a transcription of the functions here;
//  tools/selftest-cpu.mjs runs the graded self-test against this file, and
//  tools/preview.mjs renders PNGs of every scene from it.
// ===========================================================================
import { fft2d, ihash } from './ref-fft.mjs';
export { ihash };

export const LAM_R = 640e-9, LAM_G = 532e-9, LAM_B = 450e-9;
export const N_INDEX = 1.5;        // dispersionless material index (documented)
export const S_SCREEN = 0.45;      // sin(theta) half-width of the replay window
export const TWO_PI = Math.PI * 2;

export const TAU = TWO_PI;
export const frac = (t) => t - Math.floor(t);

// --- scene list ------------------------------------------------------------
export const SCENES = [
  { id: 'grating',  name: 'blazed grating' },
  { id: 'zfl',      name: 'Fresnel zone lens' },
  { id: 'axicon',   name: 'axicon / Bessel' },
  { id: 'vortex',   name: 'spiral phase plate' },
  { id: 'dammann',  name: 'Dammann lattice' },
  { id: 'holo',     name: 'GS hologram' },
];

// --- 1. blazed grating -----------------------------------------------------
export function phaseGrating(x, y, prm) {
  const period = prm.gratingPeriodPx * prm.p;
  return TAU * frac(x / period);
}

// --- 2. Fresnel zone lens --------------------------------------------------
export function phaseZFL(x, y, prm) {
  const r2 = x * x + y * y;
  return -Math.PI * r2 / (prm.lamDesign * prm.zflF);
}

// --- 3. axicon -------------------------------------------------------------
export function phaseAxicon(x, y, prm) {
  return -TAU * prm.axiconAlpha * Math.hypot(x, y) / prm.lamDesign;
}

// --- 4. spiral phase plate -------------------------------------------------
export function phaseVortex(x, y, prm) {
  return prm.vortexL * Math.atan2(y, x);
}

// --- 5. Dammann binary phase lattice --------------------------------------
// Separable 0/pi binary cell whose transitions are numerically optimised so
// that 2K+1 diffraction orders carry equal energy (see tools/dammann.mjs).
// v(t) = 0/1 per segment, cell phase = pi * (vx XOR vy).
// transitions optimised by tools/dammann.mjs (equal order energies, maximum
// total efficiency); 1-D efficiency in brackets, the 2-D lattice keeps its
// square because the element is the outer product of two 1-D cells.
export const DAMMANN = [
  { spots: 3, T: [0.3675, 0.6325], eff: 0.664 },
  { spots: 5, T: [0.0192, 0.3676, 0.6324, 0.9808], eff: 0.774 },
  { spots: 7, T: [0.1228, 0.3447, 0.3953, 0.6047, 0.6553, 0.8772], eff: 0.656 },
  { spots: 9, T: [0.0998, 0.1592, 0.3690, 0.4917, 0.5083, 0.6310, 0.8408, 0.9002], eff: 0.663 },
];

// continuous spot count 3..9 -> blended transition table ("Dammann-style")
export function dammannTable(spots) {
  const s = Math.min(9, Math.max(3, spots));
  let k = 0;
  while (k < DAMMANN.length - 1 && s > DAMMANN[k + 1].spots) k++;
  if (k === DAMMANN.length - 1) return DAMMANN[k].T.slice();
  const a = DAMMANN[k], b = DAMMANN[k + 1];
  const t = (s - a.spots) / (b.spots - a.spots);
  const n = Math.max(a.T.length, b.T.length);
  const out = [];
  for (let i = 0; i < n; i++) {
    const va = a.T[Math.min(i, a.T.length - 1)], vb = b.T[Math.min(i, b.T.length - 1)];
    out.push(va + (vb - va) * t);
  }
  return out;
}

function segValue(t, T) {
  let v = 0;
  for (let i = 0; i < T.length; i++) if (t >= T[i]) v ^= 1;
  return v;
}

export function phaseDammann(x, y, prm) {
  const c = Math.cos(prm.dammannRot), s = Math.sin(prm.dammannRot);
  const xr = x * c + y * s, yr = -x * s + y * c;
  const cell = prm.dammannSpacing * prm.p;            // cell period (pixels * p)
  const T = dammannTable(prm.dammannSpots);           // blended design table
  const vx = segValue(frac(xr / cell), T);
  const vy = segValue(frac(yr / cell), T);
  return Math.PI * (vx ^ vy);
}

// --- quantisation of the *design* phase ------------------------------------
export function quantise(phi, levels) {
  if (!levels || levels < 2) return phi % TAU;
  const step = TAU / levels;
  return step * Math.round(phi / step);
}

// --- full design phase (sum of weighted scenes) ----------------------------
export function designPhase(i, j, prm) {
  const N = prm.N;
  const x = (i - N / 2 + 0.5) * prm.p;
  const y = (j - N / 2 + 0.5) * prm.p;
  let phi = 0;
  for (let k = 0; k < SCENES.length; k++) {
    const w = prm.w[k];
    if (w <= 0) continue;
    phi += w * [
      phaseGrating, phaseZFL, phaseAxicon, phaseVortex, phaseDammann, phaseHolo,
    ][k](x, y, prm);
  }
  return phi;
}

// --- 6. Gerchberg-Saxton hologram -----------------------------------------
// runs on a reduced M x M design grid; the phase is upsampled to the mask grid
// by complex-exponential (phase-preserving) interpolation.
export function gsSolve(prm, M, iters, warm) {
  const t = Math.log2(M) | 0;
  let buf = warm && warm.length === 2 * M * M ? Float64Array.from(warm) : new Float64Array(2 * M * M);
  if (!(warm && warm.length === 2 * M * M)) {
    // quadratic-phase init to spread the spectrum and limit speckle
    for (let j = 0; j < M; j++)
      for (let i = 0; i < M; i++) {
        const x = (i - M / 2 + 0.5) / M, y = (j - M / 2 + 0.5) / M;
        const q = -22 * (x * x + y * y);
        const r = ihash(i | (j << 12)) / 4294967296;
        const ph = q + TAU * r;
        buf[(j * M + i) * 2] = Math.cos(ph);
        buf[(j * M + i) * 2 + 1] = Math.sin(ph);
      }
  }
  const tgt = gsTarget(prm, M);
  const target = tgt.amp, lit = tgt.lit;
  const scratch = new Float64Array(2 * M * M);   // ping-pong partner #1
  const work = new Float64Array(2 * M * M);
  const scratch2 = new Float64Array(2 * M * M);  // never aliases `F` below
  const scratch3 = new Float64Array(2 * M * M);
  // The mask this element can physically present is  exp(i*arg(buf)) : a pure
  // phase.  Every iteration therefore starts by forcing unit modulus.  The loop
  // finishes with the inverse transform, which leaves the mask plane "free"
  // (arbitrary amplitude) - so one more unit-modulus projection is applied
  // after the last iteration, and it is *that* field which is returned and
  // measured.  (Measuring the free field instead reports the constrained replay
  // we asked for, i.e. an error of ~0 - a very convincing lie.)
  for (let it = 0; it <= iters; it++) {
    for (let i = 0; i < M * M; i++) { work[2 * i] = buf[2 * i]; work[2 * i + 1] = buf[2 * i + 1]; }
    // amplitude constraint in the mask plane
    let amax = 0;
    for (let i = 0; i < M * M; i++) {
      const m = Math.hypot(work[2 * i], work[2 * i + 1]) || 1e-30;
      work[2 * i] /= m; work[2 * i + 1] /= m;
      amax = Math.max(amax, m);
    }
    if (it === iters) {   // final unit-modulus mask: hand it back
      for (let i = 0; i < 2 * M * M; i++) buf[i] = work[i];
      break;
    }
    const F = fft2d(work, scratch, M);              // far field
    for (let i = 0; i < M * M; i++) {               // amplitude constraint in replay
      const m = Math.hypot(F[2 * i], F[2 * i + 1]) || 1e-30;
      const a = target[i];
      F[2 * i] *= a / m; F[2 * i + 1] *= a / m;     // (norm: |F|<=1 so a is "relative amplitude")
    }
    // NOTE: fft2d ping-pongs between the two buffers it is given and may return
    // either, so the inverse step must get buffers that are neither `F` nor
    // each other.  (Passing F as both source and destination silently
    // corrupts the transform - this cost me one very confusing afternoon.)
    const g = fft2d(F, scratch2, M, { inverse: true });
    // fft2d(..., {inverse:true}) is the exact inverse of the 1/M^2-normalised
    // forward transform (verified against a naive DFT and by round-trip), so no
    // extra scaling belongs here.
    for (let i = 0; i < M * M; i++) { buf[2 * i] = g[2 * i]; buf[2 * i + 1] = g[2 * i + 1]; }
  }
  // RMS amplitude error.  Report it (a) over the whole replay window and
  // (b) restricted to the target pixels: (a) is dominated by the energy that
  // the phase-only mask physically cannot put into the image ("speckle
  // budget"), (b) is what actually says whether the picture is right.
  // `.slice()` matters: fft2d ping-pongs through the buffers you hand it, so
  // transforming `buf` in place would leave the mask we are about to return
  // holding a stale intermediate stage of the transform.
  const F = fft2d(buf.slice(), scratch3, M);
  let num = 0, den = 0, nw = 0, dw = 0, eImg = 0, eTot = 0, peak = 0;
  for (let i = 0; i < M * M; i++) {
    const I = F[2 * i] * F[2 * i] + F[2 * i + 1] * F[2 * i + 1];
    const a = Math.sqrt(I);
    const d = a - target[i];
    num += d * d; den += target[i] * target[i];
    eTot += I; peak = Math.max(peak, I);
    if (lit[i]) { nw += d * d; dw += target[i] * target[i]; eImg += I; }
  }
  return {
    field: buf, target, lit, M,
    rms: Math.sqrt(num / (den || 1)),          // whole window, normalised
    rmsWin: Math.sqrt(nw / (dw || 1)),         // target pixels only
    efficiency: eImg / (eTot || 1),            // energy landing on the image
    peak,
  };
}

// text -> target amplitude on the GS design grid. Bundled 5x7 font, so the
// result is identical on every machine (see tools/gen-font.mjs).
import { GS_CHARS, GS_FONT_B64 } from './font5x7.mjs';
let FONT = null;
export function fontData() {
  if (!FONT) FONT = Buffer.from(GS_FONT_B64, 'base64');
  return FONT;
}
export function gsTarget(prm, M, raw) {
  const A = new Float32Array(M * M);
  const lit = new Uint8Array(M * M);
  const text = (prm.text || '').toUpperCase();
  if (text.length === 0) {
    // bundled 1-bit glyph (ring + 4 spokes, 16x16) as fallback
    for (let y = 0; y < M; y++)
      for (let x = 0; x < M; x++) {
        const u = (2 * (x + 0.5)) / M - 1, v = (2 * (y + 0.5)) / M - 1;
        const r = Math.hypot(u, v);
        const on = (r > 0.38 && r < 0.46) || (r < 0.06 && ((x ^ y) & 6) === 0) ? 1 : 0;
        lit[y * M + x] = on;
        A[y * M + x] = on;
      }
    return { amp: normaliseTarget(A, M, prm, raw), lit };
  }
  // layout: 5x7 glyphs, 1 px gap, block of text centred in the window
  const font = fontData();
  const n = text.length;
  const gw = 6 * n - 1, gh = 7;
  const win = prm.holoWindow * M;                       // text width in the grid
  const scale = Math.max(1, Math.floor(win / gw));
  const ox = Math.round((M - gw * scale) / 2 + prm.holoOffX * M);
  const oy = Math.round((M - gh * scale) / 2 + prm.holoOffY * M);
  for (let c = 0; c < n; c++) {
    const idx = GS_CHARS.indexOf(text[c]);
    const gi = idx < 0 ? 0 : idx;
    for (let gx = 0; gx < 5; gx++) {
      const col = font[gi * 5 + gx];
      for (let gy = 0; gy < 7; gy++) {
        if (!(col & (1 << gy))) continue;
        const x0 = ox + (c * 6 + gx) * scale, y0 = oy + gy * scale;
        for (let dy = 0; dy < scale; dy++)
          for (let dx = 0; dx < scale; dx++) {
            const x = x0 + dx, y = y0 + dy;
            if (x >= 0 && x < M && y >= 0 && y < M) { A[y * M + x] = 1; lit[y * M + x] = 1; }
          }
      }
    }
  }
  return { amp: normaliseTarget(A, M, prm, raw), lit };
}

// Amplitude target for GS.
//
// A phase-only mask always radiates total replay energy sum|F|^2 = 1 (Parseval),
// so the target is scaled to carry holoEfficiency of that energy.  The
// background must be given a *non-zero* amplitude: with a target of exactly
// zero outside the picture, "field == 0 everywhere" satisfies both constraints
// perfectly and the iteration de-collapses straight into darkness.  A small
// uniform background level keeps the problem well posed (and is physically
// honest: a real DOE replay has speckle everywhere, it is never black).
export function normaliseTarget(A, M, prm, raw) {
  if (raw) return A;
  let nLit = 0;
  for (let i = 0; i < M * M; i++) if (A[i] > 0) nLit++;
  const n = M * M;
  const eff = prm.holoEfficiency || 0.8;
  // amplitudes, not fractions: the constraint is that the *sum* of the
  // intensities over the image pixels is `eff` of the total replay energy,
  // and the total is 1 (Parseval).  So aImg^2 * nLit = eff.
  const aImg = Math.sqrt(eff / Math.max(nLit, 1));
  const aBg = Math.sqrt((1 - eff) / Math.max(n - nLit, 1));
  for (let i = 0; i < n; i++) A[i] = A[i] > 0 ? aImg : aBg;
  return A;
}

// linear interpolation of the GS phase field, done on the unit circle so that
// 2*pi wraps do not create seams
export function holoPhaseAt(u, v, prm) {
  const M = prm.holoM;
  const f = prm.holoField;
  if (!f) return 0;
  const x = u * M - 0.5, y = v * M - 0.5;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = x - x0, fy = y - y0;
  let cr = 0, ci = 0;
  for (let b = 0; b < 4; b++) {
    const xi = Math.min(M - 1, Math.max(0, x0 + (b & 1)));
    const yi = Math.min(M - 1, Math.max(0, y0 + (b >> 1)));
    const w = ((b & 1) ? fx : 1 - fx) * ((b >> 1) ? fy : 1 - fy);
    cr += w * f[(yi * M + xi) * 2];
    ci += w * f[(yi * M + xi) * 2 + 1];
  }
  return Math.atan2(ci, cr);
}

export function phaseHolo(x, y, prm) {
  if (!prm.holoField) return 0;
  const u = x / (prm.N * prm.p) + 0.5;
  const v = y / (prm.N * prm.p) + 0.5;
  return holoPhaseAt(u, v, prm);
}

// --- replay field ----------------------------------------------------------
// returns [I_R, I_G, I_B] as Float32Array(N*N), normalised so that
// sum(I) == sum |f|^2 / N^2 == 1 for a unit modulus mask.
export function replay(prm) {
  const N = prm.N;
  const out = [];
  const a = new Float64Array(2 * N * N);
  const b = new Float64Array(2 * N * N);
  for (const lam of [prm.lamR, prm.lamG, prm.lamB]) {
    const r = prm.lamDesign / lam;      // fixed etch depth => phi_L = phi_d * Ld/L
    for (let j = 0; j < N; j++)
      for (let i = 0; i < N; i++) {
        const phi = prm.phaseFn(i, j) * r;
        a[(j * N + i) * 2] = Math.cos(phi);
        a[(j * N + i) * 2 + 1] = Math.sin(phi);
      }
    const F = fft2d(a, b, N);
    const I = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) I[i] = F[2 * i] * F[2 * i] + F[2 * i + 1] * F[2 * i + 1];
    out.push(I);
  }
  return out;
}

// screen sample of one channel: sx,sy are direction cosines (sin theta).
// Cutoff: sin^2(tx)+sin^2(ty) <= (lambda/2p)^2, which in bin units is exactly
// the inscribed circle radius N/2 of the FFT grid.
export function sampleReplay(I, N, lam, p, sx, sy) {
  const binU = (sx * N * p) / lam, binV = (sy * N * p) / lam;
  if (binU * binU + binV * binV > (N / 2) * (N / 2)) return 0;
  // natural-order FFT: bin b lives at array index b mod N, so negative
  // frequencies wrap to the tail of the array.
  const wrap = (b) => { const m = b % N; return m < 0 ? m + N : m; };
  const xf = wrap(binU) - 0.5, yf = wrap(binV) - 0.5;
  const x0 = Math.floor(xf), y0 = Math.floor(yf);
  const fx = xf - x0, fy = yf - y0;
  let acc = 0;
  for (let c = 0; c < 4; c++) {
    const xi = Math.min(N - 1, Math.max(0, x0 + (c & 1)));
    const yi = Math.min(N - 1, Math.max(0, y0 + (c >> 1)));
    acc += I[yi * N + xi] * ((c & 1) ? fx : 1 - fx) * ((c >> 1) ? fy : 1 - fy);
  }
  return acc;
}

// --- wavelength -> linear sRGB (Wyman/Sloan/Shirley multi-lobe fit) --------
function gauss(x, mu, s1, s2) {
  const t = (x - mu) * (x < mu ? 1 / s1 : 1 / s2);
  return Math.exp(-0.5 * t * t);
}
export function spectrumRGB(lamNm) {
  const X = 1.056 * gauss(lamNm, 599.8, 37.9, 31.0) + 0.362 * gauss(lamNm, 442.0, 16.0, 26.7)
    - 0.065 * gauss(lamNm, 501.1, 20.4, 26.2);
  const Y = 0.821 * gauss(lamNm, 568.8, 46.9, 40.5) + 0.286 * gauss(lamNm, 530.9, 16.3, 31.1);
  const Z = 1.217 * gauss(lamNm, 437.0, 11.8, 36.0) + 0.681 * gauss(lamNm, 459.0, 26.0, 13.8);
  let r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  let g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  let bl = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
  r = Math.max(0, r); g = Math.max(0, g); bl = Math.max(0, bl);
  const m = Math.max(r, g, bl) || 1;
  return [r / m, g / m, bl / m];
}

// --- fake SEM shading of the etched height map -----------------------------
export function heightOf(phiDesign, prm) {
  return (phiDesign / TAU) * (prm.lamDesign / (N_INDEX - 1)); // metres
}
// Fake e-beam shading.  slopeX/slopeY are the *dimensionless* physical surface
// slopes dz/dx, dz/dy (heights in metres over metres), so a 2*pi step of a
// 600 nm-pitch element really is a ~42 deg wall and shades accordingly.
export function semShade(slopeX, slopeY, grainSeed, exag = 1.0) {
  let nx = -slopeX * exag, ny = -slopeY * exag, nz = 1;
  const m = Math.hypot(nx, ny, nz);
  nx /= m; ny /= m; nz /= m;
  const lx = -0.42, ly = -0.55, lz = 0.72;   // tilt-shift e-beam detector
  const lam = Math.max(0, nx * lx + ny * ly + nz * lz);
  const edge = Math.min(1, Math.hypot(slopeX, slopeY) * 0.9);  // SE edge bloom
  let s = 0.20 + 0.62 * lam + 0.42 * edge;
  const grain = ihash(grainSeed) / 4294967296 - 0.5;
  s += grain * 0.05;
  return Math.min(1, Math.max(0, s));
}
