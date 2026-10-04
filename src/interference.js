/**
 * interference.js — coherent superposition, partial coherence, and holographic
 * recording/reconstruction.
 *
 * Everything a diffractive projector does is interference. The point of this
 * module is to make the *complex* nature of that interference explicit:
 *
 *   - coherent sums add complex amplitudes:   U = sum_k a_k e^{i phi_k}
 *   - incoherent sums add intensities:        I = sum_k |a_k|^2
 *   - partial coherence interpolates between the two: a set of mutually
 *     incoherent *modes*, each of which interferes with itself. This is the
 *     right model for an LED/ multimode source, and it is what turns a crisp
 *     simulated projection into an honest one (speckle washes out, contrast
 *     drops, uniformity improves).
 *
 * Also included: classical two/multi-beam fringe synthesis, Zernike and
 * Laguerre–Gauss/Hermite–Gauss mode generators (the standard aberration and
 * mode bases used to test optical elements), speckle with a controllable
 * correlation length, phase unwrapping, and a full hologram record/reconstruct
 * pair (the interference-based route to a CGH — the way holograms were
 * originally made, and still the way optically-recorded CGHs are produced).
 */

import { ComplexField } from './field.js';
import { propagateAngularSpectrumInPlace, wrapPhase } from './propagate.js';
import { fft2d, fftshift2d, ifftshift2d } from './fft.js';
import { makeRng } from './doe.js';
import { visibility } from './metrics.js';

/* ------------------------------------------------------------------ *
 * Superposition
 * ------------------------------------------------------------------ */

/** Coherent sum of complex fields: U = Σ U_k. */
export function superpose(fields) {
  return ComplexField.sum(fields);
}

/**
 * Incoherent sum of fields, returned as a complex field whose *modulus* equals
 * sqrt(Σ|U_k|²) and whose phase is 0. Use this when the sources do not share a
 * common phase reference.
 */
export function incoherentSum(fields) {
  return ComplexField.sumIncoherent(fields);
}

/**
 * Partial coherence: mutually incoherent modes with given powers. Each mode is
 * propagated independently and the resulting intensities are added — the
 * standard modal (Wolf) description.
 *
 * @param {ComplexField[]} modes
 * @param {number[]} [weights] relative powers (default all 1)
 * @returns {{intensity: Float64Array, coherence: number, modeCount: number}}
 */
export function modalIntensity(modes, weights = null) {
  const n = modes[0].size;
  const intensity = new Float64Array(n);
  let wsum = 0;
  for (let m = 0; m < modes.length; m++) {
    const w = weights ? weights[m] : 1;
    wsum += w;
    const { re, im } = modes[m];
    for (let k = 0; k < n; k++) intensity[k] += w * (re[k] * re[k] + im[k] * im[k]);
  }
  if (wsum > 0) for (let k = 0; k < n; k++) intensity[k] /= wsum;
  return { intensity, coherence: modes.length > 1 ? 0 : 1, modeCount: modes.length };
}

/**
 * Build the transverse modes of a partially coherent source: a Gaussian
 * envelope times a random phase screen with correlation length `sigma`
 * (approximated by a smoothed random phase), then average intensities.
 *
 * This is the cheap and remarkably faithful way to model the speckle
 * reduction that real projectors get from a spatially incoherent source.
 */
export function partiallyCoherentModes(nx, ny, dx, dy, { modes = 8, sigma = 40e-6, waist = 1e-3, seed = 7 } = {}) {
  const out = [];
  const rng = makeRng(seed);
  const sigmaPix = Math.max(1, sigma / dx);
  for (let m = 0; m < modes; m++) {
    const f = new ComplexField(nx, ny, dx, dy);
    const ph = new Float64Array(nx * ny);
    for (let k = 0; k < ph.length; k++) ph[k] = (rng() * 2 - 1) * Math.PI;
    smoothInPlace(ph, nx, ny, sigmaPix);
    for (let j = 0; j < ny; j++) {
      const y = f.y(j);
      for (let i = 0; i < nx; i++) {
        const x = f.x(i);
        const k = i + j * nx;
        const a = Math.exp(-(x * x + y * y) / (waist * waist));
        f.re[k] = a * Math.cos(ph[k]);
        f.im[k] = a * Math.sin(ph[k]);
      }
    }
    out.push(f);
  }
  return out;
}

/** Separable box-blur (used to shape the correlation length of random screens). */
function smoothInPlace(arr, nx, ny, radius) {
  const r = Math.max(1, Math.round(radius));
  const tmp = new Float64Array(arr.length);
  const norm = 1 / (2 * r + 1);
  for (let j = 0; j < ny; j++) {
    let acc = 0;
    for (let i = -r; i <= r; i++) acc += arr[(((i % nx) + nx) % nx) + j * nx];
    for (let i = 0; i < nx; i++) {
      tmp[i + j * nx] = acc * norm;
      acc -= arr[(((i - r) % nx) + nx) % nx + j * nx];
      acc += arr[(((i + r + 1) % nx) + nx) % nx + j * nx];
    }
  }
  for (let i = 0; i < nx; i++) {
    let acc = 0;
    for (let j = -r; j <= r; j++) acc += tmp[i + (((j % ny) + ny) % ny) * nx];
    for (let j = 0; j < ny; j++) {
      arr[i + j * nx] = acc * norm;
      acc -= tmp[i + ((((j - r) % ny) + ny) % ny) * nx];
      acc += tmp[i + ((((j + r + 1) % ny) + ny) % ny) * nx];
    }
  }
  return arr;
}

/* ------------------------------------------------------------------ *
 * Classical interference patterns
 * ------------------------------------------------------------------ */

/**
 * Two-beam interference of plane waves with wavevectors (angles) and phases.
 * Returns the complex field plus the analytic intensity, so you can check the
 * simulation against the textbook formula
 *
 *    I = I1 + I2 + 2 sqrt(I1 I2) cos(delta)  ,  V = 2 sqrt(I1 I2)/(I1 + I2)
 *
 * @param {{thetaX?: number, thetaY?: number, phase?: number, amplitude?: number}[]} beams
 */
export function multipleBeamInterference(nx, ny, dx, dy, lambda, beams) {
  const field = new ComplexField(nx, ny, dx, dy);
  const k = (2 * Math.PI) / lambda;
  for (let b = 0; b < beams.length; b++) {
    const { thetaX = 0, thetaY = 0, phase = 0, amplitude = 1 } = beams[b];
    for (let j = 0; j < ny; j++) {
      const y = field.y(j);
      for (let i = 0; i < nx; i++) {
        const x = field.x(i);
        const ph = k * (Math.sin(thetaX) * x + Math.sin(thetaY) * y) + phase;
        // accumulate (2 - phase)/pi to keep the synthetic phase bounded, then
        // rotate into the field: this is a coherent sum of the beams.
        const idx = i + j * nx;
        field.re[idx] += amplitude * Math.cos(ph);
        field.im[idx] += amplitude * Math.sin(ph);
      }
    }
  }
  const intensity = field.intensity();
  return {
    field,
    intensity,
    visibility: visibility(intensity),
    analyticVisibility: analyticTwoBeamVisibility(beams),
  };
}

function analyticTwoBeamVisibility(beams) {
  if (beams.length !== 2) return null;
  const a1 = beams[0].amplitude ?? 1;
  const a2 = beams[1].amplitude ?? 1;
  return (2 * a1 * a2) / (a1 * a1 + a2 * a2);
}

/**
 * Fringe period of two plane waves interfering at half-angle theta (radians):
 * Lambda = lambda / (2 sin(theta/2)). Used by the tests to validate the
 * sampled interference against the analytic fringe spacing.
 */
export function fringePeriod(lambda, theta) {
  return lambda / (2 * Math.sin(theta / 2));
}

/** N-beam grating: beams arranged on a cone of half-angle theta, N-fold symmetric. */
export function nBeamInterference(nx, ny, dx, dy, lambda, { n = 6, theta = 0.02, phase = 0 } = {}) {
  const beams = [];
  for (let b = 0; b < n; b++) {
    const a = (2 * Math.PI * b) / n;
    beams.push({ thetaX: theta * Math.cos(a), thetaY: theta * Math.sin(a), phase: phase * b, amplitude: 1 });
  }
  beams.push({ thetaX: 0, thetaY: 0, phase: 0, amplitude: 0.2 });
  return multipleBeamInterference(nx, ny, dx, dy, lambda, beams);
}

/* ------------------------------------------------------------------ *
 * Mode bases (useful to build and to diagnose complex fields)
 * ------------------------------------------------------------------ */

/** Zernike polynomial Z_n^m on the unit disc (Noll-normalised, |m| <= n). */
export function zernikePhase(nx, ny, n, m, amplitude = 1) {
  const out = new Float64Array(nx * ny);
  const absM = Math.abs(m);
  if ((n - absM) % 2 !== 0) return out;
  for (let j = 0; j < ny; j++) {
    const v = (j - ny / 2) / (ny / 2);
    for (let i = 0; i < nx; i++) {
      const u = (i - nx / 2) / (nx / 2);
      const r = Math.hypot(u, v);
      if (r > 1) continue;
      const th = Math.atan2(v, u);
      let radial = 0;
      for (let s = 0; s <= (n - absM) / 2; s++) {
        const c =
          ((s % 2 === 0 ? 1 : -1) * factorial(n - s)) /
          (factorial(s) * factorial((n + absM) / 2 - s) * factorial((n - absM) / 2 - s));
        radial += c * Math.pow(r, n - 2 * s);
      }
      const ang = m >= 0 ? Math.cos(m * th) : Math.sin(-m * th);
      out[i + j * nx] = amplitude * radial * ang;
    }
  }
  return out;
}

function factorial(x) {
  let r = 1;
  for (let i = 2; i <= x; i++) r *= i;
  return r;
}

/** Laguerre–Gauss mode LG(p, l) as a complex field with waist w0. */
export function lgMode(nx, ny, dx, dy, { p = 0, l = 0, w0 = 300e-6, amplitude = 1 } = {}) {
  const f = new ComplexField(nx, ny, dx, dy);
  for (let j = 0; j < ny; j++) {
    const y = f.y(j);
    for (let i = 0; i < nx; i++) {
      const x = f.x(i);
      const r = Math.hypot(x, y);
      const th = Math.atan2(y, x);
      const rho = (2 * r * r) / (w0 * w0);
      const lag = laguerre(p, Math.abs(l), rho);
      const a = amplitude * Math.pow(r / w0, Math.abs(l)) * lag * Math.exp(-(r * r) / (w0 * w0));
      const ph = l * th;
      const k = i + j * nx;
      f.re[k] = a * Math.cos(ph);
      f.im[k] = a * Math.sin(ph);
    }
  }
  return f;
}

function laguerre(p, l, x) {
  // L_p^l(x) via the standard recurrence
  let l0 = 1;
  let l1 = 1 + l - x;
  if (p === 0) return l0;
  if (p === 1) return l1;
  for (let k = 2; k <= p; k++) {
    const l2 = ((2 * k - 1 + l - x) * l1 - (k - 1 + l) * l0) / k;
    l0 = l1;
    l1 = l2;
  }
  return l1;
}

/** Hermite–Gauss mode HG(m, n). */
export function hgMode(nx, ny, dx, dy, { m = 0, n = 0, w0 = 300e-6, amplitude = 1 } = {}) {
  const f = new ComplexField(nx, ny, dx, dy);
  for (let j = 0; j < ny; j++) {
    const y = f.y(j);
    for (let i = 0; i < nx; i++) {
      const x = f.x(i);
      const a =
        amplitude *
        hermite(m, (Math.SQRT2 * x) / w0) *
        hermite(n, (Math.SQRT2 * y) / w0) *
        Math.exp(-(x * x + y * y) / (w0 * w0));
      f.re[i + j * nx] = a;
    }
  }
  return f;
}

function hermite(n, x) {
  let h0 = 1;
  let h1 = 2 * x;
  if (n === 0) return h0;
  if (n === 1) return h1;
  for (let k = 2; k <= n; k++) {
    const h2 = 2 * x * h1 - 2 * (k - 1) * h0;
    h0 = h1;
    h1 = h2;
  }
  return h1;
}

/** Speckle field: random phase with correlation length sigma (fully developed
 * speckle after propagation). */
export function speckleField(nx, ny, dx, dy, { sigma = 20e-6, seed = 3, phaseOnly = false } = {}) {
  const f = new ComplexField(nx, ny, dx, dy);
  const rng = makeRng(seed);
  const ph = new Float64Array(nx * ny);
  for (let k = 0; k < ph.length; k++) ph[k] = (rng() * 2 - 1) * Math.PI;
  // A correlation length of one pixel or less means no smoothing at all:
  // independent phases are exactly what a ground-glass screen looks like, and
  // they are what produce a fully developed speckle pattern (contrast -> 1) in
  // the far field. Smoothing is only needed to *reduce* the contrast.
  // Only smooth when the requested correlation length genuinely spans several
  // pixels; at a pixel or so the screen behaves as an independent-scatterer
  // diffuser, which is the case that gives fully developed speckle.
  const radius = Math.round(sigma / dx);
  if (radius >= 2) smoothInPlace(ph, nx, ny, radius);
  for (let k = 0; k < f.size; k++) {
    f.re[k] = Math.cos(ph[k]);
    f.im[k] = Math.sin(ph[k]);
  }
  if (phaseOnly) {
    for (let k = 0; k < f.size; k++) {
      f.re[k] = 1;
      f.im[k] = 0;
    }
    f.multiplyPhase(ph);
  }
  return f;
}

/* ------------------------------------------------------------------ *
 * Holographic recording (interference route to a CGH)
 * ------------------------------------------------------------------ */

/**
 * Record a hologram by interfering an object wave with a reference wave — the
 * Gabor/Leith–Upatnieks geometry. The recorded transmittance of the plate is
 *
 *    t = |O + R|^2 = |O|^2 + |R|^2 + O R* + O* R
 *
 * and the two cross terms are exactly the interference fringes carrying the
 * object information. We return the intensity (the raw recording), plus
 * amplitude/phase-only encodings of it, because those are what a
 * manufacturable element can actually implement.
 *
 * @param {ComplexField} object
 * @param {ComplexField} reference
 * @param {{encode?: 'raw'|'phase'|'binary', levels?: number}} opts
 */
export function recordHologram(object, reference, opts = {}) {
  const { encode = 'raw', levels = 0 } = opts;
  const n = object.size;
  if (reference.size !== n) throw new Error('recordHologram: shape mismatch');
  const sum = new ComplexField(object.nx, object.ny, object.dx, object.dy);
  for (let k = 0; k < n; k++) {
    sum.re[k] = object.re[k] + reference.re[k];
    sum.im[k] = object.im[k] + reference.im[k];
  }
  const recorded = sum.intensity();
  let norm = 0;
  for (let k = 0; k < n; k++) if (recorded[k] > norm) norm = recorded[k];
  if (norm > 0) for (let k = 0; k < n; k++) recorded[k] /= norm;

  const plate = new ComplexField(object.nx, object.ny, object.dx, object.dy);
  if (encode === 'raw') {
    for (let k = 0; k < n; k++) plate.re[k] = recorded[k];
  } else if (encode === 'binary') {
    for (let k = 0; k < n; k++) plate.re[k] = recorded[k] > 0.5 ? 1 : 0;
  } else if (encode === 'phase') {
    // kinoform: keep only the phase of the recorded field but with the
    // reference curvature removed, which is what a lithographic DOE can make
    let ph = new Float64Array(n);
    for (let k = 0; k < n; k++) ph[k] = Math.atan2(sum.im[k], sum.re[k]);
    ph = wrapPhase(ph);
    for (let k = 0; k < n; k++) {
      plate.re[k] = Math.cos(ph[k]);
      plate.im[k] = Math.sin(ph[k]);
    }
  } else {
    throw new Error(`recordHologram: unknown encoding "${encode}"`);
  }
  if (levels > 1 && encode === 'phase') {
    const step = (2 * Math.PI) / levels;
    for (let k = 0; k < n; k++) {
      const ph = Math.round(Math.atan2(plate.im[k], plate.re[k]) / step) * step;
      plate.re[k] = Math.cos(ph);
      plate.im[k] = Math.sin(ph);
    }
  }
  return { plate, recorded, sum };
}

/**
 * Reconstruct a recorded hologram: illuminate the plate with the reference and
 * propagate. The virtual image forms at -z, the real (conjugate) image at +z;
 * both appear because a real-valued recording keeps both cross terms. That
 * twin-image behaviour is physics, not a bug — the double-plane option below
 * shows it explicitly.
 */
export function reconstructHologram(plate, reference, lambda, z, opts = {}) {
  const illum = new ComplexField(plate.nx, plate.ny, plate.dx, plate.dy);
  for (let k = 0; k < plate.size; k++) {
    illum.re[k] = plate.re[k] * reference.re[k] - plate.im[k] * reference.im[k];
    illum.im[k] = plate.re[k] * reference.im[k] + plate.im[k] * reference.re[k];
  }
  const plus = illum.clone();
  const minus = illum.clone();
  propagateAngularSpectrumInPlace(plus, lambda, z, { bandLimit: false, ...opts });
  propagateAngularSpectrumInPlace(minus, lambda, -z, { bandLimit: false, ...opts });
  return { real: plus, virtual: minus, illumination: illum };
}

/* ------------------------------------------------------------------ *
 * Phase analysis
 * ------------------------------------------------------------------ */

/**
 * Phase unwrapping using Itoh's method: unwrap every row, then every column,
 * then repeat (a couple of passes are usually enough for smooth fields, which
 * is what simulated DOE fields are). Good enough to visualise and to compare
 * against the designed (unwrapped) profile.
 */
export function unwrapPhase2d(wrapped, nx, ny, { passes = 2 } = {}) {
  const out = Float64Array.from(wrapped);
  for (let p = 0; p < passes; p++) {
    for (let j = 0; j < ny; j++) {
      let prev = out[j * nx];
      for (let i = 1; i < nx; i++) {
        const k = i + j * nx;
        const d = out[k] - out[k - 1];
        if (d > Math.PI) out[k] -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
        else if (d < -Math.PI) out[k] -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
        prev = out[k];
      }
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 1; j < ny; j++) {
        const k = i + j * nx;
        const d = out[k] - out[k - nx];
        if (d > Math.PI) out[k] -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
        else if (d < -Math.PI) out[k] -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      }
    }
  }
  return out;
}

/**
 * Interference of the discrete diffraction orders of an element with itself:
 * this is how a "multi-order" DOE projects a set of spots (a spot array
 * generator, the workhorse of structured-light projectors). Each order gets a
 * linear ramp and an amplitude weight; the sum is a single complex field that
 * can be compared with the design target.
 */
export function diffractionOrders(nx, ny, dx, dy, lambda, z, orders) {
  const field = new ComplexField(nx, ny, dx, dy);
  const k = (2 * Math.PI) / lambda;
  for (const { mx = 0, my = 0, amplitude = 1, phase = 0 } of orders) {
    const fx = mx / (nx * dx);
    const fy = my / (ny * dy);
    for (let j = 0; j < ny; j++) {
      const y = field.y(j);
      for (let i = 0; i < nx; i++) {
        const x = field.x(i);
        // A tilted plane wave of spatial frequency (fx, fy) deflects the beam by
        // sin(theta) = fx * lambda, so the spot it focuses to sits at
        // (fx*lambda*z, fy*lambda*z) on the screen. The phase across the element
        // is therefore k*sin(theta)*r = 2*pi*fx*x (the propagation distance z
        // cancels out of the element-plane phase: it only sets where the spot
        // lands, not the ramp that puts it there).
        const ph = k * lambda * (fx * x + fy * y) + phase;
        const idx = i + j * nx;
        field.re[idx] += amplitude * Math.cos(ph);
        field.im[idx] += amplitude * Math.sin(ph);
      }
    }
  }
  return field;
}

/**
 * Compare a simulated intensity plane with the analytic fringe field, returning
 * the fringe visibility and the dominant fringe frequency/period from the FFT
 * peak. An intensity is real and therefore has a Hermitian spectrum, so the
 * sign of the reported frequency is a mirror-image artefact; only its
 * magnitude is physical.
 */
export function fringeAnalysis(intensity, nx, ny, dx, dy) {
  const peak = (() => {
    const re = Float64Array.from(intensity);
    const im = new Float64Array(nx * ny);
    const mean = re.reduce((a, b) => a + b, 0) / re.length;
    for (let k = 0; k < re.length; k++) re[k] -= mean;
    fftshift2d(re, im, nx, ny, false);
    fft2d(re, im, nx, ny, false);
    fftshift2d(re, im, nx, ny, true);
    let best = 0;
    let bx = 0;
    let by = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if (i === nx / 2 && j === ny / 2) continue; // the mean is not a fringe
        const k = i + j * nx;
        const mag = re[k] * re[k] + im[k] * im[k];
        if (mag > best) {
          best = mag;
          bx = i;
          by = j;
        }
      }
    }
    const fx = (bx - nx / 2) / (nx * dx);
    const fy = (by - ny / 2) / (ny * dy);
    return { fx, fy, period: 1 / Math.hypot(fx, fy) };
  })();
  return { visibility: visibility(intensity), peak };
}
