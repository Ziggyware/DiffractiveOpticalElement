/**
 * doe.js — synthesis of diffractive optical elements by phase retrieval.
 *
 * The design problem
 * ------------------
 * A phase-only DOE is a surface relief (or a liquid-crystal pattern) that
 * imposes a phase phi(x, y) on the illumination and nothing else:
 *
 *      U_out(x, y) = A_in(x, y) * exp( i * phi(x, y) )
 *
 * We want the *complex* field at a downstream plane to become a prescribed
 * irradiance I_target. This is the phase-retrieval problem, and the algorithms
 * below are the standard (and some non-standard) ways of solving it:
 *
 *   gerchbergSaxton            — alternating projections between the DOE plane
 *                                and the target plane (Fienup's error-reduction)
 *   weightedGerchbergSaxton    — Di Leonardo et al., Opt. Express 15, 1913
 *                                (2007): multiplicative weight feedback that
 *                                flattens the intensity of beam-shaping targets
 *   mixedRegionAmplitude        — Fienup's hybrid input-output (HIO) with
 *                                feedback, for escaping stagnation
 *   gerchbergSaxtonMultiPlane  — several target planes along the axis, i.e. a
 *                                focus stack from a *single* flat element
 *   fourierCgh                 — the DFT (far-field / lens focal plane) variant,
 *                                designed entirely in the frequency domain
 *
 * All loops are constrained to be *unitary* (band-limit disabled, energy
 * conserving propagators), so every reported efficiency is a real flux ratio
 * rather than an artefact of windowing.
 *
 * Phase quantisation
 * ------------------
 * Real DOEs have 2^b phase levels (multi-mask lithography) or are binary. The
 * quantisers here use Floyd–Steinberg error diffusion on the *wrapped* phase,
 * which preserves the diffraction efficiency far better than plain rounding.
 */

import { ComplexField, discMask, rectMask } from './field.js';
import {
  propagateAngularSpectrumInPlace,
  propagateFresnelFFT,
  lensPhase,
  wrapPhase,
} from './propagate.js';
import { fft2d, fftshift2d, ifftshift2d } from './fft.js';
import { maskedSum, regionStats, rmsError } from './metrics.js';

/** Options shared by every design routine, with sane defaults. */
export const DEFAULT_DESIGN = {
  lambda: 532e-9,
  z: 0.03, // screen distance [m]
  iterations: 60,
  seed: 12345,
  feedback: 0, // HIO feedback 0..1 (0 = pure Gerchberg-Saxton)
  beta: 0.9, // weight-update exponent for the weighted variant
  historyEvery: 1,
  randomRestarts: 1,
  onProgress: null,
};

/* ------------------------------------------------------------------ *
 * Deterministic RNG (xorshift32) so designs are perfectly reproducible
 * ------------------------------------------------------------------ */

export function makeRng(seed = 1) {
  let s = seed >>> 0 || 1;
  return function next() {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

/**
 * Random phase in (-pi, pi] — the generic initial guess for phase retrieval.
 * `smooth` > 1 low-pass filters it, which converges faster for smooth targets.
 */
export function randomPhase(nx, ny, seed = 1) {
  const rng = makeRng(seed);
  const p = new Float64Array(nx * ny);
  for (let k = 0; k < p.length; k++) p[k] = (rng() * 2 - 1) * Math.PI;
  return p;
}

/** Constant/zero phase — a good initial guess for on-axis beam shaping. */
export function flatPhase(nx, ny) {
  return new Float64Array(nx * ny);
}

/* ------------------------------------------------------------------ *
 * Basic phase primitives (thin element building blocks)
 * ------------------------------------------------------------------ */

/** Linear phase ramp — steers the beam by angle theta (tilt in x, y). */
export function prismPhase(nx, ny, dx, dy, lambda, thetaX = 0, thetaY = 0) {
  const p = new Float64Array(nx * ny);
  const k = (2 * Math.PI) / lambda;
  for (let j = 0; j < ny; j++) {
    const y = (j - ny / 2) * dy;
    for (let i = 0; i < nx; i++) {
      const x = (i - nx / 2) * dx;
      p[i + j * nx] = k * (Math.sin(thetaX) * x + Math.sin(thetaY) * y);
    }
  }
  return p;
}

/**
 * Axicon (conical) phase: a ring of zero order — Bessel-like beam generator.
 */
export function axiconPhase(nx, ny, dx, dy, lambda, angle = 1e-3) {
  const p = new Float64Array(nx * ny);
  const k = (2 * Math.PI) / lambda;
  for (let j = 0; j < ny; j++) {
    const y = (j - ny / 2) * dy;
    for (let i = 0; i < nx; i++) {
      const x = (i - nx / 2) * dx;
      p[i + j * nx] = -k * Math.sin(angle) * Math.hypot(x, y);
    }
  }
  return p;
}

/**
 * Helical (vortex) phase exp(i*l*phi): optical angular momentum, produces a
 * doughnut focus. l is the topological charge.
 */
export function vortexPhase(nx, ny, l = 1) {
  const p = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      p[i + j * nx] = l * Math.atan2(j - ny / 2, i - nx / 2);
    }
  }
  return p;
}

/**
 * Fresnel zone plate / kinoform lens: the wrapped quadratic phase of a lens
 * with focal length f. This is the simplest DOE that "projects" (it forms a
 * real focal spot at z = f), and it is the starting point for the projector
 * designs below because it puts the DC order where we want it.
 */
export function kinoformLens(nx, ny, dx, dy, lambda, f) {
  const field = new ComplexField(nx, ny, dx, dy);
  const p = lensPhase(field, lambda, f, { wrap: true });
  return p;
}

/** Wrapped modulo 2*pi of an arbitrary phase profile (kinoform wrapping). */
export function kinoform(phi, levels = 0) {
  const p = wrapPhase(Float64Array.from(phi));
  return levels > 1 ? quantizePhase(p, levels) : p;
}

/* ------------------------------------------------------------------ *
 * Quantisation
 * ------------------------------------------------------------------ */

/**
 * Round a wrapped phase to `levels` uniformly spaced levels in (-pi, pi].
 * levels = 4 -> 2-bit, levels = 8 -> 3-bit, ...
 */
export function quantizePhase(phi, levels = 4) {
  const out = new Float64Array(phi.length);
  const step = (2 * Math.PI) / levels;
  for (let k = 0; k < phi.length; k++) {
    out[k] = Math.round(phi[k] / step) * step;
  }
  return wrapPhase(out);
}

/**
 * Floyd–Steinberg error diffusion quantisation of a wrapped phase.
 *
 * Wrapping makes this non-obvious: the "error" is a phase difference, so it is
 * computed on the unit circle (via the wrapped difference) and pushed to
 * neighbours as a phase offset. The result keeps the local average phase (and
 * therefore the diffraction efficiency into the designed order) much closer to
 * the continuous solution than nearest-level rounding does.
 */
export function errorDiffusionQuantize(phi, levels = 4, { snake = true, wrapFirst = true } = {}) {
  const n = phi.length;
  const out = new Float64Array(n);
  const buf = wrapFirst ? wrapPhase(Float64Array.from(phi)) : Float64Array.from(phi);
  const step = (2 * Math.PI) / levels;
  const twoPi = 2 * Math.PI;

  // Non-power-of-two safety: we need a grid shape to know neighbours.
  // The caller passes a square power-of-two grid, so infer nx from the length.
  const nx = Math.round(Math.sqrt(n));
  const ny = n / nx;
  if (nx * ny !== n) throw new Error('errorDiffusionQuantize: expected a square grid');

  for (let j = 0; j < ny; j++) {
    const leftToRight = snake ? j % 2 === 0 : true;
    const iStart = leftToRight ? 0 : nx - 1;
    const iEnd = leftToRight ? nx : -1;
    const iStep = leftToRight ? 1 : -1;
    for (let i = iStart; i !== iEnd; i += iStep) {
      const k = i + j * nx;
      const v = buf[k];
      const q = Math.round(v / step) * step;
      out[k] = q;
      // wrapped error in (-pi, pi]
      let e = v - q;
      if (e > Math.PI) e -= twoPi;
      else if (e <= -Math.PI) e += twoPi;
      // Floyd–Steinberg kernel, mirrored for right-to-left rows
      const push = (di, dj, w) => {
        const ii = i + iStep * di;
        const jj = j + dj;
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) return;
        buf[ii + jj * nx] += e * w;
      };
      push(1, 0, 7 / 16);
      push(-1, 1, 3 / 16);
      push(0, 1, 5 / 16);
      push(1, 1, 1 / 16);
    }
  }
  return wrapPhase(out);
}

/**
 * Binary amplitude quantisation (Lee-style) of a complex field: encode a
 * complex sample by choosing on/off over a small cell so the *local mean*
 * reproduces the desired complex value. Useful when the element is an
 * amplitude mask (e.g. a printed transparency) rather than a phase element.
 */
export function leeEncode(field, cells = 2) {
  const { nx, ny } = field;
  const out = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j += cells) {
    for (let i = 0; i < nx; i += cells) {
      const k = i + j * nx;
      const amp = Math.hypot(field.re[k], field.im[k]);
      const ph = Math.atan2(field.im[k], field.re[k]);
      // fraction of "open" subcells encodes amplitude; their position encodes phase
      const open = Math.round(amp * cells * cells);
      const start = Math.floor(((ph + Math.PI) / (2 * Math.PI)) * cells * cells) % (cells * cells);
      for (let s = 0; s < cells * cells; s++) {
        const di = s % cells;
        const dj = (s / cells) | 0;
        const idx = i + di + (j + dj) * nx;
        if (idx >= nx * ny) continue;
        const rank = (s - start + cells * cells) % (cells * cells);
        out[idx] = rank < open ? 1 : 0;
      }
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Region-of-interest helpers
 * ------------------------------------------------------------------ */

/**
 * Soft mask that is 1 inside the target and falls to 0 over `edge` pixels —
 * used to keep the design energy inside a signal window without the ringing a
 * hard edge would cause.
 */
export function softRoi(nx, ny, { edge = 3, inset = 0 } = {}) {
  const m = new Float64Array(nx * ny);
  const cx = (nx - 1) / 2;
  const cy = (ny - 1) / 2;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const dxE = Math.min(i, nx - 1 - i) - inset;
      const dyE = Math.min(j, ny - 1 - j) - inset;
      const d = Math.min(dxE, dyE);
      let v = 1;
      if (d < 0) v = 0;
      else if (d < edge) v = 0.5 * (1 - Math.cos((Math.PI * d) / edge));
      m[i + j * nx] = v;
    }
  }
  return m;
}

/**
 * Ring (annular) mask, in normalised radius units of the half-window.
 */
export function ringMask(nx, ny, inner = 0.5, outer = 0.95, { soft = 0.02 } = {}) {
  const m = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    const v = (j - ny / 2) / (ny / 2);
    for (let i = 0; i < nx; i++) {
      const u = (i - nx / 2) / (nx / 2);
      const r = Math.hypot(u, v);
      let val = 0;
      const lo = inner + soft;
      const hi = outer - soft;
      if (r >= lo && r <= hi) val = 1;
      else if (r > inner && r < lo) val = 0.5 * (1 - Math.cos((Math.PI * (r - inner)) / soft));
      else if (r > hi && r < outer) val = 0.5 * (1 + Math.cos((Math.PI * (r - hi)) / soft));
      m[i + j * nx] = val;
    }
  }
  return m;
}

/* ------------------------------------------------------------------ *
 * Gerchberg–Saxton (and friends)
 * ------------------------------------------------------------------ */

/**
 * Alternating-projection phase retrieval between the DOE plane and one target
 * plane at distance z (angular-spectrum model, unitary propagation).
 *
 * @param {Float64Array} illumination amplitude of the illuminating wave on the
 *        DOE grid (0 outside the clear aperture)
 * @param {Float64Array} targetAmp desired *amplitude* at the target plane
 * @param {object} opts see DEFAULT_DESIGN, plus:
 *        - nx, ny, dx, dy        grid geometry
 *        - initPhase             Float64Array initial guess
 *        - roi                   Float64Array mask limiting where the target is enforced
 *        - weights               Float64Array multiplicative target weights
 *        - escape                number of iterations before feedback kicks in
 * @returns {{phase: Float64Array, error: number[], efficiency: number[],
 *            amplitude: Float64Array, intensity: Float64Array, iterations: number}}
 */
export function gerchbergSaxton(illumination, targetAmp, opts = {}) {
  const o = { ...DEFAULT_DESIGN, ...opts };
  const nx = o.nx ?? Math.round(Math.sqrt(illumination.length));
  const ny = o.ny ?? nx;
  const dx = o.dx ?? 8e-6;
  const dy = o.dy ?? dx;
  const { lambda, z, iterations } = o;
  const roi = o.roi ?? null;
  const weights = o.weights ?? null;

  const doe = new ComplexField(nx, ny, dx, dy);
  const plane = new ComplexField(nx, ny, dx, dy);
  const phase = o.initPhase ? Float64Array.from(o.initPhase) : flatPhase(nx, ny);

  const ampAtTarget = new Float64Array(nx * ny);
  const intensity = new Float64Array(nx * ny);
  const error = [];
  const efficiency = [];

  const targetPower = powerOfAmp(targetAmp, dx, dy, roi);

  for (let it = 0; it < iterations; it++) {
    // --- forward: DOE -> target plane ------------------------------------
    doe.setAmplitudePhase(illumination, phase);
    plane.copyFrom(doe);
    propagateAngularSpectrumInPlace(plane, lambda, z, { bandLimit: false });

    // --- measurement / cost ---------------------------------------------
    let errNum = 0;
    let errDen = 0;
    let inRoi = 0;
    let total = 0;
    for (let k = 0; k < plane.size; k++) {
      const a = Math.hypot(plane.re[k], plane.im[k]);
      const i2 = a * a;
      ampAtTarget[k] = a;
      intensity[k] = i2;
      total += i2;
      const m = roi ? roi[k] : 1;
      const t = targetAmp[k] * (weights ? weights[k] : 1);
      if (m > 0) {
        inRoi += i2 * m;
        if (t > 0 || i2 > 0) {
          const d = a * m - t * m;
          errNum += d * d;
          errDen += t * t;
        }
      }
    }
    error.push(errDen > 0 ? Math.sqrt(errNum / errDen) : 0);
    efficiency.push(total > 0 ? (inRoi * dx * dy) / (total * dx * dy) : 0);

    // --- apply the target-plane constraint ------------------------------
    const fb = it >= (o.escape ?? 0) ? o.feedback : 0;
    for (let k = 0; k < plane.size; k++) {
      const m = roi ? roi[k] : 1;
      const t = targetAmp[k] * (weights ? weights[k] : 1);
      const a = ampAtTarget[k];
      let newAmp;
      if (m > 0) {
        newAmp = t;
        if (fb > 0 && a > 0) newAmp = t + fb * (a - t); // HIO-style relaxation
      } else {
        newAmp = a * (1 - fb) + fb * a; // outside the ROI the field is left as-is
      }
      if (a > 0) {
        const s = newAmp / a;
        plane.re[k] *= s;
        plane.im[k] *= s;
      }
    }

    // --- back-propagate and re-impose the DOE-plane amplitude -----------
    propagateAngularSpectrumInPlace(plane, lambda, -z, { bandLimit: false });
    const nextPhase = new Float64Array(nx * ny);
    for (let k = 0; k < plane.size; k++) {
      const ur = plane.re[k];
      const ui = plane.im[k];
      // keep the measured phase; re-impose the illumination amplitude
      nextPhase[k] = Math.atan2(ui, ur);
    }
    // Optional smooth phase unwrap-free low-pass on the phase is deliberately
    // NOT applied: the wrapped phase is what the element implements.
    for (let k = 0; k < phase.length; k++) phase[k] = nextPhase[k];

    if (o.onProgress && (it % (o.progressEvery ?? 10) === 0 || it === iterations - 1)) {
      o.onProgress({ iteration: it, error: error[it], efficiency: efficiency[it] });
    }
  }

  // Final forward pass with the returned phase, for honest reporting.
  doe.setAmplitudePhase(illumination, phase);
  propagateAngularSpectrumInPlace(doe, lambda, z, { bandLimit: false });
  const finalIntensity = new Float64Array(nx * ny);
  const finalAmp = new Float64Array(nx * ny);
  for (let k = 0; k < doe.size; k++) {
    const a = Math.hypot(doe.re[k], doe.im[k]);
    finalAmp[k] = a;
    finalIntensity[k] = a * a;
  }

  return {
    phase,
    error,
    efficiency,
    amplitude: finalAmp,
    intensity: finalIntensity,
    iterations,
    field: doe,
  };
}

function powerOfAmp(amp, dx, dy, mask) {
  let s = 0;
  for (let k = 0; k < amp.length; k++) s += amp[k] * amp[k] * (mask ? mask[k] : 1);
  return s * dx * dy;
}

/**
 * Weighted Gerchberg–Saxton (Di Leonardo et al. 2007) for flat-top beam
 * shaping and spot-array generation.
 *
 * The trick: instead of enforcing the target amplitude directly, enforce
 * w_k * target_k, and adapt w multiplicatively using the measured intensity.
 * The weights redistribute the available light to the dark parts of the
 * target, which is what makes the result uniform instead of "target-shaped but
 * peaked in the middle".
 */
export function weightedGerchbergSaxton(illumination, targetAmp, opts = {}, baseFn = gerchbergSaxton) {
  const o = { ...DEFAULT_DESIGN, ...opts };
  const nx = o.nx ?? Math.round(Math.sqrt(illumination.length));
  const ny = o.ny ?? nx;
  const dx = o.dx ?? 8e-6;
  const dy = o.dy ?? dx;
  const weights = new Float64Array(nx * ny).fill(0);
  let active = 0;
  for (let k = 0; k < weights.length; k++) {
    if ((o.roi ? o.roi[k] : 1) > 0) {
      weights[k] = 1;
      active++;
    }
  }
  let result = null;
  // Best-iterate bookkeeping. The weighted objective is not monotone in the
  // *unweighted* error, and in geometries where the demanded power is not
  // achievable (a Fourier-type projection with a bounded element, for
  // instance) the weight feedback would otherwise be allowed to degenerate
  // into a hot spot. Keeping the best unweighted iterate makes the weighted
  // variant never worse than the unweighted one, which is what makes it safe
  // to offer as a default.
  let bestPhase = null;
  let bestRmse = Infinity;
  let bestSnapshot = null;
  const history = { error: [], efficiency: [] };
  for (let it = 0; it < o.iterations; it++) {
    result = baseFn(illumination, targetAmp, {
      ...o,
      weights,
      iterations: 1,
      initPhase: result ? result.phase : o.initPhase,
      onProgress: null,
    });
    {
      let num = 0;
      let den = 0;
      for (let k = 0; k < weights.length; k++) {
        if ((o.roi ? o.roi[k] : 1) <= 0) continue;
        const d = result.amplitude[k] - targetAmp[k];
        num += d * d;
        den += targetAmp[k] * targetAmp[k];
      }
      const rmse = den > 0 ? Math.sqrt(num / den) : 0;
      if (rmse < bestRmse) {
        bestRmse = rmse;
        bestPhase = Float64Array.from(result.phase);
        bestSnapshot = { amplitude: Float64Array.from(result.amplitude), intensity: Float64Array.from(result.intensity) };
      }
    }
    // Multiplicative weight update on the amplitude ratio. Weights only ever
    // *redistribute* the available light, so after each update they are
    // renormalised to keep the total demanded power equal to the target's own
    // power — otherwise the loop would drift into demanding light that does not
    // exist and the feedback would run away.
    const beta = o.beta ?? 0.9;
    let wsum = 0;
    let tsum = 0;
    for (let k = 0; k < weights.length; k++) {
      if ((o.roi ? o.roi[k] : 1) <= 0) continue;
      const measured = result.amplitude[k];
      const desired = targetAmp[k];
      const ratio = measured > 1e-30 ? desired / measured : 2;
      weights[k] *= Math.pow(Math.min(Math.max(ratio, 0.25), 4), beta);
      weights[k] = Math.min(Math.max(weights[k], 1e-4), 1e4); // bounded feedback
      wsum += weights[k] * weights[k] * desired * desired;
      tsum += desired * desired;
    }
    const norm = wsum > 0 ? Math.sqrt(tsum / wsum) : 1;
    for (let k = 0; k < weights.length; k++) weights[k] *= norm;
    history.error.push(bestRmse);
    history.efficiency.push(result.efficiency[result.efficiency.length - 1]);
    if (o.onProgress && (it % (o.progressEvery ?? 10) === 0 || it === o.iterations - 1)) {
      o.onProgress({ iteration: it, error: bestRmse, efficiency: result.efficiency[result.efficiency.length - 1] });
    }
  }
  if (bestPhase) {
    result = {
      ...result,
      iterations: o.iterations,
      error: history.error,
      efficiency: history.efficiency,
      phase: bestPhase,
      amplitude: bestSnapshot ? bestSnapshot.amplitude : result.amplitude,
      intensity: bestSnapshot ? bestSnapshot.intensity : result.intensity,
      weightedIterations: o.iterations,
    };
  }
  result.weights = weights;
  result.rmse = bestRmse;
  return result;
}

/**
 * Multi-plane phase retrieval: a single flat DOE that projects a *different*
 * image at each distance. This is the "focus stack from one element" trick and
 * is the clearest demonstration that a DOE can do true 3D projection: the same
 * element is sharp at several chosen planes and scrambled in between.
 *
 * @param {Float64Array} illumination
 * @param {{z: number, targetAmp: Float64Array, roi?: Float64Array, weight?: number}[]} planes
 */
export function gerchbergSaxtonMultiPlane(illumination, planes, opts = {}) {
  const o = { ...DEFAULT_DESIGN, ...opts };
  const nx = o.nx ?? Math.round(Math.sqrt(illumination.length));
  const ny = o.ny ?? nx;
  const dx = o.dx ?? 8e-6;
  const dy = o.dy ?? dx;
  const { lambda, iterations } = o;
  if (planes.length === 0) throw new Error('gerchbergSaxtonMultiPlane: no planes');

  const doe = new ComplexField(nx, ny, dx, dy);
  const plane = new ComplexField(nx, ny, dx, dy);
  const phase = o.initPhase ? Float64Array.from(o.initPhase) : flatPhase(nx, ny);
  const error = [];
  const efficiency = [];
  const perPlaneError = planes.map(() => []);

  for (let it = 0; it < iterations; it++) {
    let iterErr = 0;
    for (let p = 0; p < planes.length; p++) {
      const { z, targetAmp, roi = null, weight = 1 } = planes[p];
      doe.setAmplitudePhase(illumination, phase);
      plane.copyFrom(doe);
      propagateAngularSpectrumInPlace(plane, lambda, z, { bandLimit: false });

      let errNum = 0;
      let errDen = 0;
      let inRoi = 0;
      let total = 0;
      const amps = new Float64Array(plane.size);
      for (let k = 0; k < plane.size; k++) {
        const a = Math.hypot(plane.re[k], plane.im[k]);
        amps[k] = a;
        const i2 = a * a;
        total += i2;
        const m = roi ? roi[k] : 1;
        if (m > 0) {
          inRoi += i2 * m;
          const d = (a - targetAmp[k]) * m;
          errNum += d * d;
          errDen += targetAmp[k] * targetAmp[k];
        }
      }
      const e = errDen > 0 ? Math.sqrt(errNum / errDen) : 0;
      perPlaneError[p].push(e);
      iterErr += e * weight;

      for (let k = 0; k < plane.size; k++) {
        const m = roi ? roi[k] : 1;
        const t = m > 0 ? targetAmp[k] : amps[k];
        if (amps[k] > 0) {
          const s = t / amps[k];
          plane.re[k] *= s;
          plane.im[k] *= s;
        }
      }

      propagateAngularSpectrumInPlace(plane, lambda, -z, { bandLimit: false });
      for (let k = 0; k < plane.size; k++) {
        phase[k] = Math.atan2(plane.im[k], plane.re[k]);
      }
    }
    error.push(iterErr / planes.length);
    efficiency.push(1 - error[error.length - 1]);
    if (o.onProgress && (it % (o.progressEvery ?? 10) === 0 || it === iterations - 1)) {
      o.onProgress({ iteration: it, error: error[error.length - 1], perPlane: perPlaneError.map((a) => a[a.length - 1]) });
    }
  }

  return { phase, error, efficiency, perPlaneError, iterations };
}

/**
 * Fourier-domain (far-field) CGH design: the DOE plane and the reconstruction
 * plane are related by a single unitary DFT, i.e. a lens of focal length f (or
 * the Fraunhofer zone) sits between them.
 *
 * The screen coordinate of grid point (i, j) is
 *      u = (i - N/2) * lambda * f / (N * dx)
 * so the projected image has extent lambda*f/dx regardless of N — this is the
 * "classic" hologram, sharp at *every* distance (the pattern simply scales),
 * and adding a kinoform lens term converts it into a fixed-distance image.
 */
export function fourierCgh(illumination, targetAmp, opts = {}) {
  const o = { ...DEFAULT_DESIGN, ...opts };
  const nx = o.nx ?? Math.round(Math.sqrt(illumination.length));
  const ny = o.ny ?? nx;
  const dx = o.dx ?? 8e-6;
  const dy = o.dy ?? dx;
  const iterations = o.iterations;
  const roi = o.roi ?? null;
  const weights = o.weights ?? null;

  const re = new Float64Array(nx * ny);
  const im = new Float64Array(nx * ny);
  const phase = o.initPhase ? Float64Array.from(o.initPhase) : randomPhase(nx, ny, o.seed);
  const error = [];
  const efficiency = [];
  const intensity = new Float64Array(nx * ny);
  const amplitude = new Float64Array(nx * ny);

  const norm = 1 / Math.sqrt(nx * ny); // unitary DFT scaling

  for (let it = 0; it < iterations; it++) {
    // DOE plane: illumination amplitude x current phase
    for (let k = 0; k < re.length; k++) {
      const a = illumination[k];
      re[k] = a * Math.cos(phase[k]);
      im[k] = a * Math.sin(phase[k]);
    }
    // unitary forward DFT (centred spectrum)
    ifftshift2d(re, im, nx, ny);
    fft2d(re, im, nx, ny, false);
    fftshift2d(re, im, nx, ny, true);
    for (let k = 0; k < re.length; k++) {
      re[k] *= norm;
      im[k] *= norm;
    }

    let errNum = 0;
    let errDen = 0;
    let inRoi = 0;
    let total = 0;
    for (let k = 0; k < re.length; k++) {
      const a = Math.hypot(re[k], im[k]);
      amplitude[k] = a;
      intensity[k] = a * a;
      total += a * a;
      const m = roi ? roi[k] : 1;
      const t = targetAmp[k] * (weights ? weights[k] : 1);
      if (m > 0) {
        inRoi += a * a * m;
        const d = (a - t) * m;
        errNum += d * d;
        errDen += t * t;
      }
    }
    error.push(errDen > 0 ? Math.sqrt(errNum / errDen) : 0);
    efficiency.push(total > 0 ? inRoi / total : 0);

    for (let k = 0; k < re.length; k++) {
      const m = roi ? roi[k] : 1;
      const a = Math.hypot(re[k], im[k]);
      // Inside the signal region the modulus is forced to the target; outside it
      // the measured modulus is kept. Leaving the outside free is what keeps the
      // element from having to dump all the out-of-signal light back into the
      // image — the same convention the angular-spectrum loop uses.
      const t = m > 0 ? targetAmp[k] * (weights ? weights[k] : 1) : a;
      if (a > 0) {
        const s = t / a;
        re[k] *= s;
        im[k] *= s;
      } else {
        re[k] = t;
        im[k] = 0;
      }
    }

    // unitary inverse DFT back to the DOE plane
    ifftshift2d(re, im, nx, ny);
    fft2d(re, im, nx, ny, true);
    fftshift2d(re, im, nx, ny, true);
    for (let k = 0; k < re.length; k++) {
      re[k] *= Math.sqrt(nx * ny);
      im[k] *= Math.sqrt(nx * ny);
      phase[k] = Math.atan2(im[k], re[k]);
    }

    if (o.onProgress && (it % (o.progressEvery ?? 10) === 0 || it === iterations - 1)) {
      o.onProgress({ iteration: it, error: error[it], efficiency: efficiency[it] });
    }
  }

  return { phase, error, efficiency, amplitude, intensity, iterations, field: null };
}

/* ------------------------------------------------------------------ *
 * High-level element design
 * ------------------------------------------------------------------ */

/**
 * Design a projection DOE for a target irradiance at distance z.
 *
 * This is the entry point used by the projector pipeline and the web app. It
 * picks the algorithm, handles the illumination and return-of-investment
 * metrics, and optionally quantises the result to fabrication levels.
 *
 * @param {object} cfg
 *   nx, ny, dx            DOE grid geometry (pitch in m)
 *   lambda                wavelength
 *   z                     screen distance
 *   illumination          Float64Array amplitude at the DOE plane (default: disc)
 *   target                Float64Array target *irradiance* (default: uniform disc)
 *   mode                  'image-plane' | 'far-field' (Fourier CGH)
 *   algorithm             'gs' | 'wgs' | 'hio'
 *   levels                phase levels (0/1 = continuous)
 *   iterations
 *   roi                   mask where the target is enforced
 */
/**
 * Gaussian blur of a raster (separable, in-place-free). Projection targets
 * should be band-limited to what the optics can actually deliver; blurring the
 * target by ~1 resolution cell removes the unachievable high frequencies that
 * would otherwise show up as speckle in the projection.
 */
export function blurRaster(src, nx, ny, sigmaPx = 1) {
  if (sigmaPx <= 0) return Float64Array.from(src);
  const r = Math.max(1, Math.ceil(3 * sigmaPx));
  const kernel = new Float64Array(2 * r + 1);
  let ks = 0;
  for (let i = -r; i <= r; i++) {
    kernel[i + r] = Math.exp(-(i * i) / (2 * sigmaPx * sigmaPx));
    ks += kernel[i + r];
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ks;
  const tmp = new Float64Array(src.length);
  const out = new Float64Array(src.length);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      let acc = 0;
      for (let d = -r; d <= r; d++) {
        const ii = Math.min(nx - 1, Math.max(0, i + d));
        acc += src[ii + j * nx] * kernel[d + r];
      }
      tmp[i + j * nx] = acc;
    }
  }
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      let acc = 0;
      for (let d = -r; d <= r; d++) {
        const jj = Math.min(ny - 1, Math.max(0, j + d));
        acc += tmp[i + jj * nx] * kernel[d + r];
      }
      out[i + j * nx] = acc;
    }
  }
  return out;
}

export function designDOE(cfg) {
  const nx = cfg.nx ?? 256;
  const ny = cfg.ny ?? nx;
  const dx = cfg.dx ?? 8e-6;
  const dy = cfg.dy ?? dx;
  const lambda = cfg.lambda ?? 532e-9;
  const z = cfg.z ?? 0.03;
  const mode = cfg.mode ?? 'image-plane';
  // Algorithm default: the weighted variant is the right tool for proximity
  // (image-plane) beam shaping, where the demanded power is achievable; in a
  // Fourier-type geometry the demanded power is often not achievable and the
  // plain alternating-projection loop is the safer choice.
  const algorithm = cfg.algorithm && cfg.algorithm !== "auto" ? cfg.algorithm : mode === "far-field" ? "gs" : "wgs";
  const levels = cfg.levels ?? 0;
  const iterations = cfg.iterations ?? 60;

  const illumination =
    cfg.illumination ?? discMask(nx, ny, dx, dy, (Math.min(nx * dx, ny * dy) / 2) * 0.95, { soft: 2 * dx });
  const roi = cfg.roi ?? null;
  let targetIrradiance = cfg.target ?? defaultDiscTarget(nx, ny, 0.35);
  if (cfg.smooth > 0) targetIrradiance = blurRaster(targetIrradiance, nx, ny, cfg.smooth);
  const targetAmp = new Float64Array(nx * ny);
  for (let k = 0; k < targetAmp.length; k++) targetAmp[k] = Math.sqrt(Math.max(0, targetIrradiance[k]));

  // Match the demanded power to the available power so the error metric is
  // meaningful: scale the target so its total power equals the input power.
  let inPower = 0;
  for (let k = 0; k < illumination.length; k++) inPower += illumination[k] * illumination[k];
  let tPower = 0;
  for (let k = 0; k < targetAmp.length; k++) tPower += targetAmp[k] * targetAmp[k];
  if (tPower > 0) {
    const s = Math.sqrt(inPower / tPower);
    for (let k = 0; k < targetAmp.length; k++) targetAmp[k] *= s;
  }

  // Initial guess. A random phase is the better default for high-contrast
  // image targets: it spreads the illumination over the whole reconstruction
  // window instead of dumping most of the light into the undiffracted order.
  let initPhase = cfg.initPhase;
  if (!initPhase) {
    switch (cfg.init ?? 'random') {
      case 'flat':
        initPhase = flatPhase(nx, ny);
        break;
      case 'lens':
        initPhase = kinoformLens(nx, ny, dx, dy, lambda, z);
        break;
      case 'random':
      default:
        initPhase = randomPhase(nx, ny, cfg.seed ?? 12345);
        break;
    }
  }

  const common = {
    nx,
    ny,
    dx,
    dy,
    lambda,
    z,
    roi,
    iterations,
    seed: cfg.seed ?? 12345,
    feedback: algorithm === 'hio' ? (cfg.feedback ?? 0.8) : 0,
    escape: cfg.escape ?? Math.round(iterations * 0.2),
    beta: cfg.beta ?? 0.9,
    initPhase,
    onProgress: cfg.onProgress,
    progressEvery: cfg.progressEvery ?? 10,
  };

  let result;
  if (mode === 'far-field') {
    result = weightedWrapper(fourierCgh, illumination, targetAmp, common, algorithm);
    result.mode = 'far-field';
  } else {
    result = weightedWrapper(gerchbergSaxton, illumination, targetAmp, common, algorithm);
    result.mode = 'image-plane';
  }

  result.algorithm = algorithm;
  if (levels > 1) {
    result.phaseContinuous = Float64Array.from(result.phase);
    result.phase = errorDiffusionQuantize(result.phase, levels);
    result.levels = levels;
    // re-simulate with the quantised element so metrics are honest
    result.quantized = true;
  }

  result.illumination = illumination;
  result.targetAmp = targetAmp;
  result.roi = roi;
  result.lambda = lambda;
  result.z = z;
  result.nx = nx;
  result.ny = ny;
  result.dx = dx;
  result.dy = dy;
  return result;
}

function weightedWrapper(fn, illumination, targetAmp, common, algorithm) {
  if (algorithm === 'wgs') {
    // The weighted variant iterates whichever base algorithm the caller chose
    // (angular-spectrum GS or the Fourier-domain CGH loop).
    return weightedGerchbergSaxton(illumination, targetAmp, common, fn);
  }
  return fn(illumination, targetAmp, common);
}

function defaultDiscTarget(nx, ny, radiusFraction) {
  const t = new Float64Array(nx * ny);
  const r = radiusFraction * Math.min(nx, ny) * 0.5;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const d = Math.hypot(i - (nx - 1) / 2, j - (ny - 1) / 2);
      t[i + j * nx] = d <= r ? 1 : 0;
    }
  }
  return t;
}

/**
 * Simulate a designed element end-to-end: illuminate it, propagate to z, and
 * report the irradiance plane. `quantizedPhase` overrides the stored phase so
 * you can compare "ideal" vs "as fabricated".
 */
export function simulateDOE(result, { phase = null, z = null, lambda = null, bandLimit = false } = {}) {
  const { nx, ny, dx, dy } = result;
  const lam = lambda ?? result.lambda;
  const dist = z ?? result.z;
  const field = new ComplexField(nx, ny, dx, dy);
  field.setAmplitudePhase(result.illumination, phase ?? result.phase);
  propagateAngularSpectrumInPlace(field, lam, dist, { bandLimit });
  return field;
}

/**
 * Aperture-efficiency bookkeeping for a designed element: incident power,
 * power through the clear aperture, and the fraction of light that ends up in
 * the signal window at the design plane.
 */
export function efficiencyBudget(result, { phase = null, bandLimit = false } = {}) {
  const field = simulateDOE(result, { phase, bandLimit });
  const intensity = field.intensity();
  const { nx, ny, dx, dy } = result;
  let incident = 0;
  let aperture = 0;
  for (let k = 0; k < result.illumination.length; k++) {
    incident += 1; // illumination is normalised to unit peak
    aperture += result.illumination[k] * result.illumination[k];
  }
  const atScreen = maskedSum(intensity, null) * dx * dy;
  const inTarget = result.roi ? maskedSum(intensity, result.roi) * dx * dy : atScreen;
  return {
    powerAtScreen: atScreen,
    powerInSignal: inTarget,
    signalFraction: atScreen > 0 ? inTarget / atScreen : 0,
    roiStats: result.roi ? regionStats(intensity, result.roi) : regionStats(intensity, null),
    rms: result.roi ? rmsError(intensity, square(result.targetAmp), result.roi) : 0,
  };
}

function square(a) {
  const out = new Float64Array(a.length);
  for (let k = 0; k < a.length; k++) out[k] = a[k] * a[k];
  return out;
}

/** Convenience: export the phase as a fabrication-ready height map [m]. */
export function phaseToHeight(phase, lambda, n = 1.52) {
  const h = new Float64Array(phase.length);
  const s = lambda / (2 * Math.PI * (n - 1));
  const step = lambda / (n - 1); // one full 2*pi of phase
  for (let k = 0; k < phase.length; k++) {
    // an etch depth must be a non-negative fraction of a wave of retardation
    h[k] = (((phase[k] * s) % step) + step) % step;
  }
  return h;
}

/** Convenience: build the illumination profile from a named source. */
export function illuminationProfile(type, nx, ny, dx, dy, opts = {}) {
  const radius = opts.radius ?? (Math.min(nx * dx, ny * dy) / 2) * 0.95;
  const ap = discMask(nx, ny, dx, dy, radius, { soft: opts.soft ?? 2 * dx });
  if (type === 'tophat') return ap;
  if (type === 'gaussian') {
    const f = new ComplexField(nx, ny, dx, dy);
    const out = new Float64Array(nx * ny);
    const w = opts.waist ?? radius * 0.6;
    for (let j = 0; j < ny; j++) {
      const y = f.y(j);
      for (let i = 0; i < nx; i++) {
        const x = f.x(i);
        out[i + j * nx] = ap[i + j * nx] * Math.exp(-(x * x + y * y) / (w * w));
      }
    }
    return out;
  }
  if (type === 'annulus') return ringMask(nx, ny, opts.inner ?? 0.6, opts.outer ?? 0.98, { soft: 0.02 });
  if (type === 'rect') return rectMask(nx, ny, dx, dy, opts.width ?? nx * dx * 0.9, opts.height ?? ny * dy * 0.9, { soft: 2 * dx });
  // default: plane wave truncated by the aperture
  return ap;
}
