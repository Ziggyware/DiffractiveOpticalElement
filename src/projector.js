/**
 * projector.js — the end-to-end "true projection" pipeline.
 *
 * What makes a projection *true* (as opposed to a decorative diffraction
 * pattern) is that the element is designed against a real optical model of the
 * whole system, and the result is judged in the projection plane:
 *
 *   source -> illumination profile -> clear aperture -> DOE phase -> free space
 *   -> screen irradiance -> comparison against the target image
 *
 * Two physically distinct projection regimes are supported, and they behave in
 * genuinely different ways (which the workbench lets you see side by side):
 *
 *   'image-plane'  a real image is formed at z. The image is sharp *only* near
 *                  z: defocus blurs it, exactly like a camera defocus. Designed
 *                  with the angular-spectrum model, energy conserving.
 *
 *   'far-field'    the element is a Fourier CGH: the image exists at infinity
 *                  and is therefore sharp at every distance, scaling linearly
 *                  with z (a "throw ratio" of lambda*z/... ). This is how
 *                  holographic projection heads actually work, and why they are
 *                  paired with a lens in practice.
 *
 * The pipeline reports radiometric metrics (efficiency, uniformity, contrast,
 * SNR, zero-order leakage), a defocus curve, and a focus stack, so every claim
 * about the projection can be checked against the model.
 */

import { ComplexField, discMask } from './field.js';
import { propagateAngularSpectrumInPlace, propagateFresnelFFT, focalPlaneSpectrum, lensPhase } from './propagate.js';
import { designDOE, gerchbergSaxtonMultiPlane, illuminationProfile, errorDiffusionQuantize } from './doe.js';
import {
  correlation,
  maskedSum,
  regionStats,
  rmsError,
  rmsRadius,
  spotUniformity,
  ssim,
  zeroOrderFraction,
  summarizeMetrics,
} from './metrics.js';

/* ------------------------------------------------------------------ *
 * 5x7 bitmap font — enough to project real text ("DOE", "ARENA", "532NM")
 * ------------------------------------------------------------------ */

const GLYPHS = {
  A: '01110/10001/10001/11111/10001/10001/10001',
  B: '11110/10001/10001/11110/10001/10001/11110',
  C: '01110/10001/10000/10000/10000/10001/01110',
  D: '11110/10001/10001/10001/10001/10001/11110',
  E: '11111/10000/10000/11110/10000/10000/11111',
  F: '11111/10000/10000/11110/10000/10000/10000',
  G: '01110/10001/10000/10111/10001/10001/01111',
  H: '10001/10001/10001/11111/10001/10001/10001',
  I: '01110/00100/00100/00100/00100/00100/01110',
  J: '00111/00010/00010/00010/00010/10010/01100',
  K: '10001/10010/10100/11000/10100/10010/10001',
  L: '10000/10000/10000/10000/10000/10000/11111',
  M: '10001/11011/10101/10101/10001/10001/10001',
  N: '10001/11001/10101/10011/10001/10001/10001',
  O: '01110/10001/10001/10001/10001/10001/01110',
  P: '11110/10001/10001/11110/10000/10000/10000',
  Q: '01110/10001/10001/10001/10101/10010/01101',
  R: '11110/10001/10001/11110/10100/10010/10001',
  S: '01111/10000/10000/01110/00001/00001/11110',
  T: '11111/00100/00100/00100/00100/00100/00100',
  U: '10001/10001/10001/10001/10001/10001/01110',
  V: '10001/10001/10001/10001/10001/01010/00100',
  W: '10001/10001/10001/10101/10101/11011/10001',
  X: '10001/10001/01010/00100/01010/10001/10001',
  Y: '10001/10001/01010/00100/00100/00100/00100',
  Z: '11111/00001/00010/00100/01000/10000/11111',
  0: '01110/10001/10011/10101/11001/10001/01110',
  1: '00100/01100/00100/00100/00100/00100/01110',
  2: '01110/10001/00001/00010/00100/01000/11111',
  3: '11111/00010/00100/00010/00001/10001/01110',
  4: '00010/00110/01010/10010/11111/00010/00010',
  5: '11111/10000/11110/00001/00001/10001/01110',
  6: '00110/01000/10000/11110/10001/10001/01110',
  7: '11111/00001/00010/00100/01000/01000/01000',
  8: '01110/10001/10001/01110/10001/10001/01110',
  9: '01110/10001/10001/01111/00001/00010/01100',
  ' ': '00000/00000/00000/00000/00000/00000/00000',
  '-': '00000/00000/00000/11111/00000/00000/00000',
  '.': '00000/00000/00000/00000/00000/01100/01100',
  ':': '00000/01100/01100/00000/01100/01100/00000',
  '!': '00100/00100/00100/00100/00100/00000/00100',
  '+': '00000/00100/00100/11111/00100/00100/00000',
  '/': '00001/00010/00010/00100/01000/01000/10000',
  '%': '11001/11010/00010/00100/01000/01011/10011',
  '*': '00000/10101/01110/11111/01110/10101/00000',
};

/** Render text into a 1-bit raster (values 0/1) using the 5x7 font. */
export function renderText(text, { scale = 8, spacing = 1 } = {}) {
  const chars = text.toUpperCase().split('');
  const glyphW = 5 * spacing + (5 - 5); // 5 columns plus 1 spacing column
  const width = chars.length * (5 + spacing) - spacing;
  const height = 7;
  const raster = new Float64Array(width * height);
  chars.forEach((ch, ci) => {
    const g = GLYPHS[ch] ?? GLYPHS[' '];
    const rows = g.split('/');
    for (let y = 0; y < 7; y++) {
      const row = rows[y];
      for (let x = 0; x < 5; x++) {
        if (row[x] === '1') raster[x + ci * (5 + spacing) + y * width] = 1;
      }
    }
  });
  // upscale by `scale` with nearest-neighbour (keeps glyph edges crisp)
  const w2 = width * scale;
  const h2 = height * scale;
  const out = new Float64Array(w2 * h2);
  for (let y = 0; y < h2; y++) {
    for (let x = 0; x < w2; x++) {
      out[x + y * w2] = raster[Math.floor(x / scale) + Math.floor(y / scale) * width];
    }
  }
  return { data: out, width: w2, height: h2 };
}

/** Bilinear resample of a scalar raster onto a new raster (aspect preserving). */
export function resampleRaster(src, sw, sh, dw, dh, { fit = 'contain', background = 0, gain = 1 } = {}) {
  const out = new Float64Array(dw * dh).fill(background);
  const s = fit === 'contain' ? Math.min(dw / sw, dh / sh) : Math.max(dw / sw, dh / sh);
  const tw = sw * s;
  const th = sh * s;
  const ox = (dw - tw) / 2;
  const oy = (dh - th) / 2;
  for (let y = 0; y < dh; y++) {
    const fy = (y - oy) / s;
    const y0 = Math.floor(fy);
    if (y0 < 0 || y0 >= sh) continue;
    const ty = fy - y0;
    const y1 = Math.min(y0 + 1, sh - 1);
    for (let x = 0; x < dw; x++) {
      const fx = (x - ox) / s;
      const x0 = Math.floor(fx);
      if (x0 < 0 || x0 >= sw) continue;
      const tx = fx - x0;
      const x1 = Math.min(x0 + 1, sw - 1);
      const v =
        src[x0 + y0 * sw] * (1 - tx) * (1 - ty) +
        src[x1 + y0 * sw] * tx * (1 - ty) +
        src[x0 + y1 * sw] * (1 - tx) * ty +
        src[x1 + y1 * sw] * tx * ty;
      out[x + y * dw] = v * gain;
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Targets
 * ------------------------------------------------------------------ */

/**
 * Build a projection target.
 *
 * @param {string} kind 'dots' | 'text' | 'image' | 'disc' | 'ring' | 'checker' | 'lines' | 'grid' | 'zone'
 * @param {number} nx
 * @param {number} ny
 * @param {object} opts kind specific
 * @returns {{irradiance: Float64Array, roi: Float64Array, name: string}}
 */
export function makeTarget(kind, nx, ny, opts = {}) {
  const irr = new Float64Array(nx * ny);
  const cx = (nx - 1) / 2;
  const cy = (ny - 1) / 2;
  let name = kind;

  if (kind === 'image') {
    const { raster, width, height, inset = 0.85, gamma = 1 } = opts;
    if (!raster) throw new Error('makeTarget("image"): pass {raster, width, height}');
    const dst = Math.round(Math.min(nx, ny) * inset);
    const res = resampleRaster(raster, width, height, nx, ny, { fit: 'contain' });
    // The resample already covers the window; `dst` only documents the intent.
    for (let k = 0; k < irr.length; k++) irr[k] = gamma === 1 ? res[k] : Math.pow(Math.max(0, res[k]), gamma);
    name = opts.name ?? 'image';
  } else if (kind === 'text') {
    const { text = 'DOE', scale = 6, spacing = 1 } = opts;
    const g = renderText(text, { scale, spacing });
    const res = resampleRaster(g.data, g.width, g.height, nx, ny, { fit: 'contain' });
    irr.set(res);
    name = `text:${text}`;
  } else if (kind === 'disc') {
    const r = (opts.radius ?? 0.35) * Math.min(nx, ny) * 0.5;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        irr[i + j * nx] = Math.hypot(i - cx, j - cy) <= r ? 1 : 0;
      }
    }
  } else if (kind === 'ring') {
    const r0 = (opts.inner ?? 0.25) * Math.min(nx, ny) * 0.5;
    const r1 = (opts.outer ?? 0.4) * Math.min(nx, ny) * 0.5;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const r = Math.hypot(i - cx, j - cy);
        irr[i + j * nx] = r >= r0 && r <= r1 ? 1 : 0;
      }
    }
  } else if (kind === 'dots' || kind === 'grid') {
    const cols = opts.cols ?? 8;
    const rows = opts.rows ?? 8;
    const pitchX = nx / cols;
    const pitchY = ny / rows;
    const sigma = (opts.sigma ?? 0.22) * Math.min(pitchX, pitchY);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const fy = ((j + 0.5) / pitchY) % 1 - 0.5;
        const fx = ((i + 0.5) / pitchX) % 1 - 0.5;
        const dxp = fx * pitchX;
        const dyp = fy * pitchY;
        let v = Math.exp(-(dxp * dxp + dyp * dyp) / (2 * sigma * sigma));
        if (opts.missing && opts.missing.some(([mi, mj]) => mi === ((i / pitchX) | 0) && mj === ((j / pitchY) | 0))) v = 0;
        irr[i + j * nx] = v;
      }
    }
    name = kind;
  } else if (kind === 'checker') {
    const cells = opts.cells ?? 8;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const u = Math.floor((i / nx) * cells);
        const v = Math.floor((j / ny) * cells);
        irr[i + j * nx] = (u + v) % 2 === 0 ? 1 : 0;
      }
    }
  } else if (kind === 'lines') {
    const period = opts.period ?? 12;
    const duty = opts.duty ?? 0.5;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        irr[i + j * nx] = ((i % period) / period) < duty ? 1 : 0;
      }
    }
  } else if (kind === 'zone') {
    const period = opts.period ?? 16;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const r = Math.hypot(i - cx, j - cy);
        irr[i + j * nx] = Math.sin((Math.PI * r * r) / period) > 0 ? 1 : 0;
      }
    }
  } else {
    throw new Error(`makeTarget: unknown kind "${kind}"`);
  }

  const roi = opts.roi ?? autoRoi(irr, nx, ny, { dilate: opts.dilate ?? 2, soft: opts.softEdge ?? 2 });
  const grid = kind === 'dots' || kind === 'grid' ? { cols: opts.cols ?? 8, rows: opts.rows ?? 8 } : null;
  return { irradiance: irr, roi, name, grid };
}

/**
 * Region of interest = a soft, slightly dilated mask around the target's
 * support. Restricting the phase-retrieval constraint to this region is what
 * lets the algorithm spend the light where the image actually is.
 */
export function autoRoi(irr, nx, ny, { dilate = 2, soft = 2, threshold = 0.05 } = {}) {
  let mx = 0;
  for (let k = 0; k < irr.length; k++) if (irr[k] > mx) mx = irr[k];
  const bin = new Float64Array(irr.length);
  for (let k = 0; k < irr.length; k++) bin[k] = irr[k] > threshold * mx ? 1 : 0;
  // separable dilation
  const tmp = new Float64Array(irr.length);
  const r = Math.max(1, Math.round(dilate));
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      let v = 0;
      for (let d = -r; d <= r && v === 0; d++) {
        const ii = i + d;
        if (ii < 0 || ii >= nx) continue;
        if (bin[ii + j * nx] > 0) v = 1;
      }
      tmp[i + j * nx] = v;
    }
  }
  const dil = new Float64Array(irr.length);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      let v = 0;
      for (let d = -r; d <= r && v === 0; d++) {
        const jj = j + d;
        if (jj < 0 || jj >= ny) continue;
        if (tmp[i + jj * nx] > 0) v = 1;
      }
      dil[i + j * nx] = v;
    }
  }
  // soft edge via a small box blur
  if (soft <= 0) return dil;
  const out = new Float64Array(irr.length);
  const s = Math.max(1, Math.round(soft));
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      let acc = 0;
      let n = 0;
      for (let dj = -s; dj <= s; dj++) {
        const jj = j + dj;
        if (jj < 0 || jj >= ny) continue;
        for (let di = -s; di <= s; di++) {
          const ii = i + di;
          if (ii < 0 || ii >= nx) continue;
          acc += dil[ii + jj * nx];
          n++;
        }
      }
      out[i + j * nx] = n > 0 ? acc / n : 0;
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * The projector
 * ------------------------------------------------------------------ */

export const DEFAULT_PROJECTOR = {
  // geometry
  n: 256, // design grid (power of two)
  pitch: 8e-6, // element pixel pitch [m]
  lambda: 532e-9,
  distance: 0.03, // DOE -> screen [m]
  aperture: 0.95, // clear aperture diameter as a fraction of the window
  apodization: 2, // soft edge width in pixels
  // source
  source: 'tophat', // 'tophat' | 'gaussian' | 'annulus' | 'plane'
  waist: 0.6, // gaussian waist as a fraction of the aperture radius
  inner: 0.6, // annulus source inner radius (fraction of aperture radius)
  outer: 0.98, // annulus source outer radius
  // design
  mode: 'image-plane', // 'image-plane' | 'far-field'
  algorithm: 'auto', // 'auto' | 'gs' | 'wgs' | 'hio'
  iterations: 60,
  levels: 0, // 0 = continuous phase, else 2/4/8/16 fabrication levels
  seed: 12345,
  quantize: false,
  // target
  target: { kind: 'text', text: 'DOE' },
};

export class ProjectorSystem {
  /** @param {object} cfg see DEFAULT_PROJECTOR */
  constructor(cfg = {}) {
    this.cfg = { ...DEFAULT_PROJECTOR, ...cfg, target: { ...DEFAULT_PROJECTOR.target, ...(cfg.target ?? {}) } };
    this.nx = this.cfg.n;
    this.ny = this.cfg.n;
    this.dx = this.cfg.pitch;
    this.dy = this.cfg.pitch;
    this.lambda = this.cfg.lambda;
    // Far-field mode is only mode-matched (screen window == element window)
    // when the throw equals the critical distance N*dx^2/lambda, so that is the
    // default there; any other distance simply changes the image scale. An
    // explicit distance always wins (including one inherited from the defaults
    // for the image-plane head).
    if (cfg.distance === undefined && this.cfg.mode === 'far-field') {
      this.cfg.distance = (this.cfg.n * this.cfg.pitch * this.cfg.pitch) / this.cfg.lambda;
    }
    this.z = this.cfg.distance;
    this.target = null;
    this.result = null;
    this.setTarget(this.cfg.target);
  }

  setTarget(spec) {
    if (spec && spec.irradiance) {
      this.target = {
        irradiance: spec.irradiance,
        roi: spec.roi ?? autoRoi(spec.irradiance, this.nx, this.ny),
        name: spec.name ?? 'custom',
      };
    } else {
      this.target = makeTarget(spec.kind, this.nx, this.ny, spec);
    }
    return this.target;
  }

  /** Mode-matched throw distance N*dx^2/lambda for this grid [m]. */
  matchDistance() {
    return (this.nx * this.dx * this.dx) / this.lambda;
  }

  /** Illuminating amplitude on the DOE grid. */
  illumination() {
    const radius = ((Math.min(this.nx * this.dx, this.ny * this.dy) / 2) * this.cfg.aperture) / 1;
    const opts = {
      radius,
      soft: this.cfg.apodization * this.dx,
      waist: radius * this.cfg.waist,
      inner: this.cfg.inner ?? 0.6,
      outer: this.cfg.outer ?? 0.98,
    };
    return illuminationProfile(this.cfg.source, this.nx, this.ny, this.dx, this.dy, opts);
  }

  /**
   * Derived optical geometry: what this element can and cannot do. These are
   * the numbers a real projector design is traded against.
   */
  geometry() {
    const n = this.nx;
    const p = this.dx;
    const D = 2 * ((n * p) / 2) * this.cfg.aperture;
    const thetaMax = Math.asin(Math.min(1, this.lambda / (2 * p)));
    const fourierWindow = (this.lambda * this.z) / p; // far-field extent (independent of n)
    const imageWindow = n * p; // image-plane extent (equals the element window)
    const window = this.cfg.mode === 'far-field' ? fourierWindow : imageWindow;
    const spot = (1.22 * this.lambda * this.z) / D; // diffraction-limited spot at the screen
    const cells = window / spot;
    return {
      gridN: n,
      pitchUm: p * 1e6,
      apertureMm: D * 1e3,
      wavelengthNm: this.lambda * 1e9,
      distanceMm: this.z * 1e3,
      mode: this.cfg.mode,
      screenWindowMm: window * 1e3,
      imageWindowMm: imageWindow * 1e3,
      fourierWindowMm: fourierWindow * 1e3,
      screenPitchUm: (this.cfg.mode === 'far-field' ? fourierWindow / n : p) * 1e6,
      maxDeflectionDeg: (thetaMax * 180) / Math.PI,
      diffractionSpotUm: spot * 1e6,
      resolvableSpotsPerAxis: cells,
      addressablePixels: Math.round(cells * cells),
      throwRatio: this.z / window,
      fNumber: this.z / D,
      fresnelCriticalMm: (((n * p * p) / this.lambda) * 1e3),
      undersampledChirp: this.z < (n * p * p) / this.lambda,
      pixelCount: n * n,
      phaseLevels: this.cfg.levels > 1 ? this.cfg.levels : 'continuous',
    };
  }

  /** Run the phase-retrieval design. */
  design(opts = {}) {
    const illumination = this.illumination();
    const result = designDOE({
      nx: this.nx,
      ny: this.ny,
      dx: this.dx,
      dy: this.dy,
      lambda: this.lambda,
      z: this.z,
      mode: this.cfg.mode,
      algorithm: this.cfg.algorithm,
      iterations: opts.iterations ?? this.cfg.iterations,
      levels: opts.levels ?? this.cfg.levels,
      seed: opts.seed ?? this.cfg.seed,
      illumination,
      target: this.target.irradiance,
      roi: this.target.roi,
      onProgress: opts.onProgress,
      progressEvery: opts.progressEvery ?? 10,
      beta: opts.beta ?? 0.9,
      escape: opts.escape,
      initPhase: opts.initPhase,
    });
    // Report the iteration count that was actually run (the weighted loop drives
    // the base algorithm one iteration at a time).
    result.iterations = opts.iterations ?? this.cfg.iterations;
    result.target = this.target;
    this.result = result;
    return result;
  }

  /**
   * Simulate the element at a plane distance `z` (defaults to the design
   * distance). `pad` zero-pads the element before propagating, which is what
   * you want when looking far from the design plane, where light legitimately
   * leaves the window and would otherwise wrap around.
   */
  simulate({ z = null, phase = null, bandLimit = false, pad = 1, mode = null } = {}) {
    if (!this.result) throw new Error('ProjectorSystem.simulate: call design() first');
    const dist = z ?? this.z;
    const useMode = mode ?? this.cfg.mode;
    const r = this.result;

    if (useMode === 'far-field') {
      // Holographic head: the element is followed by a projection lens of focal
      // length equal to the throw, so the screen sits in the lens focal plane
      // and the field there is the (unitary) Fourier transform of the element.
      if (!(dist > 0)) throw new Error('ProjectorSystem.simulate: far-field mode needs a positive throw');
      const focal = dist;
      const field = new ComplexField(this.nx, this.ny, this.dx, this.dy);
      field.setAmplitudePhase(r.illumination, phase ?? r.phase);
      return focalPlaneSpectrum(field, this.lambda, focal);
    }

    let field = new ComplexField(this.nx, this.ny, this.dx, this.dy);
    field.setAmplitudePhase(r.illumination, phase ?? r.phase);
    if (pad > 1) field = field.resample(this.nx * pad, this.ny * pad);
    propagateAngularSpectrumInPlace(field, this.lambda, dist, { bandLimit });
    if (pad > 1) field = field.resample(this.nx, this.ny); // centre crop
    return field;
  }

  /**
   * Simulation with a *fixed* projection lens and a screen that can move off the
   * focal plane. This is the real behaviour of a projector head at a throw that
   * does not match its lens: the image scales with the distance and defocuses
   * away from the focal plane.
   *
   * Two things to know before comparing the result with a target:
   *  - the field comes back on the *element* grid (same pitch and window), so a
   *    magnified image is only visible through the central part of the screen
   *    window. Use `simulate()` when you want the screen sampled on its own
   *    pitch (that is what the metrics and the workbench do);
   *  - `scale` is the geometric magnification z/f of the ray picture; the
   *    diffractive blur on top of it is genuine diffraction, not a resampling
   *    artefact.
   *
   * @returns {{field: ComplexField, scale: number, focal: number, z: number}}
   */
  simulateWithFixedLens({ z = null, focal = null, phase = null, pad = 2 } = {}) {
    if (!this.result) throw new Error('ProjectorSystem.simulateWithFixedLens: call design() first');
    const r = this.result;
    const dist = z ?? this.z;
    const foc = focal ?? this.z;
    let field = this.applyLens({ focal: foc, phase });
    if (pad > 1) field = field.resample(this.nx * pad, this.ny * pad);
    propagateAngularSpectrumInPlace(field, this.lambda, dist, { bandLimit: false });
    if (pad > 1) field = field.resample(this.nx, this.ny);
    return { field, scale: dist / foc, focal: foc, z: dist };
  }

  /** The element field with a projection lens phase applied (no propagation). */
  applyLens({ focal = null, phase = null } = {}) {
    const r = this.result;
    const foc = focal ?? this.z;
    const field = new ComplexField(this.nx, this.ny, this.dx, this.dy);
    field.setAmplitudePhase(r.illumination, phase ?? r.phase);
    const lens = lensPhase(field, this.lambda, foc, { wrap: true });
    field.multiplyPhase(lens);
    return field;
  }

  /** Simulate with the magnified Fresnel kernel — the real far-field screen. */
  simulateFarField({ z = null, phase = null } = {}) {
    if (!this.result) throw new Error('ProjectorSystem.simulateFarField: call design() first');
    const dist = z ?? this.z;
    const r = this.result;
    const field = new ComplexField(this.nx, this.ny, this.dx, this.dy);
    field.setAmplitudePhase(r.illumination, phase ?? r.phase);
    return propagateFresnelFFT(field, this.lambda, dist);
  }

  /**
   * Projection-quality metrics for a simulated screen field.
   * All quantities are computed in the projection plane, against the target.
   */
  screenMetrics(field, { target = null, roi = null, phase = null } = {}) {
    const tgt = target ?? this.target;
    const irr = field.intensity();
    const mask = roi ?? tgt.roi;
    // The zero-order leak is a property of the *element*: the fraction of the
    // incident light that the phase pattern leaves in the undiffracted beam.
    // Measured at the screen it would be phase-sensitive and geometry-dependent
    // (see the docstring of zeroOrderFraction), so build the element field here.
    const element = new ComplexField(this.nx, this.ny, this.dx, this.dy);
    element.setAmplitudePhase(this.result.illumination, phase ?? this.result.phase);
    // The screen grid carries its own pitch: in the far-field geometry it is
    // lambda*f/(N*dx), not the element pitch. Using the element pitch here
    // would scale every radiometric number by the magnification squared.
    const sdx = field.dx;
    const sdy = field.dy;
    const cell = sdx * sdy;
    const intensityTarget = new Float64Array(tgt.irradiance.length);
    for (let k = 0; k < intensityTarget.length; k++) intensityTarget[k] = tgt.irradiance[k];
    // scale the target to the same total power for a fair comparison
    const pt = maskedSum(irr, mask) * cell;
    const tt = maskedSum(intensityTarget, null) * cell;
    const s = tt > 0 ? pt / tt : 0;
    for (let k = 0; k < intensityTarget.length; k++) intensityTarget[k] *= s;

    // Uniformity is only meaningful over the illuminated part of the target:
    // count pixels above 10% of the ROI peak, which is the usual projector
    // convention and keeps the dark background out of the min/max.
    // Uniformity depends on what the target is: for a spot array the meaningful
    // figure is the min/max over the *spots*, not over pixels (a single dim edge
    // pixel would otherwise define the number).
    const provisional = regionStats(irr, mask);
    const stats = regionStats(irr, mask, { threshold: 0.1 * provisional.max });
    const spots = tgt.grid ? spotUniformity(irr, this.nx, this.ny, tgt.grid) : null;
    const incident = maskedSum(this.illumination(), null) * this.dx * this.dy;
    const total = maskedSum(irr, null) * cell;
    const inSignal = maskedSum(irr, mask) * cell;
    const rms = rmsError(irr, intensityTarget, mask);
    const corr = correlation(irr, intensityTarget, mask);
    const ss = ssim(irr, intensityTarget, this.nx, this.ny, { window: 8, weight: 'target' });
    const rr = rmsRadius(irr, this.nx, sdx, 0, 0, sdy);
    return {
      efficiency: incident > 0 ? inSignal / incident : 0,
      fluxFraction: total > 0 ? inSignal / total : 0,
      uniformity: spots ? spots.uniformity : stats.uniformity,
      spotUniformity: spots ? spots.uniformity : null,
      flatness: stats.flatness,
      contrast: stats.cv,
      rmse: rms,
      fidelity: (corr + 1) / 2,
      correlation: corr,
      ssim: ss,
      snr: rms > 0 ? 20 * Math.log10(1 / rms) : Infinity,
      zeroOrder: zeroOrderFraction(element),
      rmsRadius: rr.r,
      meanIntensity: stats.mean,
      peakIntensity: stats.max,
      powerInSignal: inSignal,
      powerAtScreen: total,
      powerIncident: incident,
    };
  }

  /**
   * Defocus curve: how the projected image degrades as the screen moves away
   * from the design distance. This is the quantitative statement of "the image
   * is only in focus at z" — and the easiest way to tell the two projection
   * regimes apart.
   */
  defocusCurve({ points = 21, span = 0.6, pad = 2, mode = null } = {}) {
    const z0 = this.z * (1 - span);
    const z1 = this.z * (1 + span);
    const out = [];
    for (let i = 0; i < points; i++) {
      const z = z0 + ((z1 - z0) * i) / (points - 1);
      const field = this.simulate({ z, pad, mode });
      const m = this.screenMetrics(field);
      out.push({
        z,
        zMm: z * 1e3,
        fidelity: m.fidelity,
        correlation: m.correlation,
        ssim: m.ssim,
        rmse: m.rmse,
        efficiency: m.efficiency,
        uniformity: m.uniformity,
      });
    }
    return out;
  }

  /** Focus stack: irradiance planes along the axis, for the volume view. */
  focusStack({ points = 9, span = 0.5, pad = 2, mode = null } = {}) {
    const planes = [];
    for (let i = 0; i < points; i++) {
      const z = this.z * (1 - span) + (this.z * 2 * span * i) / (points - 1);
      const field = this.simulate({ z, pad, mode });
      planes.push({ z, intensity: field.intensity() });
    }
    return planes;
  }

  /**
   * Baseline: the same system with a flat element (no DOE). Shows what the
   * phase-retrieval design actually buys: without it, the screen shows the
   * aperture's diffraction pattern, not the image.
   */
  baseline({ z = null, pad = 2 } = {}) {
    const dist = z ?? this.z;
    const field = new ComplexField(this.nx, this.ny, this.dx, this.dy);
    field.setAmplitudePhase(this.illumination(), 0);
    let f = pad > 1 ? field.resample(this.nx * pad, this.ny * pad) : field;
    propagateAngularSpectrumInPlace(f, this.lambda, dist, { bandLimit: false });
    if (pad > 1) f = f.resample(this.nx, this.ny);
    return f;
  }

  /**
   * Design a single element that projects one image at z1 and a *different*
   * image at z2 — true multi-plane (focus-stack) projection. Uses the
   * multi-plane Gerchberg–Saxton loop.
   */
  designMultiPlane(planeSpecs, opts = {}) {
    const illumination = this.illumination();
    const planes = planeSpecs.map((p) => {
      const t = p.irradiance ? p : makeTarget(p.kind, this.nx, this.ny, p);
      const amp = new Float64Array(this.nx * this.ny);
      for (let k = 0; k < amp.length; k++) amp[k] = Math.sqrt(Math.max(0, t.irradiance[k]));
      return { z: p.z, targetAmp: amp, roi: t.roi ?? autoRoi(t.irradiance, this.nx, this.ny), weight: p.weight ?? 1 };
    });
    const res = gerchbergSaxtonMultiPlane(illumination, planes, {
      nx: this.nx,
      ny: this.ny,
      dx: this.dx,
      dy: this.dy,
      lambda: this.lambda,
      iterations: opts.iterations ?? this.cfg.iterations,
      seed: opts.seed ?? this.cfg.seed,
      initPhase: opts.initPhase,
      onProgress: opts.onProgress,
    });
    const levels = opts.levels ?? this.cfg.levels;
    const out = {
      ...res,
      illumination,
      lambda: this.lambda,
      nx: this.nx,
      ny: this.ny,
      dx: this.dx,
      dy: this.dy,
      z: this.z,
      mode: 'multi-plane',
      planes,
      target: this.target,
      roi: this.target.roi,
    };
    if (levels > 1) out.phase = errorDiffusionQuantize(out.phase, levels);
    this.result = out;
    return out;
  }

  /** Metrics summary string (CLI/test friendly). */
  report(metrics) {
    return summarizeMetrics(metrics);
  }
}

/**
 * Sweep a design over an interesting parameter (iterations, levels) and return
 * the metric trajectory — used by the CLI and by the benchmark tool to show
 * convergence rather than assert it.
 */
export function convergenceSweep(cfg, { checkpoints = [1, 2, 5, 10, 20, 40, 80] } = {}) {
  const sys = new ProjectorSystem(cfg);
  const out = [];
  for (const it of checkpoints) {
    const r = sys.design({ iterations: it });
    const field = sys.simulate();
    const m = sys.screenMetrics(field);
    out.push({ iterations: it, ...m, err: r.error[r.error.length - 1] });
  }
  return out;
}
