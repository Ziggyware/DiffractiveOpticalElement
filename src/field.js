/**
 * field.js — monochromatic scalar complex fields.
 *
 * A ComplexField is a sampled optical field
 *
 *      U(x, y) = A(x, y) * exp(i * phi(x, y))
 *
 * on a uniform nx-by-ny grid with pitches (dx, dy), stored as two Float64Array
 * planes in row-major order. Sample (i, j) sits at
 *
 *      x = (i - nx/2) * dx,   y = (j - ny/2) * dy
 *
 * i.e. the optical axis is at the *centre* of the array (a half-pixel offset
 * that cancels in every symmetric operation we perform, and which keeps
 * fftshift/ifftshift pairings trivial for even (power-of-two) sizes).
 *
 * All physical quantities are SI: metres, radians, watts.
 */

import { fft2d, ifftshift2d, fftshift2d, isPow2 } from './fft.js';

export class ComplexField {
  /**
   * @param {number} nx number of samples along x (power of two)
   * @param {number} ny number of samples along y (power of two)
   * @param {number} dx sample pitch along x [m]
   * @param {number} dy sample pitch along y [m] (defaults to dx)
   */
  constructor(nx, ny = nx, dx = 1, dy = dx) {
    if (!isPow2(nx) || !isPow2(ny)) {
      throw new Error(`ComplexField: grid must be power-of-two, got ${nx}x${ny}`);
    }
    this.nx = nx;
    this.ny = ny;
    this.dx = dx;
    this.dy = dy;
    this.re = new Float64Array(nx * ny);
    this.im = new Float64Array(nx * ny);
  }

  get size() {
    return this.nx * this.ny;
  }

  /** Physical extent along x [m]. */
  get extentX() {
    return this.nx * this.dx;
  }

  /** Physical extent along y [m]. */
  get extentY() {
    return this.ny * this.dy;
  }

  /** x coordinate of sample column i [m]. */
  x(i) {
    return (i - this.nx / 2) * this.dx;
  }

  /** y coordinate of sample row j [m]. */
  y(j) {
    return (j - this.ny / 2) * this.dy;
  }

  idx(i, j) {
    return i + j * this.nx;
  }

  clone() {
    const f = new ComplexField(this.nx, this.ny, this.dx, this.dy);
    f.re.set(this.re);
    f.im.set(this.im);
    return f;
  }

  /** Copy the contents of another (identically shaped) field into this one. */
  copyFrom(other) {
    if (other.nx !== this.nx || other.ny !== this.ny) {
      throw new Error('ComplexField.copyFrom: shape mismatch');
    }
    this.re.set(other.re);
    this.im.set(other.im);
    this.dx = other.dx;
    this.dy = other.dy;
    return this;
  }

  fill(re = 0, im = 0) {
    this.re.fill(re);
    this.im.fill(im);
    return this;
  }

  zero() {
    return this.fill(0, 0);
  }

  /** |U|^2 for sample (i, j). */
  intensityAt(i, j) {
    const k = i + j * this.nx;
    return this.re[k] * this.re[k] + this.im[k] * this.im[k];
  }

  /** |U| for sample (i, j). */
  amplitudeAt(i, j) {
    const k = i + j * this.nx;
    return Math.hypot(this.re[k], this.im[k]);
  }

  /** atan2(im, re) for sample (i, j), in (-pi, pi]. */
  phaseAt(i, j) {
    const k = i + j * this.nx;
    return Math.atan2(this.im[k], this.re[k]);
  }

  /** New plane containing |U|. */
  amplitude(out = new Float64Array(this.size)) {
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) out[k] = Math.hypot(re[k], im[k]);
    return out;
  }

  /** New plane containing |U|^2 (irradiance, arbitrary units until scaled). */
  intensity(out = new Float64Array(this.size)) {
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) out[k] = re[k] * re[k] + im[k] * im[k];
    return out;
  }

  /** New plane containing the wrapped phase atan2(im, re). */
  phase(out = new Float64Array(this.size)) {
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) out[k] = Math.atan2(im[k], re[k]);
    return out;
  }

  /** In-place multiplication by exp(i * p[k]). */
  multiplyPhase(p) {
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) {
      const c = Math.cos(p[k]);
      const s = Math.sin(p[k]);
      const r = re[k];
      const i2 = im[k];
      re[k] = r * c - i2 * s;
      im[k] = r * s + i2 * c;
    }
    return this;
  }

  /** In-place multiplication by exp(-i * p[k]) (adjoint of multiplyPhase). */
  multiplyConjPhase(p) {
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) {
      const c = Math.cos(p[k]);
      const s = -Math.sin(p[k]);
      const r = re[k];
      const i2 = im[k];
      re[k] = r * c - i2 * s;
      im[k] = r * s + i2 * c;
    }
    return this;
  }

  /**
   * Set this field from amplitude and phase planes (either may be a scalar).
   */
  setAmplitudePhase(amp, phi = 0) {
    const { re, im } = this;
    const scalarA = typeof amp === 'number';
    const scalarP = typeof phi === 'number';
    for (let k = 0; k < re.length; k++) {
      const a = scalarA ? amp : amp[k];
      const p = scalarP ? phi : phi[k];
      re[k] = a * Math.cos(p);
      im[k] = a * Math.sin(p);
    }
    return this;
  }

  /** Uniform plane wave: constant amplitude, constant phase. */
  setPlaneWave(amp = 1, phi = 0) {
    return this.setAmplitudePhase(amp, phi);
  }

  /**
   * Total optical power crossing the grid: P = integral |U|^2 dA [W].
   * (Irradiance is in W/m^2 when amplitude is sqrt(W/m^2).)
   */
  power() {
    const { re, im } = this;
    let s = 0;
    for (let k = 0; k < re.length; k++) s += re[k] * re[k] + im[k] * im[k];
    return s * this.dx * this.dy;
  }

  /** Peak irradiance. */
  peakIntensity() {
    const { re, im } = this;
    let m = 0;
    for (let k = 0; k < re.length; k++) {
      const v = re[k] * re[k] + im[k] * im[k];
      if (v > m) m = v;
    }
    return m;
  }

  /** Scale the field in place so that peak irradiance becomes `peak`. */
  normalizePeak(peak = 1) {
    const p = this.peakIntensity();
    if (p > 0) {
      const s = Math.sqrt(peak / p);
      this.scale(s);
    }
    return this;
  }

  /** Scale the field in place so that total power becomes `power`. */
  normalizePower(power = 1) {
    const p = this.power();
    if (p > 0) this.scale(Math.sqrt(power / p));
    return this;
  }

  /** In-place scalar multiplication. */
  scale(s) {
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) {
      re[k] *= s;
      im[k] *= s;
    }
    return this;
  }

  /** In-place multiply by exp(i * phase) where phase = const (global phase). */
  globalPhase(phi) {
    const c = Math.cos(phi);
    const s = Math.sin(phi);
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) {
      const r = re[k];
      const i2 = im[k];
      re[k] = r * c - i2 * s;
      im[k] = r * s + i2 * c;
    }
    return this;
  }

  /** Element-wise product with another field (same shape). */
  multiply(other) {
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) {
      const r = re[k];
      const i2 = im[k];
      const or = other.re[k];
      const oi = other.im[k];
      re[k] = r * or - i2 * oi;
      im[k] = r * oi + i2 * or;
    }
    return this;
  }

  /** Add another field in place. */
  add(other) {
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) {
      re[k] += other.re[k];
      im[k] += other.im[k];
    }
    return this;
  }

  /**
   * Coherent sum of several fields (interference).
   * @param {ComplexField[]} fields
   */
  static sum(fields) {
    if (fields.length === 0) throw new Error('ComplexField.sum: no fields');
    const out = fields[0].clone();
    for (let i = 1; i < fields.length; i++) out.add(fields[i]);
    return out;
  }

  /**
   * Incoherent (intensity) sum of several fields — the physically correct way
   * to combine mutually incoherent sources.
   * @param {ComplexField[]} fields
   */
  static sumIncoherent(fields) {
    if (fields.length === 0) throw new Error('ComplexField.sumIncoherent: no fields');
    const out = fields[0].clone();
    // Accumulate intensities and store as sqrt for a phase-free representation.
    const acc = new Float64Array(out.size);
    for (const f of fields) {
      for (let k = 0; k < acc.length; k++) {
        acc[k] += f.re[k] * f.re[k] + f.im[k] * f.im[k];
      }
    }
    for (let k = 0; k < acc.length; k++) {
      out.re[k] = Math.sqrt(acc[k]);
      out.im[k] = 0;
    }
    return out;
  }

  /**
   * Array of spatial frequencies of the DFT grid, in cycles/metre, in
   * *shifted* (fftshift) order: sample i holds f_x = (i - nx/2) / (nx*dx).
   */
  freqX(i) {
    return (i - this.nx / 2) / (this.nx * this.dx);
  }

  freqY(j) {
    return (j - this.ny / 2) / (this.ny * this.dy);
  }

  /** Sample index spacing of the frequency grid (cycles/m). */
  get dfx() {
    return 1 / (this.nx * this.dx);
  }

  get dfy() {
    return 1 / (this.ny * this.dy);
  }

  /** Centred (shifted) spectrum of the field. */
  spectrum() {
    const re = Float64Array.from(this.re);
    const im = Float64Array.from(this.im);
    ifftshift2d(re, im, this.nx, this.ny);
    fft2d(re, im, this.nx, this.ny, false);
    fftshift2d(re, im, this.nx, this.ny, true);
    return { re, im, nx: this.nx, ny: this.ny, dfx: this.dfx, dfy: this.dfy };
  }

  /**
   * Bilinear sampling of a scalar image (row-major, `width` by `height`,
   * values typically 0..1) onto this grid.
   *
   * The image is mapped so that it exactly covers the sampling window; use
   * `scale` < 1 to inset the image with zero padding.
   *
   * @param {Float64Array|Float32Array|number[]} img source pixels, row-major
   * @param {number} width
   * @param {number} height
   * @param {number} scale fraction of the window covered by the image
   * @param {number} fill value outside the image
   */
  sampleImage(img, width, height, scale = 1, fill = 0) {
    const { nx, ny } = this;
    // The raster covers the central `scale` fraction of the window: grid sample
    // i sits at (i + 0.5)/nx across the window, which maps to the image
    // coordinate ((i + 0.5)/nx - (1 - scale)/2) * width / scale. Each sample
    // then *integrates the image over the box it covers* (an area average, not a
    // point sample), which is what makes the mapping resolution-independent:
    // sampling an nx-pixel image onto an nx grid returns the image unchanged,
    // sampling it onto a finer grid is a proper antialiased upsampling, and
    // sampling it onto a coarser grid is an area downsample. Outside the image
    // (either in space, when scale < 1, or in space beyond the raster's own
    // footprint) the `fill` value is used.
    const s = Math.max(scale, 1e-9);
    const off = (1 - s) / 2;
    const boxW = width / (nx * s); // image pixels covered by one grid sample
    const boxH = height / (ny * s);
    for (let j = 0; j < ny; j++) {
      const cy = (((j + 0.5) / ny - off) * height) / s;
      for (let i = 0; i < nx; i++) {
        const cx = (((i + 0.5) / nx - off) * width) / s;
        // separable: average over the box in x and y
        let acc = 0;
        const ax = cx - boxW / 2;
        const bx = cx + boxW / 2;
        const ay = cy - boxH / 2;
        const by = cy + boxH / 2;
        const lox = Math.max(ax, 0);
        const hix = Math.min(bx, width);
        const loy = Math.max(ay, 0);
        const hiy = Math.min(by, height);
        const area = boxW * boxH;
        if (hix > lox && hiy > loy) {
          const i0 = Math.max(0, Math.floor(lox));
          const i1 = Math.min(width - 1, Math.floor(hix - 1e-12));
          const j0 = Math.max(0, Math.floor(loy));
          const j1 = Math.min(height - 1, Math.floor(hiy - 1e-12));
          for (let jj = j0; jj <= j1; jj++) {
            const wy = Math.min(by, jj + 1) - Math.max(ay, jj);
            for (let ii = i0; ii <= i1; ii++) {
              const wx = Math.min(bx, ii + 1) - Math.max(ax, ii);
              acc += img[ii + jj * width] * wx * wy;
            }
          }
        }
        acc += fill * (area - (hix - lox) * (hiy - loy));
        this.re[i + j * nx] = acc / area;
      }
    }
    this.im.fill(0);
    return this;
  }

  /** Fill amplitude from an analytic function fn(x, y, i, j). */
  fillFrom(fn) {
    for (let j = 0; j < this.ny; j++) {
      const y = this.y(j);
      for (let i = 0; i < this.nx; i++) {
        const k = i + j * this.nx;
        this.re[k] = fn(this.x(i), y, i, j) ?? 0;
      }
    }
    this.im.fill(0);
    return this;
  }

  /** Largest |U| on the grid. */
  maxAmplitude() {
    let m = 0;
    for (let k = 0; k < this.re.length; k++) {
      const a = Math.hypot(this.re[k], this.im[k]);
      if (a > m) m = a;
    }
    return m;
  }

  /** Min/max of the irradiance plane. */
  intensityRange() {
    const { re, im } = this;
    let mn = Infinity;
    let mx = -Infinity;
    for (let k = 0; k < re.length; k++) {
      const v = re[k] * re[k] + im[k] * im[k];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    return { min: mn, max: mx };
  }

  /**
   * Pad (or crop) onto a new grid, keeping the field centred. Zero padding is
   * the standard way to give a propagating field room to spread without
   * wrapping around the DFT window.
   */
  resample(nx2, ny2, dx2 = this.dx, dy2 = this.dy) {
    const out = new ComplexField(nx2, ny2, dx2, dy2);
    if (dx2 === this.dx && dy2 === this.dy && Number.isInteger(nx2) && Number.isInteger(ny2)) {
      const ox = Math.floor((nx2 - this.nx) / 2);
      const oy = Math.floor((ny2 - this.ny) / 2);
      for (let j = 0; j < this.ny; j++) {
        const jj = j + oy;
        if (jj < 0 || jj >= ny2) continue;
        for (let i = 0; i < this.nx; i++) {
          const ii = i + ox;
          if (ii < 0 || ii >= nx2) continue;
          const s = i + j * this.nx;
          const d = ii + jj * nx2;
          out.re[d] = this.re[s];
          out.im[d] = this.im[s];
        }
      }
      return out;
    }
    // General case: bilinear resample in physical coordinates.
    for (let j = 0; j < ny2; j++) {
      const y = out.y(j);
      const fy = y / this.dy + this.ny / 2 - 0.5;
      const j0 = Math.floor(fy);
      const ty = fy - j0;
      for (let i = 0; i < nx2; i++) {
        const x = out.x(i);
        const fx = x / this.dx + this.nx / 2 - 0.5;
        const i0 = Math.floor(fx);
        const tx = fx - i0;
        let sr = 0;
        let si = 0;
        for (let b = 0; b < 2; b++) {
          const jj = j0 + b;
          if (jj < 0 || jj >= this.ny) continue;
          const wy = b === 0 ? 1 - ty : ty;
          for (let a = 0; a < 2; a++) {
            const ii = i0 + a;
            if (ii < 0 || ii >= this.nx) continue;
            const wx = a === 0 ? 1 - tx : tx;
            const w = wx * wy;
            const s = ii + jj * this.nx;
            sr += this.re[s] * w;
            si += this.im[s] * w;
          }
        }
        const d = i + j * nx2;
        out.re[d] = sr;
        out.im[d] = si;
      }
    }
    return out;
  }

  /** Crop to a centred sub-grid of physical size (w, h) metres. */
  cropCentred(width, height) {
    const nx2 = Math.max(2, Math.round(width / this.dx));
    const ny2 = Math.max(2, Math.round(height / this.dy));
    return this.resample(nx2, ny2);
  }

  /** Human-readable summary, handy in REPL/tests. */
  describe() {
    const r = this.intensityRange();
    return `ComplexField ${this.nx}x${this.ny} @ ${(this.dx * 1e6).toFixed(3)}um  extent=${(
      this.extentX * 1e3
    ).toFixed(3)}x${(this.extentY * 1e3).toFixed(3)}mm  I=[${r.min.toExponential(3)}, ${r.max.toExponential(
      3,
    )}]`;
  }
}

/** Circular boolean mask (disc) on a grid, as a Float64Array of 0/1. */
export function discMask(nx, ny, dx, dy, radius, { soft = 0, cx = 0, cy = 0 } = {}) {
  const m = new Float64Array(nx * ny);
  const f = new ComplexField(nx, ny, dx, dy);
  for (let j = 0; j < ny; j++) {
    const y = f.y(j) - cy;
    for (let i = 0; i < nx; i++) {
      const x = f.x(i) - cx;
      const r = Math.hypot(x, y);
      let v = 0;
      if (soft > 0) {
        // Raised-cosine (apodized) edge — suppresses the Airy rings that a
        // hard aperture would impose on the projected image.
        v = r <= radius - soft ? 1 : r >= radius ? 0 : 0.5 * (1 + Math.cos((Math.PI * (r - radius + soft)) / soft));
      } else {
        v = r <= radius ? 1 : 0;
      }
      m[i + j * nx] = v;
    }
  }
  return m;
}

/** Rectangular boolean mask, as a Float64Array of 0/1. */
export function rectMask(nx, ny, dx, dy, width, height, { soft = 0 } = {}) {
  const m = new Float64Array(nx * ny);
  const f = new ComplexField(nx, ny, dx, dy);
  for (let j = 0; j < ny; j++) {
    const y = Math.abs(f.y(j));
    for (let i = 0; i < nx; i++) {
      const x = Math.abs(f.x(i));
      const fx = soft > 0 ? edge(x, width / 2, soft) : x <= width / 2 ? 1 : 0;
      const fy = soft > 0 ? edge(y, height / 2, soft) : y <= height / 2 ? 1 : 0;
      m[i + j * nx] = fx * fy;
    }
  }
  return m;
}

function edge(x, half, soft) {
  if (x <= half - soft) return 1;
  if (x >= half) return 0;
  return 0.5 * (1 + Math.cos((Math.PI * (x - half + soft)) / soft));
}

/** Super-Gaussian (flattened Gaussian) amplitude profile of order p. */
export function superGaussian(nx, ny, dx, dy, radius, order = 8) {
  const a = new Float64Array(nx * ny);
  const f = new ComplexField(nx, ny, dx, dy);
  for (let j = 0; j < ny; j++) {
    const y = f.y(j);
    for (let i = 0; i < nx; i++) {
      const x = f.x(i);
      const r = Math.hypot(x, y) / radius;
      a[i + j * nx] = Math.exp(-Math.pow(r, order));
    }
  }
  return a;
}
