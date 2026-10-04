/**
 * metrics.js — figure-of-merit definitions for projection quality.
 *
 * Every metric here is defined on *irradiance* planes (Float64Array, W/m^2) so
 * that the numbers mean the same thing whether they came from a simulation or
 * from a real measurement of a projected screen.
 */

/** Sum of a plane over a mask (mask may be null for "everything"). */
export function maskedSum(plane, mask = null) {
  let s = 0;
  if (mask === null) {
    for (let k = 0; k < plane.length; k++) s += plane[k];
    return s;
  }
  for (let k = 0; k < plane.length; k++) s += plane[k] * mask[k];
  return s;
}

/** Area-weighted power of an irradiance plane [W] on a grid of pitch (dx, dy). */
export function powerOf(plane, dx, dy, mask = null) {
  return maskedSum(plane, mask) * dx * dy;
}

/**
 * Region statistics inside a mask: mean, min, max, std, uniformity.
 *
 * `uniformity` is the standard projector metric 1 - (max-min)/(max+min), which
 * equals 2*min/(max+min), computed over the *illuminated* part of the region.
 * The `threshold` argument is absolute (in irradiance units) and defaults to 0;
 * callers that need the usual "ignore the dark background" behaviour pass a
 * fraction of the region peak (ProjectorSystem#screenMetrics uses 10%).
 * `flatness` = 1 - std/mean and `cv` = std/mean are also returned; flatness is
 * the smoother figure of merit because it uses every pixel rather than the two
 * extremes.
 */
export function regionStats(plane, mask, { threshold = 0 } = {}) {
  let n = 0;
  let sum = 0;
  let sumSq = 0;
  let mn = Infinity;
  let mx = -Infinity;
  for (let k = 0; k < plane.length; k++) {
    if (mask && mask[k] <= 0) continue;
    const v = plane[k];
    if (v <= threshold) continue;
    n++;
    sum += v;
    sumSq += v * v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (n === 0) return { n: 0, mean: 0, min: 0, max: 0, std: 0, uniformity: 0, flatness: 0, cv: 0 };
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  const std = Math.sqrt(variance);
  return {
    n,
    mean,
    min: mn,
    max: mx,
    std,
    uniformity: mx + mn > 0 ? 1 - (mx - mn) / (mx + mn) : 1,
    flatness: mean > 0 ? 1 - std / mean : 1,
    cv: mean > 0 ? std / mean : 0,
  };
}

/**
 * Relative RMS error between an irradiance plane and a target, normalised by
 * the target power (the classic GS cost function).
 */
export function rmsError(plane, target, mask = null) {
  let num = 0;
  let den = 0;
  for (let k = 0; k < plane.length; k++) {
    const m = mask ? mask[k] : 1;
    if (m <= 0) continue;
    const d = plane[k] - target[k];
    num += d * d;
    den += target[k] * target[k];
  }
  return den > 0 ? Math.sqrt(num / den) : 0;
}

/** Normalised cross-correlation between an irradiance plane and a target. */
export function correlation(plane, target, mask = null) {
  let sp = 0;
  let st = 0;
  let spp = 0;
  let stt = 0;
  let n = 0;
  for (let k = 0; k < plane.length; k++) {
    if (mask && mask[k] <= 0) continue;
    const p = plane[k];
    const t = target[k];
    sp += p;
    st += t;
    spp += p * p;
    stt += t * t;
    n++;
  }
  if (n === 0) return 0;
  const mp = sp / n;
  const mt = st / n;
  let cov = 0;
  let vp = 0;
  let vt = 0;
  for (let k = 0; k < plane.length; k++) {
    if (mask && mask[k] <= 0) continue;
    const dp = plane[k] - mp;
    const dt = target[k] - mt;
    cov += dp * dt;
    vp += dp * dp;
    vt += dt * dt;
  }
  return vp > 0 && vt > 0 ? cov / Math.sqrt(vp * vt) : 0;
}

/** Peak signal-to-noise ratio between image and target, in dB. */
export function psnr(plane, target, mask = null) {
  let mse = 0;
  let n = 0;
  let peak = 0;
  for (let k = 0; k < plane.length; k++) {
    if (mask && mask[k] <= 0) continue;
    const d = plane[k] - target[k];
    mse += d * d;
    peak = Math.max(peak, target[k]);
    n++;
  }
  if (n === 0 || mse === 0) return Infinity;
  mse /= n;
  return 10 * Math.log10((peak * peak) / mse);
}

/**
 * Fraction of the power that stayed in the zero (undiffracted) order, measured
 * from the *element-plane* field: |mean(U)|^2 / mean(|U|^2). For a good projector
 * DOE this is small — DC leakage is what shows up as a bright spot in the middle
 * of a projected image. Pass the field at the DOE, not at the screen: at the
 * screen the "DC bin" is a phase-sensitive quantity that depends on the screen
 * distance, whereas the element-plane number is a property of the design.
 */
export function zeroOrderFraction(field) {
  let sr = 0;
  let si = 0;
  let p = 0;
  const n = field.size;
  for (let k = 0; k < n; k++) {
    sr += field.re[k];
    si += field.im[k];
    p += field.re[k] * field.re[k] + field.im[k] * field.im[k];
  }
  if (p === 0) return 0;
  return (sr * sr + si * si) / (n * p);
}

/**
 * Root-mean-square emittance-style "spot size" of an irradiance plane, in
 * metres. Used to demonstrate defocus: the projected spot broadens away from
 * the design plane.
 */
export function rmsRadius(plane, nx, dx, cx = null, cy = null, dy = dx) {
  const ny = plane.length / nx;
  const x0 = cx ?? 0;
  const y0 = cy ?? 0;
  let p = 0;
  let sx = 0;
  let sy = 0;
  for (let j = 0; j < ny; j++) {
    const y = (j - ny / 2) * dy - y0;
    for (let i = 0; i < nx; i++) {
      const x = (i - nx / 2) * dx - x0;
      const v = plane[i + j * nx];
      p += v;
      sx += v * x * x;
      sy += v * y * y;
    }
  }
  if (p === 0) return { x: 0, y: 0, r: 0 };
  const rx = Math.sqrt(sx / p);
  const ry = Math.sqrt(sy / p);
  return { x: rx, y: ry, r: Math.hypot(rx, ry) };
}

/**
 * Structural similarity (mean SSIM) between a projected image and the target,
 * in the usual block-windowed form (8x8 by default) with the standard
 * luminance * contrast * structure decomposition.
 *
 * `dataRange` sets the dynamic range L used for the stabilisers C1 = (0.01 L)^2
 * and C2 = (0.03 L)^2 (defaults to the larger of the two peaks) and
 * `weight: 'target'` makes each window count in proportion to the target
 * energy it contains, which is what you want for sparse images (text, spot
 * arrays) where most of the plane is deliberately dark.
 */
export function ssim(plane, target, nx, ny, { window = 8, dataRange = null, weight = null } = {}) {
  // SSIM's stabilisers are defined relative to the dynamic range of the data:
  // C1 = (K1*L)^2 and C2 = (K2*L)^2 with K1 = 0.01, K2 = 0.03. Hard-coding L = 1
  // while the planes carry irradiance in W/m^2 makes the luminance and contrast
  // terms collapse and the metric stops responding to image quality. Unless the
  // caller supplies a range we take the larger of the two peaks, so both planes
  // are compared on one common scale.
  let L = dataRange;
  if (L === null) {
    let mx = 0;
    for (let k = 0; k < plane.length; k++) {
      const p = plane[k];
      const t = target[k];
      if (p > mx) mx = p;
      if (t > mx) mx = t;
    }
    L = mx > 0 ? mx : 1;
  }
  const C1 = 0.01 * L * (0.01 * L);
  const C2 = 0.03 * L * (0.03 * L);
  let total = 0;
  let weightSum = 0;
  for (let wy = 0; wy + window <= ny; wy += window) {
    for (let wx = 0; wx + window <= nx; wx += window) {
      let sp = 0;
      let st = 0;
      let spp = 0;
      let stt = 0;
      let spt = 0;
      const n = window * window;
      for (let j = 0; j < window; j++) {
        for (let i = 0; i < window; i++) {
          const k = wx + i + (wy + j) * nx;
          const p = plane[k];
          const t = target[k];
          sp += p;
          st += t;
          spp += p * p;
          stt += t * t;
          spt += p * t;
        }
      }
      const mp = sp / n;
      const mt = st / n;
      const vp = Math.abs(spp / n - mp * mp);
      const vt = Math.abs(stt / n - mt * mt);
      const cpt = spt / n - mp * mt;
      const num = (2 * mp * mt + C1) * (2 * cpt + C2);
      const den = (mp * mp + mt * mt + C1) * (vp + vt + C2);
      // With `weight='target'` each window counts in proportion to how much of
      // the image it contains. That matters for sparse targets (text, a spot
      // array, a logo): most of the plane is intentionally dark, and a uniform
      // block average would let those empty blocks - where the only thing to
      // see is the reconstruction's speckle haze - dominate the score. The haze
      // is already measured by RMSE, zero-order leak and flux fraction.
      const wt = weight === 'target' ? st / n : 1;
      total += (den !== 0 ? num / den : 0) * wt;
      weightSum += wt;
    }
  }
  return weightSum > 0 ? total / weightSum : 0;
}

/**
 * Contrast / visibility of a fringe pattern: (I_max - I_min)/(I_max + I_min)
 * inside a mask. 1 = perfect fringes, 0 = no modulation.
 */
export function visibility(plane, mask = null) {
  const st = regionStats(plane, mask);
  return st.max + st.min > 0 ? (st.max - st.min) / (st.max + st.min) : 0;
}

/**
 * Spot-array uniformity: the metric that actually matters for a multispot DOE.
 *
 * The irradiance is split into a (cols x rows) grid of cells, the power in each
 * cell is integrated, and the standard min/max uniformity 2*min/(max+min) is
 * returned over the cells. This is the number a structured-light projector
 * datasheet quotes, and it is far more informative than a per-pixel min/max
 * (which a single dim edge pixel can drag to zero).
 *
 * @param {Float64Array} plane irradiance at the screen
 * @param {number} nx @param {number} ny grid size
 * @param {{cols?: number, rows?: number, minPeak?: number}} opts
 * @returns {{uniformity: number, cells: number[], min: number, max: number, mean: number, cv: number}}
 */
export function spotUniformity(plane, nx, ny, { cols = 8, rows = 8, minPeak = 0.05 } = {}) {
  const cells = new Array(cols * rows).fill(0);
  let peak = 0;
  for (let k = 0; k < plane.length; k++) if (plane[k] > peak) peak = plane[k];
  const cut = minPeak * peak;
  for (let j = 0; j < ny; j++) {
    const cj = Math.min(rows - 1, Math.floor((j / ny) * rows));
    for (let i = 0; i < nx; i++) {
      const ci = Math.min(cols - 1, Math.floor((i / nx) * cols));
      const v = plane[i + j * nx];
      if (v >= cut) cells[ci + cj * cols] += v;
    }
  }
  let mn = Infinity;
  let mx = 0;
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (const c of cells) {
    if (c <= 0) continue; // a cell that was never meant to be lit is not a defect
    mn = Math.min(mn, c);
    mx = Math.max(mx, c);
    sum += c;
    sumSq += c * c;
    n++;
  }
  if (n === 0) return { uniformity: 0, cells, min: 0, max: 0, mean: 0, cv: 0 };
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
  return {
    uniformity: mx + mn > 0 ? (2 * mn) / (mx + mn) : 1,
    cells,
    min: mn,
    max: mx,
    mean,
    cv: mean > 0 ? std / mean : 0,
  };
}

/** Human-readable metric dump used by the CLI tools and the server. */
export function summarizeMetrics(m) {
  const f = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : String(v));
  return [
    `efficiency      ${f(m.efficiency * 100, 2)} %`,
    `uniformity      ${f(m.uniformity * 100, 2)} %`,
    `flatness (1-cv) ${f(m.flatness * 100, 2)} %`,
    `contrast        ${f(m.contrast * 100, 2)} %`,
    `RMSE            ${f(m.rmse * 100, 2)} %`,
    `SSIM            ${f(m.ssim, 4)}`,
    `SNR             ${f(m.snr, 2)} dB`,
    `zero-order leak ${f(m.zeroOrder * 100, 3)} %`,
  ].join('\n');
}
