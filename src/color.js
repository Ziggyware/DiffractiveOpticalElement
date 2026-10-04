/**
 * color.js — colour maps and plane-to-RGB conversion. Pure functions, no DOM,
 * so the same code renders in Node (CLI/PNG export) and in the browser.
 */

/** Piecewise-linear interpolation through a control-point colour map. */
function ramp(stops) {
  return (t) => {
    const x = Math.min(1, Math.max(0, t));
    for (let i = 1; i < stops.length; i++) {
      const [p0, c0] = stops[i - 1];
      const [p1, c1] = stops[i];
      if (x <= p1) {
        const f = p1 === p0 ? 0 : (x - p0) / (p1 - p0);
        return [
          Math.round(c0[0] + (c1[0] - c0[0]) * f),
          Math.round(c0[1] + (c1[1] - c0[1]) * f),
          Math.round(c0[2] + (c1[2] - c0[2]) * f),
        ];
      }
    }
    const c = stops[stops.length - 1][1];
    return [c[0], c[1], c[2]];
  };
}

export const COLORMAPS = {
  gray: ramp([
    [0, [0, 0, 0]],
    [1, [255, 255, 255]],
  ]),
  viridis: ramp([
    [0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1, [253, 231, 37]],
  ]),
  inferno: ramp([
    [0, [0, 0, 4]],
    [0.25, [87, 16, 110]],
    [0.5, [188, 55, 84]],
    [0.75, [249, 142, 9]],
    [1, [252, 255, 164]],
  ]),
  turbo: ramp([
    [0, [48, 18, 59]],
    [0.2, [65, 122, 246]],
    [0.4, [26, 219, 172]],
    [0.6, [175, 240, 91]],
    [0.8, [251, 126, 34]],
    [1, [122, 4, 3]],
  ]),
  magenta: ramp([
    [0, [0, 0, 0]],
    [0.5, [160, 20, 120]],
    [1, [255, 220, 255]],
  ]),
  phase: ramp([
    [0.0, [255, 60, 60]],
    [0.2, [255, 220, 40]],
    [0.4, [60, 230, 120]],
    [0.6, [40, 200, 255]],
    [0.8, [130, 80, 255]],
    [1.0, [255, 60, 60]],
  ]),
};

/**
 * Map a scalar plane to RGB bytes with a colour map.
 * @param {Float64Array|Float32Array} plane
 * @param {object} opts
 *   - map: colormap name (default viridis)
 *   - min, max: range; if omitted, computed from the data
 *   - gamma: display gamma (default 1)
 *   - log: logarithmic scaling (useful for diffraction patterns with huge
 *     dynamic range — the zero order would otherwise blow out the image)
 *   - logFloor: fraction of max mapped to black in log mode (default 1e-4)
 *   - percentile: use the given percentile (0..1] as max, to clip hot pixels
 */
export function planeToRGB(plane, opts = {}) {
  const {
    map = 'viridis',
    min = null,
    max = null,
    gamma = 1,
    log = false,
    logFloor = 1e-4,
    percentile = null,
    width = null,
    height = null,
  } = opts;
  const cmap = COLORMAPS[map] ?? COLORMAPS.viridis;
  let lo = min;
  let hi = max;
  if (lo === null || hi === null) {
    if (percentile !== null) {
      const sorted = Float64Array.from(plane).sort();
      hi = hi ?? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * percentile))];
      lo = lo ?? sorted[0];
    } else {
      let mn = Infinity;
      let mx = -Infinity;
      for (let k = 0; k < plane.length; k++) {
        const v = plane[k];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      lo = lo ?? mn;
      hi = hi ?? mx;
    }
  }
  const n = plane.length;
  const out = new Uint8Array(n * 3);
  const span = hi - lo || 1;
  const lf = Math.log10(Math.max(1e-12, logFloor));
  for (let k = 0; k < n; k++) {
    let t;
    if (log) {
      const v = Math.max(plane[k], hi * logFloor);
      t = (Math.log10(v / hi) - lf) / -lf; // 0 at hi*logFloor, 1 at hi
    } else {
      t = (plane[k] - lo) / span;
    }
    if (gamma !== 1) t = Math.pow(Math.max(0, Math.min(1, t)), gamma);
    const [r, g, b] = cmap(t);
    out[k * 3] = r;
    out[k * 3 + 1] = g;
    out[k * 3 + 2] = b;
  }
  return out;
}

/** Grayscale 8-bit plane (for exporting a height map / grayscale target). */
export function planeToGray(plane, { min = null, max = null, invert = false } = {}) {
  let lo = min;
  let hi = max;
  if (lo === null || hi === null) {
    let mn = Infinity;
    let mx = -Infinity;
    for (let k = 0; k < plane.length; k++) {
      if (plane[k] < mn) mn = plane[k];
      if (plane[k] > mx) mx = plane[k];
    }
    lo = lo ?? mn;
    hi = hi ?? mx;
  }
  const span = hi - lo || 1;
  const out = new Uint8Array(plane.length);
  for (let k = 0; k < plane.length; k++) {
    let t = Math.max(0, Math.min(1, (plane[k] - lo) / span));
    if (invert) t = 1 - t;
    out[k] = Math.round(t * 255);
  }
  return out;
}

/** Deep-red palette hint for a wavelength (useful for labelling renders). */
export function wavelengthColor(lambda) {
  const nm = lambda * 1e9;
  if (nm < 450) return [90, 60, 255];
  if (nm < 495) return [40, 140, 255];
  if (nm < 570) return [80, 255, 120];
  if (nm < 590) return [255, 220, 60];
  if (nm < 620) return [255, 140, 40];
  return [255, 70, 60];
}
