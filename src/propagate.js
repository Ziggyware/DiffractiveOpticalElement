/**
 * propagate.js — scalar diffraction propagation.
 *
 * Two propagators, both derived from the Rayleigh–Sommerfeld / Helmholtz
 * solution and both *exactly* invertible in their discretisation, which is what
 * makes them safe to iterate inside a phase-retrieval loop:
 *
 *  1. Angular spectrum (plane-wave decomposition), uniform sampling:
 *
 *         H(fx, fy) = exp( i * 2*pi*z*sqrt(1/lambda^2 - fx^2 - fy^2) )
 *
 *     Exact for paraxial and non-paraxial fields, and (up to the evanescent
 *     cutoff) unitary — forward+backward returns the original field to machine
 *     precision. Band-limited per Matsushima & Shimobaba (Opt. Express 17,
 *     19662, 2009) so that no aliasing is introduced when the field spreads
 *     beyond the sampling window.
 *
 *  2. Single-FFT Fresnel (Fraunhofer-style), *magnified* sampling:
 *
 *         U2 = e^{ikz}/(i*lambda*z) * P_out * DFT{ U1 * P_in }
 *         P_in  = exp( i*pi*(x^2+y^2)/(lambda*z) )
 *         P_out = exp( i*pi*(u^2+v^2)/(lambda*z) )
 *         du = lambda*z/(N*dx)   (output pitch grows with z)
 *
 *     This is the physically correct kernel for a projection screen sitting at
 *     a distance, and it is energy conserving (Parseval), so radiometric
 *     comparisons (efficiency, non-uniformity) are meaningful.
 *
 * All quantities are SI. Fields are ComplexField instances (see field.js).
 */

import { ComplexField } from './field.js';
import { fft2d, ifftshift2d, fftshift2d } from './fft.js';

export const DEFAULT_WAVELENGTH = 532e-9; // green — the classic DOE/pointer line

/** Soft (raised-cosine) window: 1 below `inner`, 0 above `lim`. */
function softWindow(v, lim, inner) {
  if (v <= inner) return 1;
  if (v >= lim) return 0;
  return 0.5 * (1 + Math.cos((Math.PI * (v - inner)) / (lim - inner)));
}

/**
 * Local bandwidth limit of the angular spectrum at distance z (cycles/m).
 * Beyond this frequency the transfer function samples alias into the window.
 */
export function localFrequencyLimit(n, d, lambda, z) {
  const df = 1 / (n * d);
  return 1 / (lambda * Math.sqrt(Math.pow(2 * df * z, 2) + 1));
}

/**
 * Angular-spectrum transfer function on the centred (fftshift) frequency grid.
 * @returns {{re: Float64Array, im: Float64Array, propagatingFraction: number}}
 */
export function angularSpectrumTransferFunction(nx, ny, dx, dy, lambda, z, opts = {}) {
  const { bandLimit = true, evanescent = true, softEdge = 0.02 } = opts;
  const re = new Float64Array(nx * ny);
  const im = new Float64Array(nx * ny);
  const dfx = 1 / (nx * dx);
  const dfy = 1 / (ny * dy);
  const invL2 = 1 / (lambda * lambda);

  const fxLim = bandLimit ? localFrequencyLimit(nx, dx, lambda, z) : Infinity;
  const fyLim = bandLimit ? localFrequencyLimit(ny, dy, lambda, z) : Infinity;
  // Nyquist guard: never let the soft edge exceed what the grid can express.
  const fxn = 1 / (2 * dx);
  const fyn = 1 / (2 * dy);

  let propagating = 0;
  for (let j = 0; j < ny; j++) {
    const fy = (j - ny / 2) * dfy;
    const limY = Math.min(fyLim, fyn);
    const wy = fyLim === Infinity ? 1 : softWindow(Math.abs(fy), limY, limY * (1 - softEdge));
    for (let i = 0; i < nx; i++) {
      const fx = (i - nx / 2) * dfx;
      const limX = Math.min(fxLim, fxn);
      const wx = fxLim === Infinity ? 1 : softWindow(Math.abs(fx), limX, limX * (1 - softEdge));
      const arg = invL2 - fx * fx - fy * fy;
      const k = i + j * nx;
      if (arg >= 0) {
        const phase = 2 * Math.PI * z * Math.sqrt(arg);
        const w = wx * wy;
        re[k] = w * Math.cos(phase);
        im[k] = w * Math.sin(phase);
        if (w > 0.5) propagating++;
      } else if (evanescent) {
        re[k] = 0;
        im[k] = 0;
      } else {
        // Propagating correction (rarely wanted): keep the wave, drop decay.
        const w = wx * wy;
        const phase = 2 * Math.PI * z * Math.sqrt(-arg);
        re[k] = w * Math.cos(phase);
        im[k] = w * Math.sin(phase);
        propagating++;
      }
    }
  }
  return { re, im, propagatingFraction: propagating / (nx * ny) };
}

/** Angular spectrum propagation; the input field is left untouched. */
export function propagateAngularSpectrum(field, lambda, z, opts = {}) {
  const out = field.clone();
  propagateAngularSpectrumInPlace(out, lambda, z, opts);
  return out;
}

/** In-place angular spectrum propagation (z may be negative: back-propagation). */
export function propagateAngularSpectrumInPlace(field, lambda, z, opts = {}) {
  if (z === 0) return field;
  const { re, im, nx, ny } = field;
  ifftshift2d(re, im, nx, ny);
  fft2d(re, im, nx, ny, false);
  fftshift2d(re, im, nx, ny, true);

  const H = angularSpectrumTransferFunction(nx, ny, field.dx, field.dy, lambda, z, opts);
  for (let k = 0; k < re.length; k++) {
    const hr = H.re[k];
    const hi = H.im[k];
    const ur = re[k];
    const ui = im[k];
    re[k] = ur * hr - ui * hi;
    im[k] = ur * hi + ui * hr;
  }

  ifftshift2d(re, im, nx, ny);
  fft2d(re, im, nx, ny, true);
  fftshift2d(re, im, nx, ny, true);
  return field;
}

/**
 * Critically sampled distance for the single-FFT Fresnel kernel: below it the
 * chirp exp(i*pi*r^2/(lambda*z)) is under-sampled on the given grid and the
 * magnified transform aliases. Use the ASM instead in that regime (or crop the
 * input window).
 */
export function fresnelMinDistance(n, d, lambda) {
  return (n * d * d) / lambda;
}

/**
 * Single-FFT Fresnel propagation to a *magnified* output grid.
 *
 * @param {ComplexField} field input on the DOE grid
 * @param {number} lambda
 * @param {number} z propagation distance, > 0 (sign convention: forward)
 * @param {{throwOnUndersampled?: boolean}} opts
 * @returns {{field: ComplexField, dx: number, dy: number, z: number,
 *            magnification: number, undersampled: boolean, zCritical: number}}
 */
export function propagateFresnelFFT(field, lambda, z, opts = {}) {
  const { throwOnUndersampled = false } = opts;
  const { nx, ny, dx, dy } = field;
  const az = Math.abs(z);
  const k = (2 * Math.PI) / lambda;
  const zCritical = fresnelMinDistance(nx, dx, lambda);
  const undersampled = az < zCritical;
  if (undersampled && throwOnUndersampled) {
    throw new Error(
      `propagateFresnelFFT: z=${(az * 1e3).toFixed(3)} mm is below the critical distance ` +
        `z_c=${(zCritical * 1e3).toFixed(3)} mm for a ${nx}x${ny} grid at ${(dx * 1e6).toFixed(
          2,
        )} um pitch — the Fresnel chirp is under-sampled. Use the angular spectrum instead.`,
    );
  }

  const dux = (lambda * az) / (nx * dx);
  const duy = (lambda * az) / (ny * dy);

  // Discrete form of
  //   U2(u,v) = e^{ikz}/(i*lambda*z) * Int U1 * exp(i*pi*((u-x)^2+(v-y)^2)/(lambda*z))
  // expanded into two quadratic factors plus a Fourier kernel:
  //   exp(-2*pi*i*(u*x + v*y)/(lambda*z))
  //                   = e^{-2*pi*i*m*i/N} * (-1)^m * (-1)^i * e^{-i*pi*N/2}
  // on the *centred* grids x_i = (i-N/2)*dx, u_m = (m-N/2)*du with
  // du = lambda*z/(N*dx). The two checkerboards are therefore part of the exact
  // transform (not display cosmetics), and the residual e^{-i*pi*N/2} is a
  // global phase. Verified against a brute-force Fresnel integral in the tests.
  const re = new Float64Array(nx * ny);
  const im = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    const y = field.y(j);
    for (let i = 0; i < nx; i++) {
      const x = field.x(i);
      const q = (Math.PI * (x * x + y * y)) / (lambda * az);
      const c = (i + j) % 2 ? -Math.cos(q) : Math.cos(q);
      const s = (i + j) % 2 ? -Math.sin(q) : Math.sin(q);
      const idx = i + j * nx;
      const ur = field.re[idx];
      const ui = field.im[idx];
      re[idx] = ur * c - ui * s;
      im[idx] = ur * s + ui * c;
    }
  }

  fft2d(re, im, nx, ny, false); // unscaled forward DFT

  const amp = (dx * dy) / (lambda * az); // |prefactor|
  const kz = k * z - (Math.PI * (nx + ny)) / 2; // includes the global e^{-i*pi*N/2}
  const ckz = Math.cos(kz);
  const skz = Math.sin(kz);
  // e^{ikz} * (1/(i*lambda*z)) / |prefactor|^-1 = (sin(kz) - i*cos(kz)) * amp
  const pr = amp * skz;
  const pj = -amp * ckz;

  for (let j = 0; j < ny; j++) {
    const v = field.y(j) * 0 + (j - ny / 2) * duy;
    for (let i = 0; i < nx; i++) {
      const u = (i - nx / 2) * dux;
      const q = (Math.PI * (u * u + v * v)) / (lambda * az);
      const c = (i + j) % 2 ? -Math.cos(q) : Math.cos(q);
      const s = (i + j) % 2 ? -Math.sin(q) : Math.sin(q);
      const idx = i + j * nx;
      const ur = re[idx];
      const ui = im[idx];
      const tr = ur * pr - ui * pj;
      const ti = ur * pj + ui * pr;
      re[idx] = tr * c - ti * s;
      im[idx] = tr * s + ti * c;
    }
  }

  const out = new ComplexField(nx, ny, dux, duy);
  out.re = re;
  out.im = im;
  return {
    field: out,
    dx: dux,
    dy: duy,
    z,
    magnification: dux / dx,
    undersampled,
    zCritical,
  };
}

/**
 * Exact analytic inverse of propagateFresnelFFT: maps the magnified plane back
 * onto the DOE grid. Applying forward then inverse returns the input to machine
 * precision (test/propagate.test.js).
 */
export function propagateFresnelFFTInverse(field, lambda, z, opts = {}) {
  const { dx, dy } = opts;
  if (!dx || !dy) {
    throw new Error("propagateFresnelFFTInverse: pass the DOE-plane pitch as {dx, dy}");
  }
  const { nx, ny } = field;
  const az = Math.abs(z);
  const k = (2 * Math.PI) / lambda;
  const dux = field.dx;
  const duy = field.dy;

  const re = new Float64Array(nx * ny);
  const im = new Float64Array(nx * ny);

  // 1) conjugate output quadratic factor, with the output checkerboard
  for (let j = 0; j < ny; j++) {
    const v = (j - ny / 2) * duy;
    for (let i = 0; i < nx; i++) {
      const u = (i - nx / 2) * dux;
      const q = (Math.PI * (u * u + v * v)) / (lambda * az);
      const sg = (i + j) % 2 ? -1 : 1;
      const c = sg * Math.cos(q);
      const s = -sg * Math.sin(q);
      const idx = i + j * nx;
      const ur = field.re[idx];
      const ui = field.im[idx];
      re[idx] = ur * c - ui * s;
      im[idx] = ur * s + ui * c;
    }
  }

  // 2) the *inverse* DFT. Using the forward DFT twice would leave a point
  //    reflection behind, because DFT^2 = N^2 * P with P the index reversal —
  //    a trap that stays invisible on centred, symmetric test fields (a
  //    symmetric Gaussian is its own reflection) but breaks every general
  //    field. The inverse transform is therefore the honest one here.
  fft2d(re, im, nx, ny, true);

  // 3) divide by A = dx*dy*e^{i*kg}/(i*lambda*az):
  //    1/A = lambda*az*e^{-i*kg} / (dx*dy)
  //    with kg = k*z - pi*(Nx+Ny)/2 - pi/2  (the last term is 1/i).
  const amp = (lambda * az) / (dx * dy);
  const kg = k * z - (Math.PI * (nx + ny)) / 2 - Math.PI / 2;
  const cg = Math.cos(kg);
  const sg2 = Math.sin(kg);
  const pr = amp * cg;
  const pj = -amp * sg2;

  for (let j = 0; j < ny; j++) {
    const y = (j - ny / 2) * dy;
    for (let i = 0; i < nx; i++) {
      const x = (i - nx / 2) * dx;
      const q = (Math.PI * (x * x + y * y)) / (lambda * az);
      const sg = (i + j) % 2 ? -1 : 1;
      const c = sg * Math.cos(q);
      const s = -sg * Math.sin(q);
      const idx = i + j * nx;
      const ur = re[idx];
      const ui = im[idx];
      const tr = ur * pr - ui * pj;
      const ti = ur * pj + ui * pr;
      re[idx] = tr * c - ti * s;
      im[idx] = tr * s + ti * c;
    }
  }

  const out = new ComplexField(nx, ny, dx, dy);
  out.re = re;
  out.im = im;
  return out;
}

export function propagateFraunhofer(field, lambda, z, opts = {}) {
  return propagateFresnelFFT(field, lambda, z, opts);
}

/**
 * Field in the focal plane of a thin lens of focal length f placed immediately
 * after the element (the "lens-FFT" identity).
 *
 * A thin lens cancels the quadratic factor that the single-FFT Fresnel kernel
 * would otherwise need, so the focal-plane field is the *plain* Fourier
 * transform of the element field:
 *
 *      U_f(u, v) = (1/(i*lambda*f)) * exp(i*pi*(u^2+v^2)/(lambda*f)) * FFT{U_1}
 *      u_m       = (m - N/2) * lambda*f/(N*dx)
 *
 * Two properties make this the natural model for a holographic projection head:
 * the transform is exactly unitary (Parseval), so any efficiency quoted from it
 * is a real flux ratio; and the image scale lambda*f/dx is independent of N, so
 * changing the lens changes the throw without changing the blur.
 *
 * Mode matching: the focal-plane window N*du equals the element window N*dx
 * exactly when f = N*dx^2/lambda (the critical distance), which is why the
 * projector defaults to that throw distance in this mode.
 */
export function focalPlaneSpectrum(field, lambda, f) {
  const { nx, ny, dx, dy } = field;
  const re = Float64Array.from(field.re);
  const im = Float64Array.from(field.im);
  ifftshift2d(re, im, nx, ny);
  fft2d(re, im, nx, ny, false);
  fftshift2d(re, im, nx, ny, true);

  const dux = (lambda * f) / (nx * dx);
  const duy = (lambda * f) / (ny * dy);
  // The discretisation itself is already unitary: the output pitch grows in
  // exact proportion to the transform gain, so power in == power out.
  const amp0 = (dx * dy) / (lambda * f);
  for (let j = 0; j < ny; j++) {
    const v = (j - ny / 2) * duy;
    for (let i = 0; i < nx; i++) {
      const u = (i - nx / 2) * dux;
      const idx = i + j * nx;
      const q = (Math.PI * (u * u + v * v)) / (lambda * f) - Math.PI / 2;
      const amp = amp0;
      const c = amp * Math.cos(q);
      const s = amp * Math.sin(q);
      const ur = re[idx];
      const ui = im[idx];
      re[idx] = ur * c - ui * s;
      im[idx] = ur * s + ui * c;
    }
  }
  const out = new ComplexField(nx, ny, dux, duy);
  out.re = re;
  out.im = im;
  return out;
}

/**
 * Generic entry point.
 * @param {ComplexField} field
 * @param {number} lambda
 * @param {number} z
 * @param {{method?: 'asm'|'fresnel', pad?: number, bandLimit?: boolean}} opts
 */
export function propagate(field, lambda, z, opts = {}) {
  const { method = 'asm', pad = 1 } = opts;
  let src = field;
  if (pad > 1) src = field.resample(field.nx * pad, field.ny * pad);
  if (method === 'fresnel') return propagateFresnelFFT(src, lambda, z, opts).field;
  return propagateAngularSpectrum(src, lambda, z, { bandLimit: opts.bandLimit ?? true });
}

/**
 * Thin-lens phase: phi(x,y) = -k*(r^2)/(2f). Positive f converges.
 * @returns {Float64Array} wrapped phase in (-pi, pi]
 */
export function lensPhase(field, lambda, f, opts = {}) {
  const { cx = 0, cy = 0, wrap = true } = opts;
  const k = (2 * Math.PI) / lambda;
  const phi = new Float64Array(field.size);
  for (let j = 0; j < field.ny; j++) {
    const y = field.y(j) - cy;
    for (let i = 0; i < field.nx; i++) {
      const x = field.x(i) - cx;
      phi[i + j * field.nx] = (-k * (x * x + y * y)) / (2 * f);
    }
  }
  return wrap ? wrapPhase(phi) : phi;
}

/** Wrap phases into (-pi, pi]. */
export function wrapPhase(phi) {
  const twoPi = 2 * Math.PI;
  const pi = Math.PI;
  for (let i = 0; i < phi.length; i++) {
    let p = phi[i] % twoPi;
    if (p > pi) p -= twoPi;
    else if (p <= -pi) p += twoPi;
    phi[i] = p;
  }
  return phi;
}

/**
 * Sample the field as it flies through the volume: returns the irradiance on a
 * stack of planes between zStart and zEnd. This is what makes the depth of a
 * "true projection" visible — the image snaps into focus at exactly one plane.
 */
export function focusSweep(field, lambda, zStart, zEnd, steps, opts = {}) {
  const planes = [];
  for (let s = 0; s <= steps; s++) {
    const z = zStart + ((zEnd - zStart) * s) / steps;
    planes.push({ z, field: propagateAngularSpectrum(field, lambda, z, opts) });
  }
  return planes;
}
