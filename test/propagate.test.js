import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ComplexField, discMask, superGaussian } from '../src/field.js';
import {
  propagateAngularSpectrum,
  propagateFresnelFFT,
  propagateFresnelFFTInverse,
  fresnelMinDistance,
  angularSpectrumTransferFunction,
  localFrequencyLimit,
  lensPhase,
  wrapPhase,
} from '../src/propagate.js';

const LAMBDA = 532e-9;

function relativeError(a, b) {
  let num = 0;
  let den = 0;
  for (let k = 0; k < a.re.length; k++) {
    num += (a.re[k] - b.re[k]) ** 2 + (a.im[k] - b.im[k]) ** 2;
    den += b.re[k] ** 2 + b.im[k] ** 2;
  }
  return Math.sqrt(num / den);
}

function randomField(n, dx, seed = 5) {
  const f = new ComplexField(n, n, dx, dx);
  let s = seed;
  const rnd = () => {
    s = (1103515245 * s + 12345) % 2147483648;
    return s / 2147483648;
  };
  for (let k = 0; k < f.size; k++) {
    f.re[k] = rnd() - 0.5;
    f.im[k] = rnd() - 0.5;
  }
  return f;
}

function gaussianField(n, dx, w0) {
  const f = new ComplexField(n, n, dx, dx);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = f.x(i);
      const y = f.y(j);
      f.re[i + j * n] = Math.exp(-(x * x + y * y) / (w0 * w0));
    }
  }
  return f;
}

test('angular spectrum propagation is unitary and invertible', () => {
  const n = 64;
  const dx = 8e-6;
  const f = randomField(n, dx);
  const z = 25e-3;
  const fwd = propagateAngularSpectrum(f, LAMBDA, z, { bandLimit: false });
  const back = propagateAngularSpectrum(fwd, LAMBDA, -z, { bandLimit: false });
  assert.ok(relativeError(back, f) < 1e-8, `round trip ${relativeError(back, f)}`);
  assert.ok(Math.abs(fwd.power() / f.power() - 1) < 1e-12, 'power conserved');
});

test('angular spectrum reproduces the analytic Gaussian beam width', () => {
  // w(z) = w0 * sqrt(1 + (lambda z / (pi w0^2))^2)
  const n = 256;
  const dx = 6e-6;
  const w0 = 120e-6;
  const z = 0.1;
  const f = gaussianField(n, dx, w0);
  const out = propagateAngularSpectrum(f, LAMBDA, z, { bandLimit: false });
  const irr = out.intensity();
  let total = 0;
  let weighted = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const r2 = out.x(i) ** 2 + out.y(j) ** 2;
      const v = irr[i + j * n];
      total += v;
      weighted += v * r2;
    }
  }
  const measured = Math.sqrt(weighted / total);
  // Gaussian beam: w(z) = w0 sqrt(1 + (z/z_R)^2), z_R = pi w0^2 / lambda, and the
  // second-moment radius of I = exp(-2 r^2 / w^2) is w/sqrt(2).
  const analytic = w0 * Math.sqrt(1 + ((LAMBDA * z) / (Math.PI * w0 * w0)) ** 2);
  const analyticRms = analytic / Math.SQRT2;
  assert.ok(
    Math.abs(measured / analyticRms - 1) < 0.01,
    `measured ${measured.toExponential(4)} m vs analytic ${analyticRms.toExponential(4)} m`,
  );
});

test('single-FFT Fresnel matches a brute-force Fresnel integral', () => {
  // Direct evaluation of U2(u,v) = A * sum U1 * exp(i pi ((u-x)^2+(v-y)^2)/(lambda z))
  const n = 8;
  const dx = 20e-6;
  const z = 50e-3;
  const f = gaussianField(n, dx, 60e-6);
  f.im[10] = 0.25; // make it genuinely complex
  const du = (LAMBDA * z) / (n * dx);
  const A = (dx * dx) / (LAMBDA * z);
  const kz = (2 * Math.PI * z) / LAMBDA;
  const refR = new Float64Array(n * n);
  const refI = new Float64Array(n * n);
  for (let v = 0; v < n; v++) {
    for (let u = 0; u < n; u++) {
      const U = (u - n / 2) * du;
      const V = (v - n / 2) * du;
      let sr = 0;
      let si = 0;
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const x = f.x(i);
          const y = f.y(j);
          const q = (Math.PI * ((U - x) ** 2 + (V - y) ** 2)) / (LAMBDA * z) + kz - Math.PI / 2;
          const vr = f.re[i + j * n];
          const vi = f.im[i + j * n];
          sr += (vr * Math.cos(q) - vi * Math.sin(q)) * A;
          si += (vr * Math.sin(q) + vi * Math.cos(q)) * A;
        }
      }
      refR[u + v * n] = sr;
      refI[u + v * n] = si;
    }
  }
  const out = propagateFresnelFFT(f, LAMBDA, z).field;
  let num = 0;
  let den = 0;
  for (let k = 0; k < n * n; k++) {
    num += (out.re[k] - refR[k]) ** 2 + (out.im[k] - refI[k]) ** 2;
    den += refR[k] ** 2 + refI[k] ** 2;
  }
  assert.ok(Math.sqrt(num / den) < 1e-9, `rel err ${Math.sqrt(num / den)}`);
});

test('single-FFT Fresnel is energy conserving and exactly invertible', () => {
  const n = 128;
  const dx = 8e-6;
  const z = 30e-3;
  const f = randomField(n, dx, 21);
  const fwd = propagateFresnelFFT(f, LAMBDA, z);
  assert.ok(Math.abs(fwd.field.power() / f.power() - 1) < 1e-12, 'Parseval: power out == power in');
  const back = propagateFresnelFFTInverse(fwd.field, LAMBDA, z, { dx, dy: dx });
  assert.ok(relativeError(back, f) < 1e-8, `round trip ${relativeError(back, f)}`);
});

test('the two propagators agree where both are valid (z = z_c)', () => {
  // At the critical distance the magnified Fresnel grid pitch equals the
  // element pitch, so the two independent kernels must give the same intensity.
  const n = 128;
  const dx = 8e-6;
  const zc = fresnelMinDistance(n, dx, LAMBDA);
  const f = gaussianField(n, dx, 150e-6);
  const asm = propagateAngularSpectrum(f, LAMBDA, zc, { bandLimit: false });
  const fres = propagateFresnelFFT(f, LAMBDA, zc).field;
  const a = asm.intensity();
  const b = fres.intensity();
  let num = 0;
  let den = 0;
  for (let k = 0; k < a.length; k++) {
    num += (a[k] - b[k]) ** 2;
    den += a[k] * a[k];
  }
  assert.ok(Math.sqrt(num / den) < 1e-4, `intensity mismatch ${Math.sqrt(num / den)}`);
});

test('thin lens focuses a collimated beam at the focal length', () => {
  const n = 256;
  const dx = 8e-6;
  const f = 40e-3;
  const field = new ComplexField(n, n, dx, dx);
  const ap = discMask(n, n, dx, dx, ((n * dx) / 2) * 0.6, { soft: 0 }); // hard edge matches the analytic Airy solution
  const phi = lensPhase(field, LAMBDA, f, { wrap: false });
  field.setAmplitudePhase(ap, phi);
  const out = propagateAngularSpectrum(field, LAMBDA, f, { bandLimit: false });
  const irr = out.intensity();
  let peakIdx = 0;
  for (let k = 0; k < irr.length; k++) if (irr[k] > irr[peakIdx]) peakIdx = k;
  const i = peakIdx % n;
  const j = (peakIdx / n) | 0;
  assert.equal(i, n / 2, 'focus on axis in x');
  assert.equal(j, n / 2, 'focus on axis in y');
  // Airy first zero at r = 1.22 lambda f / D encircles 83.8% of the energy for a
  // uniformly illuminated circular aperture — a robust, sampling-insensitive
  // check (the 2nd moment of an Airy pattern diverges, so it cannot be used).
  const D = 2 * ((n * dx) / 2) * 0.6;
  const airy = (1.22 * LAMBDA * f) / D;
  const radii = [];
  let total = 0;
  for (let jj = 0; jj < n; jj++) {
    for (let ii = 0; ii < n; ii++) {
      const r = Math.hypot(out.x(ii), out.y(jj));
      const v = irr[ii + jj * n];
      total += v;
      radii.push([r, v]);
    }
  }
  radii.sort((a, b) => a[0] - b[0]);
  let acc = 0;
  let r838 = 0;
  for (const [r, v] of radii) {
    acc += v;
    if (acc / total >= 0.838) {
      r838 = r;
      break;
    }
  }
  assert.ok(
    Math.abs(r838 / airy - 1) < 0.15,
    `83.8% encircled-energy radius ${(r838 * 1e6).toFixed(2)} um vs Airy ${(airy * 1e6).toFixed(2)} um`,
  );
});

test('band-limited transfer function removes frequencies beyond the local limit', () => {
  const n = 128;
  const dx = 8e-6;
  const z = 0.05;
  const lim = localFrequencyLimit(n, dx, LAMBDA, z);
  const df = 1 / (n * dx);
  const H = angularSpectrumTransferFunction(n, n, dx, dx, LAMBDA, z, { bandLimit: true, softEdge: 0.01 });
  // Find the first index above the limit and check its modulus is ~0
  const idxOver = Math.floor(lim / df) + 4;
  let power = 0;
  for (const k of [idxOver, idxOver + 1, idxOver + 2]) {
    const i = n / 2 + k;
    if (i >= n) continue;
    const j = n / 2;
    const idx = i + j * n;
    power += H.re[idx] ** 2 + H.im[idx] ** 2;
  }
  assert.ok(power < 1e-6, `power beyond the band limit ${power}`);
});

test('wrapPhase maps into (-pi, pi]', () => {
  const p = wrapPhase(new Float64Array([0, Math.PI, -Math.PI, 3 * Math.PI, 3.5 * Math.PI, -3.5 * Math.PI]));
  assert.equal(p[0], 0);
  assert.ok(p[1] <= Math.PI && p[1] > 0);
  assert.ok(p[2] <= Math.PI && p[2] > -Math.PI - 1e-12);
  assert.ok(Math.abs(p[3] - Math.PI) < 1e-12 || Math.abs(p[3] + Math.PI) < 1e-12);
});

test('super-Gaussian helper produces the expected profile', () => {
  const a = superGaussian(64, 64, 8e-6, 8e-6, 100e-6, 10);
  const f = new ComplexField(64, 64, 8e-6, 8e-6);
  assert.ok(a[f.idx(32, 32)] > 0.999);
  assert.ok(a[f.idx(0, 0)] < 1e-3);
});
