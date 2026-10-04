#!/usr/bin/env node
/**
 * render-demo.mjs — run the real design/simulation pipeline and write the
 * figures used by the README and the docs. Everything here comes out of the
 * library: no mock-ups, no external assets.
 *
 *   node tools/render-demo.mjs [--n 256] [--out docs/img]
 *
 * Outputs (docs/img/):
 *   target-text.png                 the target irradiance (the "slide")
 *   doe-phase.png                   the designed element's phase profile
 *   projection-imageplane.png        what the element projects at the design z
 *   projection-farfield.png          the same target through a Fourier-type head
 *   defocus-strip.png                projections along the axis (focus + away)
 *   convergence.png                  phase-retrieval error and efficiency
 *   spot-array.png                   an 8x8 spot-array generator
 *   hologram-record.png              an off-axis hologram recording
 *   hologram-reconstruct.png         and the image recovered from it
 *   modes-compared.png               image-plane vs far-field side by side
 */

import { mkdirSync } from 'node:fs';
import { ProjectorSystem, convergenceSweep } from '../src/projector.js';
import { encodePNGNode } from '../src/png.js';
import { planeToGray, planeToRGB } from '../src/color.js';
import { ComplexField } from '../src/field.js';
import { fft2d, fftshift2d, ifftshift2d } from '../src/fft.js';
import { recordHologram, reconstructHologram } from '../src/interference.js';

const args = process.argv.slice(2);
const getArg = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : dflt;
};
const N = getArg('n', 256);
const outIdx = args.indexOf('--out');
const OUT = outIdx >= 0 ? args[outIdx + 1] : 'docs/img';
mkdirSync(OUT, { recursive: true });

const write = async (name, { width, height, rgb = null, gray = null }) => {
  const png = await encodePNGNode({ width, height, rgb, gray });
  const { writeFileSync } = await import('node:fs');
  writeFileSync(`${OUT}/${name}`, png);
  console.log(`  ${OUT}/${name}  ${width}x${height}  ${(png.length / 1024).toFixed(1)} KiB`);
};

const t0 = Date.now();
console.log(`rendering demo figures at n=${N} ...`);

/* ------------------------------------------------------------------ *
 * 1. Image-plane projection of a text slide
 * ------------------------------------------------------------------ */
const imageSystem = new ProjectorSystem({
  n: N,
  pitch: 8e-6,
  lambda: 532e-9,
  distance: 30e-3,
  mode: 'image-plane',
  iterations: 80,
  seed: 12345,
  aperture: 0.95,
  apodization: 2,
  target: { kind: 'text', text: 'DOE' },
});
const imageDesign = imageSystem.design();
const imageField = imageSystem.simulate();
const metrics = imageSystem.screenMetrics(imageField);
const geometry = imageSystem.geometry();

console.log('image-plane design:', imageDesign.algorithm);
console.log(
  `  rmse ${(metrics.rmse * 100).toFixed(2)}%  correlation ${metrics.correlation.toFixed(4)}  ` +
    `efficiency ${(metrics.efficiency * 100).toFixed(1)}%  zero-order ${(metrics.zeroOrder * 100).toFixed(3)}%`,
);

await write('target-text.png', {
  width: N,
  height: N,
  gray: planeToGray(imageSystem.target.irradiance, { min: 0, max: 1 }),
});
await write('doe-phase.png', {
  width: N,
  height: N,
  rgb: planeToRGB(imageDesign.phase, { map: 'phase', min: -Math.PI, max: Math.PI }),
});
await write('projection-imageplane.png', {
  width: N,
  height: N,
  rgb: planeToRGB(imageField.intensity(), { map: 'inferno', percentile: 0.999 }),
});

/* ------------------------------------------------------------------ *
 * 2. The same target through a Fourier-type (holographic) head
 * ------------------------------------------------------------------ */
const farSystem = new ProjectorSystem({
  n: N,
  pitch: 8e-6,
  lambda: 532e-9,
  mode: 'far-field',
  iterations: 80,
  seed: 12345,
  aperture: 0.95,
  apodization: 2,
  target: { kind: 'text', text: 'DOE' },
});
farSystem.design();
const farField = farSystem.simulate();
const farMetrics = farSystem.screenMetrics(farField);
console.log(
  `far-field design: rmse ${(farMetrics.rmse * 100).toFixed(2)}%  correlation ${farMetrics.correlation.toFixed(4)}  ` +
    `efficiency ${(farMetrics.efficiency * 100).toFixed(1)}%  window ${(farSystem.geometry().screenWindowMm).toFixed(2)} mm`,
);
await write('projection-farfield.png', {
  width: N,
  height: N,
  rgb: planeToRGB(farField.intensity(), { map: 'inferno', percentile: 0.999 }),
});

/* ------------------------------------------------------------------ *
 * 3. Defocus strip: the image-plane head is sharp only at the design z
 * ------------------------------------------------------------------ */
{
  const zs = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6].map((f) => f * imageSystem.z);
  // Half-resolution frames: the strip is about focus, not pixel detail, and this
  // keeps the committed figure small.
  const M = Math.max(64, N >> 1);
  const bin = N / M;
  const strip = new Float64Array(M * M * zs.length);
  const labels = [];
  zs.forEach((z, idx) => {
    const irr = imageSystem.simulate({ z, pad: 1 }).intensity();
    let mx = 0;
    for (const v of irr) mx = Math.max(mx, v);
    for (let j = 0; j < M; j++) {
      for (let i = 0; i < M; i++) {
        let acc = 0;
        for (let bj = 0; bj < bin; bj++) {
          for (let bi = 0; bi < bin; bi++) acc += irr[i * bin + bi + (j * bin + bj) * N];
        }
        strip[idx * M * M + i + j * M] = mx > 0 ? acc / (bin * bin * mx) : 0;
      }
    }
    labels.push(`${(z * 1e3).toFixed(1)} mm`);
  });
  // lay the frames out in a row with a thin separator
  const gap = 4;
  const W = M * zs.length + gap * (zs.length - 1);
  const frame = new Float64Array(W * M);
  zs.forEach((z, idx) => {
    for (let j = 0; j < M; j++) {
      for (let i = 0; i < M; i++) {
        frame[i + idx * (M + gap) + j * W] = strip[idx * M * M + i + j * M];
      }
    }
  });
  console.log('defocus strip over', labels.join(', '));
  await write('defocus-strip.png', {
    width: W,
    height: M,
    rgb: planeToRGB(frame, { map: 'inferno', min: 0, max: 1 }),
  });
}

/* ------------------------------------------------------------------ *
 * 4. Convergence curves (error and efficiency vs iteration)
 * ------------------------------------------------------------------ */
{
  const sweepCfg = { n: 128, pitch: 8e-6, lambda: 532e-9, distance: 30e-3, mode: 'image-plane', target: { kind: 'text', text: 'DOE' } };
  const sweep = convergenceSweep(sweepCfg, { checkpoints: [1, 2, 3, 5, 8, 12, 18, 26, 36, 50, 70, 100] });
  const W = 560;
  const H = 320;
  const img = new Uint8Array(W * H * 3);
  // background
  for (let k = 0; k < W * H; k++) {
    img[k * 3] = 18;
    img[k * 3 + 1] = 18;
    img[k * 3 + 2] = 24;
  }
  const plot = (vals, lo, hi, color) => {
    let prev = null;
    vals.forEach((v, idx) => {
      const x = Math.round(40 + ((W - 60) * idx) / Math.max(1, vals.length - 1));
      const y = Math.round(H - 30 - ((H - 60) * (v - lo)) / (hi - lo || 1));
      if (prev) {
        const [px, py] = prev;
        const steps = Math.max(Math.abs(x - px), Math.abs(y - py));
        for (let s = 0; s <= steps; s++) {
          const xi = Math.round(px + ((x - px) * s) / steps);
          const yi = Math.round(py + ((y - py) * s) / steps);
          for (let d = -1; d <= 1; d++) {
            const k = xi + (yi + d) * W;
            if (k >= 0 && k < W * H) {
              img[k * 3] = color[0];
              img[k * 3 + 1] = color[1];
              img[k * 3 + 2] = color[2];
            }
          }
        }
      }
      prev = [x, y];
    });
  };
  // axes
  for (let x = 40; x < W - 20; x++) {
    const k = x + (H - 30) * W;
    img[k * 3] = 90;
    img[k * 3 + 1] = 90;
    img[k * 3 + 2] = 100;
  }
  for (let y = 30; y < H - 30; y++) {
    const k = 40 + y * W;
    img[k * 3] = 90;
    img[k * 3 + 1] = 90;
    img[k * 3 + 2] = 100;
  }
  const rmse = sweep.map((s) => s.rmse);
  const eff = sweep.map((s) => s.efficiency);
  plot(rmse, 0, Math.max(...rmse) || 1, [255, 90, 80]);
  plot(eff, 0, 1, [90, 220, 140]);
  console.log('convergence:', sweep.map((s) => `${s.iterations}:${(s.rmse * 100).toFixed(0)}%`).join(' '));
  await write('convergence.png', { width: W, height: H, rgb: img });
}

/* ------------------------------------------------------------------ *
 * 5. Spot-array generator (the classic multispot DOE)
 * ------------------------------------------------------------------ */
{
  const n = 128;
  const sys = new ProjectorSystem({
    n,
    pitch: 8e-6,
    lambda: 532e-9,
    distance: 25e-3,
    mode: 'image-plane',
    iterations: 60,
    seed: 7,
    target: { kind: 'dots', cols: 8, rows: 8, sigma: 0.15 },
  });
  sys.design();
  const m = sys.screenMetrics(sys.simulate());
  console.log(
    `spot array: rmse ${(m.rmse * 100).toFixed(1)}%  efficiency ${(m.efficiency * 100).toFixed(1)}%  ` +
      `spot uniformity ${(m.spotUniformity * 100).toFixed(1)}%  ssim ${m.ssim.toFixed(3)}`,
  );
  await write('spot-array.png', {
    width: n,
    height: n,
    rgb: planeToRGB(sys.simulate().intensity(), { map: 'inferno', percentile: 0.999 }),
  });
}

/* ------------------------------------------------------------------ *
 * 6. Off-axis hologram: record, then recover the image from the sideband
 * ------------------------------------------------------------------ */
{
  const n = 128;
  const dx = 8e-6;
  const lambda = 532e-9;
  const carrier = 16; // fringes across the window
  const sinTheta = (lambda * carrier) / (n * dx);
  const k = (2 * Math.PI) / lambda;
  const object = new ComplexField(n, n, dx, dx);
  const reference = new ComplexField(n, n, dx, dx);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = object.x(i);
      const y = object.y(j);
      object.re[i + j * n] =
        0.05 +
        Math.max(
          Math.exp(-((x - 120e-6) ** 2 + (y + 60e-6) ** 2) / (120e-6) ** 2),
          0.8 * Math.exp(-((x + 180e-6) ** 2 + (y - 120e-6) ** 2) / (70e-6) ** 2),
        );
      reference.re[i + j * n] = Math.cos(k * sinTheta * x);
      reference.im[i + j * n] = Math.sin(k * sinTheta * x);
    }
  }
  const { plate, recorded } = recordHologram(object, reference, { encode: 'raw' });
  const { real } = reconstructHologram(plate, reference, lambda, 5e-3);
  await write('hologram-record.png', { width: n, height: n, rgb: planeToRGB(recorded, { map: 'gray' }) });

  // Isolate the real image: transform the reconstruction, keep the sideband
  // around the carrier, and transform back. This is the numerical equivalent of
  // a spatial filter in the reconstruction beam.
  const irr = real.intensity();
  let mean = 0;
  for (const v of irr) mean += v;
  mean /= irr.length;
  const re = new Float64Array(irr.length);
  const im = new Float64Array(irr.length);
  for (let p = 0; p < irr.length; p++) re[p] = irr[p] - mean;
  fft2d(re, im, n, n);
  fftshift2d(re, im, n, n);
  let best = -1;
  let bestI = 0;
  for (let i = 0; i < n; i++) {
    const fx = i - n / 2;
    if (Math.abs(fx) < 6 || Math.abs(fx) > 26) continue;
    let energy = 0;
    for (let j = 0; j < n; j++) {
      const fy = j - n / 2;
      if (Math.abs(fy) > 3) continue;
      const p = i + j * n;
      energy += re[p] * re[p] + im[p] * im[p];
    }
    if (energy > best) {
      best = energy;
      bestI = i;
    }
  }
  const fc = bestI - n / 2;
  for (let j = 0; j < n; j++) {
    const fy = j - n / 2;
    for (let i = 0; i < n; i++) {
      const p = i + j * n;
      const fx = i - n / 2;
      if (Math.abs(fx - fc) <= 6 && Math.abs(fy) <= 6) continue;
      re[p] = 0;
      im[p] = 0;
    }
  }
  ifftshift2d(re, im, n, n);
  fft2d(re, im, n, n, true);
  const env = new Float64Array(irr.length);
  for (let p = 0; p < irr.length; p++) env[p] = Math.hypot(re[p], im[p]);
  console.log(`hologram: carrier at ${fc} cycles/window, image recovered from the sideband`);
  await write('hologram-reconstruct.png', { width: n, height: n, rgb: planeToRGB(env, { map: 'inferno' }) });
}

/* ------------------------------------------------------------------ *
 * 7. Modes compared: the same text through both head types
 * ------------------------------------------------------------------ */
{
  const strip = new Float64Array(N * 2 * N + 8 * N);
  const W = N * 2 + 8;
  const put = (src, col) => {
    let mx = 0;
    for (const v of src) mx = Math.max(mx, v);
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        strip[i + col * (N + 8) + j * W] = mx > 0 ? src[i + j * N] / mx : 0;
      }
    }
  };
  put(imageField.intensity(), 0);
  put(farField.intensity(), 1);
  await write('modes-compared.png', {
    width: W,
    height: N,
    rgb: planeToRGB(strip, { map: 'inferno', min: 0, max: 1 }),
  });
}

/* ------------------------------------------------------------------ *
 * report
 * ------------------------------------------------------------------ */
console.log(`\ngeometry: aperture ${geometry.apertureMm.toFixed(3)} mm, screen window ${geometry.screenWindowMm.toFixed(3)} mm,`);
console.log(`  f/${geometry.fNumber.toFixed(2)}, max deflection ${geometry.maxDeflectionDeg.toFixed(2)} deg, ~${Math.round(geometry.resolvableSpotsPerAxis)} resolvable spots per axis`);
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
