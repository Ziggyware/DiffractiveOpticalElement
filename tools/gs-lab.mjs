// GS convergence lab: quality vs init / grid / iterations, plus a PNG of the
// reconstructed window for eyeballing.  node tools/gs-lab.mjs
import fs from 'node:fs';
import { encodePNG } from './png.mjs';
import { fft2d } from './ref-fft.mjs';
import { defaultParams } from './preview.mjs';
import { gsSolve, gsTarget, TAU } from './doe.mjs';

const base = { ...defaultParams(512), w: [0, 0, 0, 0, 0, 1], text: 'DOE' };

function analyse(res, M) {
  let e = 0, tot = 0, peak = 0;
  let inL = 0, nL = 0, outL = 0, nOut = 0;
  // res.field is the unit-modulus mask (never transform it twice)
  const Ff = fft2d(res.field.slice(), new Float64Array(2 * M * M), M);
  for (let i = 0; i < M * M; i++) {
    const I = Ff[2 * i] ** 2 + Ff[2 * i + 1] ** 2;
    tot += I; peak = Math.max(peak, I);
    if (res.target[i] > 0) { e += I; nL++; inL += I; } else { nOut++; outL += I; }
  }
  const meanLit = inL / (nL || 1), meanDark = outL / (nOut || 1);
  return { eff: e / tot, contrast: meanLit / (meanDark || 1e-30), peak, meanLit, meanDark };
}

function render(res, M, name, span = 0.55, cx = 0.5, cy = 0.5, gamma = 0.5) {
  const W = 300;
  const img = new Uint8Array(W * W * 3);
  const tmp = new Float64Array(2 * M * M);
  const Ff = fft2d(res.field.slice(), tmp, M);
  let peak = 0;
  for (let i = 0; i < M * M; i++) peak = Math.max(peak, Ff[2 * i] ** 2 + Ff[2 * i + 1] ** 2);
  for (let py = 0; py < W; py++)
    for (let px = 0; px < W; px++) {
      const x = cx * M - M / 2 + (px / W - 0.5) * span * M;
      const y = cy * M - M / 2 + (py / W - 0.5) * span * M;
      const xi = ((Math.round(x) % M) + M) % M, yi = ((Math.round(y) % M) + M) % M;
      const I = Ff[(yi * M + xi) * 2] ** 2 + Ff[(yi * M + xi) * 2 + 1] ** 2;
      const v = Math.round(255 * Math.min(1, Math.pow(I / (peak * 0.35), gamma)));
      const o = (py * W + px) * 3;
      img[o] = v; img[o + 1] = v; img[o + 2] = v;
    }
  fs.writeFileSync('/home/user/previews/' + name + '.png', encodePNG(img, W, W));
}

for (const M of [256, 512]) {
  for (const iters of [8, 20, 60]) {
    const r = gsSolve(base, M, iters, null);
    const a = analyse(r, M);
    console.log(
      `M=${M} iters=${iters}  rms(window)=${r.rms.toFixed(4)}  rms(target px)=${r.rmsWin.toFixed(4)}` +
      `  eff=${(r.efficiency * 100).toFixed(1)}%  img/bg=${a.contrast.toFixed(2)}`
    );
  }
}
const r = gsSolve(base, 512, 60, null);
render(r, 512, 'gs-window', 0.42, 0.665, 0.605);
console.log('wrote /home/user/previews/gs-window.png');
