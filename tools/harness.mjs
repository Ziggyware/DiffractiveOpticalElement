// Runs the real index.html JavaScript against the CPU WebGL2 mock, drives a few
// frames and then runs the piece's own graded self-test.
//   node tools/harness.mjs [url-fragment]
//
// This is a stand-in for a browser run: it executes every line of application
// logic (render graph, GS loop, URL codec, self-test) but the *shader* maths
// comes from tools/glmock.mjs, not from a driver.  GLSL compilation, float16
// support and real GPU numerics stay unverified.
import fs from 'node:fs';
import vm from 'node:vm';
import { createGL, stats } from './glmock.mjs';

const ROOT = '/home/user/DiffractiveOpticalElement';
const html = fs.readFileSync(ROOT + '/index.html', 'utf8');
const frag = process.argv[2] || '';

/* ---------------- DOM mock ---------------- */
function makeEl(id) {
  const el = {
    id, style: {}, children: [], textContent: '', value: '', hidden: false, files: [],
    className: '', dataset: {},
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      toggle(c) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); },
      contains(c) { return this._s.has(c); },
    },
    appendChild(c) { this.children.push(c); return c; },
    append(...c) { this.children.push(...c); },
    remove() { },
    addEventListener() { }, removeEventListener() { },
    setPointerCapture() { }, click() { },
    getBoundingClientRect: () => ({ width: 900, height: 620, left: 0, top: 0 }),
    parentElement: { getBoundingClientRect: () => ({ width: 900, height: 620 }) },
    getContext(kind) { return kind === 'webgl2' ? gl : null; },
    captureStream() { return {}; },
    querySelectorAll: () => [],
  };
  return el;
}
const elements = new Map();
const gl = createGL({});
let rafCb = null;
const sandbox = {
  console,
  performance: { now: () => Number(process.hrtime.bigint()) / 1e6 },
  requestAnimationFrame: (cb) => { rafCb = cb; return 1; },
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout,
  setInterval: () => 0,
  atob: (b) => Buffer.from(b, 'base64').toString('binary'),
  btoa: (b) => Buffer.from(b, 'binary').toString('base64'),
  TextEncoder,
  Math, JSON, Date, Object, Array, Float32Array, Float64Array, Uint8Array, Uint32Array, Int16Array, Uint16Array, DataView, ArrayBuffer,
  isFinite, isNaN, parseFloat, parseInt, RegExp, String, Number, Boolean, Error, Map, Set, Promise,
  history: { replaceState() { } },
  matchMedia: () => ({ matches: false }),
  URL: { createObjectURL: () => 'blob:x', revokeObjectURL() { } },
  location: { hash: frag ? '#' + frag : '', search: '', pathname: '/' },
  navigator: { userAgent: 'node' },
  window: null,
  document: {
    readyState: 'complete',
    getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, makeEl(id));
      const el = elements.get(id);
      if (id === 'gl') { el.getContext = (k) => k === 'webgl2' ? gl : null; el.width = 900; el.height = 620; }
      return el;
    },
    createElement: (tag) => makeEl('created:' + tag),
    querySelectorAll: () => [],
    addEventListener() { },
    body: makeEl('body'),
    documentElement: makeEl('html'),
  },
};
sandbox.addEventListener = () => { };
sandbox.removeEventListener = () => { };
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
sandbox.self = sandbox;
sandbox.MediaRecorder = undefined;
sandbox.AudioContext = undefined;

/* the canvas the mock renders into when the default framebuffer is bound */
gl._canvasTex = { w: 900, h: 620, data: new Float32Array(900 * 620 * 4) };

/* shader sources: the mock needs them before the app compiles them */
const shaderSrc = {};
for (const m of html.matchAll(/<script id="([^"]+)" type="x-[^"]*">([\s\S]*?)<\/script>/g))
  shaderSrc[m[1]] = m[2];

/* rewrite the app so $("id").textContent returns the shader source for shaders */
const jsBlocks = [];
for (const m of html.matchAll(/<script(?![^>]*type=)[^>]*>([\s\S]*?)<\/script>/g)) jsBlocks.push(m[1]);
const code = jsBlocks.join('\n');

const ctx = vm.createContext(sandbox);
/* pre-fill the shader "elements" so getElementById(...).textContent works */
for (const id in shaderSrc) {
  const el = sandbox.document.getElementById(id);
  el.textContent = shaderSrc[id];
}
/* track attachShader so linkProgram can find the sources */
const realCreateProgram = gl.createProgram;
gl.createProgram = () => {
  const p = realCreateProgram();
  p.shaders = [];
  return p;
};
const realAttach = gl.attachShader;
gl.attachShader = (p, s, ...rest) => { if (!p.shaders.includes(s)) p.shaders.push(s); return realAttach(p, s, ...rest); };

try {
  vm.runInContext(code, ctx, { filename: 'index.html.js' });
} catch (e) {
  console.error('APPLICATION THREW:', e && e.stack || e);
  process.exit(1);
}

/* drive some frames */
const report = sandbox.DOE && sandbox.DOE.report;
if (!report) { console.error('boot did not run (window.DOE missing)'); process.exit(1); }

let t = 0;
for (let i = 0; i < 5; i++) {
  t += 16.7;
  try { rafCb(t); } catch (e) {
    console.error('FRAME ' + i + ' THREW:', e && e.stack || e);
    process.exit(1);
  }
}
console.log('5 frames rendered ·', JSON.stringify(report()));
{ const pr = sandbox.window.DOE.probe();
  console.log('probe: gsRMS', pr.gsRMS, 'gsIters', pr.gsIters);
  const holo = pr.holo.tex.data; let mn=1e9,mx=0;
  for (let i=0;i<holo.length;i+=4){const a=Math.hypot(holo[i],holo[i+1]);mn=Math.min(mn,a);mx=Math.max(mx,a);}
  console.log('holo field |f| range', mn.toFixed(4), mx.toFixed(4), 'texture', pr.holo.tex.w+'x'+pr.holo.tex.h);
  const I = pr.I[1].tex.data; let s2=0; for (let i=0;i<I.length;i+=4) s2+=I[i];
  console.log('green replay energy sum', s2.toFixed(6), 'tex', pr.I[1].tex.w+'x'+pr.I[1].tex.h);
  // design tap: the SEM levels should span the quantisation steps
  const D = pr.design.tex.data; let lmin=1e9,lmax=-1e9;
  for (let i=0;i<D.length;i+=4){ lmin=Math.min(lmin,D[i+2]); lmax=Math.max(lmax,D[i+2]); }
  console.log('design phase level range', lmin, lmax);
}
console.log('gl mock stats:', JSON.stringify(stats));

/* ------------------------------------------------------------------ *
 * Physics checks of the OUTPUT of the app's own pipeline (through the
 * mock shaders).  Each scene has a claim that must hold in the replay. */
function replayStats(ch) {
  const t = sandbox.window.DOE.probe().I[ch].tex;
  const N = t.w;
  const I = t.data;
  let total = 0, peak = 0, pi = 0;
  for (let i = 0; i < N * N; i++) { const v = I[4 * i]; total += v; if (v > peak) { peak = v; pi = i; } }
  const bin = (x, y) => { const bx = x < N / 2 ? x : x - N; const by = y < N / 2 ? y : y - N; return [bx, by]; };
  const at = (x, y) => I[4 * (((y % N + N) % N) * N + ((x % N + N) % N))];
  const radial = (r0, r1) => { let s = 0, n = 0; for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const [bx, by] = bin(x, y); const r = Math.hypot(bx, by); if (r >= r0 && r <= r1) { s += at(x, y); n++; } } return s / Math.max(1, n); };
  return { N, total, peak, pi, at, radial, mean: total / (N * N) };
}
function renderSceneOnce(scene, iters) {
  const D = sandbox.window.DOE;
  D.P.N = 256; D.setN(256);
  D.P.mix = 0;                       // no cross-fade: wTarget must be the pure scene
  D.setScene(scene);
  D.snapWeights();
  if (scene === 5) { D.P.holoWin = 0.5; D.setN(256); for (let i = 0; i < 3; i++) D.gsIterate(Math.max(1, Math.round((iters || 90) / 3))); }
  D.renderScene(0);
}
let vfail = 0;
const vok = (name, cond, detail) => { console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}   ${detail}`); if (!cond) vfail++; };

console.log('\n---- pipeline physics (through the mock shaders) ----');
{
  /* blaze: on-design a single spot at bin N/8 with the zero order nulled;
     coarser quantisation and off-design illumination both bring it back */
  renderSceneOnce(0);
  let st = replayStats(1);
  const want = st.N / 8;
  const bx = st.pi % st.N, by = (st.pi / st.N) | 0;
  const spot = st.at(want, 0) / st.total;
  vok('blaze: single spot at bin N/8', bx === want && by === 0 && spot > 0.9,
    `peak bin (${bx},${by}) of ${st.N} = N/8, ${(spot * 100).toFixed(1)}% of the energy in that one spot`);

  sandbox.window.DOE.P.levels = 4;                   // 4 divides the 8 samples: still an exact staircase
  sandbox.window.DOE.renderScene(0);
  const dc4 = replayStats(1).at(0, 0) / replayStats(1).total;
  sandbox.window.DOE.P.levels = 5;                   // odd counts leave the staircase uneven
  sandbox.window.DOE.renderScene(0);
  const dc5 = replayStats(1).at(0, 0) / replayStats(1).total;
  sandbox.window.DOE.P.levels = 8;
  sandbox.window.DOE.renderScene(0);
  const dc8 = replayStats(1).at(0, 0) / replayStats(1).total;

  sandbox.window.DOE.P.levels = 0;                   // "continuous": no etch steps at all
  sandbox.window.DOE.renderScene(0);
  const st0 = replayStats(1);
  const dc0 = st0.at(0, 0) / st0.total, spot0 = st0.at(st0.N / 8, 0) / st0.total;

  sandbox.window.DOE.P.lamIllum = 640e-9;            // fixed etch depth, r = lamDesign/lamIllum
  sandbox.window.DOE.renderScene(0);
  const st2 = replayStats(1);
  const r = 532 / 640;
  const want0 = Math.pow(Math.sin(Math.PI * r) / (8 * Math.sin(Math.PI * r / 8)), 2);
  const got0 = st2.at(0, 0) / st2.total;
  sandbox.window.DOE.P.lamIllum = 532e-9;
  /* quantised staircases: mean of e^(i 2 pi round(L k / 8) / L) over k, |.|^2 / 64 */
  const q = (L) => {
    let sr = 0, si = 0;
    for (let k = 0; k < 8; k++) { const l = Math.round(L * k / 8); sr += Math.cos(2 * Math.PI * l / L); si += Math.sin(2 * Math.PI * l / L); }
    return (sr * sr + si * si) / 64;
  };
  vok('zero order cancelled on design, revived by odd level counts and by off-design light',
    dc8 < 1e-6 && dc4 < 1e-6 && Math.abs(dc5 - q(5)) < 2e-3 && Math.abs(got0 - want0) < 0.005 &&
    dc0 < 1e-6 && spot0 > 0.9,
    `zero-order power: 8 levels ${dc8.toExponential(2)} (theory ~0), 4 levels ${dc4.toExponential(2)} ` +
    `(theory ~0), 5 levels ${dc5.toFixed(4)} vs theory ${q(5).toFixed(4)}; ` +
    `"continuous" ${dc0.toExponential(2)} with ${(spot0 * 100).toFixed(1)}% in the spot; ` +
    `at 640 nm ${got0.toFixed(4)} vs closed form ${want0.toFixed(4)}`);
}
{
  /* axicon: a bright ring at sin(theta) = NA, i.e. bin = NA*N*p/lambda */
  sandbox.window.DOE.P.axiconNA = 0.18;
  renderSceneOnce(2);
  const st = replayStats(1);
  const wantBin = 0.18 * st.N * sandbox.window.DOE.P.p / 532e-9;
  const inRing = st.radial(wantBin - 2, wantBin + 2);
  const nearCtr = st.radial(0, 2);
  vok('axicon: Bessel ring at sin(theta)=NA', inRing > nearCtr * 3,
    `ring/centre = ${(inRing / (nearCtr || 1e-30)).toFixed(2)} (bin ${wantBin.toFixed(0)})`);
}
{
  /* vortex: on-axis null (optical donut) for l != 0 */
  sandbox.window.DOE.P.charge = 3;
  renderSceneOnce(3);
  const st = replayStats(1);
  const ctr = st.at(0, 0);
  const ring = Math.max(st.radial(8, 20), 1e-30);
  const donut = st.total / (st.N * st.N);
  vok('vortex l=3: on-axis null', ctr < donut && ctr < ring,
    `centre ${ctr.toExponential(2)} vs mean ${donut.toExponential(2)} vs ring ${ring.toExponential(2)}`);
}
{
  /* hologram: the letters carry more energy than the background */
  const D = sandbox.window.DOE;
  D.P.text = 'DOE'; D.P.N = 256; D.setN(256); D.setScene(5); D.snapWeights();
  for (let i = 0; i < 5; i++) D.gsIterate(12);
  D.renderScene(0);
  const st = replayStats(1);
  const tgt = sandbox.window.DOE.gsTarget('DOE', D.holoM(), D.P.holoWin, 0, 0, D.P.holoEff);
  let lit = 0, nl = 0, bg = 0, nb = 0;
  for (let i = 0; i < st.N * st.N; i++) {
    const v = st.at(0, 0) * 0 + st.I ? 0 : 0;   // unused
  }
  const I = sandbox.window.DOE.probe().I[1].tex.data;
  const M = D.holoM(), N = st.N;
  // map the target grid onto the replay grid by direction sine (the target grid
  // spans the same window as the replay window in the GS model)
  for (let j = 0; j < M; j++) for (let i = 0; i < M; i++) {
    const gx = i % N, gy = j % N;
    const v = I[4 * (gy * N + gx)];
    if (tgt.lit[j * M + i]) { lit += v; nl++; } else { bg += v; nb++; }
  }
  const contrast = (lit / nl) / ((bg / nb) || 1e-30);
  vok('hologram: letters brighter than background', contrast > 3 && lit / nl > 1.2 * (1 / (N * N)),
    `contrast ${contrast.toFixed(2)}, mean lit ${(lit / nl).toExponential(2)} vs 1/N^2 = ${(1 / (N * N)).toExponential(2)}  (gsRMS ${sandbox.window.DOE.probe().gsRMS.toFixed(4)})`);
}
console.log(vfail ? `\n  ${vfail} pipeline physics check(s) failed\n` : '\n  all pipeline physics checks pass\n');

/* run the piece's own self-test */
const st = sandbox.window.__selftestResult;
// __selftestResult is only set by runSelfTest; call it through the button handler
const btn = elements.get('btnSelf');
if (btn && btn.onclick) btn.onclick();
const res = sandbox.window.__selftestResult;
if (res) {
  console.log('\n---- self-test output ----');
  for (const l of res.lines) console.log('  ' + l);
  console.log('  result:', res.pass ? 'ALL PASS' : 'FAILURES');
  process.exit(res.pass ? 0 : 2);
} else {
  console.log('self-test did not report');
  process.exit(3);
}
