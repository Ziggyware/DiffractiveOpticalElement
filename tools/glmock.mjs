// A CPU mock of the small slice of WebGL2 that index.html uses.
//
// It exists so tools/harness.mjs can execute the real application code (the
// exact JS from index.html, unmodified) without a GPU: every fragment shader
// gets a hand-written JS twin here, the FFT ping-pong and the render graph run
// for real, and gl.readPixels hands data back so the graded self-test can run
// end to end.  What this *cannot* check is the driver-side parts: GLSL
// compilation, float16 support, texture-format legality and actual GPU
// numerics.  Those are stated as unverified in the deliverable.

const K = {
  DEPTH_TEST: 1, BLEND: 2, CULL_FACE: 3,
  TEXTURE_2D: 3553, TEXTURE_MIN_FILTER: 10241, TEXTURE_MAG_FILTER: 10240,
  TEXTURE_WRAP_S: 10242, TEXTURE_WRAP_T: 10243, NEAREST: 9728, CLAMP_TO_EDGE: 33071,
  RGBA32F: 34836, RGBA16F: 34842, RG32F: 33328, RG16F: 33327, R8: 33321,
  RGBA: 6408, RG: 33319, RED: 6403, FLOAT: 5126, HALF_FLOAT: 5131, UNSIGNED_BYTE: 5121,
  FRAMEBUFFER: 36160, COLOR_ATTACHMENT0: 36064, FRAMEBUFFER_COMPLETE: 36053,
  VERTEX_SHADER: 35633, FRAGMENT_SHADER: 35632, COMPILE_STATUS: 35713, LINK_STATUS: 35714,
  ACTIVE_UNIFORMS: 35718, TRIANGLES: 4, COLOR_BUFFER_BIT: 16384,
  SRC_ALPHA: 770, ONE_MINUS_SRC_ALPHA: 771, TEXTURE0: 33984, NO_ERROR: 0,
};

const TAU = 6.283185307179586;
const clamp = (x, a, b) => x < a ? a : x > b ? b : x;

/* ------------------------------------------------------------------ */
/* fragment shader twins.  Each takes (gl, prog, ctx) and returns the four
   channels for pixel (x, y) in *fragment* coordinates.                 */
/* ------------------------------------------------------------------ */
let stats = { reads: 0, writes: 0, nan: 0, racy: 0 };

function texel(t, x, y) {
  if (!t) return [0, 0, 0, 0];
  x = clamp(Math.round(x), 0, t.w - 1);
  y = clamp(Math.round(y), 0, t.h - 1);
  const o = (y * t.w + x) * 4;
  const d = t.data;
  return [d[o], d[o + 1], d[o + 2], d[o + 3]];
}
const u = (p, n) => p.uval[n];

/* fs.fft */
function fsFFT(gl, prog, x, y) {
  const N = u(prog, 'uN') | 0, bits = u(prog, 'uBits') | 0, stage = u(prog, 'uStage') | 0;
  const axis = u(prog, 'uAxis') | 0, perm = u(prog, 'uPerm') | 0;
  const sign = u(prog, 'uSign'), norm = u(prog, 'uNorm');
  const src = gl._bound[0];
  const phys = axis === 0 ? x : y, oth = axis === 0 ? y : x;
  let l = phys;
  if (perm === 1) { l = 0; let v = phys; for (let i = 0; i < bits; i++) { l = (l << 1) | (v & 1); v >>= 1; } }
  const half = N >> 1;
  const i = l >> 1, k = l >> (stage + 1), M = N >> stage;
  const a = texel(src, ...(axis === 0 ? [i, oth] : [oth, i]));
  const b = texel(src, ...(axis === 0 ? [i + half, oth] : [oth, i + half]));
  let r0, r1;
  if ((l & 1) === 0) { r0 = a[0] + b[0]; r1 = a[1] + b[1]; }
  else {
    const ang = sign * TAU * k / M, c = Math.cos(ang), s = Math.sin(ang);
    const d0 = a[0] - b[0], d1 = a[1] - b[1];
    r0 = d0 * c - d1 * s; r1 = d0 * s + d1 * c;
  }
  return [r0 * norm, r1 * norm, 0, 1];
}

/* fs.tap */
function fsTAP(gl, prog, x, y) {
  const N = u(prog, 'uN'), P = u(prog, 'uP'), lamD = u(prog, 'uLamD');
  const scale = u(prog, 'uScale'), levels = u(prog, 'uLevels') | 0;
  const W = prog.uval['uW'] || [0, 0, 0, 0, 0, 0];
  const holoM = u(prog, 'uHoloM') | 0;
  const holo = gl._bound[1];
  const gx = x + 0.5 - 0.5, gy = y + 0.5 - 0.5;
  const X = (gx - N * 0.5 + 0.5) * P, Y = (gy - N * 0.5 + 0.5) * P;
  const r = Math.hypot(X, Y);
  let phi = 0;
  if (W[0] > 0) {
    const ix = x;                                        // floor(gl_FragCoord.x) with a pixel-centre convention
    const fr = ((((ix - N * 0.5) / u(prog, 'uGratingPx')) % 1) + 1) % 1;   // GLSL fract()
    phi += W[0] * TAU * fr;
  }
  if (W[1] > 0) phi += W[1] * (-Math.PI * (X * X + Y * Y) / (lamD * u(prog, 'uZflF')));
  if (W[2] > 0) phi += W[2] * (-TAU * u(prog, 'uAxicon') * r / lamD);
  if (W[3] > 0) phi += W[3] * u(prog, 'uCharge') * Math.atan2(Y, X);
  if (W[4] > 0) {
    const rot = u(prog, 'uRotCell'), c = Math.cos(rot), s = Math.sin(rot), cell = u(prog, 'uCellPx') * P;
    const T = prog.uval['uDT'] || [], n = u(prog, 'uDTn') | 0;
    const seg = (t) => { let v = 0; for (let i = 0; i < n && i < 9; i++) if (t >= T[i]) v = 1 - v; return v; };
    const fr = (a) => a - Math.floor(a);
    const vx = seg(fr((X * c + Y * s) / cell)), vy = seg(fr((-X * s + Y * c) / cell));
    phi += W[4] * Math.PI * (vx !== vy ? 1 : 0);
  }
  if (W[5] > 0) {
    const uu = (x + 0.5) / N, vv = (y + 0.5) / N;
    const hx = uu * holoM - 0.5, hy = vv * holoM - 0.5;
    const x0 = Math.floor(hx), y0 = Math.floor(hy), fx = hx - x0, fy = hy - y0;
    let ar = 0, ai = 0;
    for (let b = 0; b < 4; b++) {
      const cx = clamp(x0 + (b & 1), 0, holoM - 1), cy = clamp(y0 + (b >> 1), 0, holoM - 1);
      const wgt = ((b & 1) ? fx : 1 - fx) * ((b >> 1) ? fy : 1 - fy);
      const t = texel(holo, cx, cy);
      ar += wgt * t[0]; ai += wgt * t[1];
    }
    phi += W[5] * Math.atan2(ai, ar);
  }
  let lvl = 0, q = phi;                       // uLevels < 2 == continuous
  if (levels >= 2) { const st = TAU / levels; lvl = Math.floor(phi / st + 0.5); q = lvl * st; }
  return [Math.cos(q * scale), Math.sin(q * scale), lvl, 1];
}

/* fs.intensity */
function fsINT(gl, prog, x, y) {
  const c = texel(gl._bound[0], x, y);
  return [c[0] * c[0] + c[1] * c[1], 0, 0, 1];
}
/* fs.reduce */
function fsRED(gl, prog, x, y) {
  const s = gl._bound[0];
  const sel = prog.uval['uSel'] || [0, 0, 1];
  let sum = [0, 0, 0, 0];
  for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    const t = texel(s, 2 * x + dx, 2 * y + dy);
    for (let i = 0; i < 4; i++) sum[i] += t[i];
  }
  let acc = sum[0] * sel[0] + sum[1] * sel[1] + sum[2] * sel[2];
  if (u(prog, 'uMax') > 0.5) acc = Math.max(sum[0], sum[1], sum[2], sum[3]);
  return [acc, 0, 0, 1];
}
/* fs.gsmask */
function fsMASK(gl, prog, x, y) {
  const c = texel(gl._bound[0], x, y);
  const m = Math.hypot(c[0], c[1]);
  if (m < 1e-24) return [1, 0, 0, 1];
  let ph = Math.atan2(c[1], c[0]);
  const q = u(prog, 'uQuant') | 0;
  if (q >= 2) { const st = TAU / q; ph = Math.floor(ph / st + 0.5) * st; }
  return [Math.cos(ph), Math.sin(ph), 0, 1];
}
/* fs.gsreplay */
function fsGREP(gl, prog, x, y) {
  const c = texel(gl._bound[0], x, y);
  const a = texel(gl._bound[1], x, y)[0];
  const m = Math.hypot(c[0], c[1]);
  const ux = m > 1e-24 ? c[0] / m : 1, uy = m > 1e-24 ? c[1] / m : 0;
  const d = m - a;
  return [ux * a, uy * a, d * d, 1];
}
/* fs.sem */
function fsSEM(gl, prog, x, y) {
  const R = prog.uval['uRect'], rect = [R[2], R[3]];
  const fx = (x - R[0]) / rect[0], fy = (y - R[1]) / rect[1];
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return [0, 0, 0, 1];
  const zoom = u(prog, 'uZoom'), N = u(prog, 'uN') | 0;
  const pan = prog.uval['uPan'] || [0, 0];
  const uu = 0.5 + pan[0] + (fx - 0.5) / zoom, vv = 0.5 + pan[1] + (fy - 0.5) / zoom;
  const mx = uu * N, my = vv * N;
  const tapT = gl._bound[0];
  const t = (dx, dy) => texel(tapT, Math.floor(mx) + dx, Math.floor(my) + dy);
  const dphi = (a, b) => Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
  const mpp = N / (zoom * rect[0]);
  const h = clamp(Math.floor(mpp), 1, 8);
  const t0 = t(0, 0);
  const Pp = u(prog, 'uP');
  const dxr = 0.5 * (dphi(t0, t(h, 0)) - dphi(t(-h, 0), t0)) / (h * Pp);
  const dyr = 0.5 * (dphi(t0, t(0, h)) - dphi(t(0, -h), t0)) / (h * Pp);
  const hUnit = u(prog, 'uLamD') / Math.max(u(prog, 'uNmat'), 1e-3);
  let hx = dxr * hUnit / TAU, hy = dyr * hUnit / TAU;
  const flat = u(prog, 'uFlat');
  const rough = flat * (1 - 0.5 * Math.min(1, zoom / 10));
  const rnd = (px, py, s) => {
    let z = (Math.imul(px | 0, 1973) + Math.imul(py | 0, 9277) + Math.imul(s | 0, 26699)) | 0;
    z = ((z ^ 61) ^ (z >>> 16)) >>> 0; z = (z + (z << 3)) >>> 0; z = (z ^ (z >>> 4)) >>> 0;
    z = Math.imul(z, 0x27d4eb2d) >>> 0; z = (z ^ (z >>> 15)) >>> 0;
    return z / 4294967296;
  };
  hx += (rnd(mx, my, u(prog, 'uSeed')) - 0.5) * rough;
  hy += (rnd(mx * 0.37 + 11, my * 0.37 + 11, u(prog, 'uSeed') + 3) - 0.5) * rough;
  const nl = Math.hypot(-hx, -hy, 1);
  const lam = Math.max(0, (-hx / nl) * -0.42 + (-hy / nl) * -0.55 + (1 / nl) * 0.72);
  const edge = Math.min(1, Math.hypot(hx, hy) * 0.55);
  let s = 0.16 + 0.6 * lam + 0.42 * Math.pow(edge, 1.3);
  s += (rnd(mx * 3.1, my * 3.1, u(prog, 'uSeed') + 7) - 0.5) * 0.05;
  s *= 0.94 + 0.06 * Math.cos(fy * 640);
  s *= 1 - 0.3 * Math.pow(Math.hypot(fx - 0.5, fy - 0.5) * 1.4, 3);
  let lum = clamp(Math.pow(clamp(s, 0, 1), 1 / 1.6), 0, 1);
  const span = (1 / zoom) * N * Pp;
  const barLen = clamp(u(prog, 'uBarLenM') / span, 0.04, 0.55) * rect[0];
  const b0x = R[0] + rect[0] * 0.07, b0y = R[1] + rect[1] * 0.90;
  const qx = x - b0x, qy = y - b0y;
  if (qy > -1.2 && qy < 1.2 && qx > -1 && qx < barLen + 1) lum = 1;
  if ((Math.abs(qx) < 2 || Math.abs(qx - barLen) < 2) && qy > -6 && qy < 1.2) lum = 1;
  return [lum * 1.0, lum * 1.0, lum * 1.04, 1];
}
/* fs.replay */
function fsREP(gl, prog, x, y) {
  const R = prog.uval['uRect'], rect = [R[2], R[3]];
  const fx = (x - R[0]) / rect[0], fy = (y - R[1]) / rect[1];
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return [0, 0, 0, 1];
  const N = u(prog, 'uN') | 0, Pp = u(prog, 'uP');
  const S = u(prog, 'uS'), zoom = u(prog, 'uZoom');
  const pan = prog.uval['uPan'] || [0, 0];
  const sc = Math.max(rect[0], rect[1]) * 0.5 / S * zoom;
  const sx = (x - (R[0] + rect[0] * 0.5)) / sc + pan[0];
  const sy = (y - (R[1] + rect[1] * 0.5)) / sc + pan[1];
  const sample = (t, binx, biny) => {
    const w = (a) => { const v = a % N; return v < 0 ? v + N : v; };
    const xf = w(binx) - 0.5, yf = w(biny) - 0.5;
    const x0 = Math.floor(xf), y0 = Math.floor(yf), ax = xf - x0, ay = yf - y0;
    const g = (a, b) => texel(t, ((a % N) + N) % N, ((b % N) + N) % N)[0];
    return (g(x0, y0) * (1 - ax) + g(x0 + 1, y0) * ax) * (1 - ay) +
      (g(x0, y0 + 1) * (1 - ax) + g(x0 + 1, y0 + 1) * ax) * ay;
  };
  const chan = (t, lam) => {
    const bx = sx * N * Pp / lam, by = sy * N * Pp / lam;
    if (bx * bx + by * by > N * N * 0.25) return 0;
    return sample(t, bx, by);
  };
  const lamR = u(prog, 'uLamR'), lamG = u(prog, 'uLamG'), lamB = u(prog, 'uLamB');
  const CR = prog.uval['uColR'] || [1, 0, 0], CG = prog.uval['uColG'] || [0, 1, 0], CB = prog.uval['uColB'] || [0, 0, 1];
  const g = Math.pow(2, u(prog, 'uExp'));
  const IR = chan(gl._bound[0], lamR) * g, IG = chan(gl._bound[1], lamG) * g, IB = chan(gl._bound[2], lamB) * g;
  let col = [IR * CR[0] + IG * CG[0] + IB * CB[0],
             IR * CR[1] + IG * CG[1] + IB * CB[1],
             IR * CR[2] + IG * CG[2] + IB * CB[2]];
  const meanI = 1 / (N * N);
  col = col.map((c) => Math.log2(1 + (c / meanI) * 0.65) / 7);
  if (u(prog, 'uMark') > 0.5) {
    const ctrx = R[0] + rect[0] * 0.5 - pan[0] * sc, ctry = R[1] + rect[1] * 0.5 - pan[1] * sc;
    const dr = Math.hypot(x - ctrx, y - ctry);
    const dash = (Math.atan2(y - ctry, x - ctrx) / TAU * 40) % 1 >= 0.5 ? 1 : 0;
    const ring = (rad) => (1 - smoothstep(0, 1.3, Math.abs(dr - rad))) * dash;
    const rR = ring(lamR * 0.5 / Pp * sc), rG = ring(lamG * 0.5 / Pp * sc), rB = ring(lamB * 0.5 / Pp * sc);
    for (let i = 0; i < 3; i++) col[i] += 0.055 * (rR * CR[i] + rG * CG[i] + rB * CB[i]);
    const dx = Math.abs(x - ctrx), dy = Math.abs(y - ctry);
    const cross = ((dx < 6 && dy < 0.55) || (dy < 6 && dx < 0.55)) ? 0.05 : 0;
    const tt = (0.7 + 0.3 * Math.sin(u(prog, 'uT') * 1.9));
    for (let i = 0; i < 3; i++) col[i] += cross * tt + (1 - smoothstep(7, 11, dr)) * 0.03 * tt;
  }
  const vig = 1 - 0.26 * Math.pow(Math.hypot(fx - 0.5, fy - 0.5) * 1.4, 3);
  return col.map((c) => Math.pow(Math.max(c, 0) * vig, 1 / 2.2)).concat([1]);
}
function smoothstep(a, b, t) { const x = clamp((t - a) / (b - a), 0, 1); return x * x * (3 - 2 * x); }
/* fs.over */
function fsOVER(gl, prog, x, y) {
  const W = prog.uval['uRes'], split = u(prog, 'uSplit');
  let a = 0.85 * (1 - smoothstep(0, 1.1, Math.abs(x - split)));
  let c = [0.22, 0.26, 0.32];
  if (u(prog, 'uBorder') > 0.5) {
    const e = Math.min(Math.min(x, y), Math.min(W[0] - 1 - x, W[1] - 1 - y));
    const b = 1 - smoothstep(0, 2, e);
    if (b > 0) { a = Math.max(a, b * 0.55); c = c.map((v) => v * (1 - b) + [0.55, 0.42, 0.22][0] * b); }
  }
  if (u(prog, 'uWmOn') > 0.5) {
    const sc = Math.max(1, Math.floor(W[1] / 260));
    const size = prog.uval['uWmSize'] || [1, 1];
    const ox = W[0] - (size[0] * sc + 12), oy = W[1] - (size[1] * sc + 12);
    const ux = (x - ox) / sc, uy = (y - oy) / sc;
    if (ux >= 0 && uy >= 0 && ux <= size[0] && uy <= size[1]) {
      const g = texel(gl._bound[0], ux, uy)[0];
      if (g > 0.5) { a = Math.max(a, 0.8); c = [0.92, 0.94, 0.97]; }
    }
  }
  return [c[0], c[1], c[2], a];
}

const IMPL = {
  oFFT: fsFFT, oTap: fsTAP, oI: fsINT, oR: fsRED,
  oG_quant: fsMASK, oG_target: fsGREP, oCol_sem: fsSEM, oCol_replay: fsREP, oCol_over: fsOVER,
};
function implFor(src) {
  if (src.includes('oFFT')) return IMPL.oFFT;
  if (src.includes('oTap')) return IMPL.oTap;
  if (src.includes('oI =')) return IMPL.oI;
  if (src.includes('oR =')) return IMPL.oR;
  if (src.includes('uQuant')) return IMPL.oG_quant;
  if (src.includes('uTarget')) return IMPL.oG_target;
  if (src.includes('uBarLenM')) return IMPL.oCol_sem;
  if (src.includes('uIR')) return IMPL.oCol_replay;
  if (src.includes('uSplit')) return IMPL.oCol_over;
  throw new Error('glmock: no CPU twin for this shader');
}

/* ------------------------------------------------------------------ */
export function createGL(canvas) {
  const gl = { canvas, _K: K };
  for (const k in K) gl[k] = K[k];
  gl._bound = {};          // active texture unit -> texture object
  gl._units = [null, null, null, null];
  gl._activeUnit = 0;
  gl._t = new Map();
  gl._p = new Map();
  gl._sh = new Map();
  gl._program = null;
  gl._fb = null;
  gl._texUnit = 0;
  gl._stat = () => stats;

  const ok = () => true;
  Object.assign(gl, {
    getExtension: (n) => ({ name: n }),
    getShaderParameter: ok, getShaderInfoLog: () => "", getProgramInfoLog: () => "",
    getError: () => 0,
    shaderSource: (s, src) => { s.src = src; },
    compileShader: ok, attachShader: ok,
    createShader: (t) => ({ t }),
    createProgram: () => ({ uniforms: new Map(), uval: {}, src: '' }),
    linkProgram: (p) => {
      // harvest uniforms + the fragment source so we can pick a CPU twin
      for (const sh of p.shaders || []) if (sh.src) p.src += sh.src;
      const set = new Set();
      for (const d of p.src.matchAll(/uniform\s+\w+\s+([^;]+);/g))
        for (const piece of d[1].split(',')) set.add(piece.trim().replace(/\[.*$/, '').trim());
      p.uniformNames = [...set];
      p.impl = implFor(p.src);
    },
    getProgramParameter: (p, what) => what === K.LINK_STATUS ? true
      : what === K.ACTIVE_UNIFORMS ? p.uniformNames.length : 0,
    getActiveUniform: (p, i) => ({ name: p.uniformNames[i] }),
    getUniformLocation: (p, n) => ({ p, n }),
    useProgram: (p) => { gl._program = p; },
    createTexture: () => ({ w: 0, h: 0, data: null }),
    deleteTexture: (t) => gl._t.delete(t),
    deleteFramebuffer: (f) => { },
    bindTexture: (target, t) => { gl._units[gl._activeUnit] = t || null; },
    activeTexture: (unit) => { gl._activeUnit = unit - K.TEXTURE0; },
    texParameteri: ok,
    texImage2D: (target, lvl, internal, w, h, border, fmt, type, data) => {
      const t = gl._units[gl._activeUnit];
      if (!t) throw new Error('glmock: texImage2D with no bound texture');
      t.w = w; t.h = h; t.internal = internal;
      t.data = new Float32Array(w * h * 4);
      if (data) {
        const stride = (fmt === K.RED) ? 1 : (fmt === K.RG ? 2 : 4);
        for (let i = 0; i < w * h; i++)
          for (let c = 0; c < stride && c < 4; c++) {
            const v = data[i * stride + c];
            t.data[i * 4 + c] = (type === K.UNSIGNED_BYTE) ? (v / 255) : v;
          }
        if (stride === 1) { t.data[0] = t.data[0]; }
        if (fmt === K.RED || fmt === K.RG) for (let i = 0; i < w * h; i++) t.data[i * 4 + 3] = 1;
      }
    },
    createFramebuffer: () => ({ }),
    bindFramebuffer: (target, f) => { gl._fb = f; },
    framebufferTexture2D: (target, att, tt, tex) => {
      if (!gl._fb) throw new Error('glmock: framebufferTexture2D with no framebuffer');
      gl._fb.tex = tex;
    },
    checkFramebufferStatus: () => K.FRAMEBUFFER_COMPLETE,
    viewport: (x, y, w, h) => { gl._vp = [x, y, w, h]; },
    clearColor: ok, clear: ok, enable: ok, disable: ok, blendFunc: ok, pixelStorei: ok,
    uniform1i: (l, v) => { l.p.uval[l.n] = v; },
    uniform1f: (l, v) => { l.p.uval[l.n] = v; },
    uniform2f: (l, a, b) => { l.p.uval[l.n] = [a, b]; },
    uniform3f: (l, a, b, c) => { l.p.uval[l.n] = [a, b, c]; },
    uniform4f: (l, a, b, c, d) => { l.p.uval[l.n] = [a, b, c, d]; },
    uniform1fv: (l, v) => { l.p.uval[l.n] = Array.from(v); },
    drawArrays: (mode, first, count) => {
      const p = gl._program;
      if (!p || !p.impl) throw new Error('glmock: drawArrays with no usable program');
      const [vx, vy, vw, vh] = gl._vp || [0, 0, 1, 1];
      const fb = gl._fb;
      const dst = fb ? fb.tex : gl._canvasTex;
      if (!dst) throw new Error('glmock: no render target (canvas textures are created by the harness)');
      if (fb && fb.tex === gl._units[0] && p.impl !== fsOVER) {
        // reading and writing the same texture: real GPUs give undefined results
        stats.racy++;
      }
      gl._bound = { 0: gl._units[0], 1: gl._units[1], 2: gl._units[2], 3: gl._units[3] };
      if (process.env.GLMOCK_DEBUG && p.impl === fsRED) {
        const s0 = texel(gl._units[0], 0, 0), s1 = texel(gl._units[0], 1, 0);
        console.log('   [reduce] src', gl._units[0] && gl._units[0].w + 'x' + gl._units[0].h,
          'texel(0,0)', s0.map((v) => v.toExponential(2)).join(','), 'texel(1,0)', s1.map((v) => v.toExponential(2)).join(','),
          'dst', fb && fb.tex && fb.tex.w + 'x' + fb.tex.h);
      }
      const out = new Float32Array(vw * vh * 4);
      for (let y = 0; y < vh; y++)
        for (let x = 0; x < vw; x++) {
          const c = p.impl(gl, p, vx + x, vy + y);
          const o = (y * vw + x) * 4;
          for (let k = 0; k < 4; k++) {
            const v = c[k];
            if (!isFinite(v)) stats.nan++;
            out[o + k] = v;
          }
        }
      dst.data = out; dst.w = vw; dst.h = vh;
      stats.writes += vw * vh;
    },
    readPixels: (x, y, w, h, fmt, type, out) => {
      const src = gl._fb ? gl._fb.tex : gl._canvasTex;
      if (!src || !src.data) throw new Error('glmock: readPixels with no source');
      const stride = (fmt === K.RED) ? 1 : (fmt === K.RG ? 2 : 4);
      for (let j = 0; j < h; j++)
        for (let i = 0; i < w; i++) {
          const sx = clamp(x + i, 0, src.w - 1), sy = clamp(y + j, 0, src.h - 1);
          const o = (sy * src.w + sx) * 4;
          for (let c = 0; c < stride; c++) out[(j * w + i) * stride + c] = src.data[o + c];
        }
    },
  });
  return gl;
}
export { stats };
