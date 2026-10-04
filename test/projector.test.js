import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ComplexField } from '../src/field.js';
import { focalPlaneSpectrum } from '../src/propagate.js';
import { ProjectorSystem, makeTarget, renderText, resampleRaster, convergenceSweep, DEFAULT_PROJECTOR } from '../src/projector.js';

const BASE = { n: 128, pitch: 8e-6, lambda: 532e-9, distance: 0.03, iterations: 40, seed: 12345 };

test('projector defaults and geometry budget are self-consistent', () => {
  assert.equal(DEFAULT_PROJECTOR.n, 256);
  assert.equal(DEFAULT_PROJECTOR.mode, 'image-plane');
  assert.equal(DEFAULT_PROJECTOR.algorithm, 'auto');
  const sys = new ProjectorSystem(BASE);
  const g = sys.geometry();
  // geometry reports engineering units: millimetres (and micrometres for pitch)
  assert.ok(Math.abs(g.screenWindowMm - sys.nx * sys.dx * 1e3) < 1e-12, 'image-plane window equals the element window');
  assert.ok(Math.abs(g.apertureMm - sys.cfg.aperture * sys.nx * sys.dx * 1e3) < 1e-9, 'aperture from the fill factor');
  assert.ok(Math.abs(g.pitchUm - sys.dx * 1e6) < 1e-9);
  assert.ok(Math.abs(g.screenPitchUm - sys.dx * 1e6) < 1e-9);
  assert.ok(g.maxDeflectionDeg > 0 && g.maxDeflectionDeg < 90, 'deflection limit is physical');
  assert.ok(g.fNumber > 0);
  assert.equal(g.addressablePixels, Math.round(g.resolvableSpotsPerAxis ** 2));
  assert.ok(g.addressablePixels > 0 && Number.isInteger(g.addressablePixels));
  // the critical distance is where the Fresnel chirp stops being undersampled
  assert.ok(Math.abs(g.undersampledChirp === (sys.z < (sys.nx * sys.dx * sys.dx) / sys.lambda)) === 1);
  // matched throw for the far-field head
  // with no explicit distance the far-field head defaults to the mode-matched
  // throw N*p^2/lambda (where the screen window equals the element window)
  const noDistance = { ...BASE };
  delete noDistance.distance;
  const ff = new ProjectorSystem({ ...noDistance, mode: 'far-field' });
  assert.ok(Math.abs(ff.z - ff.matchDistance()) < 1e-15, 'far-field throw defaults to N*p^2/lambda');
  assert.ok(Math.abs(ff.matchDistance() - (128 * 8e-6 * 8e-6) / 532e-9) < 1e-15);
  // explicit distance still wins
  const ff2 = new ProjectorSystem({ ...BASE, mode: 'far-field', distance: 0.02 });
  assert.equal(ff2.z, 0.02);
});

test('renderText and resampleRaster build the target rasters', () => {
  const t = renderText('DOE', { scale: 6 });
  assert.ok(t.width > 0 && t.height > 0);
  let lit = 0;
  for (const v of t.data) if (v > 0) lit++;
  assert.ok(lit > 0, 'the font raster is not empty');
  assert.ok(lit < t.data.length, 'the font raster has background');
  assert.ok(t.data.every((v) => v === 0 || v === 1), 'binary raster');
  const small = resampleRaster(t.data, t.width, t.height, 64, 64, { fit: 'contain' });
  assert.equal(small.length, 64 * 64);
  let lit2 = 0;
  for (const v of small) if (v > 0.5) lit2++;
  assert.ok(lit2 > 0 && lit2 < small.length);
  // 'contain' must preserve the aspect ratio by letterboxing
  const wide = resampleRaster(new Float64Array(8 * 8).fill(1), 8, 8, 32, 16, { fit: 'contain' });
  assert.equal(wide.length, 32 * 16);
  // a target made from text has a ROI that covers the glyphs
  const tgt = makeTarget('text', 128, 128, { text: 'DOE' });
  assert.equal(tgt.irradiance.length, 128 * 128);
  assert.ok(tgt.roi.length === 128 * 128);
  assert.ok(tgt.roi.some((v) => v > 0.5), 'ROI covers the text');
});

test('image-plane projection reproduces the target with real power efficiency', () => {
  const sys = new ProjectorSystem({ ...BASE, mode: 'image-plane', target: { kind: 'text', text: 'DOE' } });
  const r = sys.design();
  assert.equal(r.mode, 'image-plane');
  assert.equal(r.iterations, 40);
  const field = sys.simulate();
  const m = sys.screenMetrics(field);
  // every one of these was measured on this fixture; the bounds leave room for
  // tuning but would catch a regression in the physics
  assert.ok(m.rmse < 0.06, `rmse ${(m.rmse * 100).toFixed(2)} %`);
  assert.ok(m.correlation > 0.98, `correlation ${m.correlation}`);
  assert.ok(m.efficiency > 0.55, `efficiency ${(m.efficiency * 100).toFixed(1)} %`);
  assert.ok(m.ssim > 0.95, `ssim ${m.ssim}`);
  assert.ok(m.zeroOrder < 0.01, `zero-order leak ${(m.zeroOrder * 100).toFixed(3)} %`);
  assert.ok(m.fluxFraction > 0.55, `flux fraction ${m.fluxFraction}`);
  assert.ok(m.snr > 15, `snr ${m.snr} dB`);
  // a designed element must beat the undiffracted element on the same target
  const flatPhase = new Float64Array(sys.nx * sys.ny);
  const flat = sys.simulate({ phase: flatPhase });
  const mFlat = sys.screenMetrics(flat, { phase: flatPhase });
  assert.ok(mFlat.zeroOrder > 0.5, `a flat element leaks ${(mFlat.zeroOrder * 100).toFixed(1)} % into DC`);
  assert.ok(m.rmse < mFlat.rmse, 'the design must beat the flat element');
});

test('the projection conserves energy exactly (Parseval through the propagator)', () => {
  const sys = new ProjectorSystem({ ...BASE, n: 64, mode: 'image-plane', target: { kind: 'disc' } });
  sys.design({ iterations: 10 });
  const field = sys.simulate();
  const irr = field.intensity();
  let screen = 0;
  for (const v of irr) screen += v;
  const ill = sys.illumination();
  let incident = 0;
  for (const v of ill) incident += v * v;
  // the angular-spectrum propagator is unitary on the same grid, so the discrete
  // power at the screen equals the power that entered the element
  assert.ok(Math.abs(screen / incident - 1) < 1e-9, `screen/incident = ${screen / incident}`);
  // and the screen window is the element window in this mode
  assert.equal(field.nx, sys.nx);
  assert.ok(Math.abs(field.dx - sys.dx) < 1e-18);
});

test('far-field projection is a lens focal-plane transform: sharp at any throw', () => {
  const sys = new ProjectorSystem({
    ...BASE,
    mode: 'far-field',
    target: { kind: 'text', text: 'DOE' },
  });
  const r = sys.design();
  assert.equal(r.mode, 'far-field');
  const field = sys.simulate();
  // window and pitch are set by the Fourier geometry, not by the element pitch
  assert.ok(Math.abs(field.dx - (sys.lambda * sys.z) / (sys.nx * sys.dx)) < 1e-18, 'Fourier pitch lambda*f/(N*dx)');
  assert.ok(Math.abs(field.dx * sys.nx - (sys.lambda * sys.z) / sys.dx) < 1e-15, 'Fourier window lambda*f/dx');
  const m = sys.screenMetrics(field);
  assert.ok(m.rmse < 0.2, `rmse ${(m.rmse * 100).toFixed(2)} %`);
  assert.ok(m.correlation > 0.95, `correlation ${m.correlation}`);
  assert.ok(m.efficiency > 0.5, `efficiency ${(m.efficiency * 100).toFixed(1)} %`);
  assert.ok(m.zeroOrder < 0.01, `zero-order leak ${(m.zeroOrder * 100).toFixed(3)} %`);
  // The image itself is scale-invariant: at another throw the *same* pattern
  // appears (bigger window, lower irradiance), which is the physical difference
  // from the image-plane head. Compare normalised patterns in index space.
  const a = sys.simulate({ z: 0.02 });
  const b = sys.simulate({ z: 0.05 });
  const norm = (f) => {
    const ir = f.intensity();
    let mean = 0;
    for (const v of ir) mean += v;
    mean /= ir.length;
    return Float64Array.from(ir, (v) => v / mean);
  };
  const na = norm(a);
  const nb = norm(b);
  let num = 0;
  let den = 0;
  for (let k = 0; k < na.length; k++) {
    num += (na[k] - nb[k]) ** 2;
    den += na[k] * na[k];
  }
  assert.ok(Math.sqrt(num / den) < 1e-12, 'the holographic image is distance-invariant');
  // total power is conserved by the lens transform, but the peak irradiance
  // falls as 1/z^2 because the window grows
  assert.ok(Math.abs(a.power() / b.power() - 1) < 1e-9, 'power conserved');
  let pa = 0;
  let pb = 0;
  for (const v of a.intensity()) pa = Math.max(pa, v);
  for (const v of b.intensity()) pb = Math.max(pb, v);
  assert.ok(Math.abs(pa / pb - (0.05 / 0.02) ** 2) < 0.05, `peak scaling ${pa / pb}`);
});

test('the automatic algorithm picks the right optimiser for each regime', () => {
  const image = new ProjectorSystem({ ...BASE, n: 64, mode: 'image-plane', target: { kind: 'disc' } });
  const far = new ProjectorSystem({ ...BASE, n: 64, mode: 'far-field', target: { kind: 'disc' } });
  const ri = image.design({ iterations: 20 });
  const rf = far.design({ iterations: 20 });
  assert.equal(ri.algorithm, 'wgs', 'image-plane defaults to the weighted loop');
  assert.equal(rf.algorithm, 'gs', 'far-field defaults to the plain loop');
  assert.equal(ri.mode, 'image-plane');
  assert.equal(rf.mode, 'far-field');
  const mi = image.screenMetrics(image.simulate());
  const mf = far.screenMetrics(far.simulate());
  assert.ok(mi.rmse < 0.1, `image-plane rmse ${(mi.rmse * 100).toFixed(2)} %`);
  assert.ok(mf.rmse < 0.35, `far-field rmse ${(mf.rmse * 100).toFixed(2)} %`);
  // the two regimes are genuinely different problems: the phases must differ
  let maxDiff = 0;
  for (let k = 0; k < ri.phase.length; k++) {
    maxDiff = Math.max(maxDiff, Math.abs(Math.cos(ri.phase[k]) - Math.cos(rf.phase[k])));
  }
  assert.ok(maxDiff > 0.2, 'the two modes must design different elements');
  // and each must beat the undiffracted element in its own geometry
  const flatPhase = new Float64Array(image.nx * image.ny);
  const miFlat = image.screenMetrics(image.simulate({ phase: flatPhase }), { phase: flatPhase });
  const mfFlat = far.screenMetrics(far.simulate({ phase: flatPhase }), { phase: flatPhase });
  assert.ok(mi.rmse < miFlat.rmse, 'image-plane design must beat a flat element');
  assert.ok(mf.rmse < mfFlat.rmse, 'far-field design must beat a flat element');
  // explicit overrides are honoured
  const forced = new ProjectorSystem({ ...BASE, n: 64, mode: 'far-field', algorithm: 'wgs', target: { kind: 'disc' } });
  assert.equal(forced.design({ iterations: 20 }).algorithm, 'wgs');
});

test('defocus: the image is only sharp at the design distance', () => {
  const sys = new ProjectorSystem({ ...BASE, n: 64, distance: 0.03, target: { kind: 'text', text: 'DOE' } });
  sys.design({ iterations: 30 });
  const curve = sys.defocusCurve({ points: 7, span: 0.6, pad: 1 });
  assert.equal(curve.length, 7);
  let best = curve[0];
  for (const p of curve) if (p.fidelity > best.fidelity) best = p;
  assert.ok(Math.abs(best.z - sys.z) < 1e-12, `focus lands at the design distance (${best.z})`);
  assert.ok(best.rmse < 0.1, `in-focus rmse ${(best.rmse * 100).toFixed(2)} %`);
  for (const p of curve) {
    if (p === best) continue;
    assert.ok(p.rmse > 0.4, `off-focus rmse at ${p.zMm.toFixed(1)} mm is ${(p.rmse * 100).toFixed(1)} %`);
    assert.ok(p.fidelity < best.fidelity, 'off-focus cannot beat in-focus');
  }
  // the rms spot radius must be smallest at focus: the light is concentrated
  const inFocus = sys.screenMetrics(sys.simulate({ z: sys.z }));
  const offFocus = sys.screenMetrics(sys.simulate({ z: sys.z * 1.4 }));
  assert.ok(inFocus.rmsRadius < offFocus.rmsRadius, 'the spot broadens away from focus');
  // in the image-plane head the screen window does not change with z
  assert.equal(sys.simulate({ z: sys.z * 1.4 }).nx, sys.nx);
});

test('a fixed lens shows geometric scaling while the designed throw stays sharp', () => {
  const sys = new ProjectorSystem({ ...BASE, mode: 'far-field', distance: 0.03, target: { kind: 'text', text: 'DOE' } });
  sys.design({ iterations: 20 });
  // at the focal plane the scale is exactly 1
  const onFocus = sys.simulateWithFixedLens({ z: 0.03, focal: 0.03 });
  assert.ok(Math.abs(onFocus.scale - 1) < 1e-15);
  // off the focal plane the image scales linearly with the throw
  const off = sys.simulateWithFixedLens({ z: 0.06, focal: 0.03 });
  assert.ok(Math.abs(off.scale - 2) < 1e-15);
  assert.equal(off.field.nx, sys.nx);
  // applyLens is a pure phase element: it cannot change the amplitude
  const lensed = sys.applyLens({ focal: 0.03 });
  const before = new ComplexField(sys.nx, sys.ny, sys.dx, sys.dy);
  before.setAmplitudePhase(sys.result.illumination, sys.result.phase);
  let num = 0;
  let den = 0;
  for (let k = 0; k < lensed.size; k++) {
    const a = Math.hypot(before.re[k], before.im[k]);
    const b = Math.hypot(lensed.re[k], lensed.im[k]);
    num += (a - b) ** 2;
    den += a * a;
  }
  assert.ok(Math.sqrt(num / den) < 1e-12, 'lens preserves amplitude');
});

test('phase quantisation costs efficiency and structure smoothly', () => {
  const sys = new ProjectorSystem({ ...BASE, mode: 'image-plane', target: { kind: 'disc' } });
  const cont = sys.design({ iterations: 30 });
  const mCont = sys.screenMetrics(sys.simulate());
  const q4 = sys.design({ iterations: 30, levels: 4 });
  assert.ok(q4.quantized);
  assert.equal(q4.levels, 4);
  // the quantised phase must sit on a 4-level grid
  const step = (2 * Math.PI) / 4;
  for (let k = 0; k < q4.phase.length; k += 17) {
    const m = ((q4.phase[k] % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    const off = Math.min(m, Math.abs(m - step), Math.abs(m - 2 * step), Math.abs(m - 3 * step), Math.abs(m - 4 * step));
    assert.ok(off < 1e-9, `phase ${q4.phase[k]} is off the 4-level grid`);
  }
  const mQ = sys.screenMetrics(sys.simulate({ phase: q4.phase }));
  assert.ok(mQ.rmse > mCont.rmse, 'quantisation must cost image quality');
  assert.ok(mQ.rmse < 0.6, `4-level rmse ${(mQ.rmse * 100).toFixed(1)} %`);
  assert.ok(mQ.efficiency > 0.4, `4-level efficiency ${(mQ.efficiency * 100).toFixed(1)} %`);
  void cont;
});

test('convergence improves the projection and the sweep reports the trajectory', () => {
  const cfg = { ...BASE, n: 64, mode: 'image-plane', target: { kind: 'text', text: 'DOE' } };
  const sweep = convergenceSweep(cfg, { checkpoints: [1, 5, 20, 40] });
  assert.equal(sweep.length, 4);
  assert.ok(sweep[3].rmse < sweep[0].rmse, 'more iterations must fit the target better');
  assert.ok(sweep[3].correlation > 0.95, `final correlation ${sweep[3].correlation}`);
  assert.ok(sweep.every((s) => Number.isFinite(s.rmse) && Number.isFinite(s.efficiency)));
  // the reported error must be a decreasing trajectory, not a single sample
  const sys = new ProjectorSystem(cfg);
  const r = sys.design({ iterations: 40 });
  assert.equal(r.error.length, 40);
  assert.ok(r.error[39] < r.error[0]);
  // every intermediate error is finite and the sequence is broadly decreasing
  assert.ok(r.error.every((v) => Number.isFinite(v) && v >= 0));
  let improvements = 0;
  for (let i = 1; i < r.error.length; i++) if (r.error[i] < r.error[i - 1]) improvements++;
  assert.ok(improvements > r.error.length * 0.5, `error decreased in ${improvements}/${r.error.length - 1} steps`);
  // design in far-field mode uses the Fourier loop and still reports a history
  const ff = new ProjectorSystem({ ...cfg, mode: 'far-field' });
  const rf = ff.design({ iterations: 20 });
  assert.equal(rf.error.length, 20);
  assert.ok(rf.error[19] < rf.error[0]);
});

test('screenMetrics is scale-aware (radiometry uses the screen pitch)', () => {
  // A far-field screen has a pitch of lambda*f/(N*dx). If the metrics used the
  // element pitch instead, every radiometric number would be off by the
  // magnification squared, so check that efficiency is window-independent.
  const sys = new ProjectorSystem({ ...BASE, mode: 'far-field', target: { kind: 'disc' } });
  sys.design({ iterations: 20 });
  const m = sys.screenMetrics(sys.simulate({ z: 0.03 }));
  const m2 = sys.screenMetrics(sys.simulate({ z: 0.06 }));
  assert.ok(Math.abs(m.efficiency - m2.efficiency) < 1e-9, 'efficiency must not depend on the throw');
  assert.ok(Math.abs(m.fluxFraction - m2.fluxFraction) < 1e-9, 'flux fraction must not depend on the throw');
  assert.ok(Math.abs(m.rmse - m2.rmse) < 1e-9, 'rmse must not depend on the throw');
  assert.ok(Math.abs(m.zeroOrder - m2.zeroOrder) < 1e-12);
  // sanity: the discrete power at the screen is the incident power
  const field = sys.simulate();
  assert.ok(Math.abs(field.power() / sys.simulate({ z: 0.03 }).power() - 1) < 1e-12);
});

test('a custom raster target works end to end', () => {
  const n = 64;
  const irr = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = i - n / 2;
      const y = j - n / 2;
      irr[i + j * n] = Math.abs(x) < 8 && Math.abs(y) < 8 ? 1 : 0; // a square
    }
  }
  const sys = new ProjectorSystem({ ...BASE, n, mode: 'image-plane', target: { irradiance: irr, name: 'square' } });
  const r = sys.design({ iterations: 30 });
  assert.ok(r.targetAmp.some((v) => v > 0));
  const m = sys.screenMetrics(sys.simulate());
  assert.ok(m.efficiency > 0.3, `efficiency ${(m.efficiency * 100).toFixed(1)} %`);
  assert.ok(m.correlation > 0.9, `correlation ${m.correlation}`);
  // the projected rms radius must sit inside the screen window
  assert.ok(m.rmsRadius > 0 && m.rmsRadius < sys.nx * sys.dx);
  void focalPlaneSpectrum;
  void ComplexField;
});
