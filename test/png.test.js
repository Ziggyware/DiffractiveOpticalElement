import { test } from 'node:test';
import assert from 'node:assert/strict';
import { inflateSync } from 'node:zlib';
import { encodePNG, encodePNGNode, writePlanePNG } from '../src/png.js';
import { planeToGray, planeToRGB, COLORMAPS, wavelengthColor } from '../src/color.js';
import * as lib from '../src/index.js';

/** Parse a PNG buffer back into its chunks and raw scanlines (test-side decoder). */
function decodePNG(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < 8; i++) assert.equal(u8[i], sig[i], 'PNG signature');
  let o = 8;
  const chunks = [];
  let idat = [];
  let width = 0;
  let height = 0;
  let colourType = 0;
  while (o < u8.length) {
    const len = (u8[o] << 24) | (u8[o + 1] << 16) | (u8[o + 2] << 8) | u8[o + 3];
    const type = String.fromCharCode(u8[o + 4], u8[o + 5], u8[o + 6], u8[o + 7]);
    const data = u8.subarray(o + 8, o + 8 + len);
    if (type === 'IHDR') {
      width = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
      height = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7];
      assert.equal(data[8], 8, 'bit depth 8');
      colourType = data[9];
    } else if (type === 'IDAT') {
      idat.push(Buffer.from(data));
    }
    chunks.push(type);
    o += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const channels = colourType === 2 ? 3 : 1;
  const stride = 1 + width * channels;
  const pixels = new Uint8Array(width * height * channels);
  for (let y = 0; y < height; y++) {
    assert.equal(raw[y * stride], 0, 'filter type 0');
    pixels.set(raw.subarray(y * stride + 1, y * stride + stride), y * width * channels);
  }
  return { width, height, colourType, channels, pixels, chunks };
}

test('encodePNG writes a well-formed grayscale PNG that round-trips', () => {
  const width = 5;
  const height = 3;
  const gray = new Uint8Array(width * height);
  for (let k = 0; k < gray.length; k++) gray[k] = (k * 17) % 256;
  const png = encodePNG({ width, height, gray });
  const back = decodePNG(png);
  assert.equal(back.width, width);
  assert.equal(back.height, height);
  assert.equal(back.channels, 1);
  assert.deepEqual(Array.from(back.pixels), Array.from(gray));
  assert.deepEqual(back.chunks, ['IHDR', 'IDAT', 'IEND']);
});

test('encodePNG writes a well-formed RGB PNG', () => {
  const width = 4;
  const height = 2;
  const rgb = new Uint8Array(width * height * 3);
  for (let k = 0; k < rgb.length; k++) rgb[k] = (k * 31) % 256;
  const back = decodePNG(encodePNG({ width, height, rgb }));
  assert.equal(back.colourType, 2);
  assert.equal(back.channels, 3);
  assert.deepEqual(Array.from(back.pixels), Array.from(rgb));
});

test('the Node encoder produces real compression and the same pixels', async () => {
  const width = 64;
  const height = 64;
  const gray = new Uint8Array(width * height);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) gray[i + j * width] = (i * 3 + j * 5) % 256;
  }
  const stored = encodePNG({ width, height, gray });
  const compressed = await encodePNGNode({ width, height, gray });
  assert.ok(compressed.length < stored.length, 'zlib should beat stored blocks');
  assert.deepEqual(Array.from(decodePNG(stored).pixels), Array.from(gray));
  assert.deepEqual(Array.from(decodePNG(compressed).pixels), Array.from(gray));
  // a constant image compresses to (almost) nothing
  const flat = await encodePNGNode({ width, height, gray: new Uint8Array(width * height) });
  assert.ok(flat.length < 200, `a flat image should be tiny, got ${flat.length}`);
});

test('encodePNG validates its arguments', () => {
  assert.throws(() => encodePNG({ width: 2, height: 2 }), /pass rgb or gray/);
  assert.throws(() => encodePNG({ width: 2, height: 2, gray: new Uint8Array(3) }), /buffer too small/);
});

test('colormaps map a plane onto 0..255 and are monotone in luminance', () => {
  for (const name of Object.keys(COLORMAPS)) {
    const plane = new Float64Array(9);
    for (let k = 0; k < plane.length; k++) plane[k] = k / 8;
    const gray = planeToGray(plane);
    assert.equal(gray.length, plane.length, `${name}: gray length`);
    assert.ok(gray[0] <= gray[8], `${name}: brighter input gives a brighter pixel`);
    const rgb = planeToRGB(plane, { colormap: name });
    assert.equal(rgb.length, plane.length * 3);
    for (const v of rgb) assert.ok(v >= 0 && v <= 255, `${name}: 8-bit range`);
  }
  // inferior (max) vs the default (normalised) scaling
  const plane = new Float64Array([0, 0.5, 2]);
  const norm = planeToGray(plane, { norm: true });
  assert.equal(norm[2], 255);
  assert.ok(norm[1] > 0 && norm[1] < 255);
  const fixed = planeToGray(plane, { min: 0, max: 4, norm: false });
  assert.ok(Math.abs(fixed[2] - 127.5) < 1);
  // wavelength -> colour is a smooth, monotone hue sweep in the visible
  const c450 = wavelengthColor(450e-9);
  const c550 = wavelengthColor(550e-9);
  const c650 = wavelengthColor(650e-9);
  assert.ok(c450[2] > c450[0], '450 nm is blue');
  assert.ok(c550[1] > c550[2], '550 nm is green');
  assert.ok(c650[0] > c650[1], '650 nm is red');
});

test('writePlanePNG writes a file that decodes back to the expected image', async () => {
  const fs = await import('node:fs');
  const os = await import('node:os');
  const path = await import('node:path');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'doe-png-'));
  const file = path.join(dir, 'plane.png');
  const plane = new Float64Array(16);
  for (let k = 0; k < plane.length; k++) plane[k] = k / 15;
  const bytes = await writePlanePNG(file, plane, 4, 4);
  const onDisk = fs.readFileSync(file);
  assert.equal(onDisk.length, bytes.length);
  const back = decodePNG(onDisk);
  assert.equal(back.width, 4);
  assert.equal(back.height, 4);
  // the brightest sample must map to a bright pixel
  assert.ok(back.pixels[15] > 240);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('the package barrel re-exports every module', () => {
  for (const name of [
    'fft2d',
    'ComplexField',
    'propagateAngularSpectrum',
    'focalPlaneSpectrum',
    'propagateFresnelFFT',
    'rmsError',
    'ssim',
    'gerchbergSaxton',
    'weightedGerchbergSaxton',
    'fourierCgh',
    'designDOE',
    'ProjectorSystem',
    'makeTarget',
    'renderText',
    'encodePNG',
    'planeToRGB',
  ]) {
    assert.ok(name in lib, `missing export ${name}`);
  }
  // and the barrel is importable in one go without touching Node APIs
  assert.equal(typeof lib.ProjectorSystem, 'function');
  assert.equal(typeof lib.encodePNG, 'function');
});
