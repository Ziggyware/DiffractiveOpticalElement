/**
 * png.js — minimal PNG encoder (8-bit RGB / grayscale) with no dependencies.
 *
 * Only the two features needed to write simulation output are implemented. The
 * encoder itself is synchronous, dependency-free and browser-safe: it writes a
 * valid zlib stream made of *stored* (uncompressed) deflate blocks. Passing a
 * `deflate` hook lets a Node caller swap in node:zlib for real compression
 * (see encodePNGNode / writePlanePNG) without the module having to import
 * anything Node-specific.
 */

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = data.length;
  const out = new Uint8Array(12 + len);
  out[0] = (len >>> 24) & 0xff;
  out[1] = (len >>> 16) & 0xff;
  out[2] = (len >>> 8) & 0xff;
  out[3] = len & 0xff;
  const t = new TextEncoder().encode(type);
  out.set(t, 4);
  out.set(data, 8);
  const crcInput = new Uint8Array(4 + len);
  crcInput.set(t, 0);
  crcInput.set(data, 4);
  const c = crc32(crcInput);
  out[8 + len] = (c >>> 24) & 0xff;
  out[9 + len] = (c >>> 16) & 0xff;
  out[10 + len] = (c >>> 8) & 0xff;
  out[11 + len] = c & 0xff;
  return out;
}

/** Minimal stored-block deflate (fallback when zlib is unavailable). */
function storedDeflate(data) {
  const max = 65535;
  const blocks = Math.ceil(data.length / max) || 1;
  // 2 (zlib header) + 5 per stored block (BFINAL byte, LEN, NLEN) + the data
  // + 4 for the big-endian Adler-32 trailer.
  const out = new Uint8Array(2 + blocks * 5 + data.length + 4);
  let o = 0;
  out[o++] = 0x78;
  out[o++] = 0x01;
  for (let b = 0; b < blocks; b++) {
    const start = b * max;
    const len = Math.min(max, data.length - start);
    out[o++] = b === blocks - 1 ? 1 : 0;
    out[o++] = len & 0xff;
    out[o++] = (len >>> 8) & 0xff;
    out[o++] = ~len & 0xff;
    out[o++] = (~len >>> 8) & 0xff;
    out.set(data.subarray(start, start + len), o);
    o += len;
  }
  // Adler-32
  let a = 1;
  let bb = 0;
  for (let i = 0; i < data.length; i++) {
    a = (a + data[i]) % 65521;
    bb = (bb + a) % 65521;
  }
  // Adler-32 is a single 32-bit value, (b << 16) | a, written big-endian.
  const adler = ((bb << 16) | a) >>> 0;
  out[o++] = (adler >>> 24) & 0xff;
  out[o++] = (adler >>> 16) & 0xff;
  out[o++] = (adler >>> 8) & 0xff;
  out[o++] = adler & 0xff;
  return out.subarray(0, o);
}

/**
 * Encode an image to a PNG Buffer/Uint8Array.
 * @param {{width: number, height: number, rgb?: Uint8Array, gray?: Uint8Array}} image
 */
export function encodePNG({ width, height, rgb = null, gray = null, deflate = null }) {
  const channels = rgb ? 3 : 1;
  const src = rgb ?? gray;
  if (!src) throw new Error('encodePNG: pass rgb or gray');
  if (src.length < width * height * channels) {
    throw new Error(`encodePNG: buffer too small (${src.length} < ${width * height * channels})`);
  }
  const raw = new Uint8Array(height * (1 + width * channels));
  for (let y = 0; y < height; y++) {
    const rowStart = y * (1 + width * channels);
    raw[rowStart] = 0; // filter type 0 (None)
    raw.set(src.subarray(y * width * channels, (y + 1) * width * channels), rowStart + 1);
  }
  let compressed = null;
  if (deflate) {
    try {
      compressed = deflate(raw);
    } catch {
      compressed = null;
    }
  }
  if (!compressed) compressed = storedDeflate(raw);

  const ihdr = new Uint8Array(13);
  ihdr[0] = (width >>> 24) & 0xff;
  ihdr[1] = (width >>> 16) & 0xff;
  ihdr[2] = (width >>> 8) & 0xff;
  ihdr[3] = width & 0xff;
  ihdr[4] = (height >>> 24) & 0xff;
  ihdr[5] = (height >>> 16) & 0xff;
  ihdr[6] = (height >>> 8) & 0xff;
  ihdr[7] = height & 0xff;
  ihdr[8] = 8; // bit depth
  ihdr[9] = channels === 3 ? 2 : 0; // colour type
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const parts = [
    new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', compressed),
    chunk('IEND', new Uint8Array(0)),
  ];
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/**
 * Like encodePNG, but uses node:zlib for real compression when the module is
 * available (Node) and falls back to the built-in stored-deflate stream
 * everywhere else (browser, worker, any runtime without node:zlib).
 */
export async function encodePNGNode(image, { level = 6 } = {}) {
  let deflate = null;
  try {
    const zlib = await import('node:zlib');
    deflate = (raw) => zlib.deflateSync(raw, { level });
  } catch {
    deflate = null;
  }
  return encodePNG({ ...image, deflate });
}

/**
 * Convenience: render a scalar plane straight to a PNG file (Node only).
 * Returns the encoded bytes so the caller can also stream them (e.g. the dev
 * server's /api/...png routes).
 */
export async function writePlanePNG(path, plane, width, height, opts = {}) {
  const { planeToGray } = await import('./color.js');
  const gray = opts.gray ?? planeToGray(plane, opts);
  const png = await encodePNGNode({ width, height, gray });
  if (path) {
    const fs = await import('node:fs');
    fs.writeFileSync(path, png);
  }
  return png;
}
