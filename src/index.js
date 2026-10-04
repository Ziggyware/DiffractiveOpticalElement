/**
 * index.js — the whole library behind one import.
 *
 *   import { ProjectorSystem, ComplexField, propagateAngularSpectrum } from 'diffractive-optical-element';
 *
 * Everything here is plain ESM with no runtime dependencies, so the same files
 * run in Node (tests, CLI tools, the dev server) and directly in a browser
 * module context (the workbench in web/ imports them over HTTP). The only
 * Node-specific helper is the zlib compressor used for compact PNG output,
 * which is imported lazily inside encodePNGNode.
 */

export * from './fft.js';
export * from './field.js';
export * from './propagate.js';
export * from './metrics.js';
export * from './doe.js';
export * from './interference.js';
export * from './projector.js';
export * from './color.js';
export * from './png.js';
