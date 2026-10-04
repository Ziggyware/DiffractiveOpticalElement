#!/usr/bin/env node
/**
 * server.mjs — dependency-free static server for the workbench.
 *
 *   node server.mjs [--port 5173] [--host 0.0.0.0]
 *
 * It serves three things out of the repository, with no build step:
 *   /            -> index.html (the workbench)
 *   /web/*       -> the workbench's own modules
 *   /src/*       -> the library itself, imported straight into the browser as
 *                   ES modules (the library has no Node-only imports on the
 *                   paths the workbench uses, so the browser runs the same code
 *                   as the tests and the CLI)
 *   /docs/img/*  -> the figures written by tools/render-demo.mjs
 *
 * Plus two tiny JSON endpoints so the page can show what the server thinks the
 * defaults are (and so a health check has something to hit).
 */

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_PROJECTOR } from './src/projector.js';

const args = process.argv.slice(2);
const argOf = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const PORT = Number(process.env.PORT ?? argOf('port', 5173));
const HOST = process.env.HOST ?? argOf('host', '0.0.0.0');
const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)));

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.md': 'text/markdown; charset=utf-8',
};

const json = (res, code, body) => {
  const data = Buffer.from(JSON.stringify(body));
  res.writeHead(code, { 'content-type': 'application/json; charset=utf-8', 'content-length': data.length });
  res.end(data);
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host ?? 'localhost'}`);
    const pathname = decodeURIComponent(url.pathname);

    if (pathname === '/api/health') return json(res, 200, { ok: true, uptime: process.uptime(), node: process.version });
    if (pathname === '/api/defaults') return json(res, 200, DEFAULT_PROJECTOR);
    if (pathname === '/api/geometry') {
      const { ProjectorSystem } = await import('./src/projector.js');
      const sys = new ProjectorSystem(DEFAULT_PROJECTOR);
      return json(res, 200, sys.geometry());
    }

    let rel = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
    const full = join(ROOT, normalize(rel));
    if (!full.startsWith(ROOT)) {
      res.writeHead(403, { 'content-type': 'text/plain' });
      res.end('forbidden');
      return;
    }
    let target = full;
    try {
      const st = await stat(target);
      if (st.isDirectory()) target = join(target, 'index.html');
    } catch {
      res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      res.end(`not found: ${pathname}\n`);
      return;
    }
    const body = await readFile(target);
    const type = MIME[extname(target).toLowerCase()] ?? 'application/octet-stream';
    res.writeHead(200, {
      'content-type': type,
      'content-length': body.length,
      'cache-control': 'no-cache',
      // the workbench is meant to be embedded in a preview iframe
      'access-control-allow-origin': '*',
    });
    res.end(body);
  } catch (err) {
    res.writeHead(500, { 'content-type': 'text/plain; charset=utf-8' });
    res.end(`error: ${err.message}\n`);
  }
});

server.listen(PORT, HOST, () => {
  console.log(`DiffractiveOpticalElement workbench`);
  console.log(`  http://localhost:${PORT}/`);
  console.log(`  serving ${ROOT}`);
  console.log(`  the library is served from /src, so the browser runs the same modules as the tests`);
});
