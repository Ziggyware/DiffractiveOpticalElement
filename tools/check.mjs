// Static consistency checks for index.html: JS syntax, element ids, shader ids,
// uniform wiring between GLSL and the JS that sets them, and size budget.
//   node tools/check.mjs
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const ROOT = '/home/user/DiffractiveOpticalElement';
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
let fail = 0;
const bad = (m) => { console.log('  FAIL ' + m); fail++; };
const good = (m) => console.log('  ok   ' + m);

/* ---- 1. extract parts ---- */
const scripts = [];
const re = /<script([^>]*)>([\s\S]*?)<\/script>/g;
let m;
while ((m = re.exec(html))) {
  const attrs = m[1];
  const idm = /id="([^"]+)"/.exec(attrs);
  scripts.push({
    id: idm ? idm[1] : null,
    type: (/type="([^"]+)"/.exec(attrs) || [])[1] || null,
    src: m[2],
  });
}
const js = scripts.filter((s) => !s.type);
const shaders = scripts.filter((s) => s.type && s.type.startsWith('x-'));

/* ---- 2. JS syntax ---- */
fs.mkdirSync('/tmp/checkjs', { recursive: true });
js.forEach((s, i) => {
  const f = `/tmp/checkjs/part${i}.js`;
  fs.writeFileSync(f, s.src);
  try {
    execFileSync(process.execPath, ['--check', f], { stdio: 'pipe' });
    good(`JS block ${i} parses (${(s.src.length / 1024).toFixed(1)} KB)`);
  } catch (e) {
    bad(`JS block ${i} syntax error:\n${e.stderr ? e.stderr.toString() : e}`);
  }
});

/* ---- 3. shader ids used by makeProg exist ---- */
const used = new Set();
for (const mm of html.matchAll(/makeProg\(\s*vs\s*,\s*"([^"]+)"\s*\)/g)) used.add(mm[1]);
for (const mm of html.matchAll(/\$\("([a-z]+\.[a-z]+)"\)\.textContent/g)) used.add(mm[1]);
const haveIds = new Set(shaders.map((s) => s.id));
for (const u of used) {
  if (!haveIds.has(u)) bad(`makeProg references missing shader id "${u}"`);
}
for (const s of shaders) if (!used.has(s.id)) bad(`shader "${s.id}" is never compiled`);
if (haveIds.size && !fail) good(`${shaders.length} shader blocks present, all referenced`);

/* ---- 4. element ids referenced from JS exist in the HTML ---- */
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((x) => x[1]));
const jstext = js.map((s) => s.src).join('\n');
const refs = new Set([...jstext.matchAll(/\$\("([^"]+)"\)/g)].map((x) => x[1]));
for (const r of refs) if (!ids.has(r)) bad(`JS asks for #${r} which is not in the document`);
good(`${refs.size} element ids resolved (${ids.size} ids in document)`);

/* ---- 5. uniform wiring: every GLSL uniform the JS sets must be declared ---- */
const PROGRAM_OF = {
  tap: 'fs.tap', fft: 'fs.fft', intensity: 'fs.intensity', reduce: 'fs.reduce',
  gsmask: 'fs.gsmask', gsreplay: 'fs.gsreplay', sem: 'fs.sem', replay: 'fs.replay', over: 'fs.over',
};
const declared = {};
for (const s of shaders) {
  if (!s.id.startsWith('fs.')) continue;
  const set = new Set();
  for (const d of s.src.matchAll(/uniform\s+\w+\s+([^;]+);/g)) {
    for (const piece of d[1].split(',')) {
      const nm = piece.trim().replace(/\[\s*\d+\s*\]$/, '').trim();
      if (nm) set.add(nm);
    }
  }
  declared[s.id] = set;
}
/* which program does each block of JS talk to?  Track `pr = PR.x` / `pr = PR[...]`
   style assignments and the argument order of the helpers. */
const jsLines = jstext.split('\n');
let cur = null;
const problems = [];
for (const line of jsLines) {
  const asg = /pr\s*=\s*PR\.(\w+)/.exec(line) || /pr\s*=\s*PR\[/.exec(line);
  if (asg) cur = asg[1] || null;
  if (/^\s*const pr = PR\.(\w+)/.test(line)) cur = /const pr = PR\.(\w+)/.exec(line)[1];
  for (const mm of line.matchAll(/\b(u1f|u1i|u2f|u3f|u4f|setTex)\(\s*pr\s*,\s*"([^"]+)"/g)) {
    const name = mm[2];
    if (!cur) continue;                       // helper used outside a pr scope: skip
    const prog = PROGRAM_OF[cur];
    if (!prog || !declared[prog]) { problems.push(`${cur}.${name}: unknown program`); continue; }
    if (!declared[prog].has(name)) problems.push(`${mm[1]}(${cur}, "${name}") but ${prog} declares no ${name}`);
  }
}
if (problems.length) problems.forEach(bad);
else good('every uniform the JS sets is declared in the matching shader');

/* ---- 6. uniform arrays set via gl.uniformNfv ---- */
for (const mm of jstext.matchAll(/pr\.u\["([^"]+)"\]/g)) {
  const name = mm[1];
  if (![...Object.values(declared)].some((s) => s.has(name))) bad(`JS reads uniform ${name} which no shader declares`);
}
good('uniform array lookups all resolve');

/* ---- 7. no network references (self-contained) ---- */
const netRe = /(https?:)?\/\/[a-z0-9.-]+\.[a-z]{2,}/gi;
const netHits = [...html.matchAll(netRe)].map((x) => x[0]).filter((u) => !u.includes('www.w3.org'));
if (netHits.length) bad('external URL(s): ' + [...new Set(netHits)].join(', '));
else good('no external URLs (fully self-contained)');
if (/\b(fetch|XMLHttpRequest|importScripts|WebSocket|EventSource)\s*\(/.test(html)) bad('network API present');
else good('no network APIs used');
if (/src="(?!data:)/.test(html.replace(/<script/g, '<script'))) {
  const ext = [...html.matchAll(/\b(?:src|href)="([^"]+)"/g)].map((x) => x[1]).filter((u) => !u.startsWith('data:') && !u.startsWith('#'));
  if (ext.length) bad('external src/href: ' + ext.join(', '));
  else good('no external src/href');
}

/* ---- 8. size ---- */
const bytes = Buffer.byteLength(html);
if (bytes > 100 * 1024) bad(`index.html is ${(bytes / 1024).toFixed(1)} KB > 100 KB`);
else good(`size ${(bytes / 1024).toFixed(1)} KB (target <= 100 KB)`);

/* ---- 9. meta tags ---- */
for (const t of ['og:title', 'og:description', 'og:image', 'twitter:card', 'twitter:title', 'description', 'viewport'])
  if (!html.includes(t)) bad(`missing meta tag ${t}`);
good('head meta tags present');

/* ---- 10. GLSL sanity: balanced braces, no obvious illegal names ---- */
for (const s of shaders) {
  const open = (s.src.match(/{/g) || []).length, close = (s.src.match(/}/g) || []).length;
  if (open !== close) bad(`${s.id}: unbalanced braces ${open}/${close}`);
  if (!s.src.includes('#version 300 es')) bad(`${s.id}: missing #version 300 es (must be the first line)`);
  if (s.id.startsWith('fs.') && !/precision\s+highp\s+float/.test(s.src)) bad(`${s.id}: missing precision qualifier`);
}
good('shader blocks: version + precision + braces');

console.log(fail ? `\n${fail} problem(s)` : '\nall checks passed');
process.exit(fail ? 1 : 0);

/* ---- README: one title line + exactly five control lines ---- */
{
  const rd = fs.readFileSync(path.join(ROOT, 'README.md'), 'utf8').split('\n').filter((l) => l.length);
  const ctrl = rd.filter((l) => /^[1-5]\. /.test(l));
  if (rd.length === 6 && ctrl.length === 5 && rd[1] === ctrl[0])
    good(`README.md: title + 5 control lines (${rd.length} lines)`);
  else
    bad(`README.md should be 6 non-empty lines (title + 5 controls), found ${rd.length} with ${ctrl.length} control lines`);
}
