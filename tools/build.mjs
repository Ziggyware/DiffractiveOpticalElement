// Assembles index.html from the parts in /home/user/tmp plus the generated
// Open Graph image, and writes the 5-line controls README.
//   node tools/build.mjs
import fs from 'node:fs';
import path from 'node:path';

const ROOT = '/home/user/DiffractiveOpticalElement';
const TMP = '/home/user/tmp';

const head = fs.readFileSync(path.join(TMP, 'p1.html'), 'utf8');
const shaders = fs.readFileSync(path.join(TMP, 'shaders.html'), 'utf8');
const app1 = fs.readFileSync(path.join(TMP, 'app.js'), 'utf8');
const app2 = fs.readFileSync(path.join(TMP, 'app2.js'), 'utf8');

// head: keep everything up to the shaders marker, then insert parts
const cut = head.indexOf('<!-- ============================ shaders ============================ -->');
if (cut < 0) throw new Error('shader marker not found in p1.html');
let body = head.slice(0, cut);

const og = fs.readFileSync('/home/user/previews/og.b64', 'utf8').trim();
body = body.split('OG_IMAGE_DATA_URI').join('data:image/png;base64,' + og);

const out = body + shaders + '\n' + app1 + '\n' + app2;
fs.writeFileSync(path.join(ROOT, 'index.html'), out);

const bytes = Buffer.byteLength(out);
console.log('index.html written:', (bytes / 1024).toFixed(1), 'KB', bytes < 100 * 1024 ? '(under 100 KB)' : '(OVER 100 KB)');
console.log('  head   ', (Buffer.byteLength(body) / 1024).toFixed(1), 'KB');
console.log('  shaders', (Buffer.byteLength(shaders) / 1024).toFixed(1), 'KB');
console.log('  js     ', ((Buffer.byteLength(app1) + Buffer.byteLength(app2)) / 1024).toFixed(1), 'KB');
console.log('  og png ', (og.length / 1024).toFixed(1), 'KB base64');

const readme = `# DOE — the picture is not in here
1. \`1…6\` or the chips pick a scene; \`tour\` cycles one scene per 10 s (every animated quantity shares that 10 s clock, so the piece loops).
2. Scroll / pinch the **left** panel to zoom the mask 1× → 64× (past single pixels), drag to pan; scroll / pinch the **right** panel to zoom the replay.
3. Drag the divider to re-split the view; \`space\` = calm (freezes all motion, also honours \`prefers-reduced-motion\`), \`r\` = record 10 s, \`0\` = reset the view.
4. Every setting (scene, λ, pitch, levels, text, seed …) rides in the URL fragment, ≤200 chars - copy the link to reproduce the exact same frame.
5. \`?selftest\` in the URL (or the selftest button) runs the graded physics checks on screen and in the console; \`DOE.report()\` dumps the live state.
`;

fs.writeFileSync(path.join(ROOT, 'README.md'), readme);
console.log('README.md written');
