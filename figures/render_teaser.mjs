// Render teaser.jsx to teaser.svg (Figure 1). Layout by Satori; arrows are added below.
// Arrow endpoints follow named Satori layout nodes, not hard-coded coordinates.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import satori from 'satori';

const HERE    = path.dirname(fileURLToPath(import.meta.url));
const SRC     = path.join(HERE, 'teaser.jsx');
const OUT_JS  = path.join(HERE, '.teaser.compiled.mjs');
const OUT_SVG = path.join(HERE, 'teaser.svg');

await build({
  entryPoints: [SRC],
  bundle: false, format: 'esm', platform: 'node',
  jsx: 'transform', jsxFactory: 'h', jsxFragment: 'Fragment',
  outfile: OUT_JS, logLevel: 'warning',
});

const mod = await import(OUT_JS + `?t=${Date.now()}`);
const Fig = mod.default;

function loadFontDir(dir, prefix) {
  const want = [
    [`${prefix}-Light.ttf`, 300, 'normal'],
    [`${prefix}-Regular.ttf`, 400, 'normal'],
    [`${prefix}-Medium.ttf`, 500, 'normal'],
    [`${prefix}-Semibold.ttf`, 600, 'normal'],
    [`${prefix}-Bold.ttf`, 700, 'normal'],
    [`${prefix}-Black.ttf`, 900, 'normal'],
    [`${prefix}-Italic.ttf`, 400, 'italic'],
  ];
  const out = [];
  for (const [file, weight, style] of want) {
    const p = path.join(dir, file);
    if (fs.existsSync(p)) out.push({ name: prefix, weight, style, data: fs.readFileSync(p) });
  }
  return out;
}

let fonts = loadFontDir('/usr/share/fonts/truetype/lato', 'Lato');
if (fonts.length === 0) throw new Error('No Lato fonts found.');

const nodes = new Map();
const svg = await satori(Fig({}), {
  width: 1664, fonts, embedFont: true,
  onNodeDetected(node) {
    if (node.props['data-node']) nodes.set(node.props['data-node'], node);
    if (node.left < -0.5 || node.left + node.width > 1664.5) {
      throw new Error('Figure content extends beyond canvas: ' + node.textContent);
    }
  },
});
const get = id => {
  if (!nodes.has(id)) throw new Error('Missing arrow anchor: ' + id);
  return nodes.get(id);
};
const pt = (id, side) => {
  const n = get(id);
  if (side === 'top') return [n.left + n.width / 2, n.top];
  if (side === 'bottom') return [n.left + n.width / 2, n.top + n.height];
  if (side === 'left') return [n.left, n.top + n.height / 2];
  return [n.left + n.width, n.top + n.height / 2];
};
const paths = [];
function arrow(points, feedback = false) {
  const d = points.map((p, i) => (i ? 'L' : 'M') + p.join(' ')).join(' ');
  paths.push('<path d="' + d + '" fill="none" stroke="' +
    (feedback ? '#008b88' : '#297b88') + '" stroke-width="3.2" stroke-linejoin="round"' +
    (feedback ? ' stroke-dasharray="7 5"' : '') +
    ' marker-end="url(#' + (feedback ? 'feedback' : 'flow') + ')"/>');
}
function curve(start, end, horizontal = false) {
  const mid = horizontal ? (start[0] + end[0]) / 2 : (start[1] + end[1]) / 2;
  const controls = horizontal ? [[mid, start[1]], [mid, end[1]]] : [[start[0], mid], [end[0], mid]];
  paths.push('<path d="M' + start.join(' ') + ' C' + controls[0].join(' ') + ' ' +
    controls[1].join(' ') + ' ' + end.join(' ') +
    '" fill="none" stroke="#008b88" stroke-width="4" marker-end="url(#flow)"/>');
}
// condition -> response
for (const k of ['1', '2', '3']) {
  const a = pt('p' + k, 'right'), b = pt('a' + k, 'left');
  arrow([[a[0] + 6, a[1]], [b[0] - 6, b[1]]]);
}
// feedback loop: select -> execute -> observe, then back through the update bar
for (const [a, b] of [['s-select', 's-exec'], ['s-exec', 's-obs']]) {
  const from = pt(a, 'right'), to = pt(b, 'left');
  arrow([[from[0] + 4, from[1]], [to[0] - 5, to[1]]]);
}
const obs = pt('s-obs', 'bottom'), sel = pt('s-select', 'bottom'), up = get('s-update');
arrow([[obs[0], obs[1] + 3], [obs[0], up.top - 5]], true);
arrow([[sel[0], up.top - 3], [sel[0], sel[1] + 6]], true);
const icons = [];
const wrench = '<path d="M21 3a6 6 0 0 1-7.6 7.6L6 18a2.8 2.8 0 1 1-4-4l7.4-7.4A6 6 0 0 1 17 0l-4 4 3 3Z" fill="currentColor"/><circle cx="4" cy="16" r="1" fill="white"/>';
for (const [id, n] of nodes) {
  if (!id.startsWith('icon-')) continue;
  const kind = id.split('-')[1];
  let content = wrench;
  if (kind === 'document') content = '<path d="M5 2h10l5 5v15H5Z" fill="#fff" stroke="#718da7" stroke-width="1.5" stroke-linejoin="round"/><path d="M15 2v6h5" fill="#bdcfe0"/><path d="M8 10h9M8 14h9M8 18h7" stroke="#718da7" stroke-width="1.5"/>';
  if (kind === 'schema') content = '<path d="M9 2H7Q5 2 5 5v3q0 4-3 4 3 0 3 4v3q0 3 3 3h1M15 2h2q2 0 2 3v3q0 4 3 4-3 0-3 4v3q0 3-3 3h-1" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>';
  if (kind === 'stack') content = '<rect x="7" y="2" width="15" height="15" rx="2" fill="#d9e8f4" stroke="#718da7"/><rect x="4" y="5" width="15" height="15" rx="2" fill="#edf5fb" stroke="#718da7"/><rect x="1" y="8" width="15" height="15" rx="2" fill="white" stroke="#718da7"/><g transform="translate(3 10) scale(.55)">' + wrench + '</g>';
  icons.push('<g transform="translate(' + n.left + ' ' + n.top + ') scale(' + n.width / 24 +
    ')" color="' + (kind === 'danger' ? '#ca433d' : '#008b88') + '">' + content + '</g>');
}
const markers = '<defs>' + [['flow', '#008b88'], ['feedback', '#008b88']].map(([id, color]) =>
  '<marker id="' + id + '" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse">' +
  '<path d="M0 0L12 6L0 12Z" fill="' + color + '"/></marker>').join('') + '</defs>';
const result = svg.replace(/(<svg[^>]*>)/, '$1' + markers).replace('</svg>', paths.join('') + icons.join('') + '</svg>');
fs.writeFileSync(OUT_SVG, result);
console.log(`wrote ${OUT_SVG} (${result.length} bytes; ${paths.length} connectors)`);
