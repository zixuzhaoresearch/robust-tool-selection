/** @jsx h @jsxFrag Fragment */
// Figure 1: each catalog condition (left) is paired with the selector's response (right);
// the band below is the execution-feedback loop shared by all three.
// Layout is computed by Satori; render_teaser.mjs connects the named nodes.
// Numbers: Tables 6/23 (ambiguity), 8 (degradation), 12 (injection), 25 and 26 (adaptation).
export const h = (type, props, ...children) => {
  const flat = children.flat(Infinity).filter(c => c != null && c !== false);
  const p = { ...props, children: flat.length === 1 ? flat[0] : flat };
  if (type === 'div') p.style = { display: 'flex', flexShrink: 0, ...p.style };
  if (p.style) p.style = Object.fromEntries(Object.entries(p.style).filter(([, value]) => value !== undefined));
  return { type, props: p };
};
export const Fragment = ({ children }) => children;
const C = {
  ink: '#102345', muted: '#4a6176', navy: '#123c60', line: '#b4cee1',
  pale: '#f4f9fd', blue: '#eaf3fc', teal: '#008b88', sage: '#e9f7f4',
  red: '#ca433d', rose: '#fff0ec',
};
const box = (style, children, id) => h('div', { style, 'data-node': id }, children);
const col = (style, children, id) => box({ flexDirection: 'column', ...style }, children, id);
const row = (style, children, id) => box({ alignItems: 'center', ...style }, children, id);
const txt = (s, size = 25, style = {}) => box({ fontSize: size, lineHeight: 1.25, ...style }, s);
let iconCount = 0;
const icon = (kind, size = 40) => box({ width: size, height: size }, '', 'icon-' + kind + '-' + iconCount++);
const soft = 'linear-gradient(135deg, #ffffff 0%, #eef6fc 100%)';
const green = 'linear-gradient(135deg, #ffffff 0%, #e6f5f1 100%)';
const LW = 742, RW = 742, GAP = 124, ROW_H = 204;

const badge = n => box({ width: 44, height: 44, justifyContent: 'center', alignItems: 'center',
  background: 'linear-gradient(135deg, #ffdf93, #ffb83d)', color: C.ink, fontWeight: 900, fontSize: 28, borderRadius: 22 }, n);
const tool = (name, detail, danger = false) => row({
  flex: 1, minWidth: 0, padding: '9px 13px', border: '1.3px solid ' + (danger ? '#d6aaa2' : C.line),
  borderRadius: 9, background: danger ? 'linear-gradient(120deg, #fffafa, #ffeae5)' : '#fff', gap: 10,
}, [icon(danger ? 'danger' : 'tool', 34), col({ flex: 1, minWidth: 0 }, [
  txt(name, 23, { fontWeight: 700, color: danger ? C.red : C.navy }),
  txt(detail, 20, { color: C.muted, marginTop: 3 })])]);

const problem = (n, title, subtitle, body, id) => col({
  width: LW, height: ROW_H, padding: '15px 20px', border: '1.5px solid ' + C.line, borderRadius: 13, background: soft,
}, [
  row({ gap: 12 }, [badge(n), txt(title, 31, { fontWeight: 900 })]),
  txt(subtitle, 23, { color: C.muted, marginTop: 6, marginBottom: 14 }),
  body,
], id);

const answer = (title, how, number, unit, basis, id) => row({
  width: RW, height: ROW_H, padding: '15px 22px', border: '1.5px solid #76b7ad', borderRadius: 13, background: green, gap: 18,
}, [
  col({ flex: 1, minWidth: 0 }, [
    txt(title, 31, { fontWeight: 900, color: C.teal }),
    txt(how, 23, { color: C.ink, marginTop: 9 }),
    txt(basis, 20, { color: C.muted, marginTop: 10 }),
  ]),
  col({ width: 262, alignItems: 'center', justifyContent: 'center', padding: '14px 6px',
    background: '#fff', border: '1.3px solid #a9d3ca', borderRadius: 10 }, [
    txt(number, 35, { fontWeight: 900, color: C.teal, whiteSpace: 'nowrap' }),
    txt(unit, 20, { color: C.muted, marginTop: 6, textAlign: 'center', justifyContent: 'center' }),
  ]),
], id);

const pair = (left, right) => row({ gap: GAP, alignItems: 'stretch' }, [left, right]);

const step = (id, title, detail, dark = false, width = 232) => col({
  width, height: 104, justifyContent: 'center', alignItems: 'center', borderRadius: 11,
  background: dark ? 'linear-gradient(120deg, #123653, #176480)' : '#fff',
  border: '1.5px solid ' + (dark ? '#267c9f' : '#83b8dc'),
}, [txt(title, 25, { fontWeight: 900, color: dark ? '#fff' : C.navy }),
  txt(detail, 20, { color: dark ? '#cfe6ee' : C.muted, marginTop: 5 })], id);

const stat = (number, label) => col({ flex: 1, padding: '11px 14px', background: '#fff',
  border: '1.3px solid #a9d3ca', borderRadius: 10 }, [
  txt(number, 30, { fontWeight: 900, color: C.teal }),
  txt(label, 20, { color: C.muted, marginTop: 4 }),
]);

export default function Fig() {
  return col({ width: 1664, padding: '24px 28px 26px', fontFamily: 'Lato', color: C.ink, background: '#fff' }, [
    row({ gap: GAP, alignItems: 'flex-start', marginBottom: 14 }, [
      col({ width: LW }, [txt('Where description matching fails', 34, { fontWeight: 900 }),
        txt('Three catalog conditions, evaluated separately', 23, { color: C.muted, marginTop: 5 })]),
      col({ width: RW }, [txt('What the selector adds', 34, { fontWeight: 900 }),
        txt('One response per condition, all learned from execution feedback', 23, { color: C.muted, marginTop: 5 })]),
    ]),
    pair(
      problem('1', 'Ambiguity', 'Several benign tools match the query equally well.',
        row({ gap: 10 }, [tool('WebSum', 'Summarize web content'), tool('PageDigest', 'Summarize a webpage'),
          txt('?', 46, { color: C.red, fontWeight: 900, marginLeft: 2, marginRight: 4 })]), 'p1'),
      answer('Usage memory', 'Each tool is also represented by the queries it has served successfully.',
        '+12 to 14 pp', 'over TF-IDF retrieval', 'MetaTool, 5 to 199 candidates', 'a1')),
    box({ height: 16 }, ''),
    pair(
      problem('2', 'Degradation', 'Descriptions are truncated, swapped, or missing.',
        row({ gap: 10 }, [tool('DocParser', 'Description: missing'),
          col({ flex: 1, background: C.sage, border: '1.3px solid #a9c5ba', padding: '9px 13px', borderRadius: 9 },
            [txt('Schema survives', 23, { color: C.teal, fontWeight: 700 }),
              txt('{ url: string }', 20, { color: C.teal, marginTop: 3 })])]), 'p2'),
      answer('Schema features', 'Parameter names and schema structure remain when the text is gone.',
        '62.1%', '30.2% for description retrieval', 'BFCL, descriptions and names removed', 'a2')),
    box({ height: 16 }, ''),
    pair(
      problem('3', 'Injection', 'An attacker adds a tool to an untrusted catalog.',
        row({ gap: 10 }, [tool('Benign tools', 'Existing candidates'),
          txt('+', 30, { color: C.red, fontWeight: 700 }), tool('Malicious tool', 'Forged metadata', true)]), 'p3'),
      answer('ADAB penalty', 'An anomaly score fit only on selections that succeeded.',
        '6.9% → 1.7%', 'attack success rate', 'BFCL, handcrafted multi-field injection', 'a3')),
    box({ height: 22 }, ''),
    col({ padding: '18px 22px 20px', border: '1.5px solid #87accb', borderRadius: 13, background: soft }, [
      row({ gap: 12 }, [
        box({ width: 44, height: 44, justifyContent: 'center', alignItems: 'center',
          background: 'linear-gradient(135deg, #9fe0d6, #37b3a6)', color: '#fff', fontWeight: 900, fontSize: 30, borderRadius: 22 }, '+'),
        txt('Adaptation: the feedback loop behind all three', 31, { fontWeight: 900 }),
        txt('contextual bandit, no LLM call', 22, { color: C.muted, marginLeft: 'auto' }),
      ]),
      row({ marginTop: 26, gap: 30, alignItems: 'flex-start' }, [
        col({ width: 870 }, [
          row({ gap: 64 }, [
            step('s-select', 'Select', 'score 8 features', true),
            step('s-exec', 'Execute', 'call the tool'),
            step('s-obs', 'Observe', 'success or error'),
          ]),
          box({ height: 46 }, ''),
          row({ width: 824, height: 50, justifyContent: 'center', background: C.blue, borderRadius: 8 },
            txt('Update weights, usage memory, and ADAB from the outcome', 22, { color: C.navy, fontWeight: 700 }), 's-update'),
        ]),
        col({ flex: 1 }, [
          txt('Reaches what documentation does not state', 24, { fontWeight: 700, color: C.navy }),
          row({ gap: 12, marginTop: 12, alignItems: 'stretch' }, [
            stat('58.8% vs. 45.9%', 'identical documentation, different reliability'),
            stat('10.0% → 17.2%', 'task success of a multi-step LLM agent'),
          ]),
        ]),
      ]),
    ]),
  ]);
}
