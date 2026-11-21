// Result of a completed weight drift job (analysis kind `weight-drift`): how far the stored values of a
// candidate checkpoint moved from a baseline. The producer is studio/weight_drift.py; it reads F16, BF16
// and F32 tensors that have the same name and shape in both files. Nothing here judges quality.
//
//   renderWeightDrift(host, {job, result, artifacts}) -> cleanup
//   driftProgressText(progress) -> one sentence, or ''
//   previousCheckpoint(path, paths) -> the checkpoint of the same kind saved just before `path`, or null

import { h, fmt } from '../dom.js';
import { notice } from '../ui.js';
import { sizeText, shapeText } from '../components/weights.js';

const LISTED_MAX = 24;
const CLOSING =
  'These numbers measure how far the weights moved, not whether the model got better. Quality needs an evaluation: validation scores or a run comparison.';

// ───────── pure helpers (exported for checks) ─────────

/**
 * A measure with the digits that differ visible: three significant digits, scientific notation below
 * 0.001 and from 100,000 up ("1.81e-4"). '' when absent.
 */
export function measureText(value) {
  if (!Number.isFinite(value)) return '';
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs < 1e-3 || abs >= 1e5) return value.toExponential(2).replace('e+', 'e');
  return String(Number(value.toPrecision(3)));
}

/** relativeL2 as a percentage: "0.251 %". '' when absent. */
export function percentText(ratio) {
  if (!Number.isFinite(ratio)) return '';
  return `${measureText(ratio * 100)} %`;
}

/**
 * The cosine and its distance from 1: "0.999997, which is 3.16e-6 below 1". Close to 1 it gets as many
 * decimals as it takes to tell it from 1; further away, three significant digits. '' when absent.
 */
export function cosineText(cosine) {
  if (!Number.isFinite(cosine)) return '';
  const gap = 1 - cosine;
  if (gap <= 0) return '1, the same direction to the precision of the measurement';
  const shown = gap < 0.1 ? cosine.toFixed(Math.min(16, Math.max(3, Math.floor(-Math.log10(gap)) + 1))) : measureText(cosine);
  return `${shown}, which is ${measureText(gap)} below 1`;
}

const baseName = (path) => {
  const text = String(path ?? '');
  return text.slice(Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\')) + 1);
};
const folderOf = (path) => {
  const text = String(path ?? '');
  const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
  return cut > 0 ? text.slice(0, cut) : '';
};
const count = (n) => (Number.isFinite(n) && n > 0 ? n : 0);

/**
 * The checkpoint of the same kind saved just before `path`: the same folder and file-name stem with the
 * next smaller iteration number (models/net_g_8.safetensors -> models/net_g_2.safetensors when those are
 * the only two; net_g_ema_8 -> net_g_ema_2). Null when `path` has no iteration number or none is older.
 * @param {string} path
 * @param {string[]} paths every file of the run, in any order
 * @returns {string|null}
 */
export function previousCheckpoint(path, paths) {
  const parse = (p) => {
    const text = String(p ?? '');
    const m = /^(.*?)_(\d+)\.safetensors$/i.exec(baseName(text));
    return m ? { folder: folderOf(text), stem: m[1], step: Number(m[2]) } : null;
  };
  const self = parse(path);
  if (!self) return null;
  let best = null;
  for (const other of Array.isArray(paths) ? paths : []) {
    const o = parse(other);
    if (!o || o.folder !== self.folder || o.stem !== self.stem || !(o.step < self.step)) continue;
    if (!best || o.step > best.step) best = { ...o, path: String(other) };
  }
  return best ? best.path : null;
}

/** One plain sentence about a running drift job, or '' when its progress is absent or not understood. */
export function driftProgressText(progress) {
  if (!progress || typeof progress !== 'object' || progress.stage !== 'comparing') return '';
  const done = progress.completedTensors;
  const total = progress.totalTensors;
  if (!Number.isFinite(done) || !Number.isFinite(total)) return '';
  const read = sizeText(progress.payloadBytesRead);
  return `${fmt.int(done)} of ${fmt.plural(total, 'tensor')} compared${read ? `, ${read} of weight data read` : ''}.`;
}

/** The plain-words count lines; "compared" is always said, every other zero count is left out. */
export function tensorCountLines(tensors) {
  const t = tensors && typeof tensors === 'object' ? tensors : {};
  const known = { measured: 'compared', leftOnly: 'only in the baseline', rightOnly: 'only in the candidate', differentShape: 'with a different shape', nonTrainingFloatOrOtherType: 'not compared: whole-number or other types' };
  const lines = [`${fmt.plural(count(t.measured), 'tensor')} compared`];
  for (const key of ['leftOnly', 'rightOnly', 'differentShape', 'nonTrainingFloatOrOtherType']) {
    if (count(t[key])) lines.push(`${fmt.plural(t[key], 'tensor')} ${known[key]}`);
  }
  // A status this screen does not know yet is still counted, under its own name.
  for (const [key, n] of Object.entries(t)) {
    if (!(key in known) && count(n)) lines.push(`${fmt.plural(n, 'tensor')} with the status ${key}`);
  }
  return lines;
}

/** Tensors that were not compared value by value (every status except "measured"). */
export function uncomparedCount(tensors) {
  const t = tensors && typeof tensors === 'object' ? tensors : {};
  return Object.entries(t).reduce((sum, [key, n]) => (key === 'measured' ? sum : sum + count(n)), 0);
}

/** The one sentence under the file line. */
export function leadSentence(result) {
  const n = result?.numerical ?? {};
  const pairs = n.finitePairs;
  const changed = n.changedFinite;
  const tensors = result?.tensors ?? {};
  if (!Number.isFinite(pairs) || !Number.isFinite(changed)) return 'The job did not report how many values it compared.';
  if (pairs === 0) {
    return count(tensors.measured)
      ? 'Nothing could be measured: the compared tensors hold no pair of finite values.'
      : 'Nothing could be measured: no tensor has the same name, the same shape and a floating-point type (F16, BF16 or F32) in both files. The counts below say what was found instead.';
  }
  if (changed === 0) {
    // "The same numbers" only when every tensor was compared and every value was finite.
    const whole = uncomparedCount(tensors) === 0 && !count(n.leftNonfinite) && !count(n.rightNonfinite);
    return whole ? 'No value changed: the two files hold the same numbers.' : `None of the ${fmt.int(pairs)} compared values changed.`;
  }
  return `${fmt.int(changed)} of ${fmt.int(pairs)} compared values changed.`;
}

// ───────── renderer ─────────

const MEASURES = [
  { key: 'rmsDelta', label: 'Typical change (RMS)', text: measureText, explain: 'Root mean square of the change, over every compared value.' },
  { key: 'meanAbsoluteDelta', label: 'Mean absolute change', text: measureText, explain: 'Average size of the change, whatever its sign.' },
  { key: 'maximumAbsoluteDelta', label: 'Largest single change', text: measureText, explain: 'The one value that moved furthest.' },
  {
    key: 'relativeL2',
    label: 'Change relative to the baseline',
    text: (v) => (Number.isFinite(v) ? `${percentText(v)} of the baseline's overall size` : ''),
    explain: 'Size of all the changes together, divided by the size of all the baseline values (both as L2 norms).',
  },
  {
    key: 'cosine',
    label: 'Direction kept (cosine)',
    text: cosineText,
    explain: 'All values taken as one direction: 1 means the candidate points the same way as the baseline, 0 unrelated, -1 opposite.',
  },
];

function fileLine(left, right) {
  const file = (path) => h('span', { class: 'mono', title: path || null }, baseName(path) || 'Not reported');
  const a = folderOf(left);
  const b = folderOf(right);
  if (left && left === right) {
    return h('p', { class: 'drift__files' }, 'The baseline and the candidate are the same file: ', file(left), a ? [' in ', h('span', { class: 'mono drift__folder' }, a)] : null, '.');
  }
  if (a && a === b) {
    return h('p', { class: 'drift__files' }, 'Baseline ', file(left), ', candidate ', file(right), ', both in ', h('span', { class: 'mono drift__folder' }, a), '.');
  }
  const where = (folder) => (folder ? [' in ', h('span', { class: 'mono drift__folder' }, folder)] : null);
  return h('p', { class: 'drift__files' }, 'Baseline ', file(left), where(a), '. Candidate ', file(right), where(b), '.');
}

function nonfiniteNotices(n) {
  const line = (value, side) =>
    count(value) ? notice('warn', `${fmt.int(value)} ${value === 1 ? 'value' : 'values'} in the ${side} ${value === 1 ? 'is' : 'are'} not finite (NaN or infinity).`) : null;
  return [line(n.rightNonfinite, 'candidate'), line(n.leftNonfinite, 'baseline')];
}

function measuresGrid(n) {
  return h(
    'dl',
    { class: 'drift__measures' },
    MEASURES.map((m) => {
      const text = m.text(n[m.key]);
      return h(
        'div',
        { class: 'drift__measure' },
        h('dt', { class: 'drift__label' }, m.label),
        h('dd', { class: text ? 'drift__value' : 'drift__value drift__absent' }, text || 'Not measured'),
        h('dd', { class: 'drift__explain' }, m.explain),
      );
    }),
  );
}

function typeText(row) {
  const a = row.left?.dtype ?? '';
  const b = row.right?.dtype ?? '';
  if (!a && !b) return '';
  return a === b || !b ? a : !a ? b : `${a} in the baseline, ${b} in the candidate`;
}

function shapeCell(row) {
  const a = shapeText(row.left?.shape);
  const b = shapeText(row.right?.shape);
  return a === b || !b ? a : !a ? b : `${a} and ${b}`;
}

function movedTable(rows) {
  const largest = rows.reduce((max, row) => (Number.isFinite(row.rmsDelta) && row.rmsDelta > max ? row.rmsDelta : max), 0);
  const width = (value) => (largest > 0 && Number.isFinite(value) ? `${Math.max(0, Math.min(100, (value / largest) * 100)).toFixed(1)}%` : '0%');
  return h(
    'div',
    { class: 'drift__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Tensors that moved most' },
    h(
      'table',
      { class: 'table drift__table' },
      h('caption', { class: 'drift__caption' }, `At most ${LISTED_MAX} tensors are listed here. The file below has one row for every tensor.`),
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', { scope: 'col' }, 'Tensor'),
          h('th', { scope: 'col' }, 'Shape'),
          h('th', { scope: 'col' }, 'Type'),
          h('th', { scope: 'col', class: 'num' }, 'RMS change'),
          h('th', { scope: 'col', class: 'num' }, 'Relative to the baseline'),
          h('th', { scope: 'col', class: 'num' }, 'Largest change'),
          h('th', { scope: 'col', class: 'num' }, 'Values changed'),
        ),
      ),
      h(
        'tbody',
        null,
        rows.map((row) =>
          h(
            'tr',
            null,
            h('td', { class: 'drift__name' }, String(row.name ?? '')),
            h('td', { class: 'drift__shape' }, shapeCell(row)),
            h('td', { class: 'drift__type' }, typeText(row)),
            h(
              'td',
              { class: 'num drift__rms' },
              h('span', { class: 'drift__rms-value' }, measureText(row.rmsDelta) || 'Not measured'),
              h('span', { class: 'drift__track', 'aria-hidden': 'true' }, h('span', { class: 'drift__bar', style: { width: width(row.rmsDelta) } })),
            ),
            h('td', { class: 'num' }, percentText(row.relativeL2) || 'Not measured'),
            h('td', { class: 'num' }, measureText(row.maximumAbsoluteDelta) || 'Not measured'),
            h('td', { class: 'num' }, Number.isFinite(row.changedFinite) && Number.isFinite(row.finitePairs) ? `${fmt.int(row.changedFinite)} of ${fmt.int(row.finitePairs)}` : 'Not reported'),
          ),
        ),
      ),
    ),
  );
}

/**
 * Render a completed weight drift job into `host`.
 * @param {HTMLElement} host
 * @param {{job?: object, result: object, artifacts?: Array<{path: string, url: string}>}} answer
 *   `result` and `artifacts` as GET /api/analysis/result gives them
 * @returns {() => void} cleanup
 */
export function renderWeightDrift(host, { job = null, result = null, artifacts = [] } = {}) {
  const r = result && typeof result === 'object' ? result : {};
  const n = r.numerical && typeof r.numerical === 'object' ? r.numerical : {};
  const left = typeof r.left === 'string' ? r.left : job?.params?.left;
  const right = typeof r.right === 'string' ? r.right : job?.params?.right;
  const tensors = r.tensors && typeof r.tensors === 'object' ? r.tensors : {};
  const rows = Array.isArray(r.largestRmsChanges) ? r.largestRmsChanges.filter((row) => row && typeof row === 'object') : [];
  const measured = Number.isFinite(n.finitePairs) && n.finitePairs > 0;
  const uncompared = uncomparedCount(tensors);
  const files = Array.isArray(artifacts) ? artifacts.filter((a) => a && typeof a.url === 'string' && a.url) : [];
  const rowsFile = files.find((a) => baseName(a.path) === 'weight-drift.jsonl') ?? files.find((a) => /\.jsonl$/i.test(String(a.path ?? ''))) ?? null;

  let emptyText = '';
  if (!rows.length) emptyText = !measured ? 'No tensor is listed: nothing was measured.' : n.changedFinite === 0 ? 'No tensor is listed: no compared value changed.' : 'No tensor is listed: the job did not report any.';
  // Compared tensors that kept every value are left out of the list; say how many when the count is certain.
  const unchanged = rows.length && rows.length < LISTED_MAX ? count(tensors.measured) - rows.length : 0;
  const read = sizeText(r.payloadBytesRead);

  const root = h(
    'div',
    { class: 'drift' },
    fileLine(left, right),
    h('p', { class: 'drift__lead' }, leadSentence(r)),
    nonfiniteNotices(n),
    measured ? measuresGrid(n) : null,
    h('h4', { class: 'drift__title' }, 'Tensors'),
    h('ul', { class: 'drift__counts' }, tensorCountLines(tensors).map((text) => h('li', { class: 'drift__count' }, text))),
    uncompared
      ? notice(
          '',
          `${fmt.plural(uncompared, 'tensor')} ${uncompared === 1 ? 'was' : 'were'} not compared value by value (only tensors with the same name and shape and an F16, BF16 or F32 type are). The downloadable file below lists each one with its reason.`,
        )
      : null,
    h('h4', { class: 'drift__title' }, 'Tensors that moved most'),
    rows.length ? movedTable(rows) : h('p', { class: 'drift__note' }, emptyText),
    unchanged > 0 ? h('p', { class: 'drift__note' }, `${fmt.plural(unchanged, 'compared tensor')} ${unchanged === 1 ? 'is' : 'are'} not listed: none of ${unchanged === 1 ? 'its' : 'their'} compared values changed.`) : null,
    h(
      'div',
      { class: 'drift__files-out' },
      rowsFile
        ? h('a', { class: 'btn btn--sm', href: rowsFile.url, download: baseName(rowsFile.path) || 'weight-drift.jsonl' }, 'Download all tensor rows (JSON lines)')
        : h('p', { class: 'drift__note' }, 'The file with one row per tensor was not reported.'),
      read ? h('p', { class: 'drift__note' }, `Read ${read} of weight data.`) : null,
    ),
    h('p', { class: 'drift__closing' }, CLOSING),
  );
  host.appendChild(root);
  return () => root.remove();
}
