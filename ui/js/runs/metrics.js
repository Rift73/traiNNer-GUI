// Pure helpers for the Runs screen: group logged history series, turn history points into chart
// points (nulls kept as gaps), classify log lines; plus the older readers of a loose `metrics` payload.
// No DOM access here so the logic can be checked with plain Node.

const X_KEYS = ['iteration', 'iter', 'step', 'x'];
const Y_KEYS = ['value', 'y'];

/**
 * A reported number. Numbers pass through; numeric strings ("0.5") are read; everything else
 * (null, "nan", objects) becomes NaN so it is counted as not drawn rather than invented.
 */
function toNumber(value) {
  if (typeof value === 'number') return value;
  if (typeof value === 'string' && value.trim() !== '') return Number(value);
  return NaN;
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function pickKey(obj, keys) {
  for (const key of keys) if (Object.prototype.hasOwnProperty.call(obj, key)) return key;
  return null;
}

function seriesFor(acc, name, xIsIteration) {
  let entry = acc.get(name);
  if (!entry) {
    entry = { name, points: [], xIsIteration, dropped: 0 };
    acc.set(name, entry);
  } else if (!xIsIteration) {
    entry.xIsIteration = false;
  }
  return entry;
}

function addPoint(entry, x, y) {
  if (Number.isFinite(x) && Number.isFinite(y)) entry.points.push({ x, y });
  else entry.dropped += 1;
}

/** One array of values for one name: [[x, y]], [{iteration, value}], or [y, ...]. */
function readArray(name, list, acc, depth) {
  const first = list.find((item) => item !== null && item !== undefined);
  // Rows nested one level down: {group: [{iteration, a, b}, ...]} -> group/a, group/b.
  // A list is point objects as soon as any element carries value|y (one may lack it: counted as dropped).
  const pointObjects = list.some((item) => isPlainObject(item) && pickKey(item, Y_KEYS));
  if (isPlainObject(first) && !pointObjects) {
    if (depth === 0) readRows(list, `${name}/`, acc);
    return;
  }
  const byIndex = !Array.isArray(first) && !isPlainObject(first);
  const entry = seriesFor(acc, name, !byIndex);
  list.forEach((item, index) => {
    if (Array.isArray(item)) {
      addPoint(entry, toNumber(item[0]), toNumber(item[1]));
    } else if (isPlainObject(item)) {
      const xKey = pickKey(item, X_KEYS);
      const yKey = pickKey(item, Y_KEYS);
      if (xKey) addPoint(entry, toNumber(item[xKey]), yKey ? toNumber(item[yKey]) : NaN);
      else {
        entry.xIsIteration = false;
        addPoint(entry, index, yKey ? toNumber(item[yKey]) : NaN);
      }
    } else if (byIndex) {
      addPoint(entry, index, toNumber(item));
    } else {
      entry.dropped += 1; // a bare number inside a list of pairs: no x to place it at
    }
  });
}

/** Rows: [{iteration, name: value, ...}]. A row without an x key is placed at its index. */
function readRows(rows, prefix, acc) {
  const seen = new Map(); // name -> did any row carry a number for it
  rows.forEach((row, index) => {
    if (!isPlainObject(row)) return;
    const xKey = pickKey(row, X_KEYS);
    const x = xKey ? toNumber(row[xKey]) : index;
    for (const [key, value] of Object.entries(row)) {
      if (key === xKey || isPlainObject(value) || Array.isArray(value)) continue;
      const y = toNumber(value);
      const name = prefix + key;
      if (!seen.has(name)) seen.set(name, false);
      if (Number.isFinite(y)) seen.set(name, true);
      addPoint(seriesFor(acc, name, Boolean(xKey)), x, y);
    }
  });
  // A column that never held a number (a timestamp string, a tag) is not a metric.
  for (const [name, numeric] of seen) if (!numeric && acc.get(name)?.points.length === 0) acc.delete(name);
}

function readEntry(name, value, acc, scalars, depth) {
  if (typeof value === 'number' || (typeof value === 'string' && value.trim() !== '')) {
    scalars.push({ name, value });
  } else if (Array.isArray(value)) {
    if (value.length) readArray(name, value, acc, depth);
  } else if (isPlainObject(value) && depth === 0) {
    for (const [key, inner] of Object.entries(value)) readEntry(`${name}/${key}`, inner, acc, scalars, 1);
  }
  // Anything else (booleans, deeper nesting, null) is not a shape the contract names: ignored.
}

/**
 * @param {unknown} raw the run's `metrics` field as the API returned it
 * @returns {{series: Array<{name: string, points: Array<{x: number, y: number}>, xIsIteration: boolean, dropped: number}>, scalars: Array<{name: string, value: number|string}>}}
 */
export function normalizeMetrics(raw) {
  const acc = new Map();
  const scalars = [];
  try {
    if (Array.isArray(raw)) readRows(raw, '', acc);
    else if (isPlainObject(raw)) for (const [name, value] of Object.entries(raw)) readEntry(name, value, acc, scalars, 0);
  } catch {
    // A payload odd enough to throw is treated like an unrecognised one.
  }
  const series = [...acc.values()];
  for (const entry of series) {
    // Stable sort: equal x keep their reported order.
    if (entry.points.some((p, i) => i > 0 && p.x < entry.points[i - 1].x)) entry.points.sort((a, b) => a.x - b.x);
  }
  return { series, scalars };
}

// ───────── grouping ─────────

const VALIDATION_WORDS = /psnr|ssim|dists|lpips|topiq/;

const PANELS = [
  { id: 'loss', title: 'Loss', test: (full, leaf) => leaf.startsWith('l_') || full.includes('loss') },
  { id: 'lr', title: 'Learning rate', test: (full, leaf) => leaf === 'lr' || leaf.startsWith('lr') },
  { id: 'val', title: 'Validation', test: (full, leaf) => full.startsWith('val') || leaf.startsWith('val') || VALIDATION_WORDS.test(full) },
  { id: 'other', title: 'Other', test: () => true },
];

/** Panel id for a series name. A name under a `val…` group is validation even when it says loss. */
export function panelOf(name) {
  const full = String(name).toLowerCase();
  const leaf = full.slice(full.lastIndexOf('/') + 1);
  if (full.startsWith('val')) return 'val';
  return PANELS.find((panel) => panel.test(full, leaf)).id;
}

function magnitude(series) {
  let m = 0;
  for (const p of series.points) m = Math.max(m, Math.abs(p.y));
  return m;
}

/** Ranges further apart than this do not share a y axis (PSNR ~30 against SSIM ~0.9). */
export const SHARED_AXIS_RATIO = 20;

function rangesDiffer(list) {
  const mags = list.map(magnitude).filter((m) => m > 0);
  if (mags.length < 2) return false;
  return Math.max(...mags) / Math.min(...mags) > SHARED_AXIS_RATIO;
}

/**
 * Panels in display order, each with one or more charts.
 * @returns {Array<{id: string, title: string, charts: Array<{id: string, title: string|null, xIsIteration: boolean, series: Array}>}>}
 */
export function groupSeries(series) {
  const out = [];
  for (const panel of PANELS) {
    const members = series.filter((s) => panelOf(s.name) === panel.id);
    if (!members.length) continue;
    const charts = [];
    // Series without iterations are drawn against sample index: never on the same x axis as iterations.
    for (const byIteration of [true, false]) {
      const list = members.filter((s) => s.xIsIteration === byIteration);
      if (!list.length) continue;
      const suffix = byIteration ? '' : ' (by sample index)';
      const split = (panel.id === 'val' || panel.id === 'other') && rangesDiffer(list);
      if (split) {
        for (const s of list) charts.push({ id: `${panel.id}:${byIteration ? 'it' : 'ix'}:${s.name}`, title: s.name + suffix, xIsIteration: byIteration, series: [s] });
      } else {
        charts.push({ id: `${panel.id}:${byIteration ? 'it' : 'ix'}`, title: suffix ? panel.title + suffix : null, xIsIteration: byIteration, series: list });
      }
    }
    out.push({ id: panel.id, title: panel.title, charts });
  }
  return out;
}

// ───────── statistics ─────────

/** Last, minimum and maximum of the points reported (non-finite values never reach `points`). */
export function seriesStats(series) {
  const points = series?.points ?? [];
  if (!points.length) return { count: 0, last: NaN, lastX: NaN, min: NaN, max: NaN, minX: NaN, maxX: NaN };
  let min = points[0];
  let max = points[0];
  for (const p of points) {
    if (p.y < min.y) min = p;
    if (p.y > max.y) max = p;
  }
  const last = points[points.length - 1];
  return { count: points.length, last: last.y, lastX: last.x, min: min.y, max: max.y, minX: min.x, maxX: max.x };
}

// ───────── log ─────────

/**
 * The API returns the whole log tail on every poll. Find the largest k such that the last k lines
 * already shown equal the first k lines of the new tail, and append only what follows.
 * @param {string[]} prev lines on screen
 * @param {string[]} next the tail just fetched
 * @returns {{reset: boolean, append: string[]}} reset: replace everything with `next`
 */
export function mergeLogTail(prev, next) {
  const incoming = Array.isArray(next) ? next.map((line) => String(line ?? '')) : [];
  if (!prev.length) return { reset: true, append: incoming };
  if (!incoming.length) return { reset: false, append: [] };
  for (let k = Math.min(prev.length, incoming.length); k > 0; k -= 1) {
    const offset = prev.length - k;
    let match = true;
    // Compare from the far end first: a tail that moved on fails fast on its newest line.
    for (let i = k - 1; i >= 0; i -= 1) {
      if (prev[offset + i] !== incoming[i]) {
        match = false;
        break;
      }
    }
    if (match) return { reset: false, append: incoming.slice(k) };
  }
  return { reset: true, append: incoming };
}

/** 'error' | 'warn' | '' for one log line. */
export function classifyLogLine(line) {
  if (/ERROR|Traceback|Error:|Exception/.test(line)) return 'error';
  if (/WARNING/.test(line)) return 'warn';
  return '';
}

// ───────── logged history (GET /api/run/history) ─────────
// The helpers above read the older, loose `metrics` payload and stay for other callers. The service now
// sends `metrics` as latest values only; series come from the history route, read with the helpers below.

/** Groups of history series keys, in display order. */
export const SERIES_GROUPS = [
  { id: 'loss', title: 'Losses' },
  { id: 'lr', title: 'Learning rate' },
  { id: 'performance', title: 'Performance' },
  { id: 'memory', title: 'Memory' },
  { id: 'other', title: 'Other' },
];

/** Group id of one history series key: `l_*` losses, `lr/*`, `performance/*`, `memory/*`, else other. */
export function groupOfKey(key) {
  const k = String(key ?? '');
  if (k.startsWith('l_')) return 'loss';
  if (k === 'lr' || k.startsWith('lr/')) return 'lr';
  if (k.startsWith('performance/')) return 'performance';
  if (k.startsWith('memory/')) return 'memory';
  return 'other';
}

/**
 * Series of a history answer grouped for the picker; groups without members are left out.
 * @param {Array<{key: string, count?: number, first?: number, last?: number}>} series
 * @returns {Array<{id: string, title: string, items: Array}>}
 */
export function groupHistorySeries(series) {
  const list = Array.isArray(series) ? series.filter((s) => s && typeof s.key === 'string') : [];
  return SERIES_GROUPS.map((group) => ({
    ...group,
    items: list.filter((s) => groupOfKey(s.key) === group.id).sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0)),
  })).filter((group) => group.items.length);
}

/** What to draw before the person picks: `l_g_total` when logged, else up to three losses, else the first key. */
export function defaultHistorySelection(keys, max = 3) {
  const list = [...(keys ?? [])].map(String);
  if (list.includes('l_g_total')) return ['l_g_total'];
  const losses = list.filter((k) => k.startsWith('l_')).sort();
  if (losses.length) return losses.slice(0, max);
  return list.length ? [list.sort()[0]] : [];
}

/**
 * History points `[[iteration, number|null]]` as chart points. A null (or non-finite) value stays
 * null so the chart draws a gap there, never a zero. Entries without a finite iteration are skipped.
 * @returns {Array<{x: number, y: number|null}>} sorted by x
 */
export function historyPoints(points) {
  const out = [];
  if (!Array.isArray(points)) return out;
  for (const item of points) {
    if (!Array.isArray(item)) continue;
    const x = item[0];
    const y = item[1];
    if (typeof x !== 'number' || !Number.isFinite(x)) continue;
    out.push({ x, y: typeof y === 'number' && Number.isFinite(y) ? y : null });
  }
  if (out.some((p, i) => i > 0 && p.x < out[i - 1].x)) out.sort((a, b) => a.x - b.x);
  return out;
}

/** "12.4 of 40.2 MB" for indexing progress; '' when either number is missing. */
export function bytesProgress(done, total) {
  if (!Number.isFinite(done) || !Number.isFinite(total) || total <= 0) return '';
  const mb = (n) => {
    const v = n / 1e6;
    return v >= 100 ? String(Math.round(v)) : String(Number(v.toFixed(1)));
  };
  return `${mb(Math.min(done, total))} of ${mb(total)} MB`;
}
