// Pure diffing: a line diff (Myers O(ND) after trimming the common prefix and suffix) with
// unified hunks, and an option-level diff of two parsed configs. No DOM here.

/** Split into lines on '\n'. Joining the result with '\n' gives the input back exactly. */
export const splitLines = (text) => String(text ?? '').split('\n');

/**
 * Myers' greedy shortest edit script over two arrays of integers.
 * Returns ops [{type: 'eq'|'del'|'add', a, b}] where a/b are indices (or -1).
 */
function myers(A, B, aOff, bOff, out) {
  const n = A.length;
  const m = B.length;
  if (n === 0 && m === 0) return;
  if (n === 0) {
    for (let y = 0; y < m; y += 1) out.push({ type: 'add', a: -1, b: bOff + y });
    return;
  }
  if (m === 0) {
    for (let x = 0; x < n; x += 1) out.push({ type: 'del', a: aOff + x, b: -1 });
    return;
  }
  const max = n + m;
  const off = max + 1;
  const v = new Int32Array(2 * max + 3);
  const trace = [];
  let done = false;
  for (let d = 0; d <= max && !done; d += 1) {
    trace.push(v.slice());
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && v[off + k - 1] < v[off + k + 1]) ? v[off + k + 1] : v[off + k - 1] + 1;
      let y = x - k;
      while (x < n && y < m && A[x] === B[y]) {
        x += 1;
        y += 1;
      }
      v[off + k] = x;
      if (x >= n && y >= m) {
        done = true;
        break;
      }
    }
  }
  // walk the trace backwards from (n, m); ops come out reversed
  const rev = [];
  let x = n;
  let y = m;
  for (let d = trace.length - 1; d >= 0; d -= 1) {
    const vd = trace[d];
    const k = x - y;
    const prevK = k === -d || (k !== d && vd[off + k - 1] < vd[off + k + 1]) ? k + 1 : k - 1;
    const prevX = vd[off + prevK];
    const prevY = prevX - prevK;
    while (x > prevX && y > prevY) {
      x -= 1;
      y -= 1;
      rev.push({ type: 'eq', a: aOff + x, b: bOff + y });
    }
    if (d > 0) {
      if (x === prevX) rev.push({ type: 'add', a: -1, b: bOff + prevY });
      else rev.push({ type: 'del', a: aOff + prevX, b: -1 });
    }
    x = prevX;
    y = prevY;
  }
  for (let i = rev.length - 1; i >= 0; i -= 1) out.push(rev[i]);
}

/**
 * Line diff.
 * @returns {{aLines: string[], bLines: string[], ops: Array<{type: 'eq'|'del'|'add', a: number, b: number}>, added: number, removed: number}}
 */
export function diffLines(aText, bText) {
  const aLines = splitLines(aText);
  const bLines = splitLines(bText);
  // compare integers, not strings, inside the O(ND) loop
  const ids = new Map();
  const idOf = (line) => {
    let id = ids.get(line);
    if (id === undefined) {
      id = ids.size;
      ids.set(line, id);
    }
    return id;
  };
  const a = aLines.map(idOf);
  const b = bLines.map(idOf);
  let pre = 0;
  while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre += 1;
  let suf = 0;
  while (suf < a.length - pre && suf < b.length - pre && a[a.length - 1 - suf] === b[b.length - 1 - suf]) suf += 1;

  const ops = [];
  for (let i = 0; i < pre; i += 1) ops.push({ type: 'eq', a: i, b: i });
  myers(a.slice(pre, a.length - suf), b.slice(pre, b.length - suf), pre, pre, ops);
  for (let i = suf; i > 0; i -= 1) ops.push({ type: 'eq', a: a.length - i, b: b.length - i });

  let added = 0;
  let removed = 0;
  for (const op of ops) {
    if (op.type === 'add') added += 1;
    else if (op.type === 'del') removed += 1;
  }
  return { aLines, bLines, ops, added, removed };
}

/**
 * Group ops into unified hunks with `context` unchanged lines around each change.
 * Each hunk: {aStart, aLen, bStart, bLen, opStart, opEnd} (zero-based, ranges in lines / ops).
 */
export function buildHunks(ops, context = 3) {
  const aPos = new Int32Array(ops.length + 1);
  const bPos = new Int32Array(ops.length + 1);
  for (let i = 0; i < ops.length; i += 1) {
    aPos[i + 1] = aPos[i] + (ops[i].type !== 'add' ? 1 : 0);
    bPos[i + 1] = bPos[i] + (ops[i].type !== 'del' ? 1 : 0);
  }
  const hunks = [];
  let current = null;
  for (let i = 0; i < ops.length; i += 1) {
    if (ops[i].type === 'eq') continue;
    const start = Math.max(0, i - context);
    let j = i;
    while (j + 1 < ops.length && ops[j + 1].type !== 'eq') j += 1;
    const end = Math.min(ops.length, j + 1 + context);
    if (current && start <= current.opEnd) current.opEnd = end;
    else {
      current = { opStart: start, opEnd: end, aStart: 0, aLen: 0, bStart: 0, bLen: 0 };
      hunks.push(current);
    }
    i = j;
  }
  for (const hunk of hunks) {
    hunk.aStart = aPos[hunk.opStart];
    hunk.aLen = aPos[hunk.opEnd] - hunk.aStart;
    hunk.bStart = bPos[hunk.opStart];
    hunk.bLen = bPos[hunk.opEnd] - hunk.bStart;
  }
  return hunks;
}

/** Full diff: lines, ops, hunks and counts. */
export function diffText(aText, bText, context = 3) {
  const result = diffLines(aText, bText);
  return { ...result, hunks: buildHunks(result.ops, context) };
}

/** Rebuild B from A and the diff's hunks (proves the hunks are complete). */
export function applyHunks(diff) {
  const { aLines, bLines, ops, hunks } = diff;
  const out = [];
  let a = 0;
  for (const hunk of hunks) {
    while (a < hunk.aStart) out.push(aLines[a++]);
    for (let i = hunk.opStart; i < hunk.opEnd; i += 1) if (ops[i].type !== 'del') out.push(bLines[ops[i].b]);
    a = hunk.aStart + hunk.aLen;
  }
  while (a < aLines.length) out.push(aLines[a++]);
  return out.join('\n');
}

/** B's text with one hunk put back to A's lines ("Revert this change"). */
export function revertHunk(diff, hunk) {
  const { aLines, bLines } = diff;
  return [...bLines.slice(0, hunk.bStart), ...aLines.slice(hunk.aStart, hunk.aStart + hunk.aLen), ...bLines.slice(hunk.bStart + hunk.bLen)].join('\n');
}

// ───────── option-level diff ─────────

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/** Lists of objects that all carry a string `type` (losses, metrics lists) match by type. */
const isTypedList = (v) => Array.isArray(v) && v.length > 0 && v.every((item) => isPlainObject(item) && typeof item.type === 'string');

/**
 * Leaf options of a config, in document order.
 * @returns {{leaves: Map<string, {path: string, value: any, edit: Array<string|number>, chain: Array}>, items: Map<string, Array<string|number>>}}
 *   leaves: display path -> leaf; items: display path of each type-matched item -> its real edit path.
 *   Display paths look like `train.losses[type=mssimloss].loss_weight`; the second item of the same
 *   type is `[type=ganloss#2]`.
 */
export function flattenConfig(config) {
  const leaves = new Map();
  const items = new Map();
  const walk = (node, display, edit, chain) => {
    if (isPlainObject(node) && Object.keys(node).length > 0) {
      for (const [key, value] of Object.entries(node)) {
        const next = display ? `${display}.${key}` : key;
        walk(value, next, [...edit, key], [...chain, { typed: false, seg: key, display: next }]);
      }
      return;
    }
    if (isTypedList(node)) {
      const seen = new Map();
      node.forEach((item, index) => {
        const n = (seen.get(item.type) ?? 0) + 1;
        seen.set(item.type, n);
        const next = `${display}[type=${item.type}${n > 1 ? `#${n}` : ''}]`;
        const itemEdit = [...edit, index];
        items.set(next, itemEdit);
        walk(item, next, itemEdit, [...chain, { typed: true, seg: index, display: next }]);
      });
      return;
    }
    leaves.set(display, { path: display, value: node, edit, chain });
  };
  if (isPlainObject(config)) walk(config, '', [], []);
  return { leaves, items };
}

/** JSON with sorted keys so equal objects compare equal whatever their key order. */
export function stableStringify(value) {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  if (isPlainObject(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stableStringify(value[k])}`)
      .join(',')}}`;
  }
  return value === undefined ? 'null' : JSON.stringify(value);
}

/** Translate a leaf of B into an edit path for A; null when a type-matched item is missing in A. */
function editPathInA(leafB, flatA) {
  let path = [];
  for (const step of leafB.chain) {
    if (step.typed) {
      const real = flatA.items.get(step.display);
      if (!real) return null;
      path = [...real];
    } else path.push(step.seg);
  }
  return path;
}

/** Group name for options that are not nested under a top-level key. */
export const TOP_LEVEL_GROUP = 'Top-level options';

/**
 * Option-level comparison.
 * @returns {Array<{path: string, group: string, state: 'different'|'only-a'|'only-b'|'same', a: any, b: any, hasA: boolean, hasB: boolean, edit: Array|null}>}
 *   edit is where B's value would go in A (null when it cannot be expressed as one edit).
 */
export function diffConfigs(configA, configB) {
  const flatA = flattenConfig(configA);
  const flatB = flattenConfig(configB);
  const order = [...flatA.leaves.keys()];
  for (const key of flatB.leaves.keys()) if (!flatA.leaves.has(key)) order.push(key);
  const rows = order.map((path) => {
    const la = flatA.leaves.get(path);
    const lb = flatB.leaves.get(path);
    // Options that sit directly at the top of the file share one group; a header per single
    // option would double the table. Nested options group under their top-level key.
    const group = /[.[]/.test(path) ? path.split(/[.[]/)[0] : TOP_LEVEL_GROUP;
    let state;
    if (la && lb) state = stableStringify(la.value) === stableStringify(lb.value) ? 'same' : 'different';
    else state = la ? 'only-a' : 'only-b';
    let edit = null;
    if (state === 'different') edit = la.edit;
    else if (state === 'only-b') edit = editPathInA(lb, flatA);
    return { path, group, state, a: la?.value, b: lb?.value, hasA: Boolean(la), hasB: Boolean(lb), edit };
  });
  // group rows by top-level key, groups in order of first appearance
  const groups = new Map();
  for (const row of rows) {
    if (!groups.has(row.group)) groups.set(row.group, []);
    groups.get(row.group).push(row);
  }
  return [...groups.values()].flat();
}

/** Compact JSON-ish text for a value: strings without quotes, lists as [a, b]. */
export function compactValue(value) {
  if (value === undefined) return '';
  if (value === null) return 'null';
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return `[${value.map(compactValue).join(', ')}]`;
  if (isPlainObject(value)) return `{${Object.entries(value).map(([k, v]) => `${k}: ${compactValue(v)}`).join(', ')}}`;
  return String(value);
}
