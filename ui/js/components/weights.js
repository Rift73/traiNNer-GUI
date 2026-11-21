// Checkpoint structure dialog on POST /api/weights and POST /api/weights/compare. Structure only: the
// service reads the .safetensors header (names, shapes, types), never the stored values, so nothing here
// may say that two checkpoints hold the same values. Measuring the values is a weight drift analysis job
// (runs/analysis.js), offered from the comparison of two .safetensors files.
//
//   openWeightsDialog({path, compareWith}) -> Promise<void> when closed

import { h, mount as fill, fmt, debounce, uid } from '../dom.js';
import { api } from '../api.js';
import { openDialog, notice, tabs, copyText, toast, loading } from '../ui.js';

const PAGE = 100;
const FILTER_DELAY = 250;
const SAME_LAYOUT = 'Same layout: every tensor name, shape and type matches. The stored values were not compared.';
const DIFFERENT_LAYOUT = 'Different layout';
const VALUES_NOT_COMPARED = 'Only names, shapes and types were compared. The stored values were not.';
const isSafetensors = (p) => typeof p === 'string' && /\.safetensors$/i.test(p);

// ───────── pure helpers (exported for checks) ─────────

/** "47.5 kB" style size (decimal units) next to the exact byte count, '' when absent. */
export function sizeText(bytes) {
  if (!Number.isFinite(bytes)) return '';
  const units = ['bytes', 'kB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return unit === 0 ? fmt.plural(bytes, 'byte') : `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]} (${fmt.plural(bytes, 'byte')})`;
}

/** "16 × 3 × 3 × 3"; a tensor with no dimensions is a scalar. */
export function shapeText(shape) {
  if (!Array.isArray(shape)) return '';
  return shape.length ? shape.join(' × ') : 'Scalar';
}

/** Words for one difference kind of /api/weights/compare. */
export const DIFFERENCE_WORDS = {
  same: 'Same name, shape and type',
  leftOnly: 'Only in the first file',
  rightOnly: 'Only in the second file',
  shape: 'Different shape',
  dtype: 'Different type',
  shapeAndDtype: 'Different shape and type',
};

/** The six comparison counts as sentences, in a fixed order; absent counts are left out. */
export function countSentences(counts) {
  const c = counts ?? {};
  const line = (n, one, many) => (Number.isFinite(n) ? `${fmt.plural(n, 'tensor')} ${n === 1 ? one : many}` : null);
  return [
    line(c.same, 'has the same name, shape and type in both files.', 'have the same name, shape and type in both files.'),
    line(c.leftOnly, 'is only in the first file.', 'are only in the first file.'),
    line(c.rightOnly, 'is only in the second file.', 'are only in the second file.'),
    line(c.shape, 'differs in shape only.', 'differ in shape only.'),
    line(c.dtype, 'differs in type only.', 'differ in type only.'),
    line(c.shapeAndDtype, 'differs in shape and type.', 'differ in shape and type.'),
  ].filter(Boolean);
}

const parentFolder = (path) => {
  const text = String(path ?? '');
  const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
  return cut > 0 ? text.slice(0, cut) : '';
};

const isAbort = (error) => error?.name === 'AbortError';

// ───────── paged table (filter, count, load more) ─────────

/**
 * @param {{label: string, filterHelp: string, headers: string[], numeric: boolean[], row: (item) => Node,
 *   fetch: ({filter, after, signal}) => Promise<object>, items: (result) => object[],
 *   status: (result, shown) => string, onFirst?: (result) => void, onError?: (message) => void}} options
 */
function pagedTable({ label, filterHelp, headers, numeric, row, fetch, items, status, onFirst = null }) {
  const filterId = uid('weights-filter');
  const helpId = uid('weights-filter-help');
  let controller = null;
  let next = null;
  let shown = 0;
  let last = null;

  const filterInput = h('input', {
    class: 'input weights__filter',
    id: filterId,
    type: 'search',
    autocomplete: 'off',
    spellcheck: 'false',
    'aria-describedby': helpId,
    oninput: () => refilter(),
    onkeydown: (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        refilter.flush();
      }
    },
  });
  const statusLine = h('p', { class: 'weights__status', role: 'status' });
  const errorBox = h('div', { class: 'weights__error', hidden: true });
  const tbody = h('tbody');
  const table = h(
    'table',
    { class: 'table weights__table' },
    h('caption', { class: 'visually-hidden' }, label),
    h('thead', null, h('tr', null, headers.map((text, i) => h('th', { scope: 'col', class: numeric[i] ? 'num' : null }, text)))),
    tbody,
  );
  const moreButton = h('button', { class: 'btn btn--sm', type: 'button', hidden: true, onclick: () => load(true) }, 'Load more');
  const el = h(
    'div',
    { class: 'weights__paged' },
    h('div', { class: 'field weights__filterfield' }, h('label', { class: 'field__label', for: filterId }, 'Name contains'), filterInput, h('p', { class: 'field__help', id: helpId }, filterHelp)),
    errorBox,
    h('div', { class: 'weights__tablewrap' }, table),
    h('div', { class: 'weights__more' }, statusLine, moreButton),
  );

  const refilter = debounce(() => load(false), FILTER_DELAY);

  async function load(append) {
    if (append && (next === null || next === undefined)) return; // never re-append the first page
    controller?.abort();
    const ctrl = new AbortController();
    controller = ctrl;
    moreButton.disabled = true;
    el.setAttribute('aria-busy', 'true');
    if (!append) statusLine.textContent = 'Loading…';
    try {
      const result = await fetch({ filter: filterInput.value || undefined, after: append ? next : undefined, signal: ctrl.signal });
      if (ctrl !== controller) return;
      errorBox.hidden = true;
      fill(errorBox);
      const list = items(result);
      if (!append) {
        fill(tbody);
        shown = 0;
        if (!last && onFirst) onFirst(result);
      }
      last = result;
      const frag = document.createDocumentFragment();
      for (const item of list) frag.appendChild(row(item));
      tbody.appendChild(frag);
      shown += list.length;
      next = result?.next ?? null;
      table.hidden = !shown;
      statusLine.textContent = status(result, shown);
      moreButton.hidden = next === null || next === undefined;
    } catch (error) {
      if (isAbort(error) || ctrl !== controller) return;
      statusLine.textContent = '';
      // The first answer failing means the file itself could not be read: the host says so at the top.
      if (!last && onFirst) onFirst(null, error);
      else {
        errorBox.hidden = false;
        fill(errorBox, notice('err', error.message));
      }
    } finally {
      if (ctrl === controller) {
        moreButton.disabled = false;
        el.removeAttribute('aria-busy');
      }
    }
  }

  return {
    el,
    load: () => load(false),
    destroy() {
      refilter.cancel();
      controller?.abort();
    },
  };
}

// ───────── dialog ─────────

/**
 * @param {{path: string, compareWith?: string|null}} options
 * @returns {Promise<void>}
 */
export function openWeightsDialog({ path, compareWith = null } = {}) {
  const parts = [];
  let closed = false;
  let compareTable = null;
  let otherPath = compareWith || null;

  // ── structure of this file ──
  const summaryBox = h('div', { class: 'weights__summary', 'aria-live': 'polite' }, loading('Reading the checkpoint header…'));
  const metaBox = h('div', { class: 'weights__meta' });

  const tensorTable = pagedTable({
    label: 'Tensors in this file',
    filterHelp: 'Not case-sensitive. The service filters the names.',
    headers: ['Name', 'Shape', 'Type', 'Stored values'],
    numeric: [false, false, false, true],
    fetch: ({ filter, after, signal }) => api.weights({ path, limit: PAGE, filter, after }, { signal }),
    items: (result) => (Array.isArray(result?.tensors) ? result.tensors : []),
    row: (t) =>
      h(
        'tr',
        null,
        h('td', { class: 'mono weights__name' }, t.name),
        h('td', { class: 'num weights__shape' }, shapeText(t.shape)),
        h('td', { class: 'mono' }, t.dtype ?? ''),
        h('td', { class: 'num' }, fmt.int(t.elements)),
      ),
    status: (result, shown) => {
      const matched = result?.matched;
      const total = result?.tensorCount;
      if (!Number.isFinite(matched)) return `${fmt.plural(shown, 'tensor')} shown.`;
      if (!matched) return 'No tensor names contain this text.';
      const of = Number.isFinite(total) && total !== matched ? ` (of ${fmt.int(total)} in the file)` : '';
      return `${fmt.int(shown)} of ${fmt.plural(matched, 'matching tensor')}${of} shown.`;
    },
    onFirst: (result, error) => renderSummary(result, error),
  });
  parts.push(tensorTable);

  function renderSummary(result, error) {
    if (!result) {
      fill(summaryBox, error ? notice('err', `The checkpoint could not be read: ${error.message}`) : h('p', { class: 'muted' }, 'Not reported'));
      return;
    }
    const fingerprint = typeof result.layoutSha256 === 'string' ? result.layoutSha256 : '';
    const dtypes = result.dtypeElements && typeof result.dtypeElements === 'object' ? Object.entries(result.dtypeElements) : [];
    const absent = () => h('span', { class: 'muted' }, 'Not reported');
    fill(
      summaryBox,
      h(
        'dl',
        { class: 'kv weights__kv' },
        h('dt', null, 'File'),
        h('dd', { class: 'mono weights__path' }, result.path ?? path),
        h('dt', null, 'Size'),
        h('dd', null, sizeText(result.bytes) || absent()),
        h('dt', null, 'Tensors'),
        h('dd', null, fmt.int(result.tensorCount) || absent()),
        h('dt', null, 'Stored values'),
        h(
          'dd',
          null,
          fmt.int(result.elements) || absent(),
          h('span', { class: 'weights__aside' }, 'Every number stored in the file. It may include buffers as well as trainable parameters; only the constructed model knows which are trainable.'),
        ),
        h('dt', null, 'Values per type'),
        h('dd', null, dtypes.length ? h('span', { class: 'weights__dtypes' }, dtypes.map(([type, n]) => h('span', { class: 'weights__dtype' }, h('span', { class: 'mono' }, type), ' ', fmt.int(n)))) : absent()),
        h('dt', null, 'Layout fingerprint'),
        h(
          'dd',
          { class: 'weights__fingerprint' },
          fingerprint
            ? [
                h('span', { class: 'mono', title: fingerprint }, `${fingerprint.slice(0, 16)}…`),
                h(
                  'button',
                  {
                    class: 'btn btn--sm btn--quiet',
                    type: 'button',
                    onclick: async () => toast((await copyText(fingerprint)) ? 'Copied the layout fingerprint.' : 'The fingerprint could not be copied.', { kind: 'info' }),
                  },
                  'Copy',
                ),
                h('span', { class: 'weights__aside' }, 'SHA-256 of the tensor names, shapes and types. Files with the same fingerprint have the same layout; their values may differ.'),
              ]
            : absent(),
        ),
      ),
    );
    const metadata = result.metadata && typeof result.metadata === 'object' ? Object.entries(result.metadata) : [];
    fill(
      metaBox,
      h('h3', { class: 'weights__heading' }, 'Metadata'),
      metadata.length
        ? h('dl', { class: 'kv weights__kv' }, metadata.map(([key, value]) => [h('dt', { class: 'mono' }, key), h('dd', { class: 'mono weights__metavalue' }, String(value))]))
        : h('p', { class: 'muted' }, 'The file has no metadata.'),
      result.metadataTruncated ? notice('', 'Some metadata was left out: the service returns at most 64 keys and 4,096 characters per value.') : null,
    );
  }

  const compareButton = h('button', { class: 'btn', type: 'button', onclick: () => chooseOther() }, 'Compare with another file');
  const structurePanel = h(
    'div',
    { class: 'weights__panel', role: 'tabpanel', 'aria-label': 'This file' },
    summaryBox,
    h('div', { class: 'row' }, compareButton),
    metaBox,
    h('h3', { class: 'weights__heading' }, 'Tensors'),
    tensorTable.el,
  );

  // ── comparison ──
  const comparePanel = h('div', { class: 'weights__panel', role: 'tabpanel', 'aria-label': 'Comparison', hidden: true });
  const compareHead = h('div', { class: 'weights__comparehead', 'aria-live': 'polite' });

  function showComparison(right) {
    otherPath = right;
    compareTable?.destroy();
    let first = null;
    compareTable = pagedTable({
      label: 'Tensors that differ',
      filterHelp: 'Not case-sensitive. Filters the list of differences; the counts above are for the whole files.',
      headers: ['Name', 'Difference', 'First file', 'Second file'],
      numeric: [false, false, false, false],
      fetch: ({ filter, after, signal }) => api.weightsCompare({ left: path, right, limit: PAGE, filter, after }, { signal }),
      items: (result) => (Array.isArray(result?.differences) ? result.differences : []),
      row: (d) => {
        const side = (entry) => (Array.isArray(entry) ? `${shapeText(entry[0])}, ${entry[1] ?? ''}` : 'Not in this file');
        return h(
          'tr',
          null,
          h('td', { class: 'mono weights__name' }, d.name),
          h('td', null, DIFFERENCE_WORDS[d.difference] ?? String(d.difference ?? '')),
          h('td', { class: Array.isArray(d.left) ? 'weights__shape' : 'muted' }, side(d.left)),
          h('td', { class: Array.isArray(d.right) ? 'weights__shape' : 'muted' }, side(d.right)),
        );
      },
      status: (result, shown) => (shown ? `${fmt.plural(shown, 'difference')} shown${result?.next ? '. There are more.' : '.'}` : first?.sameLayout ? 'No differences.' : 'No differing tensor names contain this text.'),
      onFirst: (result, error) => {
        first = result;
        renderCompareHead(result, error);
      },
    });
    fill(
      comparePanel,
      compareHead,
      h(
        'div',
        { class: 'row' },
        h('button', { class: 'btn', type: 'button', onclick: () => chooseOther() }, 'Compare with a different file'),
        h(
          'button',
          {
            class: 'btn btn--quiet',
            type: 'button',
            onclick: () => {
              tabStrip.setActive('file');
              showPanel('file');
            },
          },
          'Back to this file',
        ),
      ),
      h('h3', { class: 'weights__heading' }, 'Differences'),
      compareTable.el,
    );
    fill(compareHead, loading('Comparing the two headers…'));
    tabStrip.el.hidden = false;
    tabStrip.setActive('compare');
    showPanel('compare');
    compareTable.load();
  }

  function renderCompareHead(result, error) {
    if (!result) {
      fill(compareHead, error ? notice('err', `The two files could not be compared: ${error.message}`) : h('p', { class: 'muted' }, 'Not reported'));
      return;
    }
    const sentences = countSentences(result.counts);
    const left = result.left ?? path;
    const right = result.right ?? otherPath;
    // The values themselves are measured by a weight drift job, offered only for two .safetensors files.
    const drift = isSafetensors(left) && isSafetensors(right) ? h('button', { class: 'btn btn--sm', type: 'button', onclick: () => measureDrift(left, right) }, 'Measure how far the values moved…') : null;
    fill(
      compareHead,
      h('dl', { class: 'kv weights__kv' }, h('dt', null, 'First file'), h('dd', { class: 'mono weights__path' }, left), h('dt', null, 'Second file'), h('dd', { class: 'mono weights__path' }, right)),
      result.sameLayout === true
        ? notice('ok', SAME_LAYOUT, drift)
        : [
            result.sameLayout === false ? notice('warn', DIFFERENT_LAYOUT) : null,
            h('div', { class: 'weights__values' }, h('p', { class: 'weights__values-text' }, VALUES_NOT_COMPARED), drift),
          ],
      sentences.length ? h('ul', { class: 'weights__counts' }, sentences.map((text) => h('li', null, text))) : null,
    );
  }

  let measuring = false;
  async function measureDrift(left, right) {
    if (measuring) return;
    measuring = true;
    try {
      const { openAnalysisDialog } = await import('../runs/analysis.js');
      const job = await openAnalysisDialog({ kind: 'weight-drift', left, right, sourceLabel: 'the checkpoint structure comparison' });
      // A started job is opened by the analysis part; this dialog would cover it.
      if (job && !closed) closeWeights?.();
    } catch (error) {
      toast(`The analysis dialog could not be opened: ${error.message}`, { kind: 'err' });
    } finally {
      measuring = false;
    }
  }

  let choosing = false;
  async function chooseOther() {
    if (choosing) return;
    choosing = true;
    try {
      const { pickPath } = await import('./path-picker.js');
      const picked = await pickPath({
        title: 'Choose a checkpoint to compare with',
        mode: 'file',
        start: otherPath ? parentFolder(otherPath) : parentFolder(path),
        extensions: ['.safetensors'],
        confirmLabel: 'Compare',
      });
      if (!closed && picked) showComparison(picked);
    } catch (error) {
      toast(`The file picker could not be opened: ${error.message}`, { kind: 'err' });
    } finally {
      choosing = false;
    }
  }

  // ── frame ──
  const tabStrip = tabs({
    label: 'Checkpoint views',
    items: [
      { id: 'file', label: 'This file' },
      { id: 'compare', label: 'Comparison' },
    ],
    active: 'file',
    onChange: (id) => showPanel(id),
  });
  tabStrip.el.hidden = true;

  function showPanel(id) {
    structurePanel.hidden = id !== 'file';
    comparePanel.hidden = id !== 'compare';
  }

  const { closed: whenClosed, close: closeWeights } = openDialog({
    title: 'Checkpoint structure',
    wide: true,
    body: h('div', { class: 'weights' }, tabStrip.el, structurePanel, comparePanel),
    foot: (close) => [h('button', { class: 'btn', type: 'button', onclick: () => close() }, 'Close')],
  });

  tensorTable.load();
  if (compareWith) showComparison(compareWith);

  return whenClosed.then(() => {
    closed = true;
    for (const part of parts) part.destroy();
    compareTable?.destroy();
  });
}
