// Class results of one evaluation: the per-class table (support, predicted, correct, precision, recall,
// IoU as the service recorded them) and, for a chosen class, where its samples went. The same report
// serves a run's Validation tab (GET /api/run/classes, scores.js) and a completed evaluation job
// (GET /api/analysis/classes, holdout.js): the caller supplies the request. Rules in css/validation.css.
//
// mountClassReport(host, {evaluation, request, onClose?, headingText?, sort?, retry?, headingLevel?}) -> {destroy()}

import { h, mount, fmt, uid } from '../dom.js';
import { notice } from '../ui.js';

const CLASS_PAGE = 50;
const CLASS_COLUMNS = [
  ['name', 'Label', 'text'],
  ['id', 'Id', 'int'],
  ['support', 'Support', 'int'],
  ['predicted', 'Predicted', 'int'],
  ['correct', 'Correct', 'int'],
  ['precision', 'Precision', 'ratio'],
  ['recall', 'Recall', 'ratio'],
  ['iou', 'IoU', 'ratio'],
];

const reason = (error) => (error && typeof error.message === 'string' && error.message.trim()) || 'No reason was given.';
/** Fixed 4 decimals so the column aligns; '' for anything not finite. */
const decimals = (n) => (Number.isFinite(n) ? n.toFixed(4) : '');

/**
 * Mount the class results of one evaluation into `host` (appended) and move focus to its heading.
 * @param {HTMLElement} host
 * @param {object} options
 * @param {{id: number, dataset?: string, step?: number, task?: string, unit?: string}} options.evaluation
 *   the validation row whose classes are shown
 * @param {(params: object, opts?: object) => Promise<object>} options.request
 *   answers like GET /api/run/classes or GET /api/analysis/classes for {evaluation, limit, offset, target?}
 * @param {(() => void) | null} [options.onClose] when given, a "Hide classes" button beside the heading calls it
 * @param {string | null} [options.headingText] the heading; left out, "Class results: <dataset>, iteration <step>"
 * @param {{key: string, dir: 'ascending' | 'descending'} | null} [options.sort]
 *   a sort order the caller keeps between reports; the report reads it and writes the person's choice back
 * @param {boolean} [options.retry] offer "Try again" when the classes could not be read
 * @param {number} [options.headingLevel] 3 gives an h3 with an h4 under it; pass the level that fits the host
 * @returns {{destroy: () => void}}
 */
export function mountClassReport(host, { evaluation, request, onClose = null, headingText = null, sort = null, retry = false, headingLevel = 3 } = {}) {
  const row = evaluation;
  const order = sort ?? { key: 'id', dir: 'ascending' };
  const cl = { offset: 0, report: null, error: '', target: null, predictions: [], predError: '', predNext: null, predLoading: false, gen: 0 };
  let classAbort = null;
  let destroyed = false;

  const level = Number.isInteger(headingLevel) && headingLevel >= 2 && headingLevel <= 5 ? headingLevel : 3;
  const headingId = uid('scores-classes');
  const classHeading = h(`h${level}`, { class: 'val__heading', id: headingId, tabindex: '-1' });
  const classBody = h('div', { class: 'val__classbody', role: 'status' });
  const predEl = h('div', { class: 'val__predictions' });
  const head =
    typeof onClose === 'function'
      ? h('div', { class: 'val__classhead' }, classHeading, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => onClose() }, 'Hide classes'))
      : classHeading;

  async function loadClasses(focusHeading = false) {
    classAbort?.abort();
    const ctrl = new AbortController();
    classAbort = ctrl;
    cl.gen += 1;
    const gen = cl.gen;
    cl.report = null;
    cl.error = '';
    renderClasses(focusHeading);
    try {
      const report = await request({ evaluation: row.id, limit: CLASS_PAGE, offset: cl.offset }, { signal: ctrl.signal });
      if (gen !== cl.gen || destroyed) return;
      cl.report = report;
    } catch (error) {
      if (error?.name === 'AbortError' || gen !== cl.gen || destroyed) return;
      cl.error = reason(error);
    }
    renderClasses(false);
  }

  async function loadTarget(label, more = false) {
    const gen = cl.gen;
    if (!more) {
      cl.target = label;
      cl.predictions = [];
      cl.predNext = 0;
    }
    cl.predError = '';
    cl.predLoading = true;
    renderPredictions();
    try {
      const report = await request({ evaluation: row.id, limit: CLASS_PAGE, offset: cl.predNext ?? 0, target: label.id });
      if (gen !== cl.gen || destroyed || cl.target !== label) return;
      cl.predictions = [...cl.predictions, ...(Array.isArray(report?.predictions) ? report.predictions : [])];
      cl.predNext = Number.isInteger(report?.nextPrediction) ? report.nextPrediction : null;
    } catch (error) {
      if (gen !== cl.gen || destroyed) return;
      cl.predError = reason(error);
    }
    cl.predLoading = false;
    renderPredictions();
  }

  function renderClasses(focusHeading) {
    if (destroyed) return;
    if (!classBody.parentNode) host.append(head, classBody, predEl);
    classHeading.textContent = headingText ?? `Class results: ${row.dataset ?? 'dataset not reported'}, iteration ${fmt.int(row.step)}`;
    if (focusHeading) classHeading.focus();
    const r = cl.report;
    // Read again from the heading, which takes focus: the button that asked is replaced by the answer.
    if (cl.error) mount(classBody, notice('err', cl.error, retry ? h('button', { class: 'btn btn--sm', type: 'button', onclick: () => loadClasses(true) }, 'Try again') : null));
    else if (!r) mount(classBody, h('p', { class: 'validation__status' }, 'Loading class results…'));
    else if (!r.recorded) mount(classBody, h('p', null, 'This evaluation recorded no class counts.'));
    else mount(classBody, classTable(r));
    renderPredictions();
  }

  function cellFor(kind, value, title) {
    if (value === null || value === undefined) return h('td', { class: kind === 'text' ? '' : 'num muted', title: title || null }, 'Not defined');
    if (kind === 'text') return h('td', null, String(value));
    if (kind === 'ratio') return h('td', { class: 'num', title: String(value) }, decimals(value));
    return h('td', { class: 'num' }, fmt.int(value));
  }

  function classTable(r) {
    const labels = Array.isArray(r.labels) ? r.labels : [];
    const cols = CLASS_COLUMNS.filter(([key]) => labels.some((l) => key in l));
    const { key: sortKey, dir } = order;
    const sorted = [...labels].sort((a, b) => {
      const x = a[sortKey];
      const y = b[sortKey];
      const rank = x === null || x === undefined ? 1 : y === null || y === undefined ? -1 : typeof x === 'string' ? x.localeCompare(y) : x - y;
      return dir === 'ascending' ? rank : -rank;
    });
    // The backend's unit word ("annotated_pixel", "image"), written as words and counted.
    const word = typeof r.unit === 'string' && r.unit ? r.unit.replace(/_/g, ' ') : '';
    const unit = word ? (r.total === 1 || word.endsWith('s') ? word : `${word}s`) : '(unit not reported)';
    const from = cl.offset + 1;
    const to = cl.offset + labels.length;
    // First the counts, then (muted) how to read the table. Precision, recall and IoU come from the
    // service, so they are "as recorded"; only the browser's own share column is called derived.
    const capText = [
      h('span', { class: 'val__capline' }, `${fmt.plural(r.labelCount, 'class', 'classes')}, ${fmt.int(r.total)} ${unit} in total, ${fmt.int(r.correct)} correct.`),
      h(
        'span',
        { class: 'val__capnote' },
        `Showing classes ${fmt.int(from)} to ${fmt.int(to)}. Sorting applies to this page. Precision, recall and IoU as recorded; "Not defined" where the denominator was zero.`,
      ),
    ];
    const headRow = h(
      'tr',
      null,
      cols.map(([key, label, kind]) =>
        h(
          'th',
          { scope: 'col', class: kind === 'text' ? '' : 'num', 'aria-sort': key === sortKey ? dir : 'none' },
          h('button', { class: 'val__sort', type: 'button', onclick: () => sortBy(key) }, label, key === sortKey ? h('span', { 'aria-hidden': 'true' }, dir === 'ascending' ? ' ▲' : ' ▼') : null),
        ),
      ),
    );
    const body = sorted.map((l) =>
      h(
        'tr',
        { 'aria-current': cl.target && cl.target.id === l.id ? 'true' : null },
        cols.map(([key, , kind]) =>
          key === 'name'
            ? h('td', null, h('button', { class: 'btn btn--sm btn--quiet val__classbtn', type: 'button', onclick: () => loadTarget(l) }, l.name ?? `Class ${l.id}`))
            : cellFor(kind, l[key], 'The denominator was zero for this class.'),
        ),
      ),
    );
    const pager = h(
      'div',
      { class: 'val__actions' },
      h('button', { class: 'btn btn--sm', type: 'button', disabled: cl.offset === 0, onclick: () => page(-1) }, 'Previous classes'),
      h('button', { class: 'btn btn--sm', type: 'button', disabled: r.next === null || r.next === undefined, onclick: () => page(1) }, 'Next classes'),
    );
    return [h('table', { class: 'table val__table' }, h('caption', { class: 'val__caption' }, capText), h('thead', null, headRow), h('tbody', null, body)), pager];
  }

  function sortBy(key) {
    order.dir = order.key === key && order.dir === 'ascending' ? 'descending' : 'ascending';
    order.key = key;
    renderClasses(false);
    classBody.querySelector?.('th[aria-sort="ascending"] button, th[aria-sort="descending"] button')?.focus();
  }

  function page(dir) {
    cl.offset = Math.max(0, cl.offset + dir * CLASS_PAGE);
    loadClasses(false);
  }

  function renderPredictions() {
    const label = cl.target;
    if (!label) {
      predEl.replaceChildren();
      return;
    }
    const support = Number.isFinite(label.support) && label.support > 0 ? label.support : null;
    const rows = cl.predictions.map((p) => {
      const share = support && Number.isFinite(p.count) ? p.count / support : null;
      return h(
        'tr',
        null,
        h('td', null, p.name ?? `Class ${p.prediction}`, p.prediction === label.id ? h('span', { class: 'val__best' }, 'correct') : null),
        h('td', { class: 'num' }, fmt.int(p.prediction)),
        h('td', { class: 'num' }, fmt.int(p.count)),
        h(
          'td',
          { class: 'num' },
          share === null
            ? h('span', { class: 'muted' }, 'Not defined')
            : h('span', { class: 'val__share' }, h('span', { class: 'val__sharebar', 'aria-hidden': 'true' }, h('span', { class: 'val__sharefill', style: { width: `${Math.min(100, share * 100)}%` } })), `${(share * 100).toFixed(1)}%`),
        ),
      );
    });
    mount(
      predEl,
      h(`h${level + 1}`, { class: 'val__subheading' }, `Where ${label.name ?? `class ${label.id}`} went`),
      cl.predError ? notice('err', cl.predError) : null,
      rows.length
        ? h(
            'table',
            { class: 'table val__table' },
            h('caption', { class: 'val__caption' }, `Predictions inside this class, most first. Share is derived: count divided by the class's support (${fmt.int(label.support)}).`),
            h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Predicted as'), h('th', { scope: 'col', class: 'num' }, 'Id'), h('th', { scope: 'col', class: 'num' }, 'Count'), h('th', { scope: 'col', class: 'num' }, 'Share'))),
            h('tbody', null, rows),
          )
        : !cl.predLoading && !cl.predError
          ? h('p', null, 'No predictions were recorded for this class.')
          : null,
      h('p', { class: 'validation__status', role: 'status' }, cl.predLoading ? 'Loading predictions…' : ''),
      cl.predNext !== null && !cl.predLoading ? h('button', { class: 'btn btn--sm', type: 'button', onclick: () => loadTarget(label, true) }, 'Load more predictions') : null,
    );
  }

  loadClasses(true);

  return {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      classAbort?.abort();
      cl.gen += 1;
      head.remove();
      classBody.remove();
      predEl.remove();
    },
  };
}
