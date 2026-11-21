// Scores view of the Validation tab: the numbers each validation pass measured (GET /api/run/validation)
// and, for runs that recorded them, per-class tables (GET /api/run/classes, drawn by classes.js). Loaded
// by validation.js with a dynamic import. Shows only what the API returned; derived values are labelled
// as derived.
//
// A run can hold scores in three places: the recorded summary (this view), the averages the trainer
// printed in its log (GET /api/run/checkpoint-selection reads them) and a score per image
// (GET /api/run/evaluations, what the compare screen uses). When the summary is empty the view says so
// and shows the other two, each named by where it comes from, never as if they were the summary.
//
// mountScores(host, {run, galleryHas(step), onShowImages(step)}) -> {setRun(run), galleryChanged(), destroy()}

import { h, mount, fmt, uid } from '../dom.js';
import { api } from '../api.js';
import { notice, emptyState } from '../ui.js';
import { runStatus } from '../state/runs.js';
import { go, compareHref } from '../actions.js';
import { createChart } from '../components/chart.js';
import { mountClassReport } from './classes.js';

const PAGE = 100;
const REFRESH_MS = 10000;
/** Panel width (CSS px) from which two metric charts sit side by side. */
const TWO_COLUMNS = 1100;
/** Plot height divided by column width. */
const ASPECT = 0.32;
const HIGHER = new Set(['psnr', 'ssim', 'miou', 'accuracy', 'top1', 'top5', 'precision', 'recall', 'f1']);
const LOWER = new Set(['lpips', 'dists', 'loss']);

const reason = (error) => (error && typeof error.message === 'string' && error.message.trim()) || 'No reason was given.';
/** "higher" | "lower" | null: only where the direction is certain. */
export function direction(metric) {
  const key = String(metric).toLowerCase();
  if (HIGHER.has(key)) return 'higher';
  if (LOWER.has(key)) return 'lower';
  return null;
}
/** Fixed 4 decimals so the column aligns; '' for anything not finite. */
export const decimals = (n) => (Number.isFinite(n) ? n.toFixed(4) : '');
/** Words for a nonfinite marker from the API ("nan", "inf", "-inf"); unknown words are shown as they are. */
export function nonfiniteText(value) {
  const key = String(value ?? '').trim().toLowerCase();
  if (key === 'nan') return 'Not a number';
  if (key === 'inf' || key === '+inf' || key === 'infinity') return 'Infinite';
  if (key === '-inf' || key === '-infinity') return 'Negative infinite';
  return key ? String(value) : 'Not a number';
}
const shortAttempt = (a) => (typeof a === 'string' ? a.slice(0, 8) : '');
/** Series read for the fallback table; a run rarely logs more than a few metrics per dataset. */
const MAX_OTHER_SERIES = 8;
/** Where a ranked point's score was read from (studio/checkpoint_selection.py `source`). */
const OTHER_SOURCE = { log: 'Training log', sampleMean: 'Mean of the scores per image', summary: 'Recorded summary' };
const wordList = (items) => (items.length <= 1 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`);

export function mountScores(host, { run, galleryHas = () => false, onShowImages = () => {} } = {}) {
  let current = run ?? null;
  let destroyed = false;
  const st = { phase: 'idle', error: '', rows: [], ids: new Set(), next: null, more: false, moreError: '', dataset: null };
  let listAbort = null;
  let timer = 0;
  let charts = [];
  let chartKey = '';
  const showButtons = new Map(); // evaluation id -> button
  // What the run holds when no summary was recorded: logged averages and scores per image.
  const other = { id: null, phase: 'idle', error: '', series: [], cut: 0, perImage: null };
  let otherAbort = null;

  // ── layout ──
  const datasetId = uid('scores-dataset');
  const datasetSelect = h('select', { class: 'select val__select', id: datasetId, onchange: () => setDataset(datasetSelect.value) });
  const datasetField = h('div', { class: 'validation__field', hidden: true }, h('label', { class: 'validation__label', for: datasetId }, 'Validation dataset'), datasetSelect);
  const statusEl = h('p', { class: 'validation__status', role: 'status' });
  const errorEl = h('div', { hidden: true });
  const emptyEl = h('div', { hidden: true });
  const chartsEl = h('div', { class: 'val__charts' });
  const thead = h('thead');
  const tbody = h('tbody');
  const caption = h('caption', { class: 'val__caption' });
  const table = h('table', { class: 'table val__table' }, caption, thead, tbody);
  const tableWrap = h('div', { class: 'val__tablewrap', hidden: true }, table);
  const moreButton = h('button', { class: 'btn', type: 'button', onclick: () => loadEarlier() }, 'Load earlier');
  const moreError = h('p', { class: 'validation__moreerror', role: 'status' });
  const moreEl = h('div', { class: 'validation__more', hidden: true }, moreButton, moreError);
  const classesEl = h('section', { class: 'val__classes', hidden: true, 'aria-label': 'Class results' });
  const root = h('div', { class: 'val__scores' }, h('div', { class: 'validation__bar' }, datasetField), statusEl, errorEl, emptyEl, chartsEl, tableWrap, moreEl, classesEl);
  host.appendChild(root);

  // ── loading ──
  function accept(page, { prepend = false } = {}) {
    const list = Array.isArray(page?.evaluations) ? page.evaluations.filter((r) => r && Number.isInteger(r.id)) : [];
    const fresh = list.filter((r) => !st.ids.has(r.id));
    for (const r of fresh) st.ids.add(r.id);
    st.rows = prepend ? [...fresh, ...st.rows] : [...st.rows, ...fresh];
    st.rows.sort((x, y) => y.id - x.id);
    return fresh.length;
  }

  async function load() {
    if (!current) return;
    listAbort?.abort();
    const ctrl = new AbortController();
    listAbort = ctrl;
    st.phase = 'loading';
    st.error = '';
    renderStatus();
    try {
      const page = await api.validation(current.id, { limit: PAGE }, { signal: ctrl.signal });
      if (destroyed || listAbort !== ctrl) return;
      st.rows = [];
      st.ids = new Set();
      accept(page);
      st.next = Number.isInteger(page?.next) ? page.next : null;
      st.phase = 'ready';
    } catch (error) {
      if (error?.name === 'AbortError' || destroyed || listAbort !== ctrl) return;
      st.phase = 'error';
      st.error = reason(error);
    }
    renderAll();
    schedule();
  }

  /** While the run trains: new rows every 10 s, added on top; nothing focused is rebuilt. */
  async function poll() {
    timer = 0;
    if (destroyed || !current || !runStatus(current.status).active || st.phase !== 'ready') return;
    try {
      const page = await api.validation(current.id, { limit: 50 });
      if (destroyed) return;
      if (accept(page, { prepend: true })) renderAll();
      else if (!st.rows.length) loadOther({ quiet: true });
    } catch {
      // A missed refresh is retried at the next tick; the rows shown stay as they are.
    }
    schedule();
  }

  function schedule() {
    clearTimeout(timer);
    timer = 0;
    if (!destroyed && current && runStatus(current.status).active) timer = setTimeout(poll, REFRESH_MS);
  }

  async function loadEarlier() {
    if (st.next === null || st.more || !current) return;
    st.more = true;
    st.moreError = '';
    renderMore();
    try {
      const page = await api.validation(current.id, { limit: PAGE, before: st.next });
      if (destroyed) return;
      accept(page);
      st.next = Number.isInteger(page?.next) ? page.next : null;
      st.more = false;
      renderAll();
    } catch (error) {
      if (destroyed) return;
      st.more = false;
      st.moreError = `Earlier results could not be loaded: ${reason(error)}`;
      renderMore();
    }
  }

  // ── rendering ──
  const datasets = () => [...new Set(st.rows.map((r) => r.dataset).filter((d) => typeof d === 'string'))];
  const shownRows = () => st.rows.filter((r) => r.dataset === st.dataset);
  const metricsOf = (rows) => [...new Set(rows.flatMap((r) => [...Object.keys(r.metrics ?? {}), ...Object.keys(r.nonfinite ?? {})]))].sort();

  function setDataset(name) {
    st.dataset = name;
    closeClasses();
    renderAll();
  }

  function renderAll() {
    const names = datasets();
    if (!names.includes(st.dataset)) st.dataset = st.rows[0]?.dataset ?? null;
    const optKey = JSON.stringify([names, st.dataset]);
    if (datasetSelect.dataset.key !== optKey) {
      datasetSelect.dataset.key = optKey;
      mount(datasetSelect, names.map((n) => h('option', { value: n }, n)));
      if (st.dataset !== null) datasetSelect.value = st.dataset;
    }
    datasetField.hidden = names.length < 2;
    renderStatus();
    renderCharts();
    renderTable();
    renderMore();
  }

  function renderStatus() {
    const rows = shownRows();
    let text = '';
    if (st.phase === 'loading') text = 'Loading validation results…';
    else if (st.phase === 'ready' && rows.length) {
      text = `${fmt.plural(rows.length, 'measured validation')}${typeof st.dataset === 'string' ? ` of ${st.dataset}` : ''}${st.next !== null ? ', earlier ones are available' : ''}.`;
    }
    if (statusEl.textContent !== text) statusEl.textContent = text;
    statusEl.hidden = !text;
    const errKey = st.phase === 'error' ? st.error : '';
    if (errorEl.dataset.key !== errKey) {
      errorEl.dataset.key = errKey;
      mount(errorEl, errKey ? notice('err', `The validation results could not be loaded: ${errKey}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => load() }, 'Try again')) : null);
    }
    errorEl.hidden = !errKey;
    const empty = st.phase === 'ready' && !st.rows.length;
    emptyEl.hidden = !empty;
    if (!empty) {
      if (emptyEl.dataset.key) {
        emptyEl.dataset.key = '';
        mount(emptyEl);
      }
      return;
    }
    if (current && other.id !== current.id) loadOther();
    else renderOther();
  }

  // ── no recorded summary: the scores the run has elsewhere ──

  async function loadOther({ quiet = false } = {}) {
    if (!current) return;
    const id = current.id;
    otherAbort?.abort();
    const ctrl = new AbortController();
    otherAbort = ctrl;
    if (!quiet || other.id !== id) {
      Object.assign(other, { id, phase: 'loading', error: '', series: [], cut: 0, perImage: null });
      renderOther();
    }
    // One part failing does not hide the other: each is asked for on its own.
    const alone = (promise) =>
      promise.catch((error) => {
        if (error?.name === 'AbortError') throw error;
        return null;
      });
    try {
      const [selection, perImage] = await Promise.all([api.checkpointSelection(id, {}, { signal: ctrl.signal }), alone(api.evaluations(id, { signal: ctrl.signal }))]);
      const listed = (Array.isArray(selection?.series) ? selection.series : []).filter((row) => row && typeof row.dataset === 'string' && typeof row.metric === 'string');
      const asked = listed.slice(0, MAX_OTHER_SERIES);
      const answers = await Promise.all(asked.map((row) => alone(api.checkpointSelection(id, { dataset: row.dataset, metric: row.metric }, { signal: ctrl.signal }))));
      if (destroyed || otherAbort !== ctrl) return;
      Object.assign(other, {
        phase: 'ready',
        error: '',
        series: asked.map((row, i) => ({ dataset: row.dataset, metric: row.metric, points: Array.isArray(answers[i]?.points) ? answers[i].points.filter((point) => point && Number.isFinite(point.step)) : [] })),
        cut: listed.length - asked.length,
        perImage: Array.isArray(perImage) ? perImage.filter((row) => row && typeof row.dataset === 'string') : null,
      });
    } catch (error) {
      if (error?.name === 'AbortError' || destroyed || otherAbort !== ctrl) return;
      // A background refresh that fails leaves what is shown as it is.
      if (quiet && other.phase === 'ready') return;
      Object.assign(other, { phase: 'error', error: reason(error) });
    }
    renderOther();
  }

  /** The averages of the ranking, one row per dataset, attempt and iteration. */
  function loggedTable() {
    const rows = new Map();
    const metrics = new Set();
    const sources = new Set();
    for (const series of other.series) {
      for (const point of series.points) {
        const key = JSON.stringify([series.dataset, point.attempt ?? '', point.step]);
        if (!rows.has(key)) rows.set(key, { dataset: series.dataset, attempt: point.attempt ?? '', step: point.step, cells: {}, sources: new Set() });
        const row = rows.get(key);
        row.cells[series.metric] = point;
        row.sources.add(point.source ?? '');
        metrics.add(series.metric);
        sources.add(point.source ?? '');
      }
    }
    if (!rows.size) return null;
    const list = [...rows.values()].sort((x, y) => x.dataset.localeCompare(y.dataset) || y.step - x.step || x.attempt.localeCompare(y.attempt));
    const names = [...metrics].sort();
    const severalDatasets = new Set(list.map((row) => row.dataset)).size > 1;
    const severalAttempts = new Set(list.map((row) => row.attempt)).size > 1;
    const onlyLog = sources.size === 1 && sources.has('log');
    const sourceWords = (set) => [...set].map((code) => OTHER_SOURCE[code] ?? (code ? String(code) : 'Not reported')).join(', ');
    const table = h(
      'table',
      { class: 'table val__table' },
      h(
        'caption',
        { class: 'val__caption' },
        onlyLog
          ? 'One average per validation pass, over all its images, as the trainer printed it in its log. Newest first.'
          : 'One average per validation pass, over all its images. The last column says where each was read from. Newest first.',
      ),
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          severalDatasets ? h('th', { scope: 'col' }, 'Dataset') : null,
          h('th', { scope: 'col', class: 'num' }, 'Iteration'),
          severalAttempts ? h('th', { scope: 'col' }, 'Attempt') : null,
          names.map((name) => h('th', { scope: 'col', class: 'num' }, name)),
          onlyLog ? null : h('th', { scope: 'col' }, 'Read from'),
        ),
      ),
      h(
        'tbody',
        null,
        list.map((row) =>
          h(
            'tr',
            null,
            severalDatasets ? h('td', null, row.dataset) : null,
            h('td', { class: 'num' }, fmt.int(row.step)),
            severalAttempts ? h('td', { class: 'mono', title: row.attempt }, shortAttempt(row.attempt) || 'Not reported') : null,
            names.map((name) => {
              const point = row.cells[name];
              if (!point) return h('td', { class: 'num muted' }, 'Not reported');
              if (point.nonfinite !== null && point.nonfinite !== undefined) return h('td', { class: 'num val__nonfinite', title: String(point.nonfinite) }, nonfiniteText(point.nonfinite));
              return Number.isFinite(point.value) ? h('td', { class: 'num', title: String(point.value) }, decimals(point.value)) : h('td', { class: 'num muted' }, 'Not reported');
            }),
            onlyLog ? null : h('td', null, sourceWords(row.sources)),
          ),
        ),
      ),
    );
    return { table, onlyLog, dataset: severalDatasets ? '' : list[0].dataset };
  }

  /** "3 images at iterations 1 and 2 (psnr, ssim)", one sentence per dataset. */
  function perImageLines() {
    const sets = new Map();
    for (const row of other.perImage ?? []) {
      if (!sets.has(row.dataset)) sets.set(row.dataset, { steps: new Set(), metrics: new Set(), counts: new Set() });
      const set = sets.get(row.dataset);
      if (Number.isFinite(row.step)) set.steps.add(row.step);
      if (typeof row.metric === 'string') set.metrics.add(row.metric);
      if (Number.isFinite(row.sampleCount)) set.counts.add(row.sampleCount);
    }
    return [...sets].map(([dataset, set]) => {
      const steps = [...set.steps].sort((x, y) => x - y);
      const counts = [...set.counts].sort((x, y) => x - y);
      const images = !counts.length ? 'Scores per image' : counts.length === 1 ? fmt.plural(counts[0], 'image') : `${fmt.int(counts[0])} to ${fmt.int(counts[counts.length - 1])} images`;
      const where = !steps.length ? '' : steps.length <= 6 ? ` at ${steps.length === 1 ? 'iteration' : 'iterations'} ${wordList(steps.map((n) => fmt.int(n)))}` : ` at ${fmt.int(steps.length)} iterations, ${fmt.int(steps[0])} to ${fmt.int(steps[steps.length - 1])}`;
      const metrics = set.metrics.size ? ` (${[...set.metrics].sort().join(', ')})` : '';
      return `${dataset}: ${images}${where}${metrics}.`;
    });
  }

  function renderOther() {
    if (emptyEl.hidden) return;
    const logged = other.phase === 'ready' ? loggedTable() : null;
    const lines = other.phase === 'ready' ? perImageLines() : [];
    const key = JSON.stringify([other.id, other.phase, other.error, other.series, other.cut, other.perImage]);
    if (emptyEl.dataset.key === key) return;
    // Rebuilt only when what was read changes; the button that had focus gets it back.
    const focused = emptyEl.contains(document.activeElement) ? document.activeElement.dataset?.focus : null;
    emptyEl.dataset.key = key;
    if (other.phase === 'loading' || other.phase === 'idle') {
      mount(emptyEl, h('p', { class: 'validation__status', role: 'status' }, 'No validation summary was recorded for this run. Looking for its other scores…'));
      return;
    }
    if (other.phase === 'error') {
      mount(
        emptyEl,
        notice(
          'err',
          `No validation summary was recorded for this run, and its other scores could not be read: ${other.error}`,
          h('button', { class: 'btn btn--sm', type: 'button', dataset: { focus: 'other-retry' }, onclick: () => loadOther() }, 'Try again'),
        ),
      );
      if (focused) emptyEl.querySelector(`[data-focus="${focused}"]`)?.focus();
      return;
    }
    if (!logged && !lines.length) {
      mount(
        emptyEl,
        emptyState({
          title: 'This run has no validation scores',
          text:
            other.perImage === null
              ? 'No validation summary was recorded and its training log holds no validation score. Whether scores per image were kept could not be read.'
              : 'No validation summary was recorded, its training log holds no validation score and no score per image was kept.',
        }),
      );
      return;
    }
    mount(
      emptyEl,
      h(
        'div',
        { class: 'val__other' },
        h('p', null, 'No validation summary was recorded for this run; that is the record this view normally shows. This is what the run has instead.'),
        h('h3', { class: 'val__othertitle' }, logged && !logged.onlyLog ? 'Averages per validation pass' : 'Averages from the training log', logged?.dataset ? h('span', { class: 'muted' }, ` (${logged.dataset})`) : null),
        logged
          ? [
              h('div', { class: 'val__tablewrap' }, logged.table),
              other.cut > 0 ? h('p', { class: 'val__othernote' }, `${fmt.plural(other.cut, 'more metric')} of this run ${other.cut === 1 ? 'is' : 'are'} not listed here. The Files tab ranks every one of them.`) : null,
            ]
          : h('p', { class: 'muted' }, "The run's training log holds no validation score."),
        h('h3', { class: 'val__othertitle' }, 'Scores per image'),
        lines.length
          ? [
              h('p', null, 'The run also kept the score of every validation image:'),
              h('ul', { class: 'val__otherlist' }, lines.map((line) => h('li', null, line))),
              h(
                'div',
                { class: 'row row--wrap' },
                h('button', { class: 'btn', type: 'button', dataset: { focus: 'other-compare' }, onclick: () => current && go(compareHref(current.id)) }, 'Compare with…'),
                h('span', { class: 'val__othernote' }, 'The compare screen sets them against another run or evaluation, image by image.'),
              ),
            ]
          : h(
              'p',
              { class: 'muted' },
              other.perImage === null
                ? 'Whether the run kept a score per image could not be read.'
                : ['No score per image was kept. A run whose recipe turns on ', h('span', { class: 'mono' }, 'val.record_samples'), ' keeps them and can then be compared image by image.'],
            ),
        h('p', { class: 'val__othernote' }, 'Results by class come only with the recorded summary. The trainer records one for runs started from this app and for recipes that keep scores per image; a run made by an older version of the trainer can lack it.'),
      ),
    );
    if (focused) emptyEl.querySelector(`[data-focus="${focused}"]`)?.focus();
  }

  function seriesFor(rows, metric) {
    // One series per attempt: a resume starts a separate line, never joined to the one before.
    const byAttempt = new Map();
    for (const r of [...rows].sort((x, y) => x.step - y.step || x.id - y.id)) {
      if (!Number.isFinite(r.step)) continue;
      const key = r.attempt ?? '';
      if (!byAttempt.has(key)) byAttempt.set(key, []);
      const v = r.metrics?.[metric];
      byAttempt.get(key).push({ x: r.step, y: Number.isFinite(v) && !(metric in (r.nonfinite ?? {})) ? v : null });
    }
    const many = byAttempt.size > 1;
    return [...byAttempt].map(([attempt, points]) => ({ name: many ? `Attempt ${shortAttempt(attempt) || 'not reported'}` : metric, points }));
  }

  // Chart layout: one per row, two side by side once the panel is at least TWO_COLUMNS wide; the
  // plot height follows the column width (a fixed aspect), in steps so a window drag rarely rebuilds.
  let chartHeight = 150;
  let chartCols = 1;
  const chartObserver = new ResizeObserver(() => fitCharts());
  chartObserver.observe(chartsEl);

  function fitCharts() {
    const width = chartsEl.clientWidth;
    if (!width) return;
    const cols = width >= TWO_COLUMNS ? 2 : 1;
    const gap = parseFloat(getComputedStyle(chartsEl).columnGap) || 0;
    const column = cols === 2 ? (width - gap) / 2 : width;
    const height = Math.max(140, Math.min(320, Math.round((column * ASPECT) / 20) * 20));
    if (cols === chartCols && height === chartHeight) return;
    chartCols = cols;
    chartHeight = height;
    chartsEl.dataset.cols = String(cols);
    chartKey = ''; // createChart takes its height once: rebuild at the new size
    renderCharts();
  }

  function renderCharts() {
    const rows = shownRows();
    const metrics = metricsOf(rows);
    const key = JSON.stringify([st.dataset, metrics, chartHeight]);
    if (key !== chartKey) {
      chartKey = key;
      for (const c of charts) c.chart.destroy();
      chartsEl.replaceChildren();
      charts = metrics.map((metric) => {
        const box = h('figure', { class: 'val__chart' }, h('figcaption', { class: 'val__charttitle' }, metric, direction(metric) ? h('span', { class: 'muted' }, ` (${direction(metric)} is better)`) : null));
        chartsEl.appendChild(box);
        // A validation metric is a handful of points: no log scale, no smoothing.
        return { metric, chart: createChart(box, { title: metric, height: chartHeight, scaleToggle: false, smoothing: false }) };
      });
    }
    for (const { metric, chart } of charts) chart.update({ series: seriesFor(rows, metric), xLabel: 'Iteration' });
    chartsEl.hidden = !charts.length;
  }

  function bestIds(rows, metric) {
    const dir = direction(metric);
    if (!dir) return new Set();
    let best = null;
    for (const r of rows) {
      const v = r.metrics?.[metric];
      if (!Number.isFinite(v) || metric in (r.nonfinite ?? {})) continue;
      if (best === null || (dir === 'higher' ? v > best : v < best)) best = v;
    }
    return new Set(rows.filter((r) => r.metrics?.[metric] === best && best !== null).map((r) => r.id));
  }

  function renderTable() {
    const rows = shownRows();
    tableWrap.hidden = !rows.length;
    if (!rows.length) return;
    const metrics = metricsOf(rows);
    const key = JSON.stringify([st.dataset, metrics, rows.map((r) => r.id)]);
    if (table.dataset.key === key) return;
    // Rebuilt only when the rows change (a new validation or Load earlier), never on an idle poll;
    // the button that had focus gets it back.
    const focusedId = tbody.contains(document.activeElement) ? document.activeElement.dataset?.focus : null;
    table.dataset.key = key;
    caption.textContent = `Newest first. "Best" marks the best value among the rows shown, only for metrics whose direction is known.`;
    // The attempt is a column only where the rows shown come from more than one (a resume or a retry);
    // for a run trained in one go it would repeat one id down the table.
    const severalAttempts = new Set(rows.map((r) => r.attempt ?? '')).size > 1;
    mount(
      thead,
      h('tr', null, h('th', { scope: 'col', class: 'num' }, 'Iteration'), severalAttempts ? h('th', { scope: 'col' }, 'Attempt') : null, metrics.map((m) => h('th', { scope: 'col', class: 'num' }, m)), h('th', { scope: 'col' }, h('span', { class: 'visually-hidden' }, 'Actions'))),
    );
    const best = new Map(metrics.map((m) => [m, bestIds(rows, m)]));
    showButtons.clear();
    mount(
      tbody,
      rows.map((r) => {
        // Every metric cell ends with the same fixed-width tag slot (empty unless "best"), so the
        // digits of a column end at the same x in every row.
        const slot = (tag) => h('span', { class: 'val__tagslot' }, tag ? h('span', { class: 'val__tag' }, tag) : null);
        const cells = metrics.map((m) => {
          if (r.nonfinite && m in r.nonfinite) return h('td', { class: 'num val__nonfinite', title: String(r.nonfinite[m]) }, nonfiniteText(r.nonfinite[m]), slot(''));
          const v = r.metrics?.[m];
          if (!Number.isFinite(v)) return h('td', { class: 'num muted' }, 'Not reported', slot(''));
          return h('td', { class: 'num', title: String(v) }, decimals(v), slot(best.get(m).has(r.id) ? 'best' : ''));
        });
        const show = h('button', { class: 'btn btn--sm', type: 'button', hidden: !galleryHas(r.step), dataset: { focus: `show-${r.id}` }, onclick: () => onShowImages(r.step) }, 'Show images');
        showButtons.set(r.id, { button: show, step: r.step });
        const classes = r.hasClassCounts ? h('button', { class: 'btn btn--sm', type: 'button', dataset: { focus: `classes-${r.id}` }, onclick: () => openClasses(r) }, 'Class results') : null;
        return h(
          'tr',
          null,
          h('td', { class: 'num' }, Number.isFinite(r.step) ? fmt.int(r.step) : 'Not reported'),
          severalAttempts ? h('td', { class: 'mono', title: r.attempt ?? '' }, shortAttempt(r.attempt) || 'Not reported') : null,
          cells,
          h('td', null, h('div', { class: 'val__actions' }, show, classes)),
        );
      }),
    );
    if (focusedId) tbody.querySelector?.(`[data-focus="${focusedId}"]`)?.focus();
  }

  function renderMore() {
    moreEl.hidden = !(st.phase === 'ready' && st.next !== null);
    moreButton.disabled = st.more;
    moreButton.textContent = st.more ? 'Loading earlier results…' : 'Load earlier';
    moreError.textContent = st.moreError;
    moreError.hidden = !st.moreError;
  }

  // ── class results (classes.js) ──
  // One report at a time, under the table. The sort order the person chose stays for the next one.
  const classSort = { key: 'id', dir: 'ascending' };
  let classReport = null;

  function closeClasses() {
    classReport?.destroy();
    classReport = null;
    classesEl.hidden = true;
    classesEl.replaceChildren();
  }

  function openClasses(row) {
    classReport?.destroy();
    classesEl.hidden = false;
    classReport = mountClassReport(classesEl, { evaluation: row, request: (params, opts) => api.classes(current.id, params, opts), sort: classSort });
  }

  load();

  return {
    setRun(next) {
      if (destroyed || !next) return;
      const before = current?.id;
      current = next;
      if (next.id !== before) {
        closeClasses();
        chartKey = '';
        table.dataset.key = '';
        otherAbort?.abort();
        Object.assign(other, { id: null, phase: 'idle', error: '', series: [], cut: 0, perImage: null });
        load();
      } else if (!timer && runStatus(next.status).active && st.phase === 'ready') schedule();
    },
    /** The gallery loaded: show or hide "Show images" per row without rebuilding the table. */
    galleryChanged() {
      for (const { button, step } of showButtons.values()) button.hidden = !galleryHas(step);
    },
    destroy() {
      destroyed = true;
      clearTimeout(timer);
      chartObserver.disconnect();
      listAbort?.abort();
      otherAbort?.abort();
      classReport?.destroy();
      classReport = null;
      for (const c of charts) c.chart.destroy();
      charts = [];
      root.remove();
    },
  };
}
