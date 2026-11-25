// Telemetry tab of a run: series logged by the trainer (GET /api/run/history), drawn on charts the
// person picks, beside the latest values the service read (`run.metrics`). Indexing of a long log is
// incremental on the service: while `more` is true this module asks again and shows how far it got.
// Background refreshes update nodes in place; the series picker is never rebuilt.
// Beside the log's name: "Copy log path", the place of that file on the training PC as the service gave it
// (`sourcePaths` of the same answer; a Windows and a WSL form when the service runs in WSL). The page
// never builds a path itself.

import { h, fmt, uid } from '../dom.js';
import { api, watchHistory } from '../api.js';
import { runStatus } from '../state/runs.js';
import { emptyState, notice, loading, toast, copyText, openCopyDialog } from '../ui.js';
import { createChart } from '../components/chart.js';
import { SERIES_GROUPS, groupOfKey, groupHistorySeries, defaultHistorySelection, historyPoints, bytesProgress } from './metrics.js';

const RETRY_MS = 10000;
const BUDGET = 1000;
const MAX_ROUNDS = 4000; // each call indexes at most 4 MiB more; this bounds one discovery at about 16 GB

const setText = (node, text) => {
  if (node.textContent !== text) node.textContent = text;
};

/** Put `node` at position `index` of `parent` only when it is not already there. */
function placeAt(parent, node, index) {
  if (parent.children[index] !== node) parent.insertBefore(node, parent.children[index] ?? null);
}

/** `sourcePaths` of a history answer as {native, windows?}, or null when the service sent no usable path. */
function readPaths(value) {
  const text = (v) => (typeof v === 'string' && v.trim() ? v : null);
  const native = text(value?.native);
  if (!native) return null;
  return { native, windows: text(value?.windows) };
}

/**
 * @param {HTMLElement} host
 * @param {{run: object}} options
 * @returns {{setRun(run: object): void, setShown(shown: boolean): void, source(): {runId:string, source:string}|null, destroy(): void}}
 *   `source()` is the log the person chose here, or null for the full history.
 */
export function mountTelemetry(host, { run: initialRun }) {
  let run = initialRun;
  const id = run.id;
  let shown = false;
  let destroyed = false;
  let controller = null;
  let timer = 0;
  let cycling = false;
  let firstDone = false;
  let stopWatching;
  let changedDuringRead = false;

  let sources = [];
  let chosenSource = null; // null follows the checkpoint lineage
  let resumeSteps = [];
  let partialHistory = false;
  let activeSource = null;
  // Where the file of `activeSource` is on the training PC, from the same answer. `pathsOld` from the
  // moment another log is picked until its answer is in: a path of the log before is never copied.
  let paths = null;
  let pathsOld = false;
  let seriesList = [];
  let selected = null; // Set of keys once decided
  const data = new Map(); // key -> {points, shown, matched, downsampled, error}
  let noLogs = false;
  let noLogsReason = ''; // the service's words when it has no experiment folder for the run
  let loadError = '';

  // ── nodes ──
  const statusEl = h('p', { class: 'runs__tele-status muted', role: 'status', 'aria-live': 'polite' });
  const errorSlot = h('div');
  const sourceId = uid('runs-source');
  const sourceSelect = h('select', { class: 'select', id: sourceId, onchange: () => pickSource(sourceSelect.value) });
  // A run with one log has nothing to pick: its name stands where the list would be.
  const sourceName = h('span', { class: 'mono runs__tele-name', hidden: true });
  const copyWindows = h('button', { class: 'btn btn--sm', type: 'button', hidden: true, onclick: () => copyPath('windows') });
  const copyNative = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => copyPath('native') });
  const copyGroup = h('div', { class: 'runs__tele-copy', role: 'group', 'aria-label': 'Copy log path', hidden: true }, copyWindows, copyNative);
  const sourceRow = h('div', { class: 'runs__tele-source', hidden: true }, h('label', { for: sourceId }, 'Training history'), sourceSelect, sourceName, copyGroup);
  const pickerEl = h('div', { class: 'runs__picker' });
  const picker = h(
    'aside',
    { class: 'runs__picker-pane', 'aria-labelledby': `${sourceId}-pick` },
    h('h3', { class: 'runs__section-title', id: `${sourceId}-pick` }, 'Logged series'),
    h('p', { class: 'runs__hint muted' }, 'Tick the series to draw.'),
    pickerEl,
  );
  const chartsEl = h('div', { class: 'runs__charts' });
  const chartsNote = h('p', { class: 'runs__pad muted', hidden: true }, 'Tick a series to draw it.');
  const latestBody = h('tbody');
  let valNote = null;
  const latestEmpty = h('p', { class: 'muted' }, 'Not reported yet.');
  const latest = h(
    'section',
    { class: 'runs__latest', 'aria-labelledby': `${sourceId}-latest` },
    h('h3', { class: 'runs__section-title', id: `${sourceId}-latest` }, 'Latest values'),
    h('p', { class: 'runs__hint muted' }, 'The newest value of each number the service read from the trainer log.'),
    h(
      'p',
      { class: 'runs__hint muted', ref: (node) => (valNote = node), hidden: true },
      'Validation scores are shown here as latest values only. Their history over iterations is in the Validation tab under Scores, and the run comparison holds the per-image record.',
    ),
    latestEmpty,
    h('table', { class: 'table runs__table' }, h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Name'), h('th', { scope: 'col', class: 'num' }, 'Value'))), latestBody),
  );
  const mainEl = h('div', { class: 'runs__tele-main' }, chartsNote, chartsEl);
  const grid = h('div', { class: 'runs__tele' }, picker, mainEl, latest);
  const emptySlot = h('div');
  const loadingEl = h('div', { class: 'runs__pad' }, loading('Reading the trainer log…'));
  host.append(h('div', { class: 'runs__tele-top' }, statusEl, sourceRow), errorSlot, loadingEl, emptySlot, grid);
  grid.hidden = true;

  const groupEls = new Map(); // group id -> {fieldset, list}
  const items = new Map(); // key -> {li, input, count}
  const charts = new Map(); // group id -> {section, chart, notes}
  const latestRows = new Map(); // name -> {tr, value}

  renderLatest();
  const onVisibility = () => {
    if (document.hidden) stopTimer();
    else if (shown && !cycling && (!firstDone || isActive())) schedule(0);
  };
  document.addEventListener('visibilitychange', onVisibility);

  // ── data ──

  const isActive = () => runStatus(run.status).active;

  function watchRun() {
    if (!shown || !isActive()) {
      stopWatching?.();
      stopWatching = null;
    } else if (!stopWatching) {
      stopWatching = watchHistory({ kind: 'comparison', id }, () => {
        if (cycling) changedDuringRead = true;
        else schedule(0);
      });
    }
  }

  function stopTimer() {
    clearTimeout(timer);
    timer = 0;
  }

  function schedule(ms) {
    stopTimer();
    if (destroyed || !shown || document.hidden) return;
    timer = setTimeout(cycle, ms);
  }

  async function cycle() {
    stopTimer();
    controller?.abort();
    controller = new AbortController();
    const { signal } = controller;
    cycling = true;
    changedDuringRead = false;
    let again = false;
    try {
      await discover(signal);
      if (signal.aborted || destroyed) return;
      await fetchSelected(signal);
      if (signal.aborted || destroyed) return;
      loadError = '';
    } catch (error) {
      if (error?.name === 'AbortError' || destroyed) return;
      if (error.status === 404 && chosenSource) {
        // The chosen log no longer belongs to this run: return to full history.
        chosenSource = null;
        data.clear();
        loadError = '';
        again = true;
      } else if (error.status === 404) {
        // No experiment folder yet (a launch that failed early, or one still starting): nothing logged.
        sources = [];
        activeSource = null;
        paths = null;
        seriesList = [];
        noLogs = true;
        noLogsReason = error.message;
        loadError = '';
      } else {
        loadError = `The logged series could not be read: ${error.message}`;
      }
    } finally {
      if (!signal.aborted && !destroyed) {
        cycling = false;
        if (again) schedule(0);
        else {
          firstDone = true;
          render();
          // A series ticked while this round ran has no points yet: fetch it now.
          const missing = !loadError && !noLogs && selected && [...selected].some((k) => !data.has(k));
          // Keep up with a run that is still writing its log; a finished log does not change.
          if (missing || changedDuringRead) schedule(0);
          else if (loadError && isActive()) schedule(RETRY_MS);
        }
      }
    }
  }

  function requestHistory(params, signal) {
    const owner = chosenSource?.runId ?? id;
    const scope = chosenSource ? { source: chosenSource.source } : { scope: 'training' };
    return api.history(owner, { ...scope, ...params }, { signal });
  }

  async function discover(signal) {
    for (let round = 0; round < MAX_ROUNDS; round += 1) {
      const res = await requestHistory({}, signal);
      if (signal.aborted || destroyed) return;
      applyDiscovery(res);
      if (!res || !res.more) return;
      showIndexing(res);
    }
  }

  function applyDiscovery(res) {
    if (!chosenSource) sources = res?.attemptSources ?? [];
    resumeSteps = res?.resumeSteps ?? [];
    partialHistory = Boolean(res?.partialHistory);
    activeSource = typeof res?.source === 'string' ? res.source : null;
    paths = activeSource ? readPaths(res?.sourcePaths) : null;
    pathsOld = false;
    seriesList =Array.isArray(res?.series) ? res.series.filter((s) => s && typeof s.key === 'string') : [];
    noLogs = !sources.length && !seriesList.length;
    noLogsReason = '';
    const keys = new Set(seriesList.map((s) => s.key));
    if (selected === null && seriesList.length) selected = new Set(defaultHistorySelection([...keys]));
    if (selected) for (const key of [...data.keys()]) if (!selected.has(key)) data.delete(key);
    renderPicker();
    renderSource();
  }

  function showIndexing(res) {
    const text = bytesProgress(res.indexedBytes, res.sourceBytes);
    setText(statusEl, text ? `Reading the trainer log: ${text}.` : 'Reading the trainer log…');
  }

  async function fetchSelected(signal) {
    if (!selected) return;
    for (const key of [...selected]) {
      if (signal.aborted || destroyed) return;
      await fetchKey(key, signal);
    }
  }

  async function fetchKey(key, signal) {
    const known = seriesList.some((s) => s.key === key);
    if (!known) {
      data.set(key, { points: [], error: 'Not logged in this trainer log.' });
      return;
    }
    try {
      const res = await requestHistory({ metric: key, budget: BUDGET, ...charts.get(groupOfKey(key))?.chart.iterationRange() }, signal);
      if (signal.aborted || destroyed || !selected?.has(key)) return;
      const points = historyPoints(res?.points);
      data.set(key, {
        points,
        shown: Array.isArray(res?.points) ? res.points.length : 0,
        matched: Number.isFinite(res?.matchedPoints) ? res.matchedPoints : null,
        downsampled: res?.downsampled === true,
        error: '',
      });
      if (res?.more) showIndexing(res);
    } catch (error) {
      if (error?.name === 'AbortError') throw error;
      if (error.status === 400) data.set(key, { points: [], error: error.message });
      else throw error;
    }
  }

  function pickSource(value) {
    controller?.abort();
    chosenSource = value ? sources.find((item) => sourceKey(item) === value) ?? null : null;
    data.clear();
    pathsOld = true;
    renderSource();
    schedule(0);
  }

  /** Copy where the shown log is, in one of the forms the service gave. Starts inside the press (the iPhone asks for that). */
  function copyPath(form) {
    const text = pathsOld ? null : paths?.[form];
    if (!text) return;
    const both = Boolean(paths.windows);
    const what = both ? (form === 'windows' ? 'Windows path' : 'WSL path') : 'path';
    const said = `Copied the ${what} of ${activeSource}.`;
    copyText(text).then((ok) => {
      if (destroyed) return;
      if (ok) toast(said, { kind: 'ok' });
      else openCopyDialog({ title: both ? `Copy the ${what} of the log` : 'Copy the log path', text, onCopied: () => toast(said, { kind: 'ok' }) });
    });
  }

  function toggleKey(key, on) {
    if (!selected) selected = new Set();
    if (on) selected.add(key);
    else {
      selected.delete(key);
      data.delete(key);
    }
    renderCharts();
    if (on && !cycling) {
      const ctrl = new AbortController();
      const prev = controller;
      controller = ctrl;
      prev?.abort();
      fetchKey(key, ctrl.signal)
        .then(() => {
          if (!destroyed && !ctrl.signal.aborted) renderCharts();
        })
        .catch((error) => {
          if (error?.name !== 'AbortError' && !destroyed) {
            data.set(key, { points: [], error: error.message });
            renderCharts();
          }
        });
    }
  }

  // ── render ──

  function render() {
    loadingEl.hidden = firstDone || Boolean(loadError);
    errorSlot.replaceChildren(...(loadError ? [notice('err', loadError, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => schedule(0) }, 'Try again'))] : []));
    if (!cycling) setText(statusEl, statusText());
    const empty = firstDone && noLogs;
    grid.hidden = !firstDone || empty;
    if (empty) {
      const reason = noLogsReason ? ` The service says: ${noLogsReason}` : '';
      const text = isActive()
        ? `The trainer has not written its log yet. Charts appear here after its first progress line.${reason}`
        : `The service found no trainer log (train_*.log) for this run, so there are no logged series to draw.${reason}`;
      if (emptySlot.dataset.text !== text) {
        emptySlot.dataset.text = text;
        emptySlot.replaceChildren(emptyState({ title: 'No trainer log', text }));
      }
    } else if (emptySlot.dataset.text) {
      delete emptySlot.dataset.text;
      emptySlot.replaceChildren();
    }
    renderPicker();
    renderSource();
    renderCharts();
    renderLatest();
  }

  function statusText() {
    if (!firstDone || noLogs) return '';
    const n = seriesList.length;
    const from = activeSource ? ` from ${activeSource}` : ' across the full training history';
    const markers = resumeSteps.length ? ` Dashed lines mark resumes at ${resumeSteps.map((step) => fmt.int(step)).join(', ')}.` : '';
    const coverage = partialHistory ? ' Some earlier history could not be linked.' : '';
    return `${fmt.plural(n, 'series', 'series')} logged${from}.${markers}${coverage}${isActive() ? ' Refreshed every 10 s while the run is active.' : ''}`;
  }

  function sourceKey(item) {
    return `${item.runId}/${item.source}`;
  }

  function renderSource() {
    const many = sources.length > 1;
    sourceRow.hidden = !sources.length;
    if (sourceRow.hidden) return;
    sourceSelect.hidden = !many;
    sourceName.hidden = many;
    if (many) {
      const wanted = JSON.stringify(sources);
      if (sourceSelect.dataset.list !== wanted) {
        sourceSelect.dataset.list = wanted;
        const logs = sources.filter((item) => !item.source.startsWith('TensorBoard/'));
        const events = sources.filter((item) => item.source.startsWith('TensorBoard/'));
        sourceSelect.replaceChildren(h('option', { value: '' }, 'Full training history'), ...[...logs, ...events].map((item) => {
          const index = logs.indexOf(item);
          const label = index < 0 ? item.source : `Attempt ${logs.length - index} · ${item.source}`;
          return h('option', { value: sourceKey(item) }, label);
        }));
      }
      sourceSelect.value = chosenSource ? sourceKey(chosenSource) : '';
    } else setText(sourceName, sources[0].source);

    // One button per form of the path the service gave. With both, each says which it copies; the path
    // itself is the button's description (its tooltip).
    copyGroup.hidden = !paths;
    if (!paths) return;
    const both = Boolean(paths.windows);
    copyWindows.hidden = !both;
    setText(copyWindows, 'Copy Windows path');
    setText(copyNative, both ? 'Copy WSL path' : 'Copy log path');
    const describe = (button, text) => {
      const title = pathsOld ? 'Reading the log you picked…' : text;
      if (button.getAttribute('title') !== title) button.setAttribute('title', title);
      if (pathsOld) button.setAttribute('aria-disabled', 'true');
      else button.removeAttribute('aria-disabled');
    };
    describe(copyWindows, `For an assistant or program that runs on Windows: ${paths.windows ?? ''}`);
    describe(copyNative, both ? `For an assistant or program that runs inside WSL: ${paths.native}` : `Where this log is on the training PC: ${paths.native}`);
  }

  function renderPicker() {
    const groups = groupHistorySeries(seriesList);
    const keepGroups = new Set(groups.map((g) => g.id));
    for (const [gid, g] of groupEls) {
      if (!keepGroups.has(gid)) {
        g.fieldset.remove();
        groupEls.delete(gid);
      }
    }
    const keepKeys = new Set(seriesList.map((s) => s.key));
    for (const [key, item] of items) {
      // A focused checkbox stays even if its series left the list for a moment.
      if (!keepKeys.has(key) && document.activeElement !== item.input) {
        item.li.remove();
        items.delete(key);
      }
    }
    groups.forEach((group, gi) => {
      let g = groupEls.get(group.id);
      if (!g) {
        const list = h('ul', { class: 'runs__picker-list' });
        g = { fieldset: h('fieldset', { class: 'runs__picker-group' }, h('legend', { class: 'runs__picker-legend' }, group.title), list), list };
        groupEls.set(group.id, g);
      }
      placeAt(pickerEl, g.fieldset, gi);
      group.items.forEach((s, si) => {
        let item = items.get(s.key);
        if (!item) {
          const inputId = uid('runs-series');
          const input = h('input', { type: 'checkbox', id: inputId, onchange: () => toggleKey(s.key, input.checked) });
          const count = h('span', { class: 'runs__picker-count num muted' });
          const li = h('li', { class: 'runs__picker-item' }, input, h('label', { for: inputId, class: 'runs__picker-label' }, h('span', { class: 'mono' }, s.key), count));
          item = { li, input, count };
          items.set(s.key, item);
        }
        if (item.li.parentNode !== g.list || g.list.children[si] !== item.li) placeAt(g.list, item.li, si);
        const on = Boolean(selected?.has(s.key));
        if (item.input.checked !== on) item.input.checked = on;
        setText(item.count, Number.isFinite(s.count) ? fmt.plural(s.count, 'point') : '');
      });
    });
  }

  function renderCharts() {
    const keys = selected ? [...selected] : [];
    chartsNote.hidden = !firstDone || keys.length > 0 || noLogs;
    const byGroup = new Map();
    for (const key of keys) {
      const gid = groupOfKey(key);
      if (!byGroup.has(gid)) byGroup.set(gid, []);
      byGroup.get(gid).push(key);
    }
    for (const [gid, entry] of charts) {
      if (!byGroup.has(gid)) {
        entry.chart.destroy();
        entry.section.remove();
        charts.delete(gid);
      }
    }
    let index = 0;
    for (const group of SERIES_GROUPS) {
      const groupKeys = byGroup.get(group.id);
      if (!groupKeys) continue;
      let entry = charts.get(group.id);
      if (!entry) {
        const section = h('section', { class: 'runs__chart' }, h('h3', { class: 'runs__section-title' }, group.title));
        const chart = createChart(section, { title: group.title, height: 220, onXRangeChange: cycle });
        const notes = h('ul', { class: 'runs__chart-notes' });
        section.appendChild(notes);
        entry = { section, chart, notes };
        charts.set(group.id, entry);
      }
      placeAt(chartsEl, entry.section, index);
      index += 1;
      const series = groupKeys.sort().map((key) => ({ name: key, points: data.get(key)?.points ?? [] }));
      entry.chart.update({ series, xLabel: 'Iteration', markers: resumeSteps });
      const notes = [];
      for (const key of groupKeys) {
        const d = data.get(key);
        if (!d) notes.push(`${key}: loading…`);
        else if (d.error) notes.push(`${key}: ${d.error}`);
        else {
          const gaps = d.points.filter((p) => p.y === null).length;
          if (d.downsampled && d.matched !== null) {
            notes.push(`${key}: showing ${fmt.int(d.shown)} of ${fmt.int(d.matched)} logged points; each step keeps its first, last, lowest and highest value.`);
          }
          if (gaps) notes.push(`${key}: ${fmt.plural(gaps, 'point has', 'points have')} no number in the log and ${gaps === 1 ? 'is' : 'are'} drawn as a gap.`);
        }
      }
      const text = notes.join('\n');
      if (entry.notes.dataset.text !== text) {
        entry.notes.dataset.text = text;
        entry.notes.replaceChildren(...notes.map((n) => h('li', null, n)));
      }
    }
  }

  function renderLatest() {
    const metrics = run.metrics && typeof run.metrics === 'object' && !Array.isArray(run.metrics) ? run.metrics : {};
    const names = Object.keys(metrics)
      .filter((k) => typeof metrics[k] === 'number' && Number.isFinite(metrics[k]))
      .sort();
    latestEmpty.hidden = names.length > 0;
    latestBody.parentNode.hidden = names.length === 0;
    // Values with no history series of their own (validation scores such as <dataset>/psnr).
    const logged = new Set(seriesList.map((s) => s.key));
    if (valNote) valNote.hidden = !names.some((n) => !logged.has(n) && !n.startsWith('l_'));
    const keep = new Set(names);
    for (const [name, row] of latestRows) {
      if (!keep.has(name)) {
        row.tr.remove();
        latestRows.delete(name);
      }
    }
    names.forEach((name, i) => {
      let row = latestRows.get(name);
      if (!row) {
        const value = h('td', { class: 'num' });
        row = { tr: h('tr', null, h('th', { scope: 'row', class: 'mono runs__metric-name' }, name), value), value };
        latestRows.set(name, row);
      }
      placeAt(latestBody, row.tr, i);
      setText(row.value, fmt.num(metrics[name], 6));
    });
  }

  return {
    setRun(next) {
      const wasActive = isActive();
      const newAttempt = next.attemptId && next.attemptId !== run.attemptId;
      run = next;
      renderLatest();
      // A run that just became active (resumed) starts refreshing again; one that ended gets a last read.
      if (newAttempt) firstDone = false;
      watchRun();
      if (shown && !cycling && (!firstDone || isActive() !== wasActive)) schedule(0);
    },
    setShown(on) {
      shown = on;
      watchRun();
      if (!on) {
        stopTimer();
        return;
      }
      if (!cycling && (!firstDone || isActive())) schedule(0);
    },
    source: () => chosenSource,
    destroy() {
      destroyed = true;
      stopWatching?.();
      stopTimer();
      controller?.abort();
      document.removeEventListener('visibilitychange', onVisibility);
      for (const entry of charts.values()) entry.chart.destroy();
      charts.clear();
      host.replaceChildren();
    },
  };
}
