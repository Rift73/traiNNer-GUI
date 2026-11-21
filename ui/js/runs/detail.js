// One run: header (status, reported progress and telemetry, actions) and the tabs Telemetry,
// Validation, Log, Files and Recipe. The full detail (GET /api/run) is read once, then the run is
// followed with GET /api/run/monitor and its log cursor; the full detail is read again only when the
// status changes, when Files or Recipe is opened, or on Refresh. Nothing shown is estimated here.
// Files: each checkpoint row is one radio choice and a single action bar acts on the chosen file; above it,
// "Which checkpoint measured best?" (GET /api/run/checkpoint-selection, read once per run and series).

import { h, mount as mountNodes, fmt, uid } from '../dom.js';
import { api, artifactUrl } from '../api.js';
import { persist } from '../store.js';
import { runs, runStatus, isImported, runSourceLabel } from '../state/runs.js';
import { catalog } from '../state/catalog.js';
import { conn } from '../state/conn.js';
import { compareHref, queueHref, runHref, curriculaHref, openAsRecipe } from '../actions.js';
import { toast, confirmDialog, confirmOnPhone, promptDialog, tabs, badge, emptyState, notice, loading, copyText } from '../ui.js';
import { highlightedPre } from '../recipe/yaml-highlight.js';
import { createLogView } from './log.js';
import { mountTelemetry } from './telemetry.js';
import { mountProvenance } from './provenance.js';

const TABS = [
  { id: 'telemetry', label: 'Telemetry' },
  { id: 'validation', label: 'Validation' },
  { id: 'samples', label: 'Training samples' },
  { id: 'log', label: 'Log' },
  { id: 'files', label: 'Files' },
  { id: 'recipe', label: 'Recipe' },
  { id: 'provenance', label: 'Provenance' },
];
const FILL_TABS = new Set(['log', 'recipe']);
/** Fields taken from a monitor answer (when present); everything else comes from the full detail. */
const MONITOR_FIELDS = ['status', 'iteration', 'iterationsPerSecond', 'eta', 'vram', 'telemetryIteration', 'error', 'returncode', 'metrics'];
const FOLDERS = ['models/', 'models/resume_models/', 'training_states/'];
/** Curriculum phase modes in words (studio/curricula.py: the first phase starts, later ones continue or fine-tune). */
const PHASE_MODES = { start: 'Start', continue: 'Continue', finetune: 'Fine-tune' };

const hasProgress = (run) => Number.isFinite(run?.iteration) && Number.isFinite(run?.total);
const nameOf = (run) => run?.name ?? String(run?.id ?? '');
const setText = (node, text) => {
  if (node.textContent !== text) node.textContent = text;
};

/** Put `node` at position `index` of `parent` only when it is not already there. */
function placeAt(parent, node, index) {
  if (parent.children[index] !== node) parent.insertBefore(node, parent.children[index] ?? null);
}

/** Replace the children of `slot` only when `key` changed, so a re-render does not drop focus or re-announce. */
function mountKeyed(slot, key, build) {
  if (slot.dataset.key === key) return;
  slot.dataset.key = key;
  mountNodes(slot, build());
}

export function progressBar(label) {
  const fill = h('div', { class: 'progress__bar' });
  const el = h('div', { class: 'progress', role: 'progressbar', 'aria-label': label, 'aria-valuemin': '0' }, fill);
  const set = (run) => {
    const kind = runStatus(run.status).kind;
    el.className = `progress${kind ? ` progress--${kind}` : ''}`;
    el.setAttribute('aria-valuemax', String(run.total));
    el.setAttribute('aria-valuenow', String(run.iteration));
    fill.style.width = `${run.total > 0 ? Math.max(0, Math.min(100, (run.iteration / run.total) * 100)) : 0}%`;
  };
  // Callers call set() only when iteration and total are real numbers.
  return { el, set };
}

/** What the run head says about a curriculum phase, as plain values (also the re-render key), or null. */
function phaseFacts(c) {
  if (!c || typeof c !== 'object' || typeof c.curriculumId !== 'string' || !c.curriculumId) return null;
  const number = /^phase-(\d+)$/.exec(String(c.phaseId ?? ''));
  const handoff = (c.mode === 'continue' || c.mode === 'finetune') && c.handoff && typeof c.handoff === 'object' && typeof c.handoff.runId === 'string' && c.handoff.runId ? c.handoff : null;
  return {
    curriculumId: c.curriculumId,
    label: number ? `Phase ${Number(number[1])} of a curriculum` : 'A phase of a curriculum',
    mode: PHASE_MODES[c.mode] ?? '',
    range: Number.isFinite(c.startStep) && Number.isFinite(c.targetStep) ? `iterations ${fmt.int(c.startStep)} to ${fmt.int(c.targetStep)}` : '',
    from: handoff ? { runId: handoff.runId, name: runs.byId?.(handoff.runId)?.name ?? '', step: Number.isFinite(handoff.step) ? handoff.step : null } : null,
  };
}

/** "Phase 2 of a curriculum: Continue, iterations 1,000 to 2,000, took over from <run> at iteration 1,000." */
function phaseLine(p) {
  const words = [p.mode, p.range].filter(Boolean).join(', ');
  return h(
    'p',
    { class: 'runs__phase' },
    h('a', { href: curriculaHref(p.curriculumId) }, p.label),
    words ? `: ${words}` : '',
    p.from
      ? [
          `${words ? ',' : ':'} took over from `,
          h('a', { href: runHref(p.from.runId), class: p.from.name ? null : 'mono' }, p.from.name || p.from.runId),
          p.from.step !== null ? ` at iteration ${fmt.int(p.from.step)}` : '',
        ]
      : null,
    '.',
  );
}

/**
 * @param {HTMLElement} host
 * @param {{id: string, summary?: object|null}} options
 * @returns {{id: string, onSummary(run: object): void, tick(): void, destroy(): void}}
 */
export function createRunDetail(host, { id, summary = null }) {
  let run = { id, ...(summary ?? {}) };
  let loaded = false;
  let destroyed = false;
  let cursor = null;
  let detailCtrl = null;
  let detailBusy = false;
  let pollCtrl = null;
  let pollTimer = 0;
  let polling = false;
  const busy = { stop: false, resume: false };

  // ── header ──
  const errorSlot = h('div', { class: 'runs__error' });
  // Focusable by script: the runs list hands the focus here when a run is opened from its row.
  const nameEl = h('h2', { class: 'runs__name', tabindex: '-1' });
  const badgeSlot = h('span');
  const importedSlot = h('span');
  const stopBtn = h('button', { class: 'btn btn--danger btn--sm', type: 'button', onclick: stopRun }, 'Stop training');
  const resumeBtn = h('button', { class: 'btn btn--sm', type: 'button', onclick: resumeRun }, 'Resume');
  // Offered only where the service says this run can continue with another number of data loader workers.
  const workersBtn = h('button', { class: 'btn btn--sm', type: 'button', hidden: true, onclick: resumeWithWorkers }, 'Resume with other workers…');
  const replayBtn = h('button', { class: 'btn btn--sm', type: 'button', onclick: replay }, 'Replay');
  const compareLink = h('a', { class: 'btn btn--sm', href: compareHref(id) }, 'Compare with…');
  // Refresh reads the detail again, and tells the parts that keep their own answers (the Files tab's ranking).
  const refreshBtn = h('button', { class: 'btn btn--quiet btn--sm', type: 'button', onclick: () => {
    fetchDetail();
    for (const { part } of panels.values()) part?.refresh?.();
  } }, 'Refresh');
  const importedNote = h('p', { class: 'runs__note muted', hidden: true });
  const phaseSlot = h('div', { class: 'runs__phase-slot' });
  const failSlot = h('div');
  // The disk warning of a training run (watchDisk below). Declared here: the first renderHead() runs before
  // the rest of this function has been reached.
  const diskSlot = h('div');
  const DISK_EVERY_MS = 300000;
  let diskAt = 0;
  let diskBusy = false;
  let diskOff = false; // the service does not know this run: no further readings
  let diskText = '';
  const bar = progressBar('Training progress');
  const progressText = h('span', { class: 'runs__progress-text num' });
  const progressEl = h('div', { class: 'runs__progress' }, bar.el, progressText);
  const kv = h('dl', { class: 'kv runs__kv' });
  const kvRows = new Map();
  const head = h(
    'header',
    { class: 'runs__head' },
    h('div', { class: 'runs__title-row' }, nameEl, badgeSlot, importedSlot, h('div', { class: 'runs__actions' }, stopBtn, resumeBtn, workersBtn, replayBtn, compareLink, refreshBtn)),
    importedNote,
    phaseSlot,
    failSlot,
    diskSlot,
    progressEl,
    kv,
  );

  // ── tabs ──
  const saved = persist.get('runs-tab');
  let activeTab = TABS.some((t) => t.id === saved) ? saved : 'telemetry';
  const strip = tabs({ items: TABS, active: activeTab, label: 'Run detail', onChange: (tabId) => switchTab(tabId) });
  const panels = new Map(); // tab id -> {panel, part}
  const panelHost = h('div', { class: 'runs__panels' });
  for (const t of TABS) {
    const panel = h('div', { class: FILL_TABS.has(t.id) ? 'runs__panel runs__panel--fill' : 'runs__panel', role: 'tabpanel', id: `${host.id || 'runs'}-${t.id}-${Math.random().toString(36).slice(2, 7)}`, tabindex: '-1', hidden: true });
    strip.el.querySelector(`[data-tab="${t.id}"]`)?.setAttribute('aria-controls', panel.id);
    panel.setAttribute('aria-label', t.label);
    panels.set(t.id, { panel, part: null });
    panelHost.appendChild(panel);
  }
  const bodyLoading = h('div', { class: 'runs__pad' }, loading('Loading this run…'));
  const notFoundSlot = h('div');
  host.append(errorSlot, notFoundSlot, head, strip.el, bodyLoading, panelHost);
  panelHost.hidden = true;

  // The log view exists from the start: the monitor feeds it whichever tab is open.
  const logPart = createLogView(panels.get('log').panel, { getName: () => nameOf(run) });
  panels.get('log').part = { update() {}, setShown: (on) => logPart.setShown(on), destroy: () => logPart.destroy() };

  Promise.resolve()
    .then(() => catalog.load?.())
    .catch(() => {}); // without the catalog, task ids are shown in plain words
  const unsubCatalog = catalog.subscribe(() => renderHead());
  const onVisibility = () => {
    if (!document.hidden && loaded && !polling) schedulePoll(0);
  };
  document.addEventListener('visibilitychange', onVisibility);

  renderHead();
  fetchDetail();

  // ── header rendering ──

  function renderHead() {
    setText(nameEl, nameOf(run));
    const st = runStatus(run.status);
    const imported = isImported(run);
    mountKeyed(badgeSlot, `${st.label}|${st.kind}`, () => badge(st.label, st.kind));
    // "Imported" only when the status word lacks it; where the run comes from is said once, in the note below.
    const tag = imported && !st.imported ? 'Imported' : '';
    mountKeyed(importedSlot, tag, () => (tag ? badge(tag) : null));
    stopBtn.hidden = !(st.active && !imported);
    resumeBtn.hidden = !(st.resumable && !imported);
    workersBtn.hidden = !(st.resumable && !imported && run.resumeWorkerOverride === true);
    stopBtn.disabled = busy.stop;
    resumeBtn.disabled = busy.resume;
    workersBtn.disabled = busy.resume;
    refreshBtn.disabled = detailBusy;

    importedNote.hidden = !imported;
    if (imported) setText(importedNote, `${runSourceLabel(run)}. It can be read and compared, but not stopped or resumed.`);

    // A run that is one phase of a curriculum (GET /api/run gives `curriculum` only then).
    // Its links keep focus: the line is rebuilt on a later render once focus has left it.
    const phase = phaseFacts(run.curriculum);
    if (!phaseSlot.contains(document.activeElement)) mountKeyed(phaseSlot, phase ? JSON.stringify(phase) : '', () => (phase ? phaseLine(phase) : null));

    // A failed run (or any reported error) gets a banner with a way to the reason, which is in the log.
    const errText = typeof run.error === 'string' && run.error ? run.error : '';
    const failed = st.kind === 'err';
    const bannerText = errText ? `${failed ? 'Training failed' : 'The trainer reported an error'}: ${errText}` : failed ? 'Training failed.' : '';
    mountKeyed(failSlot, bannerText, () =>
      bannerText ? notice('err', bannerText, h('button', { class: 'btn btn--sm', type: 'button', onclick: showLog }, 'Show the log')) : null,
    );

    watchDisk();

    progressEl.hidden = !hasProgress(run);
    if (hasProgress(run)) {
      bar.set(run);
      const pct = fmt.percent(run.iteration, run.total);
      setText(progressText, `Iteration ${fmt.int(run.iteration)} of ${fmt.int(run.total)}${pct ? ` (${pct})` : ''}`);
    }

    const facts = [];
    if (run.task) facts.push(['Task', catalog.taskLabel(run.task)]);
    if (run.source === 'archive') facts.push(['Experiments folder', { mono: typeof run.sourceRoot === 'string' ? run.sourceRoot : '' }]);
    // A run that is not active keeps the trainer's last numbers: its estimate is stale, its speed is the last one.
    if (Number.isFinite(run.iterationsPerSecond)) facts.push([st.active ? 'Reported speed' : 'Last reported speed', `${fmt.num(run.iterationsPerSecond, 4)} it/s`]);
    if (st.active && typeof run.eta === 'string' && run.eta) facts.push(["Trainer's estimate", run.eta]);
    const vram = run.vram && typeof run.vram === 'object' ? run.vram : null;
    if (vram && (Number.isFinite(vram.peakAllocatedGB) || Number.isFinite(vram.peakReservedGB))) {
      const parts = [];
      if (Number.isFinite(vram.peakAllocatedGB)) parts.push(`${fmt.num(vram.peakAllocatedGB, 3)} GB allocated`);
      if (Number.isFinite(vram.peakReservedGB)) parts.push(`${fmt.num(vram.peakReservedGB, 3)} GB reserved`);
      facts.push(['Peak memory', parts.join(', ')]);
    }
    if (Number.isFinite(run.telemetryIteration)) facts.push(['Speed and memory as of', `iteration ${fmt.int(run.telemetryIteration)}`]);
    const started = fmt.dateTime(run.startedAt);
    if (started) facts.push(['Started', started]);
    const updated = fmt.dateTime(run.updatedAt);
    const ago = fmt.ago(run.updatedAt);
    // fmt.ago falls back to the date itself for a time ahead of the browser clock: do not print it twice.
    if (updated) facts.push(['Last update', ago && ago !== updated ? `${updated} (${ago})` : updated]);
    if (Number.isFinite(run.seed)) facts.push(['Seed', String(run.seed)]);
    if (Number.isFinite(run.returncode)) facts.push(['Exit code', String(run.returncode)]);
    if (run.queueEntryId) facts.push(['Queue entry', { href: queueHref(run.queueEntryId), text: 'Open its queue entry' }]);

    const keys = new Set(facts.map(([k]) => k));
    for (const [k, parts] of kvRows) {
      if (!keys.has(k)) {
        parts.dt.remove();
        parts.dd.remove();
        kvRows.delete(k);
      }
    }
    facts.forEach(([k, value], i) => {
      let parts = kvRows.get(k);
      if (!parts) {
        parts = { dt: h('dt', null, k), dd: h('dd', { class: 'num' }), key: '' };
        kvRows.set(k, parts);
      }
      // Order follows `facts`; only rows out of place move (moving a focused link would drop focus).
      placeAt(kv, parts.dt, i * 2);
      placeAt(kv, parts.dd, i * 2 + 1);
      if (typeof value === 'string') {
        if (parts.key !== `t:${value}`) {
          parts.key = `t:${value}`;
          parts.dd.textContent = value;
        }
      } else if (value.mono !== undefined) {
        if (parts.key !== `m:${value.mono}`) {
          parts.key = `m:${value.mono}`;
          mountNodes(parts.dd, h('span', { class: 'mono' }, value.mono || 'Not reported'));
        }
      } else if (parts.key !== `a:${value.href}`) {
        parts.key = `a:${value.href}`;
        mountNodes(parts.dd, h('a', { href: value.href }, value.text));
      }
    });
  }

  function showError(message) {
    const retry = () => h('button', { class: 'btn btn--sm', type: 'button', onclick: () => fetchDetail() }, 'Try again');
    mountKeyed(errorSlot, message ? `err:${message}` : '', () => (message ? notice('err', `This run could not be refreshed: ${message}`, retry()) : null));
  }

  function showNotFound() {
    stopPolling();
    diskOff = true;
    head.hidden = true;
    strip.el.hidden = true;
    panelHost.hidden = true;
    bodyLoading.hidden = true;
    errorSlot.replaceChildren();
    mountKeyed(notFoundSlot, 'missing', () =>
      h(
        'div',
        { class: 'runs__pad' },
        emptyState({
          title: 'Run not found',
          text: `The service has no run with the id ${id}. It may have been removed, or the address is from another installation.`,
          actions: [h('a', { class: 'btn btn--primary', href: '#/runs' }, 'Show all runs')],
        }),
      ),
    );
  }

  // ── data ──

  function updateParts() {
    for (const { part } of panels.values()) part?.update?.(run);
  }

  async function fetchDetail() {
    detailCtrl?.abort();
    const ctrl = new AbortController();
    detailCtrl = ctrl;
    detailBusy = true;
    refreshBtn.disabled = true;
    try {
      const data = await api.run(id, { signal: ctrl.signal });
      if (destroyed || ctrl.signal.aborted) return;
      if (!data || typeof data !== 'object') throw new Error('The service returned no run.');
      const { log, logCursor, logReset, ...fields } = data;
      // The detail replaces the summary: a field the detail omits is not carried over as if reported.
      run = { id, ...fields };
      if (!loaded || cursor === null) {
        logPart.replace(Array.isArray(log) ? log : []);
        cursor = typeof logCursor === 'string' ? logCursor : null;
      }
      const first = !loaded;
      loaded = true;
      runs.upsert(run);
      showError(null);
      bodyLoading.hidden = true;
      panelHost.hidden = false;
      renderHead();
      if (first) showTab(activeTab);
      updateParts();
      if (!polling) schedulePoll(pollDelay());
    } catch (error) {
      if (error?.name === 'AbortError' || destroyed) return;
      if (error.status === 404) {
        showNotFound();
        return;
      }
      showError(error.message);
      if (!loaded) mountNodes(bodyLoading, h('p', { class: 'muted' }, 'The detail of this run is not available.'));
    } finally {
      if (detailCtrl === ctrl) {
        detailBusy = false;
        if (!destroyed) refreshBtn.disabled = false;
      }
    }
  }

  function pollDelay() {
    const st = runStatus(run.status);
    if (st.active) return activeTab === 'log' ? 2500 : 5000;
    return 15000;
  }

  function stopPolling() {
    clearTimeout(pollTimer);
    pollTimer = 0;
    pollCtrl?.abort();
  }

  function schedulePoll(ms) {
    clearTimeout(pollTimer);
    pollTimer = 0;
    if (destroyed || document.hidden || !loaded) return;
    pollTimer = setTimeout(poll, ms);
  }

  async function poll() {
    pollTimer = 0;
    if (destroyed || document.hidden) return;
    pollCtrl?.abort();
    const ctrl = new AbortController();
    pollCtrl = ctrl;
    polling = true;
    try {
      const res = await api.monitor(id, cursor ?? undefined, { signal: ctrl.signal });
      if (destroyed || ctrl.signal.aborted) return;
      applyMonitor(res);
      showError(null);
    } catch (error) {
      if (error?.name === 'AbortError' || destroyed) return;
      if (error.status === 404) {
        showNotFound();
        return;
      }
      // A cursor the service no longer accepts: start over from its tail on the next poll.
      if (error.status === 400 && /cursor/i.test(error.message)) cursor = null;
      else showError(error.message);
    } finally {
      if (pollCtrl === ctrl) {
        polling = false;
        if (!destroyed && head.hidden === false) schedulePoll(pollDelay());
      }
    }
  }

  function applyMonitor(res) {
    if (!res || typeof res !== 'object') return;
    const { log, logCursor, logReset, ...fields } = res;
    const prevStatus = run.status;
    // Only the fields that move are taken from a monitor answer, and only when present: for imported runs
    // it leaves out task and total, and dates the log alone (not the recipe), so its updatedAt is not used.
    // A status change reads the full detail again, which replaces everything.
    const patch = {};
    for (const key of MONITOR_FIELDS) if (fields[key] !== undefined) patch[key] = fields[key];
    run = { ...run, ...patch, id };
    if (logReset) logPart.replace(Array.isArray(log) ? log : []);
    else logPart.append(Array.isArray(log) ? log : []);
    if (typeof logCursor === 'string') cursor = logCursor;
    runs.upsert({ id, ...patch });
    renderHead();
    updateParts();
    if (fields.status !== undefined && fields.status !== prevStatus) fetchDetail();
  }

  // ── tabs ──

  /** "Show the log": open the Log tab and put focus on its lines, scrolled to the newest. */
  function showLog() {
    if (!loaded) return;
    if (activeTab !== 'log') switchTab('log');
    logPart.focusEnd();
  }

  // ── disk space, while the run trains ──
  // A full disk ends a run. While this run trains here, the service's forecast (GET /api/run/checkpoint-storage)
  // is read when the run is opened and then at most every five minutes, and only a shortage is said, above
  // the tabs whichever of them is open. The whole forecast is on the Files tab. Never read for a run that is
  // not training, and never while the page is hidden.
  function watchDisk() {
    if (destroyed) return;
    if (!runStatus(run.status).active || isImported(run)) {
      diskAt = 0;
      diskText = '';
    }
    // Drawn on every header render (it changes nothing when the sentence is the same), so a change that
    // was held back while the warning's button had the focus is caught up without another reading.
    renderDisk();
    if (!runStatus(run.status).active || isImported(run)) return;
    if (diskBusy || diskOff || document.hidden || Date.now() - diskAt < DISK_EVERY_MS) return;
    diskAt = Date.now();
    diskBusy = true;
    // Bounded: a reading that never answers must not end all later readings.
    const ctrl = new AbortController();
    const giveUp = setTimeout(() => ctrl.abort(), 30000);
    Promise.all([import('./storage.js'), api.checkpointStorage(id, { signal: ctrl.signal })])
      .then(([mod, report]) => {
        // The run may have stopped while this was read: a warning is only for a run that trains.
        if (destroyed || typeof mod?.storageVerdict !== 'function' || !runStatus(run.status).active) return;
        const verdict = mod.storageVerdict(report);
        diskText = verdict.kind === 'tight' || verdict.kind === 'full' ? verdict.sentence : '';
        renderDisk();
      })
      .catch((error) => {
        // The forecast is an extra: when it cannot be read, nothing is claimed. A run the service does
        // not know (404) is not asked about again.
        if (error?.status === 404) diskOff = true;
      })
      .finally(() => {
        clearTimeout(giveUp);
        diskBusy = false;
      });
  }
  function renderDisk() {
    // Not rebuilt under the focus of its own button; the next header render catches up.
    if (diskSlot.contains(document.activeElement)) return;
    mountKeyed(diskSlot, diskText, () =>
      diskText ? notice('warn', `Disk space for checkpoints: ${diskText}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: showDisk }, 'Show the disk space')) : null,
    );
  }
  function showDisk() {
    if (!loaded) return;
    if (activeTab !== 'files') switchTab('files');
    const target = panels.get('files').panel.querySelector('.runs__storage-host');
    target?.scrollIntoView?.({ block: 'start' });
    target?.focus?.({ preventScroll: true });
  }

  function switchTab(tabId) {
    activeTab = tabId;
    persist.set('runs-tab', tabId);
    showTab(tabId);
    if (tabId === 'files' || tabId === 'recipe') fetchDetail();
    // The log polls faster while it is on screen.
    if (loaded && !polling && runStatus(run.status).active) schedulePoll(tabId === 'log' ? 0 : pollDelay());
  }

  function showTab(tabId) {
    strip.setActive(tabId);
    for (const [key, entry] of panels) {
      const on = key === tabId;
      entry.panel.hidden = !on;
      if (on && !entry.part) {
        entry.part = createPart(key, entry.panel);
        if (loaded) entry.part?.update?.(run);
      }
      entry.part?.setShown?.(on);
    }
  }

  function createPart(key, panel) {
    if (key === 'telemetry') {
      const t = mountTelemetry(panel, { run });
      // "How fast this run trained": a closed block under the charts; nothing is read until it is opened.
      const timingHost = h('div', { class: 'runs__timing-host' });
      panel.append(timingHost);
      let timing = null;
      let partGone = false;
      const others = () =>
        runs
          .get()
          .list.filter((other) => String(other.id) !== String(id))
          .map((other) => ({ id: String(other.id), name: nameOf(other) }));
      import('./timing.js')
        .then((mod) => {
          if (partGone || destroyed || typeof mod?.mountRunTiming !== 'function') return;
          timing = mod.mountRunTiming(timingHost, { runId: String(id), runs: others });
        })
        .catch(() => {
          // the block is an extra: the charts above stand without it
        });
      return {
        update: (r) => t.setRun(r),
        setShown: (on) => t.setShown(on),
        destroy: () => {
          partGone = true;
          timing?.destroy();
          t.destroy();
        },
      };
    }
    if (key === 'validation') return validationPart(panel);
    if (key === 'samples') return samplesPart(panel);
    if (key === 'files') return mountFilesTab(panel, { getRun: () => run });
    if (key === 'recipe') return recipePart(panel);
    if (key === 'provenance') {
      // Read when the tab is first opened (this part is created then).
      const p = mountProvenance(panel, { run });
      return { update: (r) => p.setRun(r), destroy: () => p.destroy() };
    }
    return null;
  }

  function validationPart(panel) {
    let handle = null;
    let gone = false;
    mountNodes(panel, h('div', { class: 'runs__pad' }, loading('Loading validation images…')));
    import('./validation.js')
      .then((mod) => {
        if (gone || destroyed) return;
        panel.replaceChildren();
        handle = mod.mountValidation(panel, { run });
      })
      .catch((error) => {
        if (gone || destroyed) return;
        mountNodes(panel, h('div', { class: 'runs__pad' }, notice('err', `The validation images could not be shown: ${error.message}`)));
      });
    return {
      update: (r) => handle?.setRun?.(r),
      destroy() {
        gone = true;
        try {
          handle?.destroy?.();
        } catch {
          // A failing part must not stop the cleanup of the others.
        }
      },
    };
  }

  /** Training samples (runs/samples.js, loaded on demand); it polls only while the run is active and the tab shown. */
  function samplesPart(panel) {
    let handle = null;
    let gone = false;
    let shownNow = true;
    mountNodes(panel, h('div', { class: 'runs__pad' }, loading('Loading the training samples…')));
    import('./samples.js')
      .then((mod) => {
        if (gone || destroyed) return;
        panel.replaceChildren();
        handle = mod.mountTrainingSamples(panel, { runId: id, active: runStatus(run.status).active });
        handle.setShown?.(shownNow);
      })
      .catch((error) => {
        if (gone || destroyed) return;
        mountNodes(panel, h('div', { class: 'runs__pad' }, notice('err', `The training samples could not be shown: ${error.message}`)));
      });
    return {
      update: (r) => handle?.setActive?.(runStatus(r.status).active),
      setShown(on) {
        shownNow = on;
        handle?.setShown?.(on);
      },
      destroy() {
        gone = true;
        try {
          handle?.destroy?.();
        } catch {
          // A failing part must not stop the cleanup of the others.
        }
      },
    };
  }

  // ── actions ──

  async function stopRun() {
    const name = nameOf(run);
    const ok = await confirmDialog({
      title: `Stop training ${name}?`,
      body: 'The trainer is asked to stop this run. Whether it can be resumed later depends on the training states it saved.',
      confirmLabel: 'Stop training',
      danger: true,
    });
    if (!ok || destroyed) return;
    act('stop', () => runs.stop(id), `Stop training: requested for ${name}.`, 'Stop training was refused', stopBtn);
  }

  async function resumeRun() {
    // This starts training: the phone edition asks first (the desktop page resumes at once).
    const ok = await confirmOnPhone({
      title: `Resume ${nameOf(run)}?`,
      body: 'Training starts again from the newest training state this run saved.',
      confirmLabel: 'Resume',
    });
    if (!ok || destroyed) return;
    act('resume', () => runs.resume(id), `Resume: ${nameOf(run)} is starting again from its newest training state.`, 'Resume was refused');
  }

  async function resumeWithWorkers() {
    const name = nameOf(run);
    const current = Number.isInteger(run.trainWorkers) ? run.trainWorkers : null;
    const answer = await promptDialog({
      title: `Resume ${name} with other workers`,
      label: 'Data loader workers for the training data',
      value: current === null ? '' : String(current),
      help: `${current === null ? '' : `The run used ${current}. `}The run continues from its newest training state with this number. The service checks that the run's checkpoint allows the change.`,
      confirmLabel: 'Resume',
      validate: (v) => (/^[1-9]\d*$/.test(v.trim()) ? null : 'Enter a whole number, 1 or more.'),
    });
    if (answer === null || answer === undefined || destroyed) return;
    const count = Number(String(answer).trim());
    act(
      'resume',
      () => runs.resume(id, count === current ? undefined : { trainWorkers: count }),
      `Resume: ${name} is starting again from its newest training state with ${count} data loader worker${count === 1 ? '' : 's'}.`,
      'Resume was refused',
      workersBtn,
    );
  }

  /** Run one action with its button disabled, toast the outcome (the backend's own refusal text), then refetch. */
  async function act(kind, request, done, failed, opener = null) {
    // A disabled button cannot keep the focus: the one that started the action gets it back with the answer.
    const held = opener ?? ([stopBtn, resumeBtn, workersBtn].includes(document.activeElement) ? document.activeElement : null);
    busy[kind] = true;
    renderHead();
    try {
      await request();
      toast(done, { kind: 'ok' });
    } catch (error) {
      toast(`${failed}: ${error.message}`, { kind: 'err' });
    } finally {
      busy[kind] = false;
      if (!destroyed) {
        renderHead();
        const lost = !document.activeElement || document.activeElement === document.body;
        if (held && lost && !held.hidden && !held.disabled && !document.querySelector('dialog[open]')) held.focus({ preventScroll: true });
        fetchDetail();
      }
    }
  }

  function replay() {
    import('./queue.js')
      .then((mod) => mod.openReplayDialog(run))
      .catch((error) => toast(`Replay could not be opened: ${error.message}`, { kind: 'err' }));
  }

  // ── recipe ──

  function recipePart(panel) {
    const empty = h('p', { class: 'runs__pad muted' }, 'The service has no saved recipe for this run.');
    const copyBtn = h('button', { class: 'btn btn--sm', type: 'button', onclick: copy }, 'Copy');
    const openBtn = h('button', { class: 'btn btn--sm btn--primary', type: 'button', onclick: open }, 'Open as recipe');
    const replayRecipeBtn = h('button', { class: 'btn btn--sm', type: 'button', onclick: replay }, 'Replay');
    // The recipe alone is "Copy"; the run file adds the seed, how the run started and where its inputs
    // were, for repeating it or fine-tuning from it on another machine.
    const runFileBtn = h(
      'button',
      { class: 'btn btn--sm', type: 'button', onclick: saveRunFile, title: 'A file that describes this run (recipe, seed, starting point, inputs), to repeat it or fine-tune from it on another machine.' },
      'Save a run file…',
    );
    const preSlot = h('div', { class: 'runs__yaml-wrap' });
    const hint = h('p', { class: 'muted grow' });
    const block = h('div', { class: 'runs__recipe', hidden: true }, h('div', { class: 'runs__toolbar' }, hint, copyBtn, openBtn, replayRecipeBtn, runFileBtn), preSlot);
    panel.append(empty, block);
    let yaml = null;

    async function saveRunFile() {
      try {
        const mod = await import('./runfile.js');
        await mod.openRunFileExport({ run, artifacts: Array.isArray(run.artifacts) ? run.artifacts : [], preselect: null });
      } catch (error) {
        toast(`The run file dialog could not be opened: ${error?.message ?? error}`, { kind: 'err' });
      }
    }

    async function copy() {
      const ok = await copyText(yaml ?? '');
      toast(ok ? 'Copied the recipe.' : 'The recipe was not copied: the browser refused access to the clipboard.', { kind: ok ? 'ok' : 'err' });
    }

    function open() {
      const name = nameOf(run);
      openAsRecipe({ yaml: yaml ?? '', name: run.name ?? '', label: `run ${name}` });
    }

    return {
      update(r) {
        const next = typeof r.yaml === 'string' ? r.yaml : '';
        empty.hidden = Boolean(next);
        block.hidden = !next;
        setText(hint, isImported(r) ? 'The recipe found beside this experiment, read-only.' : 'The recipe this run was launched with, read-only.');
        // A run file is made from an attempt whose trainer has exited.
        runFileBtn.hidden = Boolean(runStatus(r.status).active);
        if (next === yaml) return; // unchanged: keep the node, so a text selection in it survives
        yaml = next;
        const pre = highlightedPre(yaml, 'yaml__pre runs__yaml');
        pre.tabIndex = 0;
        pre.setAttribute('aria-label', 'Recipe of this run');
        mountNodes(preSlot, pre);
      },
      destroy() {
        panel.replaceChildren();
      },
    };
  }

  return {
    id,
    /** The list poll saw this run: a status change while the monitor was idle triggers a poll now. */
    onSummary(next) {
      if (destroyed || !loaded || polling) return;
      if (next && next.status !== run.status) schedulePoll(0);
    },
    tick() {
      if (loaded) renderHead();
    },
    destroy() {
      destroyed = true;
      stopPolling();
      detailCtrl?.abort();
      document.removeEventListener('visibilitychange', onVisibility);
      unsubCatalog();
      for (const { part } of panels.values()) {
        try {
          part?.destroy?.();
        } catch {
          // keep cleaning up the rest
        }
      }
      host.replaceChildren();
    },
  };
}

// ───────── Files tab: choose a checkpoint, then act on it ─────────

const CHECKPOINT_RE = /\.(safetensors|pth)$/i;
const isCheckpoint = (path) => CHECKPOINT_RE.test(String(path ?? ''));
const fileName = (path) => String(path ?? '').split('/').pop();
const EVALUATE_TIP = "Measure this checkpoint on the recipe's validation data, on the CPU.";
const FINETUNE_TIP = 'Prepare a new recipe that starts from these weights with a fresh optimizer and iteration count. Nothing starts.';
const AVERAGE_TIP = 'Make one new weights file from several checkpoints of this run, each counting the same. It is computed on the CPU.';

/** Iteration number in a checkpoint file name (net_g_8.safetensors -> 8), or null (net_g_latest.safetensors). */
function iterationOf(path) {
  const m = /_(\d+)\.(safetensors|pth)$/i.exec(fileName(path));
  return m ? Number(m[1]) : null;
}

/** The files grouped by folder, in the order the tab shows them: known folders first, then the rest by name. */
export function groupFiles(list) {
  const byFolder = new Map();
  for (const a of list) {
    const p = String(a.path ?? '');
    const folder = p.includes('/') ? p.slice(0, p.lastIndexOf('/') + 1) : '';
    if (!byFolder.has(folder)) byFolder.set(folder, []);
    byFolder.get(folder).push(a);
  }
  const order = [...FOLDERS.filter((f) => byFolder.has(f)), ...[...byFolder.keys()].filter((f) => !FOLDERS.includes(f)).sort()];
  return order.map((folder) => ({ folder, items: byFolder.get(folder) }));
}

/**
 * The checkpoint chosen when the person has not chosen one: of the checkpoint paths, in the order the list
 * shows them, the first with the highest iteration number. An unnumbered file (net_g_latest) only when no
 * file is numbered. Raw and EMA weights are not ranked against each other: the list order decides.
 * @param {string[]} paths every file path, in display order
 * @returns {string|null}
 */
export function defaultCheckpoint(paths) {
  let best = null;
  let bestStep = -1;
  for (const p of paths) {
    if (!isCheckpoint(p)) continue;
    const step = iterationOf(p);
    if (step !== null && step > bestStep) {
      best = p;
      bestStep = step;
    }
  }
  return best ?? paths.find(isCheckpoint) ?? null;
}

/** Inference weights of the run's own network (generator or autoencoder), the files a fine-tune can start from. */
// `best-…`: the weights a run kept because they scored best on a validation dataset (val.save_best).
const PRIMARY_WEIGHTS_RE = /^(net_(g|ae)(_ema)?_[^/]+|best-[^/]+)\.(safetensors|pth)$/i;
export const isPrimaryWeights = (path) => /^models\//.test(String(path ?? '')) && PRIMARY_WEIGHTS_RE.test(fileName(path));

/** A numbered safetensors checkpoint of one network role, the kind the service averages (studio/analysis_jobs.py). */
/** The most checkpoints the averaging dialog lists (the service's largest page). */
const AVERAGE_LIST_MAX = 250;
const AVERAGE_RE = /^(net_g_ema|net_g|net_ae_ema|net_ae|net_d)_(\d+)\.safetensors$/i;

/**
 * The checkpoints that can be averaged with this one: the numbered safetensors files of the same network
 * role in the same folder, this one included, by iteration. Empty when the file is not such a checkpoint.
 * @param {string} path
 * @param {string[]} paths every checkpoint path of the run
 * @returns {Array<{step: number, path: string, role: string}>}
 */
export function averageCandidates(path, paths) {
  const own = AVERAGE_RE.exec(fileName(path));
  if (!own) return [];
  const role = own[1].toLowerCase();
  const folder = String(path).slice(0, String(path).lastIndexOf('/') + 1);
  const found = [];
  for (const p of paths) {
    const text = String(p ?? '');
    if (text.slice(0, text.lastIndexOf('/') + 1) !== folder) continue;
    const m = AVERAGE_RE.exec(fileName(text));
    if (m && m[1].toLowerCase() === role) found.push({ step: Number(m[2]), path: text, role });
  }
  return found.sort((a, b) => a.step - b.step);
}

/**
 * The actions the bar offers for one file, in bar order.
 * `actions` is what the service says the run's recipe can do (summary.actions of its check: predict,
 * holdout, export, each present or absent); when it is not known, nothing is held back here and the start
 * request stays the authority.
 * @param {{path: string, root: string|null, analysis: boolean, yaml?: string, task?: string,
 *   actions?: object|null, finetune?: boolean, average?: boolean}} facts `average`: the run is managed here and
 *   holds at least two checkpoints that can be averaged with this one
 * @returns {Array<'inspect'|'drift'|'predict'|'evaluate'|'export'|'finetune'|'average'|'copy'>}
 */
export function fileActions({ path, root, analysis, yaml, task, actions = null, finetune = false, average = false }) {
  const located = Boolean(root && path);
  const safetensors = /\.safetensors$/i.test(String(path ?? ''));
  const hasRecipe = typeof yaml === 'string' && yaml.length > 0;
  // Prediction, evaluation and export build the network from the run's recipe.
  const withRecipe = located && analysis && isCheckpoint(path) && hasRecipe;
  const known = Boolean(actions) && typeof actions === 'object';
  const offers = (name, otherwise = true) => (known ? Boolean(actions[name]) && typeof actions[name] === 'object' : otherwise);
  // What a person does next with a checkpoint comes first (look, measure, continue, ship); the tools that
  // look inside the file follow, drawn quieter.
  const ids = [];
  if (withRecipe && offers('predict')) ids.push('predict');
  if (withRecipe && offers('holdout')) ids.push('evaluate');
  // A new recipe that starts from these weights; it reads the run's recipe, not the folder.
  if (finetune && hasRecipe && isPrimaryWeights(path)) ids.push('finetune');
  if (withRecipe && offers('export')) ids.push('export');
  // One new weights file from several checkpoints of this run; it needs the run, not its folder or recipe.
  if (average && analysis) ids.push('average');
  if (located && safetensors) ids.push('inspect');
  if (located && safetensors && analysis) ids.push('drift');
  if (path) ids.push('copy');
  return ids;
}

/**
 * The Files tab of one run.
 * @param {HTMLElement} panel
 * @param {{getRun: () => object, loadAnalysis?: () => Promise<object>, loadWeights?: () => Promise<object>,
 *   selection?: (id: string, params: object, opts: object) => Promise<object>}} options
 * @returns {{update(run: object): void, destroy(): void}}
 */
export function mountFilesTab(panel, { getRun, loadAnalysis = () => import('./analysis.js'), loadWeights = () => import('../components/weights.js'), loadDrift = () => import('./drift.js'), loadRunfile = () => import('./runfile.js'), loadFrontier = () => import('./frontier.js'), loadStorage = () => import('./storage.js'), check = (yaml, opts) => api.preflight(yaml, opts), selection = (id, params, opts) => api.checkpointSelection(id, params, opts) } = {}) {
  const groupName = uid('runs-file');
  // What the service does not offer for this run's task and network, said once above the list.
  const onnxNote = h('p', { class: 'runs__files-note muted', hidden: true });
  const bestHost = h('div', { class: 'runs__best-host' });
  const barLabel = h('p', { class: 'runs__file-bar-label', 'aria-live': 'polite' });
  const toolbar = h('div', { class: 'runs__file-toolbar', role: 'toolbar', onkeydown: rove });
  const barNote = h('p', { class: 'runs__file-bar-note muted', hidden: true });
  const bar = h('div', { class: 'runs__file-bar', hidden: true }, barLabel, toolbar, barNote);
  const empty = h('p', { class: 'runs__inset muted' }, 'This run has no saved checkpoints or training states yet. Validation images are under Validation.');
  const groupsEl = h('div', { class: 'runs__files' });
  // Selection reads top to bottom: which checkpoint measured best, how two scores trade off, the file
  // chosen and what to do with it, every file, and last the disk space the checkpoints need.
  const frontierHost = h('div', { class: 'runs__frontier-host' });
  // Focusable by script: the run header's disk warning leads here.
  const storageHost = h('div', { class: 'runs__storage-host', tabindex: '-1' });
  panel.append(onnxNote, bestHost, frontierHost, bar, empty, groupsEl, storageHost);

  let last = null;
  let root = null;
  let gone = false;
  let analysis = null; // the analysis part, when it loads; without it the analysis buttons are left out
  let selected = null; // relative path of the chosen checkpoint
  let checkpoints = []; // relative paths of the checkpoint files, in display order
  let listKey = '';
  let barKey = '';
  let drift = null; // runs/drift.js when it loads: it suggests the baseline for "Measure drift…"
  let shownNow = true;
  let filesSig = null; // the file list and the run's progress last seen, to know when to re-read the ranking
  let progressSig = null;
  const rows = new Map(); // path -> {li, radio}

  const run = () => getRun() ?? last ?? {};
  // The trainer root comes with GET /api/status; when it arrives later, the buttons appear.
  const unsubConn = conn.subscribe(() => {
    if (last) part.update(last);
  });
  Promise.resolve()
    .then(() => loadAnalysis())
    .then((mod) => {
      if (gone || typeof mod?.openAnalysisDialog !== 'function') return;
      analysis = mod;
      if (last) part.update(last);
    })
    .catch(() => {}); // a missing part only removes its buttons
  Promise.resolve()
    .then(() => loadDrift())
    .then((mod) => {
      if (gone || typeof mod?.previousCheckpoint !== 'function') return;
      drift = mod;
      if (last) part.update(last);
    })
    .catch(() => {}); // without it, "Measure drift…" opens with no baseline suggested
  let runfile = null; // runs/runfile.js when it loads: "Fine-tune from this…"
  Promise.resolve()
    .then(() => loadRunfile())
    .then((mod) => {
      if (gone || typeof mod?.openFineTuneDialog !== 'function') return;
      runfile = mod;
      if (last) part.update(last);
    })
    .catch(() => {}); // a missing part only removes its button
  // Two parts that read their own answers from the service; one that fails to load is left out.
  let frontier = null; // runs/frontier.js: weighing two scores against each other
  let storage = null; // runs/storage.js: the disk space the checkpoints take and still need
  Promise.resolve()
    .then(() => loadFrontier())
    .then((mod) => {
      if (gone || typeof mod?.mountCheckpointFrontier !== 'function') return;
      frontier = mod.mountCheckpointFrontier(frontierHost, { runId: String(run().id ?? ''), hasFile: (path) => checkpoints.includes(path), onPick: (path) => choose(path, { focus: true }) });
    })
    .catch(() => {});
  Promise.resolve()
    .then(() => loadStorage())
    .then((mod) => {
      if (gone || typeof mod?.mountCheckpointStorage !== 'function') return;
      storage = mod.mountCheckpointStorage(storageHost, { runId: String(run().id ?? ''), getRun: run });
      if (last) storage.update(last);
    })
    .catch(() => {});

  // What the run's recipe can do, as the service resolves it (summary.actions of the recipe check, which
  // changes nothing). Read once per recipe text; until it arrives, and when it cannot be read, the fixed
  // rule in fileActions applies and the start request stays the authority.
  let actionsYaml = null;
  let actions = null;
  let unavailable = null; // the service's own sentence per action it does not offer (summary.unavailableActions)
  let network = '';
  function readActions(r) {
    const yaml = typeof r.yaml === 'string' ? r.yaml : '';
    if (!yaml || yaml === actionsYaml) return;
    actionsYaml = yaml;
    actions = null;
    unavailable = null;
    network = '';
    Promise.resolve()
      .then(() => check(yaml))
      .then((answer) => {
        if (gone || actionsYaml !== yaml) return;
        const found = answer?.summary?.actions;
        actions = found && typeof found === 'object' ? found : null;
        const why = answer?.summary?.unavailableActions;
        unavailable = why && typeof why === 'object' ? why : null;
        network = typeof answer?.summary?.architecture === 'string' ? answer.summary.architecture : '';
        if (last) part.update(last);
      })
      .catch(() => {});
  }

  // File versions as the ranking last reported them: a job started on such a file carries its version, so
  // the service refuses a file that was replaced since it was measured.
  const versions = new Map();
  const best = mountBestCheckpoint(bestHost, {
    runId: String(run().id ?? ''),
    request: selection,
    hasFile: (path) => checkpoints.includes(path),
    onPick: (path) => choose(path, { focus: true }),
    onArtifacts: (list) => {
      versions.clear();
      for (const a of list) if (typeof a?.path === 'string' && typeof a.version === 'string') versions.set(a.path, a.version);
      renderBar();
    },
  });

  /** Absolute experiment folder of the run on the trainer machine, or null when the service did not give it. */
  function experimentRoot(r) {
    const trim = (p) => p.replace(/\/+$/, '');
    if (!r.name) return null;
    if (r.source === 'archive') return typeof r.sourceRoot === 'string' && r.sourceRoot ? `${trim(r.sourceRoot)}/${r.name}` : null;
    if (isImported(r)) {
      const trainer = conn.get().info?.trainerRoot;
      return typeof trainer === 'string' && trainer ? `${trim(trainer)}/experiments/${r.name}` : null;
    }
    // A run started here: its launch record names the run folder, which holds experiments/<name>/.
    const managed = r.invocation?.env?.TRAINNER_MANAGED_RUN_ROOT;
    return typeof managed === 'string' && managed ? `${trim(managed)}/experiments/${r.name}` : null;
  }

  const failed = (error) => toast(`The analysis dialog could not be opened: ${error?.message ?? error}`, { kind: 'err' });
  /** Open the analysis dialog; a refusal, a throw or a rejected promise all end in the same toast. */
  function openAnalysis(params) {
    try {
      Promise.resolve(analysis.openAnalysisDialog({ ...params, sourceLabel: `run ${nameOf(run())}` })).catch(failed);
    } catch (error) {
      failed(error);
    }
  }

  function inspect(abs) {
    Promise.resolve()
      .then(() => loadWeights())
      .then((mod) => mod.openWeightsDialog({ path: abs }))
      .catch((error) => toast(`Inspect structure could not be opened: ${error.message}`, { kind: 'err' }));
  }

  /** A new recipe that starts from the chosen weights. Nothing is saved or started: it opens as a recipe. */
  async function fineTune(path) {
    const r = run();
    try {
      const picked = await runfile.openFineTuneDialog({ run: r, artifact: { path, name: fileName(path), version: versions.get(path) } });
      if (picked && typeof picked.yaml === 'string') openAsRecipe({ yaml: picked.yaml, name: picked.name ?? '', label: picked.label ?? `fine-tune of ${nameOf(r)}` });
    } catch (error) {
      toast(`The fine-tune dialog could not be opened: ${error?.message ?? error}`, { kind: 'err' });
    }
  }

  /** The averaging dialog, with the candidates as they are now (a checkpoint may have appeared since the bar was drawn). */
  let averageAsking = false;
  async function average(path) {
    if (averageAsking) return;
    const r = run();
    const local = averageCandidates(path, checkpoints);
    const own = local.find((c) => c.path === path);
    if (!own) {
      toast('This run no longer has two checkpoints of this kind to average.', { kind: 'err' });
      return;
    }
    // The service names the checkpoints of this network that can be averaged, each with its current
    // version. A service that does not answer that question is given the files this page lists; the
    // start request stays the authority either way.
    let list = local.map((c) => ({ step: c.step, path: c.path, version: versions.get(c.path) ?? null }));
    let more = false;
    averageAsking = true;
    try {
      const answer = await api.averageCandidates(String(r.id ?? ''), own.role, { limit: AVERAGE_LIST_MAX });
      const named = Array.isArray(answer?.checkpoints) ? answer.checkpoints.filter((c) => c && Number.isFinite(c.step)) : null;
      if (named) {
        list = named
          .map((c) => ({ step: c.step, path: typeof c.path === 'string' ? c.path : '', version: typeof c.version === 'string' && c.version ? c.version : null }))
          .sort((a, b) => a.step - b.step);
        more = answer.nextBefore !== null && answer.nextBefore !== undefined;
      }
    } catch {
      // kept: the list from this page
    } finally {
      averageAsking = false;
    }
    if (gone) return;
    if (list.length < 2) {
      toast('This run no longer has two checkpoints of this kind to average.', { kind: 'err' });
      return;
    }
    openAnalysis({
      kind: 'average-checkpoints',
      average: { run: { id: String(r.id ?? ''), name: nameOf(r) }, role: own.role, checkpoints: list, selected: [own.step], more },
    });
  }

  async function copyPath(path) {
    const ok = await copyText(String(path));
    toast(ok ? 'Copied the path.' : 'The path was not copied: the browser refused access to the clipboard.', { kind: ok ? 'ok' : 'err' });
  }

  /** Arrow keys, Home and End move between the toolbar's buttons; Tab leaves it (one tab stop). */
  function rove(event) {
    const buttons = [...toolbar.children].filter((el) => el.tagName === 'BUTTON');
    const i = buttons.indexOf(document.activeElement);
    if (i < 0 || !buttons.length) return;
    const n = buttons.length;
    const j = { ArrowRight: (i + 1) % n, ArrowLeft: (i - 1 + n) % n, Home: 0, End: n - 1 }[event.key];
    if (j === undefined) return;
    event.preventDefault();
    buttons.forEach((b, k) => {
      b.tabIndex = k === j ? 0 : -1;
    });
    buttons[j].focus();
  }

  function choose(path, { focus = false } = {}) {
    if (!checkpoints.includes(path)) return;
    selected = path;
    syncRows();
    renderBar();
    if (focus) {
      const entry = rows.get(path);
      entry?.li.scrollIntoView?.({ block: 'center' });
      entry?.radio.focus();
    }
  }

  function syncRows() {
    for (const [path, entry] of rows) {
      const on = path === selected;
      entry.radio.checked = on;
      entry.li.className = on ? 'runs__file runs__file--choice runs__file--selected' : 'runs__file runs__file--choice';
    }
  }

  function renderBar() {
    const r = run();
    if (!selected) {
      bar.hidden = true;
      return;
    }
    bar.hidden = false;
    const label = fileName(selected);
    // The selection is announced through the live label, which changes only when the selection does.
    if (barLabel.dataset.file !== selected) {
      barLabel.dataset.file = selected;
      mountNodes(barLabel, 'With ', h('span', { class: 'mono' }, label), ':');
    }
    // A run file is made from an attempt whose trainer has exited, so a training run does not offer it.
    const exited = !runStatus(r.status).active;
    // Averaging is for a run started here (the service resolves the files from the run), with two or more
    // numbered safetensors checkpoints of the chosen file's network.
    const canAverage = !isImported(r) && averageCandidates(selected, checkpoints).length >= 2;
    const ids = fileActions({ path: selected, root, analysis: Boolean(analysis), yaml: r.yaml, task: r.task, actions, finetune: Boolean(runfile) && exited, average: canAverage });
    const located = Boolean(root);
    const noRecipe = located && Boolean(analysis) && !(typeof r.yaml === 'string' && r.yaml);
    const version = versions.get(selected) ?? null;
    const key = `${selected}|${root}|${ids.join(',')}|${noRecipe}|${Boolean(drift)}|${version}`;
    // Never rebuilt while one of its buttons has focus; a later update catches up.
    if (key === barKey || toolbar.contains(document.activeElement)) return;
    barKey = key;
    toolbar.setAttribute('aria-label', `Actions for ${label}`);
    const abs = located ? `${root}/${selected}` : null;
    const previous = ids.includes('drift') && drift ? drift.previousCheckpoint(selected, checkpoints) : null;
    const button = (text, onclick, extra = {}) => h('button', { class: 'btn btn--sm', type: 'button', onclick, ...extra }, text);
    const quiet = { class: 'btn btn--sm btn--quiet' };
    const make = {
      inspect: () => button('Inspect structure', () => inspect(abs), quiet),
      drift: () =>
        button('Measure drift…', () => openAnalysis({ kind: 'weight-drift', right: abs, left: previous ? `${root}/${previous}` : undefined }), {
          ...quiet,
          title: previous ? `How far the stored values moved since ${fileName(previous)}.` : 'How far the stored values moved from another checkpoint.',
        }),
      predict: () => button('Try on an image', () => openAnalysis({ kind: 'predict', yaml: run().yaml, weights: abs, weightsVersion: version })),
      evaluate: () => button('Evaluate…', () => openAnalysis({ kind: 'evaluate', yaml: run().yaml, weights: abs, weightsVersion: version }), { title: EVALUATE_TIP }),
      export: () => button('Export ONNX', () => openAnalysis({ kind: 'export', yaml: run().yaml, weights: abs, weightsVersion: version })),
      finetune: () => button('Fine-tune from this…', () => fineTune(selected), { title: FINETUNE_TIP }),
      average: () => button('Average with others…', () => average(selected), { title: AVERAGE_TIP }),
      copy: () => h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => copyPath(selected) }, 'Copy path'),
    };
    const buttons = ids.map((id) => make[id]());
    buttons.forEach((b, k) => {
      b.dataset.action = ids[k];
      b.tabIndex = k === 0 ? 0 : -1;
    });
    mountNodes(toolbar, buttons);
    const note = !located
      ? "The structure and the analysis jobs are not offered: the service did not report this run's folder."
      : noRecipe && isCheckpoint(selected)
        ? "Prediction, evaluation and export need this run's recipe, which was not found with it."
        : '';
    barNote.hidden = !note;
    setText(barNote, note);
  }

  function renderList(r, groups) {
    const list = groups.flatMap((g) => g.items);
    const next = `${r.id}|${list.map((a) => `${a.name}\u0000${a.path}\u0000${a.url}`).join('\n')}`;
    // Rebuilt only when the list changed, and never while a link, radio or button in it has focus.
    if (next === listKey || groupsEl.contains(document.activeElement)) return;
    listKey = next;
    rows.clear();
    const row = (a) => {
      const name = a.name ?? a.path ?? '';
      const path = String(a.path ?? '');
      const label = fileName(name);
      const link = h('a', { class: 'runs__file-link mono', href: artifactUrl(r.id, a), download: label }, name);
      if (isCheckpoint(path)) {
        // The whole row chooses the file; the name inside stays a download link.
        const radio = h('input', { class: 'runs__file-radio', type: 'radio', name: groupName, value: path, onchange: () => radio.checked && choose(path) });
        const li = h('li', { class: 'runs__file runs__file--choice' }, h('label', { class: 'runs__file-choice' }, radio, link));
        rows.set(path, { li, radio });
        return li;
      }
      const copy = a.path ? h('div', { class: 'runs__file-actions', role: 'group', 'aria-label': `Actions for ${label}` }, h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => copyPath(a.path) }, 'Copy path')) : null;
      return h('li', { class: 'runs__file' }, link, copy);
    };
    mountNodes(
      groupsEl,
      groups.map(({ folder, items }) =>
        h('section', { class: 'runs__file-group' }, h('h3', { class: 'runs__section-title mono' }, folder || 'Experiment folder'), h('ul', { class: 'runs__file-list' }, items.map(row))),
      ),
    );
  }

  const part = {
    update(r) {
      last = r;
      const list = Array.isArray(r.artifacts) ? r.artifacts.filter((a) => a && typeof a === 'object' && (a.path || a.url)) : [];
      root = experimentRoot(r);
      const groups = groupFiles(list);
      const ordered = groups.flatMap((g) => g.items.map((a) => String(a.path ?? '')));
      checkpoints = ordered.filter(isCheckpoint);
      // The choice survives refreshes while its file is still listed.
      if (!selected || !checkpoints.includes(selected)) selected = defaultCheckpoint(ordered);
      empty.hidden = list.length > 0;
      readActions(r);
      // Said once, in words: which of the usual actions the service does not offer for this run, with the
      // service's own reason when it gives one. Nothing is assumed before its answer arrives.
      const reason = (name) => (typeof unavailable?.[name] === 'string' ? unavailable[name].trim() : '');
      const whose = `this run's network${network ? ` (${network})` : ''}`;
      // "Offered" as fileActions reads it: the action is listed with its description.
      const missing = (name, label, hint = '') => {
        if (!actions || (actions[name] && typeof actions[name] === 'object')) return '';
        const why = reason(name);
        return why ? `${label} is not offered for this run: ${/[.!?]$/.test(why) ? why : `${why}.`}` : `${label} is not offered for ${whose}.${hint}`;
      };
      const noPredict = missing('predict', 'Prediction');
      const note = !checkpoints.length || !(typeof r.yaml === 'string' && r.yaml)
        ? ''
        : [noPredict, missing('holdout', 'Evaluation'), missing('export', 'ONNX export', noPredict || !root || !analysis ? '' : ' Prediction on the CPU is.')].filter(Boolean).join(' ');
      onnxNote.hidden = !note;
      setText(onnxNote, note);
      renderList(r, groups);
      syncRows();
      renderBar();
      best.filesChanged();
      storage?.update(r);
      // The ranking is computed afresh by the service: read it again when a checkpoint appears or goes
      // (a "missing" weight can become "available"), and, at most every 15 s, when the run moves on.
      const nextFiles = ordered.join('\n');
      const nextProgress = `${r.status}|${r.iteration}`;
      if (filesSig !== null && nextFiles !== filesSig) {
        best.reread();
        frontier?.refresh();
      } else if (progressSig !== null && nextProgress !== progressSig && shownNow) best.reread({ minGap: 15000 });
      filesSig = nextFiles;
      progressSig = nextProgress;
    },
    /** The tab shown again: the ranking may have changed while it was hidden. */
    setShown(on) {
      if (on && !shownNow) {
        best.reread();
        frontier?.refresh();
      }
      shownNow = Boolean(on);
    },
    /** The run's Refresh. */
    refresh() {
      best.reread();
      frontier?.refresh();
    },
    destroy() {
      gone = true;
      unsubConn();
      best.destroy();
      frontier?.destroy();
      storage?.destroy();
      panel.replaceChildren();
    },
  };
  return part;
}

// ───────── "Which checkpoint measured best?" (GET /api/run/checkpoint-selection) ─────────

/** Why a measured score could not be tied to its weights file (studio/checkpoint_selection.py), in words. */
export const CHECKPOINT_REASONS = {
  repeatedStep: 'this iteration was recorded more than once (for example after a resume), so it is not certain which save it belongs to',
  saveHistoryUnindexed: 'the training log could not be read in full, so the saves it records could not all be checked',
  noSingleCompletedTrajectory: 'the log does not show one uninterrupted training from start to finish (it was resumed or stopped, or its log is missing)',
  inferenceWeightsUnknown: 'the service cannot tell which of the saved weights this run uses for inference',
  saveFailed: 'the trainer reported that saving at this iteration failed',
  numberedSaveUnrecorded: 'the log does not record a numbered save at this iteration',
  weightsChanged: 'the weights file changed while it was being checked',
  ambiguousWeightFormat: 'more than one weights file carries this iteration number (for example .pth and .safetensors)',
  // A measurement that recorded which save it used (val.record_samples), and what it found instead.
  weightIdentityMismatch: 'the file now at this place is not the save this measurement recorded (it was replaced, for example by a later attempt)',
  weightIdentityUnassociated: 'the file carries a save ID, but this measurement has no record to compare it with',
  invalidWeightAssociation: 'the record that ties this measurement to a file is not usable',
  numberedSaveUnmatched: 'no numbered save was recorded for this measurement',
  stepNotNumbered: 'this measurement was not taken at an iteration with a numbered save',
  // What the trainer itself noted while measuring (traiNNer/training/checkpoint_association.py).
  inferenceWeightsUnobserved: 'the trainer did not observe which weights were measured',
  inferenceWeightsChanged: 'the weights changed while they were being measured',
  inferenceRoleUnknown: 'the trainer could not tell which of its networks was measured',
  optimizedInferenceWeightsUnproven: 'the measurement ran an optimized copy of the network, whose weights could not be shown to equal the saved ones',
  optimizerWeightTransform: 'the optimizer switches the weights between training and evaluation, so the saved file may not hold the measured values',
  inferenceModeTransform: 'the network is transformed for evaluation, so the saved file may not hold the measured values',
  exportedStateDiffers: 'the saved weights were not the same set as the measured ones',
  inferenceStateUntracked: 'the trainer could not track the measured weights',
  inferenceStateNotSerializable: 'the measured weights could not be compared with a saved file',
};
/** How an available file was tied to its score: the stronger and the weaker evidence, in words. */
const EVIDENCE_WORDS = {
  recordedSaveId: 'Matched by the save ID this measurement recorded.',
  retainedValidationWeights: 'The run kept these weights because they scored best: the file holds exactly what was measured, also between two numbered saves.',
  uniqueLoggedSave: 'Matched through the training log, the weaker evidence: a file replaced later would go unnoticed.',
};
const SOURCE_WORDS = { summary: 'measured summary', sampleMean: 'mean of per-image scores', log: 'from the log text' };
const NONFINITE_WORDS = { nan: 'NaN', inf: 'infinite', '-inf': 'negative infinite' };

/** A score as the service gave it; a null with a non-finite marker is NaN or infinite, never 0. */
export function scoreText(point) {
  if (typeof point?.value === 'number' && Number.isFinite(point.value)) return String(point.value);
  return NONFINITE_WORDS[point?.nonfinite] ?? 'Not reported';
}

/** One sentence (or '') about what is known of a point's weights; `pick` makes the file name a button. */
function weightsWords(checkpoint, pick) {
  const c = checkpoint && typeof checkpoint === 'object' ? checkpoint : {};
  if (c.status === 'available') {
    const name = c.artifact?.name ?? fileName(c.artifact?.path);
    const how = EVIDENCE_WORDS[c.evidence];
    return ['Its weights are ', pick && c.artifact?.path ? pick(c.artifact.path, name) : h('span', { class: 'mono' }, name || 'not named'), '.', how ? ` ${how}` : ''];
  }
  if (c.status === 'missing') return 'Its weights are no longer in the run folder.';
  if (c.status === 'invalid') return 'Its weights file is damaged or incomplete.';
  if (c.status === 'unverified') {
    const why = CHECKPOINT_REASONS[c.reason];
    return `Its weights could not be matched with certainty${why ? `: ${why}` : ''}.`;
  }
  return '';
}

/** Short words for the weights column of the observations table. */
function weightsCell(checkpoint) {
  const c = checkpoint && typeof checkpoint === 'object' ? checkpoint : {};
  if (c.status === 'available') return h('span', { class: 'mono' }, c.artifact?.name ?? fileName(c.artifact?.path));
  if (c.status === 'missing') return 'No longer in the run folder';
  if (c.status === 'invalid') return 'Damaged or incomplete';
  if (c.status === 'unverified') return h('span', { title: CHECKPOINT_REASONS[c.reason] ?? null }, 'Not matched with certainty');
  return '';
}

const samePoint = (a, b) =>
  Boolean(a && b) && a.step === b.step && scoreText(a) === scoreText(b) && (a.attempt ?? null) === (b.attempt ?? null) && (a.log ?? null) === (b.log ?? null) && (a.line ?? null) === (b.line ?? null);

/**
 * "Which checkpoint measured best?": the run's recorded series, and for the chosen one the best observed score
 * and the best checkpoint that is still there, as two answers. Never polled: the host calls reread() when the
 * answer may have changed (Refresh, the tab shown again, a new iteration or status, a changed file list).
 * A re-read keeps the chosen series and direction and leaves the block alone when the answer is the same.
 * @param {HTMLElement} host
 * @param {{runId: string, request?: Function, hasFile?: (path: string) => boolean, onPick?: (path: string) => void}} options
 * @returns {{reread(options?: {minGap?: number}): void, filesChanged(): void, destroy(): void}}
 */
export function mountBestCheckpoint(host, { runId, request = (id, params, opts) => api.checkpointSelection(id, params, opts), hasFile = () => true, onPick = () => {}, onArtifacts = () => {} } = {}) {
  const box = h('section', { class: 'runs__best', 'aria-label': 'Which checkpoint measured best?' });
  host.appendChild(box);
  const body = h('div', { class: 'runs__best-body', 'aria-live': 'polite', tabindex: '-1' });
  const selectId = uid('runs-best-series');
  let series = [];
  let seriesSig = null; // JSON of the series list on screen; null when the head is not built
  let chosenKey = null; // dataset + metric of the chosen series, kept across re-reads
  let direction = null; // a direction the person chose for a metric the service could not orient
  let answer = null;
  let answerSig = '';
  let boxState = ''; // 'loading' | 'empty' | 'error' | 'series'
  let detailsOpen = false;
  let pickKey = '';
  let ctrl = null;
  let lastRead = 0;
  let trailing = 0;
  let gone = false;

  const keyOf = (s) => `${s.dataset}\u0000${s.metric}`;
  const chosen = () => series.find((s) => keyOf(s) === chosenKey) ?? null;
  const abort = () => {
    ctrl?.abort();
    ctrl = new AbortController();
    return ctrl;
  };
  const isAbort = (error, c) => gone || c !== ctrl || error?.name === 'AbortError';
  const retryButton = (onclick) => h('button', { class: 'btn btn--sm', type: 'button', dataset: { focus: 'retry' }, onclick }, 'Try again');

  /** Keep keyboard focus on "the same" control after `area` is rebuilt (matched by data-focus). */
  function keepFocus(area, rebuild, wanted = null) {
    const active = typeof document !== 'undefined' ? document.activeElement : null;
    const had = Boolean(active && area.contains(active));
    const key = wanted ?? (had ? active.dataset?.focus : null);
    rebuild();
    if (!key && !had) return;
    const escaped = key && typeof CSS !== 'undefined' && CSS.escape ? CSS.escape(key) : String(key ?? '').replace(/"/g, '\\"');
    const target = key ? area.querySelector(`[data-focus="${escaped}"]`) : null;
    (target ?? (had ? body : null))?.focus?.();
  }

  function showBox(state, build) {
    if (boxState === state && state !== 'series') return; // the same empty or error line stays as it is
    boxState = state;
    keepFocus(box, () => mountNodes(box, build()));
  }

  function loadSeries({ quiet = false } = {}) {
    const c = abort();
    lastRead = Date.now();
    if (!quiet || !boxState) {
      seriesSig = null;
      boxState = 'loading';
      mountNodes(box, loading('Reading the recorded validation scores…'));
    }
    Promise.resolve()
      .then(() => request(runId, {}, { signal: c.signal }))
      .then((res) => {
        if (gone || c !== ctrl) return;
        series = Array.isArray(res?.series) ? res.series.filter((s) => s && typeof s.dataset === 'string' && typeof s.metric === 'string') : [];
        if (!series.length) {
          seriesSig = null;
          answer = null;
          answerSig = '';
          showBox('empty', () =>
            h('div', { class: 'runs__best-error' }, h('p', { class: 'runs__best-empty muted' }, 'This run has no recorded validation scores, so no checkpoint can be ranked.'), retryButton(() => loadSeries())),
          );
          return;
        }
        if (!chosen()) {
          chosenKey = keyOf(series[0]);
          direction = null;
        }
        const sig = JSON.stringify(series.map(keyOf));
        if (sig !== seriesSig || boxState !== 'series') {
          seriesSig = sig;
          boxState = 'series';
          const select = h(
            'select',
            {
              class: 'select runs__best-select',
              id: selectId,
              onchange: () => {
                chosenKey = keyOf(series[Number(select.value)]);
                direction = null;
                ask();
              },
            },
            series.map((s, i) => h('option', { value: String(i), selected: keyOf(s) === chosenKey }, `${s.dataset}: ${s.metric}`)),
          );
          keepFocus(box, () =>
            mountNodes(box, h('div', { class: 'runs__best-head' }, h('h3', { class: 'runs__best-title' }, 'Which checkpoint measured best?'), h('label', { class: 'runs__best-label', for: selectId }, 'Validation score'), select), body),
          );
        }
        ask({ quiet: quiet && Boolean(answerSig), keepCtrl: c });
      })
      .catch((error) => {
        if (isAbort(error, c)) return;
        // A background re-read that fails leaves what is on screen; a requested read says what happened.
        if (quiet && boxState === 'series' && answerSig) return;
        seriesSig = null;
        boxState = '';
        showBox('error', () => h('div', { class: 'runs__best-error' }, h('p', { class: 'runs__best-note' }, `The recorded validation scores could not be read: ${error.message}`), retryButton(() => loadSeries())));
      });
  }

  function ask({ quiet = false, focus = null, keepCtrl = null } = {}) {
    const c = keepCtrl ?? abort();
    const s = chosen();
    if (!s) return;
    const params = { dataset: s.dataset, metric: s.metric };
    if (direction) params.direction = direction;
    if (!quiet) {
      // A choice made with a button keeps the button on screen until the answer replaces it, so focus is not lost.
      if (focus) body.setAttribute('aria-busy', 'true');
      else {
        answer = null;
        answerSig = '';
        pickKey = '';
        mountNodes(body, loading('Ranking the recorded scores…'));
      }
    }
    Promise.resolve()
      .then(() => request(runId, params, { signal: c.signal }))
      .then((res) => {
        if (gone || c !== ctrl) return;
        body.removeAttribute('aria-busy');
        const next = res && typeof res === 'object' ? res : {};
        const sig = JSON.stringify(next);
        if (sig === answerSig) return; // unchanged: nothing is rebuilt
        answer = next;
        answerSig = sig;
        keepFocus(body, renderAnswer, focus);
      })
      .catch((error) => {
        if (isAbort(error, c)) return;
        body.removeAttribute('aria-busy');
        if (quiet && answerSig) return;
        // A refusal (for example a direction opposite to the recorded one) is shown in the service's words.
        const refused = error?.status === 400 || error?.status === 422;
        const chose = Boolean(direction);
        answer = null;
        answerSig = '';
        keepFocus(
          body,
          () =>
            mountNodes(
              body,
              h(
                'div',
                { class: 'runs__best-error' },
                h('p', { class: 'runs__best-note' }, refused ? `The service refused: ${error.message}` : `The scores could not be ranked: ${error.message}`),
                h(
                  'button',
                  {
                    class: 'btn btn--sm',
                    type: 'button',
                    dataset: { focus: 'retry' },
                    onclick: () => {
                      if (refused) direction = null;
                      ask();
                    },
                  },
                  refused && chose ? 'Ask without a direction' : 'Try again',
                ),
              ),
            ),
          focus ? 'retry' : null,
        );
      });
  }

  /** The key of what the answer shows that depends on the file list (which picks are buttons). */
  function currentPickKey() {
    const paths = [answer?.bestObserved, answer?.bestAvailable].map((p) => (p?.checkpoint?.status === 'available' ? p.checkpoint.artifact?.path : null)).filter(Boolean);
    return paths.map((p) => `${p}:${hasFile(p)}`).join('|');
  }

  function renderAnswer() {
    const a = answer ?? {};
    pickKey = currentPickKey();
    // Every file the answer ties to a score, with its version, for whoever starts a job on one of them.
    onArtifacts(
      [a.bestObserved, a.bestAvailable, ...(Array.isArray(a.points) ? a.points : [])]
        .map((p) => (p?.checkpoint?.status === 'available' ? p.checkpoint.artifact : null))
        .filter((artifact) => artifact && typeof artifact === 'object'),
    );
    const dir = a.direction === 'higher' || a.direction === 'lower' ? a.direction : null;
    const pick = (path, name) =>
      hasFile(path)
        ? h('button', { class: 'runs__best-pick', type: 'button', title: 'Select this file in the list below', dataset: { focus: `pick:${path}` }, onclick: () => onPick(path) }, name)
        : h('span', { class: 'mono' }, name);
    const dirWords = dir ? `${dir} is better${a.directionSource === 'explicit' ? ', as you chose' : ''}` : '';
    // A run measured by more than one attempt (it was resumed, or replayed in place) can hold the same
    // iteration twice. Only then does an observation name its attempt (by the start of the id the service
    // gave it): an iteration measured once needs no such mark.
    const points = Array.isArray(a.points) ? a.points.filter((p) => p && typeof p === 'object') : [];
    const attemptOf = (p) => (typeof p?.attempt === 'string' && p.attempt ? p.attempt : null);
    const attemptsAt = new Map(); // iteration -> the attempts that measured it
    for (const p of [a.bestObserved, a.bestAvailable, ...points]) {
      if (!p || !attemptOf(p)) continue;
      if (!attemptsAt.has(p.step)) attemptsAt.set(p.step, new Set());
      attemptsAt.get(p.step).add(attemptOf(p));
    }
    const repeated = (p) => (attemptsAt.get(p?.step)?.size ?? 0) > 1;
    const severalAttempts = [...attemptsAt.values()].some((set) => set.size > 1);
    const attemptShort = (p) => (attemptOf(p) ? attemptOf(p).slice(0, 8) : '');
    const where = (point) => `Iteration ${fmt.int(point.step)}${repeated(point) && attemptOf(point) ? `, attempt ${attemptShort(point)}` : ''}`;
    const line = (point) => [`${where(point)}: `, h('span', { class: 'runs__best-value' }, scoreText(point)), ` (${dirWords}). `, weightsWords(point.checkpoint, pick)];
    const tie = (point) => (Number.isFinite(point?.tieCount) && point.tieCount > 1 ? h('dd', { class: 'runs__best-tie' }, `${fmt.int(point.tieCount)} observations share this score; the earliest is shown.`) : null);

    const parts = [];
    if (!dir) {
      if (a.directionSource === 'conflicting') {
        parts.push(h('p', { class: 'runs__best-line' }, 'The service does not know whether higher or lower is better for this metric: its records disagree, so it does not rank it.'));
      } else {
        const choose = (d) => () => {
          direction = d;
          ask({ focus: `dir:${d}` });
        };
        parts.push(
          h('p', { class: 'runs__best-line' }, 'The service does not know whether higher or lower is better for this metric.'),
          h(
            'div',
            { class: 'runs__best-choice', role: 'group', 'aria-label': 'Which direction is better' },
            h('button', { class: 'btn btn--sm', type: 'button', dataset: { focus: 'dir:higher' }, onclick: choose('higher') }, 'Higher is better'),
            h('button', { class: 'btn btn--sm', type: 'button', dataset: { focus: 'dir:lower' }, onclick: choose('lower') }, 'Lower is better'),
          ),
        );
      }
    } else {
      const bo = a.bestObserved && typeof a.bestObserved === 'object' ? a.bestObserved : null;
      const ba = a.bestAvailable && typeof a.bestAvailable === 'object' ? a.bestAvailable : null;
      const answers = [];
      if (bo && ba && samePoint(bo, ba)) {
        answers.push(h('div', { class: 'runs__best-answer' }, h('dt', null, 'Best score observed, also the best checkpoint still available'), h('dd', null, line(bo)), tie(bo)));
      } else {
        answers.push(
          h('div', { class: 'runs__best-answer' }, h('dt', null, 'Best score observed'), h('dd', null, bo ? line(bo) : 'None: no observation has a score that can be ranked.'), tie(bo)),
          h('div', { class: 'runs__best-answer' }, h('dt', null, 'Best checkpoint still available'), h('dd', null, ba ? line(ba) : 'None: no numbered checkpoint could be matched to a measured score.'), tie(ba)),
        );
      }
      parts.push(h('dl', { class: 'runs__best-answers' }, answers));
    }

    // Coverage, in small print: what the answer rests on.
    const small = [];
    const count = (n) => (Number.isFinite(n) ? n : 0);
    small.push(`${fmt.plural(count(a.observedCount), 'measured observation')}, ${fmt.int(count(a.rankableCount))} rankable, ${fmt.int(count(a.availableCount))} with weights that could be matched.`);
    const history = a.history && typeof a.history === 'object' ? a.history : {};
    if (history.logCoverage === 'partial') {
      const logs = Number.isFinite(history.unindexedLogs) && history.unindexedLogs > 0 ? `${fmt.plural(history.unindexedLogs, 'training log')} could not be searched in full` : 'Some older log text was not searched';
      small.push(history.logPurpose === 'saveEvidence' ? `${logs}, so the saves recorded there could not be checked against these scores.` : `${logs}, so scores printed only there may be missing here.`);
    }
    if (Number.isFinite(a.unplacedLogCount) && a.unplacedLogCount > 0) {
      small.push(`${fmt.plural(a.unplacedLogCount, 'score')} in the log text could not be placed at an iteration and ${a.unplacedLogCount === 1 ? 'is' : 'are'} not ranked.`);
    }
    // Nothing could be matched, and only through the log (no measurement here recorded a save of its own):
    // say what makes the match certain in a later run.
    const LOG_REASONS = ['repeatedStep', 'saveHistoryUnindexed', 'noSingleCompletedTrajectory', 'numberedSaveUnrecorded'];
    if (count(a.observedCount) > 0 && count(a.availableCount) === 0 && points.length && points.every((p) => LOG_REASONS.includes(p.checkpoint?.reason))) {
      small.push('A run whose recipe turns on val.record_samples records, with each measurement, the save it used. The match then does not depend on the log.');
    }
    parts.push(h('div', { class: 'runs__best-small' }, small.map((text) => h('p', { class: 'runs__best-note' }, text))));

    if (points.length) {
      const shown = Number.isFinite(a.observedCount) && a.observedCount > points.length ? h('p', { class: 'runs__best-note' }, `The first ${fmt.int(points.length)} of ${fmt.int(a.observedCount)} are listed.`) : null;
      const details = h(
        'details',
        { class: 'runs__best-details', open: detailsOpen },
        h('summary', { class: 'runs__best-summary', dataset: { focus: 'details' } }, dir ? 'All ranked observations' : 'All observations (not ranked)'),
        shown,
        h(
          'div',
          { class: 'runs__best-table-wrap', tabindex: '0', role: 'region', dataset: { focus: 'table' }, 'aria-label': dir ? 'All ranked observations' : 'All observations' },
          h(
            'table',
            { class: 'table runs__best-table' },
            h(
              'thead',
              null,
              h(
                'tr',
                null,
                h('th', { scope: 'col', class: 'num' }, 'Iteration'),
                severalAttempts ? h('th', { scope: 'col' }, 'Attempt') : null,
                h('th', { scope: 'col', class: 'num' }, 'Value'),
                h('th', { scope: 'col' }, 'Source'),
                h('th', { scope: 'col' }, 'Weights'),
              ),
            ),
            h(
              'tbody',
              null,
              points.map((p) =>
                h(
                  'tr',
                  null,
                  h('td', { class: 'num' }, fmt.int(p.step)),
                  severalAttempts ? h('td', { class: 'mono', title: attemptOf(p) }, attemptShort(p) || 'Not recorded') : null,
                  h('td', { class: 'num' }, scoreText(p)),
                  h('td', null, SOURCE_WORDS[p.source] ?? ''),
                  h('td', null, weightsCell(p.checkpoint)),
                ),
              ),
            ),
          ),
        ),
      );
      // Its open state survives every re-render.
      details.addEventListener('toggle', () => {
        detailsOpen = Boolean(details.open);
      });
      parts.push(details);
    }
    mountNodes(body, parts);
  }

  loadSeries();

  return {
    /** Read the series and the chosen answer again; `minGap` (ms) throttles background reasons. */
    reread({ minGap = 0 } = {}) {
      if (gone) return;
      const wait = lastRead + minGap - Date.now();
      if (minGap > 0 && wait > 0) {
        if (!trailing) {
          trailing = setTimeout(() => {
            trailing = 0;
            if (!gone) loadSeries({ quiet: true });
          }, wait);
        }
        return;
      }
      clearTimeout(trailing);
      trailing = 0;
      loadSeries({ quiet: true });
    },
    /** The file list changed: a pick becomes a button (or plain text) when its file appears (or goes). */
    filesChanged() {
      if (!answer || gone || currentPickKey() === pickKey) return;
      keepFocus(body, renderAnswer);
    },
    destroy() {
      gone = true;
      clearTimeout(trailing);
      ctrl?.abort();
      box.remove();
    },
  };
}
