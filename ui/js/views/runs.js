// Runs screen frame: the queue block and the run list on the left, and by route one of
//   #/runs                          the selected or newest run (or the empty state)
//   #/runs/<runId>                  one run (runs/detail.js)
//   #/runs/queue[/<entryId>]        the queue (runs/queue.js, loaded on demand)
//   #/runs/compare/<left>[/<right>] two runs or evaluation jobs side by side; a side is a run id or
//                                   job:<id> (runs/evaluation.js, loaded on demand)
//   #/runs/analysis[/<jobId>]       analysis jobs (runs/analysis.js, loaded on demand)
//   #/runs/studies[/<studyId>]      studies (runs/studies.js, loaded on demand)
//   #/runs/curricula[/<id>]         curricula, training in phases (runs/curricula.js, loaded on demand)
// Parts written elsewhere are loaded with import(); a missing or failing part shows a notice in its
// own host and never takes the screen down. Background polls update rows in place.

import { h, mount as mountNodes, fmt } from '../dom.js';
import { runs, runStatus, isImported, runSourceLabel } from '../state/runs.js';
import { runHref, compareHref, queueHref, go } from '../actions.js';
import { badge, emptyState, notice, loading } from '../ui.js';
import { createRunDetail, progressBar } from '../runs/detail.js';
import { mountArchives } from '../runs/archives.js';

let view = null;

/**
 * @param {HTMLElement} el
 * @param {{param?: string|null}} [options] the text after `#/runs/`, URL-decoded
 * @returns {() => void} cleanup
 */
export function mount(el, { param } = {}) {
  view = createView(el, param ?? '');
  return () => {
    view?.destroy();
    view = null;
  };
}

/** Same screen, another route: switch the main area without remounting the list. */
export function setParam(param) {
  view?.setParam(param ?? '');
}

/** @returns {{kind: 'auto'} | {kind: 'run', id: string} | {kind: 'queue', entryId: string|null} | {kind: 'compare', leftId: string|null, rightId: string|null} | {kind: 'analysis', jobId: string|null} | {kind: 'studies', studyId: string|null} | {kind: 'curricula', curriculumId: string|null}} */
export function parseRoute(param) {
  const p = String(param ?? '').replace(/^\/+|\/+$/g, '');
  if (!p) return { kind: 'auto' };
  const parts = p.split('/');
  if (parts[0] === 'queue') return { kind: 'queue', entryId: parts.length > 1 && parts[1] ? parts.slice(1).join('/') : null };
  if (parts[0] === 'compare') return { kind: 'compare', leftId: parts[1] || null, rightId: parts[2] || null };
  if (parts[0] === 'analysis') return { kind: 'analysis', jobId: parts.length > 1 && parts[1] ? parts.slice(1).join('/') : null };
  if (parts[0] === 'studies') return { kind: 'studies', studyId: parts.length > 1 && parts[1] ? parts.slice(1).join('/') : null };
  if (parts[0] === 'curricula') return { kind: 'curricula', curriculumId: parts.length > 1 && parts[1] ? parts.slice(1).join('/') : null };
  return { kind: 'run', id: p };
}

const hasProgress = (run) => Number.isFinite(run?.iteration) && Number.isFinite(run?.total);
const setText = (node, text) => {
  if (node.textContent !== text) node.textContent = text;
};
const time = (iso) => {
  const t = iso ? new Date(iso).getTime() : NaN;
  return Number.isNaN(t) ? null : t;
};

/** Active first, then newest `updatedAt`, then newest `startedAt`; API order breaks ties. */
function orderRuns(list) {
  return list
    .map((run, index) => ({ run, index }))
    .sort((a, b) => {
      const act = Number(runStatus(b.run.status).active) - Number(runStatus(a.run.status).active);
      if (act) return act;
      for (const key of ['updatedAt', 'startedAt']) {
        const ta = time(a.run[key]);
        const tb = time(b.run[key]);
        if (ta !== null && tb !== null && ta !== tb) return tb - ta;
        if ((ta === null) !== (tb === null)) return ta === null ? 1 : -1;
      }
      return a.index - b.index;
    })
    .map((item) => item.run);
}

/** Put `node` at position `index` of `parent` only when it is not already there (moves drop focus and selection). */
function placeAt(parent, node, index) {
  if (parent.children[index] !== node) parent.insertBefore(node, parent.children[index] ?? null);
}

/** Replace the children of `slot` only when `key` changed, so a re-render does not drop focus or re-announce. */
function mountKeyed(slot, key, build) {
  if (slot.dataset.key === key) return;
  slot.dataset.key = key;
  mountNodes(slot, build());
}

function createView(el, initialParam) {
  let route = parseRoute(initialParam);
  let filter = 'all';
  let main = null; // {kind, id?, detail?, handle?, host, gone, args}
  let destroyed = false;
  const releaseWatch = runs.watch();

  // ── left column ──
  const queueHost = h('div', { class: 'runs__queue' });
  const archivesHost = h('div', { class: 'runs__archives-host' });
  const studiesLink = h('a', { class: 'runs__nav-link', href: '#/runs/studies' }, 'Studies');
  const curriculaLink = h('a', { class: 'runs__nav-link', href: '#/runs/curricula' }, 'Curricula');
  const analysisLink = h('a', { class: 'runs__nav-link', href: '#/runs/analysis' }, 'Analysis jobs');
  const allBtn = h('button', { class: 'btn btn--sm', type: 'button', 'aria-pressed': 'true', onclick: () => setFilter('all') }, 'All');
  const activeBtn = h('button', { class: 'btn btn--sm', type: 'button', 'aria-pressed': 'false', onclick: () => setFilter('active') }, 'Active');
  const listStatus = h('div', { class: 'runs__list-status', role: 'status' });
  const listEl = h('ul', { class: 'runs__items', 'aria-label': 'Training runs' });
  const listPane = h(
    'aside',
    { class: 'pane runs__list', 'aria-label': 'Queue and runs' },
    queueHost,
    h('nav', { class: 'runs__nav', 'aria-label': 'Studies, curricula and analysis' }, studiesLink, curriculaLink, analysisLink),
    h('div', { class: 'pane__head' }, h('h1', { class: 'pane__title runs__h1' }, 'Runs'), h('div', { class: 'seg', role: 'group', 'aria-label': 'Show runs' }, allBtn, activeBtn)),
    h('div', { class: 'runs__list-body' }, listStatus, listEl),
    archivesHost,
  );
  const mainEl = h('section', { class: 'runs__detail', 'aria-label': 'Run detail' });
  const root = h('div', { class: 'runs' }, listPane, mainEl);
  el.appendChild(root);

  const rows = new Map(); // id -> row parts
  // Read once now; it re-reads itself after an add or a forget.
  const archives = mountArchives(archivesHost);

  // ── queue block (another builder's part) ──
  let queueSummary = null;
  let queueGone = false;
  mountNodes(queueHost, loading('Loading the queue…'));
  import('../runs/queue.js')
    .then((mod) => {
      if (queueGone) return;
      queueHost.replaceChildren();
      queueSummary = mod.mountQueueSummary(queueHost, { selectedId: route.kind === 'queue' ? route.entryId : null });
    })
    .catch((error) => {
      if (queueGone) return;
      mountNodes(queueHost, notice('err', `The queue could not be shown: ${error.message}`));
    });

  function setFilter(next) {
    filter = next;
    allBtn.setAttribute('aria-pressed', String(next === 'all'));
    activeBtn.setAttribute('aria-pressed', String(next === 'active'));
    renderList();
  }

  /** What the main area is about, for the per-row "Compare" button: a run id, or the compare view's left side (a run id or job:<id>). */
  function openRunId() {
    if (main?.kind === 'run') return main.id;
    if (route.kind === 'compare') return route.leftId;
    return null;
  }

  // ── list ──

  function makeRow(run) {
    const id = String(run.id);
    const name = h('span', { class: 'runs__row-name truncate' });
    const status = h('span', { class: 'runs__row-status' });
    const tag = h('span', { class: 'runs__row-tag' });
    const progressText = h('span', { class: 'runs__row-progress num' });
    const bar = progressBar('Training progress');
    const telemetry = h('span', { class: 'runs__row-telemetry num' });
    const updated = h('span', { class: 'runs__row-updated muted' });
    const source = h('span', { class: 'runs__row-source muted truncate' });
    // Opening a run from its row takes the focus to the run (its name), so the next Tab is in the detail
    // and not in the rest of a long list. Only this explicit choice does it: a run that opens by itself
    // (the first of the list, a link from elsewhere) moves nothing.
    const link = h(
      'a',
      {
        class: 'runs__row',
        href: runHref(id),
        onclick: (event) => {
          if (event.defaultPrevented || event.button > 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
          focusWanted = id;
          if (main?.kind === 'run' && main.id === id) focusOpenRun();
        },
      },
      h('span', { class: 'runs__row-top' }, name, tag, status),
      source,
      progressText,
      bar.el,
      telemetry,
      updated,
    );
    const compareBtn = h('button', {
      class: 'btn btn--sm btn--quiet runs__row-compare',
      type: 'button',
      hidden: true,
      onclick: () => {
        const left = openRunId();
        if (left && left !== id) go(compareHref(left, id));
      },
    }, 'Compare');
    const li = h('li', { class: 'runs__item' }, link, compareBtn);
    return { li, link, name, status, tag, source, progressText, bar, telemetry, updated, compareBtn };
  }

  function updateRow(row, run) {
    const id = String(run.id);
    const label = run.name ?? id;
    setText(row.name, label);
    row.link.title = label;
    const st = runStatus(run.status);
    mountKeyed(row.status, `${st.label}|${st.kind}`, () => badge(st.label, st.kind));
    // The tag only when the status word does not already say "Imported".
    const imported = isImported(run) && !st.imported;
    mountKeyed(row.tag, imported ? 'imported' : '', () => (imported ? badge('Imported') : null));
    const prog = hasProgress(run);
    row.progressText.hidden = !prog;
    row.bar.el.hidden = !prog;
    if (prog) {
      setText(row.progressText, `${fmt.int(run.iteration)} of ${fmt.int(run.total)}`);
      row.bar.set(run);
    }
    // Speed and the trainer's estimate only while the run is active: afterwards they are the last, stale values.
    const tele = [];
    if (st.active && Number.isFinite(run.iterationsPerSecond)) tele.push(`${fmt.num(run.iterationsPerSecond, 3)} it/s`);
    if (st.active && typeof run.eta === 'string' && run.eta) tele.push(`trainer's estimate ${run.eta}`);
    row.telemetry.hidden = !tele.length;
    setText(row.telemetry, tele.join(', '));
    const from = run.source === 'archive' ? runSourceLabel(run) : '';
    row.source.hidden = !from;
    setText(row.source, from);
    const ago = fmt.ago(run.updatedAt);
    row.updated.hidden = !ago;
    if (ago) setText(row.updated, `updated ${ago}`);
    const open = openRunId();
    if (open !== null && String(open) === id) row.link.setAttribute('aria-current', 'true');
    else row.link.removeAttribute('aria-current');
    const canCompare = open !== null && String(open) !== id;
    row.compareBtn.hidden = !canCompare;
    if (canCompare) row.compareBtn.setAttribute('aria-label', `Compare ${label} with ${String(open).startsWith('job:') ? 'the chosen evaluation' : 'the open run'}`);
  }

  function renderList() {
    if (destroyed) return;
    const s = runs.get();
    const loaded = s.phase === 'ready' || (s.phase === 'error' && s.list.length > 0);
    const ordered = orderRuns(s.list);
    const shown = filter === 'active' ? ordered.filter((r) => runStatus(r.status).active) : ordered;
    if (s.phase === 'error') {
      mountKeyed(listStatus, `error:${s.error}`, () =>
        notice('err', `The run list could not be loaded: ${s.error}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => runs.refresh() }, 'Try again')),
      );
    } else if (!loaded && !s.list.length) {
      mountKeyed(listStatus, 'loading', () => loading('Loading runs…'));
    } else if (!ordered.length) {
      mountKeyed(listStatus, 'none', () => h('p', { class: 'runs__list-note muted' }, 'No runs yet.'));
    } else if (filter === 'active' && !shown.length) {
      mountKeyed(listStatus, 'none-active', () => h('p', { class: 'runs__list-note muted' }, 'No run is active.'));
    } else {
      mountKeyed(listStatus, '', () => null);
    }
    const keep = new Set(shown.map((r) => String(r.id)));
    for (const [id, row] of rows) {
      if (!keep.has(id)) {
        row.li.remove();
        rows.delete(id);
      }
    }
    shown.forEach((run, index) => {
      const id = String(run.id);
      let row = rows.get(id);
      if (!row) {
        row = makeRow(run);
        rows.set(id, row);
      }
      updateRow(row, run);
      placeAt(listEl, row.li, index);
    });
    if (route.kind === 'auto') resolveAuto();
    else if (main?.kind === 'run') main.detail.onSummary(runs.byId(main.id));
  }

  // ── main area ──

  /**
   * Narrow layout: a part stacks its list above the open item, which then lies a screen or more below
   * the place that was clicked. Bring the open item to the top of the part, and keep it there for a
   * moment while the list and the item are still being read (both grow as their answers arrive).
   * Side by side, the item is already in view and nothing moves.
   */
  const OPEN_ITEM = '.queue__detail, .analysis__detail, .studies__main, .curricula__main';
  let revealWatch = null;
  function revealOpenItem(host) {
    revealWatch?.disconnect();
    revealWatch = null;
    const item = host.querySelector(OPEN_ITEM);
    const side = item?.previousElementSibling;
    if (!item || !side) return;
    // Only the part itself is scrolled. scrollIntoView would also move every clipped ancestor, and the
    // person cannot scroll those back.
    const bring = () => {
      if (item.offsetTop < side.offsetTop + side.offsetHeight - 1) return;
      host.scrollTop += item.getBoundingClientRect().top - host.getBoundingClientRect().top;
    };
    bring();
    if (typeof ResizeObserver !== 'function') return;
    const watch = new ResizeObserver(bring);
    watch.observe(item);
    watch.observe(side);
    revealWatch = watch;
    setTimeout(() => {
      watch.disconnect();
      if (revealWatch === watch) revealWatch = null;
    }, 1500);
  }
  const hasOpenItem = (args) => Object.values(args ?? {}).some(Boolean);

  function closeMain() {
    if (!main) return;
    revealWatch?.disconnect();
    revealWatch = null;
    main.gone = true;
    try {
      if (main.detail) main.detail.destroy();
      else main.handle?.destroy?.();
    } catch {
      // a failing part must not keep the screen from switching
    }
    main = null;
    delete mainEl.dataset.key;
    mainEl.replaceChildren();
  }

  function showMessage(key, build) {
    if (main?.kind === 'message' && main.key === key) return;
    closeMain();
    main = { kind: 'message', key };
    mountNodes(mainEl, build());
  }

  /** The run the person just opened from its row; its detail takes the focus once (see makeRow). */
  let focusWanted = null;
  function focusOpenRun() {
    // The wish is spent by whatever opens next: it never waits for a later, unrelated open of the same run.
    const wanted = focusWanted;
    focusWanted = null;
    if (main?.kind !== 'run' || wanted !== main.id) return;
    const title = main.host.querySelector('.runs__name');
    if (title) title.focus({ preventScroll: true });
  }

  function openRun(id) {
    if (main?.kind === 'run' && main.id === id) return;
    closeMain();
    const host = h('div', { class: 'runs__part' });
    mainEl.appendChild(host);
    main = { kind: 'run', id, host, detail: createRunDetail(host, { id, summary: runs.byId(id) }) };
    // After `main` is set: select() notifies the list, which must see this run as the open one.
    if (String(runs.get().selectedId) !== id) runs.select(id);
    focusOpenRun();
  }

  /** Mount a part from another module; `args` is read at mount time so a route change while loading wins. */
  function openPart(kind, args, load, start, failText) {
    // A part that failed to load is opened afresh by the next route change, which is its way to try again.
    if (main?.kind === kind && !main.failed) {
      const moved = JSON.stringify(main.args) !== JSON.stringify(args);
      main.args = args;
      if (main.handle) {
        main.update(main.handle, args);
        if (moved && hasOpenItem(args)) revealOpenItem(main.host);
      }
      return;
    }
    closeMain();
    const host = h('div', { class: 'runs__part' });
    mainEl.appendChild(host);
    const m = { kind, host, handle: null, gone: false, args, update: start.update };
    main = m;
    mountNodes(host, h('div', { class: 'runs__pad' }, loading('Loading…')));
    load()
      .then((mod) => {
        if (m.gone) return;
        host.replaceChildren();
        m.handle = start.mount(mod, host, m.args);
        if (hasOpenItem(m.args)) revealOpenItem(host);
      })
      .catch((error) => {
        if (m.gone) return;
        m.failed = true;
        mountNodes(host, h('div', { class: 'runs__pad' }, notice('err', `${failText}: ${error.message}`)));
      });
  }

  function resolveAuto() {
    const s = runs.get();
    const list = s.list;
    if (!list.length) {
      if (s.phase === 'ready') {
        showMessage('empty', () =>
          h(
            'div',
            { class: 'runs__pad' },
            emptyState({
              title: 'No runs yet',
              text: 'A run starts from a recipe. On the Recipe screen, check a recipe, then choose Start training to train now, or Add to queue to keep a copy that starts later. Experiments the trainer already has in its experiments folder also appear here.',
              actions: [h('a', { class: 'btn btn--primary', href: '#/recipe' }, 'Open the Recipe screen'), h('a', { class: 'btn', href: queueHref() }, 'Open the queue')],
            }),
          ),
        );
      } else if (s.phase === 'error') {
        showMessage('error', () => h('div', { class: 'runs__pad' }, notice('err', `The runs could not be loaded: ${s.error}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => runs.refresh() }, 'Try again'))));
      } else {
        showMessage('loading', () => h('div', { class: 'runs__pad' }, loading('Loading runs…')));
      }
      return;
    }
    if (main?.kind === 'run' && list.some((r) => String(r.id) === main.id)) {
      main.detail.onSummary(runs.byId(main.id));
      return;
    }
    const byId = (id) => list.find((r) => String(r.id) === String(id));
    const ordered = orderRuns(list);
    const target = byId(s.selectedId) ?? ordered.find((r) => runStatus(r.status).active) ?? ordered[0];
    openRun(String(target.id));
    refreshRowMarks();
  }

  function refreshRowMarks() {
    for (const run of runs.get().list) {
      const row = rows.get(String(run.id));
      if (row) updateRow(row, run);
    }
  }

  function applyRoute() {
    queueSummary?.setSelected?.(route.kind === 'queue' ? route.entryId : null);
    if (route.kind === 'run') openRun(route.id);
    else if (route.kind === 'queue') {
      openPart(
        'queue',
        { entryId: route.entryId },
        () => import('../runs/queue.js'),
        { mount: (mod, host, a) => mod.mountQueueView(host, { entryId: a.entryId }), update: (handle, a) => handle.setEntry?.(a.entryId) },
        'The queue could not be shown',
      );
    } else if (route.kind === 'compare') {
      openPart(
        'compare',
        { leftId: route.leftId, rightId: route.rightId },
        () => import('../runs/evaluation.js'),
        { mount: (mod, host, a) => mod.mountRunCompare(host, { leftId: a.leftId, rightId: a.rightId }), update: (handle, a) => handle.setRuns?.(a.leftId, a.rightId) },
        'The comparison could not be shown',
      );
    } else if (route.kind === 'analysis') {
      openPart(
        'analysis',
        { jobId: route.jobId },
        () => import('../runs/analysis.js'),
        { mount: (mod, host, a) => mod.mountAnalysisView(host, { jobId: a.jobId }), update: (handle, a) => handle.setJob?.(a.jobId) },
        'The analysis jobs could not be shown',
      );
    } else if (route.kind === 'studies') {
      openPart(
        'studies',
        { studyId: route.studyId },
        () => import('../runs/studies.js'),
        { mount: (mod, host, a) => mod.mountStudiesView(host, { studyId: a.studyId }), update: (handle, a) => handle.setStudy?.(a.studyId) },
        'The studies could not be shown',
      );
    } else if (route.kind === 'curricula') {
      openPart(
        'curricula',
        { curriculumId: route.curriculumId },
        () => import('../runs/curricula.js'),
        { mount: (mod, host, a) => mod.mountCurriculaView(host, { curriculumId: a.curriculumId }), update: (handle, a) => handle.setCurriculum?.(a.curriculumId) },
        'The curricula could not be shown',
      );
    } else {
      if (main && main.kind !== 'run' && main.kind !== 'message') closeMain();
      resolveAuto();
    }
    for (const [kind, link] of [['analysis', analysisLink], ['studies', studiesLink], ['curricula', curriculaLink]]) {
      if (route.kind === kind) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    refreshRowMarks();
  }

  const unsubscribe = runs.subscribe(renderList);
  // Relative times ("updated 5 s ago") age between polls; re-render them on a slow tick.
  const agoTimer = setInterval(() => {
    refreshRowMarks();
    main?.detail?.tick();
  }, 5000);
  renderList();
  applyRoute();

  return {
    setParam(next) {
      route = parseRoute(next);
      applyRoute();
    },
    destroy() {
      destroyed = true;
      unsubscribe();
      clearInterval(agoTimer);
      releaseWatch();
      closeMain();
      queueGone = true;
      try {
        queueSummary?.destroy?.();
      } catch {
        // keep cleaning up
      }
      queueSummary = null;
      archives.destroy();
      root.remove();
    },
  };
}
