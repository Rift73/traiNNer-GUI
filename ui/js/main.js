// Shell: the pipeline rail, the next-step strip and the router.

import { h, mount, fmt } from './dom.js';
import { persist } from './store.js';
import { conn } from './state/conn.js';
import { recipe } from './state/recipe.js';
import { data, AUDIT_TASKS } from './state/data.js';
import { runs, runStatus } from './state/runs.js';
import { queue } from './state/queue.js';
import { catalog } from './state/catalog.js';
import { countIssues, loading, notice, toast, confirmOnPhone } from './ui.js';
import { startTraining, addToQueue, go, runHref, queueHref } from './actions.js';

const VIEWS = {
  data: { title: 'Data', load: () => import('./views/data.js') },
  recipe: { title: 'Recipe', load: () => import('./views/recipe.js') },
  runs: { title: 'Runs', load: () => import('./views/runs.js') },
};
// The order a first training takes: a recipe says which data it needs, the data is chosen and checked
// against it, then it runs.
const ORDER = ['recipe', 'data', 'runs'];

const railEl = document.getElementById('rail');
const nextEl = document.getElementById('next');
const viewEl = document.getElementById('view');

// ───────── station summaries (all read from real state) ─────────

/** True when the open recipe's data is audited by the service as a job, not by the paired folder check. */
function auditedRecipe() {
  const r = recipe.get();
  return r.phase === 'open' && AUDIT_TASKS.includes(String(r.summary?.task ?? ''));
}

/** Where the data audit of the open recipe stands, in the words both the rail and the next step use. */
function auditState() {
  const audit = data.get().audit;
  if (!audit) return 'none';
  if (audit.status === 'starting' || audit.status === 'running') return 'running';
  if (audit.status !== 'completed') return audit.status; // 'failed' | 'cancelled'
  if (!data.auditIsCurrent()) return 'stale';
  if (audit.errors) return 'errors';
  return audit.warnings ? 'warnings' : 'ok';
}

function dataStation() {
  const d = data.get();
  if (auditedRecipe()) {
    const audit = d.audit;
    switch (auditState()) {
      case 'running':
        return { state: '', text: 'Checking data…' };
      case 'failed':
        return { state: 'err', text: 'Data check failed' };
      case 'stale':
        return { state: '', text: 'Data check out of date' };
      case 'errors':
        return { state: 'err', text: `Data checked, ${fmt.plural(audit.errors, 'error')}` };
      case 'warnings':
        return { state: 'warn', text: `Data checked, ${fmt.plural(audit.warnings, 'warning')}` };
      case 'ok':
        return { state: 'ok', text: 'Data checked' };
      default:
        return { state: '', text: 'Data not checked' };
    }
  }
  if (d.phase === 'scanning') return { state: '', text: 'Checking folders…' };
  if (d.phase === 'error') return { state: 'err', text: 'Folder check failed' };
  if (d.phase === 'done' && d.result) {
    const counts = d.result.counts ?? {};
    const what = counts.paired > 0 ? fmt.plural(counts.paired, 'pair') : Number.isFinite(counts.gt) ? fmt.plural(counts.gt, 'image') : 'Checked';
    const { errors, warnings } = countIssues(d.result.issues);
    if (errors) return { state: 'err', text: `${what}, ${fmt.plural(errors, 'error')}` };
    if (warnings) return { state: 'warn', text: `${what}, ${fmt.plural(warnings, 'warning')}` };
    return { state: 'ok', text: what };
  }
  return { state: '', text: 'Folders not checked' };
}

function recipeStation() {
  const r = recipe.get();
  if (r.phase === 'empty') return { state: '', text: 'No recipe open' };
  const name = r.summary?.name || r.name || 'Untitled';
  if (r.check === 'failed') return { state: 'err', text: 'Check could not run' };
  // While a re-check is running, keep showing the last verdict instead of flickering on every edit.
  if (!r.hasResult) return { state: '', text: `${name}, checking…` };
  const { errors, warnings } = countIssues(r.issues);
  if (errors || !r.ok) return { state: 'err', text: `${name}, ${fmt.plural(errors, 'error')}` };
  if (warnings) return { state: 'warn', text: `${name}, ${fmt.plural(warnings, 'warning')}` };
  return { state: 'ok', text: name };
}

function runsStation() {
  const r = runs.get();
  if (r.phase === 'error') return { state: 'err', text: 'Run list unavailable' };
  if (r.phase === 'idle' || r.phase === 'loading') return { state: '', text: 'Loading…' };
  // The queue is part of this station: released entries are runs that have not started yet.
  const released = queue.pending().length;
  const held = queue.held().length;
  const active = runs.active();
  if (active.length === 1) {
    const run = active[0];
    const progress = Number.isFinite(run.iteration) && Number.isFinite(run.total) ? `, ${fmt.int(run.iteration)} of ${fmt.int(run.total)}` : '';
    return { state: 'run', text: `${runStatus(run.status).label}${progress}${released ? `, ${released} queued` : ''}` };
  }
  if (active.length > 1) return { state: 'run', text: `${active.length} runs active${released ? `, ${released} queued` : ''}` };
  if (released) {
    return queue.get().paused ? { state: 'warn', text: `${released} queued, queue paused` } : { state: 'run', text: `${released} queued, waiting to start` };
  }
  const onHold = held ? `, ${held} on hold` : '';
  if (r.list.length === 0) return { state: '', text: held ? `${held} on hold in the queue` : 'No runs yet' };
  return { state: '', text: `${fmt.plural(r.list.length, 'run')}${onHold}` };
}

const STATION_STATE = { data: dataStation, recipe: recipeStation, runs: runsStation };
const STATE_WORD = { ok: 'ready', warn: 'needs attention', err: 'has errors', run: 'active', '': '' };

// ───────── rail ─────────

const stationEls = {};
let connEl;
let connText;
let themeButton;

function buildRail() {
  const stations = h(
    'nav',
    { class: 'stations', 'aria-label': 'Workflow' },
    ORDER.map((id, index) => {
      const mark = h('span', { class: 'station__mark', 'aria-hidden': 'true' }, index + 1);
      const stateText = h('span', { class: 'station__state' });
      const link = h(
        'a',
        { class: 'station', href: `#/${id}`, 'aria-keyshortcuts': `Alt+${index + 1}` },
        mark,
        h('span', { class: 'station__name' }, VIEWS[id].title),
        stateText,
      );
      stationEls[id] = { link, stateText };
      return link;
    }),
  );
  connText = h('span', { class: 'conn__text' });
  // The connection state is also the way in to "what is this trainer machine": interpreter, packages, disk.
  connEl = h(
    'button',
    {
      class: 'conn',
      type: 'button',
      'aria-haspopup': 'dialog',
      onclick: async () => {
        if (conn.get().state !== 'online') {
          conn.check();
          return;
        }
        try {
          const part = await import('./components/environment.js');
          await part.openEnvironmentDialog();
        } catch (error) {
          console.error(error);
          toast(`The trainer environment could not be shown: ${error.message}`, { kind: 'err' });
        }
      },
    },
    h('span', { class: 'conn__dot', 'aria-hidden': 'true' }),
    connText,
  );
  themeButton = h(
    'button',
    {
      class: 'btn btn--sm',
      type: 'button',
      onclick: () => {
        const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.dataset.theme = next;
        persist.set('theme', next);
        renderTheme();
      },
    },
    'Dark theme',
  );
  mount(
    railEl,
    h('a', { class: 'wordmark', href: '#/recipe', 'aria-label': 'traiNNer GUI' }, 'traiNNer', h('small', null, ' GUI')),
    stations,
    h('div', { class: 'rail__end' }, connEl, themeButton),
  );
  renderTheme();
}

function renderTheme() {
  themeButton.setAttribute('aria-pressed', String(document.documentElement.dataset.theme === 'dark'));
}

function renderRail() {
  for (const id of ORDER) {
    const { state, text } = STATION_STATE[id]();
    const { link, stateText } = stationEls[id];
    link.dataset.state = state;
    if (stateText.textContent !== text) stateText.textContent = text;
    link.title = STATE_WORD[state] ? `${VIEWS[id].title}: ${STATE_WORD[state]}` : VIEWS[id].title;
  }
  const c = conn.get();
  connEl.dataset.state = c.state;
  if (c.state === 'online') {
    connText.textContent = [c.info?.name, c.info?.version].filter(Boolean).join(' ') || 'Trainer service connected';
    connEl.title = `${c.info?.trainerRoot ? `Trainer folder: ${c.info.trainerRoot}. ` : ''}Show the trainer environment.`;
  } else {
    connText.textContent = c.state === 'offline' ? 'Trainer service offline' : 'Connecting…';
    connEl.title = c.state === 'offline' ? `${c.error ?? ''} Try again.`.trim() : '';
  }
}

// ───────── next step ─────────

function nextStep() {
  const c = conn.get();
  if (c.state === 'offline') {
    return {
      tone: 'err',
      text: 'The trainer service is not answering, so nothing can be loaded, checked or started.',
      action: { label: 'Try again', run: () => conn.check() },
    };
  }
  if (c.state === 'checking') return { tone: '', text: 'Connecting to the trainer service…' };

  const active = runs.active();
  if (active.length) {
    const run = active[0];
    const progress = Number.isFinite(run.iteration) && Number.isFinite(run.total) ? `, iteration ${fmt.int(run.iteration)} of ${fmt.int(run.total)}` : '';
    return {
      tone: 'run',
      text: `${runStatus(run.status).label}: ${run.name ?? run.id}${progress}.`,
      action: { label: 'Open run', href: runHref(run.id) },
    };
  }

  // Released queue entries are next in line for the trainer; say why nothing is training yet.
  const released = queue.pending().sort((a, b) => (a.order ?? Infinity) - (b.order ?? Infinity));
  if (released.length) {
    const first = released[0];
    const count = released.length === 1 ? '1 released queue entry' : `${released.length} released queue entries`;
    if (queue.get().paused) {
      return {
        tone: 'warn',
        text: `The queue is paused, so ${count} will not start.`,
        action: {
          label: 'Continue the queue',
          // Continuing lets released entries start training: the phone edition asks first.
          run: async () => {
            if (!(await confirmOnPhone({ title: 'Continue the queue?', body: 'Released entries start when the trainer is free.', confirmLabel: 'Continue the queue' }))) return;
            queue.setPaused(false).catch((error) => toast(`The queue was not continued: ${error.message}`, { kind: 'err' }));
          },
        },
        second: { label: 'Open queue', href: queueHref(first.id) },
      };
    }
    return {
      tone: 'run',
      text: `Next in the queue: ${first.name ?? first.id}. ${first.blockedReason ?? 'It starts when the trainer is free.'}`,
      action: { label: 'Open queue', href: queueHref(first.id) },
    };
  }

  const r = recipe.get();
  if (r.phase === 'empty') {
    const held = queue.held();
    if (held.length) {
      return {
        tone: '',
        text: `${held.length === 1 ? '1 queue entry is' : `${held.length} queue entries are`} on hold. An entry starts only after you release it.`,
        action: { label: 'Open queue', href: queueHref(held[0].id) },
        second: { label: 'Choose a recipe', href: '#/recipe' },
      };
    }
    return {
      tone: '',
      text: 'Choose a recipe to start from. A recipe holds every training option in one file.',
      action: { label: 'Choose a recipe', href: '#/recipe' },
    };
  }
  if (r.check === 'failed') {
    return { tone: 'err', text: `The recipe could not be checked: ${r.checkError}`, action: { label: 'Check again', run: () => recipe.checkNow() } };
  }
  if (!r.hasResult) return { tone: '', text: 'Checking the recipe…' };

  const { errors, warnings } = countIssues(r.issues);
  if (errors || !r.ok) {
    // A template names example folders. When the only errors are data paths that are not there, the
    // step that fixes them is choosing real data, not reading an error list.
    const errorIssues = r.issues.filter((issue) => issue.level === 'error');
    const missing = errorIssues.filter(isMissingDataPath);
    if (missing.length && missing.length === errorIssues.length) {
      return {
        tone: 'warn',
        text: `The recipe's data is not on the trainer machine yet (${fmt.plural(missing.length, 'path')} not found). Choose your own on the Data screen.`,
        hereText: `The recipe's data is not on the trainer machine yet (${fmt.plural(missing.length, 'path')} not found). Choose your own with Change… below.`,
        action: { label: 'Choose data', href: '#/data', primary: true },
        second: { label: 'Show errors', run: () => showRecipeIssues() },
      };
    }
    return {
      tone: 'err',
      text: `The recipe has ${fmt.plural(errors, 'error')}. Training cannot start until ${errors === 1 ? 'it is' : 'they are'} fixed.`,
      action: { label: 'Show errors', run: () => showRecipeIssues() },
    };
  }

  const passed = warnings ? `The recipe passed its check with ${fmt.plural(warnings, 'warning')}.` : 'The recipe passed its check.';

  // Classification, segmentation and detection: the service audits the recipe's data as a job.
  if (auditedRecipe()) {
    const audit = data.get().audit;
    const state = auditState();
    if (state === 'running') return { tone: '', text: "Checking the recipe's data…", action: { label: 'Open Data', href: '#/data' } };
    if (state === 'errors') {
      return {
        tone: 'err',
        text: `The data check found ${fmt.plural(audit.errors, 'error')} in the recipe's data.`,
        action: { label: 'Open Data', href: '#/data' },
      };
    }
    if (state === 'failed') {
      return { tone: 'err', text: `The data check did not finish: ${audit.error ?? 'the job failed.'}`, action: { label: 'Open Data', href: '#/data' } };
    }
    if (state !== 'ok' && state !== 'warnings') {
      return {
        tone: '',
        text:
          state === 'stale'
            ? "The recipe's data settings changed after the last data check. Check the data again before training."
            : "Check the recipe's data before training: class names, annotations, masks and image sizes.",
        // On the Data screen the sentence names the button that is there.
        hereText:
          state === 'stale'
            ? "The recipe's data settings changed after the last data check. Use Check again below before training."
            : "Check the recipe's data before training with Check the data below: class names, annotations, masks and image sizes.",
        action: { label: 'Check data', href: '#/data' },
        second: { label: 'Start training', run: () => startTraining() },
      };
    }
    return {
      tone: 'ok',
      text: `${passed} ${state === 'warnings' ? `The data check reported ${fmt.plural(audit.warnings, 'warning')}.` : 'Its data was checked.'}`,
      action: { label: 'Start training', run: () => startTraining(), primary: true },
      second: { label: 'Add to queue', run: () => addToQueue() },
    };
  }

  const d = data.get();
  const hasFolders = recipe.datasetFolders().some((set) => set.gt.length || set.lq.length);
  if (d.phase === 'done' && d.result && countIssues(d.result.issues).errors) {
    return { tone: 'err', text: 'The image folders you checked have errors.', action: { label: 'Open Data', href: '#/data' } };
  }
  if (hasFolders && d.phase !== 'done' && d.phase !== 'scanning') {
    return {
      tone: '',
      text: "Check the recipe's image folders before training: that they exist, how many pairs they hold, and their sizes.",
      hereText: "Check the recipe's image folders before training with Check these folders below: that they exist, how many pairs they hold, and their sizes.",
      action: { label: 'Check folders', href: '#/data' },
      second: { label: 'Start training', run: () => startTraining() },
    };
  }
  return {
    tone: 'ok',
    text: passed,
    action: { label: 'Start training', run: () => startTraining(), primary: true },
    second: { label: 'Add to queue', run: () => addToQueue() },
  };
}

// The dataset fields that hold a place on disk, and the service's wording for one that is not there.
const DATA_PATH_FIELD = /^datasets\.[^.]+\.(dataroot_gt|dataroot_lq|annotation_file|meta_info)(\.\d+)?$/;

function isMissingDataPath(issue) {
  return DATA_PATH_FIELD.test(String(issue.path ?? '')) && /^Path was not found/i.test(String(issue.message ?? ''));
}

function showRecipeIssues() {
  go('#/recipe');
  const first = recipe.get().issues.find((issue) => issue.level === 'error');
  // Give the view a moment to mount before asking it to scroll.
  setTimeout(() => recipe.requestFocus(first?.path ?? ''), 60);
}

let lastNextKey = '';

function actionButton(action, primary) {
  const cls = `btn btn--sm${primary ? ' btn--primary' : ''}`;
  return action.href
    ? h('a', { class: cls, href: action.href }, action.label)
    : h('button', { class: cls, type: 'button', onclick: action.run }, action.label);
}

function renderNext() {
  const step = nextStep();
  // A button that only leads to the screen the person is already on is left out; the sentence may then
  // point at the place on that screen instead (`hereText`).
  const screen = `#/${parseHash().name}`;
  const here = (action) => Boolean(action?.href) && action.href === screen;
  const text = here(step.action) && step.hereText ? step.hereText : step.text;
  const action = here(step.action) ? null : step.action;
  const second = here(step.second) ? null : step.second;
  const key = JSON.stringify([step.tone, text, action?.label, second?.label]);
  if (key === lastNextKey) return;
  lastNextKey = key;
  nextEl.dataset.tone = step.tone;
  mount(
    nextEl,
    h('span', { class: 'next__label' }, 'Next'),
    h('span', { class: 'next__text' }, text),
    action ? actionButton(action, action.primary) : null,
    second ? actionButton(second, false) : null,
  );
}

// ───────── router ─────────

let current = { name: null, cleanup: null, module: null };
let navToken = 0;

function parseHash() {
  const [, name = '', ...rest] = location.hash.replace(/^#/, '').split('/');
  return { name, param: rest.length ? decodeURIComponent(rest.join('/')) : null };
}

async function route() {
  const { name, param } = parseHash();
  if (!VIEWS[name]) {
    const fallback = persist.get('view', 'recipe');
    location.replace(`#/${VIEWS[fallback] ? fallback : 'recipe'}`);
    return;
  }
  persist.set('view', name);
  for (const id of ORDER) {
    if (id === name) stationEls[id].link.setAttribute('aria-current', 'page');
    else stationEls[id].link.removeAttribute('aria-current');
  }
  document.title = `${VIEWS[name].title} – traiNNer GUI`;
  // The next-step line leaves out a button that leads to the screen now open.
  renderNext();

  // Same view, different parameter: let the view handle it without a remount.
  if (current.name === name && current.module?.setParam) {
    current.module.setParam(param);
    return;
  }

  const token = ++navToken;
  // A control inside the screen led to another screen: its focus would fall to the top of the page with
  // the old screen (or has already, when the button was disabled while it worked). Go on from the new
  // screen instead. Focus in the rail (a station link) stays there; the first screen moves nothing.
  const active = document.activeElement;
  const focusWasInside = current.name !== null && (!active || active === document.body || viewEl.contains(active));
  if (current.cleanup) {
    try {
      current.cleanup();
    } catch (error) {
      console.error(error);
    }
  }
  current = { name, cleanup: null, module: null };
  mount(viewEl, loading(`Opening ${VIEWS[name].title}…`));
  try {
    const module = await VIEWS[name].load();
    if (token !== navToken) return;
    mount(viewEl);
    const cleanup = module.mount(viewEl, { param });
    current = { name, cleanup: typeof cleanup === 'function' ? cleanup : null, module };
    // Only when nothing took the focus meanwhile (a dialog, a field the new screen focused itself).
    if (focusWasInside && (document.activeElement === document.body || !document.activeElement)) viewEl.focus({ preventScroll: true });
  } catch (error) {
    if (token !== navToken) return;
    console.error(error);
    mount(viewEl, h('div', { style: { padding: '16px' } }, notice('err', `The ${VIEWS[name].title} screen failed to open: ${error.message}`)));
  }
}

// ───────── boot ─────────

buildRail();
recipe.restore();
for (const store of [conn, recipe, data, runs, queue]) {
  store.subscribe(() => {
    renderRail();
    renderNext();
  });
}
renderRail();
renderNext();

window.addEventListener('hashchange', route);
// "Skip to content" moves the focus only. Followed as a link it would replace the address (#/runs/<id>) with
// "#view", which is no screen: the open run, job or comparison would be dropped.
document.querySelector('.skip-link')?.addEventListener('click', (event) => {
  event.preventDefault();
  viewEl.focus();
});
window.addEventListener('keydown', (event) => {
  if (!event.altKey || event.ctrlKey || event.metaKey) return;
  const index = ['1', '2', '3'].indexOf(event.key);
  if (index >= 0) {
    event.preventDefault();
    go(`#/${ORDER[index]}`);
  }
});

conn.check();
runs.refresh();
queue.refresh();
// Task names, networks and losses: fetched once, read by every screen through state/catalog.js.
catalog.load().catch(() => {});
route();
