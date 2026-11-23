// "Training videos": the panel of a clip-training dataset (VideoClipDataset) in the options form, and on
// the Data screen. A view of state/video-prep.js, which owns the folder check and the preparation job, so
// the panel can be thrown away with a rebuild of the form at any time. The form never calls cleanups: the
// panel stops listening by itself once its element has left the page.
//
//   mountVideoPrep(host, {onReveal}) -> cleanup   (onReveal is accepted for callers; the panel needs none:
//                                                  revealing it is recipe.requestFocus('datasets.train.dataroot_gt'))
//   askAndPrepare() -> Promise<void>   (the phone edition asks first; used by the next-step line too)

import { h, mount, fmt, uid } from '../dom.js';
import { recipe } from '../state/recipe.js';
import { videoPrep, counts, jobActive, exampleUnchosen } from '../state/video-prep.js';
import { badge, toast, confirmOnPhone } from '../ui.js';
import { analysisHref, go } from '../actions.js';
import { matchesExtensions, folderOfPath, baseName } from '../components/path-picker.js';

const VIDEO_EXTENSIONS = ['.mkv', '.mp4', '.m2ts', '.mov'];
const ROW_LIMIT = 50;
const LEAD =
  'Training reads the video files as they are: MKV, MP4, M2TS or MOV. No ZIP and no extracted frames. Each video needs an index, a small file written once into a folder named index next to the video.';
const UNSUPPORTED = 'The trainer service that is running does not offer this yet. Close traiNNer GUI and open it again.';
const CANCELLED = 'Preparation cancelled. Indexes already written are kept; preparing again continues with the rest.';
const ORDER = { invalid: 0, stale: 1, missing: 2, ready: 3 };
const FILE_BADGE = { invalid: ['Cannot be used', 'err'], stale: ['Out of date', 'warn'], missing: ['No index', 'warn'], ready: ['Ready', 'ok'] };
const GT_PATH = ['datasets', 'train', 'dataroot_gt'];

const prepareLabel = (n) => (n === 1 ? 'Prepare 1 video' : `Prepare ${fmt.int(n)} videos`);

/** Start the preparation; on the phone edition it asks first (a press on glass is easy to make by accident). */
export async function askAndPrepare() {
  const { fixable } = counts(videoPrep.get().check.answer);
  const ok = await confirmOnPhone({
    title: `${prepareLabel(fixable)}?`,
    body: 'This reads each video on the trainer machine and writes its index next to it.',
    confirmLabel: prepareLabel(fixable),
  });
  if (ok) await videoPrep.prepare();
}

/**
 * Bring the panel into view from anywhere (the next-step line, the Data screen): the Recipe screen, its
 * Datasets section, the panel at the top of the form with its title focused. The form may still be loading,
 * so it is asked again for a moment until the title has the focus.
 */
export function revealVideoPanel() {
  if (!/^#\/recipe(\/|$)/.test(location.hash)) go('#/recipe');
  let tries = 0;
  const attempt = () => {
    recipe.requestFocus('datasets.train');
    if (document.activeElement?.classList?.contains('vprep__title')) return;
    tries += 1;
    if (tries < 30) setTimeout(attempt, 100);
  };
  setTimeout(attempt, 60);
}

/** A notice in the panel's one status region: the same look as ui.js notice(), without a live region of its own. */
function box(kind, body, ...actions) {
  return h('div', { class: `notice${kind ? ` notice--${kind}` : ''}` }, h('div', { class: 'notice__body' }, body), actions.length ? h('div', { class: 'vprep__noticeacts' }, actions) : null);
}

/** The reason of one file, without the path the service puts in front of it. */
function reasonOf(file) {
  // The badge already says it: "No index".
  if (file.status === 'missing') return '';
  const text = String(file.reason ?? '').trim();
  let out = text;
  const lead = typeof file.path === 'string' && file.path ? `${file.path}: ` : null;
  if (lead && out.startsWith(lead)) out = out.slice(lead.length);
  // The index file the service names instead: "<folder>/index/<stem>.json: …".
  else out = out.replace(/^.*?[\\/]index[\\/][^\\/]+?\.json: /, '');
  // What to do is the panel's button, not the row's.
  out = out.replace(/ Prepare video indexes( again)?\.$/, '');
  if (out === text) return text;
  return out ? `${out.charAt(0).toUpperCase()}${out.slice(1)}` : text;
}

const gtValue = () => recipe.get().config?.datasets?.train?.dataroot_gt;

function writeGt(value) {
  return recipe.applyEdits([{ path: GT_PATH, value }]).catch((error) => toast(`dataroot_gt was not changed: ${error?.message ?? error}`, { kind: 'err' }));
}

// ───────── the videos a visit to the picker chooses (pure, exported for checks) ─────────

/** A path for comparing: forward slashes, no trailing one, and a Windows drive as the trainer sees it (D:\x is /mnt/d/x). */
const slashes = (p) => String(p ?? '').trim().replace(/\\/g, '/').replace(/\/+$/, '').replace(/^([A-Za-z]):(\/|$)/, (_, d) => `/mnt/${d.toLowerCase()}/`).replace(/\/+$/, '');
const isAbsolute = (p) => /^\//.test(slashes(p)) || /^[A-Za-z]:\//.test(slashes(p));
export const isVideoFile = (p) => matchesExtensions(slashes(p), VIDEO_EXTENSIONS);
const asList = (value) => (Array.isArray(value) ? value : [value]).filter((v) => typeof v === 'string' && v.trim() !== '');

/**
 * dataroot_gt after a visit that ticked `picked` (absolute paths): the visit defines the set of files. An
 * entry is dropped when it is an absolute video file (the picker stood for those), a folder holding a
 * picked file, a relative file naming a picked one, or every entry when they are the template's untouched
 * example. Everything else stays in its place; the picked files follow; no path twice.
 * @returns {{value: string[], replaced: string[]}} replaced: the folder entries the files now stand for
 */
export function mergePicked(current, picked, { example = false } = {}) {
  const files = picked.map(slashes);
  const replaced = [];
  const kept = asList(current).filter((entry) => {
    if (example) return false;
    const e = slashes(entry);
    if (isVideoFile(e)) {
      if (isAbsolute(e)) return false;
      const rel = e.replace(/^\.\//, '');
      return !files.some((f) => f === rel || f.endsWith(`/${rel}`));
    }
    const rel = e.replace(/^\.\//, '');
    const holds = isAbsolute(e) ? files.some((f) => f.startsWith(`${e}/`)) : files.some((f) => f.startsWith(`${rel}/`) || f.includes(`/${rel}/`));
    if (holds) replaced.push(entry);
    return !holds;
  });
  const seen = new Set();
  const value = [];
  for (const p of [...kept, ...picked]) {
    const key = slashes(p);
    if (seen.has(key)) continue;
    seen.add(key);
    value.push(p);
  }
  return { value, replaced };
}

/**
 * The lines of the "Videos" value: a folder or a lone file on a line of its own; several files of one folder
 * as the folder once and their names (at most 8, then "and N more").
 * @returns {Array<{path: string} | {folder: string, count: number, names: string[], more: number}>}
 */
export function groupRoots(roots) {
  const groups = new Map();
  const order = [];
  for (const root of roots) {
    if (!isVideoFile(root)) {
      order.push({ path: root });
      continue;
    }
    const folder = folderOfPath(root);
    if (!groups.has(folder)) {
      const group = { folder, files: [] };
      groups.set(folder, group);
      order.push(group);
    }
    groups.get(folder).files.push(root);
  }
  return order.map((item) => {
    if (!item.files) return item;
    if (item.files.length === 1) return { path: item.files[0] };
    const names = item.files.map((f) => baseName(f));
    return { folder: item.folder, count: names.length, names: names.slice(0, 8), more: Math.max(0, names.length - 8) };
  });
}

/** Whether the readiness rows were opened, kept across rebuilds of the form. */
let readyOpen = false;

export function mountVideoPrep(host, { onReveal } = {}) {
  const titleId = uid('vprep-title');
  const labelId = uid('vprep-label');
  let disposed = false;
  let attached = false;

  const badgeSlot = h('span', { class: 'vprep__badge' });
  const paths = h('div', { class: 'vprep__paths', role: 'list', 'aria-labelledby': labelId });
  const chooseButton = h('button', { class: 'btn btn--sm vprep__btn', type: 'button', onclick: () => chooseFolder() }, 'Choose folder…');
  const videosButton = h('button', { class: 'btn btn--sm btn--quiet vprep__btn', type: 'button', onclick: () => chooseVideos() }, 'Choose videos…');
  const checkButton = h('button', { class: 'btn btn--sm vprep__btn', type: 'button', onclick: () => videoPrep.check({ force: true }) }, 'Check videos');
  const prepareButton = h('button', { class: 'btn btn--sm btn--primary vprep__btn', type: 'button', hidden: true, onclick: () => askAndPrepare() }, '');
  const actions = h('div', { class: 'vprep__actions' }, checkButton, prepareButton);

  // While a preparation runs: real numbers only, so no bar until the service names a total.
  const bar = h('div', { class: 'progress progress--run vprep__bar', role: 'progressbar', 'aria-label': 'Videos indexed', 'aria-valuemin': '0', hidden: true }, h('div', { class: 'progress__bar' }));
  const episodeLine = h('p', { class: 'vprep__episode', hidden: true });
  const countLine = h('p', { class: 'vprep__count', hidden: true });
  const waitLine = h('p', { class: 'vprep__wait', hidden: true });
  const pollLine = h('p', { class: 'vprep__pollerror', hidden: true });
  const cancelButton = h('button', { class: 'btn btn--sm vprep__btn', type: 'button', onclick: () => doCancel() }, 'Cancel');
  const jobLink = h('a', { class: 'btn btn--sm btn--quiet vprep__btn', href: '#/runs/analysis' }, 'Open the job');
  const jobBox = h('div', { class: 'vprep__job', hidden: true }, bar, episodeLine, countLine, waitLine, pollLine, h('div', { class: 'vprep__actions' }, cancelButton, jobLink));

  const status = h('div', { class: 'vprep__status', role: 'status' });
  const files = h('div', { class: 'vprep__files' });

  const root = h(
    'section',
    { class: 'vprep', 'aria-labelledby': titleId },
    h('div', { class: 'vprep__head' }, h('h4', { class: 'vprep__title', id: titleId, tabindex: '-1' }, 'Training videos'), badgeSlot),
    h('p', { class: 'vprep__lead' }, LEAD),
    h('div', { class: 'vprep__source' }, h('span', { class: 'vprep__label', id: labelId }, 'Videos'), paths, h('div', { class: 'vprep__pick' }, chooseButton, videosButton)),
    actions,
    jobBox,
    status,
    files,
  );
  host.appendChild(root);

  async function chooseFolder() {
    const { pickPath } = await import('../components/path-picker.js');
    const current = gtValue();
    const list = (Array.isArray(current) ? current : [current]).filter((v) => typeof v === 'string' && v);
    const picked = await pickPath({ title: 'Folder with the training videos', mode: 'folder', start: list[0] || '', confirmLabel: 'Use this folder' });
    if (typeof picked !== 'string' || !picked) return;
    writeGt(Array.isArray(current) ? [picked] : picked);
  }

  /** "Choose videos…": several files ticked in one visit, which then are the recipe's set of video files. */
  async function chooseVideos() {
    const { pickPath } = await import('../components/path-picker.js');
    const current = gtValue();
    const list = asList(current);
    const selected = list.filter((p) => isVideoFile(p) && isAbsolute(p));
    const folder = list.find((p) => !isVideoFile(p));
    const firstFile = list.find((p) => isVideoFile(p));
    const start = folder ?? (firstFile ? folderOfPath(firstFile) : '');
    const picked = await pickPath({
      title: 'Choose the training videos',
      mode: 'file',
      multiple: true,
      selected,
      start,
      extensions: VIDEO_EXTENSIONS,
      confirmLabel: (n) => (n === 1 ? 'Use 1 video' : n ? `Use ${n} videos` : 'Use these videos'),
    });
    if (!Array.isArray(picked) || !picked.length) return;
    const { value, replaced } = mergePicked(gtValue(), picked, { example: Boolean(videoPrep.get().example) });
    try {
      await recipe.applyEdits([{ path: GT_PATH, value }]);
    } catch (error) {
      toast(`The videos were not chosen: ${error?.message ?? error}`, { kind: 'err' });
      return;
    }
    const names = replaced.map((p) => baseName(p) || p);
    const instead = names.length ? ` They replace the folder${names.length > 1 ? 's' : ''} ${names.join(', ')}.` : '';
    toast(`Chose ${fmt.plural(picked.length, 'video')} for training.${instead}`, { kind: 'ok' });
  }

  async function doCancel() {
    try {
      await videoPrep.cancel();
    } catch (error) {
      toast(`The preparation could not be cancelled: ${error?.message ?? error}`, { kind: 'err' });
    }
  }

  let lastPaths = null;
  let lastStatus = null;
  let lastFiles = null;

  function renderPaths(roots) {
    const key = JSON.stringify(roots);
    if (key === lastPaths) return;
    lastPaths = key;
    const groups = roots.length ? groupRoots(roots) : [];
    // File names need the row's width: with a group of files the two buttons move under the value.
    paths.parentElement?.classList.toggle('vprep__source--stacked', groups.some((item) => item.path === undefined));
    if (!roots.length) {
      mount(paths, h('p', { class: 'vprep__path vprep__path--none', role: 'listitem' }, 'No folder chosen'));
      return;
    }
    // Several files of one folder: the folder once, then their names, so ten files stay readable.
    mount(
      paths,
      groups.map((item) =>
        item.path !== undefined
          ? h('p', { class: 'vprep__path mono', role: 'listitem' }, item.path)
          : h(
              'div',
              { class: 'vprep__group', role: 'listitem' },
              h('p', { class: 'vprep__path mono' }, item.folder),
              h('p', { class: 'vprep__names' }, `${fmt.int(item.count)} video files: `, h('span', { class: 'mono' }, item.names.join(', ')), item.more ? ` and ${fmt.int(item.more)} more` : ''),
            ),
      ),
    );
  }

  function badgeFor(s) {
    if (jobActive(s.job)) return badge('Preparing', 'run');
    if (s.check.phase === 'checking') return badge('Checking');
    if (exampleUnchosen(s)) return badge('No folder chosen');
    if (s.roots.length && s.check.phase === 'error') return badge('Needs attention', 'err');
    if (!s.roots.length || s.check.phase !== 'done' || !s.check.answer) return badge('Not checked');
    const c = counts(s.check.answer);
    if (c.invalid) return badge(`${fmt.int(c.invalid)} cannot be used`, 'err');
    if (c.pending) return badge(`${fmt.int(c.pending)} of ${fmt.int(c.total)} need an index`, 'warn');
    return badge(`${fmt.int(c.ready)} ready`, 'ok');
  }

  /** The check's sentence; the box of a job that ended goes above it. */
  function statusParts(s) {
    const parts = [];
    const job = s.job;
    if (job && !jobActive(job)) {
      if (job.status === 'failed') {
        const again = h('button', { class: 'btn btn--sm vprep__btn', type: 'button', onclick: () => videoPrep.prepare({ again: Boolean(job.retry) }) }, 'Try again');
        const open = job.id ? h('a', { class: 'btn btn--sm btn--quiet vprep__btn', href: analysisHref(job.id) }, 'Open the job') : null;
        parts.push(box('err', job.error || 'The preparation failed.', again, open));
      } else if (job.status === 'cancelled') parts.push(box('', CANCELLED));
    }
    if (!s.roots.length) parts.push(h('p', { class: 'vprep__hint' }, 'Choose the folder that holds the training videos.'));
    else if (s.check.phase === 'unsupported') parts.push(box('warn', UNSUPPORTED));
    // A template's example path the person never chose: ask for a folder, do not show the refusal.
    else if (exampleUnchosen(s)) parts.push(box('', "This is the template's example path. Choose the folder that holds your training videos.", h('button', { class: 'btn btn--sm vprep__btn', type: 'button', onclick: () => chooseFolder() }, 'Choose folder…')));
    else if (s.check.phase === 'error') parts.push(box('err', s.check.error || 'The videos could not be checked.', h('button', { class: 'btn btn--sm vprep__btn', type: 'button', onclick: () => videoPrep.check({ force: true }) }, 'Check again')));
    else if (s.check.answer && !jobActive(job)) {
      const c = counts(s.check.answer);
      if (c.invalid) parts.push(box('err', `${fmt.int(c.invalid)} of ${fmt.plural(c.total, 'video')} cannot be used for clip training. Leave them out with Choose videos…, or change what the reason below names, then check again.`));
      else if (c.pending) {
        const stale = (Array.isArray(s.check.answer.files) ? s.check.answer.files : []).some((f) => f?.status === 'stale');
        parts.push(box('warn', `${fmt.int(c.pending)} of ${fmt.plural(c.total, 'video')} need an index. Preparing reads each one once on the trainer machine; indexed videos are reused.${stale ? " An index goes out of date when the video or the recipe's descale options change." : ''}`));
      }
      else if (c.total === 1) parts.push(box('ok', 'The video is indexed. Nothing to prepare.'));
      else if (c.total > 0) parts.push(box('ok', `All ${fmt.int(c.total)} videos are indexed. Nothing to prepare.`));
      else parts.push(box('warn', 'No videos were found here.'));
    } else if (s.check.phase === 'checking' && !s.check.answer) {
      parts.push(h('div', { class: 'loading vprep__loading' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), 'Checking the videos…'));
    }
    if (jobActive(job)) parts.push(h('p', { class: 'visually-hidden' }, 'Preparing the training videos.'));
    if (job?.status === 'completed' && s.outcome) {
      const written = Number.isFinite(s.outcome.written) ? s.outcome.written : null;
      const skipped = Number.isFinite(s.outcome.skipped) ? s.outcome.skipped : null;
      if (written !== null && skipped !== null) parts.push(h('p', { class: 'vprep__last' }, `Last preparation: ${fmt.plural(written, 'index', 'indexes')} written, ${fmt.int(skipped)} reused.`));
    }
    return parts;
  }

  function statusKey(s) {
    const job = s.job;
    return JSON.stringify([
      s.roots.length,
      exampleUnchosen(s),
      s.check.phase,
      s.check.error,
      s.check.answer ? [s.check.answer.total, s.check.answer.ready, s.check.answer.pending, s.check.answer.invalid, (s.check.answer.files ?? []).some?.((f) => f?.status === 'stale')] : null,
      job ? [job.id, jobActive(job), job.status === 'failed' || job.status === 'cancelled' || job.status === 'completed' ? job.status : '', job.error, job.retry] : null,
      s.outcome ? [s.outcome.written, s.outcome.skipped] : null,
    ]);
  }

  function renderFiles(answer) {
    if (answer === lastFiles) return;
    lastFiles = answer;
    const list = (Array.isArray(answer?.files) ? answer.files : []).filter((f) => f && typeof f === 'object');
    if (!list.length) {
      mount(files);
      return;
    }
    const sorted = [...list].sort((a, b) => (ORDER[a.status] ?? 2.5) - (ORDER[b.status] ?? 2.5));
    const ready = sorted.filter((f) => f.status === 'ready');
    const other = sorted.filter((f) => f.status !== 'ready');
    const row = (f) => {
      const [word, kind] = FILE_BADGE[f.status] ?? [String(f.status ?? 'Unknown'), ''];
      const reason = reasonOf(f);
      return h('li', { class: 'vprep__file' }, badge(word, kind), h('span', { class: 'vprep__name mono', title: typeof f.path === 'string' ? f.path : null }, String(f.name ?? f.path ?? '')), reason && f.status !== 'ready' ? h('span', { class: 'vprep__reason' }, reason) : null);
    };
    const shown = other.slice(0, ROW_LIMIT);
    const details = ready.length
      ? h(
          'details',
          { class: 'vprep__ready', open: readyOpen, ontoggle: (event) => (readyOpen = event.currentTarget.open) },
          h('summary', { class: 'vprep__summary' }, other.length ? `${fmt.int(ready.length)} ready` : ready.length === 1 ? 'Show the video' : `Show the ${fmt.int(ready.length)} videos`),
          h('ul', { class: 'vprep__list' }, ready.map(row)),
        )
      : null;
    mount(
      files,
      shown.length ? h('ul', { class: 'vprep__list', 'aria-label': 'Videos that need attention' }, shown.map(row)) : null,
      other.length > shown.length ? h('p', { class: 'vprep__more' }, `and ${fmt.int(other.length - shown.length)} more`) : null,
      details,
    );
  }

  function renderJob(s) {
    const job = s.job;
    const active = jobActive(job);
    jobBox.hidden = !active;
    actions.hidden = active;
    if (!active) return;
    const p = job.progress && typeof job.progress === 'object' ? job.progress : {};
    const total = Number.isFinite(p.total) && p.total > 0 ? p.total : null;
    const current = Number.isFinite(p.current) ? Math.min(p.current, total ?? p.current) : null;
    const counted = total !== null && current !== null;
    bar.hidden = !counted;
    if (counted) {
      bar.setAttribute('aria-valuemax', String(total));
      bar.setAttribute('aria-valuenow', String(current));
      bar.firstChild.style.width = `${Math.min(100, (current / total) * 100)}%`;
    } else {
      // No numbers yet (a new job starts without progress): nothing of an earlier bar stays behind.
      bar.removeAttribute('aria-valuemax');
      bar.removeAttribute('aria-valuenow');
      bar.firstChild.style.width = '0%';
    }
    const queued = job.status === 'queued';
    // What the service published, as it published it: the video being read, when the event says so.
    const name = counted && typeof p.episode === 'string' && p.episode ? p.episode : '';
    const episode = !name ? '' : p.action === 'indexing' ? `Indexing ${name}` : p.action === 'indexed' ? `Indexed ${name}` : p.action === 'skipped' ? `Reused the index of ${name}` : '';
    episodeLine.hidden = !episode;
    if (episodeLine.textContent !== episode) episodeLine.textContent = episode;
    const done = counted ? `${fmt.int(current)} of ${fmt.int(total)} done` : queued ? '' : 'Starting…';
    countLine.hidden = !done;
    if (countLine.textContent !== done) countLine.textContent = done;
    waitLine.hidden = !queued;
    waitLine.textContent = 'Waiting for another analysis job to finish.';
    pollLine.hidden = !job.pollError;
    pollLine.textContent = job.pollError ?? '';
    const cancelling = job.status === 'cancelling' || Boolean(job.cancelAsked);
    cancelButton.hidden = !job.id;
    cancelButton.disabled = cancelling;
    const cancelText = cancelling ? 'Cancelling…' : 'Cancel';
    if (cancelButton.textContent !== cancelText) cancelButton.textContent = cancelText;
    jobLink.hidden = !job.id;
    if (job.id) jobLink.setAttribute('href', analysisHref(job.id));
  }

  function render() {
    if (disposed) return;
    if (root.isConnected) attached = true;
    else if (attached) {
      // Thrown away with a rebuild of the form (or its screen): stop listening.
      cleanup();
      return;
    }
    const s = videoPrep.get();
    mount(badgeSlot, badgeFor(s));
    renderPaths(s.roots);
    const checking = s.check.phase === 'checking';
    checkButton.disabled = checking || !s.roots.length;
    const checkText = checking ? 'Checking…' : 'Check videos';
    if (checkButton.textContent !== checkText) checkButton.textContent = checkText;
    const c = counts(s.check.answer);
    const canPrepare = s.check.phase === 'done' && Boolean(s.check.answer) && c.fixable > 0 && c.invalid === 0;
    prepareButton.hidden = !canPrepare;
    const label = prepareLabel(c.fixable);
    if (prepareButton.textContent !== label) prepareButton.textContent = label;
    renderJob(s);
    const key = statusKey(s);
    if (key !== lastStatus) {
      // A button in the region that holds the focus is put back on its replacement.
      const focused = status.contains(document.activeElement) ? document.activeElement.textContent : null;
      lastStatus = key;
      mount(status, statusParts(s));
      if (focused) [...status.querySelectorAll('button, a')].find((el) => el.textContent === focused)?.focus({ preventScroll: true });
    }
    // While a job runs the rows would contradict it (it indexes them one by one): hidden until the next answer.
    renderFiles(jobActive(s.job) || s.check.phase === 'error' || s.check.phase === 'unsupported' || !s.roots.length ? null : s.check.answer);
  }

  const unsubscribe = videoPrep.subscribe(render);
  // The form builds its rows before putting them on the page.
  const firstPaint = setTimeout(render, 0);
  render();

  function cleanup() {
    if (disposed) return;
    disposed = true;
    clearTimeout(firstPaint);
    unsubscribe();
  }
  return cleanup;
}
