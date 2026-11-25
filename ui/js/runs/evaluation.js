// Two sides side by side (#/runs/compare/<left>[/<right>]). A side is a training run (its id) or a
// completed CPU evaluation job (`job:<id>`), or a TensorBoard event file (`tb:<encoded path>`):
//   1. the recorded scores of both (POST /api/compare), exactly as the service returns them: image by
//      image, or the whole-dataset score for tasks scored over the whole dataset;
//   2. both sides' saved outputs for one image on the comparison stage;
//   3. one logged series of both runs on one chart (GET /api/run/history; training runs and TensorBoard files);
//   4. the recipe options that differ (a run's yaml or a job's evaluation.yml, parsed by POST /api/preflight).
// Left and right keep their sides everywhere: --series-1 is the left side, --series-2 the right side.
//
//   const part = mountRunCompare(host, { leftId, rightId });  part.setRuns(left, right);  part.destroy();

import { h, mount, fmt, uid } from '../dom.js';
import { api, watchHistory } from '../api.js';
import { runs, runStatus } from '../state/runs.js';
import { catalog } from '../state/catalog.js';
import { go, compareHref, runHref, analysisHref } from '../actions.js';
import { notice, loading, emptyState } from '../ui.js';
import { pickPath, parentOf } from '../components/path-picker.js';
import { createChart } from '../components/chart.js';
import { createCompare } from '../components/compare.js';
import { diffConfigs, compactValue } from '../recipe/diff.js';

const SIDES = ['left', 'right'];
const SIDE_WORD = { left: 'Left', right: 'Right' };
const KIND_NOUN = { run: 'run', job: 'evaluation', tensorboard: 'TensorBoard run' };
const TB_PREFIX = 'tb:';
const JOB_PREFIX = 'job:';
const JOB_RECIPE = 'evaluation.yml'; // the recipe an evaluation job's evaluator ran with, among its files
const JOBS_MAX_AGE = 15000; // the list of evaluation jobs is read again when older than this (ms)
const HISTORY_BUDGET = 2048;
const DEFAULT_SERIES = 'l_g_total';
const POOLED = 'pooledDataset';
const POOLED_SENTENCE =
  'These are the evaluator’s scores over the whole dataset, not means of per-image scores, so there is no image-by-image list.';
const RECORD_HINT =
  'The recipe option val.record_samples turns the per-image record on. It saves the scores of the metrics validation already computes, so validation metrics (val.metrics_enabled) must be on as well.';

/**
 * Each file path is encoded inside its side token so slashes survive route decoding.
 * @returns {{kind: 'run'|'job'|'tensorboard', id: string} | null}
 */
function parseSide(token) {
  const text = token === null || token === undefined ? '' : String(token);
  if (!text) return null;
  if (text.startsWith(TB_PREFIX)) {
    try {
      let path = decodeURIComponent(text.slice(TB_PREFIX.length));
      if (baseName(path).startsWith('events.out.tfevents.')) path = parentOf(path);
      return path ? { kind: 'tensorboard', id: path } : null;
    } catch {
      return null;
    }
  }
  if (!text.startsWith(JOB_PREFIX)) return { kind: 'run', id: text };
  const id = text.slice(JOB_PREFIX.length);
  return id ? { kind: 'job', id } : null;
}

/** The same side as /api/compare takes it: the run id, or {job: id}. */
const sideParam = (ref) => (ref.kind === 'job' ? { job: ref.id } : ref.id);

const baseName = (path) => {
  const text = String(path ?? '').replace(/[\\/]+$/, '');
  const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
  return cut < 0 ? text : text.slice(cut + 1);
};

/** "net_g_8.safetensors, 20 Nov 2026, 21:55": what tells one evaluation job from another. */
function jobShortName(job) {
  const file = baseName(job?.params?.weights);
  const when = fmt.dateTime(job?.finishedAt ?? job?.createdAt);
  return [file, when].filter(Boolean).join(', ');
}

function jobLabel(job) {
  const file = baseName(job?.params?.weights);
  const when = fmt.dateTime(job?.finishedAt ?? job?.createdAt);
  if (!file) return `Evaluation job ${job?.id ?? ''}${when ? `, ${when}` : ''}`.trim();
  return `Evaluation of ${file}${when ? `, ${when}` : ''}`;
}

/** What each status of /api/compare counts means, in the order they are listed. */
const COUNT_TEXT = {
  matched: 'Recorded in both evaluations from the same source file, with a finite score on both sides. Only these images are compared.',
  leftOnly: 'Recorded only in the left evaluation, so those images are not compared.',
  rightOnly: 'Recorded only in the right evaluation, so those images are not compared.',
  unverifiedSource: 'At least one evaluation did not record which source file the image came from, so it cannot be confirmed to be the same image; not compared.',
  changedSource: 'The source image changed between the two evaluations, so those images are not compared.',
  nonfinite: 'The score is not a finite number on at least one side; not compared.',
  missingScore: 'The image was recorded, but this metric has no score for it on at least one side; not compared.',
};

/** The counts of a whole-dataset comparison (studio/pooled_comparison.py), in the order they are listed. */
const POOLED_COUNT_TEXT = {
  matched: 'Scored inputs recorded on both sides with the same source identity (image and target). The whole-dataset scores are compared only when this is every scored input of both sides.',
  recordedLeft: 'Scored inputs the left side recorded.',
  recordedRight: 'Scored inputs the right side recorded.',
  leftUnverified: 'Recorded on the left side without a source identity, so they cannot be confirmed to be the same inputs.',
  rightUnverified: 'Recorded on the right side without a source identity, so they cannot be confirmed to be the same inputs.',
};

/** Settings that must agree before two evaluations can be compared. */
const SETTING_TEXT = {
  definition: 'How the metric is defined (its settings)',
  input_protocol: 'How the validation inputs were prepared for the model',
  better: 'Which direction counts as better',
  // whole-dataset scores (classification, segmentation, detection)
  metric: 'How the metric is defined (its settings)',
  labels: 'The classes the evaluator scored (ids and names)',
  task: 'The task the evaluator scored',
  version: 'The version of the recorded evaluation protocol',
};

const isAbort = (error) => error?.name === 'AbortError';

function directionWords(better) {
  if (better === 'higher') return 'Higher is better';
  if (better === 'lower') return 'Lower is better';
  return 'Not reported';
}

/**
 * How one side was measured, from the `inference` record of a comparison ({device, precision,
 * tileSize, tileOverlap}); null when nothing readable was recorded. Unknown fields are shown as recorded.
 */
function inferenceText(settings) {
  if (!isContainer(settings) || Array.isArray(settings)) return null;
  const { device, precision, tileSize, tileOverlap, ...rest } = settings;
  const bits = [];
  if (typeof device === 'string' && device) bits.push(`device ${device}`);
  if (typeof precision === 'string' && precision) bits.push(`precision ${precision}`);
  if (tileSize === 0) bits.push('whole images, no tiles');
  else if (Number.isFinite(tileSize)) {
    bits.push(Number.isFinite(tileOverlap) ? `tiles of ${fmt.int(tileSize)} px overlapping by ${fmt.int(tileOverlap)} px` : `tiles of ${fmt.int(tileSize)} px`);
  }
  for (const [key, value] of Object.entries(rest)) {
    if (value !== null && value !== undefined) bits.push(`${key} ${leafText(value)}`);
  }
  return bits.length ? bits.join(', ') : null;
}

/** Whether two recorded settings objects hold the same values (key order does not matter). */
function sameSettings(a, b) {
  const text = (value) => JSON.stringify([...flattenLeaves(value).entries()].sort(([x], [y]) => (x < y ? -1 : x > y ? 1 : 0)));
  return text(a) === text(b);
}

const MAX_PLACES = 6;

/**
 * Decimal places for one group of numbers shown together (a column pair, a summary block): enough
 * to show 4 significant digits of the smallest non-zero magnitude in the group, at most 6, so the
 * numbers line up ("27.73 / 27.90", "+0.1750 / -0.4000").
 */
function placesFor(values) {
  let smallest = Infinity;
  for (const v of values) if (Number.isFinite(v) && v !== 0) smallest = Math.min(smallest, Math.abs(v));
  if (!Number.isFinite(smallest)) return 0;
  return Math.max(0, Math.min(MAX_PLACES, 3 - Math.floor(Math.log10(smallest))));
}

/** Scores are raw floats from the service: shown with the group's fixed places, the exact value in `title`. */
function score(n, places) {
  return Number.isFinite(n) ? n.toFixed(places) : 'Not reported';
}

/** Signed difference with the group's fixed places: "+0.1750", "-0.4000". */
function signed(n, places) {
  if (!Number.isFinite(n)) return 'Not reported';
  const text = n.toFixed(places);
  return n > 0 ? `+${text}` : text;
}

/** A difference for reading: signed, fixed places, a tie as "No difference". */
function delta(n, places) {
  if (n === 0) return 'No difference';
  return signed(n, places);
}

/**
 * One set of places for every score of a comparison (means and per-image scores) and one for every
 * difference (summary and per-image), so a column never mixes precisions.
 */
function comparisonPlaces(res) {
  const rows = [...(Array.isArray(res.improvements) ? res.improvements : []), ...(Array.isArray(res.regressions) ? res.regressions : [])];
  const s = res.summary ?? {};
  return {
    score: placesFor([s.meanLeft, s.meanRight, ...rows.flatMap((r) => [r.left, r.right])]),
    delta: placesFor([s.meanDelta, s.minDelta, s.maxDelta, ...rows.map((r) => r.delta)]),
  };
}

const isContainer = (v) => v !== null && typeof v === 'object';
const hasSide = (pair, side) => isContainer(pair) && Object.prototype.hasOwnProperty.call(pair, side);

/**
 * Leaf values of a recorded setting by path: dots for keys, [i] for list indices
 * ("evaluator", "crop.border", "channels[0]"). An empty object or list is a leaf of its own; a
 * value that is not an object or list is the single leaf at path ''.
 * @returns {Map<string, any>}
 */
function flattenLeaves(value, prefix = '', out = new Map()) {
  if (Array.isArray(value)) {
    if (!value.length) out.set(prefix, value);
    value.forEach((item, i) => flattenLeaves(item, `${prefix}[${i}]`, out));
  } else if (isContainer(value)) {
    const keys = Object.keys(value);
    if (!keys.length) out.set(prefix, value);
    for (const key of keys) flattenLeaves(value[key], prefix ? `${prefix}.${key}` : key, out);
  } else if (value !== undefined) {
    out.set(prefix, value);
  }
  return out;
}

/** Text of one recorded leaf value, exactly as recorded (strings without quotes unless empty). */
function leafText(value) {
  if (typeof value === 'string') return value === '' ? '""' : value;
  return JSON.stringify(value) ?? String(value);
}

/** A table cell for one side of a setting; "Not set" (muted) when that side has no such value. */
function leafCell(present, value) {
  return present ? h('td', { class: 'mono evaluation__value' }, leafText(value)) : h('td', { class: 'muted' }, 'Not set');
}

/** title text holding the exact value the service returned */
const exact = (n) => (Number.isFinite(n) ? `Exact value: ${n}` : null);

/** Last path segment of a recorded sample id ("0/0007.png" -> "0007.png"). */
const sampleLabel = (sample) => String(sample).split('/').filter(Boolean).pop() || String(sample);

function bytes(n) {
  if (!Number.isFinite(n)) return '';
  const units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB'];
  let value = n;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return i ? `${value.toFixed(1)} ${units[i]}` : `${fmt.int(n)} bytes`;
}

function swatch(side) {
  return h('span', { class: `evaluation__swatch evaluation__swatch--${side}`, 'aria-hidden': 'true' });
}

/** A child AbortController that also aborts when `parent` does. Call release() when finished. */
function childController(parent) {
  const ctl = new AbortController();
  const onAbort = () => ctl.abort();
  if (parent.aborted) ctl.abort();
  else parent.addEventListener('abort', onAbort, { once: true });
  ctl.release = () => parent.removeEventListener('abort', onAbort);
  return ctl;
}

function section(title, lead, ...children) {
  const titleId = uid('evaluation-title');
  return h(
    'section',
    { class: 'evaluation__section', 'aria-labelledby': titleId },
    h('h2', { class: 'evaluation__title', id: titleId }, title),
    lead ? h('p', { class: 'evaluation__lead' }, lead) : null,
    children,
  );
}

function taskWords(task) {
  if (!task) return '';
  try {
    return catalog.taskLabel(task) || task;
  } catch {
    return task;
  }
}

/** Where a run came from, in the words the pickers use; '' for a run this service launched. */
function originWord(run) {
  if (run?.source === 'archive') return 'added folder';
  if (run?.source === 'cli') return 'imported';
  return '';
}

function optionLabel(run) {
  const origin = originWord(run);
  const word = origin || String(run.status ?? '').toLowerCase() || 'status not reported';
  return `${run.name || run.id} (${word})`;
}

/**
 * Each id identifies a managed run, an evaluation job, or a TensorBoard event file.
 * @param {HTMLElement} host
 * @param {{leftId?: string|null, rightId?: string|null}} options
 * @returns {{setRuns(leftId: string|null, rightId: string|null): void, destroy(): void}}
 */
export function mountRunCompare(host, { leftId = null, rightId = null } = {}) {
  const ids = { left: null, right: null }; // the tokens, as the route and the pickers carry them
  const refs = { left: null, right: null }; // parsed side tokens
  const setSides = (left, right) => {
    refs.left = parseSide(left);
    refs.right = parseSide(right);
    ids.left = refs.left ? String(left) : null;
    ids.right = refs.right ? String(right) : null;
  };
  setSides(leftId, rightId);
  let pendingRight = null; // a right side chosen while the left is still empty (the route needs a left side)
  let pair = null; // {left, right (parsed sides), signal, ctl, detail, cleanups}
  let destroyed = false;

  // ───────── completed evaluation jobs (the pickers' second group, and each job side's facts) ─────────

  let jobs = [];
  let jobsPhase = 'idle'; // 'idle' | 'loading' | 'ready' | 'error'
  let jobsError = '';
  let jobsAt = 0;
  let jobsShown = '';
  const jobsCtl = new AbortController();
  const jobDetail = new Map(); // job id -> {job, result, artifacts, settled, promise}

  async function loadJobs() {
    if (destroyed || jobsPhase === 'loading') return;
    if (jobsAt && Date.now() - jobsAt < JOBS_MAX_AGE) return;
    const first = jobsPhase === 'idle';
    if (first) jobsPhase = 'loading';
    jobsAt = Date.now();
    try {
      const list = await api.analysisJobs({ signal: jobsCtl.signal });
      if (destroyed) return;
      jobs = (Array.isArray(list) ? list : []).filter((job) => job?.kind === 'evaluate' && job.status === 'completed');
      jobsPhase = 'ready';
      jobsError = '';
    } catch (error) {
      if (destroyed || isAbort(error)) return;
      // A list read earlier stays in use; only a list that was never read is reported as unreadable.
      if (jobsPhase !== 'ready') {
        jobsPhase = 'error';
        jobsError = error?.message ?? 'no answer';
      }
    }
    // Nothing is touched when the answer is the one already shown (this also runs on background refreshes).
    const shown = [jobsPhase, jobsError, ...jobs.map((job) => `${job.id}=${jobLabel(job)}`)].join('|');
    if (shown === jobsShown) return;
    jobsShown = shown;
    fillSelects();
    updateFacts();
    if (!pair) renderIdle();
  }

  function jobOf(id) {
    return jobs.find((job) => job.id === id) ?? jobDetail.get(id)?.job ?? null;
  }

  /** The job and its result, read once per job for the facts card and the recipe part. */
  function readJob(id) {
    const known = jobDetail.get(id);
    if (known && (!known.settled || (known.job && known.result))) return known.promise;
    const entry = { job: known?.job ?? null, result: null, artifacts: [], settled: false, promise: null };
    entry.promise = Promise.allSettled([api.analysisJob(id, { signal: jobsCtl.signal }), api.analysisResult(id, {}, { signal: jobsCtl.signal })]).then(
      ([job, result]) => {
        if (job.status === 'fulfilled' && job.value && typeof job.value === 'object') entry.job = job.value;
        if (result.status === 'fulfilled') {
          entry.result = isContainer(result.value?.result) ? result.value.result : null;
          entry.artifacts = Array.isArray(result.value?.artifacts) ? result.value.artifacts : [];
        }
        entry.settled = true;
        if (!destroyed) {
          fillSelects();
          updateFacts();
          if (!pair) renderIdle(); // the "same on both sides" sentence names the job
        }
        return entry;
      },
    );
    jobDetail.set(id, entry);
    return entry.promise;
  }

  // ───────── side pickers ─────────

  const selectIds = { left: uid('evaluation-left'), right: uid('evaluation-right') };
  const selects = {};
  let lastLogFolder = '';
  async function browseLog(side) {
    const ref = parseSide(currentValue(side));
    const path = await pickPath({
      title: 'Choose a TensorBoard run folder',
      mode: 'folder',
      showFilesInitially: true,
      start: ref?.kind === 'tensorboard' ? ref.id : lastLogFolder,
      confirmLabel: 'Compare full history',
    });
    if (!path || destroyed) return;
    lastLogFolder = parentOf(path);
    choose(side, TB_PREFIX + encodeURIComponent(path));
  }
  const optionSignature = { left: '', right: '' };
  for (const side of SIDES) {
    selects[side] = h('select', {
      class: 'select evaluation__select',
      id: selectIds[side],
      onchange: () => choose(side, selects[side].value),
      onblur: () => fillSelect(side),
    });
  }
  const swapBtn = h('button', { class: 'btn evaluation__swap', type: 'button', onclick: swap }, 'Swap');
  const picker = h(
    'div',
    { class: 'evaluation__picker' },
    SIDES.map((side) =>
      h(
        'div',
        { class: 'evaluation__pick' },
        h('label', { class: 'evaluation__label', for: selectIds[side] }, swatch(side), SIDE_WORD[side]),
        selects[side],
        h('button', { class: 'btn', type: 'button', onclick: () => browseLog(side),
          'aria-label': `Browse TensorBoard run folder for the ${side} side` }, 'Browse TensorBoard run folder…'),
      ),
    ),
    swapBtn,
  );

  const facts = h('div', { class: 'evaluation__facts' });
  const body = h('div', { class: 'evaluation__body' });
  const root = h('div', { class: 'evaluation' }, picker, facts, body);
  host.appendChild(root);

  function currentValue(side) {
    return side === 'right' && !ids.left ? pendingRight : ids[side];
  }

  function fillSelect(side) {
    const select = selects[side];
    if (document.activeElement === select) return; // rebuilt on blur; an open list is never replaced
    const list = runs.get().list ?? [];
    const value = currentValue(side) ?? '';
    const ref = parseSide(value);
    const runEntries = list.map((run) => [run.id, optionLabel(run)]);
    const jobEntries = jobs.map((job) => [`${JOB_PREFIX}${job.id}`, jobLabel(job)]);
    if (ref?.kind === 'run' && !list.some((run) => run.id === ref.id)) runEntries.unshift([value, `${value} (not in the run list)`]);
    if (ref?.kind === 'job' && !jobs.some((job) => job.id === ref.id)) {
      const job = jobOf(ref.id);
      // Only a list that was read can say the job is not among the completed evaluation jobs.
      const label = job ? jobLabel(job) : `Evaluation job ${ref.id}`;
      jobEntries.unshift([value, jobsPhase === 'ready' ? `${label} (not in the list of completed evaluation jobs)` : label]);
    }
    // A list that could not be read says so even when the chosen job is known from its own record.
    let jobNote = '';
    if (jobsPhase === 'error') jobNote = `Could not read the evaluation jobs: ${jobsError}`;
    else if (!jobEntries.length) jobNote = jobsPhase === 'ready' ? 'No completed evaluation job yet' : 'Reading the evaluation jobs…';
    // The left side can only be changed, not cleared: the route always names a left side.
    const placeholder = side === 'right' || !value ? 'Choose a run or an evaluation job' : '';
    const signature = [placeholder, value, ...runEntries.map((e) => e.join('=')), '#', ...jobEntries.map((e) => e.join('=')), jobNote].join('|');
    if (signature === optionSignature[side]) return;
    optionSignature[side] = signature;
    const options = (entries) => entries.map(([id, label]) => h('option', { value: id }, label));
    mount(
      select,
      placeholder ? h('option', { value: '' }, placeholder) : null,
      ref?.kind === 'tensorboard' ? h('optgroup', { label: 'TensorBoard' }, h('option', { value }, baseName(ref.id))) : null,
      h('optgroup', { label: 'Training runs' }, runEntries.length ? options(runEntries) : h('option', { disabled: true }, 'No runs yet')),
      h('optgroup', { label: 'Evaluation jobs' }, options(jobEntries), jobNote ? h('option', { disabled: true }, jobNote) : null),
    );
    select.value = value;
  }

  function fillSelects() {
    for (const side of SIDES) fillSelect(side);
    swapBtn.disabled = !(ids.left && ids.right);
  }

  function choose(side, value) {
    const next = { left: ids.left, right: ids.left ? ids.right : pendingRight, [side]: value || null };
    if (next.left) {
      go(compareHref(next.left, next.right));
      return;
    }
    pendingRight = next.right;
    fillSelects();
    renderIntro();
  }

  function swap() {
    if (ids.left && ids.right) go(compareHref(ids.right, ids.left));
  }

  // ───────── facts row ─────────

  const factRefs = {};

  function runOf(side) {
    const ref = refs[side];
    if (ref?.kind !== 'run') return null;
    return runs.byId(ref.id) ?? pair?.detail?.[side] ?? null;
  }

  /** The side's name standing on its own (headings, column heads): a run's name, or "Evaluation of <file>, <date>". */
  function nameOf(side) {
    const ref = refs[side];
    if (!ref) return '';
    if (ref.kind === 'tensorboard') return baseName(ref.id);
    if (ref.kind === 'job') {
      const job = jobOf(ref.id);
      return job ? jobLabel(job) : `Evaluation job ${ref.id}`;
    }
    return runOf(side)?.name || ref.id;
  }

  /** "run" or "evaluation": what the side is called inside a sentence. */
  const nounOf = (side) => KIND_NOUN[refs[side]?.kind] ?? 'run';

  /** The side's name after its noun, where "Evaluation of" would repeat the noun. */
  function shortNameOf(side) {
    const ref = refs[side];
    if (ref?.kind !== 'job') return nameOf(side);
    return jobShortName(jobOf(ref.id)) || `job ${ref.id}`;
  }

  /** "left run (name)" / "left evaluation (file, date)". */
  const sideRef = (side) => `${side} ${nounOf(side)} (${shortNameOf(side)})`;

  /** "The two runs", "The two evaluations", "The evaluation and the run". */
  function bothWords() {
    const [l, r] = SIDES.map(nounOf);
    return l === r ? `The two ${l}s` : `The ${l} and the ${r}`;
  }

  function buildFacts() {
    for (const side of SIDES) delete factRefs[side];
    if (!ids.left || !ids.right) {
      mount(facts);
      return;
    }
    const cards = SIDES.map((side) => {
      const ref = refs[side];
      if (ref.kind === 'tensorboard') {
        return h('div', { class: `evaluation__fact evaluation__fact--${side}` },
          h('div', { class: 'evaluation__fact-head' }, swatch(side),
            h('span', { class: 'evaluation__side-word' }, `${SIDE_WORD[side]} TensorBoard run`),
            h('span', { class: 'evaluation__fact-name' }, nameOf(side))),
          h('div', { class: 'evaluation__value mono' }, ref.id));
      }
      if (ref.kind === 'job') {
        readJob(ref.id);
        const parts = {
          kind: 'job',
          name: h('a', { class: 'evaluation__fact-name', href: analysisHref(ref.id) }),
          checkpoint: h('dd', { class: 'evaluation__value' }),
          data: h('dd', null),
          measured: h('dd', null),
          finished: h('dd', null),
        };
        factRefs[side] = parts;
        return h(
          'div',
          { class: `evaluation__fact evaluation__fact--${side}` },
          h('div', { class: 'evaluation__fact-head' }, swatch(side), h('span', { class: 'evaluation__side-word' }, `${SIDE_WORD[side]} evaluation job`), parts.name),
          h(
            'dl',
            { class: 'kv evaluation__kv' },
            h('dt', null, 'Checkpoint'), parts.checkpoint,
            h('dt', null, 'Data'), parts.data,
            h('dt', null, 'Measured on'), parts.measured,
            h('dt', null, 'Finished'), parts.finished,
          ),
        );
      }
      const parts = {
        kind: 'run',
        name: h('a', { class: 'evaluation__fact-name', href: runHref(ref.id) }),
        task: h('dd', null),
        status: h('dd', null),
        iteration: h('dd', { class: 'num' }),
        speedLabel: h('dt', null, 'Reported speed'),
        speed: h('dd', { class: 'num' }),
      };
      factRefs[side] = parts;
      return h(
        'div',
        { class: `evaluation__fact evaluation__fact--${side}` },
        h('div', { class: 'evaluation__fact-head' }, swatch(side), h('span', { class: 'evaluation__side-word' }, `${SIDE_WORD[side]} run`), parts.name),
        h(
          'dl',
          { class: 'kv evaluation__kv' },
          h('dt', null, 'Task'), parts.task,
          h('dt', null, 'Status'), parts.status,
          h('dt', null, 'Iteration'), parts.iteration,
          parts.speedLabel, parts.speed,
        ),
      );
    });
    mount(facts, cards);
    updateFacts();
  }

  /** Facts of an evaluation job: only what the job and its result say; the rest is "Not reported". */
  function updateJobFacts(side, parts, set) {
    const id = refs[side].id;
    const detail = jobDetail.get(id);
    const job = jobOf(id);
    const result = detail?.result ?? null;
    const reading = detail && !detail.settled;
    set(parts.name, nameOf(side));
    parts.name.setAttribute('href', analysisHref(id));
    const weights = job?.params?.weights ?? result?.weights;
    set(parts.checkpoint, (typeof weights === 'string' && baseName(weights)) || (reading ? 'Reading…' : 'Not reported'));
    if (typeof weights === 'string' && weights) parts.checkpoint.setAttribute('title', weights);
    else parts.checkpoint.removeAttribute('title');
    const split = job?.params?.split;
    let data = reading ? 'Reading…' : 'Not reported';
    if (typeof split === 'string' && split) data = split;
    else if (job?.params && (split === null || split === undefined)) data = 'All validation and test data';
    set(parts.data, data);
    const measured = [result?.device, result?.precision].filter((v) => typeof v === 'string' && v).join(', ');
    set(parts.measured, measured || (reading ? 'Reading…' : 'Not reported'));
    set(parts.finished, fmt.dateTime(job?.finishedAt) || (reading ? 'Reading…' : 'Not reported'));
  }

  /** Text updates in place, so a focused name link keeps its focus during background refreshes. */
  function updateFacts() {
    for (const side of SIDES) {
      const parts = factRefs[side];
      if (!parts || !ids[side]) continue;
      const set = (el, text) => {
        if (el.textContent !== text) el.textContent = text;
      };
      if (parts.kind === 'job') {
        updateJobFacts(side, parts, set);
        continue;
      }
      const run = runOf(side);
      const runId = refs[side].id;
      set(parts.name, run?.name || runId);
      parts.name.setAttribute('href', runHref(runId));
      set(parts.task, taskWords(run?.task) || 'Not reported');
      const status = run ? runStatus(run.status) : null;
      const origin = run?.source === 'archive' ? ', from an added folder' : run?.source === 'cli' && !status?.imported ? ', imported' : '';
      set(parts.status, status ? `${status.label}${origin}` : 'Not reported');
      const it = run?.iteration;
      const total = run?.total;
      set(
        parts.iteration,
        Number.isFinite(it) && Number.isFinite(total) && total > 0
          ? `${fmt.int(it)} of ${fmt.int(total)}`
          : Number.isFinite(it)
            ? fmt.int(it)
            : 'Not reported',
      );
      // A run that is not training any more keeps its last reported speed, which is no longer current.
      set(parts.speedLabel, status?.active ? 'Reported speed' : 'Last reported speed');
      set(parts.speed, Number.isFinite(run?.iterationsPerSecond) ? `${fmt.num(run.iterationsPerSecond, 3)} it/s` : 'Not reported');
    }
  }

  // ───────── fewer than two sides ─────────

  function renderIntro() {
    const next = ids.left && !ids.right ? 'Choose the right side' : pendingRight ? 'Choose the left side' : 'Choose both sides';
    mount(body, emptyState({
      title: 'Compare runs and evaluations',
      text: `${next} above. Browse a TensorBoard run folder to compare its logged values over iterations. Runs and evaluation jobs can also compare recorded scores, images and recipes.`,
    }));
  }

  function renderSame() {
    mount(body, h('p', { class: 'evaluation__lead', role: 'status' },
      `Both sides are the same ${nounOf('left')}. Choose a different source on one side.`));
  }

  /** What the body shows while there is no pair to compare. */
  function renderIdle() {
    if (ids.left && ids.left === ids.right) renderSame();
    else renderIntro();
  }

  // ───────── one pair ─────────

  function endPair() {
    if (!pair) return;
    pair.ctl.abort();
    for (const fn of pair.cleanups.splice(0)) {
      try {
        fn();
      } catch {
        // A part that fails to clean up must not keep the others from cleaning up.
      }
    }
    pair = null;
  }

  function startPair() {
    endPair();
    loadJobs();
    fillSelects();
    buildFacts();
    if (!ids.left || !ids.right) {
      renderIntro();
      return;
    }
    if (ids.left === ids.right && refs.left.kind !== 'tensorboard') {
      // A TensorBoard folder can show a different writer on each side.
      renderSame();
      return;
    }
    const ctl = new AbortController();
    pair = { left: refs.left, right: refs.right, ctl, signal: ctl.signal, detail: {}, cleanups: [] };
    const current = pair;
    if (SIDES.some((side) => current[side].kind === 'tensorboard')) {
      mount(body, historyPart(current));
      return;
    }
    const outputs = outputsPart(current);
    const evaluation = evaluationPart(current, outputs);
    const history = historyPart(current);
    const recipes = recipePart(current);
    mount(body, evaluation, outputs.el, history, recipes);
  }

  const live = (p) => !destroyed && pair === p && !p.signal.aborted;

  // ───────── 2. recorded evaluation ─────────

  const listOf = (ref, opts) => (ref.kind === 'job' ? api.analysisEvaluations(ref.id, opts) : api.evaluations(ref.id, opts));

  function evaluationPart(p, outputs) {
    const status = h('div', { class: 'evaluation__status', role: 'status', 'aria-live': 'polite' }, loading('Reading the recorded evaluations…'));
    const controls = h('div', { class: 'evaluation__controls' });
    const result = h('div', { class: 'evaluation__result' });
    // One short line is announced when a result arrives; the tables themselves are not read out.
    const announce = h('p', { class: 'visually-hidden', role: 'status', 'aria-live': 'polite' });
    p.announce = announce;
    const el = section(
      'Recorded evaluation',
      'The scores each side recorded, compared image by image on the images both share. A task scored over the whole dataset (classification, segmentation, detection) is compared by that one score instead.',
      status,
      controls,
      announce,
      result,
    );

    // A job side's own record is awaited too, so every sentence below can name the job (readJob never rejects).
    const named = Promise.all(SIDES.filter((side) => p[side].kind === 'job').map((side) => readJob(p[side].id)));
    Promise.allSettled(SIDES.map((side) => listOf(p[side], { signal: p.signal }))).then(async (settled) => {
      await named;
      if (!live(p)) return;
      const lists = {};
      const problems = [];
      settled.forEach((outcome, i) => {
        const side = SIDES[i];
        if (outcome.status === 'fulfilled') lists[side] = Array.isArray(outcome.value) ? outcome.value : [];
        else if (!isAbort(outcome.reason)) problems.push(`${SIDE_WORD[side]} ${nounOf(side)} (${shortNameOf(side)}): ${outcome.reason?.message ?? 'no answer'}`);
      });
      if (problems.length) {
        mount(status, notice('err', h('span', null, 'Could not read the recorded evaluations. ', problems.join(' '))));
        return;
      }
      const empty = SIDES.filter((side) => !lists[side].length);
      if (empty.length) {
        // The recipe option only helps a training run; an evaluation job records what its task allows.
        const hint = empty.some((side) => p[side].kind === 'run') ? ` ${RECORD_HINT}` : '';
        mount(
          status,
          notice(
            'warn',
            h(
              'span',
              null,
              empty
                .map((side) => (p[side].kind === 'job' ? `The ${sideRef(side)} recorded no scores that can be compared. ` : `The ${sideRef(side)} has no recorded evaluation. `))
                .join(''),
              'Recorded evaluations are what this part compares.',
              hint,
            ),
          ),
        );
        return;
      }
      offerPickers(p, lists, status, controls, result, outputs);
    }).catch((error) => {
      if (!live(p)) return;
      mount(status, notice('err', `Could not show the recorded evaluations: ${error?.message ?? 'unknown error'}`));
    });
    return el;
  }

  function offerPickers(p, lists, status, controls, result, outputs) {
    // dataset + metric pairs that both sides recorded. Each side's measurements are listed newest
    // iteration first; several measurements of one iteration are told apart by their evaluation id.
    const byKey = (list) => {
      const map = new Map();
      for (const row of list) {
        const key = `${row.dataset}\u0000${row.metric}`;
        if (!map.has(key)) map.set(key, { dataset: row.dataset, metric: row.metric, better: row.better, aggregation: row.aggregation, steps: new Map() });
        const steps = map.get(key).steps;
        if (!steps.has(row.step)) steps.set(row.step, []);
        const group = steps.get(row.step);
        if (!group.some((m) => m.evaluation === row.evaluation)) group.push({ step: row.step, sampleCount: row.sampleCount, evaluation: row.evaluation });
      }
      for (const entry of map.values()) {
        entry.choices = [...entry.steps.keys()]
          .sort((a, b) => b - a)
          .flatMap((step) => {
            const group = entry.steps.get(step);
            // Without an evaluation id on every row the measurements cannot be named, so the iteration stays one choice.
            if (group.length < 2 || !group.every((m) => Number.isInteger(m.evaluation))) return [{ ...group[0], index: 1, of: 1 }];
            return group
              .slice()
              .sort((a, b) => a.evaluation - b.evaluation)
              .map((m, i) => ({ ...m, index: i + 1, of: group.length }));
          });
        entry.repeated = entry.choices.some((c) => c.of > 1);
      }
      return map;
    };
    const left = byKey(lists.left);
    const right = byKey(lists.right);
    const maps = { left, right };
    const common = [...left.keys()].filter((key) => right.has(key));
    if (!common.length) {
      const names = (map) => [...map.values()].map((e) => `${e.dataset}, ${e.metric}`).join('; ');
      mount(
        status,
        notice(
          'warn',
          h(
            'span',
            null,
            `${bothWords()} recorded no dataset and metric in common, so their scores cannot be compared. `,
            'Scores are matched by the name each recipe gives its dataset. ',
            `Left: ${names(left)}. Right: ${names(right)}.`,
          ),
        ),
      );
      return;
    }
    mount(status);

    // Which of an iteration's measurements an answer names ("measurement 1 of 2"), from the lists read above.
    p.measurementOf = (side, res) => {
      const entry = maps[side].get(`${res.dataset}\u0000${res.metric}`);
      const choice = entry?.choices.find((c) => c.step === res[`${side}Step`] && c.evaluation === res[`${side}Evaluation`]);
      return choice && choice.of > 1 ? choice : null;
    };

    const kinds = { left: p.left.kind, right: p.right.kind };
    const runSides = SIDES.filter((side) => kinds[side] === 'run');
    const pairId = uid('evaluation-pair');
    const pairSelect = h(
      'select',
      { class: 'select evaluation__select', id: pairId, onchange: () => onPairChange() },
      common.map((key) => h('option', { value: key }, `${left.get(key).dataset}, ${left.get(key).metric}`)),
    );
    pairSelect.value = common[0];
    const modeName = uid('evaluation-mode');
    const latestId = uid('evaluation-latest');
    const explicitId = uid('evaluation-explicit');
    const latestRadio = h('input', { type: 'radio', name: modeName, id: latestId, value: 'latest', checked: true, onchange: () => onModeChange() });
    const explicitRadio = h('input', { type: 'radio', name: modeName, id: explicitId, value: 'explicit', onchange: () => onModeChange() });
    // Two runs meet at their latest common iteration; next to an evaluation job a run is taken at
    // its latest measurement. An evaluation job has one measurement and no iteration to choose.
    const words =
      runSides.length === 1
        ? { legend: `Iteration of the ${runSides[0]} run`, latest: 'Its latest measurement', explicit: 'Choose an iteration' }
        : { legend: 'Iterations', latest: 'Latest common iteration', explicit: 'Choose an iteration for each run' };
    const stepIds = { left: uid('evaluation-step-left'), right: uid('evaluation-step-right') };
    const stepSelects = {};
    const stepPicks = {};
    for (const side of SIDES) {
      stepSelects[side] = h('select', { class: 'select evaluation__select', id: stepIds[side], onchange: () => compareNow() });
      stepPicks[side] = h(
        'div',
        { class: 'evaluation__pick', hidden: true },
        h('label', { class: 'evaluation__label', for: stepIds[side] }, swatch(side), `${SIDE_WORD[side]} ${kinds[side] === 'job' ? 'measurement' : 'iteration'}`),
        stepSelects[side],
      );
    }
    const stepFields = h('div', { class: 'evaluation__steps', hidden: true }, SIDES.map((side) => stepPicks[side]));
    const oneNote = h('p', { class: 'evaluation__quiet', hidden: true });

    mount(
      controls,
      h(
        'div',
        { class: 'evaluation__pick' },
        h('label', { class: 'evaluation__label', for: pairId }, 'Dataset and metric'),
        pairSelect,
      ),
      runSides.length
        ? h(
            'fieldset',
            { class: 'evaluation__fieldset' },
            h('legend', { class: 'evaluation__legend' }, words.legend),
            h('label', { class: 'evaluation__radio', for: latestId }, latestRadio, words.latest),
            h('label', { class: 'evaluation__radio', for: explicitId }, explicitRadio, words.explicit),
          )
        : null,
      stepFields,
      oneNote,
    );

    const entryOf = (side) => maps[side].get(pairSelect.value);
    const choiceValue = (c) => (c.of > 1 ? `${c.step}:${c.evaluation}` : String(c.step));
    // A run's iteration is chosen only on request; an evaluation job's measurement only when it has several.
    const needsChoice = (side) => (kinds[side] === 'run' ? explicitRadio.checked : entryOf(side).choices.length > 1);
    const chosen = (side) => (needsChoice(side) ? (entryOf(side).choices.find((c) => choiceValue(c) === stepSelects[side].value) ?? null) : null);

    function showFields() {
      let any = false;
      for (const side of SIDES) {
        const show = needsChoice(side);
        stepPicks[side].hidden = !show;
        any = any || show;
      }
      stepFields.hidden = !any;
    }

    function fillSteps() {
      const single = [];
      for (const side of SIDES) {
        const entry = entryOf(side);
        const isJob = kinds[side] === 'job';
        const unit = entry.aggregation === POOLED ? 'sample' : 'image';
        mount(
          stepSelects[side],
          entry.choices.map((c) => {
            let text = fmt.int(c.step);
            if (isJob) text = `Measurement ${c.index} of ${c.of}`;
            else if (entry.repeated) text = c.of > 1 ? `Iteration ${fmt.int(c.step)}, measurement ${c.index} of ${c.of}` : `Iteration ${fmt.int(c.step)}`;
            const count = Number.isFinite(c.sampleCount) ? ` (${fmt.plural(c.sampleCount, unit)})` : '';
            return h('option', { value: choiceValue(c), title: c.of > 1 ? `evaluation ${c.evaluation}` : null }, `${text}${count}`);
          }),
        );
        if (entry.choices.length) stepSelects[side].value = choiceValue(entry.choices[0]); // newest first
        if (isJob && entry.choices.length < 2) single.push(side);
      }
      if (single.length === 2) oneNote.textContent = 'Each evaluation job has one measurement, so there is no iteration to choose.';
      else if (single.length) oneNote.textContent = `The ${sideRef(single[0])} is compared by its one measurement.`;
      oneNote.hidden = !single.length;
      showFields();
    }

    function onPairChange() {
      fillSteps();
      compareNow();
    }

    function onModeChange() {
      showFields();
      compareNow();
    }

    let ctl = null;
    let token = 0;
    p.cleanups.push(() => ctl?.abort());

    async function compareNow() {
      ctl?.abort();
      ctl?.release();
      ctl = childController(p.signal);
      const mine = ++token;
      const entry = left.get(pairSelect.value);
      const params = { left: sideParam(p.left), right: sideParam(p.right), dataset: entry.dataset, metric: entry.metric };
      for (const side of SIDES) {
        const choice = chosen(side);
        if (!choice) continue;
        // An evaluation job has no iteration to send; an evaluation id goes only with a choice among several.
        if (kinds[side] === 'run') params[`${side}Step`] = choice.step;
        if (choice.of > 1) params[`${side}Evaluation`] = choice.evaluation;
      }
      outputs.reset();
      mount(result, loading('Comparing the recorded scores…'));
      try {
        const reply = await api.compare(params, { signal: ctl.signal });
        if (mine !== token || !live(p)) return;
        mount(result, renderComparison(reply, outputs, p));
        p.announce.textContent = announceText(reply, p);
      } catch (error) {
        if (isAbort(error) || mine !== token || !live(p)) return;
        if (error?.status === 400) {
          // The service's refusal in its own words, then what to do about it here when that is known.
          let extra = '';
          if (/val\.record_samples/.test(error.message)) extra = ` ${RECORD_HINT}`;
          else if (/^Multiple attempts recorded this iteration/.test(error.message) && runSides.length && !explicitRadio.checked) {
            extra = ` Choose “${words.explicit}” above to pick one measurement of that iteration.`;
          }
          mount(result, notice('warn', `${error.message}${extra}`));
        } else {
          mount(result, notice('err', `Could not compare the evaluations: ${error?.message ?? 'no answer'}`));
        }
      }
    }

    fillSteps();
    compareNow();
  }

  /** What one side of an answer was measured at, in words: never the raw step of an evaluation job (-1). */
  function measuredWords(p, side, res) {
    const choice = p.measurementOf?.(side, res) ?? null;
    const repeat = choice ? `measurement ${choice.index} of ${choice.of}` : '';
    const title = choice ? `evaluation ${choice.evaluation}` : null;
    if (p[side].kind === 'job') return { text: choice ? `Measurement ${choice.index} of ${choice.of}` : 'Its one measurement', title };
    const step = res[`${side}Step`];
    if (!Number.isFinite(step) || step < 0) return { text: 'Not reported', title };
    return { text: repeat ? `${fmt.int(step)}, ${repeat}` : fmt.int(step), title };
  }

  function announceText(reply, p) {
    if (reply?.comparable !== true) return `${bothWords()} could not be compared; the reason is shown below.`;
    if (reply.aggregation === POOLED) return 'The whole-dataset scores of both sides are shown below.';
    const s = reply.summary ?? {};
    const where = SIDES.map((side) => (p[side].kind === 'job' ? `the ${side} evaluation` : `iteration ${measuredWords(p, side, reply).text} of the ${side} run`)).join(' with ');
    return `Compared ${where}: ${fmt.int(s.improved)} improved, ${fmt.int(s.tied)} tied, ${fmt.int(s.regressed)} regressed.`;
  }

  /** One quiet line per side saying how it was measured, and a warning when the two differ. */
  function inferenceLines(res) {
    const recorded = isContainer(res.inference) ? res.inference : {};
    if (!SIDES.some((side) => isContainer(recorded[side]))) return [];
    const lines = SIDES.map((side) => {
      const text = isContainer(recorded[side]) ? inferenceText(recorded[side]) : null;
      return h(
        'p',
        { class: 'evaluation__quiet' },
        swatch(side),
        text ? ` ${SIDE_WORD[side]} measured with: ${text}.` : ` ${SIDE_WORD[side]}: how it was measured was not recorded.`,
      );
    });
    const out = [h('div', { class: 'evaluation__measured' }, lines)];
    if (SIDES.every((side) => isContainer(recorded[side])) && !sameSettings(recorded.left, recorded.right)) {
      out.push(notice('warn', 'The two sides were not measured the same way: the device, precision or tile settings recorded above differ.'));
    }
    return out;
  }

  function renderComparison(res, outputs, p) {
    const parts = [];
    const pooled = res.aggregation === POOLED;
    const summary = !pooled && res.comparable === true ? res.summary : null;
    const places = comparisonPlaces(res);
    const bothRuns = SIDES.every((side) => p[side].kind === 'run');

    // what was compared, per side
    // a cell is text, or [text, title] when the shown text is rounded or stands for an id
    const row = (label, cells, cls) =>
      h(
        'tr',
        null,
        h('th', { scope: 'row' }, label),
        cells.map((cell) => (Array.isArray(cell) ? h('td', { class: cls, title: cell[1] }, cell[0]) : h('td', { class: cls }, cell))),
      );
    const measured = SIDES.map((side) => measuredWords(p, side, res));
    parts.push(
      h(
        'dl',
        { class: 'kv evaluation__kv' },
        h('dt', null, 'Dataset'), h('dd', { class: 'mono' }, String(res.dataset ?? 'Not reported')),
        h('dt', null, 'Metric'), h('dd', { class: 'mono' }, String(res.metric ?? 'Not reported')),
        h('dt', null, 'Direction'), h('dd', null, directionWords(res.better)),
      ),
      h(
        'table',
        { class: 'table evaluation__table' },
        h('caption', { class: 'visually-hidden' }, 'What each side of the comparison recorded'),
        h(
          'thead',
          null,
          h(
            'tr',
            null,
            h('th', { scope: 'col' }, h('span', { class: 'visually-hidden' }, 'Recorded value')),
            SIDES.map((side) => h('th', { scope: 'col', class: 'num' }, swatch(side), ` ${SIDE_WORD[side]}: ${nameOf(side)}`)),
          ),
        ),
        h(
          'tbody',
          null,
          row(bothRuns ? 'Iteration' : 'Iteration or measurement', measured.map((m) => [m.text, m.title]), 'num'),
          row(pooled ? 'Samples in the evaluation' : 'Images in the evaluation', [fmt.int(res.leftCount) || 'Not reported', fmt.int(res.rightCount) || 'Not reported'], 'num'),
          summary
            ? row(
                'Mean score over the matched images',
                [
                  [score(summary.meanLeft, places.score), exact(summary.meanLeft)],
                  [score(summary.meanRight, places.score), exact(summary.meanRight)],
                ],
                'num',
              )
            : null,
        ),
      ),
      inferenceLines(res),
    );

    // settings differ: nothing else is comparable
    if (res.comparable === false && res.differences && typeof res.differences === 'object') {
      const entries = Object.entries(res.differences);
      const scalars = entries.filter(([, value]) => !isContainer(value?.left) && !isContainer(value?.right));
      const nested = entries.filter(([, value]) => isContainer(value?.left) || isContainer(value?.right));
      const sideHeads = () => SIDES.map((side) => h('th', { scope: 'col' }, swatch(side), ` ${SIDE_WORD[side]}`));
      const settingHead = (key) => [h('span', { class: 'mono' }, key), SETTING_TEXT[key] ? h('span', { class: 'evaluation__hint' }, SETTING_TEXT[key]) : null];
      parts.push(
        notice(
          'warn',
          h(
            'span',
            null,
            'These two evaluations cannot be compared: they were recorded with different settings. The settings that differ are listed below. ',
            'In practice the scores were produced under different evaluation settings, so their difference would not measure the models. Record both sides with the same settings to compare them.',
          ),
        ),
      );
      if (scalars.length) {
        parts.push(
          h(
            'table',
            { class: 'table evaluation__table' },
            h('caption', { class: 'evaluation__caption' }, 'Settings that differ'),
            h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Setting'), sideHeads())),
            h(
              'tbody',
              null,
              scalars.map(([key, value]) =>
                h('tr', null, h('th', { scope: 'row' }, settingHead(key)), SIDES.map((side) => leafCell(hasSide(value, side), value?.[side]))),
              ),
            ),
          ),
        );
      }
      for (const [key, value] of nested) {
        // A nested setting is compared leaf by leaf, so the few values that differ are not lost in two
        // long blocks; the full values stay available below the table.
        const leftLeaves = flattenLeaves(value?.left);
        const rightLeaves = flattenLeaves(value?.right);
        const paths = [...leftLeaves.keys()];
        for (const path of rightLeaves.keys()) if (!leftLeaves.has(path)) paths.push(path);
        const differing = paths.filter(
          (path) => leftLeaves.has(path) !== rightLeaves.has(path) || JSON.stringify(leftLeaves.get(path)) !== JSON.stringify(rightLeaves.get(path)),
        );
        parts.push(
          h(
            'div',
            { class: 'evaluation__setting' },
            h('h3', { class: 'evaluation__subtitle' }, settingHead(key)),
            h('p', { class: 'evaluation__lead' }, `${fmt.int(differing.length)} of ${fmt.int(paths.length)} values differ.`),
            differing.length
              ? h(
                  'div',
                  { class: 'evaluation__scroll' },
                  h(
                    'table',
                    { class: 'table evaluation__table' },
                    h('caption', { class: 'visually-hidden' }, `Values of ${key} that differ`),
                    h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Path'), sideHeads())),
                    h(
                      'tbody',
                      null,
                      differing.map((path) =>
                        h(
                          'tr',
                          null,
                          h('th', { scope: 'row', class: 'mono evaluation__value' }, path || '(whole value)'),
                          leafCell(leftLeaves.has(path), leftLeaves.get(path)),
                          leafCell(rightLeaves.has(path), rightLeaves.get(path)),
                        ),
                      ),
                    ),
                  ),
                )
              : null,
            h(
              'details',
              { class: 'evaluation__details' },
              h('summary', { class: 'evaluation__summary-toggle' }, 'Full recorded settings'),
              h(
                'div',
                { class: 'evaluation__full' },
                SIDES.map((side) =>
                  h(
                    'div',
                    { class: 'evaluation__full-side' },
                    h('p', { class: 'evaluation__label' }, swatch(side), `${SIDE_WORD[side]}: ${nameOf(side)}`),
                    h('pre', { class: 'evaluation__json mono' }, hasSide(value, side) ? (JSON.stringify(value[side], null, 2) ?? 'Not reported') : 'Not set'),
                  ),
                ),
              ),
            ),
          ),
        );
      }
      return parts;
    }

    if (pooled) {
      // Whole-dataset scores: compared only when both sides scored exactly the same inputs.
      if (res.comparable === true) parts.push(renderPooled(res));
      else {
        const reason = typeof res.reason === 'string' && res.reason ? res.reason : 'The service gave no reason.';
        parts.push(notice('warn', h('span', null, 'The whole-dataset scores of the two sides cannot be compared. ', reason)));
        if (isContainer(res.nonfinite)) {
          const word = (value) => (value === null || value === undefined ? 'no non-finite value reported' : leafText(value));
          parts.push(
            h(
              'dl',
              { class: 'kv evaluation__kv' },
              SIDES.map((side) => [h('dt', null, swatch(side), ` ${SIDE_WORD[side]} score, as recorded`), h('dd', { class: 'mono' }, word(res.nonfinite[side]))]),
            ),
          );
        }
      }
      if (res.counts && typeof res.counts === 'object') parts.push(renderCounts(res.counts, POOLED_COUNT_TEXT, 'Scored inputs by match status', 'Inputs'));
      return parts;
    }

    if (res.comparable === false) {
      parts.push(notice('warn', 'No image matched between the two evaluations, so nothing can be compared image by image. The counts below say why.'));
    }

    if (summary) parts.push(renderSummary(res, summary, places));
    if (res.counts && typeof res.counts === 'object') parts.push(renderCounts(res.counts, COUNT_TEXT, 'Images by match status', 'Images'));

    if (summary) {
      parts.push(sampleTables(res, outputs, places));
      const listed = (res.improvements?.length ?? 0) + (res.regressions?.length ?? 0);
      if (listed) outputs.reveal(); // with every image tied there is no sample to choose
    }
    return parts;
  }

  /** The evaluator's score over the whole dataset on each side, and their difference. */
  function renderPooled(res) {
    const pooled = isContainer(res.pooled) ? res.pooled : {};
    const scorePlaces = placesFor([pooled.left, pooled.right]);
    const deltaPlaces = placesFor([pooled.delta]);
    let verdict = '';
    if (Number.isFinite(pooled.delta) && pooled.delta !== 0 && (res.better === 'higher' || res.better === 'lower')) {
      const rightBetter = res.better === 'higher' ? pooled.delta > 0 : pooled.delta < 0;
      verdict = `, the right ${nounOf('right')} is ${rightBetter ? 'better' : 'worse'}`;
    }
    return h(
      'div',
      { class: 'evaluation__summary' },
      h('h3', { class: 'evaluation__subtitle' }, 'Whole-dataset score'),
      h(
        'dl',
        { class: 'kv evaluation__kv' },
        SIDES.map((side) => [
          h('dt', null, swatch(side), ` ${SIDE_WORD[side]}: ${nameOf(side)}`),
          h('dd', { class: 'num', title: exact(pooled[side]) }, score(pooled[side], scorePlaces)),
        ]),
        h('dt', null, 'Difference (right minus left)'),
        h('dd', { class: 'num', title: exact(pooled.delta) }, `${delta(pooled.delta, deltaPlaces)}${verdict}`),
      ),
      h('p', { class: 'evaluation__sentence' }, POOLED_SENTENCE),
    );
  }

  function renderSummary(res, s, places) {
    const matched = res.counts?.matched;
    const items = [];
    items.push(
      h(
        'dl',
        { class: 'kv evaluation__kv' },
        h('dt', null, 'Mean difference (right minus left)'), h('dd', { class: 'num', title: exact(s.meanDelta) }, delta(s.meanDelta, places.delta)),
        h('dt', { title: 'The most negative difference, right minus left' }, 'Smallest difference'), h('dd', { class: 'num', title: exact(s.minDelta) }, delta(s.minDelta, places.delta)),
        h('dt', { title: 'The most positive difference, right minus left' }, 'Largest difference'), h('dd', { class: 'num', title: exact(s.maxDelta) }, delta(s.maxDelta, places.delta)),
      ),
    );
    if (Number.isFinite(s.meanDelta) && s.meanDelta !== 0) {
      const rightBetter = res.better === 'higher' ? s.meanDelta > 0 : s.meanDelta < 0;
      const winner = rightBetter ? 'right' : 'left';
      const over = Number.isFinite(matched) ? ` over the ${fmt.plural(matched, 'matched image')}` : '';
      items.push(
        h(
          'p',
          { class: 'evaluation__sentence' },
          swatch(winner),
          ` On average${over}, the ${sideRef(winner)} scores better on ${res.metric}: the mean difference, right minus left, is ${signed(s.meanDelta, places.delta)} and ${directionWords(res.better).toLowerCase()}.`,
        ),
      );
    }
    const right = nounOf('right');
    const parts = [
      { key: 'improved', label: 'Improved', help: `the right ${right} scores better`, n: s.improved },
      { key: 'tied', label: 'Tied', help: 'same score', n: s.tied },
      { key: 'regressed', label: 'Regressed', help: `the right ${right} scores worse`, n: s.regressed },
    ];
    if (parts.every((part) => Number.isFinite(part.n))) {
      const total = parts.reduce((sum, part) => sum + part.n, 0);
      const words = parts.map((part) => `${part.label} ${fmt.int(part.n)}`).join(', ');
      items.push(
        h(
          'div',
          { class: 'evaluation__split' },
          total > 0
            ? h(
                'div',
                { class: 'evaluation__bar', role: 'img', 'aria-label': `Matched images: ${words}` },
                parts
                  .filter((part) => part.n > 0)
                  .map((part) =>
                    h(
                      'span',
                      { class: `evaluation__bar-part evaluation__bar-part--${part.key}`, style: { flexGrow: String(part.n) }, title: `${part.label}: ${fmt.int(part.n)}` },
                      fmt.int(part.n),
                    ),
                  ),
              )
            : null,
          h(
            'ul',
            { class: 'evaluation__bar-legend' },
            parts.map((part) =>
              h(
                'li',
                { class: 'evaluation__bar-key' },
                h('span', { class: `evaluation__bar-mark evaluation__bar-mark--${part.key}`, 'aria-hidden': 'true' }),
                h('strong', { class: 'num' }, fmt.int(part.n)),
                ` ${part.label.toLowerCase()} (${part.help})`,
              ),
            ),
          ),
        ),
      );
    }
    return h('div', { class: 'evaluation__summary' }, items);
  }

  function renderCounts(counts, texts, caption, unit) {
    const keys = [...Object.keys(texts).filter((k) => k in counts), ...Object.keys(counts).filter((k) => !(k in texts))];
    return h(
      'table',
      { class: 'table evaluation__table' },
      h('caption', { class: 'evaluation__caption' }, caption),
      h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Status'), h('th', { scope: 'col', class: 'num' }, unit), h('th', { scope: 'col' }, 'What it means'))),
      h(
        'tbody',
        null,
        keys.map((key) =>
          h(
            'tr',
            null,
            h('th', { scope: 'row', class: 'mono' }, key),
            h('td', { class: 'num' }, fmt.int(counts[key]) || 'Not reported'),
            h('td', null, texts[key] ?? 'Reported by the service without a description.'),
          ),
        ),
      ),
    );
  }

  function sampleTables(res, outputs, places) {
    const buttons = [];
    const pick = (sample, item) => {
      for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.sample === sample));
      outputs.show(sample, res, item);
    };
    const right = nounOf('right');
    const table = (title, rows, emptyText) => {
      const list = Array.isArray(rows) ? rows : [];
      return h(
        'div',
        { class: 'evaluation__table-wrap' },
        h('h3', { class: 'evaluation__subtitle' }, title),
        list.length
          ? h(
              'table',
              { class: 'table evaluation__table' },
              h(
                'thead',
                null,
                h(
                  'tr',
                  null,
                  h('th', { scope: 'col' }, 'Sample'),
                  h('th', { scope: 'col', class: 'num' }, swatch('left'), ' Left'),
                  h('th', { scope: 'col', class: 'num' }, swatch('right'), ' Right'),
                  h('th', { scope: 'col', class: 'num' }, 'Difference (right minus left)'),
                ),
              ),
              h(
                'tbody',
                null,
                list.map((item) => {
                  const sample = String(item.sample);
                  const button = h(
                    'button',
                    {
                      class: 'evaluation__sample mono',
                      type: 'button',
                      'aria-pressed': 'false',
                      dataset: { sample },
                      title: `${sample}: show both sides’ saved outputs for this image`,
                      onclick: () => pick(sample, item),
                    },
                    sampleLabel(sample),
                  );
                  buttons.push(button);
                  return h(
                    'tr',
                    null,
                    h('th', { scope: 'row' }, button),
                    h('td', { class: 'num', title: exact(item.left) }, score(item.left, places.score)),
                    h('td', { class: 'num', title: exact(item.right) }, score(item.right, places.score)),
                    h('td', { class: 'num', title: exact(item.delta) }, delta(item.delta, places.delta)),
                  );
                }),
              ),
            )
          : h('p', { class: 'evaluation__lead' }, emptyText),
      );
    };
    return h(
      'div',
      { class: 'evaluation__samples' },
      h('p', { class: 'evaluation__lead' }, 'Choose a sample to see both sides’ saved outputs for it on the comparison stage below.'),
      h(
        'div',
        { class: 'evaluation__tables' },
        table('Largest improvements (up to 10)', res.improvements, `No matched image scores better in the right ${right}.`),
        table('Largest regressions (up to 10)', res.regressions, `No matched image scores worse in the right ${right}.`),
      ),
    );
  }

  // ───────── 3. the two outputs for one image ─────────

  function outputsPart(p) {
    const status = h('div', { class: 'evaluation__status', role: 'status', 'aria-live': 'polite' });
    const aBar = h('div', { class: 'evaluation__a-choice', hidden: true });
    const stageHost = h('div', { class: 'evaluation__stage', hidden: true });
    const el = section(
      'Outputs for one image',
      'The left side’s saved output (A) against the right side’s (B), or the recorded ground truth against the right output. Choose a sample in the tables above.',
      status,
      aBar,
      stageHost,
    );
    el.hidden = true; // shown once there are samples to choose from
    let stage = null;
    let ctl = null;
    let token = 0;
    p.cleanups.push(() => {
      ctl?.abort();
      stage?.destroy();
      stage = null;
    });

    // The saved images of a run (GET /api/run/images) or of an evaluation job (GET /api/analysis/images,
    // the same shape). An evaluation job has one observation, so no iteration is asked for.
    async function find(side, sample, dataset, step, signal) {
      const ref = p[side];
      const images = ref.kind === 'job' ? api.analysisImages : api.images;
      const query = ref.kind === 'job' ? { image: sample, limit: 1 } : { image: sample, iteration: step, limit: 1 };
      let reply = await images(ref.id, { ...query, dataset }, { signal });
      let image = Array.isArray(reply?.images) ? reply.images[0] : null;
      if (!image && dataset !== undefined && dataset !== null) {
        reply = await images(ref.id, query, { signal });
        image = Array.isArray(reply?.images) ? reply.images[0] : null;
      }
      return image ?? null;
    }

    /**
     * What the stage says under an output. A file the comparison itself named is the version recorded
     * with the scores, which the gallery entry (a later export can replace it) does not describe, so
     * only the compared iteration is added to it.
     */
    function outputNote(side, image, step, linked) {
      const bits = ['model output'];
      const iteration = linked ? step : image?.iteration;
      if (p[side].kind === 'run' && Number.isFinite(iteration) && iteration >= 0) bits.push(`iteration ${fmt.int(iteration)}`);
      if (linked) {
        bits.push('the file recorded with the scores');
        return bits.join(', ');
      }
      if (image?.name) bits.push(image.name);
      const dims = fmt.dims(image?.width, image?.height);
      bits.push(dims ? `original file, ${dims}` : 'original file');
      return bits.join(', ');
    }

    /** "left run, iteration 8" / "left evaluation": where a saved image was looked for. */
    function where(side, step) {
      const it = p[side].kind === 'run' && Number.isFinite(step) && step >= 0 ? `, iteration ${fmt.int(step)}` : '';
      return `${side} ${nounOf(side)}${it}`;
    }

    function reset() {
      token += 1;
      ctl?.abort();
      mount(status);
      aBar.hidden = true;
      mount(aBar);
      if (stage) stage.setSources({ a: null, b: null });
      stageHost.hidden = true;
      el.hidden = true;
    }

    function reveal() {
      el.hidden = false;
    }

    /** `item` is the improvement or regression row; it may carry leftImageUrl / rightImageUrl. */
    async function show(sample, res, item = null) {
      reveal();
      ctl?.abort();
      ctl?.release();
      ctl = childController(p.signal);
      const mine = ++token;
      mount(status, loading(`Looking up the saved outputs for ${sample}…`));
      const steps = { left: res.leftStep, right: res.rightStep };
      const linked = {};
      for (const side of SIDES) {
        const url = item?.[`${side}ImageUrl`];
        linked[side] = typeof url === 'string' && url ? url : null;
      }
      // The gallery is still asked: it knows the image's size and the recorded ground truth.
      const settled = await Promise.allSettled(SIDES.map((side) => find(side, sample, res.dataset, steps[side], ctl.signal)));
      if (mine !== token || !live(p)) return;
      const images = {};
      const lines = [];
      settled.forEach((outcome, i) => {
        const side = SIDES[i];
        images[side] = outcome.status === 'fulfilled' ? outcome.value : null;
        if (linked[side]) return; // the comparison named this side's file itself
        if (outcome.status === 'fulfilled') {
          if (!outcome.value) lines.push(`No saved image for this sample in ${nameOf(side)} (${where(side, steps[side])}).`);
        } else if (!isAbort(outcome.reason)) {
          lines.push(`Could not look up the saved image in ${nameOf(side)} (${where(side, steps[side])}): ${outcome.reason?.message ?? 'no answer'}`);
        }
      });

      const source = (side) => {
        const image = images[side];
        const url = linked[side] ?? image?.url;
        if (!url) return null;
        // Both sides are model outputs (the same output key); the label names the side and what it is.
        return { label: `${SIDE_WORD[side]} ${nounOf(side)}: ${shortNameOf(side)}`, role: 'out', load: () => Promise.resolve(url), note: outputNote(side, image, steps[side], Boolean(linked[side])) };
      };
      const left = source('left');
      const right = source('right');
      const gtOwner = images.left?.gtUrl ? 'left' : images.right?.gtUrl ? 'right' : null;
      const gt = gtOwner
        ? {
            label: 'Ground truth',
            role: 'gt',
            load: () => Promise.resolve(images[gtOwner].gtUrl),
            note: `recorded by the ${gtOwner} ${nounOf(gtOwner)}, a preview of at most 1536 px`,
          }
        : null;

      const head = h('p', { class: 'evaluation__lead' }, 'Sample ', h('span', { class: 'mono' }, sample), '.');
      mount(status, head, lines.length ? notice('warn', lines.join(' ')) : null);
      if (!left && !right && !gt) {
        aBar.hidden = true;
        stageHost.hidden = true;
        if (stage) stage.setSources({ a: null, b: null });
        return;
      }

      let aRole = left ? 'left' : 'gt';
      const apply = () => {
        stageHost.hidden = false;
        if (!stage) stage = createCompare(stageHost, { emptyText: 'Choose a sample in the tables above.' });
        stage.setSources({ a: aRole === 'gt' ? gt : left, b: right });
      };
      if (gt && left) {
        const buttons = {};
        const choose = (role) => {
          aRole = role;
          for (const [key, b] of Object.entries(buttons)) b.setAttribute('aria-pressed', String(key === role));
          apply();
        };
        buttons.left = h('button', { class: 'btn btn--sm', type: 'button', 'aria-pressed': 'true', onclick: () => choose('left') }, `Left ${nounOf('left')} output`);
        buttons.gt = h('button', { class: 'btn btn--sm', type: 'button', 'aria-pressed': 'false', onclick: () => choose('gt') }, 'Ground truth');
        mount(aBar, h('span', { class: 'evaluation__a-label' }, 'Show as A'), h('div', { class: 'seg', role: 'group', 'aria-label': 'Image shown as A' }, buttons.left, buttons.gt));
        aBar.hidden = false;
      } else {
        mount(aBar);
        aBar.hidden = true;
      }
      apply();
    }

    return { el, show, reset, reveal };
  }

  // ───────── 4. logged history ─────────

  function historyPart(p) {
    if (SIDES.some((side) => p[side].kind === 'job')) {
      // Only training writes a log, so nothing is asked of the service for an evaluation job.
      return section('Logged history', null, h('p', { class: 'evaluation__lead' }, 'An evaluation job has no training log, so there is no logged value to chart.'));
    }
    const hasArchive = SIDES.some((side) => p[side].kind === 'tensorboard');
    function read(side, params, signal) {
      const ref = p[side];
      return ref.kind === 'tensorboard'
        ? api.tensorboardHistory(ref.id, params, { signal })
        : api.history(ref.id, { scope: hasArchive ? 'comparison' : 'training', ...params }, { signal });
    }
    const progress = { left: '', right: '' };
    const status = h('div', { class: 'evaluation__status', role: 'status', 'aria-live': 'polite' }, loading('Reading the training logs…'));
    const controls = h('div', { class: 'evaluation__controls' });
    const writerControls = h('div', { class: 'evaluation__controls' });
    const selectedWriter = { left: '', right: '' };
    const lastMetric = { left: '', right: '' };
    let generation = 0;
    const indexes = {};
    const curves = {};
    let controlKey = null;
    let drawCurrent;
    const chartHost = h('div', { class: 'evaluation__chart' });
    const notes = h('ul', { class: 'evaluation__notes', 'aria-live': 'polite' });
    const el = section(
      'Logged history',
      hasArchive ? 'Compare scalar values by iteration. Select the same dataset and metric on both sides.' : 'Compare a value recorded in both training logs by iteration.',
      writerControls,
      status,
      controls,
      chartHost,
      notes,
    );
    let chart = null;
    let ctl = null;
    let token = 0;
    p.cleanups.push(() => {
      ctl?.abort();
      chart?.destroy();
      chart = null;
    });

    const showProgress = () => {
      const text = SIDES.map((side) => progress[side]).filter(Boolean).join(' ');
      mount(status, loading(text || 'Reading the training logs…'));
    };

    async function index(side, version) {
      if (indexes[side]) return indexes[side];
      const source = selectedWriter[side];
      let reply;
      let previous;
      do {
        reply = await read(side, { source }, p.signal);
        if (!live(p) || version !== generation) return reply;
        if (!reply?.more) break;
        const cursor = `${reply.indexedBytes}/${reply.sourceBytes}`;
        if (previous === cursor) throw new Error('Indexing stopped before the full history could be read.');
        previous = cursor;
        const done = bytes(reply.indexedBytes);
        const total = bytes(reply.sourceBytes);
        progress[side] = done && total ? `${SIDE_WORD[side]} run: ${done} of ${total} of the training log indexed.` : `${SIDE_WORD[side]} run: indexing the training log.`;
        showProgress();
      } while (live(p) && version === generation);
      progress[side] = '';
      indexes[side] = reply;
      return reply;
    }

    async function reloadHistory(background) {
      const version = ++generation;
      ctl?.abort();
      token++;
      if (!background) showProgress();
      const settled = await Promise.allSettled(SIDES.map((side) => index(side, version)));
      if (!live(p) || version !== generation) return;
      const info = {};
      const lines = [];
      settled.forEach((outcome, i) => {
        const side = SIDES[i];
        if (outcome.status === 'fulfilled') info[side] = outcome.value;
        else if (!isAbort(outcome.reason)) lines.push(`Could not read ${nameOf(side)}: ${outcome.reason?.message ?? 'no answer'}`);
      });
      if (lines.length) throw new Error(lines.join(' '));
      const layout = JSON.stringify(SIDES.map((side) => [
        selectedWriter[side], info[side].sources, info[side].source, info[side].writerCount,
        info[side].partialHistory, info[side].resumeSteps, info[side].series?.map((s) => s.key),
      ]));
      if (layout === controlKey && drawCurrent) {
        return drawCurrent(background);
      }
      controlKey = layout;
      drawCurrent = null;
      mount(writerControls, SIDES.filter((side) => info[side].writerCount > 1).map((side) => {
        const id = uid('history-files');
        const select = h('select', { class: 'select evaluation__select', id,
          onchange: () => { selectedWriter[side] = select.value; refreshSide(side); } },
          h('option', { value: '' }, `Full history (${info[side].writerCount} event files)`),
          info[side].sources.map((path) => h('option', { value: path, title: path }, baseName(path))));
        select.value = selectedWriter[side];
        return h('div', { class: 'evaluation__pick' },
          h('label', { class: 'evaluation__label', for: id }, `${SIDE_WORD[side]} history`), select);
      }));
      for (const side of SIDES) {
        if (info[side].partialHistory) lines.push(`${SIDE_WORD[side]}: some earlier training history could not be linked.`);
      }
      const keysOf = (reply) => (Array.isArray(reply.series) ? reply.series.map((s) => s.key) : []);
      const keys = { left: keysOf(info.left), right: keysOf(info.right) };
      const rightKeys = new Set(keys.right);
      const common = keys.left.filter((key) => rightKeys.has(key));
      if (SIDES.some((side) => !keys[side].length) || (!hasArchive && !common.length)) {
        chart?.destroy();
        chart = null;
        mount(chartHost);
        mount(controls);
        mount(notes);
      }
      if (SIDES.some((side) => !keys[side].length)) {
        mount(status, notice('', SIDES.filter((side) => !keys[side].length)
          .map((side) => `${SIDE_WORD[side]} has no scalar values in the selected history.`).join(' ')));
        return;
      }
      if (!hasArchive && !common.length) {
        mount(status, notice('', 'The two runs have no logged values with matching names.'));
        return;
      }
      mount(status, h('p', { class: 'evaluation__lead' },
        SIDES.map((side) => `${SIDE_WORD[side]} source: ${info[side].source || 'Full training history'}. `), lines.join(' ')));
      const preferred = (hasArchive && common.find((key) => key.toLowerCase().endsWith('/psnr')))
        || (common.includes(DEFAULT_SERIES) && DEFAULT_SERIES) || common[0];
      const choices = {};
      function valueControl(side, options) {
        const id = uid('evaluation-series');
        const select = h('select', { class: 'select evaluation__select', id,
          onchange: () => {
            if (hasArchive && side === 'left') choices.right.value = rightKeys.has(select.value) ? select.value : '';
            draw();
          } }, options.map((key) => h('option', { value: key }, key)));
        select.value = options.includes(lastMetric[side]) ? lastMetric[side] : preferred || options[0];
        choices[side] = select;
        const label = hasArchive ? `${SIDE_WORD[side]} logged value` : 'Logged value';
        return h('div', { class: 'evaluation__pick' }, h('label', { class: 'evaluation__label', for: id }, label), select);
      }
      mount(controls, hasArchive ? SIDES.map((side) => valueControl(side, keys[side])) : valueControl('left', common));
      if (!hasArchive) choices.right = choices.left;
      if (hasArchive) {
        choices.right.insertBefore(h('option', { value: '' }, 'Choose a comparable value'), choices.right.firstChild);
        choices.right.value = keys.right.includes(lastMetric.right) ? lastMetric.right : preferred || '';
      }
      drawCurrent = draw;
      return draw(background);

      async function draw(quiet = false) {
        ctl?.abort();
        ctl?.release();
        ctl = childController(p.signal);
        const mine = ++token;
        const selected = { left: choices.left.value, right: choices.right.value };
        Object.assign(lastMetric, selected);
        if (!selected.right) {
          chart?.destroy();
          chart = null;
          mount(notes, h('li', null, 'Choose the corresponding logged value on the right.'));
          return;
        }
        if (!quiet) mount(notes, h('li', null, loading('Reading both series…')));
        const signal = ctl.signal;
        const results = await Promise.allSettled(SIDES.map(async (side) => {
          const params = { metric: selected[side], budget: HISTORY_BUDGET, ...chart?.iterationRange(), source: selectedWriter[side] };
          const key = JSON.stringify(params);
          if (curves[side]?.key === key) return curves[side].reply;
          const reply = await read(side, params, signal);
          if (!signal.aborted) curves[side] = { key, reply };
          return reply;
        }));
        if (mine !== token || !live(p)) return;
        if (quiet) {
          const failed = results.find((r) => r.status === 'rejected');
          if (failed) throw failed.reason;
        }
        const series = [];
        const items = [];
        results.forEach((outcome, i) => {
          const side = SIDES[i];
          const word = SIDE_WORD[side];
          const key = selected[side];
          if (outcome.status !== 'fulfilled') {
            if (!isAbort(outcome.reason)) items.push(`${word} run: could not read ${key}: ${outcome.reason?.message ?? 'no answer'}`);
            return;
          }
          const reply = outcome.value ?? {};
          const raw = Array.isArray(reply.points) ? reply.points : [];
          const points = [];
          let gaps = 0;
          let real = 0;
          for (const point of raw) {
            const [x, y] = Array.isArray(point) ? point : [];
            if (!Number.isFinite(x)) continue;
            // null (a non-finite logged value) stays null: the chart breaks the line there.
            if (Number.isFinite(y)) real += 1;
            else gaps += 1;
            points.push({ x, y: Number.isFinite(y) ? y : null });
          }
          series.push({ name: `${word}: ${nameOf(side)}${selected.left !== selected.right ? ` · ${key}` : ''}`, points });
          const matched = Number.isFinite(reply.matchedPoints) ? reply.matchedPoints : raw.length;
          let text = reply.downsampled
            ? `${word} run: ${fmt.int(raw.length)} of ${fmt.int(matched)} logged points drawn (downsampled by the service: each stretch of iterations keeps its first, last, lowest and highest point).`
            : `${word} run: all ${fmt.plural(matched, 'logged point')} drawn.`;
          if (gaps) text += ` ${fmt.plural(gaps, 'point has', 'points have')} no finite value; the line breaks there.`;
          if (!real) text += ` No finite value of ${key} to draw.`;
          items.push(text);
        });
        // Both series are always passed, the left one first by name, so the left run takes the
        // first series colour and the right run the second even when one side has no values.
        if (!chart) chart = createChart(chartHost, { title: 'Logged history', onXRangeChange: () => drawCurrent?.() });
        const markers = [...new Set(SIDES.flatMap((side) => info[side].resumeSteps || []))];
        chart.update({ series, xLabel: 'Iteration', markers, yKey: JSON.stringify(selected) });
        mount(notes, items.map((text) => h('li', { class: 'evaluation__note' }, text)));
      }
    }

    function refreshSide(side, background = false) {
      delete indexes[side];
      delete curves[side];
      return refreshHistory(background);
    }

    function refreshHistory(background = false) {
      return reloadHistory(background).catch((error) => {
        if (live(p) && !isAbort(error)) {
          controlKey = null;
          mount(status, notice('err', `Could not show the logged history: ${error.message}`));
          if (background) throw error;
        }
      });
    }
    refreshHistory();
    for (const side of SIDES) {
      const ref = p[side];
      const params = ref.kind === 'tensorboard' ? { kind: 'tensorboard', path: ref.id }
        : { kind: hasArchive ? 'comparison' : 'training', id: ref.id };
      p.cleanups.push(watchHistory(params, () => refreshSide(side, true)));
    }
    return el;
  }

  // ───────── 5. recipe differences ─────────

  function recipePart(p) {
    const status = h('div', { class: 'evaluation__status', role: 'status', 'aria-live': 'polite' }, loading('Reading both recipes…'));
    const result = h('div', { class: 'evaluation__result' });
    const anyJob = SIDES.some((side) => p[side].kind === 'job');
    const el = section(
      'Recipe differences',
      anyJob
        ? `The options the two recipes set differently, as written in each. An evaluation job is compared by ${JOB_RECIPE}, the recipe the service wrote for its evaluator, so options that only matter for training can be missing from it.`
        : 'The options the two saved recipes set differently, as written in each recipe.',
      status,
      result,
    );

    /** A run's saved recipe, or the recipe an evaluation job's evaluator ran with; '' when there is none. */
    async function recipeOf(side) {
      const ref = p[side];
      if (ref.kind === 'run') {
        const detail = (await api.run(ref.id, { signal: p.signal })) ?? {};
        p.detail[side] = detail;
        return typeof detail.yaml === 'string' ? detail.yaml : '';
      }
      const detail = await readJob(ref.id);
      const artifact = detail.artifacts.find((a) => a?.path === JOB_RECIPE && typeof a.url === 'string' && a.url);
      if (!artifact) return '';
      // A plain GET of the file's address, as the job's result lists it.
      let response;
      try {
        response = await fetch(artifact.url, { signal: p.signal });
      } catch (error) {
        if (isAbort(error)) throw error;
        throw new Error('The trainer service did not answer.');
      }
      if (!response.ok) throw new Error(`The service answered ${response.status} for ${JOB_RECIPE}.`);
      return response.text();
    }

    (async () => {
      const details = await Promise.allSettled(SIDES.map(recipeOf));
      if (!live(p)) return;
      const yaml = {};
      const lines = [];
      const missing = [];
      details.forEach((outcome, i) => {
        const side = SIDES[i];
        if (outcome.status === 'fulfilled') {
          if (outcome.value.trim()) yaml[side] = outcome.value;
          else missing.push(side);
        } else if (!isAbort(outcome.reason)) {
          lines.push(`Could not read the recipe of the ${sideRef(side)}: ${outcome.reason?.message ?? 'no answer'}`);
        }
      });
      updateFacts();
      if (lines.length) {
        mount(status, notice('err', lines.join(' ')));
        return;
      }
      if (missing.length) {
        mount(
          status,
          h(
            'p',
            { class: 'evaluation__lead' },
            missing
              .map((side) =>
                p[side].kind === 'job'
                  ? `The recipe of the ${sideRef(side)} is not available to compare.`
                  : `${nameOf(side)} (${side} run) has no saved recipe, so the recipes are not compared.`,
              )
              .join(' '),
          ),
        );
        return;
      }
      const checks = await Promise.allSettled(SIDES.map((side) => api.preflight(yaml[side], { signal: p.signal })));
      if (!live(p)) return;
      const config = {};
      checks.forEach((outcome, i) => {
        const side = SIDES[i];
        if (outcome.status === 'fulfilled' && outcome.value?.config && typeof outcome.value.config === 'object') config[side] = outcome.value.config;
        else if (outcome.status === 'fulfilled') {
          const first = outcome.value?.issues?.[0]?.message;
          lines.push(`The recipe of the ${sideRef(side)} could not be read as options${first ? `: ${first}` : '.'}`);
        } else if (!isAbort(outcome.reason)) {
          lines.push(`Could not read the recipe of the ${sideRef(side)}: ${outcome.reason?.message ?? 'no answer'}`);
        }
      });
      if (lines.length) {
        mount(status, notice('warn', lines.join(' ')));
        return;
      }
      const rows = diffConfigs(config.left, config.right).filter((r) => r.state !== 'same');
      if (!rows.length) {
        mount(status, h('p', { class: 'evaluation__lead' }, 'Both recipes set the same options to the same values.'));
        return;
      }
      mount(status, h('p', { class: 'evaluation__lead' }, `${fmt.plural(rows.length, 'option')} ${rows.length === 1 ? 'differs' : 'differ'}.`));
      const cell = (has, value) => (has ? h('td', { class: 'mono evaluation__value' }, compactValue(value)) : h('td', { class: 'muted' }, 'Not set'));
      mount(
        result,
        h(
          'div',
          { class: 'evaluation__scroll' },
          h(
            'table',
            { class: 'table evaluation__table' },
            h(
              'thead',
              null,
              h(
                'tr',
                null,
                h('th', { scope: 'col' }, 'Option'),
                h('th', { scope: 'col' }, swatch('left'), ` Left: ${nameOf('left')}`),
                h('th', { scope: 'col' }, swatch('right'), ` Right: ${nameOf('right')}`),
              ),
            ),
            h(
              'tbody',
              null,
              rows.map((r) => h('tr', null, h('th', { scope: 'row', class: 'mono evaluation__value' }, r.path), cell(r.hasA, r.a), cell(r.hasB, r.b))),
            ),
          ),
        ),
      );
    })().catch((error) => {
      if (isAbort(error) || !live(p)) return;
      mount(status, notice('err', `Could not compare the recipes: ${error?.message ?? 'no answer'}`));
    });

    return el;
  }

  // ───────── lifecycle ─────────

  const unsubscribe = runs.subscribe(() => {
    if (destroyed) return;
    fillSelects();
    updateFacts();
    loadJobs(); // at most once per JOBS_MAX_AGE: an evaluation job that finished meanwhile joins the pickers
    if (pair) return;
    renderIdle();
  });
  if (runs.get().phase === 'idle') runs.refresh();

  startPair();

  return {
    setRuns(nextLeft, nextRight) {
      if (destroyed) return;
      const l = parseSide(nextLeft) ? String(nextLeft) : null;
      const r = parseSide(nextRight) ? String(nextRight) : null;
      if (l === ids.left && r === ids.right) return;
      setSides(l, r);
      pendingRight = null;
      startPair();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      jobsCtl.abort();
      unsubscribe();
      endPair();
      root.remove();
    },
  };
}
