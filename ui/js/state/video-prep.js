// The training videos of a clip-training recipe (datasets.train of type VideoClipDataset): whether each
// video has its index (POST /api/video-index/status), and the analysis job that writes the missing ones
// (kind index-video-clips). The state lives here, not in the panel that shows it, so the answer and a
// running job outlive a rebuild of the options form and a change of screen, and the next-step line can
// say where they stand.
//
//   videoPrep.get() -> {applies, roots, check, job, outcome}
//     applies   the open recipe trains on video clips
//     roots     datasets.train.dataroot_gt as a list (folders and single video files)
//     check     {phase: 'idle'|'checking'|'done'|'error'|'unsupported', answer, error, signature}
//     job       null, or the last preparation {id, status, progress, result, error, cancelAsked, retry,
//               pollError}; status 'starting' (the start request is out), then the service's own words
//     outcome   the result {total, ready, written, skipped} of the last completed preparation, or null
//   videoPrep.check({force}), prepare({again}), cancel(), restore(), subscribe(fn)
//
// Nothing here waits for the recipe to pass its check: the videos can be prepared while the weights and
// other options are still unset.

import { createStore } from '../store.js';
import { api, newRequestId } from '../api.js';
import { recipe } from './recipe.js';

export const VIDEO_DATASET = 'videoclipdataset';
export const VIDEO_INDEX_KIND = 'index-video-clips';
/** The options of the training dataset an index depends on, besides the videos themselves. */
const DESCALE_KEYS = ['descale_size', 'descale_kernel', 'descale_border', 'descale_linear_light'];
const CHECK_DELAY = 500;
const JOB_POLL_MS = 1000;
const RETRY_MS = 3000;
const RESULT_ATTEMPTS = 3;
const OFFLINE = 'The trainer service did not answer.';
const TERMINAL = new Set(['completed', 'failed', 'cancelled']);

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const messageOf = (error) => (error?.offline ? OFFLINE : error?.message ?? String(error));
/** The same test as the other parts that meet a service without the route (runs/storage.js). */
const isMissingRoute = (error) => error?.status === 404 || /unknown api endpoint/i.test(String(error?.message ?? ''));

/**
 * What the video check of a recipe depends on, from its checked config: the training videos and the
 * four descale options. null when datasets.train is not a VideoClipDataset.
 * @returns {{roots: string[], signature: string} | null}
 */
export function inputs(config) {
  const train = config?.datasets?.train;
  if (!isPlainObject(train) || String(train.type ?? '').toLowerCase() !== VIDEO_DATASET) return null;
  const gt = train.dataroot_gt;
  const roots = (Array.isArray(gt) ? gt : [gt]).filter((v) => typeof v === 'string' && v.trim() !== '');
  return { roots, signature: JSON.stringify([roots, ...DESCALE_KEYS.map((k) => (k in train ? train[k] : null))]) };
}

/** A job that has not ended yet (starting, queued, running, cancelling, or a state this module does not know). */
export function jobActive(job) {
  return Boolean(job) && !TERMINAL.has(job.status) && !(job.status === 'failed');
}

/** Counts of a status answer; `fixable` are the videos a preparation can index (pending includes the invalid ones). */
export function counts(answer) {
  const n = (v) => (Number.isFinite(v) ? v : 0);
  const total = n(answer?.total);
  const ready = n(answer?.ready);
  const pending = Number.isFinite(answer?.pending) ? answer.pending : Math.max(0, total - ready);
  const invalid = n(answer?.invalid);
  return { total, ready, pending, invalid, fixable: Math.max(0, pending - invalid) };
}

const store = createStore({
  applies: false,
  roots: [],
  /** The videos are still the example path of the template the recipe started from. */
  example: false,
  check: { phase: 'idle', answer: null, error: null, signature: null },
  job: null,
  outcome: null,
});

let checkTimer = 0;
let checkCtl = null;
/** Signatures whose last answer was an error: never asked again by themselves. */
const failedSignatures = new Set();
let pollTimer = 0;
let pollCtl = null;
let startToken = null;
/** The body of the last start, sent again as it is by "Try again" after no answer or a busy service. */
let lastBody = null;
let restored = false;

const sameList = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

function setJob(patch) {
  store.set((s) => ({ job: s.job ? { ...s.job, ...patch } : { id: null, status: 'running', progress: null, result: null, error: null, ...patch } }));
}

// ───────── folder check ─────────

/** Ask the service where the indexes stand, for the recipe text as it is now (saved or not). */
async function check({ force = false } = {}) {
  clearTimeout(checkTimer);
  checkTimer = 0;
  const r = recipe.get();
  const inp = r.phase === 'open' ? inputs(r.config) : null;
  if (!inp || !inp.roots.length) return;
  if (!force && store.get().check.phase === 'unsupported') return;
  checkCtl?.abort();
  const ctl = new AbortController();
  checkCtl = ctl;
  const { signature } = inp;
  store.set((s) => ({ check: { ...s.check, phase: 'checking', error: null, signature } }));
  try {
    const answer = await api.videoIndexStatus(r.yaml, { signal: ctl.signal });
    if (ctl !== checkCtl) return;
    checkCtl = null;
    failedSignatures.delete(signature);
    store.set({ check: { phase: 'done', answer: isPlainObject(answer) ? answer : null, error: null, signature } });
  } catch (error) {
    if (error?.name === 'AbortError' || ctl !== checkCtl) return;
    checkCtl = null;
    if (isMissingRoute(error)) {
      store.set({ check: { phase: 'unsupported', answer: null, error: null, signature } });
      return;
    }
    failedSignatures.add(signature);
    store.set({ check: { phase: 'error', answer: null, error: messageOf(error), signature } });
  }
}

/**
 * A check refused while the videos are still the template's example path (a place the person never
 * chose): the panel and the shell then ask for a folder instead of showing the refusal.
 */
export function exampleUnchosen(s = store.get()) {
  return Boolean(s.example) && s.check.phase === 'error';
}

/** The roots of the template the open recipe started from, read while its text was still the template's. */
let templateBase = { yaml: null, roots: null };
/** The signature last seen, so a finished job's outcome is dropped when the videos or options change. */
let seenSignature = null;

/** Called on every change of the recipe store: follow which videos the open recipe names. */
function onRecipe() {
  const r = recipe.get();
  const inp = r.phase === 'open' ? inputs(r.config) : null;
  const applies = Boolean(inp);
  const roots = inp?.roots ?? [];
  const fromTemplate = r.phase === 'open' && /^template /.test(r.baseLabel ?? '');
  if (fromTemplate && r.configCurrent && r.yaml === r.baseYaml && inp) templateBase = { yaml: r.baseYaml, roots: JSON.stringify(roots) };
  const example = Boolean(inp) && fromTemplate && templateBase.yaml === r.baseYaml && templateBase.roots === JSON.stringify(roots);
  const s = store.get();
  if (applies !== s.applies || !sameList(roots, s.roots) || example !== s.example) store.set({ applies, roots, example });
  // Another folder or other descale options: what the last finished job did is about something else.
  if (inp && seenSignature !== null && inp.signature !== seenSignature && !jobActive(store.get().job) && (store.get().outcome || store.get().job)) store.set({ outcome: null, job: null });
  if (inp) seenSignature = inp.signature;
  if (!applies || !roots.length) {
    clearTimeout(checkTimer);
    checkTimer = 0;
    if (checkCtl) {
      checkCtl.abort();
      checkCtl = null;
    }
    if (store.get().check.phase !== 'idle' && store.get().check.phase !== 'unsupported') store.set({ check: { phase: 'idle', answer: null, error: null, signature: null } });
    // The refusal that was on screen is gone with the recipe: opening the same videos again asks again,
    // or the panel would sit on "Not checked" with nothing to go on.
    failedSignatures.clear();
    return;
  }
  if (!restored) restore();
  const now = store.get();
  if (jobActive(now.job) || now.check.phase === 'unsupported') return;
  if (inp.signature === now.check.signature || failedSignatures.has(inp.signature)) return;
  clearTimeout(checkTimer);
  checkTimer = setTimeout(() => {
    checkTimer = 0;
    const later = recipe.get();
    const again = later.phase === 'open' ? inputs(later.config) : null;
    const cur = store.get();
    if (!again || !again.roots.length || jobActive(cur.job) || cur.check.phase === 'unsupported') return;
    if (again.signature === cur.check.signature || failedSignatures.has(again.signature)) return;
    check();
  }, CHECK_DELAY);
}

// ───────── the preparation job ─────────

function stopPolling() {
  clearTimeout(pollTimer);
  pollTimer = 0;
  pollCtl?.abort();
  pollCtl = null;
}

function schedulePoll(delay = JOB_POLL_MS) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, delay);
}

/** A job has ended: what is on disk changed, so the folder check and the recipe check are asked again. */
function finished() {
  stopPolling();
  check({ force: true });
  // The recipe check names how many indexes are ready, which a cancelled or failed run changes too.
  if (recipe.get().phase === 'open') recipe.checkNow();
}

/** A job answer (start, poll, cancel or the list): show it, and keep following it while it runs. */
function take(job) {
  if (!isPlainObject(job)) return;
  const current = store.get().job;
  const id = job.id !== undefined && job.id !== null ? String(job.id) : current?.id ?? null;
  if (current?.id && id && current.id !== id) return; // an answer about a job this store no longer follows
  const same = current && current.id === id;
  // The progress as the service published it; null (the first seconds of a job) stays null.
  const progress = isPlainObject(job.progress) ? job.progress : null;
  const base = { id, progress, pollError: null, retry: false, cancelAsked: same ? Boolean(current.cancelAsked) : false };
  if (job.status === 'completed') {
    store.set({ job: { ...base, status: same && current.status !== 'completed' ? current.status : 'running', result: null, error: null } });
    readResult(id);
    return;
  }
  if (job.status === 'failed') {
    store.set({ job: { ...base, status: 'failed', result: null, error: typeof job.error === 'string' && job.error ? job.error : 'The job failed; the service gave no reason.' } });
    finished();
    return;
  }
  if (job.status === 'cancelled') {
    store.set({ job: { ...base, status: 'cancelled', result: null, error: null } });
    finished();
    return;
  }
  store.set({ job: { ...base, status: typeof job.status === 'string' && job.status ? job.status : 'running', result: null, error: null } });
  schedulePoll();
}

async function readResult(id, attempt = 1) {
  stopPolling();
  const ctl = new AbortController();
  pollCtl = ctl;
  try {
    const answer = await api.analysisResult(id, {}, { signal: ctl.signal });
    if (ctl !== pollCtl) return;
    pollCtl = null;
    const result = isPlainObject(answer?.result) ? answer.result : null;
    setJob({ status: 'completed', result, error: null });
    store.set({ outcome: result });
  } catch (error) {
    if (error?.name === 'AbortError' || ctl !== pollCtl) return;
    pollCtl = null;
    if (attempt < RESULT_ATTEMPTS) {
      pollTimer = setTimeout(() => readResult(id, attempt + 1), JOB_POLL_MS * attempt);
      return;
    }
    // The indexes were written; only the counts are missing.
    setJob({ status: 'completed', result: null, error: null });
    store.set({ outcome: null });
  }
  finished();
}

async function poll() {
  pollTimer = 0;
  const job = store.get().job;
  if (!job?.id || !jobActive(job)) return;
  const ctl = new AbortController();
  pollCtl = ctl;
  try {
    const next = await api.analysisJob(job.id, { signal: ctl.signal });
    if (ctl !== pollCtl) return;
    pollCtl = null;
    take(next);
  } catch (error) {
    if (error?.name === 'AbortError' || ctl !== pollCtl) return;
    pollCtl = null;
    if (error?.status === 404) {
      // The service no longer has this job: nothing is left to follow.
      store.set({ job: null });
      check({ force: true });
      return;
    }
    // The job may still be running on the trainer machine; only the answer is missing.
    setJob({ pollError: `The progress could not be read: ${messageOf(error)}` });
    schedulePoll(RETRY_MS);
  }
}

/**
 * Start the preparation for the recipe text as it is now. One request id per press; `again` sends the
 * last body as it was (after no answer or a busy service), so a retry never starts a second job.
 * @returns {Promise<void>}
 */
async function prepare({ again = false } = {}) {
  const r = recipe.get();
  if (r.phase !== 'open' || !inputs(r.config)) return;
  if (jobActive(store.get().job)) return;
  const body = again && lastBody && lastBody.yaml === r.yaml ? lastBody : { kind: VIDEO_INDEX_KIND, yaml: r.yaml, requestId: newRequestId() };
  lastBody = body;
  stopPolling();
  const token = {};
  startToken = token;
  store.set({ job: { id: null, status: 'starting', progress: null, result: null, error: null, cancelAsked: false, retry: false, pollError: null } });
  try {
    const job = await api.analysisStart(body);
    if (token !== startToken) return;
    startToken = null;
    if (!isPlainObject(job) || job.id === undefined || job.id === null) throw new Error('The service answered without a job.');
    store.set((s) => ({ job: { ...s.job, id: String(job.id) } }));
    take(job);
  } catch (error) {
    if (token !== startToken) return;
    startToken = null;
    // No answer, or the service is busy with another job: the same request can simply be sent again.
    const retry = Boolean(error?.offline || error?.status === 409);
    const message = error?.offline ? OFFLINE : `The preparation could not be started: ${messageOf(error)}`;
    store.set({ job: { id: null, status: 'failed', progress: null, result: null, error: message, cancelAsked: false, retry, pollError: null } });
    // A refusal (an invalid video, a folder that changed) is best explained by a fresh check.
    if (!error?.offline) check({ force: true });
  }
}

/** Ask the service to cancel the job being followed. Rejects when the request fails (the job runs on). */
async function cancel() {
  const job = store.get().job;
  if (!job?.id || !jobActive(job) || job.cancelAsked) return;
  setJob({ cancelAsked: true });
  try {
    const next = await api.analysisCancel(job.id);
    if (store.get().job?.id !== job.id) return;
    if (isPlainObject(next) && next.status) take(next);
  } catch (error) {
    if (store.get().job?.id === job.id) setJob({ cancelAsked: false });
    throw new Error(messageOf(error));
  }
}

/** True when a job's roots name the recipe's videos: the same paths, or absolute ones ending in a relative one. */
function sameRoots(jobRoots, recipeRoots) {
  if (!Array.isArray(jobRoots) || jobRoots.length !== recipeRoots.length) return false;
  const clean = (p) => String(p).replace(/\\/g, '/').replace(/\/+$/, '');
  return recipeRoots.every((root) => {
    const r = clean(root).replace(/^\.\//, '');
    return jobRoots.some((a) => {
      const abs = clean(a);
      return abs === r || (r && !r.startsWith('/') && abs.endsWith(`/${r}`));
    });
  });
}

/**
 * Once per page load, when a clip-training recipe is first open: follow a preparation of its videos that
 * is still running (started in another window, on the phone, or before a reload).
 */
async function restore() {
  if (restored) return;
  const r = recipe.get();
  const inp = r.phase === 'open' ? inputs(r.config) : null;
  if (!inp || !inp.roots.length) return;
  restored = true;
  if (store.get().job) return;
  let list;
  try {
    list = await api.analysisJobs();
  } catch {
    return;
  }
  if (store.get().job) return; // a start from this page came first
  const found = (Array.isArray(list) ? list : [])
    .filter((job) => job?.kind === VIDEO_INDEX_KIND && job.id !== undefined && jobActive(job) && sameRoots(job.params?.roots, inp.roots))
    .sort((a, b) => String(b.createdAt ?? '').localeCompare(String(a.createdAt ?? '')))[0];
  if (!found) return;
  clearTimeout(checkTimer);
  checkTimer = 0;
  store.set({ job: { id: String(found.id), status: found.status, progress: isPlainObject(found.progress) ? found.progress : null, result: null, error: null, cancelAsked: false, retry: false, pollError: null } });
  take(found);
}

export const videoPrep = {
  get: store.get,
  subscribe: store.subscribe,
  check,
  prepare,
  cancel,
  restore,
  /** A preparation has been asked for and has not ended. */
  active: () => jobActive(store.get().job),
};

recipe.subscribe(onRecipe);
onRecipe();
