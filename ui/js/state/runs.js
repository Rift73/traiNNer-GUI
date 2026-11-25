// The list of training runs, kept fresh by polling GET /api/runs.
// Detail for one run (metrics, log, artifacts) is fetched by the Runs view itself.

import { api, newRequestId } from '../api.js';
import { createStore, persist } from '../store.js';

const store = createStore({
  /**
   * [{id, name, source: 'managed'|'cli', status, iteration?, total?, startedAt?, updatedAt?, task?,
   *   queueEntryId?, error?, returncode?, iterationsPerSecond?, eta?, estimatedFinishAt?, vram?, telemetryIteration?}]
   *   eta: the time left as the trainer wrote it ("3:22:06"); estimatedFinishAt: when that estimate ends, a UTC
   *   date-time the service works out from the time of the log line; null or absent when it cannot (an older
   *   service, an undated log line).
   */
  list: [],
  /** 'idle' | 'loading' | 'ready' | 'error' */
  phase: 'idle',
  error: null,
  selectedId: persist.get('run', null),
});

// The service reports: starting, running, stopping, completed, stopped, failed, and "imported" for
// experiments found in the trainer's experiments folder that it did not launch (read-only).
const ACTIVE = new Set(['starting', 'running', 'stopping']);
const DONE = new Set(['completed']);
const FAILED = new Set(['failed']);

/**
 * Classify a backend status word for presentation. The label is the backend's own word,
 * capitalised; unknown words are shown as they are.
 * @returns {{kind: 'run'|'ok'|'err'|'', active: boolean, label: string, imported: boolean, resumable: boolean}}
 *   imported = found on disk, not launched here: it can be read, not stopped or resumed.
 *   resumable = the backend accepts "resume" from this state (stopped or failed).
 */
export function runStatus(status) {
  const word = String(status ?? '').trim();
  const key = word.toLowerCase();
  const label = word ? word.charAt(0).toUpperCase() + word.slice(1) : 'Unknown';
  const base = { kind: '', active: false, label, imported: key === 'imported', resumable: key === 'stopped' || key === 'failed' };
  if (ACTIVE.has(key)) return { ...base, kind: 'run', active: true };
  if (DONE.has(key)) return { ...base, kind: 'ok' };
  if (FAILED.has(key)) return { ...base, kind: 'err' };
  return base;
}

/**
 * true for an experiment the service found on disk but did not launch: in the trainer's own
 * experiments folder (source "cli") or in a registered extra folder (source "archive"). Read-only.
 */
export const isImported = (run) => run?.source === 'cli' || run?.source === 'archive' || run?.status === 'imported';

/** Where a run came from, in words. */
export function runSourceLabel(run) {
  if (run?.source === 'archive') return 'From an added experiments folder';
  if (run?.source === 'cli' || run?.status === 'imported') return "Found in the trainer's experiments folder";
  return 'Started here';
}

let timer = 0;
let watchers = 0;
let inFlight = null;

function nextDelay() {
  const s = store.get();
  if (s.phase === 'error') return 8000;
  if (watchers > 0 || s.list.some((run) => runStatus(run.status).active)) return 3000;
  return 20000;
}

function schedule() {
  clearTimeout(timer);
  if (document.hidden) return;
  timer = setTimeout(refresh, nextDelay());
}

async function refresh() {
  if (inFlight) return inFlight;
  if (store.get().phase === 'idle') store.set({ phase: 'loading' });
  inFlight = (async () => {
    try {
      const list = await api.runs();
      store.set({ list: Array.isArray(list) ? list : [], phase: 'ready', error: null });
    } catch (error) {
      store.set({ phase: 'error', error: error.message });
    } finally {
      inFlight = null;
      schedule();
    }
  })();
  return inFlight;
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearTimeout(timer);
  else refresh();
});

function upsert(run) {
  if (!run || run.id === undefined) return;
  store.set((s) => {
    const index = s.list.findIndex((item) => item.id === run.id);
    const summary = { ...(index >= 0 ? s.list[index] : {}), ...pickSummary(run) };
    const list = index >= 0 ? s.list.map((item, i) => (i === index ? summary : item)) : [summary, ...s.list];
    return { list };
  });
}

function pickSummary(run) {
  const out = {};
  const keys = [
    'id', 'name', 'source', 'status', 'iteration', 'total', 'startedAt', 'updatedAt', 'task', 'queueEntryId',
    'error', 'returncode', 'iterationsPerSecond', 'eta', 'estimatedFinishAt', 'vram', 'telemetryIteration', 'attemptId', 'sourceRootId', 'sourceRoot',
  ];
  for (const key of keys) {
    if (run[key] !== undefined) out[key] = run[key];
  }
  // A finish time belongs to the estimate it was dated from: an estimate that comes without one clears the
  // earlier finish time of the entry it is merged into.
  if (out.eta !== undefined && out.estimatedFinishAt === undefined) out.estimatedFinishAt = null;
  return out;
}

function select(id) {
  store.set({ selectedId: id ?? null });
  persist.set('run', id ?? null);
}

export const runs = {
  get: store.get,
  subscribe: store.subscribe,
  refresh,
  select,
  /** Merge fresher fields for one run into the list (the Runs view calls this after GET /api/run). */
  upsert,
  /** Poll quickly while something is looking at the list. Returns a release function. */
  watch() {
    watchers += 1;
    refresh();
    return () => {
      watchers = Math.max(0, watchers - 1);
    };
  },
  active() {
    return store.get().list.filter((run) => runStatus(run.status).active);
  },
  byId: (id) => store.get().list.find((run) => run.id === id) ?? null,
  /**
   * Start training now with this recipe text. Resolves to the new run. Rejects with ApiError:
   * status 422 + issues when the recipe does not pass, 409 when the queue holds the trainer.
   * @param {string} yaml
   * @param {string} [requestId] one per confirmed start; repeating it does not start a second run
   */
  async start(yaml, requestId = newRequestId()) {
    const run = await api.startRun(yaml, requestId);
    upsert(run);
    if (run && run.id !== undefined) select(run.id);
    refresh();
    return run;
  },
  async forceStop(id, attemptId) {
    const run = await api.forceStopRun(id, attemptId);
    upsert(run);
    return run;
  },
  async stop(id) {
    const run = await api.stopRun(id);
    upsert(run);
    return run;
  },
  async resume(id, params) {
    const run = await api.resumeRun(id, params);
    upsert(run);
    return run;
  },
};
