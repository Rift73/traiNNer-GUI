// The training queue, kept fresh by polling GET /api/queue.
// An entry is a saved copy of a recipe. It starts "held": nothing runs until the person releases it.
// Released entries wait in a line and start one at a time, from its front, when the trainer is free. A
// release joins the end of the line; the person can move a waiting entry to another place in it.

import { api, newRequestId } from '../api.js';
import { createStore } from '../store.js';

const store = createStore({
  /** true = stop the active queue run at a checkpoint and suspend dispatch */
  paused: false,
  /** id of the entry being started or trained, or null */
  activeId: null,
  /**
   * [{id, name, status, groupId, recipeDigest, createdAt, updatedAt, recipePath?, sourceRunId?, order?,
   *   blockedReason?, runId?, runStatus?, error?, issues?, sourceAttemptId?, resumeStep?, additionalIterations?}]
   * in creation order.
   * A continuation (resumeStep = iteration of the checkpoint it continues stopped run sourceRunId from)
   * is saved released and starts another attempt of the same run; one that extends a completed run also carries
   * additionalIterations (how many iterations it trains beyond resumeStep). A replay carries sourceRunId
   * without resumeStep.
   */
  entries: [],
  /** 'idle' | 'loading' | 'ready' | 'error' */
  phase: 'idle',
  error: null,
});

const STATUS = {
  held: { kind: '', label: 'On hold', open: true, waiting: true },
  pending: { kind: 'run', label: 'Released', open: true, waiting: true },
  running: { kind: 'run', label: 'Training', open: true, active: true },
  stopping: { kind: 'run', label: 'Stopping', open: true, active: true },
  completed: { kind: 'ok', label: 'Completed' },
  stopped: { kind: '', label: 'Stopped' },
  failed: { kind: 'err', label: 'Failed' },
  cancelled: { kind: '', label: 'Cancelled' },
};

/**
 * Classify a queue entry state for presentation.
 * @returns {{kind: 'run'|'ok'|'err'|'', label: string, open: boolean, waiting: boolean, active: boolean}}
 *   open = not finished; waiting = held or released, not started; active = a trainer is running for it.
 */
export function entryStatus(status) {
  const word = String(status ?? '').trim();
  const known = STATUS[word.toLowerCase()];
  if (known) return { open: false, waiting: false, active: false, ...known };
  return { kind: '', label: word ? word.charAt(0).toUpperCase() + word.slice(1) : 'Unknown', open: false, waiting: false, active: false };
}

/** A continuation of a stopped run (it carries resumeStep); a replay carries sourceRunId only. */
export const isContinuation = (entry) => entry?.resumeStep !== undefined && entry?.resumeStep !== null;

/** "1st", "2nd", "11th", … */
export function ordinal(n) {
  const tens = n % 100;
  const suffix = tens >= 11 && tens <= 13 ? 'th' : { 1: 'st', 2: 'nd', 3: 'rd' }[n % 10] || 'th';
  return `${n}${suffix}`;
}

/** "Next", "2nd in line", … for released entries among `open` (queue.open() order); '' otherwise. */
export function placeWords(entry, open = queue.open()) {
  if (entry?.status !== 'pending') return '';
  const index = open.filter((item) => item.status === 'pending').findIndex((item) => item.id === entry.id);
  if (index < 0) return '';
  return index === 0 ? 'Next' : `${ordinal(index + 1)} in line`;
}

/** Status and place of an entry in words, lowercased: "released, next", "training". */
export function statusPlaceWords(entry, open = queue.open()) {
  const place = placeWords(entry, open);
  return [entryStatus(entry?.status).label.toLowerCase(), place.toLowerCase()].filter(Boolean).join(', ');
}

/** What each action does and from which states the backend accepts it. */
export const ENTRY_ACTIONS = {
  release: { label: 'Release', from: ['held'], done: 'Released' },
  hold: { label: 'Put on hold', from: ['pending'], done: 'Put on hold' },
  cancel: { label: 'Cancel entry', from: ['held', 'pending'], done: 'Cancelled' },
  forceStop: { label: 'Force stop', from: ['stopping'], done: 'Force stop requested for' },
  stop: { label: 'Stop training', from: ['running'], done: 'Stop requested for' },
};

let timer = 0;
let watchers = 0;
let inFlight = null;

function nextDelay() {
  const s = store.get();
  if (s.phase === 'error') return 8000;
  const moving = s.entries.some((entry) => entry.status === 'pending' || entryStatus(entry.status).active);
  if (watchers > 0 || moving) return 3000;
  return 20000;
}

function schedule() {
  clearTimeout(timer);
  if (document.hidden) return;
  timer = setTimeout(refresh, nextDelay());
}

function accept(snapshot) {
  store.set({
    paused: Boolean(snapshot?.paused),
    activeId: snapshot?.activeId ?? null,
    entries: Array.isArray(snapshot?.entries) ? snapshot.entries : [],
    phase: 'ready',
    error: null,
  });
}

// Raised when an answer replaces the whole list (a reorder) or a fresh reading is wanted: a poll that was
// already out then carries an older list and is dropped, so it cannot put a moved entry back.
let epoch = 0;

async function refresh() {
  if (inFlight) return inFlight;
  if (store.get().phase === 'idle') store.set({ phase: 'loading' });
  inFlight = (async () => {
    const started = epoch;
    try {
      const snapshot = await api.queue();
      if (started === epoch) accept(snapshot);
    } catch (error) {
      store.set({ phase: 'error', error: error.message });
    } finally {
      inFlight = null;
      schedule();
    }
  })();
  return inFlight;
}

/** Read the queue again without settling for a poll that is already out (its list may be the older one). */
async function reload() {
  epoch += 1;
  if (inFlight) await inFlight;
  return refresh();
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearTimeout(timer);
  else refresh();
});

function upsert(entry) {
  if (!entry || entry.id === undefined) return;
  store.set((s) => {
    const index = s.entries.findIndex((item) => item.id === entry.id);
    const entries = index >= 0 ? s.entries.map((item, i) => (i === index ? { ...item, ...entry } : item)) : [...s.entries, entry];
    return { entries };
  });
}

async function act(action, id) {
  const entry = await api.queueAction(action, id);
  upsert(entry);
  refresh();
  return entry;
}

export const queue = {
  get: store.get,
  subscribe: store.subscribe,
  refresh,
  /** Poll quickly while something is looking at the queue. Returns a release function. */
  watch() {
    watchers += 1;
    refresh();
    return () => {
      watchers = Math.max(0, watchers - 1);
    };
  },

  /** Entries that have not finished, in the order they will be considered: training first, then released by order, then held. */
  open() {
    const rank = (entry) => (entryStatus(entry.status).active ? 0 : entry.status === 'pending' ? 1 : 2);
    return store
      .get()
      .entries.filter((entry) => entryStatus(entry.status).open)
      .sort((a, b) => rank(a) - rank(b) || (a.order ?? Infinity) - (b.order ?? Infinity) || String(a.createdAt).localeCompare(String(b.createdAt)));
  },
  /** Finished entries, newest first. */
  finished() {
    return store
      .get()
      .entries.filter((entry) => !entryStatus(entry.status).open)
      .sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
  },
  /** The line: released entries that wait, in the order they will start. */
  line: () => queue.open().filter((entry) => entry.status === 'pending'),
  held: () => store.get().entries.filter((entry) => entry.status === 'held'),
  pending: () => store.get().entries.filter((entry) => entry.status === 'pending'),
  byId: (id) => store.get().entries.find((entry) => entry.id === id) ?? null,
  /** The most recent launched entry for this run. */
  forRun: (runId) => store.get().entries.filter((entry) => entry.runId === runId).sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))[0] ?? null,

  /**
   * Save a copy of this recipe text in the queue, on hold.
   * @param {string} yaml
   * @param {{recipePath?: string|null, requestId?: string}} [options]
   */
  async submit(yaml, { recipePath = null, requestId = newRequestId() } = {}) {
    const entry = await api.queueSubmit({ yaml, requestId, recipePath: recipePath || undefined });
    upsert(entry);
    refresh();
    return entry;
  },
  /** Save a held entry that repeats run `runId` with its recorded seed. */
  async replay(runId, { requestId = newRequestId() } = {}) {
    const entry = await api.queueReplay(runId, requestId);
    upsert(entry);
    refresh();
    return entry;
  },
  /**
   * Queue a released entry that continues stopped run `runId` from its newest checkpoint (or get the open
   * one). With `additionalIterations` (a positive whole number): extend completed run `runId` from its final
   * training state by that many iterations.
   */
  async resume(runId, { requestId = newRequestId(), additionalIterations = null } = {}) {
    const extra = additionalIterations === null || additionalIterations === undefined ? undefined : { additionalIterations };
    const entry = await api.queueResume(runId, requestId, extra);
    upsert(entry);
    refresh();
    return entry;
  },
  /** The continuation of run `runId` that has not finished, or null. */
  continuationOf: (runId) =>
    store.get().entries.find((entry) => entryStatus(entry.status).open && (
      (entry.pauseAttempt && entry.runId === runId) || (isContinuation(entry) && entry.sourceRunId === runId)
    )) ?? null,
  /** Every continuation of run `runId`, newest first. */
  continuationsOf: (runId) =>
    store
      .get()
      .entries.filter((entry) => isContinuation(entry) && entry.sourceRunId === runId)
      .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt))),
  /**
   * Put the line in another order. `expected`: the ids of the line as the person saw it; `order`: the same
   * ids as wanted. The answer is the whole queue. A refusal (the line changed meanwhile) is thrown after the
   * queue has been read again, so the store already shows the line as it is.
   */
  async reorder(expected, order) {
    epoch += 1;
    let snapshot;
    try {
      snapshot = await api.queueReorder(expected, order);
    } catch (error) {
      await reload();
      throw error;
    }
    epoch += 1;
    accept(snapshot);
    schedule();
    return snapshot;
  },
  /** Allow a held entry to start when the trainer is free. */
  release: (id) => act('release', id),
  /** Take a released entry back out of the line. */
  hold: (id) => act('hold', id),
  /** Withdraw an entry that has not started. */
  cancel: (id) => act('cancel', id),
  /** Stop the training an entry started. */
  stop: (id) => act('stop', id),
  /** Pause or continue starting released entries. */
  async setPaused(paused) {
    accept(await api.queuePause(paused));
    schedule();
  },
};
