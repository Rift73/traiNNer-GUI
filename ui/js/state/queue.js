// The training queue, kept fresh by polling GET /api/queue.
// An entry is a saved copy of a recipe. It starts "held": nothing runs until the person releases it.
// Released entries start one at a time, in release order, when the trainer is free.

import { api, newRequestId } from '../api.js';
import { createStore } from '../store.js';

const store = createStore({
  /** true = no further entry is started (one that is training continues) */
  paused: false,
  /** id of the entry being started or trained, or null */
  activeId: null,
  /**
   * [{id, name, status, groupId, recipeDigest, createdAt, updatedAt, recipePath?, sourceRunId?, order?,
   *   blockedReason?, runId?, runStatus?, error?, issues?}] in creation order
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

/** What each action does and from which states the backend accepts it. */
export const ENTRY_ACTIONS = {
  release: { label: 'Release', from: ['held'], done: 'Released' },
  hold: { label: 'Put on hold', from: ['pending'], done: 'Put on hold' },
  cancel: { label: 'Cancel entry', from: ['held', 'pending'], done: 'Cancelled' },
  stop: { label: 'Stop training', from: ['running', 'stopping'], done: 'Stop requested for' },
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

async function refresh() {
  if (inFlight) return inFlight;
  if (store.get().phase === 'idle') store.set({ phase: 'loading' });
  inFlight = (async () => {
    try {
      accept(await api.queue());
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
  held: () => store.get().entries.filter((entry) => entry.status === 'held'),
  pending: () => store.get().entries.filter((entry) => entry.status === 'pending'),
  byId: (id) => store.get().entries.find((entry) => entry.id === id) ?? null,
  /** The queue entry that started run `runId`, if any. */
  forRun: (runId) => store.get().entries.find((entry) => entry.runId === runId) ?? null,

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
