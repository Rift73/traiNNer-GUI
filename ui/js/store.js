// A minimal observable store plus guarded localStorage access.

/**
 * @template T
 * @param {T} initial
 * @returns {{get(): T, set(patch: Partial<T> | ((s: T) => Partial<T>)): void, subscribe(fn: (s: T, prev: T) => void): () => void}}
 */
export function createStore(initial) {
  let state = initial;
  const listeners = new Set();
  return {
    get: () => state,
    set(patch) {
      const next = typeof patch === 'function' ? patch(state) : patch;
      if (!next) return;
      const prev = state;
      state = { ...state, ...next };
      for (const fn of [...listeners]) fn(state, prev);
    },
    /** Returns an unsubscribe function. The listener is not called immediately. */
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

// The phone edition (mobile.html marks the document) keeps its own preferences, so looking at it on the
// machine that also runs the desktop page never changes that page's layout, theme or open recipe.
const PREFIX = typeof document !== 'undefined' && document.documentElement?.dataset.edition === 'mobile' ? 'trainner-gui:m:' : 'trainner-gui:';

/** localStorage that never throws (private mode, quota, disabled storage). */
export const persist = {
  get(key, fallback = null) {
    try {
      const raw = localStorage.getItem(PREFIX + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      if (value === null || value === undefined) localStorage.removeItem(PREFIX + key);
      else localStorage.setItem(PREFIX + key, JSON.stringify(value));
      return true;
    } catch {
      return false;
    }
  },
};

/** Tiny event channel for one-off requests between modules ("scroll to this option"). */
export function createChannel() {
  const listeners = new Set();
  return {
    emit(payload) {
      for (const fn of [...listeners]) fn(payload);
    },
    on(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
