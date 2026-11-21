// Connection to the trainer service: is it there, and what does it say it is.

import { api } from '../api.js';
import { createStore } from '../store.js';

const store = createStore({
  /** @type {'checking'|'online'|'offline'} */
  state: 'checking',
  /** {name, version, trainerRoot} as reported by GET /api/status */
  info: null,
  error: null,
});

let timer = 0;

async function check() {
  clearTimeout(timer);
  try {
    const info = await api.status();
    store.set({ state: 'online', info, error: null });
  } catch (error) {
    store.set({ state: 'offline', error: error.message });
  }
  // Look again sooner while the service is missing.
  timer = setTimeout(check, store.get().state === 'online' ? 30000 : 5000);
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) check();
});

export const conn = {
  get: store.get,
  subscribe: store.subscribe,
  check,
};
