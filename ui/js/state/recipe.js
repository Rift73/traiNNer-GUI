// The open recipe. YAML text is the single source of truth:
//   - the raw editor calls setYaml()
//   - the form calls applyEdits(), which asks the backend to rewrite the YAML (POST /api/render)
//   - every change is checked by the backend (POST /api/preflight), which returns the parsed
//     `config`, the `summary` and the `issues` that the rest of the interface reads.
// The browser never parses or writes YAML itself.

import { api, ApiError } from '../api.js';
import { createStore, createChannel, persist } from '../store.js';
import { debounce } from '../dom.js';

const EMPTY = {
  /** 'empty' = nothing open, 'open' = a recipe is loaded */
  phase: 'empty',
  /** where it was loaded from / saved to; null for pasted or unsaved text */
  path: null,
  /** document name (file name without extension) */
  name: '',
  yaml: '',
  /** the text this recipe started from, for "what did I change" */
  baseYaml: '',
  /** human label of the starting point, e.g. "template Compact_OTF_gan" */
  baseLabel: '',
  /** text as last loaded or saved; null when it was never saved */
  savedYaml: null,
  /** revision of the saved file this text was loaded from or saved to (only for recipes/<name>.yml) */
  revision: null,

  /** 'idle' | 'checking' | 'done' | 'failed' (the check itself could not run) */
  check: 'idle',
  /** true once any check of this document has answered; ok/issues/summary then hold the latest answer */
  hasResult: false,
  /** backend verdict for the current text; null until checked */
  ok: null,
  issues: [],
  /** {name, task, architecture, scale} */
  summary: null,
  /** last parsed config the backend returned */
  config: null,
  /** false when `config` belongs to older text (for example the YAML no longer parses) */
  configCurrent: false,
  checkError: null,

  /** number of form edits waiting for the backend */
  pending: 0,
  saving: false,
};

const store = createStore({ ...EMPTY });
const focusChannel = createChannel();

let checkAbort = null;
let editQueue = Promise.resolve();

const saveDraft = debounce(() => {
  const s = store.get();
  if (s.phase === 'empty') {
    persist.set('recipe', null);
    return;
  }
  persist.set('recipe', {
    path: s.path,
    name: s.name,
    yaml: s.yaml,
    baseYaml: s.baseYaml,
    baseLabel: s.baseLabel,
    savedYaml: s.savedYaml,
    revision: s.revision,
  });
}, 400);

async function runCheck() {
  const yaml = store.get().yaml;
  if (checkAbort) checkAbort.abort();
  const control = new AbortController();
  checkAbort = control;
  store.set({ check: 'checking' });
  try {
    const result = await api.preflight(yaml, { signal: control.signal });
    if (control.signal.aborted || store.get().yaml !== yaml) return;
    const hasConfig = result && result.config && typeof result.config === 'object';
    store.set((s) => ({
      check: 'done',
      hasResult: true,
      ok: Boolean(result.ok),
      issues: Array.isArray(result.issues) ? result.issues : [],
      summary: result.summary ?? null,
      config: hasConfig ? result.config : s.config,
      configCurrent: Boolean(hasConfig),
      checkError: null,
    }));
  } catch (error) {
    if (error?.name === 'AbortError') return;
    store.set({ check: 'failed', ok: null, checkError: error.message, configCurrent: false });
  }
}

const scheduleCheck = debounce(runCheck, 450);

function setYaml(yaml, { immediate = false } = {}) {
  if (store.get().phase === 'empty') return;
  if (yaml === store.get().yaml) return;
  store.set({ yaml, configCurrent: false });
  saveDraft();
  if (immediate) scheduleCheck.flush();
  else scheduleCheck();
}

// Older editor versions emitted entire object lists on one line. Reformat the
// open draft once, without racing raw edits or changing the saved file.
async function expandCompactDraft(yaml) {
  if (!/^\s*(?:(?:-\s+)?[\w.-]+:\s*)?(?:\[\s*\{|\{[^\n}]+:|\[[^\n]{80,}\])/m.test(yaml)) return;
  try {
    const result = await api.render(yaml, []);
    if (store.get().yaml === yaml && typeof result.yaml === 'string') setYaml(result.yaml);
  } catch (error) {
    // Invalid pasted drafts remain editable. Preflight displays their parser errors.
    if (!(error instanceof ApiError)) throw error;
  }
}

/**
 * Open a recipe document.
 * @param {{path?: string|null, name?: string, yaml: string}} doc
 * @param {{label?: string, saved?: boolean}} [options] label describes the starting point;
 *   saved=false marks text that does not exist on disk yet (pasted, or a template copy).
 */
function load(doc, { label = '', saved = true } = {}) {
  store.set({
    ...EMPTY,
    phase: 'open',
    path: saved ? (doc.path ?? null) : null,
    name: doc.name ?? '',
    yaml: doc.yaml,
    baseYaml: doc.yaml,
    baseLabel: label || doc.name || '',
    savedYaml: saved ? doc.yaml : null,
    revision: saved && isSavedPath(doc.path) && typeof doc.revision === 'string' ? doc.revision : null,
  });
  saveDraft();
  scheduleCheck.flush();
  void expandCompactDraft(doc.yaml);
}

/** Saved recipes live under recipes/; templates (options/…) are read-only starting points. */
const isSavedPath = (path) => typeof path === 'string' && path.startsWith('recipes/');
const savedPathFor = (name) => `recipes/${name}.yml`;

async function openPath(path, options) {
  const doc = await api.recipe(path);
  load(doc, options);
  return doc;
}

/**
 * Change options through the backend so comments and layout survive.
 * Each edit sets a value (`{path, value}`) or deletes a key or list item (`{path, remove: true}`).
 * A list cannot be extended by index: to add an item, set the whole list (`{path: listPath, value: [...]}`).
 * A new key is written as the first key of its mapping; replaced collections use indented YAML. Comments
 * outside the replaced value remain in place.
 * @param {Array<{path: Array<string|number>, value: any} | {path: Array<string|number>, remove: true}>} edits at most 128
 * @returns {Promise<void>} rejects with ApiError when the backend refuses the edit
 */
function applyEdits(edits) {
  store.set((s) => ({ pending: s.pending + 1 }));
  const job = editQueue.then(async () => {
    try {
      for (let attempt = 0; attempt < 3; attempt += 1) {
        const before = store.get().yaml;
        const result = await api.render(before, edits);
        if (!result || typeof result.yaml !== 'string') throw new ApiError('The backend returned no YAML for this edit.');
        // If the text changed while the request was out, apply the edit again on the newer text.
        if (store.get().yaml !== before) continue;
        setYaml(result.yaml, { immediate: true });
        return;
      }
      throw new ApiError('The recipe kept changing while this edit was applied. Try again.');
    } finally {
      store.set((s) => ({ pending: Math.max(0, s.pending - 1) }));
    }
  });
  editQueue = job.catch(() => {});
  return job;
}

/**
 * Save under `name` (POST /api/recipes). Resolves to the saved document.
 * When `name` is the saved recipe this text came from, its revision is sent, and the backend refuses
 * with ApiError status 409 if that file was changed by something else meanwhile. `overwrite: true`
 * skips that protection (use it only after the person chose to overwrite).
 * @param {string} name
 * @param {{overwrite?: boolean}} [options]
 */
async function save(name, { overwrite = false } = {}) {
  const s = store.get();
  const yaml = s.yaml;
  const revision = !overwrite && s.revision && s.path === savedPathFor(name) ? s.revision : undefined;
  store.set({ saving: true });
  try {
    const doc = await api.saveRecipe(name, yaml, revision);
    const savedText = typeof doc.yaml === 'string' ? doc.yaml : yaml;
    store.set({
      path: doc.path ?? null,
      name: doc.name ?? name,
      savedYaml: savedText,
      revision: typeof doc.revision === 'string' ? doc.revision : null,
    });
    if (savedText !== store.get().yaml && store.get().yaml === yaml) setYaml(savedText, { immediate: true });
    saveDraft();
    return doc;
  } finally {
    store.set({ saving: false });
  }
}

function close() {
  if (checkAbort) checkAbort.abort();
  scheduleCheck.cancel();
  store.set({ ...EMPTY });
  saveDraft.flush();
}

/** Restore the draft from the previous visit. Returns true when one was restored. */
function restore() {
  const draft = persist.get('recipe');
  if (!draft || typeof draft.yaml !== 'string') return false;
  store.set({
    ...EMPTY,
    phase: 'open',
    path: draft.path ?? null,
    name: draft.name ?? '',
    yaml: draft.yaml,
    baseYaml: typeof draft.baseYaml === 'string' ? draft.baseYaml : draft.yaml,
    baseLabel: draft.baseLabel ?? '',
    savedYaml: typeof draft.savedYaml === 'string' ? draft.savedYaml : null,
    revision: typeof draft.revision === 'string' ? draft.revision : null,
  });
  scheduleCheck.flush();
  void expandCompactDraft(draft.yaml);
  return true;
}

const toList = (value) => (Array.isArray(value) ? value : typeof value === 'string' && value ? [value] : []).filter((v) => typeof v === 'string');

export const recipe = {
  get: store.get,
  subscribe: store.subscribe,
  load,
  openPath,
  setYaml,
  applyEdits,
  save,
  close,
  restore,
  checkNow: () => scheduleCheck.flush(),

  /** The saved-recipe path a name would be written to. */
  savedPathFor,

  /** Text differs from what is on disk (or was never saved). */
  isDirty() {
    const s = store.get();
    return s.phase === 'open' && s.yaml !== s.savedYaml;
  },
  /** Text differs from the starting point. */
  hasChanges() {
    const s = store.get();
    return s.phase === 'open' && s.yaml !== s.baseYaml;
  },
  revertToBase() {
    setYaml(store.get().baseYaml, { immediate: true });
  },
  /** Use the current text as the new starting point for the Changes view. */
  rebase(label) {
    const s = store.get();
    store.set({ baseYaml: s.yaml, baseLabel: label ?? s.baseLabel });
    saveDraft();
  },

  /**
   * Ask the form and the YAML pane to bring an option into view.
   * @param {string} path issue-style path, e.g. "train.losses[2].loss_weight"
   */
  requestFocus: (path) => focusChannel.emit(path),
  onFocusRequest: (fn) => focusChannel.on(fn),

  /**
   * Image folders named by the recipe's datasets, read from the checked config.
   * @returns {Array<{key: string, name: string, type: string, gt: string[], lq: string[]}>}
   */
  /**
   * The image folders each dataset of the checked recipe names, with the facts that decide how the
   * trainer pairs and crops them (so a folder check can ask the same question the trainer will).
   * @returns {Array<{key, name, type, gt: string[], lq: string[], phase: 'train'|'val'|'test'|null,
   *   filenameTemplate: string|null, metaInfo: string|null, annotationFile: string|null,
   *   storage: 'disk'|'lmdb', gtSize: number|null}>}
   *   phase is the dataset key before its first "_" (the trainer's rule); gtSize is gt_size, or
   *   lq_size times the recipe scale.
   */
  datasetFolders() {
    const config = store.get().config;
    const datasets = config?.datasets;
    if (!datasets || typeof datasets !== 'object') return [];
    const scale = Number.isInteger(config.scale) ? config.scale : null;
    return Object.entries(datasets)
      .filter(([, ds]) => ds && typeof ds === 'object')
      .map(([key, ds]) => {
        const phase = String(key).split('_')[0];
        const gtSize = Number.isInteger(ds.gt_size) ? ds.gt_size : Number.isInteger(ds.lq_size) && scale ? ds.lq_size * scale : null;
        return {
          key,
          name: typeof ds.name === 'string' ? ds.name : key,
          type: typeof ds.type === 'string' ? ds.type : '',
          gt: toList(ds.dataroot_gt),
          lq: toList(ds.dataroot_lq),
          phase: ['train', 'val', 'test'].includes(phase) ? phase : null,
          filenameTemplate: typeof ds.filename_tmpl === 'string' && ds.filename_tmpl ? ds.filename_tmpl : null,
          metaInfo: typeof ds.meta_info === 'string' && ds.meta_info ? ds.meta_info : null,
          annotationFile: typeof ds.annotation_file === 'string' && ds.annotation_file ? ds.annotation_file : null,
          storage: ds.io_backend && typeof ds.io_backend === 'object' && ds.io_backend.type === 'lmdb' ? 'lmdb' : 'disk',
          gtSize,
        };
      });
  },
};

/** Split an issue path such as `train.losses[2].type` or `/train/losses/2/type` into segments. */
export function parsePath(path) {
  if (Array.isArray(path)) return path;
  if (typeof path !== 'string' || !path) return [];
  const out = [];
  for (const part of path.split(/[./]|\[|\]/)) {
    if (part === '') continue;
    out.push(/^\d+$/.test(part) ? Number(part) : part);
  }
  return out;
}

/** Read a nested value; returns undefined when any step is missing. */
export function getAt(root, path) {
  let node = root;
  for (const key of path) {
    if (node === null || node === undefined) return undefined;
    node = node[key];
  }
  return node;
}
