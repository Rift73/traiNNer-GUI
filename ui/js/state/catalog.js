// What the installed trainer offers: tasks, networks, losses, optimizers, schedulers, dataset types and
// metrics (GET /api/catalog, fetched once).
// Also the words each task uses for its two image folders, so a segmentation recipe is not described
// in super-resolution terms.

import { api } from '../api.js';
import { createStore } from '../store.js';

const store = createStore({
  /** 'idle' | 'loading' | 'ready' | 'error' */
  phase: 'idle',
  error: null,
  /**
   * [{id, label, networkField: 'network_g'|'network_ae', trainingOnly, architectures: string[]|null (null = any),
   *   recommendedArchitectures: string[], datasetTypes: string[] (empty = unrestricted),
   *   excludedArchitectures: string[], taskOptions: JSON Schema|null}]
   */
  tasks: [],
  /** [{id, task, registries, schemaRef}] */
  architectures: [],
  /** [{id}] */
  losses: [],
  /**
   * [{id, displayName, signatureAvailable, acceptsExtraKeywords, parameters: [{name, required, default, choices?}]}]
   * displayName: the spelling recipes write (AdamW for the id adamw); choices: the values a parameter takes.
   */
  optimizers: [],
  /** [{id, kind, allowedParents, parameters: [{name, required, default, choices?}]}] */
  schedulers: [],
  /** [{id}] */
  datasets: [],
  /** [{id, requiresReference, signatureAvailable, acceptsExtraKeywords, parameters: [{name, required, default, choices?}]}] */
  metrics: [],
  /** Counts each time the lists were read again after the first time: a form built on them rebuilds itself. */
  revision: 0,
  /** `source|generatedAt` of the answer in use; '' until the first answer. */
  stamp: '',
});

const listsOf = (result) => ({
  tasks: Array.isArray(result?.tasks) ? result.tasks : [],
  architectures: Array.isArray(result?.architectures) ? result.architectures : [],
  losses: Array.isArray(result?.losses) ? result.losses : [],
  optimizers: Array.isArray(result?.optimizers) ? result.optimizers : [],
  schedulers: Array.isArray(result?.schedulers) ? result.schedulers : [],
  datasets: Array.isArray(result?.datasets) ? result.datasets : [],
  metrics: Array.isArray(result?.metrics) ? result.metrics : [],
  // Which reading of the lists this is (the service's `source` and `generatedAt`), to tell a newer one.
  stamp: `${result?.source ?? ''}|${result?.generatedAt ?? ''}`,
});

let pending = null;

function load() {
  if (store.get().phase === 'ready') return Promise.resolve(store.get());
  if (pending) return pending;
  store.set({ phase: 'loading', error: null });
  pending = api
    .catalog()
    .then((result) => {
      store.set({ phase: 'ready', ...listsOf(result) });
      return store.get();
    })
    .catch((error) => {
      store.set({ phase: 'error', error: error.message });
      throw error;
    })
    .finally(() => {
      pending = null;
    });
  return pending;
}

/**
 * The service's lists changed (it read them again from the installation): take the new answer.
 * @param {object} result the catalog as the service just returned it
 */
function replace(result) {
  store.set({ phase: 'ready', error: null, ...listsOf(result), revision: store.get().revision + 1 });
}

/** Fallback label for a task id the catalog does not list: the id with spaces, first letter capitalised. */
function plainLabel(id) {
  const text = String(id ?? '').replace(/_/g, ' ').trim();
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : '';
}

// The catalog does not say what a task's dataroot_gt / dataroot_lq folders hold, so the wording lives
// here, keyed by task id. Unknown tasks get the neutral option names.
const FOLDER_WORDS = {
  image_restoration: {
    gt: { name: 'Ground truth', short: 'GT', what: 'the clean images the network should reproduce' },
    lq: { name: 'Low quality', short: 'LQ', what: 'the degraded inputs' },
    pair: 'pair',
    paired: true,
  },
  semantic_segmentation: {
    gt: { name: 'Masks', short: 'Mask', what: 'integer PNG masks, one class ID per pixel' },
    lq: { name: 'Images', short: 'Image', what: 'the pictures to segment' },
    pair: 'image and mask pair',
    paired: true,
  },
  unknown: {
    gt: { name: 'Target folder', short: 'Target', what: 'the option dataroot_gt' },
    lq: { name: 'Input folder', short: 'Input', what: 'the option dataroot_lq' },
    pair: 'pair',
    paired: true,
  },
};
// Classification reads one folder per split (the option dataroot_lq) with one subfolder per class
// name; it has no second folder, so `paired` is false and `gt` only names the unused option.
FOLDER_WORDS.image_classification = {
  gt: { name: 'Target folder', short: 'Target', what: 'not used: the class of an image is the name of its subfolder' },
  lq: { name: 'Class folders', short: 'Classes', what: 'one subfolder of images per class name' },
  pair: 'image',
  paired: false,
};
// Detection reads one folder of images per split plus a COCO JSON file (the option annotation_file).
FOLDER_WORDS.object_detection = {
  gt: { name: 'Target folder', short: 'Target', what: 'not used: the boxes come from the annotation file' },
  lq: { name: 'Images', short: 'Images', what: 'the pictures the annotation file refers to' },
  pair: 'image',
  paired: false,
  /** a second input that is a file, not a folder */
  file: { key: 'annotation_file', name: 'Annotation file', what: 'a COCO JSON file with images, categories and annotations' },
};
// A table task reads no image folder: each dataset is one CSV file of numbers (the option meta_info),
// and its columns are named by the task options `features` and `targets`.
FOLDER_WORDS.tabular_regression = {
  gt: { name: 'Target folder', short: 'Target', what: 'not used: the targets are columns of the CSV file' },
  lq: { name: 'Input folder', short: 'Input', what: 'not used: the features are columns of the CSV file' },
  pair: 'record',
  paired: false,
  /** the dataset is a table of records, not images: no folder is asked for, listed or shown */
  records: true,
  file: { key: 'meta_info', name: 'CSV file', what: 'a table of numbers with one named column per feature and target', extensions: ['.csv'] },
};
FOLDER_WORDS.recurrent_restoration = FOLDER_WORDS.image_restoration;
FOLDER_WORDS.autoencoder = {
  ...FOLDER_WORDS.image_restoration,
  gt: { name: 'Images', short: 'Image', what: 'the images the autoencoder learns to reconstruct' },
};

/**
 * Words for a task's two image folders.
 * @param {string|null|undefined} taskId summary.task from the recipe check
 * @returns {{gt: {name, short, what}, lq: {name, short, what}, pair: string, paired: boolean}}
 *   gt describes the option dataroot_gt, lq the option dataroot_lq. `paired: false` means the task reads
 *   only the lq folder (classification, detection), so a part must not ask for or describe a second
 *   folder. `file`, when present, is {key, name, what} for a second input that is a file (detection's
 *   annotation_file).
 */
export function folderWords(taskId) {
  if (FOLDER_WORDS[taskId]) return FOLDER_WORDS[taskId];
  if (!taskId) return FOLDER_WORDS.image_restoration;
  // A task this file has no words for: the catalog still says whether it trains on whole images
  // (one folder) or on aligned crops of two folders.
  const whole = store.get().tasks.find((task) => task.id === taskId)?.datasetGeometry === 'whole_images';
  return whole ? { ...FOLDER_WORDS.unknown, pair: 'image', paired: false } : FOLDER_WORDS.unknown;
}

export const catalog = {
  get: store.get,
  subscribe: store.subscribe,
  /** Fetch once; resolves to the store state. Safe to call from anywhere. */
  load,
  /** Take a newer catalog answer (after POST /api/catalog/refresh); `revision` goes up by one. */
  replace,
  /** Catalog entry of a task, or null when it is not listed (or the catalog has not loaded). */
  task: (id) => store.get().tasks.find((task) => task.id === id) ?? null,
  /** Display name of a task id: the catalog's label when known. */
  taskLabel: (id) => store.get().tasks.find((task) => task.id === id)?.label ?? plainLabel(id),
  /**
   * Networks a task accepts, recommended ones first.
   * @returns {{ids: string[], recommended: Set<string>, restricted: boolean}} restricted = the task
   *   lists its networks; otherwise every installed network except the task's exclusions.
   */
  networksFor(taskId) {
    const s = store.get();
    const task = s.tasks.find((item) => item.id === taskId);
    const all = s.architectures.map((item) => item.id);
    if (!task) return { ids: all, recommended: new Set(), restricted: false };
    const excluded = new Set(task.excludedArchitectures ?? []);
    const allowed = Array.isArray(task.architectures) ? task.architectures.filter((id) => !excluded.has(id)) : all.filter((id) => !excluded.has(id));
    const recommended = new Set(task.recommendedArchitectures ?? []);
    const ids = [...allowed.filter((id) => recommended.has(id)), ...allowed.filter((id) => !recommended.has(id))];
    return { ids, recommended, restricted: Array.isArray(task.architectures) };
  },
};
