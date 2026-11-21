// Image folders being checked: the inputs the person typed and the backend's real scan of them.

import { api } from '../api.js';
import { createStore, persist } from '../store.js';
import { recipe } from './recipe.js';

const saved = persist.get('data', {});

/**
 * Request fields beyond the two folders and the scale, as POST /api/data/scan names them. They describe
 * how the trainer will pair and crop the files, so they normally come from a recipe dataset
 * (see recipe.datasetFolders()) and are not kept across reloads.
 * @typedef {object} ScanOptions
 * @property {string[]} [gtRoots] several folders, paired by position with lqRoots (replaces gt)
 * @property {string[]} [lqRoots] (replaces lq)
 * @property {string[]} [validationRoots] folders to compare against for the split overlap check
 * @property {string} [filenameTemplate] applied to the ground truth name without extension; '{}' = same name
 * @property {'train'|'val'|'test'} [phase]
 * @property {number} [gtPatchSize] training crop on the ground truth side
 * @property {'folder'|'meta_info'|'lmdb'} [source]
 * @property {string} [metaInfo] manifest path, for source 'meta_info'
 * @property {string} [task] the recipe's task id
 */
const OPTION_KEYS = ['gtRoots', 'lqRoots', 'validationRoots', 'filenameTemplate', 'phase', 'gtPatchSize', 'source', 'metaInfo', 'task'];

/** Keep only known, non-empty option fields, in a fixed key order (so two equal sets compare equal). */
function cleanOptions(options) {
  const out = {};
  if (!options || typeof options !== 'object') return out;
  for (const key of OPTION_KEYS) {
    const value = options[key];
    if (value === undefined || value === null || value === '') continue;
    if (Array.isArray(value)) {
      const list = value.filter((item) => typeof item === 'string' && item.trim()).map((item) => item.trim());
      if (list.length) out[key] = list;
    } else out[key] = value;
  }
  return out;
}

const store = createStore({
  /** folder of ground-truth (high-quality) images */
  gt: typeof saved.gt === 'string' ? saved.gt : '',
  /** folder of low-quality inputs; '' when the recipe makes them on the fly */
  lq: typeof saved.lq === 'string' ? saved.lq : '',
  /** integer scale to check pair sizes against; null = do not check sizes */
  scale: Number.isInteger(saved.scale) ? saved.scale : null,
  /** @type {ScanOptions} extra request fields; {} for a plain two-folder check */
  options: {},

  /** 'idle' | 'scanning' | 'done' | 'error' */
  phase: 'idle',
  /**
   * The answer of POST /api/data/scan: {source, ok, counts:{gt,lq,paired}, roots:[{id,gt,lq?,counts}],
   * issues:[{level,path,message,rootId?}], samples:[…], truncated, limits, totals}.
   */
  result: null,
  error: null,
  /** the inputs `result` belongs to: {gt, lq, scale, options} */
  scanned: null,

  /**
   * The last data audit of a recipe (classification, segmentation and detection, where the service lists
   * and opens the files itself as an analysis job), or null. Kept for the session only.
   * @type {null | {yaml: string, signature: string|null, jobId: string|null,
   *   status: 'starting'|'running'|'completed'|'failed'|'cancelled',
   *   errors: number|null, warnings: number|null, error: string|null}}
   */
  audit: null,
});

let abort = null;

/** Patch the inputs. `options`, when given, replaces the whole option set (pass {} to clear it). */
function setInputs(patch) {
  const next = { ...patch };
  if ('options' in next) next.options = cleanOptions(next.options);
  store.set(next);
  const s = store.get();
  persist.set('data', { gt: s.gt, lq: s.lq, scale: s.scale });
}

/** The request body for the current inputs. */
function requestFor({ gt, lq, scale, options }) {
  const params = { ...options };
  // The backend refuses gt together with gtRoots, so several roots replace the single folder.
  if (!params.gtRoots) params.gt = gt;
  if (!params.lqRoots && lq) params.lq = lq;
  if (Number.isInteger(scale)) params.scale = scale;
  return params;
}

async function scan() {
  const s = store.get();
  const options = cleanOptions(s.options);
  const gt = s.gt.trim();
  if (!gt && !options.gtRoots) return;
  if (abort) abort.abort();
  const control = new AbortController();
  abort = control;
  const inputs = { gt, lq: s.lq.trim(), scale: s.scale, options };
  store.set({ phase: 'scanning', error: null });
  try {
    const result = await api.scan(requestFor(inputs), { signal: control.signal });
    if (control.signal.aborted) return;
    store.set({ phase: 'done', result, scanned: inputs, error: null });
  } catch (error) {
    if (error?.name === 'AbortError') return;
    store.set({ phase: 'error', result: null, scanned: inputs, error: error.message });
  }
}

/** Tasks whose data the service audits as a job, instead of the paired folder check. */
export const AUDIT_TASKS = ['image_classification', 'semantic_segmentation', 'object_detection'];
/** Tasks whose recipes do not upscale an image: their summaries show no scale. */
export const NO_SCALE_TASKS = [...AUDIT_TASKS, 'tabular_regression'];

/**
 * What a data audit depends on in the open recipe: its task, the task's options (class names and the
 * like) and its datasets. Editing anything else (a learning rate, the iteration count) leaves an audit
 * valid. Null while the parsed recipe does not belong to the current text.
 */
function auditSignature() {
  const r = recipe.get();
  if (!r.configCurrent || !r.config || typeof r.config !== 'object') return null;
  return JSON.stringify([r.config.task ?? null, r.config.task_options ?? null, r.config.datasets ?? null]);
}

/**
 * Record where the data audit of a recipe stands, so the shell can say it without owning the job.
 * Pass null to forget it. `yaml` is the recipe text the audit was started with; an update for the same
 * text keeps the signature taken when it started.
 */
function setAudit(audit) {
  if (!audit) {
    store.set({ audit: null });
    return;
  }
  const count = (value) => (Number.isFinite(value) ? value : null);
  const yaml = typeof audit.yaml === 'string' ? audit.yaml : '';
  const before = store.get().audit;
  const sameAudit = Boolean(before) && before.yaml === yaml;
  // The signature is taken when the audit starts. While the recipe is being re-checked it cannot be
  // read yet (the parsed recipe belongs to the earlier text); auditIsCurrent() fills it in then.
  const signature = (sameAudit && before.signature) || (yaml === recipe.get().yaml ? auditSignature() : null);
  // A status update of the audit already recorded does not change whether it is current.
  if (!sameAudit) lastCurrent = yaml === recipe.get().yaml;
  store.set({
    audit: {
      yaml,
      signature,
      jobId: audit.jobId ?? null,
      status: audit.status ?? 'running',
      errors: count(audit.errors),
      warnings: count(audit.warnings),
      error: audit.error ?? null,
    },
  });
  followAudit();
}

let lastCurrent = true;
let followTimer = null;
/** The job whose finished result could not be read, and how many times in a row. */
let unread = { jobId: null, tries: 0 };

/**
 * Keep a running audit's status true while the Data screen is not open (it follows the job itself
 * while it is). Asks the service every few seconds until the job has ended, then stops.
 *
 * A finished audit is recorded as completed only once its result was read: a result's counts are
 * absent when they are zero, so "not read" must never be stored as "completed" (it would read as
 * "no problems"). A job the service no longer has is forgotten instead of followed for ever.
 */
function followAudit() {
  const audit = store.get().audit;
  if (!audit || audit.status !== 'running' || !audit.jobId) {
    clearTimeout(followTimer);
    followTimer = null;
    return;
  }
  if (followTimer) return;
  followTimer = setTimeout(async () => {
    followTimer = null;
    const asked = store.get().audit;
    if (!asked || asked.status !== 'running' || !asked.jobId) return;
    const same = () => {
      const now = store.get().audit;
      return now && now.jobId === asked.jobId && now.status === 'running' ? now : null;
    };
    try {
      const job = await api.analysisJob(asked.jobId);
      if (job?.status === 'completed') {
        let totals = null;
        try {
          totals = (await api.analysisResult(asked.jobId))?.result?.totals ?? {};
        } catch {
          // Read again on the next tick; after a few failures say that the result could not be read.
        }
        const now = same();
        if (now && totals) {
          unread = { jobId: null, tries: 0 };
          setAudit({ ...now, status: 'completed', errors: totals.error, warnings: totals.warning });
        } else if (now) {
          unread = { jobId: asked.jobId, tries: unread.jobId === asked.jobId ? unread.tries + 1 : 1 };
          if (unread.tries >= 5) {
            unread = { jobId: null, tries: 0 };
            setAudit({ ...now, status: 'failed', error: 'The check finished, but its result could not be read.' });
          }
        }
      } else if (job?.status === 'failed' || job?.status === 'cancelled') {
        const now = same();
        if (now) setAudit({ ...now, status: job.status, error: job.status === 'failed' ? (job.error ?? null) : null });
      }
    } catch (error) {
      // The service no longer has this job: nothing is left to follow. Any other failure (no answer,
      // a busy service): ask again on the next tick.
      if (error?.status === 404 && same()) setAudit(null);
    }
    followAudit();
  }, 3000);
}

/**
 * True when the recorded audit still describes the open recipe's data. While the recipe is being
 * re-checked after an edit the previous answer is kept, so the screen does not flicker.
 */
function auditIsCurrent() {
  const audit = store.get().audit;
  if (!audit) return false;
  const r = recipe.get();
  if (r.phase !== 'open') return false;
  if (audit.yaml === r.yaml) {
    lastCurrent = true;
    // Started while the recipe was being re-checked: take the signature now that it can be read, so a
    // later edit outside the datasets does not mark this audit out of date.
    if (!audit.signature) audit.signature = auditSignature();
  } else if (!audit.signature) lastCurrent = false;
  else {
    const now = auditSignature();
    if (now !== null) lastCurrent = now === audit.signature;
  }
  return lastCurrent;
}

export const data = {
  get: store.get,
  subscribe: store.subscribe,
  setInputs,
  scan,
  setAudit,
  auditIsCurrent,
  /** True when the shown result belongs to the inputs currently typed (folders, scale and options). */
  isCurrent() {
    const s = store.get();
    return (
      Boolean(s.scanned) &&
      s.scanned.gt === s.gt.trim() &&
      s.scanned.lq === s.lq.trim() &&
      s.scanned.scale === s.scale &&
      JSON.stringify(s.scanned.options ?? {}) === JSON.stringify(cleanOptions(s.options))
    );
  },
};
