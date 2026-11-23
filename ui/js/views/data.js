// Data screen: check that the image folders a recipe trains on are there, paired and correctly
// sized, and look at real pairs before spending GPU hours. Everything shown comes from the scan.

// dom.js's mount() is imported as fill(): this module must export its own mount().
import { h, mount as fill, fmt, uid } from '../dom.js';
import { api, imageUrl, newRequestId, PREVIEW_MAX, THUMB_SIZE } from '../api.js';
import { data, AUDIT_TASKS } from '../state/data.js';
import { recipe } from '../state/recipe.js';
import { conn } from '../state/conn.js';
import { folderWords } from '../state/catalog.js';
import { analysisHref } from '../actions.js';
import { toast, issueList, notice, roleChip, countIssues, badge, openDialog } from '../ui.js';
import { createCompare } from '../components/compare.js';
import { videoPrep, counts as videoCounts, jobActive, exampleUnchosen, VIDEO_DATASET } from '../state/video-prep.js';
import { revealVideoPanel } from '../recipe/video-prep.js';

const SCALES = [1, 2, 3, 4, 8];
const EMPTY_BEFORE = 'Check your folders to look at real images here.';
const EMPTY_NONE = 'The check returned no images to look at.';
const SEGMENTATION = 'semantic_segmentation';
/** `source` of a scan answer: folder (default), meta_info (explicit manifest) or lmdb. */
const SOURCE_WORDS = { folder: 'Folders', meta_info: 'Manifest (meta_info file)', lmdb: 'LMDB databases' };
const OVERLAP_PREFIX = 'Same physical file as validation path';
const CLASSIFICATION = 'image_classification';
const DETECTION = 'object_detection';
// The trainer's image dataset that degrades the ground truth on the fly (RealESRGANDataset): the one class
// here that never reads dataroot_lq. Its clip counterpart, VIDEO_DATASET, has an entry of its own.
const OTF_DATASET = 'realesrgandataset';
/** A dataset type as the trainer looks it up: lower case, with or without the registry's _traiNNer suffix. */
const datasetKind = (type) => String(type ?? '').toLowerCase().replace(/_trainner$/, '');
const UNCHECKED_WHAT = { [CLASSIFICATION]: 'classification folders', [DETECTION]: 'detection data' };
const PACK_LINE = 'Writes a full copy as LMDB databases on the trainer machine, as a separate job.';

/** The plain sentence for a task whose dataset reads a single folder (folderWords(task).paired === false). */
export function unpairedLine(task) {
  const what = UNCHECKED_WHAT[task] ?? 'data for this task';
  const look = task === CLASSIFICATION ? ' Use Browse to look at the class subfolders.' : task === DETECTION ? ' List images shows the files in the folder.' : '';
  return `This version of the trainer service does not check ${what}: its folder check needs a ground truth folder and pairs.${look}`;
}

/**
 * What "Pack these folders into a portable dataset…" passes: exactly what was checked. The answer's
 * `roots` carry the folders as the backend resolved them; the request is used when they are absent.
 * @returns {{gtRoots: string[], lqRoots: string[], scale: number|null, filenameTemplate: string|null}|null}
 */
export function packDataset(result, scanned) {
  if (!scanned) return null;
  const roots = Array.isArray(result?.roots) ? result.roots : [];
  const fromRoots = roots.length && roots.every((root) => root?.gt && root?.lq);
  const gtRoots = fromRoots ? roots.map((root) => root.gt) : scanned.options?.gtRoots ?? (scanned.gt ? [scanned.gt] : []);
  const lqRoots = fromRoots ? roots.map((root) => root.lq) : scanned.options?.lqRoots ?? (scanned.lq ? [scanned.lq] : []);
  if (!gtRoots.length || gtRoots.length !== lqRoots.length) return null;
  return {
    gtRoots,
    lqRoots,
    scale: Number.isInteger(scanned.scale) ? scanned.scale : null,
    filenameTemplate: typeof scanned.options?.filenameTemplate === 'string' ? scanned.options.filenameTemplate : null,
  };
}
const MASK_LINE = 'Mask pixels are class IDs (0, 1, …, 255 for ignored), so a mask preview looks almost black.';

// ───────── pure helpers (exported for checks) ─────────

/** "Shown reduced from 4000 × 3000" when the preview is smaller than the file, '' otherwise or when unknown. */
export function reducedNote(width, height) {
  if (!Number.isFinite(width) || !Number.isFinite(height)) return '';
  return Math.max(width, height) > PREVIEW_MAX ? `Shown reduced from ${fmt.dims(width, height)}` : '';
}

/**
 * The open recipe's dataset whose first folders are exactly these (gt, lq), or null.
 * @param {Array<{gt: string[], lq: string[]}>} sets recipe.datasetFolders()
 */
export function recipeSetFor(sets, gt, lq) {
  const g = String(gt ?? '').trim();
  const l = String(lq ?? '').trim();
  if (!g) return null;
  return (sets ?? []).find((set) => (set.gt?.[0] ?? '') === g && (set.lq?.[0] ?? '') === l) ?? null;
}

/**
 * Issues about the two folders themselves (a missing folder comes back as an issue whose `path` is the
 * folder). The scan's `roots` carry the resolved absolute folders; the typed values are matched too.
 * @returns {{gt: object[], lq: object[]}}
 */
export function folderIssues(result, scanned) {
  const roots = Array.isArray(result?.roots) ? result.roots : [];
  const gt = new Set([scanned?.gt, ...roots.map((root) => root?.gt)].filter((p) => typeof p === 'string' && p));
  const lq = new Set([scanned?.lq, ...roots.map((root) => root?.lq)].filter((p) => typeof p === 'string' && p));
  const out = { gt: [], lq: [] };
  for (const issue of result?.issues ?? []) {
    if (typeof issue?.path !== 'string') continue;
    if (gt.has(issue.path)) out.gt.push(issue);
    else if (lq.has(issue.path)) out.lq.push(issue);
  }
  return out;
}

/**
 * "Showing the first 100 of 2,431 pairs." from the scan's `limits` and `totals` only; '' when nothing was cut.
 * @param {'samples'|'issues'} key
 */
export function limitLine(result, key, one, many) {
  const limit = result?.limits?.[key];
  const total = result?.totals?.[key];
  if (!Number.isFinite(limit) || !Number.isFinite(total) || total <= limit) return '';
  return `Showing the first ${fmt.int(limit)} of ${fmt.int(total)} ${total === 1 ? one : many}.`;
}

const capitalize = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : '');
/** "Ground truth folder"; "Target folder" and "Class folders" already say folder. */
const folderLabel = (name) => (/folders?$/i.test(name) ? name : `${name} folder`);

const PHASE_DATA = { train: 'Training data', val: 'Validation data', test: 'Test data' };
/** "training data", or "validation data val_2" when the key is more than the phase. */
const dataWords = (set) => {
  const word = (PHASE_DATA[set?.phase] ?? 'Data').toLowerCase();
  return set?.phase && set.key !== set.phase ? `${word} ${set.key}` : set?.phase ? word : `data ${set?.key ?? ''}`.trim();
};

/**
 * The recipe edit that puts a chosen path into one place of a dataset: a list keeps its other entries
 * (the first is replaced), a single path stays a single path, an absent folder becomes a one-item list.
 * @returns {{path: string[], value: string|string[]}}
 */
export function placeEdit(config, key, option, picked, mode = 'folder') {
  const current = config?.datasets?.[key]?.[option];
  let value;
  if (Array.isArray(current) && current.length) value = [picked, ...current.slice(1)];
  else if (typeof current === 'string' || mode === 'file') value = picked;
  else value = [picked];
  return { path: ['datasets', key, option], value };
}

/**
 * "Path was not found" (and "Expected a file") issues of the recipe check, by dataset place:
 * `${key}|${option}|${index}` -> message. The service reports them at datasets.<key>.<option>[.<index>].
 * @returns {Map<string, string>}
 */
export function placeIssues(issues) {
  const out = new Map();
  for (const issue of Array.isArray(issues) ? issues : []) {
    const path = Array.isArray(issue?.path) ? issue.path.join('.') : String(issue?.path ?? '');
    const match = /^datasets\.(.+)\.(dataroot_gt|dataroot_lq|annotation_file|meta_info)(?:\.(\d+))?$/.exec(path);
    if (!match) continue;
    const message = String(issue.message ?? '');
    if (!/^(Path was not found|Expected a file)/.test(message)) continue;
    out.set(`${match[1]}|${match[2]}|${match[3] ?? 0}`, message);
  }
  return out;
}

// ───────── alignment check (POST /api/data/alignment) ─────────

export const ALIGN_CHANNELS = [
  ['luma', 'Brightness'],
  ['blueMinusLuma', 'Blue against brightness'],
  ['redMinusLuma', 'Red against brightness'],
];
const ASSESSMENT_WORDS = {
  noClearShift: 'No clear shift',
  possibleShift: 'Possible shift',
  ambiguous: 'Ambiguous: several offsets fit about equally well',
  inconsistentAcrossPatches: 'The patches disagree',
  searchBoundary: 'Best match at the edge of the search range: the real shift may be larger',
  insufficientTexture: 'Not enough texture to measure',
};
const ALIGN_SCOPE = 'This is a local estimate at up to nine places in one image pair. Nothing is corrected and training is not changed.';

/** Plain words for an alignment assessment; an unknown word is shown unchanged. */
export function assessmentText(word) {
  return ASSESSMENT_WORDS[word] ?? String(word ?? '');
}

/** "0.00"-style text for a measured value (never "-0.00"); '' when absent. */
export function fixed(value, digits = 2) {
  if (!Number.isFinite(value)) return '';
  const text = value.toFixed(digits);
  return Number(text) === 0 ? (0).toFixed(digits) : text;
}

/**
 * The overlap check the open recipe allows: its training dataset's folders scanned with every validation
 * dataset's folders as `validationRoots` (the trainer's phase is the dataset key before the first "_").
 * @param {Array<{key: string, gt: string[], lq: string[]}>} sets recipe.datasetFolders()
 * @returns {{body: object, trainKey: string, valKeys: string[]}|null}
 */
export function overlapRequest(sets) {
  const phaseOf = (key) => String(key ?? '').split('_')[0];
  const list = Array.isArray(sets) ? sets : [];
  const train = list.find((set) => phaseOf(set.key) === 'train' && set.gt?.length);
  const vals = list.filter((set) => phaseOf(set.key) === 'val' && (set.gt?.length || set.lq?.length));
  if (!train || !vals.length) return null;
  const validationRoots = [...new Set(vals.flatMap((set) => [...(set.gt ?? []), ...(set.lq ?? [])]))];
  return {
    body: { gtRoots: [...train.gt], ...(train.lq?.length ? { lqRoots: [...train.lq] } : {}), validationRoots },
    trainKey: train.key,
    valKeys: vals.map((set) => set.key),
  };
}

/**
 * Scan options for a recipe dataset (recipe.datasetFolders() item), as the trainer will read it.
 * @returns {import('../state/data.js').ScanOptions}
 */
export function optionsForSet(set, task) {
  const out = {};
  const gt = Array.isArray(set?.gt) ? set.gt : [];
  const lq = Array.isArray(set?.lq) ? set.lq : [];
  if (gt.length > 1 || lq.length > 1) {
    out.gtRoots = [...gt];
    if (lq.length) out.lqRoots = [...lq];
  }
  if (typeof set?.filenameTemplate === 'string' && set.filenameTemplate && set.filenameTemplate !== '{}') out.filenameTemplate = set.filenameTemplate;
  if (set?.phase) out.phase = set.phase;
  if (set?.phase === 'train' && Number.isInteger(set?.gtSize)) out.gtPatchSize = set.gtSize;
  if (set?.storage === 'lmdb') out.source = 'lmdb';
  else if (typeof set?.metaInfo === 'string' && set.metaInfo) {
    out.source = 'meta_info';
    out.metaInfo = set.metaInfo;
  }
  if (task) out.task = task;
  return out;
}

const PHASE_WORDS = { train: 'training', val: 'validation', test: 'test' };

/**
 * task_options.class_names against the subfolders listed (exact, case-sensitive names).
 * @returns {{missing: string[], extra: string[]}} names without a folder, folders not named
 */
export function compareClassNames(names, folders) {
  const named = (Array.isArray(names) ? names : []).map(String);
  const found = (Array.isArray(folders) ? folders : []).map(String);
  const foundSet = new Set(found);
  const namedSet = new Set(named);
  return { missing: named.filter((name) => !foundSet.has(name)), extra: found.filter((name) => !namedSet.has(name)) };
}

/**
 * One line saying how the check was asked, built only from the options sent; '' when there are none.
 * @param {object} options state/data.js options
 * @param {string|null} fromKey the recipe dataset key the options came from, or null
 */
export function optionsSummary(options, fromKey) {
  const o = options ?? {};
  if (!Object.keys(o).length) return '';
  const parts = [];
  if (o.gtRoots?.length > 1 || o.lqRoots?.length > 1) parts.push(`${fmt.int(Math.max(o.gtRoots?.length ?? 0, o.lqRoots?.length ?? 0))} folder pairs`);
  if (o.filenameTemplate) parts.push(`name template ${o.filenameTemplate}`);
  if (o.phase && !fromKey) parts.push(`as ${PHASE_WORDS[o.phase] ?? o.phase} data`);
  if (Number.isInteger(o.gtPatchSize)) parts.push(`training crop ${fmt.int(o.gtPatchSize)} px`);
  if (o.source === 'lmdb') parts.push('LMDB database');
  else if (o.source === 'meta_info') parts.push(o.metaInfo ? `manifest ${o.metaInfo}` : 'manifest');
  if (o.validationRoots?.length) parts.push(`compared with ${fmt.plural(o.validationRoots.length, 'validation folder')}`);
  const head = fromKey ? `Checked as the recipe's ${fromKey} dataset pairs them` : 'Checked with these options';
  return parts.length ? `${head}: ${parts.join(', ')}.` : `${head}.`;
}

const lowerFirst = (text) => (text ? text.charAt(0).toLowerCase() + text.slice(1) : '');

/**
 * One line from an alignment answer only, e.g. "Brightness: no clear shift (dx 0.00, dy 0.00). Colour
 * channels: no clear shift." `warn` is true when any assessment is not `noClearShift` or an estimate
 * is at the search limit.
 * @returns {{text: string, warn: boolean}}
 */
export function alignmentSummary(result) {
  const evidence = result?.evidence ?? {};
  const estimate = result?.estimate ?? {};
  const said = (key) => (evidence[key]?.assessment !== undefined ? lowerFirst(assessmentText(evidence[key].assessment)) : '');
  const parts = [];
  let warn = false;
  for (const [key] of ALIGN_CHANNELS) {
    const word = evidence[key]?.assessment;
    if (word !== undefined && word !== 'noClearShift') warn = true;
    if (estimate[key]?.atLimit === true) warn = true;
  }
  const luma = said('luma');
  if (luma) {
    const est = estimate.luma ?? {};
    const shift = [Number.isFinite(est.dx) ? `dx ${fixed(est.dx)}` : '', Number.isFinite(est.dy) ? `dy ${fixed(est.dy)}` : ''].filter(Boolean).join(', ');
    const limit = est.atLimit === true ? ', at the search limit' : '';
    parts.push(`Brightness: ${luma}${shift || limit ? ` (${shift}${shift ? limit : limit.slice(2)})` : ''}.`);
  }
  const blue = said('blueMinusLuma');
  const red = said('redMinusLuma');
  if (blue && red && blue === red) parts.push(`Colour channels: ${blue}.`);
  else {
    if (blue) parts.push(`Blue against brightness: ${blue}.`);
    if (red) parts.push(`Red against brightness: ${red}.`);
  }
  return { text: parts.join(' '), warn };
}

/** The counts an evidence entry carries, in words; absent counts are left out. */
export function evidenceCounts(entry) {
  const parts = [];
  if (Number.isFinite(entry?.informativePatches)) parts.push(`${fmt.plural(entry.informativePatches, 'informative patch', 'informative patches')}`);
  if (Number.isFinite(entry?.supportingPatches)) parts.push(`${fmt.int(entry.supportingPatches)} supporting`);
  if (Number.isFinite(entry?.distinctPeaks)) parts.push(fmt.plural(entry.distinctPeaks, 'distinct peak'));
  return parts.join(', ');
}

/**
 * Error messages per sample index, for error issues whose `path` names the sample's file
 * (its name, its GT path or its LQ path). Exact string match: the backend reports these forms.
 * @returns {Map<number, string[]>}
 */
export function errorsBySample(samples, issues) {
  const byPath = new Map();
  (samples ?? []).forEach((sample, index) => {
    for (const p of [sample?.name, sample?.gt, sample?.lq]) {
      if (typeof p !== 'string' || !p) continue;
      if (!byPath.has(p)) byPath.set(p, new Set());
      byPath.get(p).add(index);
    }
  });
  const out = new Map();
  for (const issue of issues ?? []) {
    if (issue?.level !== 'error' || typeof issue.path !== 'string') continue;
    for (const index of byPath.get(issue.path) ?? []) {
      if (!out.has(index)) out.set(index, []);
      out.get(index).push(String(issue.message ?? ''));
    }
  }
  return out;
}

/** "2 errors, 1 warning" or the all-clear sentence. */
export function issueSummary(issues) {
  const { errors, warnings } = countIssues(issues);
  if (!errors && !warnings) return 'No problems found in these folders.';
  return [errors ? fmt.plural(errors, 'error') : '', warnings ? fmt.plural(warnings, 'warning') : ''].filter(Boolean).join(', ');
}

/** Count line by the contact sheet; only real numbers. */
export function countLine(shown, counts, hasPairs) {
  const paired = counts?.paired;
  const gt = counts?.gt;
  if (hasPairs && Number.isFinite(paired) && paired > 0) return `${fmt.int(shown)} of ${fmt.plural(paired, 'pair')} shown`;
  if (Number.isFinite(gt) && gt > 0) return `${fmt.int(shown)} of ${fmt.plural(gt, 'image')} shown`;
  return `${fmt.plural(shown, 'image')} shown`;
}

const scaleText = (n) => (Number.isInteger(n) ? `${n}×` : 'Sizes not checked');

// ───────── analysis jobs started from this screen (data audit, identical files) ─────────

const AUDIT_LINE = 'Lists every dataset in the recipe and opens files to check them against the recipe: class names, annotation boxes, mask values and image sizes.';
const LEAK_LINE = 'Compares the encoded bytes of every training and validation file of the recipe. Large datasets take a while.';
const AUDIT_SAMPLES = 25;
const JOB_POLL_MS = 1000;
/** Reads of a job or its result before this screen gives up and offers "Try again". */
const RESULT_ATTEMPTS = 3;
const OFFLINE = 'The trainer service did not answer.';
/** The options chosen in the audit block, kept for the session. */
const auditChoice = { limit: '100', every: false };
const REG_LINE = "Measures, on a sample of the recipe's own pairs, how far each low quality image is shifted against its ground truth.";
/** Service defaults of a pair-registration job (studio/analysis_jobs.py). */
const REG_DEFAULTS = { samples: 12, samplesMin: 1, samplesMax: 64, radius: 6, radiusMin: 2, radiusMax: 12 };
/** The options chosen in the alignment block, kept for the session. */
const regChoice = { split: null, samples: String(REG_DEFAULTS.samples), radius: String(REG_DEFAULTS.radius) };

/** What the identical-files and alignment jobs depend on in the open recipe: its task, scale and datasets. */
function datasetsSignature() {
  const r = recipe.get();
  if (!r.configCurrent || !r.config || typeof r.config !== 'object') return null;
  return JSON.stringify([r.config.task ?? null, r.config.scale ?? null, r.config.datasets ?? null]);
}

/**
 * Session memory of one kind of job started here: {yaml, signature, jobId, status, error}, and whether a
 * completed one still describes the open recipe's data (an edit elsewhere in the recipe keeps it current;
 * while the recipe is re-checked the previous answer is kept, so the marker does not flicker).
 */
function sessionJob() {
  let memory = null;
  let lastCurrent = true;
  // Listeners, so a runner mounted later hears what a closed screen's late start answer recorded.
  const listeners = new Set();
  const notify = () => {
    for (const fn of [...listeners]) fn();
  };
  return {
    get: () => memory,
    /** @returns {() => void} unsubscribe */
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    record(next) {
      if (!next) {
        memory = null;
        notify();
        return;
      }
      const signature = memory && memory.yaml === next.yaml ? memory.signature : next.yaml === recipe.get().yaml ? datasetsSignature() : null;
      if (!memory || memory.yaml !== next.yaml) lastCurrent = true;
      memory = { yaml: next.yaml ?? '', signature, jobId: next.jobId ?? null, status: next.status, error: next.error ?? null };
      notify();
    },
    stale() {
      if (!memory || memory.status !== 'completed') return false;
      const r = recipe.get();
      if (r.phase !== 'open') return true;
      if (memory.yaml === r.yaml) lastCurrent = true;
      else if (!memory.signature) lastCurrent = false;
      else {
        const now = datasetsSignature();
        if (now !== null) lastCurrent = now === memory.signature;
      }
      return !lastCurrent;
    },
  };
}
/** The last "Look for identical files" job of the session. */
const leakJob = sessionJob();
/** The last "Measure alignment" job of the session. */
const regJob = sessionJob();

function memoImport(load) {
  let pending = null;
  return () => {
    if (!pending) {
      pending = load();
      pending.catch(() => {
        pending = null;
      });
    }
    return pending;
  };
}
const loadTaskData = memoImport(() => import('../runs/taskdata.js'));
const loadHoldout = memoImport(() => import('../runs/holdout.js'));
const loadRegistration = memoImport(() => import('../runs/registration.js'));

/** The task of the open recipe as last checked (the summary, else the last parsed config), or null. */
function recipeTask() {
  const r = recipe.get();
  return r.summary?.task ?? r.config?.task ?? null;
}

/**
 * Where the audit controls stand for the open recipe: hidden (a paired restoration task), 'none' (no
 * recipe), 'yaml' (the text does not parse), or ready.
 * @returns {{show: boolean, need: ''|'none'|'yaml'}}
 */
export function auditAvailability(r = recipe.get()) {
  if (r.phase !== 'open') return { show: true, need: 'none' };
  const task = r.summary?.task ?? r.config?.task ?? null;
  const known = AUDIT_TASKS.includes(task);
  if (r.check === 'done' && !r.configCurrent) return { show: known || !task, need: 'yaml' };
  return { show: known, need: '' };
}

/** The request body of "Check the data": `auditLimit` null opens every file. */
export function auditRequest(yaml, limit, requestId) {
  return { kind: 'task-data', yaml, auditLimit: limit, sampleLimit: AUDIT_SAMPLES, requestId };
}

/** The request body of "Look for identical files" (the service's default group limit). */
export function leakageRequest(yaml, requestId) {
  return { kind: 'split-leakage', yaml, requestId };
}

/** One line from an audit result's totals: "2 errors found in all datasets." / "No problems found." */
export function auditOutcome(result) {
  const totals = result?.totals ?? {};
  const errors = Number.isFinite(totals.error) ? totals.error : 0;
  const warnings = Number.isFinite(totals.warning) ? totals.warning : 0;
  if (errors || warnings) {
    const words = [errors ? fmt.plural(errors, 'error') : '', warnings ? fmt.plural(warnings, 'warning') : ''].filter(Boolean).join(' and ');
    return { kind: errors ? 'err' : 'warn', text: `${words} found in all datasets.` };
  }
  return { kind: 'ok', text: result?.complete === true ? 'No problems found.' : 'No problems found in the files that were opened.' };
}

/** One line from a split-leakage result: whether training and validation share identical files. */
export function leakageOutcome(result) {
  const counts = result?.counts ?? {};
  if (result?.hasTrainValidationOverlap === true) {
    const groups = Number.isFinite(counts.trainValidationGroups) ? counts.trainValidationGroups : null;
    return { kind: 'warn', text: groups !== null ? `${fmt.plural(groups, 'set', 'sets')} of identical files in training and validation.` : 'Training and validation share identical files.' };
  }
  if (result?.hasTrainValidationOverlap === false) {
    return { kind: 'ok', text: result.complete === true ? 'No identical files in training and validation.' : 'No identical files in training and validation among the files compared.' };
  }
  return { kind: '', text: 'The comparison finished.' };
}

/**
 * A job started from this screen, followed inline: controls and a short status for the left pane, the
 * result for the main area. Starts with the open recipe's YAML and one request id per click; "Try again"
 * after no answer (or a busy service) repeats the same request.
 * @param {object} spec
 * @returns {{controls: HTMLElement, main: HTMLElement, start: () => void, destroy: () => void}}
 */
function createJobRunner(spec) {
  let disposed = false;
  let timer = 0;
  let ctrl = null;
  /** The job this block follows: {jobId, yaml, task}. */
  let current = null;
  let shownJobId = null;
  let cleanupResult = null;
  let part = null;
  /** True when this visit saw the job start or run: only then does finishing switch the main area. */
  let live = false;

  const needBox = h('div', { class: 'data__auditneed', hidden: true });
  const startButton = h('button', { class: `btn ${spec.quiet ? '' : 'btn--primary'}`.trim(), type: 'button', onclick: () => start() }, spec.buttonLabel);
  const cancelButton = h('button', { class: 'btn btn--sm', type: 'button', hidden: true, onclick: () => cancel() }, 'Cancel');
  const jobLink = h('a', { class: 'data__auditlink', hidden: true }, 'Open in Analysis jobs');
  const optionsBox = h('div', { class: 'data__auditcontrols' }, h('p', { class: 'field__help data__auditline' }, spec.intro), spec.options ?? null);
  const statusLine = h('p', { class: 'data__auditstatus', role: 'status', hidden: true });
  const outcomeLine = h('p', { class: 'data__auditoutcome', role: 'status', hidden: true });
  const actions = h('div', { class: 'data__auditactions' }, startButton, cancelButton, jobLink);
  const messageBox = h('div', { class: 'data__auditmessage', hidden: true });
  const staleBox = h('div', { class: 'data__auditstale', hidden: true }, spec.staleText ? notice('warn', spec.staleText, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => start() }, 'Check again')) : null);
  const controls = h('div', { class: 'data__audit' }, needBox, optionsBox, actions, statusLine, outcomeLine, messageBox, staleBox);

  // Main area: the result, or an empty state that says what the job does and repeats the button.
  const resultHost = h('div', { class: 'data__auditresult' });
  // Not a second primary button: the one in the pane, beside the options, is the screen's main action.
  const emptyStart = h('button', { class: 'btn', type: 'button', onclick: () => start() }, spec.buttonLabel);
  const emptyText = h('p', { class: 'empty__text' }, spec.intro);
  const emptyBox = h('div', { class: 'empty data__auditempty' }, h('h2', { class: 'empty__title' }, spec.emptyTitle), emptyText, h('div', { class: 'empty__actions' }, emptyStart));
  // The result stays on screen after the recipe's data changed, so it says so itself as well: the pane's
  // marker is out of sight once the result is being read.
  const mainStale = h(
    'div',
    { class: 'data__auditstale data__auditstale--main', hidden: true },
    spec.staleText ? notice('warn', `${spec.staleText} What is shown below describes the earlier settings.`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => start() }, 'Check again')) : null,
  );
  const main = h('div', { class: 'data__auditmain', role: 'region', 'aria-label': spec.emptyTitle }, emptyBox, mainStale, resultHost);

  const setOutcome = (outcome) => {
    outcomeLine.hidden = !outcome?.text;
    outcomeLine.className = `data__auditoutcome${outcome?.kind ? ` data__auditoutcome--${outcome.kind}` : ''}`;
    // The sentence says the outcome; the tone only repeats it.
    outcomeLine.textContent = outcome?.text ?? '';
  };

  let lastNeed = null;
  function renderNeed() {
    const need = spec.need ? spec.need() : '';
    if (need === lastNeed) return;
    lastNeed = need;
    optionsBox.hidden = Boolean(need);
    startButton.hidden = Boolean(need);
    emptyStart.hidden = Boolean(need);
    needBox.hidden = !need;
    fill(
      needBox,
      need === 'yaml' ? h('p', { class: 'muted' }, "The recipe's YAML could not be read, so its datasets are not known. ", h('a', { href: '#/recipe' }, 'Fix the recipe'), '.') : null,
    );
  }

  function renderStale() {
    const stale = Boolean(shownJobId && spec.stale && spec.stale());
    if (staleBox.hidden === !stale) return;
    staleBox.hidden = !stale;
    mainStale.hidden = !stale;
  }

  let focusHeld = null;

  /** 'idle' | 'starting' | 'running' | 'cancelling' | 'loading' */
  function setPhase(phase, text = '') {
    const busy = phase !== 'idle';
    // A disabled button cannot keep the focus: the one that started the job gets it back when the job ends.
    if (busy && !focusHeld && (document.activeElement === startButton || document.activeElement === emptyStart)) focusHeld = document.activeElement;
    startButton.disabled = busy;
    emptyStart.disabled = busy;
    if (!busy && focusHeld) {
      const held = focusHeld;
      focusHeld = null;
      const lost = !document.activeElement || document.activeElement === document.body;
      if (lost && !document.querySelector('dialog[open]')) (held.isConnected && held.offsetParent !== null ? held : startButton).focus({ preventScroll: true });
    }
    cancelButton.hidden = !(phase === 'running' || phase === 'cancelling');
    cancelButton.disabled = phase === 'cancelling';
    if (statusLine.textContent !== text) statusLine.textContent = text;
    statusLine.hidden = !text;
    if (emptyText.textContent !== (busy && text ? text : spec.intro)) emptyText.textContent = busy && text ? text : spec.intro;
  }

  function setMessage(node) {
    fill(messageBox, node);
    messageBox.hidden = !node;
  }

  function showLink(jobId) {
    jobLink.hidden = !jobId;
    if (jobId) jobLink.setAttribute('href', analysisHref(jobId));
  }

  function clearResult() {
    cleanupResult?.();
    cleanupResult = null;
    shownJobId = null;
    fill(resultHost);
    emptyBox.hidden = false;
    setOutcome(null);
    renderStale();
  }

  function stopFollowing() {
    clearTimeout(timer);
    timer = 0;
    ctrl?.abort();
    ctrl = null;
  }

  const fresh = () => {
    stopFollowing();
    ctrl = new AbortController();
    return ctrl;
  };

  /** Record a change of the job (shared state for the audit, session memory otherwise), once per change. */
  function record(patch) {
    const before = spec.remembered();
    const next = { yaml: current?.yaml ?? '', jobId: current?.jobId ?? null, errors: null, warnings: null, error: null, ...patch };
    if (before && before.yaml === next.yaml && before.jobId === next.jobId && before.status === next.status && before.errors === next.errors && before.warnings === next.warnings && before.error === next.error) return;
    spec.record(next);
  }

  const retryButton = (run) => h('button', { class: 'btn btn--sm', type: 'button', onclick: () => run() }, 'Try again');

  function start() {
    if (startButton.disabled || recipe.get().phase !== 'open') return;
    // One request id per click: "Try again" after no answer repeats the same request, never a second job.
    const body = spec.body(recipe.get().yaml, newRequestId());
    if (!body) return;
    begin(body, recipeTask());
  }

  let startToken = null;
  async function begin(body, task) {
    stopFollowing();
    // The start request is not aborted when the screen closes: the job it creates is still recorded,
    // and the screen follows it again when it is opened again.
    const token = {};
    startToken = token;
    clearResult();
    setMessage(null);
    showLink(null);
    current = { jobId: null, yaml: body.yaml, task };
    record({ status: 'starting' });
    setPhase('starting', 'Starting…');
    live = true;
    spec.onActivity?.();
    spec.load().catch(() => {});
    try {
      const job = spec.exact?.(body) ? await api.analysisStartExact(body) : await api.analysisStart(body);
      if (token !== startToken) return;
      startToken = null;
      if (disposed) {
        const before = spec.remembered();
        if (before && before.status === 'starting' && before.yaml === body.yaml && job?.id) spec.record({ ...before, jobId: String(job.id), status: 'running' });
        return;
      }
      follow(job);
    } catch (error) {
      if (token !== startToken) return;
      startToken = null;
      const message = error?.offline ? OFFLINE : error?.message ?? String(error);
      if (disposed) {
        const before = spec.remembered();
        if (before && before.status === 'starting' && before.yaml === body.yaml) spec.record({ ...before, status: 'failed', error: message });
        return;
      }
      setPhase('idle');
      record({ status: 'failed', error: message });
      // No answer, or the service is busy with another job: the same request can simply be sent again.
      const again = error?.offline || error?.status === 409;
      setMessage(notice('err', error?.offline ? OFFLINE : `${spec.startFailed}: ${message}`, again ? retryButton(() => begin(body, task)) : null));
    }
  }

  /** A job answer (from start, a poll or cancel): show it and keep following while it runs. */
  function follow(job) {
    if (!job || typeof job !== 'object') return;
    if (job.id !== undefined && job.id !== null) current = { ...current, jobId: String(job.id) };
    showLink(current?.jobId);
    const status = job.status;
    if (status === 'completed') {
      loadResult(job);
      return;
    }
    if (status === 'failed') {
      setPhase('idle');
      const message = typeof job.error === 'string' && job.error ? job.error : 'The job failed; the service gave no reason.';
      record({ status: 'failed', error: message });
      setMessage(notice('err', message));
      return;
    }
    if (status === 'cancelled') {
      setPhase('idle');
      record({ status: 'cancelled' });
      setMessage(notice('', spec.cancelledText));
      return;
    }
    // running, cancelling (or a state this screen does not know yet): keep asking.
    live = true;
    record({ status: 'running' });
    setMessage(null);
    const text = status === 'cancelling' ? 'Cancelling…' : (part ? spec.progress(part, job.progress, current?.task) : '') || spec.runningText;
    setPhase(status === 'cancelling' ? 'cancelling' : 'running', text);
    clearTimeout(timer);
    timer = setTimeout(poll, JOB_POLL_MS);
  }

  spec.load().then(
    (mod) => {
      part = mod;
    },
    () => {},
  );

  async function poll() {
    timer = 0;
    const jobId = current?.jobId;
    if (!jobId || disposed) return;
    const c = fresh();
    try {
      const job = await api.analysisJob(jobId, { signal: c.signal });
      if (c !== ctrl || disposed) return;
      follow(job);
    } catch (error) {
      if (error?.name === 'AbortError' || c !== ctrl || disposed) return;
      setPhase('idle');
      // The job may still be running on the trainer machine; only the answer is missing.
      setMessage(notice('err', error?.offline ? OFFLINE : `The progress could not be read: ${error?.message ?? error}`, retryButton(() => poll())));
      startButton.disabled = true;
      emptyStart.disabled = true;
    }
  }

  async function cancel() {
    const jobId = current?.jobId;
    if (!jobId) return;
    setPhase('cancelling', 'Cancelling…');
    try {
      const job = await api.analysisCancel(jobId);
      if (disposed || current?.jobId !== jobId) return;
      if (job && typeof job === 'object' && job.status) follow(job);
    } catch (error) {
      if (disposed || current?.jobId !== jobId) return;
      setPhase('running', statusLine.textContent);
      toast(`The job could not be cancelled: ${error?.offline ? OFFLINE : error?.message ?? error}`, { kind: 'err' });
    }
  }

  /**
   * Read the result of a completed job. It is recorded as completed only once it was actually read: a
   * failed read is tried again a few times, then recorded as failed with "Try again" re-reading it.
   */
  async function loadResult(job, attempt = 1) {
    const c = fresh();
    setPhase('loading', 'Loading the result…');
    let answer;
    try {
      answer = await api.analysisResult(job.id, {}, { signal: c.signal });
    } catch (error) {
      if (error?.name === 'AbortError' || c !== ctrl || disposed) return;
      if (attempt < RESULT_ATTEMPTS) {
        ctrl = null;
        timer = setTimeout(() => loadResult(job, attempt + 1), JOB_POLL_MS * attempt);
        return;
      }
      ctrl = null;
      setPhase('idle');
      record({ status: 'failed', error: spec.resultUnreadable });
      setMessage(notice('err', `${spec.resultUnreadable} ${error?.offline ? OFFLINE : error?.message ?? error}`, retryButton(() => loadResult(job))));
      return;
    }
    if (c !== ctrl || disposed) return;
    let mod;
    try {
      mod = await spec.load();
    } catch (error) {
      if (c !== ctrl || disposed) return;
      ctrl = null;
      setPhase('idle');
      // The result was read: it is complete, only this screen cannot draw it.
      record({ status: 'completed', ...spec.counts(answer?.result) });
      setOutcome(spec.outcome(answer?.result));
      setMessage(notice('err', `${spec.partMissing}: ${error?.message ?? error}`));
      return;
    }
    if (c !== ctrl || disposed) return;
    ctrl = null;
    setPhase('idle');
    setMessage(null);
    record({ status: 'completed', ...spec.counts(answer?.result) });
    setOutcome(spec.outcome(answer?.result));
    cleanupResult?.();
    fill(resultHost);
    emptyBox.hidden = true;
    try {
      cleanupResult = spec.render(mod, resultHost, { job, result: answer?.result ?? null, artifacts: Array.isArray(answer?.artifacts) ? answer.artifacts : [] });
    } catch (error) {
      cleanupResult = null;
      setMessage(notice('err', `${spec.partMissing}: ${error?.message ?? error}`));
    }
    shownJobId = String(job.id);
    if (live) spec.onActivity?.();
    live = false;
    renderStale();
  }

  /**
   * Show the remembered job again (the screen was opened again). Only a 404 means the job is gone; any
   * other failure keeps it and asks again a few times, then offers "Try again".
   */
  async function restore(remembered, attempt = 1) {
    current = { jobId: remembered.jobId, yaml: remembered.yaml, task: recipeTask() };
    showLink(remembered.jobId);
    const c = fresh();
    setPhase('loading', remembered.status === 'running' ? spec.runningText : 'Loading the last result…');
    try {
      const job = await api.analysisJob(remembered.jobId, { signal: c.signal });
      if (c !== ctrl || disposed) return;
      follow(job);
    } catch (error) {
      if (error?.name === 'AbortError' || c !== ctrl || disposed) return;
      ctrl = null;
      if (error?.status === 404) {
        // The job is gone: forget it and start over.
        setPhase('idle');
        current = null;
        showLink(null);
        setMessage(null);
        spec.record(null);
        return;
      }
      if (attempt < RESULT_ATTEMPTS) {
        timer = setTimeout(() => restore(remembered, attempt + 1), JOB_POLL_MS * attempt);
        return;
      }
      setPhase('idle');
      setMessage(notice('err', error?.offline ? OFFLINE : `The job could not be read: ${error?.message ?? error}`, retryButton(() => restore(remembered))));
      // It may still be running on the trainer machine: no second start until it answers.
      startButton.disabled = true;
      emptyStart.disabled = true;
    }
  }
  // A start sent from an earlier visit of this screen that has not answered yet: wait for its job id.
  let awaitingStart = false;
  const remembered = spec.remembered();
  if (remembered?.jobId) restore(remembered);
  else if (remembered?.status === 'starting') {
    awaitingStart = true;
    current = { jobId: null, yaml: remembered.yaml, task: recipeTask() };
    setPhase('starting', 'Starting…');
  } else if (remembered && remembered.status === 'failed' && remembered.error) {
    current = { jobId: null, yaml: remembered.yaml, task: recipeTask() };
    setMessage(notice('err', remembered.error));
  }

  /** Called when the shared state changed (recipe or data). */
  function refresh() {
    const now = spec.remembered();
    if (awaitingStart && now?.status !== 'starting') {
      awaitingStart = false;
      if (now?.jobId) restore(now);
      else {
        setPhase('idle');
        if (now?.error) setMessage(notice('err', now.error));
      }
    }
    renderNeed();
    renderStale();
  }
  renderNeed();
  renderStale();

  return {
    controls,
    main,
    start,
    refresh,
    /** True once this runner follows or shows a job (or one is remembered). */
    active: () => Boolean(current || shownJobId),
    destroy() {
      disposed = true;
      stopFollowing();
      cleanupResult?.();
      cleanupResult = null;
    },
  };
}

/**
 * "Check this recipe's data": the service's data audit of the open recipe. Where it stands is kept in
 * state/data.js (`data.get().audit`), so the rail can say it and the screen shows it again later.
 * @param {HTMLElement} host left-pane controls
 * @param {HTMLElement} [mainHost] where the result goes (the main area); defaults to `host`
 * @param {{onActivity?: () => void}} [options]
 * @returns {{start: () => void, destroy: () => void}|(() => void)}
 */
export function mountRecipeAudit(host, mainHost = host, { onActivity } = {}) {
  const ids = { limit: uid('data-audit-limit'), every: uid('data-audit-every'), unit: uid('data-audit-unit'), error: uid('data-audit-error') };
  const limitError = h('p', { class: 'field__error', id: ids.error });
  const clearLimitError = () => {
    limitError.textContent = '';
    limitInput.removeAttribute('aria-invalid');
  };
  const limitInput = h('input', {
    class: 'input input--num data__auditlimit',
    id: ids.limit,
    type: 'number',
    min: '0',
    step: '1',
    inputmode: 'numeric',
    value: auditChoice.limit,
    disabled: auditChoice.every,
    'aria-describedby': `${ids.unit} ${ids.error}`,
    oninput: () => {
      auditChoice.limit = limitInput.value;
      clearLimitError();
    },
  });
  const everyBox = h('input', {
    type: 'checkbox',
    id: ids.every,
    checked: auditChoice.every,
    onchange: () => {
      auditChoice.every = everyBox.checked;
      limitInput.disabled = everyBox.checked;
      clearLimitError();
    },
  });
  const options = [
    h(
      'div',
      { class: 'data__auditoptions' },
      h('label', { for: ids.limit }, 'Open up to'),
      limitInput,
      h('span', { id: ids.unit }, 'files in each dataset'),
      h('label', { class: 'data__auditevery', for: ids.every }, everyBox, 'Every file (slower on large datasets)'),
    ),
    limitError,
  ];
  const runner = createJobRunner({
    buttonLabel: 'Check the data',
    intro: AUDIT_LINE,
    emptyTitle: 'Data check',
    options,
    staleText: "The recipe's data settings changed after this check.",
    startFailed: 'The check could not be started',
    cancelledText: 'The check was cancelled.',
    runningText: 'Checking the data…',
    partMissing: 'The data check result could not be shown',
    resultUnreadable: 'The check finished, but its result could not be read.',
    need: () => auditAvailability().need,
    stale: () => {
      const audit = data.get().audit;
      return Boolean(audit && audit.status === 'completed' && !data.auditIsCurrent());
    },
    remembered: () => data.get().audit,
    record: (next) => data.setAudit(next),
    load: loadTaskData,
    exact: (body) => body.auditLimit === null,
    body(yaml, requestId) {
      let limit = null;
      if (!everyBox.checked) {
        const raw = limitInput.value.trim();
        const n = Number(raw);
        if (!/^\d+$/.test(raw) || !Number.isSafeInteger(n)) {
          limitError.textContent = 'Enter a whole number of files, 0 or more, or choose every file.';
          limitInput.setAttribute('aria-invalid', 'true');
          limitInput.focus();
          return null;
        }
        limit = n;
      }
      return auditRequest(yaml, limit, requestId);
    },
    progress: (mod, progress, task) => (mod?.taskDataProgressText ? mod.taskDataProgressText(progress, task ?? undefined) : ''),
    render: (mod, el, input) => mod.renderTaskData(el, input),
    outcome: auditOutcome,
    counts: (result) => ({ errors: Number.isFinite(result?.totals?.error) ? result.totals.error : null, warnings: Number.isFinite(result?.totals?.warning) ? result.totals.warning : null }),
    onActivity,
  });
  fill(host, runner.controls);
  if (mainHost === host) host.appendChild(runner.main);
  else fill(mainHost, runner.main);
  const offs = [recipe.subscribe(runner.refresh), data.subscribe(runner.refresh)];
  const destroy = () => {
    runner.destroy();
    for (const off of offs) off();
  };
  // Called as a plain cleanup function, or used for its start().
  destroy.start = runner.start;
  destroy.active = runner.active;
  return destroy;
}

/**
 * "Look for identical files": the service's split-leakage job (paired restoration recipes with training and
 * validation data). Remembered for the session in this module.
 */
export function mountLeakage(host, mainHost, { onActivity } = {}) {
  const runner = createJobRunner({
    buttonLabel: 'Look for identical files',
    // A second look, after the folders themselves are checked: not the screen's main action.
    quiet: true,
    intro: LEAK_LINE,
    emptyTitle: 'Identical files',
    startFailed: 'The comparison could not be started',
    cancelledText: 'The comparison was cancelled.',
    runningText: 'Comparing the files…',
    partMissing: 'The identical-files result could not be shown',
    resultUnreadable: 'The comparison finished, but its result could not be read.',
    staleText: "The recipe's data settings changed after this check.",
    stale: leakJob.stale,
    remembered: leakJob.get,
    record: leakJob.record,
    load: loadHoldout,
    body: (yaml, requestId) => leakageRequest(yaml, requestId),
    progress: (mod, progress) => (mod?.splitLeakageProgressText ? mod.splitLeakageProgressText(progress) : ''),
    render: (mod, el, input) => {
      if (typeof mod.renderSplitLeakage !== 'function') throw new Error('The identical-files part has no renderSplitLeakage.');
      return mod.renderSplitLeakage(el, input);
    },
    outcome: leakageOutcome,
    counts: () => ({}),
    onActivity,
  });
  fill(host, runner.controls);
  fill(mainHost, runner.main);
  const offs = [recipe.subscribe(runner.refresh), leakJob.subscribe(runner.refresh)];
  const destroy = () => {
    runner.destroy();
    for (const off of offs) off();
  };
  destroy.start = runner.start;
  destroy.active = runner.active;
  return destroy;
}

/** True when a "Look for identical files" job was started in this session. */
export const leakageStarted = () => Boolean(leakJob.get());
/** True when a "Measure alignment" job was started in this session. */
export const registrationStarted = () => Boolean(regJob.get());

/**
 * The datasets a pair-registration job accepts: PairedImageDataset (any letter case), as the service checks.
 * @param {Array<{key: string, type: string, phase: string|null}>} sets recipe.datasetFolders()
 */
export function registrationSets(sets) {
  return (Array.isArray(sets) ? sets : []).filter((set) => String(set?.type ?? '').toLowerCase() === 'pairedimagedataset');
}

/** The request body of "Measure alignment"; the seed is left to the service. */
export function registrationRequest(yaml, split, sampleLimit, radius, requestId) {
  return { kind: 'pair-registration', yaml, split, sampleLimit, radius, requestId };
}

/** One line from a pair-registration result, only from its counts: no verdict, no threshold. */
export function registrationOutcome(result) {
  const sampled = result?.sampled;
  const population = result?.population;
  const unavailable = result?.counts?.unavailable;
  if (!Number.isFinite(sampled)) return { kind: '', text: 'The measurement finished.' };
  const head = Number.isFinite(population) ? `Sampled ${fmt.int(sampled)} of ${fmt.plural(population, 'pair')}.` : `Sampled ${fmt.plural(sampled, 'pair')}.`;
  const tail = Number.isFinite(unavailable) && unavailable > 0 ? ` ${fmt.int(unavailable)} could not be measured.` : '';
  return { kind: '', text: `${head}${tail}` };
}

/**
 * "Measure alignment": the service's pair-registration job on a seeded sample of one paired dataset's pairs
 * (paired restoration recipes). Remembered for the session in this module.
 */
export function mountRegistration(host, mainHost, { onActivity } = {}) {
  const ids = { split: uid('data-reg-split'), samples: uid('data-reg-samples'), samplesHelp: uid('data-reg-samples-help'), radius: uid('data-reg-radius'), radiusHelp: uid('data-reg-radius-help'), error: uid('data-reg-error') };
  const error = h('p', { class: 'field__error', id: ids.error });
  const clearError = (input) => {
    error.textContent = '';
    input.removeAttribute('aria-invalid');
  };
  const splitSelect = h('select', { class: 'select', id: ids.split, onchange: () => (regChoice.split = splitSelect.value) });
  const samplesInput = h('input', {
    class: 'input input--num data__regnum',
    id: ids.samples,
    type: 'number',
    min: String(REG_DEFAULTS.samplesMin),
    max: String(REG_DEFAULTS.samplesMax),
    step: '1',
    inputmode: 'numeric',
    value: regChoice.samples,
    'aria-describedby': `${ids.samplesHelp} ${ids.error}`,
    oninput: () => {
      regChoice.samples = samplesInput.value;
      clearError(samplesInput);
    },
  });
  const radiusInput = h('input', {
    class: 'input input--num data__regnum',
    id: ids.radius,
    type: 'number',
    min: String(REG_DEFAULTS.radiusMin),
    max: String(REG_DEFAULTS.radiusMax),
    step: '1',
    inputmode: 'numeric',
    value: regChoice.radius,
    'aria-describedby': `${ids.radiusHelp} ${ids.error}`,
    oninput: () => {
      regChoice.radius = radiusInput.value;
      clearError(radiusInput);
    },
  });
  const options = [
    h(
      'div',
      { class: 'data__regoptions' },
      h('label', { class: 'data__reglabel', for: ids.split }, 'Which data'),
      splitSelect,
      h('label', { class: 'data__reglabel', for: ids.samples }, 'Pairs to measure'),
      h('span', { class: 'data__regfield' }, samplesInput, h('span', { class: 'data__regunit', id: ids.samplesHelp }, `${REG_DEFAULTS.samplesMin} to ${REG_DEFAULTS.samplesMax}`)),
      h('label', { class: 'data__reglabel', for: ids.radius }, 'Search radius'),
      h('span', { class: 'data__regfield' }, radiusInput, h('span', { class: 'data__regunit', id: ids.radiusHelp }, `LQ pixels, ${REG_DEFAULTS.radiusMin} to ${REG_DEFAULTS.radiusMax}`)),
    ),
    error,
  ];

  /** Offer the recipe's paired datasets; rebuilt only when they change and never while the list has focus. */
  let lastSets = '';
  function syncSplits() {
    const sets = registrationSets(recipe.datasetFolders());
    const key = JSON.stringify(sets.map((set) => [set.key, set.phase]));
    if (key === lastSets || document.activeElement === splitSelect) return;
    lastSets = key;
    fill(splitSelect, sets.map((set) => h('option', { value: set.key }, `${PHASE_DATA[set.phase] ?? 'Data'} (${set.key})`)));
    const keys = sets.map((set) => set.key);
    const chosen = keys.includes(regChoice.split) ? regChoice.split : keys.includes('train') ? 'train' : keys[0] ?? '';
    splitSelect.value = chosen;
    regChoice.split = chosen || null;
  }
  syncSplits();

  const whole = (input, lo, hi) => {
    const raw = input.value.trim();
    const n = Number(raw);
    return /^\d+$/.test(raw) && n >= lo && n <= hi ? n : null;
  };
  const runner = createJobRunner({
    buttonLabel: 'Measure alignment',
    // Not the screen's main action.
    quiet: true,
    intro: REG_LINE,
    emptyTitle: 'Alignment sample',
    options,
    staleText: "The recipe's data settings changed after this check.",
    stale: regJob.stale,
    startFailed: 'The measurement could not be started',
    cancelledText: 'The measurement was cancelled.',
    runningText: 'Measuring the pairs…',
    partMissing: 'The alignment result could not be shown',
    resultUnreadable: 'The measurement finished, but its result could not be read.',
    remembered: regJob.get,
    record: regJob.record,
    load: loadRegistration,
    body(yaml, requestId) {
      const split = splitSelect.value;
      if (!split) {
        error.textContent = 'The recipe has no paired dataset to measure.';
        return null;
      }
      const samples = whole(samplesInput, REG_DEFAULTS.samplesMin, REG_DEFAULTS.samplesMax);
      if (samples === null) {
        error.textContent = `Enter a whole number of pairs from ${REG_DEFAULTS.samplesMin} to ${REG_DEFAULTS.samplesMax}.`;
        samplesInput.setAttribute('aria-invalid', 'true');
        samplesInput.focus();
        return null;
      }
      const radius = whole(radiusInput, REG_DEFAULTS.radiusMin, REG_DEFAULTS.radiusMax);
      if (radius === null) {
        error.textContent = `Enter a search radius from ${REG_DEFAULTS.radiusMin} to ${REG_DEFAULTS.radiusMax} LQ pixels.`;
        radiusInput.setAttribute('aria-invalid', 'true');
        radiusInput.focus();
        return null;
      }
      return registrationRequest(yaml, split, samples, radius, requestId);
    },
    progress: (mod, progress) => (mod?.pairRegistrationProgressText ? mod.pairRegistrationProgressText(progress) : ''),
    render: (mod, el, input) => {
      if (typeof mod.renderPairRegistration !== 'function') throw new Error('The alignment part has no renderPairRegistration.');
      return mod.renderPairRegistration(el, input);
    },
    outcome: registrationOutcome,
    counts: () => ({}),
    onActivity,
  });
  fill(host, runner.controls);
  fill(mainHost, runner.main);
  const offs = [
    recipe.subscribe(() => {
      syncSplits();
      runner.refresh();
    }),
    regJob.subscribe(runner.refresh),
  ];
  const destroy = () => {
    runner.destroy();
    for (const off of offs) off();
  };
  destroy.start = runner.start;
  destroy.active = runner.active;
  return destroy;
}

// ───────── view ─────────

export function mount(el) {
  const unsubs = [];
  const id = { gt: uid('data-gt'), lq: uid('data-lq'), scale: uid('data-scale'), gtHelp: uid('data-gt-help'), lqHelp: uid('data-lq-help'), scaleHelp: uid('data-scale-help'), target: uid('data-target') };
  const start = data.get();

  // ── folder form (built once; never rebuilt while the person types) ──
  const gtInput = h('input', { class: 'input input--path', id: id.gt, type: 'text', value: start.gt, autocomplete: 'off', spellcheck: 'false', 'aria-describedby': id.gtHelp, oninput: () => editFolder('gt', gtInput.value) });
  const lqInput = h('input', { class: 'input input--path', id: id.lq, type: 'text', value: start.lq, autocomplete: 'off', spellcheck: 'false', 'aria-describedby': id.lqHelp, oninput: () => editFolder('lq', lqInput.value) });
  const scaleSelect = h('select', {
    class: 'select',
    id: id.scale,
    'aria-describedby': id.scaleHelp,
    onchange: () => data.setInputs({ scale: scaleSelect.value === '' ? null : Number(scaleSelect.value) }),
  });
  const checkButton = h('button', { class: 'btn btn--primary', type: 'submit' });
  const rootHelp = h('p', { class: 'field__help' });

  // Labels, help and folder issues change with the task wording and the scan; the inputs never do.
  const gtLabel = h('label', { class: 'field__label', for: id.gt }, 'Ground truth folder');
  const lqLabel = h('label', { class: 'field__label', for: id.lq }, 'Low quality folder');
  const gtHelp = h('p', { class: 'field__help', id: id.gtHelp }, 'High-quality target images.');
  const lqHelp = h('p', { class: 'field__help', id: id.lqHelp }, 'Leave empty when the recipe makes low-quality images on the fly.');
  const gtIssues = h('div', { class: 'data__fieldissues', id: uid('data-gt-issues'), hidden: true });
  const lqIssues = h('div', { class: 'data__fieldissues', id: uid('data-lq-issues'), hidden: true });
  gtInput.setAttribute('aria-describedby', `${id.gtHelp} ${gtIssues.id}`);
  lqInput.setAttribute('aria-describedby', `${id.lqHelp} ${lqIssues.id}`);
  const taskNote = h('p', { class: 'data__tasknote', hidden: true });
  let unmounted = false;
  let picking = false;

  async function browse(role) {
    if (picking) return;
    picking = true;
    const input = role === 'gt' ? gtInput : lqInput;
    const words = currentWords().words;
    try {
      const { pickPath } = await import('../components/path-picker.js');
      const path = await pickPath({ title: `Choose the ${words[role].name.toLowerCase()} folder`, mode: 'folder', start: input.value.trim() });
      if (unmounted || !path) return;
      input.value = path;
      editFolder(role, path);
    } catch (error) {
      toast(`The folder picker could not be opened: ${error.message}`, { kind: 'err' });
    } finally {
      picking = false;
    }
  }
  // aria-label (set in renderWords) names the folder; the visible word stays "Browse".
  const gtBrowse = h('button', { class: 'btn', type: 'button', onclick: () => browse('gt') }, 'Browse');
  const lqBrowse = h('button', { class: 'btn', type: 'button', onclick: () => browse('lq') }, 'Browse');

  function syncScaleSelect() {
    const scale = data.get().scale;
    // A recipe may use a scale outside the usual list; offer it rather than silently dropping it.
    const values = [...SCALES];
    if (Number.isInteger(scale) && !values.includes(scale)) values.push(scale);
    values.sort((a, b) => a - b);
    fill(
      scaleSelect,
      h('option', { value: '' }, 'Do not check sizes'),
      values.map((n) => h('option', { value: String(n) }, `${n}×`)),
    );
    scaleSelect.value = Number.isInteger(scale) ? String(scale) : '';
  }
  syncScaleSelect();

  // A task whose dataset reads only dataroot_lq (folderWords(task).paired === false, e.g. classification):
  // no ground truth field, no scale, and no check, because the scan does not cover such folders.
  const gtField = h('div', { class: 'field' }, gtLabel, h('div', { class: 'data__pathrow' }, gtInput, gtBrowse), gtHelp, gtIssues);
  const scaleField = h(
    'div',
    { class: 'field' },
    h('label', { class: 'field__label', for: id.scale }, 'Scale to check'),
    scaleSelect,
    h('p', { class: 'field__help', id: id.scaleHelp }, 'Each ground truth image should be this many times the size of its low-quality partner.'),
  );
  const checkRow = h('div', null, checkButton);

  // What the check asks beyond the two folders (state/data.js options), and how to drop it.
  const optionsText = h('span', { class: 'data__optionstext' });
  const optionsLine = h(
    'div',
    { class: 'data__options', 'aria-live': 'polite', hidden: true },
    optionsText,
    h('button', { class: 'btn btn--sm', type: 'button', onclick: () => plainCheck() }, 'Plain folder check'),
  );
  const templateId = uid('data-template');
  const templateHelpId = uid('data-template-help');
  const phaseId = uid('data-phase');
  const templateInput = h('input', {
    class: 'input input--path',
    id: templateId,
    type: 'text',
    placeholder: '{}',
    autocomplete: 'off',
    spellcheck: 'false',
    'aria-describedby': templateHelpId,
    oninput: () => editOption('filenameTemplate', templateInput.value.trim() === '{}' ? '' : templateInput.value.trim()),
  });
  const phaseSelect = h(
    'select',
    { class: 'select', id: phaseId, onchange: () => editOption('phase', phaseSelect.value) },
    h('option', { value: '' }, 'Not set (the service checks as training)'),
    h('option', { value: 'train' }, 'Training'),
    h('option', { value: 'val' }, 'Validation'),
    h('option', { value: 'test' }, 'Test'),
  );
  const moreOptions = h(
    'details',
    { class: 'data__more' },
    h('summary', null, 'More check options'),
    h(
      'div',
      { class: 'field' },
      h('label', { class: 'field__label', for: templateId }, 'Low-quality name template'),
      templateInput,
      h('p', { class: 'field__help', id: templateHelpId }, 'Applied to the ground truth file name without its extension to give the low-quality name; {} means the same name on both sides. The extension may differ.'),
    ),
    h('div', { class: 'field' }, h('label', { class: 'field__label', for: phaseId }, 'Check as'), phaseSelect, h('p', { class: 'field__help' }, 'Validation allows a ground truth image larger than scale times its partner; the excess is cropped at the top-left.')),
  );
  /** Put the current options into the two hand fields, unless the person is typing in them. */
  function syncMoreOptions() {
    const o = data.get().options ?? {};
    if (document.activeElement !== templateInput) templateInput.value = o.filenameTemplate ?? '';
    if (document.activeElement !== phaseSelect) phaseSelect.value = o.phase ?? '';
  }
  let lastOptions = '';
  function renderOptions() {
    const o = data.get().options ?? {};
    const text = optionsSummary(o, optionsFrom?.key ?? null);
    const key = `${text}|${JSON.stringify(o)}`;
    if (key === lastOptions) return;
    lastOptions = key;
    optionsText.textContent = text;
    optionsLine.hidden = !text;
    syncMoreOptions();
  }
  const unpairedNote = h('div', { hidden: true });
  // Classification lists class subfolders; any other single-folder task (detection) lists image files.
  const listButton = h('button', { class: 'btn btn--primary', type: 'button', onclick: () => (setMode('listing'), unpairedTask() === CLASSIFICATION ? listClasses(lqInput.value.trim()) : listImages(lqInput.value.trim())) }, 'List class folders');
  const listRow = h('div', { hidden: true }, listButton);

  const form = h(
    'form',
    {
      class: 'data__form',
      novalidate: true,
      onsubmit: (event) => {
        event.preventDefault(); // Enter in a field lands here too
        checkFolders();
      },
    },
    gtField,
    h('div', { class: 'field' }, lqLabel, h('div', { class: 'data__pathrow' }, lqInput, lqBrowse), lqHelp, lqIssues),
    taskNote,
    unpairedNote,
    optionsLine,
    moreOptions,
    scaleField,
    checkRow,
    listRow,
  );

  const recipeBox = h('div', { class: 'data__sets' });
  const writeBox = h('div', { class: 'data__write' });
  const overlapBox = h('div', { class: 'data__write', hidden: true });
  // Paired restoration recipes: alignment measured on a sample of the recipe's own pairs.
  const regBox = h('div', { class: 'data__write', hidden: true });
  const splitBox = h('div', { class: 'data__write', hidden: true });
  const patchBox = h('div', { class: 'data__write', hidden: true });
  const resultBox = h('div', { class: 'data__result', role: 'status' });

  // Main-area panels for the two jobs; the left pane holds only their controls and a short status.
  const auditMain = h('div', { class: 'data__jobmain', hidden: true });
  const leakMain = h('div', { class: 'data__jobmain', hidden: true });
  const regMain = h('div', { class: 'data__jobmain', hidden: true });

  // Classification, segmentation and detection recipes: the service lists and opens the files itself.
  const auditHost = h('div', { class: 'data__audit' });
  const auditSection = h('section', { class: 'data__section', 'aria-label': "Check this recipe's data", hidden: true }, h('h3', { class: 'data__subheading' }, "Check this recipe's data"), auditHost);
  const unmountAudit = mountRecipeAudit(auditHost, auditMain, { onActivity: () => setMode('audit') });
  function renderAuditSection() {
    const show = recipe.get().phase === 'open' && auditAvailability().show;
    if (auditSection.hidden === !show) return;
    auditSection.hidden = !show;
  }

  // The open recipe's datasets lead the pane; checking any two folders is the secondary tool below.
  const recipeTitle = h('h2', { class: 'data__heading' });
  const leadLine = h('p', { class: 'data__lead', hidden: true }, "These are the template's example paths. Choose your own with Change…");
  const recipeSection = h('section', { class: 'data__recipe', 'aria-label': 'Data of the open recipe', hidden: true }, recipeTitle, leadLine, recipeBox, auditSection, overlapBox, splitBox, patchBox, regBox);
  const noRecipeLine = h(
    'p',
    { class: 'data__norecipe', hidden: true },
    'No recipe is open. A recipe decides what data is needed: ',
    h('a', { href: '#/recipe' }, 'choose one first'),
    ', or check any two folders below.',
  );
  const otherDetails = h(
    'details',
    { class: 'data__other' },
    h('summary', { class: 'data__othersummary' }, 'Check other folders'),
    h('div', { class: 'data__otherbody' }, form, writeBox, h('section', { class: 'data__section', 'aria-label': 'Check result' }, h('h3', { class: 'data__subheading' }, 'Check result'), resultBox)),
  );

  const folders = h(
    'aside',
    { class: 'pane data__folders', 'aria-label': 'Data' },
    h('div', { class: 'pane__body pane__body--pad data__scroll' }, h('div', { class: 'stack' }, h('h1', null, 'Data'), rootHelp), noRecipeLine, recipeSection, otherDetails),
  );

  // ── stage side ──
  const sampleName = h('span', { class: 'data__samplename' });
  const samplePos = h('span', { class: 'data__samplepos' });
  const sampleNote = h('span', { class: 'data__samplenote', hidden: true });
  const stageHead = h('div', { class: 'stage data__stagehead' }, sampleName, sampleNote, samplePos);
  const compareHost = h('div', { class: 'data__compare' });
  const countEl = h('p', { class: 'data__count' });
  const strip = h('div', { class: 'data__strip', role: 'listbox', 'aria-orientation': 'horizontal', 'aria-label': 'Checked images' });
  const sheetEmpty = h('p', { class: 'data__sheetempty' });
  const rootSelectId = uid('data-root');
  const rootSelect = h('select', {
    class: 'select data__rootselect',
    id: rootSelectId,
    onchange: () => {
      rootFilter = rootSelect.value;
      buildSheet(shown().result);
    },
  });
  const rootSelectWrap = h('span', { class: 'data__rootfilter', hidden: true }, h('label', { for: rootSelectId }, 'Show pairs from'), rootSelect);
  const sheet = h('div', { class: 'stage data__sheet' }, h('div', { class: 'data__sheethead' }, countEl, rootSelectWrap), strip, sheetEmpty);
  // ── alignment check for the selected pair (only when asked) ──
  const radiusId = uid('data-radius');
  const radiusHelpId = uid('data-radius-help');
  const radiusInput = h('input', { class: 'input input--num data__radius', id: radiusId, type: 'number', min: '2', max: '12', step: '1', value: '6', inputmode: 'numeric', 'aria-describedby': radiusHelpId });
  const alignButton = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => runAlignment() }, 'Check alignment');
  // One line only: the stage keeps its room. The full answer opens in a dialog from "Details".
  const alignSummary = h('p', { class: 'data__alignsummary', 'aria-live': 'polite' });
  const alignDetails = h('button', { class: 'btn btn--sm', type: 'button', hidden: true, onclick: () => openAlignmentDetails() }, 'Details');
  const alignBar = h(
    'section',
    { class: 'data__align', 'aria-label': 'Alignment check', hidden: true },
    h('label', { class: 'data__alignlabel', for: radiusId }, 'Search radius'),
    radiusInput,
    h('span', { class: 'data__alignunit', id: radiusHelpId }, 'LQ pixels, 2 to 12'),
    alignButton,
    alignSummary,
    alignDetails,
  );
  // Pairs (stage, alignment, contact sheet) or, for a single-folder task, the class folders.
  const pairArea = h('div', { class: 'data__pairs' }, stageHead, compareHost, alignBar, sheet);
  const classesHost = h('section', { class: 'data__classes', 'aria-label': 'Class folders', hidden: true });
  // A two- or three-item switch when the main area has more than one thing to show.
  const switchBox = h('div', { class: 'seg', role: 'group', 'aria-label': 'Show in this area' });
  const switchRow = h('div', { class: 'data__switchrow', hidden: true }, switchBox);
  const main = h('div', { class: 'data__main' }, switchRow, pairArea, classesHost, auditMain, leakMain, regMain);

  el.appendChild(h('div', { class: 'data' }, folders, main));

  // ── main area: what it shows ──
  let mode = null;
  let lastItems = '';
  function mainItems() {
    const r = recipe.get();
    if (r.phase !== 'open') return [];
    const task = r.summary?.task ?? r.config?.task ?? null;
    if (AUDIT_TASKS.includes(task)) return [['audit', 'Data check'], ['listing', task === CLASSIFICATION ? 'Class folders' : task === DETECTION ? 'Images' : 'Folder check']];
    const extra = [...(leakageStarted() ? [['leak', 'Identical files']] : []), ...(registrationStarted() ? [['reg', 'Alignment sample']] : [])];
    if (extra.length && !recipeUnpaired()) return [['listing', 'Folder check'], ...extra];
    return [];
  }
  function renderMain() {
    const items = mainItems();
    const ids = items.map(([id]) => id);
    if (!ids.length) mode = 'listing';
    else if (!ids.includes(mode)) mode = ids[0];
    const itemsKey = JSON.stringify(items);
    if (itemsKey !== lastItems) {
      lastItems = itemsKey;
      switchRow.hidden = !items.length;
      fill(switchBox, items.map(([id, label]) => h('button', { class: 'btn btn--sm', type: 'button', dataset: { mode: id }, onclick: () => setMode(id) }, label)));
    }
    for (const button of switchBox.children) button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
    const unpaired = recipeUnpaired();
    pairArea.hidden = !(mode === 'listing' && !unpaired);
    classesHost.hidden = !(mode === 'listing' && unpaired);
    auditMain.hidden = mode !== 'audit';
    leakMain.hidden = mode !== 'leak';
    regMain.hidden = mode !== 'reg';
  }
  function setMode(next) {
    mode = next;
    renderMain();
  }

  let compare = null;
  let compareText = '';
  function ensureCompare(text) {
    // emptyText is fixed per instance, so a different empty sentence means a new instance.
    if (compare && compareText === text) return;
    compare?.destroy();
    compare = createCompare(compareHost, { emptyText: text });
    compareText = text;
  }
  // The first renderSheet() (in renderAll below) always builds the sheet and creates the stage.

  // ── regions ──
  let lastRoot;
  function renderRoot() {
    const root = conn.get().info?.trainerRoot;
    if (root === lastRoot) return;
    lastRoot = root;
    fill(
      rootHelp,
      typeof root === 'string' && root
        ? ['Paths are on the trainer machine: absolute, or relative to ', h('span', { class: 'mono' }, root), '.']
        : 'Paths are on the trainer machine: absolute, or relative to the trainer folder.',
    );
  }

  /** Task words for a pair of folders: the open recipe's task when these are its folders, else the defaults. */
  function wordsFor(gt, lq) {
    const set = recipeSetFor(recipe.datasetFolders(), gt, lq);
    const task = set ? (recipe.get().summary?.task ?? null) : null;
    return { fromRecipe: Boolean(set), task, words: folderWords(task) };
  }
  function currentWords() {
    const d = data.get();
    return wordsFor(d.gt, d.lq);
  }
  const DEFAULT_WORDS = folderWords(null);
  const isCustom = (words) => words.gt.name !== DEFAULT_WORDS.gt.name || words.lq.name !== DEFAULT_WORDS.lq.name;

  /** True when the open recipe's task reads only dataroot_lq (no pairs to check). */
  function recipeUnpaired() {
    const task = recipe.get().summary?.task;
    return Boolean(task) && folderWords(task).paired === false;
  }
  /** The open recipe's task id when it is a single-folder task, else null. */
  const unpairedTask = () => (recipeUnpaired() ? recipe.get().summary.task : null);

  let lastWords = '';
  let wasUnpaired = false;
  function renderWords() {
    const unpaired = recipeUnpaired();
    const { task, words } = unpaired ? { task: recipe.get().summary?.task, words: folderWords(recipe.get().summary?.task) } : currentWords();
    const key = JSON.stringify([task, words.gt.name, words.lq.name, unpaired]);
    if (key === lastWords) return;
    lastWords = key;
    gtField.hidden = unpaired;
    moreOptions.hidden = unpaired;
    scaleField.hidden = unpaired;
    checkRow.hidden = unpaired;
    // Classification and detection are checked by "Check this recipe's data"; only a single-folder task
    // this screen has no audit for keeps the sentence that nothing checks it.
    const noAudit = unpaired && !AUDIT_TASKS.includes(task);
    unpairedNote.hidden = !noAudit;
    fill(unpairedNote, noAudit ? notice('', unpairedLine(task)) : null);
    fill(listButton, task === CLASSIFICATION ? 'List class folders' : 'List images');
    listRow.hidden = !unpaired;
    renderMain();
    const unpairedId = unpaired ? task : false;
    if (unpairedId !== wasUnpaired) {
      const entering = unpaired && !wasUnpaired;
      wasUnpaired = unpairedId;
      if (unpaired && !entering) renderClassesIdle();
      else if (unpaired) {
        // A folder remembered from another recipe is not shown as this recipe's class folder; the stored
        // value stays until the person acts.
        const own = recipe.datasetFolders().flatMap((set) => set.lq ?? []);
        if (!own.includes(data.get().lq.trim())) lqInput.value = '';
        renderClassesIdle();
      } else {
        lqInput.value = data.get().lq;
        resetClasses();
      }
    }
    const custom = isCustom(words);
    gtLabel.textContent = folderLabel(words.gt.name);
    lqLabel.textContent = folderLabel(words.lq.name);
    gtHelp.textContent = custom ? `${capitalize(words.gt.what)}.` : 'High-quality target images.';
    lqHelp.textContent = custom ? `${capitalize(words.lq.what)}.` : 'Leave empty when the recipe makes low-quality images on the fly.';
    gtBrowse.setAttribute('aria-label', `Browse for the ${words.gt.name.toLowerCase()} folder`);
    lqBrowse.setAttribute('aria-label', `Browse for the ${words.lq.name.toLowerCase()} folder`);
    taskNote.hidden = task !== SEGMENTATION;
    taskNote.textContent = task === SEGMENTATION ? MASK_LINE : '';
  }

  let lastFieldIssues = '';
  function renderFieldIssues() {
    const d = shown();
    // Only while the fields still hold the folders that were checked.
    const found = d.result && d.current ? folderIssues(d.result, d.scanned) : { gt: [], lq: [] };
    const key = JSON.stringify(found);
    if (key === lastFieldIssues) return;
    lastFieldIssues = key;
    const line = (issue) => {
      const isError = issue.level === 'error';
      return h('p', { class: `data__fieldissue${isError ? ' data__fieldissue--err' : ''}` }, h('span', { class: 'data__fieldlevel' }, isError ? 'Error' : 'Warning'), ' ', String(issue.message ?? ''));
    };
    for (const [box, input, list] of [
      [gtIssues, gtInput, found.gt],
      [lqIssues, lqInput, found.lq],
    ]) {
      fill(box, list.map(line));
      box.hidden = !list.length;
      if (list.some((issue) => issue.level === 'error')) input.setAttribute('aria-invalid', 'true');
      else input.removeAttribute('aria-invalid');
    }
  }

  /** What the screen shows: the shared data state (every check, plain or with options, goes through it). */
  function shown() {
    const d = data.get();
    return { phase: d.phase, result: d.result, error: d.error, scanned: d.scanned, current: data.isCurrent() };
  }

  // ── check options (state/data.js `options`) ──
  /** {key, gt, lq} of the recipe dataset the current options were built from, or null (none, or set by hand). */
  let optionsFrom = null;

  /** Folder field edited by hand: options built from a recipe dataset no longer describe these folders. */
  function editFolder(role, value) {
    const patch = { [role]: value };
    if (optionsFrom) {
      const next = { gt: data.get().gt, lq: data.get().lq, ...patch };
      if (next.gt.trim() !== optionsFrom.gt || next.lq.trim() !== optionsFrom.lq) {
        optionsFrom = null;
        patch.options = {};
      }
    }
    data.setInputs(patch);
    if (patch.options) syncMoreOptions();
  }

  /** Hand-set template or phase: keep the other options, but they no longer come from a recipe dataset as is. */
  function editOption(key, value) {
    const next = { ...data.get().options };
    if (value === '' || value === null || value === undefined) delete next[key];
    else next[key] = value;
    optionsFrom = null;
    data.setInputs({ options: next });
  }

  function plainCheck() {
    optionsFrom = null;
    data.setInputs({ options: {} });
    syncMoreOptions();
    setMode('listing');
    if (data.get().gt.trim()) data.scan();
  }

  let lastButton = '';
  function renderButton() {
    const d = data.get();
    const scanning = d.phase === 'scanning';
    const ready = Boolean(d.gt.trim() || d.options?.gtRoots?.length);
    const key = `${scanning}:${ready}`;
    if (key === lastButton) return;
    lastButton = key;
    checkButton.disabled = scanning || !ready;
    fill(checkButton, scanning ? [h('span', { class: 'spinner data__spin', 'aria-hidden': 'true' }), 'Checking…'] : 'Check folders');
  }

  function useFolders(set) {
    const patch = { gt: set.gt[0] ?? '', lq: set.lq[0] ?? '' };
    const scale = recipe.get().config?.scale;
    if (Number.isInteger(scale)) patch.scale = scale;
    if (recipeUnpaired()) {
      // Only the class folder is used; nothing is checked (the scan does not cover it). The classes are listed.
      optionsFrom = null;
      data.setInputs({ gt: '', lq: set.lq[0] ?? '', options: {} });
      gtInput.value = '';
      lqInput.value = set.lq[0] ?? '';
      syncMoreOptions();
      annotationPath = set.annotationFile ?? null;
      if (set.lq[0]) {
        if (unpairedTask() === CLASSIFICATION) listClasses(set.lq[0]);
        else listImages(set.lq[0]);
      }
      return;
    }
    // Ask the question the trainer will: the dataset's own roots, template, phase, crop and storage.
    patch.options = optionsForSet(set, recipe.get().summary?.task ?? null);
    optionsFrom = { key: set.key, gt: patch.gt.trim(), lq: patch.lq.trim() };
    data.setInputs(patch);
    gtInput.value = patch.gt;
    lqInput.value = patch.lq;
    syncScaleSelect();
    syncMoreOptions();
    if (patch.gt.trim() || patch.options.gtRoots) data.scan();
  }

  /** "Check folders": the folders in the fields with the options in effect. */
  function checkFolders() {
    const d = data.get();
    if (d.phase === 'scanning' || recipeUnpaired()) return;
    setMode('listing');
    if (d.gt.trim() || d.options?.gtRoots?.length) data.scan();
  }

  // ── the open recipe's datasets: places on disk, Change…, and the check of each entry ──
  /** Refused edits, by dataset key: the service's message, shown in the entry. */
  const entryErrors = new Map();
  let changing = false;

  /** The places a dataset reads, in the task's words. */
  function placesFor(set, task, words) {
    // A table of records: the one place is the CSV file itself.
    if (words.records && words.file) return [{ option: words.file.key, label: words.file.name, paths: set.metaInfo ? [set.metaInfo] : [], mode: 'file', extensions: words.file.extensions ?? ['.csv'] }];
    const gtPlace = { option: 'dataroot_gt', label: folderLabel(words.gt.name), paths: set.gt, mode: 'folder' };
    const lqPlace = { option: 'dataroot_lq', label: folderLabel(words.lq.name), paths: set.lq, mode: 'folder' };
    if (words.paired === false) {
      const places = [lqPlace];
      if (words.file) places.push({ option: words.file.key, label: words.file.name, paths: set.annotationFile ? [set.annotationFile] : [], mode: 'file', extensions: ['.json'] });
      return places;
    }
    // Only the on-the-fly dataset makes its low-quality images itself. Every other class that reaches here
    // reads dataroot_lq (PairedImageDataset, RealESRGANPairedDataset, PairedVideoDataset, ImageMaskDataset):
    // without one it is "Not set", with Choose… to set it again.
    if (!set.lq.length && datasetKind(set.type) === OTF_DATASET) lqPlace.onTheFly = true;
    return task === SEGMENTATION ? [lqPlace, gtPlace] : [gtPlace, lqPlace];
  }

  /** One line about the folder check of the dataset loaded in it, while that result is the current one. */
  function loadedOutcome() {
    if (!optionsFrom?.key || recipeUnpaired()) return null;
    const d = shown();
    if (!d.current) return null;
    const key = optionsFrom.key;
    if (d.phase === 'scanning') return { key, kind: '', text: 'Checking…' };
    if (d.phase === 'error') return { key, kind: 'err', text: `Could not be checked: ${d.error}` };
    if (!d.result) return null;
    const found = folderIssues(d.result, d.scanned);
    const folderError = [...found.gt, ...found.lq].find((issue) => issue.level === 'error');
    if (folderError) {
      const missing = /not (found|exist)|no such|does not exist|missing/i.test(String(folderError.message ?? ''));
      return { key, kind: 'err', text: missing ? 'Folder not found' : String(folderError.message ?? 'Error'), missing };
    }
    const { errors, warnings } = countIssues(d.result.issues);
    if (errors || warnings) return { key, kind: errors ? 'err' : 'warn', text: [errors ? fmt.plural(errors, 'error') : '', warnings ? fmt.plural(warnings, 'warning') : ''].filter(Boolean).join(', ') };
    const paired = d.result.counts?.paired;
    const word = folderWords(recipe.get().summary?.task ?? null).pair;
    return { key, kind: 'ok', text: Number.isFinite(paired) ? `${fmt.plural(paired, word, `${word}s`)}, no problems` : 'No problems' };
  }

  async function changePlace(set, place) {
    if (changing) return;
    changing = true;
    let picked = null;
    try {
      const { pickPath } = await import('../components/path-picker.js');
      picked = await pickPath({
        title: `Choose the ${place.label.toLowerCase()} of the ${dataWords(set)}`,
        mode: place.mode,
        start: place.paths[0] ?? '',
        ...(place.mode === 'file' ? { extensions: place.extensions, confirmLabel: 'Use this file' } : {}),
      });
    } catch (error) {
      toast(`The ${place.mode === 'file' ? 'file' : 'folder'} picker could not be opened: ${error?.message ?? error}`, { kind: 'err' });
    } finally {
      changing = false;
    }
    if (!picked || unmounted) return;
    const edit = placeEdit(recipe.get().config, set.key, place.option, picked, place.mode);
    entryErrors.delete(set.key);
    try {
      await recipe.applyEdits([edit]);
      toast([`Set the ${dataWords(set)}'s ${place.label.toLowerCase()} in the recipe.`, ' ', h('span', { class: 'mono' }, picked)], { kind: 'ok' });
    } catch (error) {
      entryErrors.set(set.key, `The recipe was not changed: ${error?.message ?? error}`);
    }
    lastRecipe = '';
    renderRecipe();
  }

  function placeRow(set, place, missing) {
    const mark = (index) => {
      const message = missing.get(`${set.key}|${place.option}|${index}`);
      if (!message) return null;
      return badge(message.startsWith('Path was not found') ? 'Not found on the trainer machine' : message, 'err');
    };
    const [first, ...more] = place.paths;
    const value = first
      ? [
          h('span', { class: 'mono data__placepath' }, first),
          mark(0),
          more.length
            ? h(
                'details',
                { class: 'data__placemore' },
                h('summary', null, `and ${fmt.int(more.length)} more`),
                more.map((path, i) => h('div', { class: 'data__placeextra' }, h('span', { class: 'mono data__placepath' }, path), mark(i + 1))),
              )
            : null,
        ]
      : h('span', { class: 'muted' }, place.onTheFly ? 'Made on the fly' : 'Not set');
    return h(
      'div',
      { class: 'data__place' },
      h('span', { class: 'data__placelabel' }, place.label),
      h('div', { class: 'data__placevalue' }, value),
      place.onTheFly
        ? null
        : h(
            'button',
            {
              class: 'btn btn--sm data__placechange',
              type: 'button',
              dataset: { focus: `${set.key}|${place.option}` },
              'aria-label': `${first ? 'Change' : 'Choose'} the ${place.label.toLowerCase()} of the ${dataWords(set)}`,
              onclick: () => changePlace(set, place),
            },
            first ? 'Change…' : 'Choose…',
          ),
    );
  }

  let lastRecipe = '';
  let wasOpen;
  function renderRecipe() {
    const r = recipe.get();
    const open = r.phase === 'open';
    noRecipeLine.hidden = open;
    recipeSection.hidden = !open;
    if (open !== wasOpen) {
      // Closed by default while a recipe is open; the tool itself when none is.
      otherDetails.open = !open;
      wasOpen = open;
    }
    const sets = recipe.datasetFolders();
    const task = r.summary?.task ?? null;
    const missing = placeIssues(r.issues);
    const outcome = loadedOutcome();
    const key = JSON.stringify([r.phase, r.name, Boolean(r.config), r.check, r.checkError, sets, task, [...missing], outcome, [...entryErrors]]);
    if (key === lastRecipe) return;
    lastRecipe = key;
    // The same name the rail shows for the recipe: its own `name` option, then the name it was opened under.
    const recipeName = r.summary?.name || r.name;
    recipeTitle.textContent = recipeName ? `Data for ${recipeName}` : 'Data for the open recipe';
    const words = folderWords(task);
    // Checking two image folders has nothing to say about a table of records.
    otherDetails.hidden = Boolean(open && words.records);
    const focused = recipeBox.contains(document.activeElement) ? document.activeElement?.dataset?.focus ?? null : null;
    let content;
    let lead = false;
    if (!open) content = null;
    else if (!r.config && r.check === 'failed') content = h('p', { class: 'muted' }, `The recipe could not be checked, so its folders are not known: ${r.checkError}`);
    else if (!r.config) content = h('p', { class: 'muted' }, 'Reading the folders from the recipe…');
    else if (!sets.length) content = h('p', { class: 'muted' }, 'The open recipe names no datasets.');
    else {
      const all = [];
      content = sets.map((set) => {
        // Video clips are not image folders: no image rows, no image check; the recipe prepares them.
        if (String(set.type).toLowerCase() === VIDEO_DATASET) return videoEntry(set);
        const places = placesFor(set, task, words);
        for (const place of places) if (place.paths.length) all.push(missing.has(`${set.key}|${place.option}|0`));
        const here = outcome?.key === set.key;
        const several = places.some((place) => place.paths.length > 1);
        const unpaired = words.paired === false;
        const error = entryErrors.get(set.key);
        return h(
          'div',
          { class: here ? 'data__entry data__entry--shown' : 'data__entry' },
          h(
            'div',
            { class: 'data__entryhead' },
            h('span', { class: 'data__entryrole' }, PHASE_DATA[set.phase] ?? 'Data'),
            h('span', { class: 'mono data__entrykey' }, set.key),
            set.name && set.name !== set.key ? h('span', { class: 'data__entryname' }, set.name) : null,
            here ? badge('Shown in the folder check') : null,
          ),
          places.map((place) => placeRow(set, place, missing)),
          several ? h('p', { class: 'data__entryhint' }, 'Change… edits the first folder; more folders are edited in the ', h('a', { href: '#/recipe' }, "recipe's Datasets section"), '.') : null,
          error ? notice('err', error) : null,
          h(
            'div',
            { class: 'data__entryactions' },
            words.records
              ? h('p', { class: 'data__entryhint' }, 'Each row is one record. Its columns are named in the ', h('a', { href: '#/recipe' }, "recipe's Basics section"), ', under the task options: features and targets.')
              : unpaired
              ? h(
                  'button',
                  { class: 'btn btn--sm', type: 'button', dataset: { focus: `${set.key}|list` }, disabled: !set.lq.length, onclick: () => (setMode('listing'), useFolders(set)) },
                  task === CLASSIFICATION ? 'List class folders' : 'List images',
                )
              : h(
                  'button',
                  {
                    // The training folders are what a first training needs checked: the screen's main action.
                    class: `btn btn--sm${set.phase === 'train' ? ' btn--primary' : ''}`,
                    type: 'button',
                    dataset: { focus: `${set.key}|check` },
                    disabled: !set.gt.length || data.get().phase === 'scanning',
                    title: set.gt.length > 1 || set.lq.length > 1 ? `Checks all ${fmt.int(Math.max(set.gt.length, set.lq.length))} folder pairs together` : null,
                    onclick: () => (setMode('listing'), useFolders(set)),
                  },
                  'Check these folders',
                ),
            here && outcome ? h('span', { class: `data__entryoutcome${outcome.kind ? ` data__entryoutcome--${outcome.kind}` : ''}`, role: 'status' }, outcome.text) : null,
          ),
        );
      });
      lead = (all.length > 0 && all.every(Boolean)) || outcome?.missing === true;
    }
    fill(recipeBox, content);
    leadLine.hidden = !lead;
    if (focused) [...recipeBox.querySelectorAll('button')].find((button) => button.dataset.focus === focused)?.focus();
  }

  // ── a video clip dataset: its videos, one line from the recipe's video check, and the way to it ──
  // The line is one node kept across rebuilds of the entries, so it follows the check by itself.
  const videoLine = h('p', { class: 'data__entryhint data__videoline', role: 'status' });
  // Also one node: its words follow the same state ("Show progress", "Prepare in the recipe", "Show in the recipe").
  const videoButton = h('button', { class: 'btn btn--sm btn--primary', type: 'button', onclick: () => revealVideoPanel() }, 'Show in the recipe');
  function renderVideoLine() {
    const v = videoPrep.get();
    const c = v.check.phase === 'done' && v.check.answer ? videoCounts(v.check.answer) : null;
    const label = jobActive(v.job) ? 'Show progress' : c && c.fixable > 0 && !c.invalid ? 'Prepare in the recipe' : 'Show in the recipe';
    if (videoButton.textContent !== label) videoButton.textContent = label;
    const text = jobActive(v.job)
      ? 'Preparing the videos…'
      : exampleUnchosen(v)
        ? "This is the template's example path. Choose the video folder in the recipe."
      : v.check.phase === 'checking'
        ? 'Checking the videos…'
        : v.check.phase === 'error'
          ? v.check.error || 'The videos could not be checked.'
          : !c
            ? 'Videos not checked.'
            : c.invalid
              ? `${fmt.int(c.invalid)} of ${fmt.plural(c.total, 'video')} cannot be used for clip training.`
              : c.pending
                ? `${fmt.int(c.pending)} of ${fmt.plural(c.total, 'video')} need an index.`
                : `${fmt.plural(c.ready, 'video')} ready.`;
    if (videoLine.textContent !== text) videoLine.textContent = text;
  }
  function videoEntry(set) {
    return h(
      'div',
      { class: 'data__entry' },
      h('div', { class: 'data__entryhead' }, h('span', { class: 'data__entryrole' }, 'Training videos'), h('span', { class: 'mono data__entrykey' }, set.key)),
      set.gt.length ? set.gt.map((p) => h('p', { class: 'mono data__videopath' }, p)) : h('p', { class: 'muted' }, 'No folder chosen'),
      videoLine,
      h('div', { class: 'data__entryactions' }, videoButton),
    );
  }

  let lastWrite = '';
  let targetSelect = null;
  let writeButton = null;
  let writing = false;
  function renderWrite() {
    const r = recipe.get();
    const keys = r.phase === 'open' && r.configCurrent ? recipe.datasetFolders().map((set) => set.key) : [];
    const key = JSON.stringify(keys);
    if (key !== lastWrite) {
      lastWrite = key;
      const keep = targetSelect?.value;
      if (!keys.length) {
        targetSelect = null;
        writeButton = null;
        fill(writeBox);
      } else {
        targetSelect = h('select', { class: 'select', id: id.target }, keys.map((k) => h('option', { value: k }, k)));
        if (keep && keys.includes(keep)) targetSelect.value = keep;
        writeButton = h('button', { class: 'btn btn--sm', type: 'button', onclick: writeFolders }, 'Set folders in recipe');
        fill(
          writeBox,
          h('h3', { class: 'data__subheading' }, 'Write these folders into the recipe'),
          h('div', { class: 'field' }, h('label', { class: 'field__label', for: id.target }, 'Dataset in the recipe'), h('div', { class: 'row' }, targetSelect, writeButton)),
        );
      }
    }
    if (writeButton) writeButton.disabled = writing || !(recipeUnpaired() ? lqInput.value.trim() : data.get().gt.trim());
  }

  // ── class folders (single-folder tasks): what fs/list shows, nothing checked ──
  let classCtrl = null;
  let classObserver = null;
  let classGen = 0;
  const CLASS_LIMIT = 1024;
  const CLASS_THUMBS = 6;

  function resetClasses() {
    classGen += 1;
    classCtrl?.abort();
    classCtrl = null;
    classObserver?.disconnect();
    classObserver = null;
  }

  /** Heading of the right-hand side for the single-folder task: "Class folders" or "Images". */
  const unpairedHeading = () => folderWords(unpairedTask()).lq.name;

  function renderClassesIdle() {
    resetClasses();
    const task = unpairedTask();
    classesHost.setAttribute('aria-label', unpairedHeading());
    const set = recipe.datasetFolders().find((s) => s.annotationFile) ?? null;
    fill(
      classesHost,
      h('div', { class: 'data__classhead' }, h('h2', { class: 'data__heading' }, folderWords(task).records ? 'Records' : unpairedHeading())),
      h(
        'p',
        { class: 'data__classempty' },
        folderWords(task).records
          ? 'This recipe trains on tables of numbers: there are no images to list. Open a CSV file in a spreadsheet or a text editor to look at its rows.'
          : task === CLASSIFICATION ? 'Choose List class folders to look at the classes: one subfolder of images per class name.' : 'Choose List images to look at the first files in the image folder.',
      ),
      task === DETECTION ? annotationBlock(set?.annotationFile ?? null) : null,
    );
  }

  // ── detection: the image folder's first page, and the annotation file path (never read) ──
  let annotationPath = null;
  function annotationBlock(path) {
    if (path !== undefined) annotationPath = path;
    const value = h('span', { class: annotationPath ? 'mono data__classpath' : 'muted' }, annotationPath ?? 'Not set in the open recipe');
    const box = h(
      'div',
      { class: 'data__annotation' },
      h('h3', { class: 'data__subheading' }, folderWords(DETECTION).file?.name ?? 'Annotation file'),
      // Shown as the recipe has it; it is changed with Change… on the dataset, which writes it into the recipe.
      h('div', { class: 'row' }, value),
      h('p', { class: 'muted' }, 'Boxes are not drawn here. Choose Check the data to count the boxes in the annotation file, or Change… on a dataset above to choose another file.'),
    );
    return box;
  }

  const IMAGE_PAGE = 100;
  async function listImages(path) {
    const heading = unpairedHeading();
    if (!path) {
      renderClassesIdle();
      fill(classesHost.children[1], 'Enter the image folder, or choose List images on a dataset above.');
      return;
    }
    resetClasses();
    const gen = classGen;
    const ctrl = new AbortController();
    classCtrl = ctrl;
    listButton.disabled = true;
    const set = setForClassFolder(path);
    if (set?.annotationFile) annotationPath = set.annotationFile;
    const head = () => h('div', { class: 'data__classhead' }, h('h2', { class: 'data__heading' }, heading), h('span', { class: 'mono data__classpath', title: path }, path));
    const status = h('p', { class: 'data__classstatus', role: 'status' }, 'Listing the files…');
    fill(classesHost, head(), status);
    try {
      const answer = await api.fsList({ path, mode: 'files', limit: IMAGE_PAGE }, { signal: ctrl.signal });
      if (gen !== classGen || unmounted) return;
      const folder = answer?.path ?? path;
      const files = (Array.isArray(answer?.entries) ? answer.entries : []).filter((entry) => entry.kind === 'file');
      const more = answer?.next !== null && answer?.next !== undefined;
      // Only what this page says: "the first N" when the answer reports more.
      status.textContent = files.length
        ? more
          ? `The first ${fmt.plural(files.length, 'file')} of this folder; there are more.`
          : `${fmt.plural(files.length, 'file')} in this folder.`
        : 'This folder has no files directly inside it.';
      const grid = h('div', { class: 'data__imagegrid', role: 'list', 'aria-label': 'Files in the image folder' });
      const cells = files.map((file) => {
        const cell = h('div', { class: 'data__classthumb data__imagecell', role: 'listitem', title: file.name, dataset: { path: joinFolder(folder, file.name) } }, h('span', { class: 'data__nopreview' }, file.name));
        return cell;
      });
      fill(grid, cells);
      fill(classesHost, head(), status, annotationBlock(), grid);
      if (cells.length) {
        classObserver = new IntersectionObserver(
          (entries) => {
            for (const entry of entries) {
              if (!entry.isIntersecting || gen !== classGen) continue;
              classObserver?.unobserve(entry.target);
              const cell = entry.target;
              imageUrl(cell.dataset.path, { maxSize: THUMB_SIZE }).then(
                (url) => {
                  if (gen !== classGen) return;
                  const img = h('img', { class: 'data__thumbimg', alt: cell.title, draggable: 'false', decoding: 'async' });
                  img.addEventListener('error', () => fill(cell, h('span', { class: 'data__nopreview' }, 'No preview')), { once: true });
                  img.src = url;
                  fill(cell, img);
                },
                () => {
                  if (gen === classGen) fill(cell, h('span', { class: 'data__nopreview' }, 'No preview'));
                },
              );
            }
          },
          { root: grid, rootMargin: '240px 0px' },
        );
        for (const cell of cells) classObserver.observe(cell);
      }
    } catch (error) {
      if (error?.name === 'AbortError' || gen !== classGen || unmounted) return;
      fill(classesHost, head(), notice('err', `The image folder could not be listed: ${error.message}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => browse('lq') }, 'Browse')), annotationBlock());
    } finally {
      if (gen === classGen) listButton.disabled = false;
    }
  }

  /** The recipe dataset whose class folder this is, for its phase (validation may omit classes). */
  const setForClassFolder = (path) => recipe.datasetFolders().find((set) => (set.lq ?? []).includes(path)) ?? null;

  async function listClasses(path) {
    if (!path) {
      renderClassesIdle();
      fill(classesHost.lastChild, 'Enter the class folder, or choose List class folders on a dataset above.');
      return;
    }
    resetClasses();
    const gen = classGen;
    const ctrl = new AbortController();
    classCtrl = ctrl;
    listButton.disabled = true;
    const status = h('p', { class: 'data__classstatus', role: 'status' }, 'Listing the class folders…');
    fill(classesHost, h('div', { class: 'data__classhead' }, h('h2', { class: 'data__heading' }, 'Class folders'), h('span', { class: 'mono data__classpath' }, path)), status);
    try {
      const answer = await api.fsList({ path, mode: 'folders', limit: CLASS_LIMIT }, { signal: ctrl.signal });
      if (gen !== classGen || unmounted) return;
      renderClassList(path, answer, status);
    } catch (error) {
      if (error?.name === 'AbortError' || gen !== classGen || unmounted) return;
      fill(
        classesHost,
        h('div', { class: 'data__classhead' }, h('h2', { class: 'data__heading' }, 'Class folders'), h('span', { class: 'mono data__classpath' }, path)),
        notice('err', `The class folders could not be listed: ${error.message}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => browse('lq') }, 'Browse')),
      );
    } finally {
      // classCtrl stays: it also cancels the per-class listings until the next list or reset.
      if (gen === classGen) listButton.disabled = false;
    }
  }

  function renderClassList(asked, answer, status) {
    const folder = answer?.path ?? asked;
    const folders = (Array.isArray(answer?.entries) ? answer.entries : []).filter((entry) => entry.kind === 'folder').map((entry) => entry.name);
    const more = answer?.next !== null && answer?.next !== undefined;
    status.textContent = folders.length
      ? `${more ? `The first ${fmt.int(folders.length)}` : fmt.int(folders.length)} ${folders.length === 1 && !more ? 'subfolder' : 'subfolders'}${more ? '; there are more' : ''}.`
      : 'This folder has no subfolders, so it holds no classes.';
    const names = recipe.get().config?.task_options?.class_names;
    const set = setForClassFolder(asked);
    const parts = [];
    if (Array.isArray(names)) {
      const { missing, extra } = compareClassNames(names, folders);
      const isVal = set?.phase === 'val';
      const lines = [];
      if (!missing.length && !extra.length) lines.push(notice('ok', 'Every class name has a folder.'));
      if (missing.length) {
        const text = `${missing.length === 1 ? 'This class name has' : 'These class names have'} no folder: ${missing.join(', ')}.`;
        lines.push(isVal ? notice('', `${text} A validation dataset may omit classes.`) : notice('warn', text));
      }
      if (extra.length) lines.push(notice(isVal ? '' : 'warn', `${extra.length === 1 ? 'This folder is' : 'These folders are'} not in class_names: ${extra.join(', ')}.`));
      if (more) lines.push(notice('', `Only the first ${fmt.int(folders.length)} subfolders were compared.`));
      parts.push(
        h(
          'div',
          { class: 'data__classcompare' },
          h('h3', { class: 'data__subheading' }, 'Compared with task_options.class_names in the open recipe'),
          lines,
          h('p', { class: 'muted' }, 'This compares names only. Choose Check the data to open the files.'),
        ),
      );
    }
    const rows = folders.map((name) => {
      const thumbs = h('div', { class: 'data__classthumbs' });
      const count = h('span', { class: 'data__classcount' });
      const row = h('div', { class: 'data__classrow', dataset: { path: joinFolder(folder, name) } }, h('span', { class: 'mono data__classname' }, name), count, thumbs);
      row._parts = { thumbs, count };
      return row;
    });
    const list = h('div', { class: 'data__classlist', role: 'list', 'aria-label': 'Class subfolders' }, rows.map((row) => (row.setAttribute('role', 'listitem'), row)));
    fill(
      classesHost,
      h('div', { class: 'data__classhead' }, h('h2', { class: 'data__heading' }, 'Class folders'), h('span', { class: 'mono data__classpath', title: folder }, folder)),
      status,
      parts,
      list,
    );
    if (rows.length) {
      const gen = classGen;
      classObserver = new IntersectionObserver((entries) => onClassVisible(entries, gen), { root: list, rootMargin: '240px 0px' });
      for (const row of rows) classObserver.observe(row);
    }
  }

  const joinFolder = (folder, name) => `${String(folder).replace(/\/+$/, '')}/${name}`;

  function onClassVisible(entries, gen) {
    for (const entry of entries) {
      if (!entry.isIntersecting || gen !== classGen) continue;
      classObserver?.unobserve(entry.target);
      const row = entry.target;
      const { thumbs, count } = row._parts;
      const path = row.dataset.path;
      count.textContent = 'Listing…';
      api.fsList({ path, mode: 'files', limit: CLASS_THUMBS }, { signal: classCtrl?.signal }).then(
        (answer) => {
          if (gen !== classGen || unmounted) return;
          const files = (Array.isArray(answer?.entries) ? answer.entries : []).filter((e) => e.kind === 'file');
          const more = answer?.next !== null && answer?.next !== undefined;
          // Only what one page says: "at least" when the answer reports more.
          count.textContent = files.length ? `${more ? 'at least ' : ''}${fmt.plural(files.length, 'file')}` : 'No files';
          fill(
            thumbs,
            files.map((file) => {
              const cell = h('div', { class: 'data__classthumb', title: file.name });
              imageUrl(joinFolder(path, file.name), { maxSize: THUMB_SIZE }).then(
                (url) => {
                  if (gen !== classGen) return;
                  const img = h('img', { class: 'data__thumbimg', alt: file.name, draggable: 'false', decoding: 'async' });
                  img.addEventListener('error', () => fill(cell, h('span', { class: 'data__nopreview' }, 'No preview')), { once: true });
                  img.src = url;
                  fill(cell, img);
                },
                () => {
                  if (gen === classGen) fill(cell, h('span', { class: 'data__nopreview', title: file.name }, 'No preview'));
                },
              );
              return cell;
            }),
          );
        },
        (error) => {
          if (error?.name === 'AbortError' || gen !== classGen) return;
          count.textContent = `Could not list: ${error.message}`;
        },
      );
    }
  }

  // ── overlap between training and validation (a second scan, only when asked) ──
  let overlapKey = '';
  let overlapCtrl = null;
  let overlapPlan = null;
  const QUICK_LABEL = 'Quick check: same files on disk';
  const overlapButton = h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => runOverlap() }, QUICK_LABEL);
  const overlapResult = h('div', { class: 'data__overlapresult', 'aria-live': 'polite' });
  // The main action compares encoded bytes as an analysis job (restoration recipes); the instant scan stays
  // as the quieter second action.
  const leakHost = h('div', { class: 'data__leak' });
  const unmountLeak = mountLeakage(leakHost, leakMain, { onActivity: () => setMode('leak') });
  fill(
    overlapBox,
    h('h3', { class: 'data__subheading' }, 'Training and validation'),
    leakHost,
    h(
      'div',
      { class: 'data__quick' },
      h('p', { class: 'field__help' }, 'Compares physical file identity between the training folders and the validation folders of the open recipe.'),
      h('div', null, overlapButton),
      overlapResult,
    ),
  );

  // ── a validation split (split-dataset job; the four tasks whose training data the service splits) ──
  // The training dataset type it splits for each task, lowercase as it compares them (the same table as
  // the start form in runs/analysis.js).
  const SPLIT_TYPES = {
    image_restoration: ['pairedimagedataset', 'pairedvideodataset'],
    image_classification: ['classfolderdataset'],
    semantic_segmentation: ['imagemaskdataset'],
    object_detection: ['cocodetectiondataset'],
  };
  /** `what`: 'pairs' (paired folders), 'scenes' (paired frame folders, split by whole scene) or 'images'. */
  const splitLine = (what) =>
    what === 'pairs'
      ? 'Sets aside part of the training pairs for validation: the same pairs every time, as two lists of file names. Nothing is copied or moved.'
      : what === 'scenes'
        ? 'Sets aside whole scenes of the training frames for validation: the same scenes every time, as two scene lists and a recipe that uses them. Nothing is copied or moved.'
        : 'Sets aside part of the training images for validation: the same images every time, as two lists and a recipe that uses them. Nothing is copied or moved.';
  const splitLead = h('p', { class: 'field__help' });
  const splitButton = h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => openSplit() }, 'Make a validation split…');
  fill(splitBox, h('h3', { class: 'data__subheading' }, 'A validation split'), splitLead, h('div', null, splitButton));
  let splitKey = '';
  let splitting = false;
  /** Offered where the service accepts it (studio/analysis_jobs.py): datasets.train of the type it splits for the recipe's task. */
  function renderSplit() {
    const r = recipe.get();
    const config = r.phase === 'open' ? r.config : null;
    const train = config?.datasets?.train;
    const task = config?.task ?? 'image_restoration';
    const paired = task === 'image_restoration';
    const type = String(train?.type ?? '').toLowerCase();
    const show = Boolean(config) && !(paired && recipeUnpaired()) && (SPLIT_TYPES[task] ?? []).includes(type);
    const hasVal = show && Object.keys(config.datasets).some((key) => key.split('_')[0] === 'val');
    const what = !paired ? 'images' : type === 'pairedvideodataset' ? 'scenes' : 'pairs';
    const key = JSON.stringify([show, hasVal, what]);
    if (key === splitKey) return;
    splitKey = key;
    splitBox.hidden = !show;
    splitLead.textContent = hasVal ? splitLine(what) : `This recipe has no validation data. ${splitLine(what)}`;
  }
  async function openSplit() {
    if (splitting) return;
    splitting = true;
    try {
      const mod = await import('../runs/analysis.js');
      if (typeof mod.openAnalysisDialog !== 'function') throw new Error('The analysis part has no openAnalysisDialog.');
      // The dialog reads the open recipe itself; its result is a job under Runs, Analysis jobs.
      await mod.openAnalysisDialog({ kind: 'split-dataset' });
    } catch (error) {
      toast(`The split dialog could not be opened: ${error?.message ?? error}`, { kind: 'err' });
    } finally {
      splitting = false;
    }
  }

  // ── training patches (extract-patches job; restoration recipes with paired folders on disk) ──
  const PATCH_LINE = 'Cuts the pairs of a dataset into small paired patches, saved as PNG files with their list and the recipe lines that use them. The source images are not changed.';
  const patchLead = h('p', { class: 'field__help' }, PATCH_LINE, ' Make the validation split first, so patches of one image do not land on both sides.');
  const patchButton = h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => openPatches() }, 'Extract patches…');
  fill(patchBox, h('h3', { class: 'data__subheading' }, 'Patches for training'), patchLead, h('div', null, patchButton));
  let patching = false;
  /** Offered where the service accepts it (studio/analysis_jobs.py): task image_restoration and a paired dataset kept as folders. */
  function renderPatches() {
    const r = recipe.get();
    const config = r.phase === 'open' ? r.config : null;
    const show =
      Boolean(config) &&
      !recipeUnpaired() &&
      (config.task ?? 'image_restoration') === 'image_restoration' &&
      registrationSets(recipe.datasetFolders()).some((set) => set.storage === 'disk');
    if (patchBox.hidden === !show) return;
    patchBox.hidden = !show;
  }
  async function openPatches() {
    if (patching) return;
    patching = true;
    try {
      const mod = await import('../runs/analysis.js');
      if (typeof mod.openAnalysisDialog !== 'function') throw new Error('The analysis part has no openAnalysisDialog.');
      // The dialog reads the open recipe itself; its result is a job under Runs, Analysis jobs.
      await mod.openAnalysisDialog({ kind: 'extract-patches' });
    } catch (error) {
      toast(`The patch extraction dialog could not be opened: ${error?.message ?? error}`, { kind: 'err' });
    } finally {
      patching = false;
    }
  }

  // ── alignment of a sample of pairs (pair-registration job; paired restoration recipes only) ──
  const regHost = h('div', { class: 'data__leak' });
  const unmountReg = mountRegistration(regHost, regMain, { onActivity: () => setMode('reg') });
  fill(regBox, h('h3', { class: 'data__subheading' }, 'Alignment of a sample of pairs'), regHost);
  function renderRegistration() {
    const r = recipe.get();
    const task = r.summary?.task ?? r.config?.task ?? null;
    const show = r.phase === 'open' && Boolean(r.config) && !recipeUnpaired() && !AUDIT_TASKS.includes(task) && registrationSets(recipe.datasetFolders()).length > 0;
    if (regBox.hidden === !show) return;
    regBox.hidden = !show;
  }

  function renderOverlap() {
    const r = recipe.get();
    const plan = r.config ? overlapRequest(recipe.datasetFolders()) : null;
    const key = plan ? JSON.stringify(plan) : '';
    if (key === overlapKey) return;
    overlapKey = key;
    overlapPlan = plan;
    overlapCtrl?.abort();
    overlapCtrl = null;
    overlapBox.hidden = !plan;
    // Identical encoded bytes: refused by the service for anything but a restoration recipe.
    leakHost.hidden = !plan || recipeUnpaired() || AUDIT_TASKS.includes(recipe.get().summary?.task ?? recipe.get().config?.task ?? null);
    overlapButton.disabled = false;
    fill(overlapButton, QUICK_LABEL);
    fill(overlapResult);
  }

  async function runOverlap() {
    if (!overlapPlan || overlapCtrl) return;
    const plan = overlapPlan;
    const ctrl = new AbortController();
    overlapCtrl = ctrl;
    overlapButton.disabled = true;
    fill(overlapButton, h('span', { class: 'spinner', 'aria-hidden': 'true' }), 'Checking…');
    fill(overlapResult, h('p', { class: 'muted' }, 'Checking…'));
    try {
      const result = await api.scan(plan.body, { signal: ctrl.signal });
      if (ctrl !== overlapCtrl || unmounted) return;
      fill(overlapResult, overlapView(result, plan));
    } catch (error) {
      if (error?.name === 'AbortError' || ctrl !== overlapCtrl || unmounted) return;
      fill(overlapResult, notice('err', `The overlap could not be checked: ${error.message}`));
    } finally {
      if (ctrl === overlapCtrl) {
        overlapCtrl = null;
        overlapButton.disabled = false;
        fill(overlapButton, QUICK_LABEL);
      }
    }
  }

  function overlapView(result, plan) {
    const issues = Array.isArray(result?.issues) ? result.issues : [];
    const found = issues.filter((issue) => String(issue?.message ?? '').startsWith(OVERLAP_PREFIX));
    const other = issues.filter((issue) => !found.includes(issue));
    const c = result?.counts ?? {};
    const checked = [Number.isFinite(c.gt) ? fmt.plural(c.gt, 'ground truth image') : '', Number.isFinite(c.lq) && c.lq ? fmt.plural(c.lq, 'low-quality image') : '']
      .filter(Boolean)
      .join(' and ');
    const checkedText = checked ? `${checked} of ${plan.trainKey}` : `the folders of ${plan.trainKey}`;
    const against = `${fmt.plural(plan.body.validationRoots.length, 'folder')} of ${plan.valKeys.join(', ')}`;
    const capped = limitLine(result, 'issues', 'issue', 'issues');
    return h(
      'div',
      { class: 'data__overlap' },
      h('p', null, `Checked ${checkedText} against ${against}.`),
      found.length
        ? notice('warn', `${fmt.plural(found.length, 'training file')} ${found.length === 1 ? 'is' : 'are'} the same physical file as a validation file.`)
        : notice('ok', capped ? 'None of the issues shown says a training file is also a validation file.' : 'The scan reported no training file that is the same physical file as a validation file.'),
      capped ? notice('warn', `${capped} The evidence was capped, so more overlaps may exist than are listed.`) : null,
      found.length ? issueList(found) : null,
      other.length ? h('details', { class: 'data__overlapother' }, h('summary', null, `Other issues from the same scan (${fmt.int(other.length)})`), issueList(other)) : null,
    );
  }

  async function writeFolders() {
    const target = targetSelect?.value;
    const gt = data.get().gt.trim();
    const unpaired = recipeUnpaired();
    // For a single-folder task the field may be empty while an older folder is stored: use what is shown.
    const lq = unpaired ? lqInput.value.trim() : data.get().lq.trim();
    if (!target || (unpaired ? !lq : !gt)) return;
    writing = true;
    renderWrite();
    try {
      // A task that reads only dataroot_lq gets only that option written.
      await recipe.applyEdits([
        ...(unpaired ? [] : [{ path: ['datasets', target, 'dataroot_gt'], value: [gt] }]),
        ...(lq ? [{ path: ['datasets', target, 'dataroot_lq'], value: [lq] }] : []),
      ]);
      toast(`Set the folders of ${target} in the recipe.`, { kind: 'ok' });
    } catch (error) {
      toast(`The folders could not be set in the recipe: ${error.message}`, { kind: 'err' });
    } finally {
      writing = false;
      renderWrite();
    }
  }

  const figure = (label, value) =>
    h(
      'div',
      { class: 'data__figure' },
      label,
      Number.isFinite(value) ? h('span', { class: 'data__figvalue' }, fmt.int(value)) : h('span', { class: 'data__figabsent' }, 'Not reported'),
    );

  // Two parts inside the one status region: the notes change while the person types (stale or
  // not), the body only when a new result arrives, so a keystroke never re-announces the counts.
  const notesBox = h('div', { class: 'data__notes' });
  const bodyBox = h('div', { class: 'data__resultbody' });
  const packBox = h('div', { class: 'data__pack', hidden: true });
  fill(resultBox, notesBox, bodyBox, packBox);
  let lastPack = '';
  let packing = false;

  /** "Pack these folders…": only for a current paired check without errors, with exactly what was checked. */
  function renderPack() {
    const d = shown();
    const result = d.result;
    const { errors } = countIssues(result?.issues);
    const dataset = !recipeUnpaired() && result && d.current && d.phase === 'done' && result.ok !== false && !errors ? packDataset(result, d.scanned) : null;
    const key = dataset ? JSON.stringify(dataset) : '';
    if (key === lastPack) return;
    lastPack = key;
    packBox.hidden = !dataset;
    fill(
      packBox,
      dataset
        ? [
            h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => openPack(dataset) }, 'Pack these folders into a portable dataset…'),
            h('p', { class: 'muted data__packline' }, PACK_LINE),
          ]
        : null,
    );
  }

  async function openPack(dataset) {
    if (packing) return;
    packing = true;
    try {
      const mod = await import('../runs/analysis.js');
      if (typeof mod.openAnalysisDialog !== 'function') throw new Error('The analysis part has no openAnalysisDialog.');
      await mod.openAnalysisDialog({ kind: 'pack-dataset', dataset });
    } catch (error) {
      toast(`The packing dialog could not be opened: ${error?.message ?? error}`, { kind: 'err' });
    } finally {
      packing = false;
    }
  }
  let lastNotes = '';
  let lastBody = '';
  let resultSerial = 0;
  let resultRef = null;
  function renderResult() {
    renderPack();
    const d = shown();
    if (d.result !== resultRef) {
      resultRef = d.result;
      resultSerial += 1;
    }
    const current = d.current;
    // A single-folder task has no check: an older pair result is not shown as if it were about this recipe.
    const unpaired = recipeUnpaired();
    const result = unpaired ? null : d.result;
    const notesKey = JSON.stringify([!unpaired && d.phase === 'error' ? d.error : null, Boolean(result) && !current]);
    if (notesKey !== lastNotes) {
      lastNotes = notesKey;
      fill(
        notesBox,
        !unpaired && d.phase === 'error' ? notice('err', `The folders could not be checked: ${d.error}`) : null,
        result && !current ? notice('warn', 'These results are for different folders or scale. Check again to update them.') : null,
      );
      notesBox.hidden = !notesBox.firstChild;
    }
    const words = d.scanned ? wordsFor(d.scanned.gt, d.scanned.lq).words : DEFAULT_WORDS;
    const lists = unpaired ? (unpairedTask() === CLASSIFICATION ? 'classes' : 'images') : '';
    const bodyKey = JSON.stringify([resultSerial, result ? '' : d.phase, words.gt.name, words.lq.name, unpaired, lists]);
    if (bodyKey === lastBody) return;
    lastBody = bodyKey;
    const parts = [];
    if (unpaired) {
      // The sentence names the button this task really has.
      parts.push(
        h(
          'p',
          { class: 'muted' },
          AUDIT_TASKS.includes(unpairedTask())
            ? `The folder check needs pairs of folders, so it does not cover this task. Choose Check the data to check the recipe's datasets, or ${lists === 'classes' ? 'List class folders to see the classes' : 'List images to see the files'} on the right.`
            : lists === 'classes'
              ? 'Nothing is checked for this task. Choose List class folders to see the classes on the right.'
              : 'Nothing is checked for this task. Choose List images to see the files on the right.',
        ),
      );
    } else if (result) {
      const counts = result.counts ?? {};
      const issues = Array.isArray(result.issues) ? result.issues : [];
      const samples = Array.isArray(result.samples) ? result.samples : [];
      const s = d.scanned;
      const paired = Boolean(s?.lq);
      if (s) {
        parts.push(
          h(
            'dl',
            { class: 'kv data__scanned' },
            h('dt', null, words.gt.name),
            s.options?.gtRoots?.length
              ? h('dd', { class: 'mono data__value data__rootlist' }, s.options.gtRoots.map((p) => h('span', { class: 'data__rootitem' }, p)))
              : h('dd', { class: 'mono data__value' }, s.gt),
            h('dt', null, words.lq.name),
            s.options?.lqRoots?.length
              ? h('dd', { class: 'mono data__value data__rootlist' }, s.options.lqRoots.map((p) => h('span', { class: 'data__rootitem' }, p)))
              : h('dd', { class: s.lq && !s.options?.gtRoots ? 'mono data__value' : 'muted data__value' }, s.lq && !s.options?.gtRoots ? s.lq : 'None'),
            h('dt', null, 'Scale'),
            h('dd', null, scaleText(s.scale)),
          ),
        );
      }
      parts.push(
        h(
          'div',
          { class: 'data__figures' },
          // Short labels: three figures share a 300px column. The heading above says these are images.
          figure(roleChip('gt', words.gt.name), counts.gt),
          figure(roleChip('lq', words.lq.name), counts.lq),
          figure(h('span', { class: 'data__figlabel' }, 'Pairs'), counts.paired),
        ),
      );
      const facts = [];
      if (typeof result.source === 'string' && result.source !== 'folder') facts.push(h('dt', null, 'Source'), h('dd', null, SOURCE_WORDS[result.source] ?? result.source));
      if (Number.isFinite(result.headerRowsChecked)) facts.push(h('dt', null, 'Header rows checked'), h('dd', null, fmt.int(result.headerRowsChecked)));
      if (facts.length) parts.push(h('dl', { class: 'kv data__scanned' }, facts));
      const roots = Array.isArray(result.roots) ? result.roots : [];
      if (roots.length > 1) parts.push(rootsTable(roots));
      parts.push(h('p', { class: 'data__summary' }, issueSummary(issues)));
      const sampleLimit = paired ? limitLine(result, 'samples', words.pair, `${words.pair}s`) : limitLine(result, 'samples', 'image', 'images');
      const issueLimit = limitLine(result, 'issues', 'issue', 'issues');
      if (sampleLimit) parts.push(notice('', sampleLimit));
      else if (result.truncated && !issueLimit) parts.push(notice('', `Showing the first ${fmt.int(samples.length)} images. The folder holds more than the check returned.`));
      if (issueLimit) parts.push(notice('', `${issueLimit} The error and warning counts above are of the issues shown.`));
      if (issues.length) parts.push(issueList(issues));
    } else if (d.phase === 'scanning') {
      parts.push(h('p', { class: 'muted' }, 'Checking the folders…'));
    } else if (d.phase !== 'error') {
      parts.push(h('p', { class: 'muted' }, 'Enter a ground truth folder and choose Check folders.'));
    }
    fill(bodyBox, parts);
    bodyBox.hidden = !parts.length;
  }

  const tail = (path) => {
    const text = String(path ?? '');
    const parts = text.split('/').filter(Boolean);
    return parts.slice(-3).join('/') || text;
  };

  /** Counts per folder pair, as the scan's `roots` report them. */
  function rootsTable(roots) {
    const cell = (n) => h('td', { class: 'num' }, Number.isFinite(n) ? fmt.int(n) : h('span', { class: 'muted' }, 'Not reported'));
    return h(
      'table',
      { class: 'table data__roots' },
      h('caption', { class: 'data__rootscaption' }, `${fmt.int(roots.length)} folder pairs`),
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', { scope: 'col', class: 'data__rootcol' }, 'Pair'),
          h('th', { scope: 'col' }, 'Folders'),
          h('th', { scope: 'col', class: 'num data__countcol' }, 'GT'),
          h('th', { scope: 'col', class: 'num data__countcol' }, 'LQ'),
          h('th', { scope: 'col', class: 'num data__countcol' }, 'Paired'),
        ),
      ),
      h(
        'tbody',
        null,
        roots.map((root) =>
          h(
            'tr',
            null,
            h('td', null, String(root.id ?? '')),
            h('td', { class: 'mono data__rootpath', title: [root.gt, root.lq].filter(Boolean).join('\n') }, tail(root.gt), root.lq ? h('span', { class: 'muted' }, ` and ${tail(root.lq)}`) : null),
            cell(root.counts?.gt),
            cell(root.counts?.lq),
            cell(root.counts?.paired),
          ),
        ),
      ),
    );
  }

  // ── images: dataset files, and LMDB records ({path, key}) with their own small cache ──
  const lmdbCache = new Map(); // "size|path|key" -> Promise<string>
  const LMDB_CACHED = 120;
  function lmdbUrl(path, key, maxSize) {
    const id = `${maxSize || 0}|${path}|${key}`;
    const hit = lmdbCache.get(id);
    if (hit) return hit;
    const promise = api.previewBlob(path, { key, ...(maxSize ? { maxSize } : {}) }).then((blob) => URL.createObjectURL(blob));
    promise.catch(() => lmdbCache.delete(id));
    lmdbCache.set(id, promise);
    if (lmdbCache.size > LMDB_CACHED) {
      const [oldId, old] = lmdbCache.entries().next().value;
      lmdbCache.delete(oldId);
      old.then((url) => URL.revokeObjectURL(url)).catch(() => {});
    }
    return promise;
  }
  /** Image address for one side of a sample; an LMDB sample carries `gtKey` / `lqKey`. */
  function sampleImage(sample, side, maxSize) {
    const key = sample?.[`${side}Key`];
    if (typeof key === 'string' && key) return lmdbUrl(sample[side], key, maxSize);
    return imageUrl(sample[side], maxSize ? { maxSize } : {});
  }

  // ── contact sheet ──
  let sheetRef;
  let rootFilter = '';
  let sheetGen = 0;
  let observer = null;
  let tiles = [];
  let samples = [];
  let selected = -1;

  /** Offer "Show pairs from" only when the scan covered several folder pairs. */
  function syncRootFilter(result) {
    const roots = Array.isArray(result?.roots) ? result.roots : [];
    rootSelectWrap.hidden = roots.length < 2;
    if (roots.length < 2) {
      rootFilter = '';
      fill(rootSelect);
      return roots;
    }
    fill(
      rootSelect,
      h('option', { value: '' }, 'All folder pairs'),
      roots.map((root) => h('option', { value: String(root.id) }, `Pair ${root.id}: ${tail(root.gt)}`)),
    );
    if (!roots.some((root) => String(root.id) === rootFilter)) rootFilter = '';
    rootSelect.value = rootFilter;
    return roots;
  }

  function buildSheet(result) {
    sheetGen += 1;
    observer?.disconnect();
    observer = null;
    const roots = syncRootFilter(result);
    const every = Array.isArray(result?.samples) ? result.samples : [];
    samples = rootFilter ? every.filter((sample) => String(sample.rootId) === rootFilter) : every;
    const shownCounts = rootFilter ? roots.find((root) => String(root.id) === rootFilter)?.counts : result?.counts;
    selected = -1;
    const marks = errorsBySample(samples, result?.issues);
    tiles = samples.map((sample, index) => buildTile(sample, index, marks.get(index)));
    fill(strip, tiles);
    strip.hidden = !tiles.length;
    countEl.hidden = !tiles.length;
    sheetEmpty.hidden = Boolean(tiles.length);
    countEl.textContent = tiles.length ? countLine(samples.length, shownCounts, samples.some((s) => s.lq)) : '';
    const table = folderWords(recipe.get().summary?.task ?? null).records === true;
    // Only video clip datasets: nothing here to check as images.
    const videoOnly = !table && videoPrep.get().applies && recipe.datasetFolders().every((set) => String(set.type).toLowerCase() === VIDEO_DATASET || (!set.gt.length && !set.lq.length));
    const VIDEO_ONLY = 'This recipe trains from video files. They are checked and prepared in the recipe.';
    sheetEmpty.textContent = result ? 'The check returned no images to show.' : table ? 'This recipe trains on tables of numbers: there are no images to look at.' : videoOnly ? VIDEO_ONLY : 'Checked images appear here.';
    ensureCompare(result ? EMPTY_NONE : table ? 'This recipe trains on tables of numbers: there are no images to look at.' : videoOnly ? VIDEO_ONLY : EMPTY_BEFORE);
    if (tiles.length) {
      observer = new IntersectionObserver(onVisible, { root: strip, rootMargin: '0px 480px' });
      for (const tile of tiles) observer.observe(tile);
      select(0, false);
    } else {
      compare.setSources({ a: null, b: null });
      fill(sampleName);
      fill(samplePos);
      fill(sampleNote);
      sampleNote.hidden = true;
      resetAlignment();
    }
  }

  function buildTile(sample, index, errors) {
    const thumb = h('div', { class: 'data__thumb' });
    const dims = [
      fmt.dims(sample.gtWidth, sample.gtHeight) ? `GT ${fmt.dims(sample.gtWidth, sample.gtHeight)}` : '',
      fmt.dims(sample.lqWidth, sample.lqHeight) ? `LQ ${fmt.dims(sample.lqWidth, sample.lqHeight)}` : '',
    ].filter(Boolean);
    return h(
      'div',
      {
        class: `data__tile${errors ? ' data__tile--err' : ''}`,
        role: 'option',
        'aria-selected': 'false',
        tabindex: '-1',
        dataset: { index: String(index) },
        title: errors ? errors.join('\n') : null,
        onclick: () => select(index, true),
      },
      thumb,
      h('span', { class: 'data__tilename', title: sample.name }, sample.name ?? sample.id ?? ''),
      dims.length ? h('span', { class: 'data__tiledims' }, dims.join(', ')) : null,
      errors ? h('span', { class: 'data__mark' }, errors.length > 1 ? `${errors.length} errors` : 'Error') : null,
    );
  }

  function noPreview(thumb, reason) {
    fill(thumb, h('span', { class: 'data__nopreview', title: reason }, 'No preview'));
  }

  function onVisible(entries) {
    const gen = sheetGen;
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer?.unobserve(entry.target);
      const index = Number(entry.target.dataset.index);
      const sample = samples[index];
      const thumb = entry.target.firstChild;
      if (!sample?.gt) {
        noPreview(thumb, 'The check returned no ground truth path for this image.');
        continue;
      }
      // Small copies for the strip; only tiles near the visible part of the strip get here.
      sampleImage(sample, 'gt', THUMB_SIZE).then(
        (url) => {
          if (gen !== sheetGen) return;
          const img = h('img', { class: 'data__thumbimg', alt: '', draggable: 'false', decoding: 'async' });
          img.addEventListener('error', () => noPreview(thumb, 'The browser could not decode this image.'), { once: true });
          img.src = url;
          fill(thumb, img);
        },
        (error) => {
          if (gen === sheetGen) noPreview(thumb, error?.message || 'The image could not be loaded.');
        },
      );
    }
  }

  function select(index, focus) {
    if (index < 0 || index >= tiles.length) return;
    if (index !== selected) {
      const old = tiles[selected];
      if (old) {
        old.setAttribute('aria-selected', 'false');
        old.tabIndex = -1;
      }
      selected = index;
      const tile = tiles[index];
      tile.setAttribute('aria-selected', 'true');
      tile.tabIndex = 0;
      showSample();
    }
    if (focus) {
      tiles[index].focus();
      tiles[index].scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
  }

  /** Words for the checked folders (the sheet and stage belong to the last check). */
  function sheetWords() {
    const s = shown().scanned;
    return s ? wordsFor(s.gt, s.lq).words : DEFAULT_WORDS;
  }

  let stageWordsKey = '';
  function showSample() {
    const sample = samples[selected];
    if (!sample) return;
    resetAlignment();
    const words = sheetWords();
    stageWordsKey = `${words.gt.name}|${words.lq.name}`;
    const gtNote = reducedNote(sample.gtWidth, sample.gtHeight);
    const lqNote = reducedNote(sample.lqWidth, sample.lqHeight);
    // The stage gets the large preview (at most PREVIEW_MAX); `note` says when that is smaller than the file.
    compare.setSources({
      a: sample.gt ? { label: words.gt.name, role: 'gt', load: () => sampleImage(sample, 'gt'), ...(gtNote ? { note: gtNote } : {}) } : null,
      b: sample.lq ? { label: words.lq.name, role: 'lq', load: () => sampleImage(sample, 'lq'), ...(lqNote ? { note: lqNote } : {}) } : null,
    });
    fill(sampleName, h('span', { class: 'mono truncate', title: sample.gt ?? '' }, sample.name ?? sample.id ?? ''));
    fill(
      sampleNote,
      [gtNote && sample.gt ? `${words.gt.short} ${gtNote.charAt(0).toLowerCase()}${gtNote.slice(1)}` : '', lqNote && sample.lq ? `${words.lq.short} ${lqNote.charAt(0).toLowerCase()}${lqNote.slice(1)}` : '']
        .filter(Boolean)
        .join('. '),
    );
    sampleNote.hidden = !sampleNote.textContent;
    samplePos.textContent = `${sample.lq ? 'Pair' : 'Image'} ${fmt.int(selected + 1)} of ${fmt.int(tiles.length)}`;
  }

  // ── alignment ──
  let alignCtrl = null;
  let alignKey = '';
  /** {result, name} of the last measurement of the selected pair, for "Details". */
  let alignAnswer = null;
  let alignDialog = null;

  /** Show or reset the alignment bar for the selected sample; a running request for another pair is dropped. */
  function resetAlignment() {
    const sample = samples[selected];
    const key = sample?.gt && sample?.lq ? `${sample.gt}\n${sample.lq}` : '';
    if (key === alignKey) return;
    alignKey = key;
    alignCtrl?.abort();
    alignCtrl = null;
    alignBar.hidden = !key;
    alignButton.disabled = false;
    fill(alignButton, 'Check alignment');
    alignAnswer = null;
    setAlignLine('', '');
  }

  /** The one summary line; tone '' | 'warn' | 'err' (paired with words, never colour alone). */
  function setAlignLine(text, tone) {
    alignSummary.textContent = text;
    alignSummary.title = text;
    alignSummary.className = `data__alignsummary${tone ? ` data__alignsummary--${tone}` : ''}`;
    alignDetails.hidden = !alignAnswer;
  }

  async function runAlignment() {
    const sample = samples[selected];
    if (!sample?.gt || !sample?.lq || alignCtrl) return;
    const raw = radiusInput.value.trim();
    alignAnswer = null;
    if (!raw) {
      setAlignLine('Enter a search radius from 2 to 12.', 'err');
      return;
    }
    const scale = data.get().scale;
    const params = { gt: sample.gt, lq: sample.lq, radius: Number(raw), ...(Number.isInteger(scale) ? { scale } : {}) };
    const key = alignKey;
    const ctrl = new AbortController();
    alignCtrl = ctrl;
    alignButton.disabled = true;
    setAlignLine('Measuring…', '');
    try {
      const result = await api.alignment(params, { signal: ctrl.signal });
      if (ctrl !== alignCtrl || key !== alignKey || unmounted) return;
      alignAnswer = { result, name: sample.name ?? baseOf(sample.gt) };
      const line = alignmentSummary(result);
      setAlignLine(line.text || 'Measured. The answer named no channels.', line.warn ? 'warn' : '');
    } catch (error) {
      if (error?.name === 'AbortError' || ctrl !== alignCtrl || key !== alignKey || unmounted) return;
      setAlignLine(`The alignment could not be measured: ${error.message}`, 'err');
    } finally {
      if (ctrl === alignCtrl) {
        alignCtrl = null;
        alignButton.disabled = false;
      }
    }
  }

  function openAlignmentDetails() {
    if (!alignAnswer || alignDialog) return;
    const handle = openDialog({
      title: `Alignment of ${alignAnswer.name}`,
      wide: true,
      body: alignmentView(alignAnswer.result),
      foot: (close) => [h('button', { class: 'btn btn--primary', type: 'button', onclick: () => close() }, 'Close')],
    });
    alignDialog = handle;
    handle.closed.then(() => {
      alignDialog = null;
    });
  }

  const exact = (value) => (Number.isFinite(value) ? String(value) : null);
  const limitHead = () => h('th', { scope: 'col' }, h('span', { class: 'visually-hidden' }, 'At the search limit'));
  const num = (value, digits = 2) => h('td', { class: 'num', title: exact(value) }, fixed(value, digits) || h('span', { class: 'muted' }, 'Not reported'));

  function alignmentView(result) {
    const units = typeof result?.units === 'string' && result.units ? result.units : '';
    const unitHead = units ? ` (${units})` : '';
    const evidence = result?.evidence ?? {};
    const estimate = result?.estimate ?? {};
    const patches = Array.isArray(result?.patches) ? result.patches : [];
    const channelRows = ALIGN_CHANNELS.filter(([key]) => evidence[key] || estimate[key]).map(([key, label]) => {
      const ev = evidence[key] ?? {};
      const est = estimate[key] ?? {};
      return h(
        'tr',
        null,
        h('th', { scope: 'row' }, label),
        h('td', null, ev.assessment !== undefined ? assessmentText(ev.assessment) : h('span', { class: 'muted' }, 'Not reported'), evidenceCounts(ev) ? h('span', { class: 'data__aligncounts' }, evidenceCounts(ev)) : null),
        num(est.dx),
        num(est.dy),
        h('td', null, est.atLimit === true ? badge('At the search limit', 'warn') : null),
      );
    });
    const patchTable = (key, label) => {
      const rows = patches.filter((p) => p?.[key]);
      if (!rows.length) return null;
      return h(
        'table',
        { class: 'table data__aligntable' },
        h('caption', { class: 'data__aligncaption' }, label),
        h(
          'thead',
          null,
          h(
            'tr',
            null,
            ['Region (x, y, width × height)', `dx${unitHead}`, `dy${unitHead}`, 'Peak', 'Zero-offset correlation', 'Margin'].map((text, i) => h('th', { scope: 'col', class: i > 0 ? 'num' : null }, text)),
            limitHead(),
          ),
        ),
        h(
          'tbody',
          null,
          rows.map((p) => {
            const c = p[key];
            const r = p.region ?? {};
            return h(
              'tr',
              null,
              h('td', { class: 'mono' }, [r.x, r.y].every(Number.isFinite) ? `${r.x}, ${r.y}, ${fmt.dims(r.width, r.height)}` : 'Not reported'),
              num(c.dx),
              num(c.dy),
              num(c.peak, 3),
              num(c.zero, 3),
              num(c.peakMargin, 3),
              h('td', null, c.atLimit === true ? badge('At the search limit', 'warn') : null),
            );
          }),
        ),
      );
    };
    return h(
      'div',
      { class: 'data__alignbody' },
      h(
        'dl',
        { class: 'kv data__alignfacts' },
        h('dt', null, 'Pair'),
        h('dd', { class: 'mono' }, baseOf(result?.gt), ' and ', baseOf(result?.lq)),
        h('dt', null, 'Scale'),
        h('dd', null, Number.isFinite(result?.scale) ? `${result.scale}×` : h('span', { class: 'muted' }, 'Not reported')),
        h('dt', null, 'Search radius'),
        h('dd', null, Number.isFinite(result?.radius) ? `${fmt.int(result.radius)} ${units}`.trim() : h('span', { class: 'muted' }, 'Not reported')),
      ),
      h(
        'table',
        { class: 'table data__aligntable' },
        h('caption', { class: 'data__aligncaption' }, 'Estimated shift per channel'),
        h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Channel'), h('th', { scope: 'col' }, 'Assessment'), h('th', { scope: 'col', class: 'num' }, `dx${unitHead}`), h('th', { scope: 'col', class: 'num' }, `dy${unitHead}`), limitHead())),
        h('tbody', null, channelRows),
      ),
      typeof result?.direction === 'string' && result.direction ? h('p', { class: 'data__alignnote' }, `Direction: ${result.direction}.`) : null,
      patchTable('luma', 'Brightness per patch'),
      patches.some((p) => p?.blueMinusLuma || p?.redMinusLuma)
        ? h('details', { class: 'data__aligndetails' }, h('summary', null, 'Colour channels per patch'), patchTable('blueMinusLuma', 'Blue against brightness per patch'), patchTable('redMinusLuma', 'Red against brightness per patch'))
        : null,
      typeof result?.method === 'string' && result.method ? h('p', { class: 'data__alignnote' }, `Method: ${result.method}.`) : null,
      result?.alphaIgnored === true ? h('p', { class: 'data__alignnote' }, 'Transparency was ignored.') : null,
      h('p', { class: 'data__alignscope' }, ALIGN_SCOPE),
    );
  }

  const baseOf = (path) => {
    const text = String(path ?? '');
    return text.slice(text.lastIndexOf('/') + 1) || text;
  };

  /** The recipe's task wording can arrive after the check: relabel the stage without reloading the sheet. */
  function renderStageWords() {
    if (selected < 0) return;
    const words = sheetWords();
    if (`${words.gt.name}|${words.lq.name}` !== stageWordsKey) showSample();
  }

  strip.addEventListener('keydown', (event) => {
    const last = tiles.length - 1;
    const next = { ArrowRight: selected + 1, ArrowLeft: selected - 1, Home: 0, End: last }[event.key];
    if (next === undefined || !tiles.length) return;
    event.preventDefault();
    select(Math.max(0, Math.min(last, next)), true);
  });

  function renderSheet() {
    const result = shown().result;
    if (result === sheetRef) return;
    sheetRef = result;
    rootFilter = '';
    buildSheet(result);
  }

  // ── wiring ──
  const renderAll = () => {
    renderRoot();
    renderAuditSection();
    renderWords();
    renderOptions();
    renderButton();
    renderRecipe();
    renderWrite();
    renderOverlap();
    renderSplit();
    renderPatches();
    renderRegistration();
    renderMain();
    renderResult();
    renderFieldIssues();
    renderSheet();
    renderStageWords();
  };
  unsubs.push(
    data.subscribe(() => {
      renderWords();
      renderOptions();
      renderButton();
      renderRecipe();
      renderMain();
      renderWrite();
      renderResult();
      renderFieldIssues();
      renderSheet();
      renderStageWords();
    }),
  );
  unsubs.push(
    recipe.subscribe(() => {
      renderAuditSection();
      renderWords();
      renderButton();
      renderRecipe();
      renderWrite();
      renderOverlap();
      renderSplit();
      renderPatches();
      renderRegistration();
      renderMain();
      renderResult();
      renderStageWords();
    }),
  );
  unsubs.push(conn.subscribe(renderRoot));
  unsubs.push(videoPrep.subscribe(renderVideoLine));
  renderVideoLine();
  renderAll();

  return () => {
    unmounted = true;
    unmountAudit();
    unmountLeak();
    unmountReg();
    alignCtrl?.abort();
    alignDialog?.close();
    overlapCtrl?.abort();
    resetClasses();
    for (const promise of lmdbCache.values()) promise.then((url) => URL.revokeObjectURL(url)).catch(() => {});
    lmdbCache.clear();
    for (const off of unsubs) off();
    sheetGen += 1; // thumbnails still loading are dropped
    observer?.disconnect();
    compare?.destroy();
  };
}
