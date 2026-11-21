// Client for the loopback trainer API (same origin). One function per endpoint; the doc comment of each
// gives the request and the answer as the service really sends them (the service's own description is
// studio/API.md). Nothing here invents data: absent fields stay absent and errors surface as ApiError.

export class ApiError extends Error {
  /**
   * @param {string} message
   * @param {{status?: number, issues?: Array, offline?: boolean}} [info]
   */
  constructor(message, info = {}) {
    super(message);
    this.name = 'ApiError';
    /** HTTP status: 400 refused, 404 not found, 409 conflict or trainer busy, 422 recipe rejected (with issues) */
    this.status = info.status ?? 0;
    this.issues = info.issues ?? null;
    /** true when the service could not be reached at all */
    this.offline = info.offline ?? false;
  }
}

/**
 * `rawBody`: `body` is already JSON text and is sent as written. `text`: the answer is returned as the
 * text the service sent, not parsed. Both exist for documents whose exact writing matters (run files).
 */
async function request(method, url, body, { signal, binary = false, text = false, rawBody = false } = {}) {
  let response;
  try {
    response = await fetch(url, {
      method,
      signal,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : rawBody ? body : JSON.stringify(body),
    });
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    throw new ApiError('The trainer service did not answer.', { offline: true });
  }

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`.trim();
    let issues = null;
    try {
      const payload = await response.json();
      if (payload && typeof payload.error === 'string') message = payload.error;
      if (payload && Array.isArray(payload.issues)) issues = payload.issues;
    } catch {
      // Not JSON: keep the HTTP status line as the message.
    }
    throw new ApiError(message, { status: response.status, issues });
  }

  if (binary) return response.blob();
  if (text) return response.text();
  try {
    return await response.json();
  } catch {
    throw new ApiError(`${url} did not return JSON.`, { status: response.status });
  }
}

const get = (url, opts) => request('GET', url, undefined, opts);
const post = (url, body, opts) => request('POST', url, body, opts);

/** Query string from an object; undefined, null and '' are left out. */
function qs(params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== '') search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : '';
}

/** Body from an object; undefined and null are left out. */
function compact(params) {
  const out = {};
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null) out[key] = value;
  }
  return out;
}

function memo(load) {
  let pending = null;
  const run = () => {
    if (!pending) {
      pending = load().catch((error) => {
        pending = null;
        throw error;
      });
    }
    return pending;
  };
  run.reset = () => {
    pending = null;
  };
  return run;
}

/**
 * One id per thing the person asked for (a start, a queue entry, a replay). The backend derives the run
 * or entry id from it, so sending the same request twice does not start twice.
 */
export function newRequestId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  return `ui-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

/** Largest preview the backend produces (longest edge, px). Larger images arrive reduced. */
export const PREVIEW_MAX = 1536;
/** Longest edge used for grid thumbnails. */
export const THUMB_SIZE = 256;

export const api = {
  /** => {name, version, trainerRoot, stateDir} */
  status: (opts) => get('/api/status', opts),
  /** The trainer machine as it is: interpreter, package versions, platform, CPU, disk. */
  environment: (opts) => get('/api/environment', opts),

  // ───────── recipes ─────────

  /** => [{path, name, family, task, architecture, scale} | {path, name, family, error}] */
  templates: (opts) => get('/api/templates', opts),
  /** => [{path, name, task, architecture, scale} | {path, name, error}] */
  recipes: (opts) => get('/api/recipes', opts),
  /** => {path, name, yaml, revision} */
  recipe: (path, opts) => get(`/api/recipe${qs({ path })}`, opts),
  /**
   * Save as recipes/<name>.yml. With `revision` (from the last load or save of that same file) the
   * backend refuses with 409 when the file changed meanwhile; without it the file is overwritten.
   * => {path, name, yaml, revision}
   */
  saveRecipe: (name, yaml, revision, opts) => post('/api/recipes', compact({ name, yaml, revision }), opts),

  /** => {ok, issues:[{level, path, message}], summary:{name, task, architecture, scale}, config?} */
  preflight: (yaml, opts) => post('/api/preflight', { yaml }, opts),
  /**
   * edits: [{path: [string|number, ...], value} | {path, remove: true}] => {yaml}
   * A list cannot be appended to by index: set the whole list instead.
   */
  render: (yaml, edits, opts) => post('/api/render', { yaml, edits }, opts),

  /**
   * => {tasks:[{id, label, networkField, trainingOnly, architectures, recommendedArchitectures,
   *     datasetTypes, excludedArchitectures, taskOptions}], architectures:[{id, task, registries,
   *     schemaRef}], losses:[{id}], schemaUrl} (cached)
   */
  catalog: memo(() => get('/api/catalog')),
  /**
   * Read the lists again from this installation (its own networks, losses, optimizers, metrics), in a
   * separate CPU process; can take a while. Answers with the new catalog, which also carries
   * {source: 'packaged' | 'local' | …, generatedAt: ISO time | null, signatureIssues: [...]}.
   * The cached catalog and schema are dropped, so the next read sees the new lists.
   */
  catalogRefresh: async (opts) => {
    const result = await post('/api/catalog/refresh', {}, opts);
    api.catalog.reset();
    api.schema.reset();
    return result;
  },
  /** => the full JSON Schema (cached) */
  schema: memo(async () => {
    let url = '/api/schema';
    try {
      const catalog = await api.catalog();
      if (catalog && typeof catalog.schemaUrl === 'string') url = catalog.schemaUrl;
    } catch {
      // Fall back to the documented default.
    }
    return get(url);
  }),

  // ───────── data ─────────

  /**
   * {gt, lq?, scale?} => {counts:{gt,lq,paired}, issues, samples:[{id,name,gt,lq?,gtWidth?,...}],
   *   truncated, limits:{samples,issues}, totals:{samples,issues}}
   */
  scan: (params, opts) => post('/api/data/scan', params, opts),
  /**
   * => PNG Blob of the image at `path`, longest edge at most `maxSize` (16..1536, default 1536),
   * never enlarged. An image larger than that arrives reduced. For a sample inside an LMDB database
   * pass the database path and the sample's `key`.
   */
  previewBlob: (path, { maxSize, key, ...opts } = {}) => post('/api/data/preview', compact({ path, key, maxSize }), { ...opts, binary: true }),
  /**
   * List a folder on the trainer machine.
   * {path?, mode?: 'all'|'folders'|'files', filter?, hidden?, limit?, after?}
   *   => {path, parent, entries:[{kind:'folder'|'file', name}], next}
   */
  fsList: (params, opts) => post('/api/fs/list', compact(params), opts),
  /**
   * Measure how one ground-truth image and its low-quality image line up (runs on the trainer
   * machine's CPU when asked, never during training). {gt, lq, scale?, radius?, gtKey?, lqKey?}; the
   * answer is drawn by the Data screen's alignment panel (views/data.js), which names its fields.
   */
  alignment: (params, opts) => post('/api/data/alignment', compact(params), opts),

  /**
   * Structure of a .safetensors file (no tensor values are read).
   * {path, limit?, filter?, after?} => {path, bytes, tensorCount, elements, dtypeElements, layoutSha256,
   *   metadata, metadataTruncated, matched, tensors:[{name, shape, dtype, elements}], next}
   */
  weights: (params, opts) => post('/api/weights', compact(params), opts),
  /**
   * {left, right, limit?, filter?, after?} => {left, right, counts:{same, leftOnly, rightOnly, shape,
   *   dtype, shapeAndDtype}, sameLayout, differences:[{name, difference, left, right}], next}
   */
  weightsCompare: (params, opts) => post('/api/weights/compare', compact(params), opts),

  // ───────── runs ─────────

  /** => [{id, name, source, status, startedAt?, updatedAt?, task?, total?, iteration?, iterationsPerSecond?, eta?, vram?, queueEntryId?, error?}] */
  runs: (opts) => get('/api/runs', opts),
  /** => list item + {metrics (latest values), log, logCursor, logReset, yaml, artifacts:[{name, path, url}], seed?, invocation?} */
  run: (id, opts) => get(`/api/run${qs({ id })}`, opts),
  /**
   * Light update for an open run: status, reported progress and new log lines after `cursor`.
   * => list item + {metrics, log, logCursor, logReset}. Append `log` unless `logReset`.
   */
  monitor: (id, cursor, opts) => get(`/api/run/monitor${qs({ id, cursor })}`, opts),
  /** Starts training now. 422 carries `issues`; 409 means the trainer is occupied by the queue. */
  startRun: (yaml, requestId, opts) => post('/api/runs', compact({ yaml, requestId }), opts),
  stopRun: (id, opts) => post('/api/runs/stop', { id }, opts),
  /** `params.trainWorkers` (a positive integer) only for a run whose detail says `resumeWorkerOverride: true`. */
  resumeRun: (id, params, opts) => post('/api/runs/resume', { id, ...params }, opts),
  /**
   * Logged history of one attempt. Without `metric`: the list of series. With it: its points.
   * {metric?, source?, budget?, from?, to?} => {sources, source, series:[{key, count, first, last}],
   *   points:[[iteration, number|null]], indexedBytes, sourceBytes, more, metric?, matchedPoints?, downsampled?}
   * `more: true` means the log is not fully indexed yet: call again.
   */
  history: (id, params, opts) => get(`/api/run/history${qs({ id, ...params })}`, opts),
  /**
   * => [{dataset, step, sampleCount, metric, better, evaluation, aggregation}] ("higher" = higher is better),
   *   newest step first. `evaluation` is the id of the recorded measurement (0 = the older per-step record);
   *   one step can appear under several ids when it was measured by more than one attempt. `aggregation` is
   *   'meanPerSample' or 'pooledDataset' (a score over the whole dataset, not a mean of per-image scores).
   */
  evaluations: (id, opts) => get(`/api/run/evaluations${qs({ id })}`, opts),
  /**
   * Validation images a run saved.
   * {dataset?, iteration?, image?, limit? (1..500), after?} => {images:[{kind, iteration, dataset, image,
   *   name, path, url, width?, height?, provenance, lq?, lqUrl?, gt?, gtUrl?}], next, provenance}
   * `url` is the original output; lqUrl/gtUrl are previews of the inputs the run actually used.
   */
  images: (id, params, opts) => get(`/api/run/images${qs({ id, ...params })}`, opts),
  /**
   * Measured validation results of a run, newest first.
   * {dataset?, attempt?, limit? (50), before?} => {evaluations:[{id, attempt, dataset, step, task, unit,
   *   hasClassCounts, metrics:{name: number}, nonfinite:{}}], next} (`next` is the id to pass as `before`)
   */
  validation: (id, params, opts) => get(`/api/run/validation${qs({ id, ...params })}`, opts),
  /**
   * Per-class results of one evaluation (segmentation, classification; needs val.record_samples).
   * {evaluation, limit?, offset?, target?} => class ids, labels, support, predicted and correct counts,
   *   precision, recall, IoU, total and unit; with `target` the ranked predictions inside that class.
   */
  classes: (id, params, opts) => get(`/api/run/classes${qs({ id, ...params })}`, opts),
  /** => {sourceRunId, groupId, recipeDigest, yaml, sourceInvocation?} */
  replayInfo: (id, opts) => get(`/api/run/replay${qs({ id })}`, opts),
  /** What each attempt of a run actually ran with. => {attempts: [...], recorded: boolean} */
  provenance: (id, opts) => get(`/api/run/provenance${qs({ id })}`, opts),
  /** Extra experiments folders whose runs are listed read-only. */
  archives: (opts) => get('/api/archives', opts),
  /** Register an experiments folder (absolute path on the trainer machine). */
  addArchive: (path, opts) => post('/api/archives', { path }, opts),
  /** Forget a registered folder; the folder itself is not touched. => {removed} */
  removeArchive: (id, opts) => post('/api/archives/remove', { id }, opts),
  /**
   * Comparison of two recorded observations of the same dataset and metric. Each of `left` and `right` is
   * a run id or {job: <completed evaluation job id>}. {left, right, dataset, metric, leftStep?, rightStep?,
   * leftEvaluation?, rightEvaluation?} (an evaluation id tells two attempts at the same iteration apart; 0
   * is the legacy per-step record). A standalone evaluation has the single step -1. The answer names both
   * steps and evaluation ids; it is either per image (`summary`, `improvements`, `regressions`) or, for
   * classification, segmentation and detection, `aggregation: 'pooledDataset'` with `pooled: {left, right,
   * delta}`. See studio/API.md, "Comparing measured holdout results".
   */
  compare: (params, opts) => post('/api/compare', compact(params), opts),

  // ───────── run files: a run's recipe, seed, starting point and inputs as one JSON object ─────────
  // Neither call writes anything or starts anything. A run file holds no weights and no data.

  /**
   * {id, attemptId?, mode?: 'replay' | 'weights-only', weightPath?: an artifact `path` of the run
   *   (weights-only), weightsVersion?} => the bundle: {format: 'trainner-run-recipe', version, mode, digest,
   *   origin: {id, name, source, status, attemptId?, checkout?}, recipe: {submitted, effective, archived,
   *   basis}, seed: {value, evidence}, initialization: {kind, weight?}, provenance, inputs: [{field, kind,
   *   value, sourcePath?}], limitations}. Keep it as one object: `digest` covers the other fields.
   */
  reproductionExport: (params, opts) => post('/api/reproduction/export', compact(params), opts),
  /**
   * {bundle, mappings?: [{from, to}], sourceRoot?, name?, seed?, edits?} => {ok, yaml: string|null,
   *   initialization: {kind, seed, seedEvidence, optimizerState, resume}, compatibility: {environment,
   *   source, config, data}, inputs: [{field, kind, declared, sourcePath, path, mapped, status, recorded,
   *   current?}], issues: [{level, path, message, code?}], bundleDigest}. `yaml` is the candidate recipe
   *   for this machine; nothing is saved or queued.
   */
  reproductionPrepare: (params, opts) => post('/api/reproduction/prepare', compact(params), opts),
  /**
   * The same two calls for a bundle kept as text. A bundle that is parsed and written again by the browser
   * can change (1000000.0 is written back as 1000000), and the service then refuses it: its digest no
   * longer matches. So the bundle is read as the text the service sent, shown from a parsed copy, and sent
   * back (or saved to a file) as that same text.
   * reproductionExportText(params) => the bundle as text.
   * reproductionPrepareText(bundleText, {mappings?, sourceRoot?, name?, seed?, edits?}) => the answer of
   *   reproductionPrepare. `bundleText` must be valid JSON (parse it first to check and to display it).
   */
  reproductionExportText: (params, opts) => post('/api/reproduction/export', compact(params), { ...opts, text: true }),
  reproductionPrepareText: (bundleText, params, opts) => {
    const rest = JSON.stringify(compact(params));
    const body = rest === '{}' ? `{"bundle":${bundleText}}` : `{"bundle":${bundleText},${rest.slice(1)}`;
    return post('/api/reproduction/prepare', body, { ...opts, rawBody: true });
  },

  // ───────── queue ─────────

  /** => {paused, activeId, entries:[Entry]} in creation order */
  queue: (opts) => get('/api/queue', opts),
  /** => Entry + {yaml} */
  queueEntry: (id, opts) => get(`/api/queue/entry${qs({ id })}`, opts),
  /** => {groupId, entries, runIds} */
  queueGroup: (id, opts) => get(`/api/queue/group${qs({ id })}`, opts),
  /** Saves a copy of the recipe as a held entry. Nothing starts until it is released. => Entry */
  queueSubmit: ({ yaml, requestId, recipePath }, opts) => post('/api/queue', compact({ yaml, requestId, recipePath }), opts),
  /** action: 'release' | 'hold' | 'cancel' | 'stop' => Entry */
  queueAction: (action, id, opts) => post(`/api/queue/${action}`, { id }, opts),
  /** => {paused, activeId, entries} */
  queuePause: (paused, opts) => post('/api/queue/pause', { paused: Boolean(paused) }, opts),
  /** Saves a held entry that repeats run `runId` with its recorded seed. => Entry */
  queueReplay: (runId, requestId, opts) => post('/api/queue/replay', compact({ runId, requestId }), opts),

  // ───────── controlled studies (arms × seeds; every entry is created on hold) ─────────

  /** => [{id, name, createdAt, baseDigest, armCount, seedCount, entryCount}] newest first */
  studies: (opts) => get('/api/studies', opts),
  /**
   * => the list header + {baseYaml, seeds: number[], arms: [{id, name, edits, entries}]}; an entry is a
   *   queue Entry + {studyId, studyName, armId, armName, seed, runId?, runStatus?} (groupId = studyId).
   */
  study: (id, opts) => get(`/api/study${qs({ id })}`, opts),
  /**
   * {name, baseYaml, arms: [{name, edits: [{path, value} | {path, remove: true}]}], seeds: number[],
   *   requestId?} => the study detail. Every entry is saved on hold; releasing is a queue action.
   *   At most 128 arm/seed entries. Rejects with status 422 and
   *   issues [{level, path, message, armId, armName, seed?}] when an arm does not render or pass the check.
   */
  createStudy: (params, opts) => post('/api/studies', compact(params), opts),
  /**
   * {studyId, leftArmId, rightArmId, dataset, metric, step?} => the same fields + {pairs: [{seed,
   *   leftEntryId, rightEntryId, leftStatus, rightStatus, leftRunId?, rightRunId?,
   *   status: 'unpaired'|'unavailable'|'incompatible'|'compared', comparison?, error?}]}; `comparison` is
   *   the answer of `compare`. One pair per seed: results are descriptive per seed, never pooled.
   */
  studyCompare: (params, opts) => post('/api/study/compare', compact(params), opts),

  // ───────── curricula (one recipe trained in ordered phases; every phase entry is created on hold) ─────────

  /** => [{id, name, createdAt, baseDigest, phaseCount, phaseEntries: {phaseId: current queue entry id}}] newest first */
  curricula: (opts) => get('/api/curricula', opts),
  /**
   * => the list header + {baseYaml, phases: [Phase]}. Phase = {id: 'phase-N', name, mode: 'start' |
   *   'continue' | 'finetune', updates, startStep, targetStep, edits, predecessorPhaseId?, weights?:
   *   'raw' | 'ema' (fine-tune only), entry: the current queue Entry, revisions: [every queue Entry of
   *   this phase]}. A phase Entry is a queue Entry + {curriculumId, phaseId, phaseName, revision,
   *   previousEntryId?, replacedBy?, edits, handoff?: {entryId, runId, attemptId, step, seed, files}}.
   */
  curriculum: (id, opts) => get(`/api/curriculum${qs({ id })}`, opts),
  /**
   * {name, baseYaml, phases: [{name, mode, updates, edits: [{path, value} | {path, remove: true}],
   *   weights?}], requestId?} => the curriculum detail. At most 32 phases; the first is 'start', later
   *   ones 'continue' or 'finetune'. Each phase's edits apply to the base recipe on their own, not on top
   *   of the phase before. The service sets train.total_iter, path.resume_state, the pretrain paths and
   *   (for continue) manual_seed itself. Rejects with status 422 and issues [{level, path, message,
   *   phaseId}] when a phase does not render, does not pass the check, or cannot follow the one before.
   */
  createCurriculum: (params, opts) => post('/api/curricula', compact(params), opts),
  /**
   * Save a new held revision of one phase. {id, phaseId, edits?, requestId?}: leave `edits` out to
   *   reuse the phase's edits, pass [] for the base recipe's settings, or a replacement list. Mode and
   *   iteration count stay fixed. Only while the phase's current entry is held, cancelled, stopped or
   *   failed. => the new queue Entry. `edits: null` is sent as written, so it is not compacted away.
   */
  reviseCurriculumPhase: ({ id, phaseId, edits, requestId }, opts) =>
    post('/api/curriculum/retry', { id, phaseId, ...(edits === undefined ? {} : { edits }), ...(requestId ? { requestId } : {}) }, opts),
  /**
   * What one phase's recipe keeps as best weights, without saving anything. {baseYaml, edits}: the
   *   phase's edits as createCurriculum takes them. => {snapshotMetrics: {name: 'higher'|'lower'},
   *   bestValidationDatasets: string[], bestMetric: null | string}.
   */
  curriculumPhasePreview: (body, opts) => post('/api/curriculum/phase-preview', body, opts),

  /**
   * Translate a neosr TOML document into an editable recipe draft. Writes no file.
   * => {yaml, sourceSha256, needsReview, review, notes}; entries that could not be mapped are kept in a
   *   `neosr_review` section of the YAML, which must be resolved and removed before training.
   */
  importNeosr: (toml, opts) => post('/api/recipes/import-neosr', { toml }, opts),

  // ───────── analysis jobs (CPU only, started by the person, never part of training) ─────────

  /**
   * => [{id, kind: 'predict'|'evaluate'|'export'|'loss-report'|'pack-dataset'|'weight-drift'|'task-data'|
   *   'split-leakage', device, createdAt, requestDigest, params, status, finishedAt?, error?}]
   */
  analysisJobs: (opts) => get('/api/analysis', opts),
  /** => one job, same shape */
  analysisJob: (id, opts) => get(`/api/analysis/job${qs({ id })}`, opts),
  /** => {text, truncated} */
  analysisLog: (id, opts) => get(`/api/analysis/log${qs({ id })}`, opts),
  /** {offset?, limit?} => {result, artifacts:[{path, url}], next} */
  analysisResult: (id, params, opts) => get(`/api/analysis/result${qs({ id, ...params })}`, opts),
  /**
   * Start a job. {kind, yaml, requestId?, weights?, input?, height?, width?, opset?, step?, batches?,
   *   parameter?, discriminator?}: predict needs weights and input (an image or a folder); export needs
   *   weights, height and width; loss-report measures the recipe's loss terms on up to 32 batches.
   * Dataset packing takes no recipe: {kind: 'pack-dataset', dataset: {gtRoots, lqRoots, scale,
   *   filenameTemplate}, destination?, requestId?} writes a portable LMDB copy and answers with the source
   *   inventory, the two database paths and a recipe fragment.
   * Detection recipes only: predict also takes scoreThreshold (0 to 1, boxes must score above it) and
   *   topLimit (most boxes kept per image); export is refused for object detection.
   * Weight drift takes no recipe: {kind: 'weight-drift', left, right} (two .safetensors files; left is
   *   the baseline) measures how far the numbers moved, tensor by tensor.
   * Task data audit: {kind: 'task-data', yaml, auditLimit?: integer | null (null checks every file),
   *   sampleLimit?: 0 to 100} inventories each dataset of a classification, segmentation or detection
   *   recipe. `auditLimit: null` must reach the service, so pass it through `analysisStartExact`.
   * Evaluation: {kind: 'evaluate', yaml, weights, split?: a validation or test dataset key (left out =
   *   all of them), saveImages?: boolean} runs the trainer's own evaluator on the CPU and answers with
   *   {task, weights, device, precision, datasets: [{dataset, samples, metrics: {name: number},
   *   nonfinite: {}}], splits, outputCount}.
   * Split leakage: {kind: 'split-leakage', yaml, groupLimit?: 0 to 200} looks for identical encoded
   *   content between the recipe's training and validation inputs.
   */
  analysisStart: (params, opts) => post('/api/analysis', compact(params), opts),
  /** The same request with the body sent exactly as given (keeps a null, which means "no limit"). */
  analysisStartExact: (params, opts) => post('/api/analysis', params, opts),
  analysisCancel: (id, opts) => post('/api/analysis/cancel', { id }, opts),
  /** A completed evaluation job, in the shape of `evaluations` of a run (its one observation has step -1). */
  analysisEvaluations: (id, opts) => get(`/api/analysis/evaluations${qs({ id })}`, opts),
  /** A completed evaluation job's saved images, in the shape of `images` of a run. */
  analysisImages: (id, params, opts) => get(`/api/analysis/images${qs({ id, ...params })}`, opts),
  /** A completed evaluation job's measured summaries, in the shape of `validation` of a run. */
  analysisValidation: (id, params, opts) => get(`/api/analysis/validation${qs({ id, ...params })}`, opts),
  /** Per-class results of one evaluation of a completed evaluation job, in the shape of `classes` of a run. */
  analysisClasses: (id, params, opts) => get(`/api/analysis/classes${qs({ id, ...params })}`, opts),

  // ───────── which checkpoint measured best ─────────

  /**
   * {dataset?, metric?, direction?: 'higher'|'lower', limit?: 1 to 250}. Without dataset and metric =>
   *   {runId, series: [{dataset, metric, direction, directionSource}], history}. With both => {runId,
   *   dataset, metric, direction: 'higher'|'lower'|null, directionSource: 'stored'|'evaluator'|'recipe'|
   *   'explicit'|'unknown'|'conflicting', points: [Point], observedCount, rankableCount, availableCount,
   *   unplacedLogCount, history: {source, logCoverage, logPurpose, logBytesRead, logByteLimit,
   *   unindexedLogs}, bestObserved: Point + {tieCount} | null, bestAvailable: the same | null}.
   *   Point = {step, value: number|null, nonfinite: 'nan'|'inf'|'-inf'|null, source: 'summary'|
   *   'sampleMean'|'log', checkpoint: {status: 'available'|'missing'|'invalid'|'unverified', reason?,
   *   evidence?, artifact?: {name, path, role, version}}, attempt?, evaluation?, log?, line?}.
   *   The best observed score and the best checkpoint that is still there are different answers.
   */
  checkpointSelection: (id, params, opts) => get(`/api/run/checkpoint-selection${qs({ id, ...params })}`, opts),
  /**
   * Two scores of one dataset at once: the measured checkpoints that no other beats on both.
   * {dataset, x, y, xDirection?, yDirection? ('higher'|'lower'), availableOnly?} => {runId, dataset,
   *   metrics: [x, y], availableOnly, observedCount, excluded: {reason: count}, cohorts: [{attempt, task,
   *   settings, populationId, populationVerified, directions: {metric: 'higher'|'lower'}, observedCount,
   *   frontierCount, points: [{step, evaluation, values: {metric: number}, checkpoint: {status, evidence,
   *   artifact?: {name, path, role, version, saveId}}}]}]}. A cohort is one attempt with one way of
   *   measuring; its points are the ones on the frontier.
   */
  checkpointFrontier: (id, params, opts) => get(`/api/run/checkpoint-frontier${qs({ id, ...params })}`, opts),
  /**
   * Disk used by a run's checkpoints and what its remaining saves need (read-only; sizes in bytes,
   * unknown estimates null). => {runId, policy: {source, saveEvery, targetStep, keepRegularSets,
   *   retention, format}, observed: {bytes, fileCount, complete, numberedBytes, latestBytes, otherBytes,
   *   temporaryBytes, recentCompleteSets: [{step, bytes, onCadence, files: [{role, path, bytes,
   *   validationSave?}]}], numberedStateFiles, candidatesChecked}, projection: {sampleSteps,
   *   numberedSaveBytes, atomicTemporaryFileBytes, newNumberedSaveHeadroomBytes, fromStep, fromStepSource,
   *   runStatus, scenario, remainingRegularSaves, additionalFinalNumberedSaves, remainingNumberedWriteBytes,
   *   peakAdditionalNumberedBytes, futureRetainedNumberedBytes, regularRetentionWindowBytes, assumptions:
   *   [text]}, filesystem: {availableBytes, volumes: [{directories, availableBytes}], wouldFitOneNewSet,
   *   wouldFitProjectedNumberedPeak}, limitations: [text]}
   */
  checkpointStorage: (id, opts) => get(`/api/run/checkpoint-storage${qs({ id })}`, opts),
  /**
   * The numbered checkpoints of one network of a managed run that can be averaged, newest first
   * (read-only). `params`: {before?: step, limit?: 1..250}.
   * @returns {{role, checkpoints: [{step, version, path, name, bytes}], nextBefore: number|null}}
   */
  averageCandidates: (id, role, params, opts) => get(`/api/run/average-candidates${qs({ id, role, ...params })}`, opts),
  /**
   * Every rate, data-wait percentage and memory figure a run's trainer log recorded (read-only).
   * `params`: {source?, from?, to?, against?: another run's id, againstSource?}.
   * @returns {{fromStep, toStep, base: {runId, sources: [name], source, sourceFormat, indexedBytes,
   *   sourceBytes, more, series: {[key]: {loggedPoints, firstStep, lastStep, minimum, maximum, median,
   *   last}}}, against?: same shape, matchedRecentRates?: {loggedStepsInCommon, firstStep, lastStep,
   *   medianAgainstOverBase}}}
   */
  runTiming: (id, params, opts) => get(`/api/run/timing${qs({ id, ...params })}`, opts),
  /**
   * What the model was fed: the first sample of each update the recipe asked to capture
   * (`logger.training_sample_steps`), saved after the batch was fed and before the optimizer step.
   * {limit?, after?} => {samples: [{id, attempt, candidateStep, epoch, batchInEpoch (1-based),
   *   sampleIndex (0-based), rank, task, stage, tensors: {name: {shape, dtype, preview?, previewUrl?}},
   *   rawUrl (safetensors), ...task fields such as lq_path / gt_path}], next: string|null}.
   */
  trainingSamples: (id, params, opts) => get(`/api/run/training-samples${qs({ id, ...params })}`, opts),
};

/**
 * Address of a run file (`artifact.url` when the backend gave one, otherwise built from id and path).
 * @param {string} runId
 * @param {{url?: string, path?: string}} artifact
 */
export function artifactUrl(runId, artifact) {
  if (artifact && typeof artifact.url === 'string' && artifact.url) return artifact.url;
  return `/api/run/artifact${qs({ id: runId, path: artifact?.path })}`;
}

/**
 * A gallery address (`url`, `lqUrl`, `gtUrl` from api.images) limited to `maxSize` px on its longest
 * edge. These are plain GET addresses, usable directly as <img src>. Without maxSize, `url` returns
 * the original output file; lqUrl/gtUrl are previews of at most PREVIEW_MAX px either way.
 */
export function sizedUrl(url, maxSize) {
  if (typeof url !== 'string' || !url) return '';
  if (!maxSize) return url;
  return `${url}${url.includes('?') ? '&' : '?'}maxSize=${Math.round(maxSize)}`;
}

// ───────── dataset image previews ─────────
// /api/data/preview is a POST that returns bytes, so <img src> cannot point at it directly.
// imageUrl() fetches each (path, size) once, hands back an object URL, limits parallel downloads and
// evicts the oldest entries so a long browsing session does not hold every image in memory.

const MAX_PARALLEL = 4;
const MAX_CACHED = 320;
const cache = new Map(); // "size|path" -> Promise<string>
const waiting = [];
let active = 0;

function pump() {
  while (active < MAX_PARALLEL && waiting.length) {
    const job = waiting.shift();
    active += 1;
    job().finally(() => {
      active -= 1;
      pump();
    });
  }
}

/**
 * Object URL for an image file on the trainer machine. Rejects with ApiError when it cannot be read.
 * @param {string} path a `gt` / `lq` path from a scan sample (or any path the backend accepts)
 * @param {{maxSize?: number}} [options] longest edge in px (16..1536). Use THUMB_SIZE for grids; omit
 *   for the largest preview (PREVIEW_MAX), which is still a reduced copy of a larger image.
 * @returns {Promise<string>}
 */
export function imageUrl(path, { maxSize } = {}) {
  const size = maxSize ? Math.min(PREVIEW_MAX, Math.max(16, Math.round(maxSize))) : 0;
  const key = `${size}|${path}`;
  const hit = cache.get(key);
  if (hit) {
    cache.delete(key);
    cache.set(key, hit); // refresh recency
    return hit;
  }
  const promise = new Promise((resolve, reject) => {
    waiting.push(() =>
      api.previewBlob(path, size ? { maxSize: size } : {}).then(
        (blob) => resolve(URL.createObjectURL(blob)),
        (error) => {
          cache.delete(key);
          reject(error);
        },
      ),
    );
    pump();
  });
  cache.set(key, promise);
  if (cache.size > MAX_CACHED) {
    const [oldestKey, oldest] = cache.entries().next().value;
    cache.delete(oldestKey);
    oldest.then((url) => URL.revokeObjectURL(url)).catch(() => {});
  }
  return promise;
}
