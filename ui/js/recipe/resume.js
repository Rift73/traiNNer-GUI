// What a recipe resumes from: the `summary.resume` part of a recipe check (POST /api/preflight) when the
// recipe sets path.resume_state. Pure: no requests, no state, no styles of its own.
//
//   resumeFacts(resume)    -> [[label, value], ...] for the fields the service sent
//   resumeSentence(resume) -> one sentence, '' without a resume summary
//   renderResume(resume)   -> <dl class="kv"> of the facts, or null
//   resumeRefused(issues, resume?) -> true when the check refuses the resume
//
// The summary always has {state, iteration, version, files, metadataVerified}; when the state's details were
// read (metadataVerified true) it also has {epoch, nextBatch, optimizerCount, schedulerCount,
// optimizerIdentities, randomStreams, replaySeed}. `version` is an opaque fingerprint of the file on disk
// (used to notice a change), not a format version, so it is not shown.

import { h, fmt } from '../dom.js';

/** Weight files the service pairs with the state, by recipe option (path.<key>). */
const FILE_ROLES = {
  pretrain_network_g: 'raw weights',
  pretrain_network_g_ema: 'EMA weights',
  pretrain_network_ae: 'autoencoder raw weights',
  pretrain_network_ae_ema: 'autoencoder EMA weights',
  pretrain_network_d: 'discriminator',
};

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isCount = (n) => Number.isInteger(n) && n >= 0;

/** The last part of a path on the trainer machine (either separator). */
function fileName(path) {
  const text = String(path ?? '');
  const parts = text.split(/[\\/]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : text;
}

/** [label, value, title?] for every field present; the title carries full paths. */
function factRows(resume) {
  if (!isObject(resume)) return [];
  const rows = [];
  if (isCount(resume.iteration)) rows.push(['Continues at', `iteration ${fmt.int(resume.iteration)}`]);
  if (typeof resume.state === 'string' && resume.state) rows.push(['Training state', fileName(resume.state), resume.state]);

  if (isObject(resume.files)) {
    const models = Object.entries(resume.files).filter(([key, path]) => key !== 'resume_state' && typeof path === 'string' && path);
    if (models.length) {
      const text = models.map(([key, path]) => (FILE_ROLES[key] ? `${fileName(path)} (${FILE_ROLES[key]})` : `${fileName(path)} (${key})`)).join(', ');
      rows.push(['Model files', text, models.map(([key, path]) => `${key}: ${path}`).join('\n')]);
    }
  }

  if (resume.metadataVerified === false) {
    rows.push(['Saved details', 'Not read here. The trainer checks the state when it starts.']);
    return rows;
  }

  if (isCount(resume.epoch)) rows.push(['Epoch when saved', fmt.int(resume.epoch)]);
  if ('nextBatch' in resume) {
    rows.push(['Data position', isCount(resume.nextBatch) ? `next batch: number ${fmt.int(resume.nextBatch)} of its epoch, counting from 0` : 'Not saved in this state']);
  }
  if (isCount(resume.optimizerCount)) {
    const unnamed = resume.optimizerIdentities === false && resume.optimizerCount > 0;
    rows.push(['Optimizers', unnamed ? `${fmt.int(resume.optimizerCount)}, saved without names, so the trainer matches them by position` : fmt.int(resume.optimizerCount)]);
  }
  if (isCount(resume.schedulerCount)) rows.push(['Schedulers', fmt.int(resume.schedulerCount)]);
  if (typeof resume.randomStreams === 'boolean') {
    let text;
    if (!resume.randomStreams) text = 'Not possible: the state has no saved random streams';
    else if (isCount(resume.nextBatch)) text = 'Random streams and data position are saved';
    else if ('nextBatch' in resume) text = 'Not possible: the state has no saved data position';
    else text = 'Random streams are saved';
    rows.push(['Exact replay', text]);
    if (resume.randomStreams && Number.isInteger(resume.replaySeed)) rows.push(['Saved seed', String(resume.replaySeed)]);
  }
  return rows;
}

/**
 * The resume facts in plain words, only for the fields present.
 * @param {object|null|undefined} resume  `summary.resume` of a recipe check
 * @returns {Array<[string, string]>}
 */
export function resumeFacts(resume) {
  return factRows(resume).map(([label, value]) => [label, value]);
}

/**
 * One sentence for a one-line place. '' when there is no resume summary.
 * @param {object|null|undefined} resume
 * @returns {string}
 */
export function resumeSentence(resume) {
  if (!isObject(resume)) return '';
  const from = isCount(resume.iteration) ? `Continues from iteration ${fmt.int(resume.iteration)} of a saved training state` : 'Continues from a saved training state';
  return resume.metadataVerified === false ? `${from}. The trainer checks its details when it starts.` : `${from}.`;
}

const RESUME_PATH = 'path.resume_state';

/**
 * True when the recipe check refuses the resume: an error issue at path.resume_state (or below it).
 *
 * The resume inspection also reports refusals under the recipe options they concern (scale, task,
 * network_g.type, train.optim_g.type, train.scheduler, train.schedulers, checkpoint_rng, manual_seed,
 * datasets.train, datasets.train.num_worker_per_gpu), with no code that marks them. What does mark them is
 * the order of the check (studio/preflight.py): the inspection runs only when the recipe has no error so far,
 * and it is the last step, so when `summary.resume` is present every error came from it. Pass that summary
 * as the second argument to count those errors too; with `issues` alone only the path decides.
 * @param {Array<{level?: string, path?: string}>|null|undefined} issues  `issues` of the recipe check
 * @param {object|null} [resume]  `summary.resume` of the same check
 * @returns {boolean}
 */
export function resumeRefused(issues, resume = null) {
  const errors = (Array.isArray(issues) ? issues : []).filter((issue) => issue?.level === 'error');
  const atResume = (path) => typeof path === 'string' && (path === RESUME_PATH || path.startsWith(`${RESUME_PATH}.`) || path.startsWith(`${RESUME_PATH}[`));
  if (errors.some((issue) => atResume(issue.path))) return true;
  return isObject(resume) && errors.length > 0;
}

/**
 * The facts as the app's key/value list, or null when there is nothing to show.
 * @param {object|null|undefined} resume
 * @returns {HTMLElement|null}
 */
export function renderResume(resume) {
  const rows = factRows(resume);
  if (!rows.length) return null;
  return h(
    'dl',
    { class: 'kv' },
    rows.flatMap(([label, value, title]) => [h('dt', null, label), h('dd', title ? { title } : null, value)]),
  );
}
