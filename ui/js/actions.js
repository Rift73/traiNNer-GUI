// Actions that more than one screen offers.

import { h, fmt } from './dom.js';
import { api, newRequestId } from './api.js';
import { recipe } from './state/recipe.js';
import { runs } from './state/runs.js';
import { queue } from './state/queue.js';
import { data, AUDIT_TASKS, NO_SCALE_TASKS } from './state/data.js';
import { catalog, folderWords } from './state/catalog.js';
import { badge, confirmDialog, openDialog, promptDialog, toast, countIssues, issueList } from './ui.js';
import { renderResume, resumeSentence } from './recipe/resume.js';

export function go(hash) {
  if (location.hash !== hash) location.hash = hash;
}

export const runHref = (id) => `#/runs/${encodeURIComponent(id)}`;
export const queueHref = (id) => (id ? `#/runs/queue/${encodeURIComponent(id)}` : '#/runs/queue');
export const compareHref = (left, right) => `#/runs/compare/${encodeURIComponent(left)}${right ? `/${encodeURIComponent(right)}` : ''}`;
export const analysisHref = (id) => (id ? `#/runs/analysis/${encodeURIComponent(id)}` : '#/runs/analysis');
export const studiesHref = (id) => (id ? `#/runs/studies/${encodeURIComponent(id)}` : '#/runs/studies');
export const curriculaHref = (id) => (id ? `#/runs/curricula/${encodeURIComponent(id)}` : '#/runs/curricula');

/** The open recipe, or null with a toast explaining why it cannot be used yet. */
function checkedRecipe(verb) {
  const s = recipe.get();
  if (s.phase !== 'open') {
    toast(`Open a recipe before ${verb}.`, { kind: 'err' });
    return null;
  }
  if (s.pending > 0 || s.check === 'checking' || s.check === 'idle') {
    toast('The recipe is still being checked. Try again in a moment.');
    return null;
  }
  if (s.check === 'failed') {
    toast(`The recipe could not be checked: ${s.checkError}`, { kind: 'err' });
    return null;
  }
  return s;
}

// ───────── review: what is about to train ─────────

const SET_WORD = { train: 'Training data', val: 'Validation data', test: 'Test data' };

const sameList = (a, b) => a.length === b.length && a.every((item, index) => item === b[index]);

/**
 * What the last data check says about this recipe's data, in one line. A check that found errors comes
 * back as {errors, text}, so the count can be shown as an error; everything else is a plain sentence.
 * @returns {string | {errors: number, text: string}}
 */
function dataCheckLine(taskId) {
  const d = data.get();
  if (AUDIT_TASKS.includes(taskId)) {
    const audit = d.audit;
    if (!audit) return 'Not checked. The Data screen can check it.';
    if (audit.status === 'starting' || audit.status === 'running') return 'Being checked now.';
    if (audit.status === 'failed') return 'The check did not finish.';
    if (audit.status === 'cancelled') return 'The check was cancelled.';
    if (!data.auditIsCurrent()) return "Out of date: the recipe's data settings changed after the check.";
    if (audit.errors) {
      return { errors: audit.errors, text: `found by the check${audit.warnings ? `, and ${fmt.plural(audit.warnings, 'warning')}` : ''}. The Data screen lists them.` };
    }
    return audit.warnings ? `Checked: ${fmt.plural(audit.warnings, 'warning')}.` : 'Checked, no problems found.';
  }
  // Paired folders: the folder check belongs to whatever folders were typed on the Data screen, so it
  // only counts here when those are the folders of one of this recipe's datasets.
  if (d.phase === 'scanning') return 'Being checked now.';
  if (d.phase === 'error') return 'The last folder check did not finish. The Data screen can run it again.';
  if (d.phase !== 'done' || !d.result || !d.scanned || !data.isCurrent()) return 'Not checked. The Data screen can check the folders.';
  const checked = d.scanned.options?.gtRoots ?? (d.scanned.gt ? [d.scanned.gt] : []);
  const set = recipe.datasetFolders().find((item) => item.gt.length && sameList(item.gt, checked));
  if (!set) return "Not checked: the last folder check was of other folders than this recipe's.";
  const { errors, warnings } = countIssues(d.result.issues);
  const counts = d.result.counts ?? {};
  const what = counts.paired > 0 ? fmt.plural(counts.paired, 'pair') : Number.isFinite(counts.gt) ? fmt.plural(counts.gt, 'image') : 'folders';
  const which = (SET_WORD[set.phase] ?? set.key).toLowerCase();
  if (errors) return { errors, text: `in the ${which} (${what}). The Data screen lists them.` };
  return `The ${which} was checked: ${what}${warnings ? `, ${fmt.plural(warnings, 'warning')}` : ', no problems found'}.`;
}

/**
 * The facts to look over before a recipe trains. Every value is read from the checked recipe or from
 * the last data check; a value the recipe does not hold is left out.
 * @param {object} s recipe state
 */
function reviewBlock(s) {
  const config = s.configCurrent && s.config && typeof s.config === 'object' ? s.config : null;
  const taskId = s.summary?.task ? String(s.summary.task) : '';
  const words = folderWords(taskId);
  const rows = [];
  const add = (label, value) => {
    if (value === null || value === undefined || value === '') return;
    rows.push(h('dt', null, label), h('dd', null, value));
  };
  const several = (folders) => (folders.length > 1 ? `${folders[0]} and ${folders.length - 1} more` : folders[0]);

  add('Task', taskId ? catalog.taskLabel(taskId) || taskId : null);
  add('Network', s.summary?.architecture ? String(s.summary.architecture) : null);
  if (Number.isFinite(s.summary?.scale) && !NO_SCALE_TASKS.includes(taskId)) add('Scale', `${s.summary.scale}×`);
  for (const set of recipe.datasetFolders()) {
    const parts = [];
    if (set.gt.length) parts.push(`${words.gt.short}: ${several(set.gt)}`);
    if (set.lq.length) parts.push(`${words.lq.short}: ${several(set.lq)}`);
    if (set.annotationFile) parts.push(`${words.file?.name ?? 'Annotation file'}: ${set.annotationFile}`);
    add(
      SET_WORD[set.phase] ?? set.key,
      parts.length ? h('span', { class: 'stack', style: { gap: '2px' } }, parts.map((part) => h('span', null, part))) : null,
    );
  }
  // A recipe that continues a saved training state: where it continues, and what that state holds. Only
  // for a recipe the check accepted; a refused state is in the issues, and saying "continues" would be untrue.
  const resume = s.ok && !(s.issues ?? []).some((issue) => issue.level === 'error') ? s.summary?.resume : null;
  if (resume) {
    const held = renderResume(resume);
    add(
      'Starts from',
      h('span', { class: 'stack', style: { gap: '2px' } }, h('span', null, resumeSentence(resume)), held ? h('details', null, h('summary', null, 'What the saved state holds'), held) : null),
    );
  }
  // Errors in the data are the one line here that should stop the eye before training is started.
  const check = dataCheckLine(taskId);
  add('Data check', typeof check === 'string' ? check : h('span', null, badge(fmt.plural(check.errors, 'error'), 'err'), ` ${check.text}`));
  const total = config?.train?.total_iter;
  add('Iterations', Number.isFinite(total) ? fmt.int(total) : null);
  const batch = config?.datasets?.train?.batch_size_per_gpu;
  add('Batch size', Number.isFinite(batch) ? fmt.int(batch) : null);
  return rows.length ? h('dl', { class: 'kv' }, rows) : null;
}

/**
 * Start training now with the open recipe. Errors block; warnings ask first.
 * @returns {Promise<boolean>} true when a run was started
 */
export async function startTraining() {
  const s = checkedRecipe('starting training');
  if (!s) return false;
  const { errors, warnings } = countIssues(s.issues);
  if (!s.ok || errors > 0) {
    toast(`Fix the ${errors === 1 ? 'error' : `${errors} errors`} in the recipe before starting training.`, {
      kind: 'err',
      action: { label: 'Show errors', run: () => go('#/recipe') },
    });
    return false;
  }

  const name = s.summary?.name || s.name || 'this recipe';
  const warningItems = s.issues.filter((issue) => issue.level !== 'error');
  const body = h(
    'div',
    { class: 'stack' },
    h('p', null, `Training starts now on the trainer machine with the recipe text as it is${recipe.isDirty() ? ' (it has unsaved changes)' : ''}.`),
    reviewBlock(s),
    warnings > 0 ? h('p', null, `The check reported ${warnings === 1 ? '1 warning' : `${warnings} warnings`}:`) : null,
    warnings > 0 ? issueList(warningItems) : null,
  );
  const confirmed = await confirmDialog({ title: `Start training ${name}?`, body, confirmLabel: 'Start training' });
  if (!confirmed) return false;

  // One id for this confirmed start: if the answer is lost and the person tries again from the toast,
  // the service returns the run it already started instead of starting a second one.
  const requestId = newRequestId();
  const yaml = s.yaml;
  const attempt = async () => {
    try {
      const run = await runs.start(yaml, requestId);
      // The service records the run even when the trainer ends at once (it could not be launched): the
      // answer then already says failed, and "started" would be untrue. The run's page has the reason.
      if (run?.status === 'failed') {
        toast(`Training did not start: ${run.error || 'the trainer ended at once.'}`, { kind: 'err' });
      } else {
        toast(`Started training ${run?.name ?? name}.`, { kind: 'ok' });
      }
      go(run && run.id !== undefined ? runHref(run.id) : '#/runs');
      return true;
    } catch (error) {
      if (error.status === 409) {
        toast(`Training did not start: the trainer is in use. ${error.message}`, {
          kind: 'err',
          action: { label: 'Add to queue', run: () => addToQueue() },
        });
      } else if (error.status === 422 && Array.isArray(error.issues) && error.issues.length) {
        openDialog({
          title: 'Training did not start',
          body: h('div', { class: 'stack' }, h('p', null, error.message), issueList(error.issues)),
          foot: (close) => [h('button', { class: 'btn btn--primary', type: 'button', autofocus: true, onclick: () => close() }, 'Close')],
        });
        recipe.checkNow();
      } else if (error.offline) {
        toast('Training may not have started: the trainer service did not answer.', {
          kind: 'err',
          action: { label: 'Try again', run: attempt },
        });
      } else {
        toast(`Training did not start: ${error.message}`, { kind: 'err' });
      }
      return false;
    }
  };
  return attempt();
}

/**
 * Save a copy of the open recipe in the queue. The entry is on hold unless the person chooses to
 * release it at once; released entries start one at a time when the trainer is free.
 * @returns {Promise<object|null>} the queue entry, or null when nothing was added
 */
export async function addToQueue() {
  const s = checkedRecipe('adding to the queue');
  if (!s) return null;
  const { errors, warnings } = countIssues(s.issues);
  const name = s.summary?.name || s.name || 'this recipe';
  const hasErrors = !s.ok || errors > 0;

  // The two ways to queue more than one run from this recipe. They replace this dialog with their own,
  // so they are offered in the body and the foot stays with the decision about this one entry.
  let handle = null;
  const instead = hasErrors
    ? null
    : h(
        'div',
        { class: 'stack', style: { gap: 'var(--s2)', paddingTop: 'var(--s3)', borderTop: '1px solid var(--rule)' } },
        h('p', { class: 'muted' }, 'More than one run from this recipe, each saved on hold:'),
        h(
          'div',
          { class: 'row', style: { flexWrap: 'wrap', gap: 'var(--s2)' } },
          h(
            'button',
            {
              class: 'btn btn--sm',
              type: 'button',
              title: 'Queue several named variations of this recipe, each with the same seeds, to compare them.',
              onclick: () => handle?.close('study'),
            },
            'Compare variations (a study)…',
          ),
          h(
            'button',
            {
              class: 'btn btn--sm',
              type: 'button',
              title: 'Queue this recipe as ordered phases. Each phase continues the one before it exactly, or fine-tunes from its final weights.',
              onclick: () => handle?.close('curriculum'),
            },
            'Train in phases (a curriculum)…',
          ),
        ),
      );

  const body = h(
    'div',
    { class: 'stack' },
    h('p', null, `The queue keeps a copy of the recipe text as it is now${recipe.isDirty() ? ' (it has unsaved changes)' : ''}. Later edits here do not change the queued copy.`),
    h('p', null, 'An entry on hold never starts by itself. A released entry starts when the trainer is free, after the entries released before it.'),
    reviewBlock(s),
    hasErrors
      ? h(
          'div',
          { class: 'notice notice--err', role: 'status' },
          h('div', { class: 'notice__body' }, `The check found ${errors === 1 ? '1 error' : `${errors} errors`}. The entry is checked again when it starts and fails if ${errors === 1 ? 'it is' : 'they are'} still there, so it can only be added on hold.`),
        )
      : null,
    hasErrors ? issueList(s.issues.filter((issue) => issue.level === 'error')) : null,
    !hasErrors && warnings > 0 ? h('p', null, `The check reported ${warnings === 1 ? '1 warning' : `${warnings} warnings`}:`) : null,
    !hasErrors && warnings > 0 ? issueList(s.issues.filter((issue) => issue.level !== 'error')) : null,
    instead,
  );
  handle = openDialog({
    title: `Add ${name} to the queue?`,
    body,
    foot: (close) => [
      h('button', { class: 'btn', type: 'button', onclick: () => close(null) }, 'Cancel'),
      hasErrors ? null : h('button', { class: 'btn', type: 'button', onclick: () => close('release') }, 'Add and release'),
      h('button', { class: 'btn btn--primary', type: 'button', autofocus: true, onclick: () => close('hold') }, 'Add on hold'),
    ],
  });
  const choice = await handle.closed;
  if (choice === 'study') return createStudy();
  if (choice === 'curriculum') return createCurriculum();
  if (choice !== 'hold' && choice !== 'release') return null;

  let entry;
  try {
    entry = await queue.submit(s.yaml, { recipePath: s.path });
  } catch (error) {
    toast(`The recipe was not added to the queue: ${error.message}`, { kind: 'err' });
    return null;
  }
  const open = { label: 'Open queue', run: () => go(queueHref(entry.id)) };
  if (choice === 'release') {
    try {
      entry = await queue.release(entry.id);
      toast(`Added ${entry.name ?? name} to the queue and released it.`, { kind: 'ok', action: open });
    } catch (error) {
      toast(`Added ${entry.name ?? name} to the queue on hold. It was not released: ${error.message}`, { kind: 'err', action: open });
    }
  } else {
    toast(`Added ${entry.name ?? name} to the queue on hold.`, { kind: 'ok', action: open });
  }
  return entry;
}

/**
 * Open the study dialog for the open recipe: several named variations times the same seeds, every
 * entry saved on hold. The dialog itself lives in runs/studies.js.
 * @returns {Promise<object|null>} the created study, or null
 */
export async function createStudy() {
  const s = checkedRecipe('creating a study');
  if (!s) return null;
  try {
    const { openStudyDialog } = await import('./runs/studies.js');
    return await openStudyDialog({ yaml: s.yaml, sourceLabel: 'the open recipe', name: s.summary?.name || s.name || '' });
  } catch (error) {
    toast(`The study dialog could not be opened: ${error.message}`, { kind: 'err' });
    return null;
  }
}

/**
 * Open the curriculum dialog for the open recipe: ordered phases, each one saved on hold. The dialog
 * itself lives in runs/curricula.js.
 * @returns {Promise<object|null>} the created curriculum, or null
 */
export async function createCurriculum() {
  const s = checkedRecipe('planning phases');
  if (!s) return null;
  try {
    const { openCurriculumDialog } = await import('./runs/curricula.js');
    return await openCurriculumDialog({ yaml: s.yaml, sourceLabel: 'the open recipe', name: s.summary?.name || s.name || '' });
  } catch (error) {
    toast(`The phase planner could not be opened: ${error.message}`, { kind: 'err' });
    return null;
  }
}

/**
 * Open the data audit dialog for the open recipe (classification, segmentation and detection: the
 * task's own inventory of every configured dataset). The dialog lives in runs/analysis.js.
 * @returns {Promise<object|null>} the started analysis job, or null
 */
export async function auditTaskData() {
  const s = recipe.get();
  if (s.phase !== 'open' || !s.yaml) {
    toast('Open a recipe before checking its data.', { kind: 'err' });
    return null;
  }
  try {
    const { openAnalysisDialog } = await import('./runs/analysis.js');
    return await openAnalysisDialog({ kind: 'task-data', yaml: s.yaml, sourceLabel: 'the open recipe' });
  } catch (error) {
    toast(`The data check could not be opened: ${error.message}`, { kind: 'err' });
    return null;
  }
}

/**
 * Save the open recipe, asking for a name. Never overwrites another saved recipe, or a newer version
 * of this one, without asking.
 * @returns {Promise<boolean>} true when saved
 */
export async function saveRecipe() {
  const s = recipe.get();
  if (s.phase !== 'open') return false;
  const name = await promptDialog({
    title: 'Save recipe',
    label: 'Recipe name',
    value: s.path?.startsWith('recipes/') ? s.name : s.summary?.name || s.name || '',
    help: 'Saved recipes appear under "Saved recipes" when you choose a recipe.',
    confirmLabel: 'Save recipe',
    validate: (value) => (!value ? 'Enter a name.' : /[\\/<>:"|?*]/.test(value) ? 'A name cannot contain \\ / < > : " | ? *' : null),
  });
  if (!name) return false;

  const target = recipe.savedPathFor(name);
  let overwrite = false;
  if (s.path !== target) {
    // Saving under a name this text did not come from: make sure it does not silently replace a file.
    let existing = [];
    try {
      existing = await api.recipes();
    } catch {
      // The save below reports the failure.
    }
    if (Array.isArray(existing) && existing.some((item) => item.path === target)) {
      const ok = await confirmDialog({
        title: `Replace the saved recipe ${name}?`,
        body: `A saved recipe named ${name} already exists. Saving replaces its text with the open recipe.`,
        confirmLabel: 'Replace saved recipe',
        danger: true,
      });
      if (!ok) return false;
      overwrite = true;
    }
  }

  try {
    const doc = await recipe.save(name, { overwrite });
    toast(`Saved recipe ${doc.name ?? name}.`, { kind: 'ok' });
    return true;
  } catch (error) {
    if (error.status !== 409) {
      toast(`The recipe was not saved: ${error.message}`, { kind: 'err' });
      return false;
    }
  }

  // 409: the saved file changed after this text was loaded (another window, the command line, an editor).
  const ok = await confirmDialog({
    title: `${name} changed on disk`,
    body: h(
      'div',
      { class: 'stack' },
      h('p', null, `The saved recipe ${name} was changed by something else after you opened it, so it was not saved.`),
      h('p', null, 'Overwrite it with the open recipe, or cancel and save under another name to keep both.'),
    ),
    confirmLabel: 'Overwrite saved recipe',
    danger: true,
  });
  if (!ok) return false;
  try {
    const doc = await recipe.save(name, { overwrite: true });
    toast(`Saved recipe ${doc.name ?? name}.`, { kind: 'ok' });
    return true;
  } catch (error) {
    toast(`The recipe was not saved: ${error.message}`, { kind: 'err' });
    return false;
  }
}

/**
 * Open a recipe text (from a run or a queue entry) in the Recipe screen as an unsaved copy.
 * Asks first when that would discard unsaved work.
 * @param {{yaml: string, name?: string, label: string, derived?: boolean}} source
 *   `derived`: the text was made from the open recipe's current text, so its unsaved changes are in it;
 *   nothing is discarded and nothing is asked. The caller checks that the text did not change meanwhile.
 * @returns {Promise<boolean>}
 */
export async function openAsRecipe({ yaml, name = '', label, derived = false }) {
  if (typeof yaml !== 'string' || !yaml) {
    toast('There is no recipe text to open.', { kind: 'err' });
    return false;
  }
  if (!derived && recipe.get().phase === 'open' && recipe.isDirty() && recipe.hasChanges()) {
    const ok = await confirmDialog({
      title: 'Replace the open recipe?',
      body: 'The open recipe has changes that are not saved. Opening another recipe discards them.',
      confirmLabel: 'Discard changes and open',
      danger: true,
    });
    if (!ok) return false;
  }
  recipe.load({ path: null, name, yaml }, { label, saved: false });
  go('#/recipe');
  return true;
}
