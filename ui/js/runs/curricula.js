// Curricula (#/runs/curricula[/<curriculumId>]): one recipe trained in ordered phases. Creating a curriculum
// saves every phase as an ordinary queue entry on hold; releasing is the normal queue action, and a phase
// after the first can be released only once the phase before it has completed. Each phase's changes apply
// to the base recipe on their own. The service sets train.total_iter, path.resume_state, the pretrain
// paths (after phase 1) and manual_seed (Continue) itself.
//
//   const view = mountCurriculaView(host, { curriculumId });  view.setCurriculum(id);  view.destroy();
//   const curriculum = await openCurriculumDialog({ yaml, sourceLabel, name });

import { h, mount, fmt, uid, debounce } from '../dom.js';
import { api, newRequestId } from '../api.js';
import { queue, entryStatus, ENTRY_ACTIONS } from '../state/queue.js';
import { recipe, parsePath, getAt } from '../state/recipe.js';
import { runStatus } from '../state/runs.js';
import { go, curriculaHref, queueHref, runHref, openAsRecipe, createCurriculum } from '../actions.js';
import { toast, openDialog, confirmDialog, confirmOnPhone, badge, notice, loading, emptyState, issueList } from '../ui.js';
import { createCombobox, toChoices } from '../components/combobox.js';
import { readValue } from './studies.js';

const MAX_PHASES = 32;
const SLOW = 20000;
const FAST = 3000;
const MOVING = new Set(['pending', 'running', 'stopping']);
const REVISABLE = new Set(['held', 'cancelled', 'stopped', 'failed']);
const PHASE_WEIGHT_FIELDS = [
  'pretrain_network_g',
  'pretrain_network_g_ema',
  'pretrain_network_d',
  'pretrain_network_ae',
  'pretrain_network_ae_ema',
  'pretrain_network_ae_decoder',
  'pretrain_network_ae_decoder_ema',
];

const INTRO = 'A curriculum trains one recipe in ordered phases.';
const HOLD_TEXT = 'Every phase is saved on hold. Nothing trains until you release a phase.';
const OWN_TEXT = "Each phase's changes apply to the base recipe on their own, not on top of the phase before.";
const ITER_TEXT = "The curriculum sets the number of iterations itself: the recipe's own train.total_iter is replaced for each phase.";
const NO_CHANGES = "No changes: the base recipe's settings";
const HOW_TEXT = 'On the Recipe screen, choose Add to queue, then Train in phases.';
const MODE_LABEL = { start: 'Start', continue: 'Continue', finetune: 'Fine-tune' };
// Each sentence continues its mode label: "Start: begins at iteration 0."
const MODE_TEXT = {
  start: 'begins at iteration 0.',
  continue:
    "restores the previous phase's weights, optimizer, scheduler and iteration count, so optimizer and schedule settings cannot change here (use Fine-tune for that); data and loss settings can.",
  finetune: 'starts a new count at iteration 0 from weights of the previous phase.',
};
const WEIGHTS_LABEL = { raw: 'Raw weights', ema: 'EMA weights' };
const RESTORED_WARNING =
  'Continue restores this setting from the previous phase. The service refuses such a change when a curriculum is created, but it does not check revisions, so this copy may be refused or fail when it is released.';
const RESTORED_HINT = 'Continue restores this from the previous phase; use Fine-tune to change it.';
// Same wording as the queue view's confirmations.
const CANCEL_TEXT = 'Cancelling cannot be undone. The entry will never start; it stays in the finished list, where its recipe text can still be opened.';
const STOP_TEXT = 'The trainer is asked to stop this training run now. The entry does not start again by itself.';

const isAbort = (error) => error?.name === 'AbortError';
const shortDigest = (digest) => (typeof digest === 'string' ? digest.slice(0, 12) : '');
const modeLabel = (mode) => MODE_LABEL[mode] ?? String(mode ?? 'Not reported');

const BEST_LABEL = 'The best-scoring weights on a validation dataset';
const BEST_HELP = "Uses the weights the earlier phase kept as best on this dataset. That phase's recipe must keep its best weights (Validation, 'Keep the best weights by').";

/** The validation dataset of a fine-tune that starts from kept best weights ({bestValidation: name}), else null. */
const bestDataset = (weights) => (weights !== null && typeof weights === 'object' && typeof weights.bestValidation === 'string' ? weights.bestValidation : null);

/**
 * The service's preview of what a phase's recipe keeps as best weights, reduced to what the planner
 * words: {bestMetric: null | string, datasets: string[]}. null when the answer has another shape.
 */
function readPhasePreview(answer) {
  if (answer === null || typeof answer !== 'object') return null;
  const { bestMetric, bestValidationDatasets } = answer;
  if (bestMetric !== null && !(typeof bestMetric === 'string' && bestMetric)) return null;
  if (!Array.isArray(bestValidationDatasets)) return null;
  return { bestMetric, datasets: bestValidationDatasets.filter((name) => typeof name === 'string' && name) };
}

/**
 * The lines under "Validation dataset name" for a preview of phase `n`, the phase before the fine-tune.
 * @returns {{kind: 'warn'|'quiet', text: string}[]}
 */
function phasePreviewWords(n, preview, typed) {
  const { bestMetric, datasets } = preview;
  if (bestMetric === null) {
    return [{ kind: 'warn', text: `Phase ${n}'s recipe does not keep its best weights, so the service will refuse this choice. Set 'Keep the best weights by' in the base recipe, or add it to phase ${n} as a change.` }];
  }
  if (!datasets.length) return [{ kind: 'warn', text: `Phase ${n} keeps its best weights by ${bestMetric}, but none of its validation datasets is scored.` }];
  const lines = [{ kind: 'quiet', text: `Phase ${n} keeps its best weights by ${bestMetric} for: ${datasets.join(', ')}.` }];
  if (typed && !datasets.includes(typed)) lines.push({ kind: 'quiet', text: `${typed} is not one of them.` });
  return lines;
}

/** "Starts from the best weights on Val A." */
const bestLine = (weights) => `Starts from the best weights on ${bestDataset(weights)}.`;

/** The sentence after the mode label. Best weights are not the final ones, so that case names only the new count. */
const modeText = (mode, weights) => (mode === 'finetune' && bestDataset(weights) !== null ? 'starts a new count at iteration 0.' : MODE_TEXT[mode]);

/** "Fine-tune, EMA weights" (the best weights of a dataset are told by bestLine instead) */
const modeHead = (mode, weights) => `${modeLabel(mode)}${mode === 'finetune' && typeof weights === 'string' && weights ? `, ${WEIGHTS_LABEL[weights] ?? weights}` : ''}`;

/** "Fine-tune, EMA weights: starts a new count …" */
const modeLine = (mode, weights) => {
  const text = modeText(mode, weights);
  const line = text ? `${modeHead(mode, weights)}: ${text}` : modeHead(mode, weights);
  return mode === 'finetune' && bestDataset(weights) !== null ? `${line} ${bestLine(weights)}` : line;
};

/**
 * Settings a Continue phase restores from the phase before (the service's Continue compatibility rule):
 * train.optim_*, train.scheduler, train.warmup_iter, train.ema_*, network_*, scale and task.
 */
function continueRestores(path) {
  if (!Array.isArray(path) || !path.length) return false;
  const [top, second] = path;
  if (top === 'scale' || top === 'task') return true;
  if (typeof top === 'string' && top.startsWith('network_')) return true;
  if (top !== 'train' || typeof second !== 'string') return false;
  return second.startsWith('optim_') || second === 'scheduler' || second === 'warmup_iter' || second.startsWith('ema_');
}

/** A value in its JSON form, shortened in the middle past 60 characters. */
function shortJson(value) {
  const text = JSON.stringify(value) ?? String(value);
  return text.length > 60 ? `${text.slice(0, 29)}…${text.slice(-29)}` : text;
}

/** Every leaf of a parsed recipe as {path, text}; a list of plain values is offered whole as well. */
function flattenOptions(config) {
  const out = [];
  const walk = (node, path) => {
    if (Array.isArray(node)) {
      if (path.length && node.every((item) => item === null || typeof item !== 'object')) out.push(path);
      if (!node.length && path.length) return;
      node.forEach((item, i) => walk(item, [...path, i]));
    } else if (node !== null && typeof node === 'object') {
      const keys = Object.keys(node);
      if (!keys.length && path.length) out.push(path);
      for (const key of keys) walk(node[key], [...path, key]);
    } else if (path.length) {
      out.push(path);
    }
  };
  walk(config, []);
  return out.map((path) => ({ path, text: dotted(path) }));
}

/**
 * Option path help for one dialog: the base recipe's options, read once with preflight, as the list
 * every path field of the dialog offers (`paths`), and the base value of a typed path. Without an
 * answer the fields work without choices.
 * @param {string} baseYaml
 * @param {(path: Array) => boolean} excluded paths the curriculum sets itself
 * @param {() => void} onReady
 */
function optionHelp(baseYaml, excluded, onReady) {
  const help = { paths: [], config: null, ctl: new AbortController() };
  if (typeof baseYaml === 'string' && baseYaml) {
    api
      .preflight(baseYaml, { signal: help.ctl.signal })
      .then((answer) => {
        const config = answer?.config;
        if (!config || typeof config !== 'object' || Array.isArray(config)) return;
        help.config = config;
        const seen = new Set();
        const paths = [];
        for (const { path, text } of flattenOptions(config)) {
          if (excluded(path) || seen.has(text)) continue;
          seen.add(text);
          paths.push(text);
        }
        help.paths = paths;
        onReady();
      })
      .catch(() => {});
  }
  return help;
}

/** "Base recipe: 0.0002" / "Not set in the base recipe" for a typed path, or '' without a parsed recipe. */
function baseValueWords(help, path) {
  if (!help?.config || !path.length) return '';
  const value = getAt(help.config, path);
  return value === undefined ? 'Not set in the base recipe' : `Base recipe: ${shortJson(value)}`;
}

function timeEl(iso) {
  const text = fmt.dateTime(iso);
  return text ? h('time', { datetime: iso, title: iso }, text) : 'Not reported';
}

/** ['train', 'optim_g', 'lr'] -> "train.optim_g.lr"; list indices as [i]. */
function dotted(path) {
  if (!Array.isArray(path)) return String(path ?? '');
  let out = '';
  for (const part of path) out += typeof part === 'number' ? `[${part}]` : out ? `.${part}` : String(part);
  return out;
}

function valueText(value) {
  if (typeof value === 'string') return value === '' ? '""' : value;
  return JSON.stringify(value) ?? String(value);
}

/** Text for a value field that readValue reads back as the same value. */
function valueInput(value) {
  if (typeof value !== 'string') return JSON.stringify(value) ?? '';
  if (value === '' || value.trim() !== value) return JSON.stringify(value);
  try {
    JSON.parse(value);
    return JSON.stringify(value); // "true", "1e-4" and the like stay text
  } catch {
    return value;
  }
}

function statusBadge(status) {
  const st = entryStatus(status);
  return badge(st.label, st.kind);
}

/** "Iterations 0 to 4", "Iterations 0 to 2 (new count)". */
function clockWords(mode, start, target) {
  if (!Number.isFinite(start) || !Number.isFinite(target)) return 'Iterations not reported';
  return `Iterations ${fmt.int(start)} to ${fmt.int(target)}${mode === 'finetune' ? ' (new count)' : ''}`;
}

/** One phase of the schedule sentence. */
function scheduleWords(n, mode, start, target) {
  if (mode === 'continue') return `Phase ${n} continues ${fmt.int(start)} to ${fmt.int(target)}.`;
  if (mode === 'finetune') return `Phase ${n} starts a new count, ${fmt.int(start)} to ${fmt.int(target)}.`;
  return `Phase ${n} trains iterations ${fmt.int(start)} to ${fmt.int(target)}.`;
}

const validUpdates = (n) => Number.isSafeInteger(n) && n > 0;

/**
 * The service's clock rule. start: 0 to updates; continue: previous target to previous target + updates;
 * fine-tune: 0 to updates. A phase whose clock cannot be known yet is null.
 * @param {Array<{mode: string, updates: number|null}>} phases
 */
function phaseClock(phases) {
  let clock = 0;
  let known = true;
  return phases.map((phase) => {
    const ok = validUpdates(phase.updates);
    if (phase.mode === 'continue') {
      if (!ok || !known) {
        known = false;
        return null;
      }
      const startStep = clock;
      clock = startStep + phase.updates;
      return { startStep, targetStep: clock };
    }
    if (!ok) {
      known = false;
      return null;
    }
    known = true;
    clock = phase.updates;
    return { startStep: 0, targetStep: clock };
  });
}

/** The options the service sets itself for a phase of this mode (prefix match, as the service does). */
function reservedFields(mode) {
  const fields = [['train', 'total_iter'], ['path', 'resume_state']];
  if (mode !== 'start') for (const field of PHASE_WEIGHT_FIELDS) fields.push(['path', field]);
  if (mode === 'continue') fields.push(['manual_seed']);
  return fields;
}

/** The service's own refusal for an edit it would reject before rendering, or ''. */
function editProblem(mode, edit) {
  const path = Array.isArray(edit?.path) ? edit.path : [];
  for (const field of reservedFields(mode)) {
    if (field.every((key, i) => path[i] === key)) return `The curriculum sets ${field.join('.')} for this phase.`;
  }
  const setsValue = !edit.remove && edit.value !== null && edit.value !== undefined;
  if (path.length === 2 && path[0] === 'path' && (path[1] === 'ignore_resume_networks' || path[1] === 'results_root') && setsValue) {
    return `Clear path.${path[1]} in the curriculum recipe.`;
  }
  if (mode === 'continue' && path.length === 2 && path[0] === 'train' && path[1] === 'restart_scheduler' && !edit.remove && edit.value) {
    return 'Continue restores the source scheduler. Clear train.restart_scheduler.';
  }
  return '';
}

/** A phase's changes as "dotted.path = value" / "dotted.path removed" lines. */
function editsView(edits) {
  if (!Array.isArray(edits) || !edits.length) return h('p', { class: 'curricula__muted' }, `${NO_CHANGES}.`);
  return h(
    'ul',
    { class: 'curricula__changes' },
    edits.map((edit) => h('li', { class: 'curricula__change' }, edit?.remove ? `${dotted(edit.path)} removed` : `${dotted(edit?.path)} = ${valueText(edit?.value)}`)),
  );
}

/** Problems with a phase or curriculum name, in the order the service checks them. */
function nameProblem(text, what) {
  if (!text.trim()) return `Name ${what}.`;
  if (text.length > 120) return `A name has at most 120 characters; this one has ${fmt.int(text.length)}.`;
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f]/.test(text)) return 'A name cannot contain control characters such as tabs or line breaks.';
  return '';
}

// ───────── recipe check of one phase (create and revise dialogs) ─────────
// Creating a curriculum only renders each phase and applies the phase rules; it does not run the recipe
// check. So each phase is checked here the way the Recipe screen checks a recipe: render the base recipe
// with the phase's changes, then preflight the result. Neither call changes anything on the service. This
// approximates what is launched: the service also sets the iteration count and the handoff paths.

/** => {errors: issue[], refusal?: string, failed?: string, notPassed?: boolean}; rejects only on abort. */
async function checkPhaseRecipe(baseYaml, edits, signal) {
  let text = baseYaml;
  if (edits.length) {
    try {
      const rendered = await api.render(baseYaml, edits, { signal });
      text = rendered?.yaml;
      if (typeof text !== 'string') return { errors: [], failed: 'the service answered without recipe text' };
    } catch (error) {
      if (isAbort(error)) throw error;
      if (error?.offline) return { errors: [], failed: error.message };
      return { errors: [], refusal: error?.message ?? 'no answer' };
    }
  }
  try {
    const answer = await api.preflight(text, { signal });
    const errors = (Array.isArray(answer?.issues) ? answer.issues : []).filter((issue) => issue?.level === 'error');
    return { errors, notPassed: answer?.ok === false && !errors.length };
  } catch (error) {
    if (isAbort(error)) throw error;
    return { errors: [], failed: error?.message ?? 'no answer' };
  }
}

const checkFails = (result) => Boolean(result && (result.refusal || result.errors.length || result.notPassed));

/** What to show under a phase for its check result: nothing when it passes. */
function checkView(result) {
  if (!result) return null;
  if (result.refusal) return notice('err', [h('p', { class: 'curricula__issue' }, 'A change cannot be applied to the base recipe:'), h('p', { class: 'curricula__issue' }, result.refusal)]);
  if (result.errors.length) {
    return notice('err', [
      h('p', { class: 'curricula__issue' }, `This phase's recipe has ${fmt.plural(result.errors.length, 'error')}. Saved like this, the phase would fail when it starts.`),
      issueList(result.errors),
    ]);
  }
  if (result.notPassed) return notice('err', "This phase's recipe did not pass the check. Saved like this, the phase would fail when it starts.");
  if (result.failed) return notice('warn', `This phase's recipe could not be checked: ${result.failed}`);
  return null;
}

/** A result that is missing (aborted, not run) or could not be had counts as not checked: it must be asked about. */
const checkMissing = (result) => !result || Boolean(result.failed);

/** "Phase 2's recipe could not be checked.", "The recipes of phases 2 and 3 could not be checked." */
function uncheckedWords(numbers) {
  if (numbers.length === 1) return `Phase ${numbers[0]}'s recipe could not be checked.`;
  return `The recipes of phases ${numbers.slice(0, -1).join(', ')} and ${numbers[numbers.length - 1]} could not be checked.`;
}

/** JSON with sorted keys, so the same request body always gives the same text. */
function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value)
      .filter((key) => value[key] !== undefined)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`)
      .join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

/**
 * One request id per request content: the id is kept while the body (without the id) is the same as the
 * body last sent, so "Try again" after a lost answer repeats that request; any edit makes a new request.
 */
function requestIds() {
  let last = null; // {key, requestId}
  return (body) => {
    const key = stableJson(body);
    if (last?.key !== key) last = { key, requestId: newRequestId() };
    return last.requestId;
  };
}

/**
 * Keep a dialog from closing while a save is in flight. openDialog (ui.js) has no hook for this, so the
 * <dialog> it returns is used: its close() is wrapped (Close, ✕ and the backdrop all go through it), its
 * cancel event (Escape) is prevented, the close buttons are disabled and the fields set inert.
 * @returns {(on: boolean) => void}
 */
function guardDialog(dialog, closeButtons, fieldsets) {
  let locked = false;
  let refocus = null;
  const nativeClose = dialog.close.bind(dialog);
  dialog.close = (...args) => {
    if (!locked) nativeClose(...args);
  };
  dialog.addEventListener('cancel', (event) => {
    if (locked) event.preventDefault();
  });
  return (on) => {
    if (on === locked) return;
    locked = on;
    const sets = fieldsets.filter(Boolean);
    if (on) refocus = sets.some((set) => set.contains(document.activeElement)) ? document.activeElement : null;
    for (const button of closeButtons()) if (button) button.disabled = on;
    for (const set of sets) set.disabled = on;
    if (on) dialog.setAttribute('aria-busy', 'true');
    else dialog.removeAttribute('aria-busy');
    if (!on && refocus?.isConnected && (!document.activeElement || document.activeElement === document.body)) refocus.focus();
  };
}

/** "Phase 2's recipe has errors", "Phases 2 and 3 have recipe errors". */
function failingWords(numbers) {
  if (numbers.length === 1) return `Phase ${numbers[0]}'s recipe has errors and would fail when it starts.`;
  const list = `${numbers.slice(0, -1).join(', ')} and ${numbers[numbers.length - 1]}`;
  return `Phases ${list} have recipe errors and would fail when they start.`;
}

// ───────── change list editor (create and revise dialogs) ─────────

/**
 * Rows of option path + value (or "remove this option").
 * help: the dialog's optionHelp (suggestions and base values). restored: how a change to a setting that
 * Continue restores is flagged on a Continue phase: 'notice' (revise dialog) or 'hint' (planner).
 * @param {{initial?: Array, getMode: () => string, onChange: () => void, help?: object, restored?: 'notice'|'hint'}} options
 */
function createEditor({ initial = [], getMode, onChange, help = null, restored = 'hint' }) {
  const rows = [];
  const list = h('ul', { class: 'curricula__edit-list' });
  const none = h('p', { class: 'curricula__muted' }, `${NO_CHANGES}.`);
  const addBtn = h(
    'button',
    {
      class: 'btn btn--sm',
      type: 'button',
      onclick: () => {
        const row = addRow();
        onChange();
        row.path.focus();
      },
    },
    'Add a change',
  );

  function addRow(init = {}) {
    const pathId = uid('curricula-edit-path');
    const valueId = uid('curricula-edit-value');
    const removeId = uid('curricula-edit-remove');
    const row = {};
    // The base recipe's options are offered once they are read; any other path can be typed.
    const pathCombo = createCombobox({
      id: pathId,
      inputClass: 'mono',
      placeholder: 'train.optim_g.lr',
      listLabel: 'Options of the base recipe',
      getChoices: () => toChoices(help?.paths ?? []),
      attrs: { 'aria-describedby': `${pathId}-error ${pathId}-restored` },
      onPick: () => onChange(),
    });
    row.path = pathCombo.input;
    row.path.value = init.path ?? '';
    row.path.addEventListener('input', () => onChange());
    row.value = h('input', { class: 'input mono', id: valueId, type: 'text', value: init.value ?? '', autocomplete: 'off', spellcheck: 'false', 'aria-describedby': `${valueId}-read ${valueId}-base`, oninput: () => onChange() });
    row.remove = h('input', { type: 'checkbox', id: removeId, checked: Boolean(init.remove), onchange: () => onChange() });
    row.reading = h('span', { class: 'curricula__reading', id: `${valueId}-read` });
    row.base = h('span', { class: 'curricula__reading', id: `${valueId}-base` });
    row.error = h('p', { class: 'curricula__row-error', id: `${pathId}-error`, 'aria-live': 'polite' });
    row.restored = h('div', { class: 'curricula__row-note', id: `${pathId}-restored`, 'aria-live': 'polite' });
    row.restoredOn = false;
    row.el = h(
      'li',
      { class: 'curricula__edit' },
      h('div', { class: 'curricula__edit-field' }, h('label', { class: 'field__label', for: pathId }, 'Option path'), pathCombo.el),
      h('div', { class: 'curricula__edit-field' }, h('label', { class: 'field__label', for: valueId }, 'New value'), row.value, row.reading, row.base),
      h('label', { class: 'curricula__check curricula__edit-end', for: removeId }, row.remove, 'Remove this option'),
      h(
        'button',
        {
          class: 'btn btn--quiet btn--sm curricula__edit-end',
          type: 'button',
          onclick: () => {
            rows.splice(rows.indexOf(row), 1);
            row.el.remove();
            onChange();
            addBtn.focus();
          },
        },
        'Remove change',
      ),
      row.error,
      row.restored,
    );
    rows.push(row);
    list.appendChild(row.el);
    return row;
  }

  for (const edit of Array.isArray(initial) ? initial : []) {
    addRow(edit?.remove ? { path: dotted(edit.path), remove: true } : { path: dotted(edit?.path), value: valueInput(edit?.value) });
  }

  /** => {edits, problems}; reserved options are refused as soon as they are typed, the rest on submit. */
  function collect(showErrors) {
    const mode = getMode();
    const edits = [];
    const problems = [];
    none.hidden = rows.length > 0;
    for (const row of rows) {
      const path = parsePath(row.path.value.trim());
      const removing = row.remove.checked;
      row.value.disabled = removing;
      const reading = removing ? { ok: true, words: 'This option is removed from the recipe.' } : readValue(row.value.value);
      row.reading.textContent = row.value.value || removing ? reading.words : '';
      const edit = removing ? { path, remove: true } : { path, value: reading.value };
      const reserved = path.length && reading.ok ? editProblem(mode, edit) : '';
      let problem = reserved;
      if (!path.length) problem = 'Enter an option path for every change, for example train.optim_g.lr.';
      else if (!reading.ok) problem = reading.words;
      const shown = reserved || (showErrors ? problem : '');
      if (row.error.textContent !== shown) row.error.textContent = shown;
      if (shown) row.path.setAttribute('aria-invalid', 'true');
      else row.path.removeAttribute('aria-invalid');
      const base = baseValueWords(help, path);
      if (row.base.textContent !== base) row.base.textContent = base;
      // Not a blocker: on a Continue phase, flag a change to a setting Continue restores.
      const restoredNow = mode === 'continue' && !reserved && continueRestores(path);
      if (restoredNow !== row.restoredOn) {
        row.restoredOn = restoredNow;
        if (!restoredNow) mount(row.restored);
        else if (restored === 'notice') mount(row.restored, notice('warn', RESTORED_WARNING));
        else mount(row.restored, h('p', { class: 'curricula__row-hint' }, RESTORED_HINT));
      }
      if (problem) problems.push(problem);
      edits.push(edit);
    }
    return { edits, problems };
  }

  const el = h('div', { class: 'curricula__editor' }, none, list, h('div', { class: 'row' }, addBtn));
  return { el, collect, focus: () => (rows[0]?.path ?? addBtn).focus() };
}

/** A two-button choice (aria-pressed), as in the shared .seg. */
function segChoice(label, options, value, onChange) {
  const buttons = new Map();
  const el = h(
    'div',
    { class: 'seg', role: 'group', 'aria-label': label },
    options.map(([key, text]) => {
      const button = h('button', { class: 'btn btn--sm', type: 'button', 'aria-pressed': String(key === value), onclick: () => onChange(key) }, text);
      buttons.set(key, button);
      return button;
    }),
  );
  return {
    el,
    set(key) {
      for (const [k, button] of buttons) button.setAttribute('aria-pressed', String(k === key));
    },
    button: (key) => buttons.get(key),
  };
}

// ───────── create dialog ─────────

/**
 * Plan a curriculum from a recipe text. Resolves to the created curriculum detail, or null when closed
 * without creating. Opens the curriculum after creating it.
 * @param {{yaml?: string, sourceLabel?: string, name?: string}} options
 * @returns {Promise<object|null>}
 */
export function openCurriculumDialog({ yaml = '', sourceLabel = '', name = '' } = {}) {
  const requestIdFor = requestIds(); // the same id only for the same content
  const nameId = uid('curricula-name');
  const nameInput = h('input', { class: 'input', id: nameId, type: 'text', value: name, autocomplete: 'off', spellcheck: 'false', maxlength: '120' });
  const nameError = h('p', { class: 'field__error', role: 'alert' });
  const phaseList = h('ol', { class: 'curricula__plan', 'aria-label': 'Phases in training order' });
  const schedule = h('p', { class: 'curricula__schedule', role: 'status', 'aria-live': 'polite' });
  const limitLine = h('p', { class: 'curricula__muted' });
  const footStatus = h('div', { class: 'curricula__issues', 'aria-live': 'polite' });
  const phases = []; // in order; see addPhase
  let busy = false;
  let created = null;
  let sent = null; // the phases as they were sent, for placing issues by phaseId
  // Base recipe options for the path inputs. The paths the curriculum sets for any mode are not offered.
  const help = optionHelp(yaml, (path) => ['start', 'continue', 'finetune'].some((mode) => reservedFields(mode).some((field) => field.every((key, i) => path[i] === key))), () => collect(false));

  function addPhase({ mode = 'continue' } = {}) {
    const n = phases.length + 1;
    const ids = { name: uid('curricula-phase-name'), iter: uid('curricula-phase-iter'), best: uid('curricula-phase-best') };
    const phase = { mode: mode === 'finetune' ? 'finetune' : 'continue', weights: 'raw' };
    phase.title = h('h3', { class: 'curricula__plan-title' });
    phase.nameInput = h('input', { class: 'input', id: ids.name, type: 'text', value: `Phase ${n}`, autocomplete: 'off', spellcheck: 'false', maxlength: '120', oninput: () => update(phase) });
    phase.nameError = h('p', { class: 'field__error', role: 'alert' });
    phase.startLine = h('p', { class: 'curricula__muted' }, `${MODE_LABEL.start}: ${MODE_TEXT.start}`);
    phase.modeSeg = segChoice('How this phase begins', [['continue', MODE_LABEL.continue], ['finetune', MODE_LABEL.finetune]], phase.mode, (key) => {
      phase.mode = key;
      update(phase);
    });
    phase.modeSet = h(
      'div',
      { class: 'curricula__choice' },
      h('span', { class: 'field__label' }, 'How this phase begins'),
      phase.modeSeg.el,
      h(
        'dl',
        { class: 'curricula__modes' },
        h('dt', null, MODE_LABEL.continue),
        h('dd', null, MODE_TEXT.continue),
        h('dt', null, MODE_LABEL.finetune),
        h('dd', null, MODE_TEXT.finetune),
      ),
    );
    // 'best' is this form's own key: it is sent as {bestValidation: <the name typed below>}.
    phase.weightsSeg = segChoice('Weights to fine-tune from', [['raw', WEIGHTS_LABEL.raw], ['ema', WEIGHTS_LABEL.ema], ['best', BEST_LABEL]], phase.weights, (key) => {
      phase.weights = key;
      update(phase);
    });
    phase.weightsSet = h('div', { class: 'curricula__choice' }, h('span', { class: 'field__label' }, "Which weights of the previous phase"), phase.weightsSeg.el);
    // The name is typed or chosen. When the service previews the previous phase's recipe, its scored
    // validation datasets are the choices and a line below says what that phase keeps (syncPreviews).
    phase.bestNames = [];
    const bestCombo = createCombobox({
      id: ids.best,
      listLabel: 'Validation datasets of the previous phase',
      getChoices: () => toChoices(phase.bestNames),
      attrs: { 'aria-describedby': `${ids.best}-help` },
      onPick: () => update(phase),
    });
    phase.bestInput = bestCombo.input;
    phase.bestInput.addEventListener('input', () => update(phase));
    phase.bestStatus = h('div', { class: 'curricula__issues', 'aria-live': 'polite' });
    // key: the previous phase's changes the preview is for (null: none wanted); shown: what is drawn.
    phase.preview = { chosen: false, key: null, answer: null, ctl: null, timer: 0, shown: '[]' };
    phase.bestError = h('p', { class: 'field__error', role: 'alert' });
    phase.bestSet = h(
      'div',
      { class: 'field', hidden: true },
      h('label', { class: 'field__label', for: ids.best }, 'Validation dataset name'),
      bestCombo.el,
      h('p', { class: 'field__help', id: `${ids.best}-help` }, BEST_HELP),
      phase.bestStatus,
      phase.bestError,
    );
    phase.iterInput = h('input', { class: 'input input--num curricula__iter', id: ids.iter, type: 'number', min: '1', step: '1', inputmode: 'numeric', 'aria-describedby': `${ids.iter}-clock`, oninput: () => update(phase) });
    phase.iterClock = h('p', { class: 'curricula__iter-clock', id: `${ids.iter}-clock` });
    phase.iterError = h('p', { class: 'field__error', role: 'alert' });
    phase.ownHelp = h('p', { class: 'field__help' });
    phase.editor = createEditor({ getMode: () => modeAt(phase), onChange: () => update(phase), help, restored: 'hint' });
    phase.refused = false; // a service refusal is shown in phase.issues
    phase.issues = h('div', { class: 'curricula__issues', 'aria-live': 'polite' });
    phase.checkSlot = h('div', { class: 'curricula__issues', 'aria-live': 'polite' });
    phase.checked = null; // {key, result} of the last completed recipe check
    phase.current = { edits: [], ok: true }; // the changes as last read by collect()
    phase.upBtn = h('button', { class: 'btn btn--quiet btn--sm', type: 'button', onclick: () => move(phase, -1) }, 'Move up');
    phase.downBtn = h('button', { class: 'btn btn--quiet btn--sm', type: 'button', onclick: () => move(phase, 1) }, 'Move down');
    phase.removeBtn = h('button', { class: 'btn btn--quiet btn--sm', type: 'button', onclick: () => removePhase(phase) }, 'Remove phase');
    phase.el = h(
      'li',
      { class: 'curricula__plan-phase' },
      h('div', { class: 'curricula__plan-head' }, phase.title, h('div', { class: 'row row--wrap' }, phase.upBtn, phase.downBtn, phase.removeBtn)),
      h('div', { class: 'field' }, h('label', { class: 'field__label', for: ids.name }, 'Phase name'), phase.nameInput, phase.nameError),
      phase.startLine,
      phase.modeSet,
      phase.weightsSet,
      phase.bestSet,
      h('div', { class: 'field' }, h('label', { class: 'field__label', for: ids.iter }, 'Iterations'), h('div', { class: 'curricula__iter-row' }, phase.iterInput, phase.iterClock), phase.iterError),
      h('div', { class: 'curricula__plan-changes' }, h('span', { class: 'field__label' }, 'Changes'), phase.ownHelp, phase.editor.el),
      phase.checkSlot,
      phase.issues,
    );
    phases.push(phase);
    phaseList.appendChild(phase.el);
    return phase;
  }

  const modeAt = (phase) => (phases.indexOf(phase) === 0 ? 'start' : phase.mode);

  function move(phase, delta) {
    const from = phases.indexOf(phase);
    const to = from + delta;
    if (from < 0 || to < 0 || to >= phases.length) return;
    const first = phases[0];
    phases.splice(from, 1);
    phases.splice(to, 0, phase);
    if (phases[0] !== first) first.mode = 'continue'; // a phase moved out of first place becomes Continue
    for (const p of phases) phaseList.appendChild(p.el);
    clearRefusals(); // phase numbers and predecessors changed
    update();
    const button = delta < 0 ? phase.upBtn : phase.downBtn;
    (button.disabled ? (delta < 0 ? phase.downBtn : phase.upBtn) : button).focus();
  }

  function removePhase(phase) {
    const index = phases.indexOf(phase);
    if (index < 0 || phases.length <= 1) return;
    phases.splice(index, 1);
    phase.el.remove();
    stopPreview(phase);
    clearRefusals();
    update();
    (phases[index] ?? phases[index - 1])?.nameInput.focus();
  }

  const addPhaseBtn = h(
    'button',
    {
      class: 'btn btn--sm',
      type: 'button',
      onclick: () => {
        if (phases.length >= MAX_PHASES) return;
        const phase = addPhase();
        clearFoot(); // phase numbers in the foot summary no longer hold
        update();
        phase.nameInput.focus();
      },
    },
    'Add a phase',
  );

  /** Validate everything and refresh the live parts; returns the request body or null. */
  function collect(showErrors) {
    let ok = true;
    const curriculumName = nameInput.value.trim();
    const nameText = nameProblem(nameInput.value, 'the curriculum');
    nameError.textContent = showErrors || nameInput.value ? nameText : '';
    if (nameText) ok = false;

    const seen = new Set();
    const body = [];
    const plan = phases.map((phase) => {
      const raw = phase.iterInput.value.trim();
      return { mode: modeAt(phase), updates: /^\d+$/.test(raw) ? Number(raw) : null };
    });
    const clocks = phaseClock(plan);
    phases.forEach((phase, index) => {
      const n = index + 1;
      const mode = plan[index].mode;
      if (phase.title.textContent !== `Phase ${n}`) phase.title.textContent = `Phase ${n}`;
      phase.startLine.hidden = index !== 0;
      phase.modeSet.hidden = index === 0;
      phase.modeSeg.set(phase.mode);
      phase.weightsSet.hidden = mode !== 'finetune';
      phase.weightsSeg.set(phase.weights);
      const best = mode === 'finetune' && phase.weights === 'best';
      phase.bestSet.hidden = !best;
      phase.upBtn.disabled = index === 0;
      phase.downBtn.disabled = index === phases.length - 1;
      phase.removeBtn.hidden = phases.length <= 1;
      phase.ownHelp.textContent =
        index === 0
          ? 'Applied to the base recipe.'
          : `Applied to the base recipe on their own, not on top of phase ${index}.${mode === 'continue' ? ` So a Continue phase repeats any optimizer or schedule change phase ${index} made.` : ''}`;

      const phaseName = phase.nameInput.value.trim();
      let nameText = nameProblem(phase.nameInput.value, 'this phase');
      if (!nameText && seen.has(phaseName.toLowerCase())) nameText = `The phase name ${phaseName} is used twice. Phase names must differ, ignoring capitals.`;
      if (phaseName) seen.add(phaseName.toLowerCase());
      phase.nameError.textContent = showErrors || nameText.includes('twice') ? nameText : '';
      if (nameText) ok = false;

      const updates = plan[index].updates;
      const raw = phase.iterInput.value.trim();
      let iterText = '';
      if (!validUpdates(updates)) iterText = raw ? 'Enter the iterations as a whole number, 1 or more.' : 'Enter the number of iterations this phase trains.';
      phase.iterError.textContent = showErrors || raw ? iterText : '';
      if (iterText) ok = false;
      const clock = clocks[index];
      phase.iterClock.textContent = clock ? `${clockWords(mode, clock.startStep, clock.targetStep)}.` : index > 0 && mode === 'continue' && validUpdates(updates) ? `Continues where phase ${index} ends.` : '';

      const { edits, problems } = phase.editor.collect(showErrors);
      phase.current = { edits, ok: !problems.length };
      if (problems.length) ok = false;
      // The service takes the name as it is written in the previous phase's recipe (datasets.val*.name).
      const bestName = phase.bestInput.value.trim();
      const bestText = best && !bestName ? `Enter the name of a validation dataset of phase ${index}.` : '';
      phase.bestError.textContent = showErrors ? bestText : '';
      if (bestText) ok = false;
      const item = { name: phaseName, mode, updates, edits };
      if (mode === 'finetune') item.weights = best ? { bestValidation: bestName } : phase.weights;
      body.push(item);
    });

    schedule.textContent = phases
      .map((phase, index) => {
        const clock = clocks[index];
        if (clock) return scheduleWords(index + 1, plan[index].mode, clock.startStep, clock.targetStep);
        return validUpdates(plan[index].updates) ? `Phase ${index + 1} continues where phase ${index} ends.` : `Phase ${index + 1} needs its iterations.`;
      })
      .join(' ');
    addPhaseBtn.disabled = phases.length >= MAX_PHASES;
    limitLine.textContent = phases.length >= MAX_PHASES ? `A curriculum holds at most ${MAX_PHASES} phases.` : `${fmt.plural(phases.length, 'phase')} (at most ${MAX_PHASES}).`;
    if (!ok || !yaml) return null;
    return { name: curriculumName, baseYaml: yaml, phases: body };
  }

  let checkCtl = null;
  let dialogOpen = true;

  /**
   * Check the recipe of every phase whose changes are complete (force: all of them again). Phases with the
   * same changes share one check per pass. Resolves to one result per phase, in order (null: not checked).
   */
  function checkAll(force) {
    checkCtl?.abort();
    const ctl = new AbortController();
    checkCtl = ctl;
    const jobs = new Map();
    return Promise.all(
      phases.map(async (phase) => {
        const { edits, ok } = phase.current;
        if (!ok) {
          phase.checked = null;
          mount(phase.checkSlot);
          return null;
        }
        const key = JSON.stringify(edits);
        if (!force && phase.checked?.key === key) {
          mount(phase.checkSlot, checkView(phase.checked.result));
          return phase.checked.result;
        }
        if (!jobs.has(key)) jobs.set(key, checkPhaseRecipe(yaml, edits, ctl.signal));
        try {
          const result = await jobs.get(key);
          if (ctl.signal.aborted || !dialogOpen) return null;
          phase.checked = { key, result };
          mount(phase.checkSlot, checkView(result));
          return result;
        } catch (error) {
          if (isAbort(error)) return null;
          throw error;
        }
      }),
    );
  }
  const checkSoon = debounce(() => {
    if (dialogOpen && yaml && !busy) checkAll(false).catch(() => {});
  }, 700);

  // The foot status. A refusal there goes with the inputs; the lost-answer notice stays until the next save.
  let footSticky = false;
  function setFoot(node = null, sticky = false) {
    footSticky = sticky;
    mount(footStatus, node);
  }
  function clearFoot() {
    if (!busy && !footSticky) setFoot();
  }

  /** A service refusal belongs to the inputs it was given: it goes as soon as they change. */
  function clearRefusal(phase) {
    if (!phase?.refused) return;
    phase.refused = false;
    mount(phase.issues);
  }

  function clearRefusals() {
    for (const phase of phases) clearRefusal(phase);
    clearFoot();
  }

  /** An edit (of `phase`, or of the curriculum itself when null): clear what no longer applies, refresh. */
  function update(phase = null) {
    clearRefusal(phase);
    if (!phases.some((p) => p.refused)) clearFoot();
    collect(false);
    checkSoon();
    syncPreviews();
  }

  // ───────── what the previous phase keeps as best weights ─────────
  // A hint only: saving never waits for it, and the service decides. When the preview does not
  // answer, for any reason, nothing is shown and the name is typed as before.

  function stopPreview(phase) {
    const p = phase.preview;
    clearTimeout(p.timer);
    p.ctl?.abort();
    p.ctl = null;
    p.key = null;
    p.answer = null;
  }

  /** Draw a phase's preview: the suggestions and the line under the field, each only when it changed. */
  function paintPreview(phase) {
    const p = phase.preview;
    const n = phases.indexOf(phase); // the number of the phase before this one
    const lines = p.answer && n > 0 ? phasePreviewWords(n, p.answer, phase.bestInput.value.trim()) : [];
    const shown = JSON.stringify(lines);
    if (shown !== p.shown) {
      p.shown = shown;
      mount(phase.bestStatus, lines.map((line) => (line.kind === 'warn' ? notice('warn', line.text) : h('p', { class: 'curricula__muted' }, line.text))));
    }
    // Read by the name field each time its list opens.
    phase.bestNames = p.answer ? p.answer.datasets : [];
  }

  /** Ask for the preview each best-weights fine-tune needs: at once when chosen, shortly after an edit of the phase before. */
  function syncPreviews() {
    if (!dialogOpen) return;
    phases.forEach((phase, index) => {
      const p = phase.preview;
      const before = phases[index - 1];
      const chosen = Boolean(yaml) && index > 0 && modeAt(phase) === 'finetune' && phase.weights === 'best';
      const key = chosen && before.current.ok ? JSON.stringify(before.current.edits) : null;
      const first = !p.chosen;
      p.chosen = chosen;
      if (key !== p.key) {
        stopPreview(phase);
        p.key = key;
        if (key !== null) {
          const edits = before.current.edits;
          p.timer = setTimeout(() => {
            const ctl = new AbortController();
            p.ctl = ctl;
            api.curriculumPhasePreview({ baseYaml: yaml, edits }, { signal: ctl.signal }).then(
              (answer) => {
                if (ctl.signal.aborted || !dialogOpen || p.key !== key) return;
                p.answer = readPhasePreview(answer);
                paintPreview(phase);
              },
              () => {}, // no preview: the field stays as it is
            );
          }, first ? 0 : 400);
        }
      }
      paintPreview(phase);
    });
  }

  nameInput.addEventListener('input', () => update(null));
  addPhase();
  addPhase();

  // Everything the person edits, made inert while a save is in flight.
  const fields = h(
    'fieldset',
    { class: 'curricula__lock' },
    h('div', { class: 'field' }, h('label', { class: 'field__label', for: nameId }, 'Curriculum name'), nameInput, nameError),
    h('h3', { class: 'curricula__subtitle' }, 'Phases'),
    notice('', h('div', { class: 'curricula__rules' }, h('p', { class: 'curricula__rule' }, OWN_TEXT), h('p', { class: 'curricula__rule' }, ITER_TEXT))),
    h('p', { class: 'curricula__muted' }, 'An option path is dotted, for example train.optim_g.lr. A value is read as JSON when it parses (numbers, true, false, null, lists, quoted text), otherwise as plain text.'),
    phaseList,
    h('div', { class: 'row row--wrap' }, addPhaseBtn, limitLine),
    h('div', { class: 'curricula__schedule-box' }, h('span', { class: 'field__label' }, 'Schedule'), schedule),
  );
  const body = yaml
    ? h(
        'div',
        { class: 'curricula__form' },
        h('p', null, INTRO, ' Each phase is saved as a queue entry.'),
        h('p', { class: 'curricula__muted' }, 'Base recipe: ', sourceLabel || 'the recipe text given'),
        fields,
        footStatus,
        h('p', { class: 'curricula__hold' }, HOLD_TEXT),
      )
    : notice('err', 'There is no recipe text to plan phases from. Open a recipe on the Recipe screen first.');

  update();

  let saveBtn = null;
  let closeBtn = null;
  let closeDialog = null;
  let lock = () => {};

  function setBusy(on) {
    busy = on;
    if (saveBtn) {
      if (on) saveBtn.setAttribute('aria-disabled', 'true');
      else saveBtn.removeAttribute('aria-disabled');
    }
    lock(on);
  }

  /** The service's reason for a refusal, in one line (for a toast). */
  function reasonText(error) {
    if (error?.status === 422 && Array.isArray(error.issues) && error.issues.length) {
      return error.issues.map((issue) => (issue.phaseId ? `${issue.phaseId.replace('phase-', 'Phase ')}: ${issue.message}` : issue.message)).join(' ');
    }
    return error?.message ?? 'no answer';
  }

  /** Send one request body; the id follows the content. */
  async function send(request) {
    const requestId = requestIdFor(request);
    setFoot(loading('Saving the curriculum…'));
    try {
      const detail = await api.createCurriculum({ ...request, requestId });
      created = detail;
      const count = detail?.phaseCount ?? request.phases.length;
      const words = `Saved curriculum ${detail?.name ?? request.name}: ${fmt.plural(count, 'phase')} on hold.`;
      queue.refresh();
      if (dialogOpen) {
        toast(words, { kind: 'ok' });
        setBusy(false);
        closeDialog(detail);
        if (detail?.id) go(curriculaHref(detail.id));
      } else {
        // Closed anyway (the browser can force Escape): report the outcome where the person is, do not navigate.
        toast(words, { kind: 'ok', action: detail?.id ? { label: 'Open', run: () => go(curriculaHref(detail.id)) } : null });
      }
    } catch (error) {
      if (!dialogOpen) {
        if (error?.offline) toast(`The trainer service did not answer, so the curriculum ${request.name} may have been saved. Look in the Curricula list.`, { kind: 'err', action: { label: 'Open the list', run: () => go(curriculaHref()) } });
        else toast(`The curriculum ${request.name} was not saved: ${reasonText(error)}`, { kind: 'err' });
        return;
      }
      if (error?.status === 422 && Array.isArray(error.issues) && error.issues.length) {
        const byPhase = new Map();
        const general = [];
        for (const issue of error.issues) {
          const match = typeof issue?.phaseId === 'string' ? /^phase-(\d+)$/.exec(issue.phaseId) : null;
          const phase = match ? sent[Number(match[1]) - 1] : null;
          if (phase && phases.includes(phase)) {
            if (!byPhase.has(phase)) byPhase.set(phase, []);
            byPhase.get(phase).push(issue);
          } else general.push(issue);
        }
        for (const [phase, list] of byPhase) {
          phase.refused = true;
          mount(phase.issues, notice('err', [h('p', { class: 'curricula__issue' }, 'The service refused this phase:'), issueList(list)]));
        }
        const lines = [...byPhase].flatMap(([phase, list]) => list.map((issue) => h('li', null, `Phase ${phases.indexOf(phase) + 1} (${phase.nameInput.value.trim()}): ${issue.message}`)));
        setFoot(
          notice('err', [
            h('p', { class: 'curricula__issue' }, `The curriculum was not saved: the service found ${fmt.plural(error.issues.length, 'problem')}.${byPhase.size ? ' Each is shown under its phase.' : ''}`),
            lines.length ? h('ul', { class: 'curricula__summary' }, lines) : null,
            general.length ? issueList(general) : null,
          ]),
        );
      } else if (error?.offline) {
        // The request may have reached the service: never claim that nothing happened.
        setFoot(
          notice('err', [
            h('p', { class: 'curricula__issue' }, 'The trainer service did not answer, so the curriculum may have been saved.'),
            h(
              'p',
              { class: 'curricula__issue' },
              'Try again sends the same plan with the same request, so it is not saved twice. Or ',
              h('a', { href: curriculaHref(), onclick: () => closeDialog(null) }, 'look in the Curricula list'),
              ' (this closes the planner).',
            ),
          ], h('button', { class: 'btn btn--sm', type: 'button', onclick: () => tryAgain(request) }, 'Try again')),
          true,
        );
      } else {
        setFoot(notice('err', `The curriculum was not saved: ${error?.message ?? 'no answer'}`));
      }
    }
  }

  async function tryAgain(request) {
    if (busy || !dialogOpen) return;
    setBusy(true);
    try {
      await send(request);
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    if (busy) return;
    const request = collect(true);
    if (!request) {
      setFoot(notice('err', 'The curriculum was not saved: fix the fields marked above.'));
      return;
    }
    setBusy(true);
    sent = phases.slice();
    for (const phase of phases) clearRefusal(phase);
    try {
      // Every phase's recipe is checked once more. Errors, and a check that did not come back, do not block
      // saving on hold, but they are always asked about.
      checkSoon.cancel();
      setFoot(loading("Checking each phase's recipe…"));
      let results = [];
      try {
        results = await checkAll(true);
      } catch {
        results = [];
      }
      if (!dialogOpen) return;
      const failing = sent.map((phase, i) => (checkFails(results[i]) ? i + 1 : 0)).filter(Boolean);
      const unchecked = sent.map((phase, i) => (!checkFails(results[i]) && checkMissing(results[i]) ? i + 1 : 0)).filter(Boolean);
      if (failing.length || unchecked.length) {
        const words = [failing.length ? failingWords(failing) : '', unchecked.length ? uncheckedWords(unchecked) : ''].filter(Boolean).join(' ');
        setFoot(notice('err', `${words}${failing.length ? ' The problems are shown under each phase.' : ''}`));
        const ok = await confirmDialog({
          title: failing.length ? 'Save with recipe errors?' : 'Save without a recipe check?',
          body: `${words} Save the curriculum anyway?`,
          confirmLabel: 'Save anyway',
          cancelLabel: 'Keep editing',
        });
        if (!ok || !dialogOpen) return;
      }
      await send(request);
    } finally {
      setBusy(false);
    }
  }

  const { closed, dialog, close } = openDialog({
    title: 'Train in phases',
    body,
    wide: true,
    foot: (closeFn) => {
      saveBtn = h('button', { class: 'btn btn--primary', type: 'button', onclick: () => save() }, 'Save curriculum');
      closeBtn = h('button', { class: 'btn', type: 'button', onclick: () => closeFn(null) }, 'Close');
      return [closeBtn, yaml ? saveBtn : null];
    },
  });
  closeDialog = close;
  lock = guardDialog(dialog, () => [closeBtn, dialog.querySelector('.dialog__head button')], [yaml ? fields : null]);
  if (yaml) checkAll(false).catch(() => {});
  return closed.then((value) => {
    dialogOpen = false;
    checkSoon.cancel();
    if (!busy) checkCtl?.abort();
    help.ctl.abort();
    for (const phase of phases) stopPreview(phase);
    return value ?? created ?? null;
  });
}
// ───────── revise dialog ─────────

/**
 * Save a new held revision of one phase. Resolves to the new queue entry or null.
 * @param {{curriculum: object, phase: object, index: number, onSaved?: (entry: object) => void}} options
 */
function openReviseDialog({ curriculum, phase, index, onSaved = () => {} }) {
  const entry = phase.entry ?? {};
  const held = entry.status === 'held';
  const requestIdFor = requestIds(); // the same id only for the same content
  const current = Array.isArray(entry.edits) ? entry.edits : Array.isArray(phase.edits) ? phase.edits : [];
  const group = uid('curricula-revise');
  const ids = { same: `${group}-same`, none: `${group}-none`, different: `${group}-different` };
  const errorSlot = h('div', { class: 'curricula__issues', 'aria-live': 'polite' });
  let choice = held ? 'different' : 'same';
  let busy = false;
  let saved = null;

  const baseYaml = typeof curriculum.baseYaml === 'string' ? curriculum.baseYaml : '';
  const checkSlot = h('div', { class: 'curricula__issues', 'aria-live': 'polite' });
  let checkCtl = null;
  let checked = null; // {key, result} of the last completed check
  let dialogOpen = true;

  /** The changes the chosen option would save, and whether they are complete. */
  function chosenEdits() {
    if (choice === 'same') return { edits: current, ok: true };
    if (choice === 'none') return { edits: [], ok: true };
    const { edits, problems } = editor.collect(false);
    return { edits, ok: !problems.length };
  }

  /** Check the recipe the chosen changes give (force: again even when unchanged). */
  async function check(force) {
    checkCtl?.abort();
    if (!baseYaml) return null;
    const { edits, ok } = chosenEdits();
    if (!ok) {
      checked = null;
      mount(checkSlot);
      return null;
    }
    const key = JSON.stringify(edits);
    if (!force && checked?.key === key) {
      mount(checkSlot, checkView(checked.result));
      return checked.result;
    }
    const ctl = new AbortController();
    checkCtl = ctl;
    try {
      const result = await checkPhaseRecipe(baseYaml, edits, ctl.signal);
      if (ctl !== checkCtl || !dialogOpen) return null;
      checked = { key, result };
      mount(checkSlot, checkView(result));
      return result;
    } catch (error) {
      if (isAbort(error)) return null;
      throw error;
    }
  }
  const checkSoon = debounce(() => {
    if (dialogOpen && !busy) check(false).catch(() => {});
  }, 700);

  // Base recipe options for the path inputs, without the paths the curriculum sets for this phase's mode.
  const help = optionHelp(baseYaml, (path) => reservedFields(phase.mode).some((field) => field.every((key, i) => path[i] === key)), () => editor.collect(false));
  const editor = createEditor({
    initial: current,
    getMode: () => phase.mode,
    onChange: () => {
      editor.collect(false);
      checkSoon();
    },
    help,
    restored: 'notice',
  });
  // The same changes again: flag the ones Continue restores, as the editor does for each row.
  const restoredSame = phase.mode === 'continue' && current.some((edit) => continueRestores(edit?.path) && !editProblem('continue', edit));
  const sameView = h('div', { class: 'curricula__choice-body' }, editsView(current), restoredSame ? notice('warn', RESTORED_WARNING) : null);
  const editorView = h('div', { class: 'curricula__choice-body' }, h('p', { class: 'field__help' }, OWN_TEXT), editor.el);
  const radio = (key, label, hint) =>
    h(
      'div',
      { class: 'curricula__radio' },
      h('input', { type: 'radio', name: group, id: ids[key], value: key, checked: choice === key, onchange: () => setChoice(key) }),
      h('label', { for: ids[key] }, label, hint ? h('span', { class: 'curricula__hint' }, hint) : null),
    );

  function setChoice(key) {
    choice = key;
    sameView.hidden = key !== 'same';
    editorView.hidden = key !== 'different';
    if (key === 'different') editor.collect(false);
    checkSoon.cancel();
    check(false).catch(() => {});
  }

  const choices = h(
    'fieldset',
    { class: 'curricula__fieldset' },
    h('legend', { class: 'field__label' }, 'Changes in the new copy'),
    radio('same', 'The same changes again', 'Retries the phase as planned.'),
    sameView,
    radio('none', `${NO_CHANGES}`, 'The new copy has no changes of its own.'),
    radio('different', 'Different changes', 'Start from the current changes and edit them.'),
    editorView,
  );
  const body = h(
    'div',
    { class: 'curricula__form' },
    h('p', null, 'Saves a new copy of this phase on hold. The mode and the number of iterations stay as planned.', held ? ' The current copy is cancelled.' : ''),
    h(
      'dl',
      { class: 'kv curricula__kv' },
      h('dt', null, 'Phase'), h('dd', null, `Phase ${index + 1}: ${phase.name}`),
      h('dt', null, 'Mode'), h('dd', null, modeLine(phase.mode, phase.weights)),
      h('dt', null, 'Iterations'), h('dd', null, clockWords(phase.mode, phase.startStep, phase.targetStep)),
      h('dt', null, 'Current copy'), h('dd', null, statusBadge(entry.status), Number(entry.revision) > 1 ? ` Revision ${entry.revision}` : ''),
    ),
    choices,
    checkSlot,
    errorSlot,
  );
  setChoice(choice);

  let saveBtn = null;
  let closeBtn = null;
  let closeDialog = null;
  let lock = () => {};

  function setBusy(on) {
    busy = on;
    if (saveBtn) {
      if (on) saveBtn.setAttribute('aria-disabled', 'true');
      else saveBtn.removeAttribute('aria-disabled');
    }
    lock(on);
  }

  const reasonText = (error) =>
    error?.status === 422 && Array.isArray(error.issues) && error.issues.length ? error.issues.map((issue) => issue.message).join(' ') : error?.message ?? 'no answer';

  /** Send one revision request; the id follows the content. */
  async function send(params) {
    const requestId = requestIdFor(params);
    mount(errorSlot, loading('Saving the revision…'));
    try {
      const revision = await api.reviseCurriculumPhase({ ...params, requestId });
      saved = revision;
      const words = Number.isFinite(revision?.revision) ? `Saved revision ${revision.revision} of ${phase.name} on hold.` : `Saved a revision of ${phase.name} on hold.`;
      toast(words, { kind: 'ok' });
      queue.refresh();
      onSaved(revision);
      if (dialogOpen) {
        setBusy(false);
        closeDialog(revision);
      }
    } catch (error) {
      if (!dialogOpen) {
        // Closed anyway (the browser can force Escape): report the outcome where the person is.
        if (error?.offline) toast(`The trainer service did not answer, so a revision of ${phase.name} may have been saved. Look at the phase's revisions.`, { kind: 'err' });
        else toast(`The revision of ${phase.name} was not saved: ${reasonText(error)}`, { kind: 'err' });
        return;
      }
      if (error?.status === 422 && Array.isArray(error.issues) && error.issues.length) {
        mount(errorSlot, notice('err', [h('p', { class: 'curricula__issue' }, 'The revision was not saved:'), issueList(error.issues)]));
      } else if (error?.offline) {
        // The request may have reached the service: never claim that nothing happened.
        mount(
          errorSlot,
          notice(
            'err',
            "The trainer service did not answer, so the revision may have been saved. Try again sends the same request, so it is not saved twice; or close this dialog and look at the phase's revisions.",
            h('button', { class: 'btn btn--sm', type: 'button', onclick: () => tryAgain(params) }, 'Try again'),
          ),
        );
      } else {
        mount(errorSlot, notice('err', `The revision was not saved: ${error?.message ?? 'no answer'}`));
      }
    }
  }

  async function tryAgain(params) {
    if (busy || !dialogOpen) return;
    setBusy(true);
    try {
      await send(params);
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    if (busy) return;
    const params = { id: curriculum.id, phaseId: phase.id };
    if (choice === 'none') params.edits = [];
    else if (choice === 'different') {
      const { edits, problems } = editor.collect(true);
      if (problems.length) {
        mount(errorSlot, notice('err', 'The revision was not saved: fix the changes marked above.'));
        return;
      }
      params.edits = edits;
    }
    setBusy(true);
    try {
      // The chosen changes are checked once more. Errors, and a check that did not come back, do not block
      // saving on hold, but they are always asked about.
      checkSoon.cancel();
      mount(errorSlot, loading("Checking this phase's recipe…"));
      let result = null;
      try {
        result = await check(true);
      } catch {
        result = null;
      }
      if (!dialogOpen) return;
      if (checkFails(result) || checkMissing(result)) {
        mount(errorSlot);
        const words = checkFails(result) ? failingWords([index + 1]) : uncheckedWords([index + 1]);
        const ok = await confirmDialog({
          title: checkFails(result) ? 'Save with recipe errors?' : 'Save without a recipe check?',
          body: `${words} Save the revision anyway?`,
          confirmLabel: 'Save anyway',
          cancelLabel: 'Keep editing',
        });
        if (!ok || !dialogOpen) return;
      }
      await send(params);
    } finally {
      setBusy(false);
    }
  }

  const { closed, dialog, close } = openDialog({
    title: `Revise ${phase.name}`,
    body,
    wide: true,
    foot: (closeFn) => {
      saveBtn = h('button', { class: 'btn btn--primary', type: 'button', onclick: () => save() }, 'Save revision');
      closeBtn = h('button', { class: 'btn', type: 'button', onclick: () => closeFn(null) }, 'Close');
      return [closeBtn, saveBtn];
    },
  });
  closeDialog = close;
  lock = guardDialog(dialog, () => [closeBtn, dialog.querySelector('.dialog__head button')], [choices]);
  return closed.then((value) => {
    dialogOpen = false;
    checkSoon.cancel();
    checkCtl?.abort();
    help.ctl.abort();
    return value ?? saved ?? null;
  });
}
// ───────── the view ─────────

/**
 * @param {HTMLElement} host
 * @param {{curriculumId?: string|null}} options
 * @returns {{setCurriculum(id: string|null): void, destroy(): void}}
 */
export function mountCurriculaView(host, { curriculumId = null } = {}) {
  let current = curriculumId || null;
  let destroyed = false;
  let timer = 0;
  let listCtl = null;
  let detailCtl = null;
  let curriculum = null; // the last detail answer for `current`
  let detailRefs = null; // in-place parts of the rendered detail
  let listSignature = '';
  let listCount = null; // null until the list has answered
  let mainMode = ''; // 'empty' | 'pick' | 'detail' | 'loading' | 'error' | 'missing'
  const busy = new Set();
  const openState = new Map(); // "<phaseId>|<part>" -> open, so a poll does not close a <details>

  const recipeOpen = () => {
    const r = recipe.get();
    return r.phase === 'open' && typeof r.yaml === 'string' && r.yaml !== '';
  };
  let hadRecipe = recipeOpen();

  const planBtn = h('button', { class: 'btn btn--primary btn--sm', type: 'button', onclick: () => createCurriculum(), hidden: !hadRecipe }, 'Plan phases for the open recipe…');
  const listEl = h('ul', { class: 'curricula__list', 'aria-label': 'Curricula, newest first' });
  const listStatus = h('div', { class: 'curricula__status', role: 'status' }, loading('Reading the curricula…'));
  const side = h(
    'nav',
    { class: 'curricula__side', 'aria-label': 'Curricula' },
    h('div', { class: 'curricula__side-head' }, h('h2', { class: 'curricula__title' }, 'Curricula'), h('p', { class: 'curricula__muted' }, INTRO), planBtn),
    listStatus,
    listEl,
  );
  const main = h('div', { class: 'curricula__main' });
  const root = h('div', { class: 'curricula' }, side, main);
  host.appendChild(root);

  // ───────── list ─────────

  async function refreshList() {
    listCtl?.abort();
    listCtl = new AbortController();
    try {
      const list = await api.curricula({ signal: listCtl.signal });
      if (destroyed) return;
      renderList(Array.isArray(list) ? list : []);
    } catch (error) {
      if (isAbort(error) || destroyed) return;
      listSignature = '';
      mount(listStatus, notice('err', `Could not read the curricula: ${error?.message ?? 'no answer'}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => tick() }, 'Try again')));
      if (!current && listCount === null) {
        mainMode = 'error';
        detailRefs = null;
        mount(main, notice('err', 'The curricula could not be read. Use Try again in the list once the service answers.'));
      }
    }
  }

  function renderList(list) {
    const signature = JSON.stringify([current, list.map((c) => [c.id, c.name, c.createdAt, c.phaseCount])]);
    listCount = list.length;
    if (signature === listSignature) return;
    listSignature = signature;
    mount(listStatus);
    if (!list.length) {
      mount(listEl);
      mount(listStatus, h('p', { class: 'curricula__muted' }, 'No curricula yet.'));
      if (!current) renderEmptyMain();
      return;
    }
    const focusedId = listEl.contains(document.activeElement) ? document.activeElement?.dataset?.curriculum : null;
    mount(
      listEl,
      list.map((c) =>
        h(
          'li',
          { class: 'curricula__item' },
          h(
            'a',
            { class: 'curricula__link', href: curriculaHref(c.id), dataset: { curriculum: c.id }, 'aria-current': c.id === current ? 'page' : null },
            h('span', { class: 'curricula__name' }, c.name || c.id),
            h('span', { class: 'curricula__meta' }, fmt.dateTime(c.createdAt) || 'Created time not reported'),
            h('span', { class: 'curricula__meta' }, Number.isFinite(c.phaseCount) ? fmt.plural(c.phaseCount, 'phase') : 'Phase count not reported'),
          ),
        ),
      ),
    );
    if (focusedId) [...listEl.querySelectorAll('a')].find((a) => a.dataset.curriculum === focusedId)?.focus();
    if (!current) renderPickMain();
  }

  function renderEmptyMain() {
    detailRefs = null;
    mainMode = 'empty';
    hadRecipe = recipeOpen();
    mount(
      main,
      emptyState({
        title: 'No curricula yet',
        text: `${INTRO} Each phase is saved on hold as a queue entry, and a phase after the first waits for the one before it to complete. To make one: ${HOW_TEXT}`,
        actions: [
          h('a', { class: 'btn', href: '#/recipe' }, 'Go to the Recipe screen'),
          hadRecipe ? h('button', { class: 'btn btn--primary', type: 'button', onclick: () => createCurriculum() }, 'Plan phases for the open recipe…') : null,
        ].filter(Boolean),
      }),
    );
  }

  function renderPickMain() {
    detailRefs = null;
    mainMode = 'pick';
    mount(main, emptyState({ title: 'Choose a curriculum', text: `${INTRO} Choose one in the list to see its phases, what each one waits for, and its queue entries. To make another: ${HOW_TEXT}` }));
  }

  // ───────── detail ─────────

  /** A phase entry with the fresher state the queue store may hold (whichever was updated later). */
  function merged(entry) {
    if (!entry) return {};
    const fresh = queue.byId(entry.id);
    if (!fresh || String(fresh.updatedAt ?? '') < String(entry.updatedAt ?? '')) return entry;
    return { ...entry, status: fresh.status ?? entry.status, runId: fresh.runId ?? entry.runId, runStatus: fresh.runStatus ?? entry.runStatus, error: fresh.error ?? entry.error };
  }

  const phasesOf = (c) => (Array.isArray(c?.phases) ? c.phases : []);

  async function refreshDetail() {
    const id = current;
    if (!id) return;
    detailCtl?.abort();
    detailCtl = new AbortController();
    if (!detailRefs || detailRefs.id !== id) {
      mainMode = 'loading';
      mount(main, loading('Reading the curriculum…'));
    }
    try {
      const answer = await api.curriculum(id, { signal: detailCtl.signal });
      if (destroyed || id !== current) return;
      curriculum = answer;
      const shape = JSON.stringify([answer.id, answer.name, phasesOf(answer).map((p) => [p.id, p.name, p.mode, p.updates, p.startStep, p.targetStep, p.weights ?? null])]);
      if (!detailRefs || detailRefs.id !== id || detailRefs.shape !== shape) buildDetail(answer, shape);
      else {
        setRefreshProblem('');
        updateDetail();
      }
    } catch (error) {
      if (isAbort(error) || destroyed || id !== current) return;
      if (!detailRefs || detailRefs.id !== id) {
        detailRefs = null;
        if (error?.status === 404) {
          mainMode = 'missing';
          mount(
            main,
            emptyState({
              title: 'Curriculum not found',
              text: `The service has no curriculum with the id ${id}. ${error?.message && error.message !== `404 Not Found` ? error.message : ''}`.trim(),
              actions: [h('a', { class: 'btn', href: curriculaHref() }, 'All curricula')],
            }),
          );
        } else {
          mainMode = 'error';
          mount(main, notice('err', `Could not read this curriculum: ${error?.message ?? 'no answer'}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => tick() }, 'Try again')));
        }
      } else {
        // The open curriculum stays as last read; say so instead of keeping it silently.
        setRefreshProblem(error?.message ?? 'no answer');
      }
    } finally {
      schedule();
    }
  }

  /** The quiet line under the head when a poll of the open curriculum failed ('' clears it). */
  function setRefreshProblem(message) {
    const refs = detailRefs;
    if (!refs || refs.refreshMessage === message) return;
    refs.refreshMessage = message;
    mount(refs.refreshSlot, message ? notice('warn', `This curriculum could not be refreshed: ${message} It shows what was last read; trying again.`) : null);
  }

  function buildDetail(c, shape) {
    const refs = { id: c.id, shape, rows: new Map(), refreshSlot: h('div', { class: 'curricula__status' }), refreshMessage: '' };
    detailRefs = refs;
    mainMode = 'detail';
    const phases = phasesOf(c);

    const header = h(
      'header',
      { class: 'curricula__head' },
      h('h2', { class: 'curricula__title' }, c.name || c.id),
      h(
        'dl',
        { class: 'kv curricula__kv' },
        h('dt', null, 'Created'), h('dd', null, timeEl(c.createdAt)),
        h('dt', null, 'Phases'), h('dd', null, fmt.plural(c.phaseCount ?? phases.length, 'phase')),
        h('dt', null, 'Base recipe fingerprint'), h('dd', null, c.baseDigest ? h('span', { class: 'mono', title: c.baseDigest }, shortDigest(c.baseDigest)) : 'Not reported'),
      ),
      h('p', { class: 'curricula__schedule' }, phases.map((p, i) => (Number.isFinite(p.startStep) && Number.isFinite(p.targetStep) ? scheduleWords(i + 1, p.mode, p.startStep, p.targetStep) : `Phase ${i + 1}: iterations not reported.`)).join(' ')),
      h(
        'div',
        { class: 'row row--wrap' },
        h('button', { class: 'btn btn--sm', type: 'button', onclick: () => openAsRecipe({ yaml: c.baseYaml, name: c.name, label: `Base recipe of the curriculum ${c.name}` }) }, 'Open the base recipe'),
      ),
      typeof c.baseYaml === 'string' && c.baseYaml
        ? h('details', { class: 'curricula__yaml-box' }, h('summary', null, 'Base recipe text'), h('pre', { class: 'curricula__yaml mono', tabindex: '0' }, c.baseYaml))
        : null,
      h('p', { class: 'curricula__muted' }, `${HOLD_TEXT} ${OWN_TEXT}`),
      refs.refreshSlot,
    );

    const list = h(
      'ol',
      { class: 'curricula__phases', 'aria-label': 'Phases in training order' },
      phases.map((phase, index) => {
        const dyn = h('div', { class: 'curricula__phase-state' });
        const alert = h('div', { class: 'curricula__alert' });
        refs.rows.set(phase.id, { dyn, alert, sig: '', phase, index });
        return h(
          'li',
          { class: 'curricula__phase' },
          h('span', { class: 'curricula__phase-num', 'aria-hidden': 'true' }, String(index + 1)),
          h(
            'div',
            { class: 'curricula__phase-body' },
            h('h3', { class: 'curricula__phase-name' }, h('span', { class: 'visually-hidden' }, `Phase ${index + 1}: `), phase.name),
            h('p', { class: 'curricula__phase-mode' }, h('strong', null, MODE_TEXT[phase.mode] ? `${modeHead(phase.mode, phase.weights)}:` : modeHead(phase.mode, phase.weights)), MODE_TEXT[phase.mode] ? ' ' : null, MODE_TEXT[phase.mode] ? h('span', { class: 'curricula__muted-inline' }, modeText(phase.mode, phase.weights)) : null),
            phase.mode === 'finetune' && bestDataset(phase.weights) !== null ? h('p', { class: 'curricula__phase-mode' }, bestLine(phase.weights)) : null,
            h('p', { class: 'curricula__phase-clock' }, `${clockWords(phase.mode, phase.startStep, phase.targetStep)}${validUpdates(phase.updates) ? `, ${fmt.plural(phase.updates, 'iteration')}` : ''}`),
            dyn,
            alert,
          ),
        );
      }),
    );

    mount(main, header, h('section', { class: 'curricula__section', 'aria-label': 'Phases' }, h('h3', { class: 'curricula__subtitle' }, 'Phases'), list));
    updateDetail();
  }

  function phaseSignature(phase, entry, pred) {
    const revisions = (Array.isArray(phase.revisions) ? phase.revisions : []).map((r) => {
      const m = merged(r);
      return [r.id, r.revision, m.status, m.runId ?? null];
    });
    const keys = [...busy].filter((key) => key.endsWith(`|${entry.id}`));
    return JSON.stringify([entry.id, entry.status, entry.runId ?? null, entry.runStatus ?? null, entry.error ?? null, entry.revision ?? null, entry.handoff ?? null, entry.edits ?? null, pred ? [pred.status, pred.id] : null, revisions, keys]);
  }

  /** Update every phase row in place; a row's state part is rebuilt only when it changed. */
  function updateDetail() {
    const refs = detailRefs;
    if (!refs || !curriculum || curriculum.id !== refs.id) return;
    const phases = phasesOf(curriculum);
    const active = document.activeElement;
    phases.forEach((phase, index) => {
      const slot = refs.rows.get(phase.id);
      if (!slot) return;
      slot.phase = phase;
      const entry = merged(phase.entry);
      const pred = index > 0 ? merged(phases[index - 1].entry) : null;
      const sig = phaseSignature(phase, entry, pred);
      if (slot.sig === sig) return;
      slot.sig = sig;
      const hadFocus = slot.dyn.contains(active);
      const focusKey = hadFocus ? active?.dataset?.focus ?? null : null;
      mount(slot.dyn, renderPhaseState(phase, index, entry, pred, phases));
      if (hadFocus) {
        const target = [...slot.dyn.querySelectorAll('a, button, summary')].find((el) => el.dataset.focus === focusKey) ?? slot.dyn.querySelector('a');
        target?.focus();
      }
    });
  }

  function remembered(key, el) {
    if (openState.get(key)) el.open = true;
    el.addEventListener('toggle', () => openState.set(key, el.open));
    return el;
  }

  function runLink(runId, entry, label, focus) {
    const rs = entry?.runStatus ? runStatus(entry.runStatus) : null;
    return h('a', { class: 'curricula__link-inline', href: runHref(runId), dataset: { focus }, 'aria-label': `${label}${rs ? `, ${rs.label}` : ''}` }, rs ? `Run: ${rs.label.toLowerCase()}` : 'Run');
  }

  function renderPhaseState(phase, index, entry, pred, phases) {
    const n = index + 1;
    const where = `phase ${n}, ${phase.name}`;
    const st = entryStatus(entry.status);
    const parts = [];

    parts.push(
      h(
        'div',
        { class: 'curricula__state' },
        statusBadge(entry.status),
        Number(entry.revision) > 1 ? h('span', { class: 'curricula__revision' }, `Revision ${entry.revision}`) : null,
        entry.id ? h('a', { class: 'curricula__link-inline', href: queueHref(entry.id), dataset: { focus: 'entry' }, 'aria-label': `Queue entry for ${where}` }, 'Queue entry') : null,
        entry.runId ? runLink(entry.runId, entry, `Run for ${where}`, 'run') : null,
      ),
    );
    if (st.kind === 'err' && entry.error) parts.push(h('p', { class: 'curricula__entry-error' }, entry.error));

    // What the phase waits for, readable before any click.
    let canRelease = false;
    if (entry.status === 'held') {
      if (index === 0) {
        canRelease = true;
        parts.push(h('p', { class: 'curricula__gate' }, 'Ready: begins at iteration 0 when released.'));
      } else if (pred?.status === 'completed') {
        canRelease = true;
        const from = phases[index - 1]?.targetStep;
        parts.push(h('p', { class: 'curricula__gate' }, `Ready: takes over from phase ${index} at iteration ${Number.isFinite(from) ? fmt.int(from) : 'not reported'}.`));
      } else {
        parts.push(h('p', { class: 'curricula__gate' }, `Waits for phase ${index} (${phases[index - 1]?.name ?? ''}) to complete.`));
      }
    } else if (entry.status === 'cancelled' || entry.status === 'failed' || entry.status === 'stopped') {
      parts.push(h('p', { class: 'curricula__gate' }, 'To train this phase again, revise it: the new copy is saved on hold.'));
    }

    if (entry.handoff && typeof entry.handoff === 'object') parts.push(handoffView(phase, index, entry.handoff));

    parts.push(h('div', { class: 'curricula__phase-changes' }, h('span', { class: 'curricula__label' }, 'What this phase changes'), editsView(Array.isArray(entry.edits) ? entry.edits : phase.edits)));

    const buttons = [];
    for (const action of ['release', 'hold', 'cancel', 'stop']) {
      if (!ENTRY_ACTIONS[action].from.includes(entry.status)) continue;
      if (action === 'release' && !canRelease) continue;
      buttons.push(actionButton(action, phase, index, entry, where));
    }
    if (REVISABLE.has(entry.status)) {
      buttons.push(
        h(
          'button',
          { class: 'btn btn--sm', type: 'button', dataset: { focus: 'revise' }, 'aria-label': `Revise ${where}…`, onclick: () => revise(phase, index) },
          'Revise this phase…',
        ),
      );
    }
    if (buttons.length) parts.push(h('div', { class: 'curricula__actions' }, buttons));

    const earlier = (Array.isArray(phase.revisions) ? phase.revisions : []).filter((r) => r && r.id !== entry.id);
    if (earlier.length) {
      parts.push(
        remembered(
          `${phase.id}|revisions`,
          h(
            'details',
            { class: 'curricula__details' },
            h('summary', { dataset: { focus: 'revisions' } }, `Earlier revisions (${earlier.length})`),
            h(
              'ul',
              { class: 'curricula__revisions' },
              earlier.map((r) => {
                const m = merged(r);
                return h(
                  'li',
                  { class: 'curricula__revision-item' },
                  h('span', null, Number.isFinite(r.revision) ? `Revision ${r.revision}` : 'Revision not reported'),
                  statusBadge(m.status),
                  h('a', { class: 'curricula__link-inline', href: queueHref(r.id), 'aria-label': `Queue entry for revision ${r.revision ?? ''} of ${where}` }, 'Queue entry'),
                  m.runId ? runLink(m.runId, m, `Run for revision ${r.revision ?? ''} of ${where}`, `run-${r.id}`) : null,
                );
              }),
            ),
          ),
        ),
      );
    }
    return parts;
  }

  function handoffView(phase, index, handoff) {
    const files = Array.isArray(handoff.files) ? handoff.files : [];
    return h(
      'div',
      { class: 'curricula__handoff' },
      h('span', { class: 'curricula__label' }, index > 0 ? `Takes over from phase ${index}, pinned when released` : 'Pinned when released'),
      h(
        'dl',
        { class: 'kv curricula__kv' },
        h('dt', null, 'Source run'), h('dd', null, handoff.runId ? h('a', { href: runHref(handoff.runId), dataset: { focus: 'source-run' }, title: handoff.runId }, `Run of phase ${index}`) : 'Not reported'),
        h('dt', null, 'Source entry'), h('dd', null, handoff.entryId ? h('a', { href: queueHref(handoff.entryId), dataset: { focus: 'source-entry' } }, 'Queue entry') : 'Not reported'),
        h('dt', null, 'Iteration'), h('dd', null, Number.isFinite(handoff.step) ? fmt.int(handoff.step) : 'Not reported'),
        h('dt', null, 'Seed'), h('dd', null, Number.isFinite(handoff.seed) ? String(handoff.seed) : 'Not recorded'),
      ),
      files.length
        ? remembered(
            `${phase.id}|files`,
            h(
              'details',
              { class: 'curricula__details' },
              h('summary', { dataset: { focus: 'files' } }, `Pinned files (${files.length})`),
              h(
                'ul',
                { class: 'curricula__files' },
                files.map((file) => {
                  const size = Array.isArray(file?.version) && Number.isFinite(file.version[2]) ? file.version[2] : null;
                  return h(
                    'li',
                    { class: 'curricula__file' },
                    h('span', { class: 'mono' }, file?.option ?? 'Option not reported'),
                    h('span', { class: 'mono curricula__path', title: file?.source ?? null }, file?.path ?? file?.source ?? 'Path not reported'),
                    size !== null ? h('span', { class: 'curricula__meta' }, fmt.plural(size, 'byte')) : null,
                  );
                }),
              ),
            ),
          )
        : h('p', { class: 'curricula__muted' }, 'No pinned files reported.'),
    );
  }

  function actionButton(action, phase, index, entry, where) {
    const label = ENTRY_ACTIONS[action].label;
    const key = `${action}|${entry.id}`;
    return h(
      'button',
      {
        class: `btn btn--sm${action === 'release' ? ' btn--primary' : action === 'cancel' || action === 'stop' ? ' btn--quiet' : ''}`,
        type: 'button',
        'aria-label': `${label}: ${where}`,
        'aria-disabled': busy.has(key) ? 'true' : null,
        dataset: { focus: action },
        onclick: () => runAction(action, phase, index, entry, where, key),
      },
      label,
    );
  }

  async function runAction(action, phase, index, entry, where, key) {
    if (busy.has(key)) return;
    if (action === 'cancel') {
      const ok = await confirmDialog({
        title: `Cancel ${where}?`,
        body: h('div', { class: 'stack' }, h('p', null, CANCEL_TEXT), h('p', null, 'To train this phase later, revise it: the new copy is saved on hold.')),
        confirmLabel: 'Cancel entry',
        cancelLabel: 'Keep entry',
        danger: true,
      });
      if (!ok) return;
    } else if (action === 'stop') {
      const ok = await confirmDialog({ title: `Stop training ${where}?`, body: STOP_TEXT, confirmLabel: 'Stop training', cancelLabel: 'Keep training', danger: true });
      if (!ok) return;
    } else if (action === 'release') {
      // A released phase trains as soon as the trainer is free: the phone edition asks first.
      const ok = await confirmOnPhone({
        title: `Release ${where}?`,
        body: queue.get().paused ? 'The queue is paused, so it waits until you continue the queue. Then it starts when the trainer is free.' : 'It starts when the trainer is free, after the released entries ahead of it in line.',
        confirmLabel: 'Release',
      });
      if (!ok) return;
    }
    if (destroyed) return;
    const alert = detailRefs?.rows.get(phase.id)?.alert;
    if (alert) mount(alert);
    busy.add(key);
    updateDetail();
    try {
      await queue[action](entry.id);
      const done = { release: `Released ${where}.`, hold: `Put ${where} on hold.`, cancel: `Cancelled ${where}.`, stop: `Asked the trainer to stop ${where}.` }[action];
      toast(action === 'release' && queue.get().paused ? `${done} The queue is paused, so it waits until you continue the queue.` : done, { kind: 'ok' });
    } catch (error) {
      // The service's own words, in the phase's row; never only a toast.
      const slot = detailRefs?.rows.get(phase.id)?.alert;
      if (slot) mount(slot, notice('err', [h('strong', null, `${ENTRY_ACTIONS[action].label} was refused. `), error?.message ?? 'The trainer service did not answer.']));
      else toast(`${where} was not changed: ${error?.message ?? 'no answer'}`, { kind: 'err' });
    } finally {
      busy.delete(key);
      if (!destroyed) {
        refreshDetail();
        refreshList();
      }
    }
  }

  async function revise(phase, index) {
    if (!curriculum) return;
    // Refreshed on a saved revision even when its answer arrives after the dialog was closed.
    await openReviseDialog({
      curriculum,
      phase: { ...phase, entry: merged(phase.entry) },
      index,
      onSaved: () => {
        if (destroyed) return;
        const alert = detailRefs?.rows.get(phase.id)?.alert;
        if (alert) mount(alert);
        refreshDetail();
        refreshList();
      },
    });
  }

  // ───────── polling ─────────

  function schedule() {
    clearTimeout(timer);
    if (destroyed || document.hidden) return;
    const moving = current && curriculum?.id === current && phasesOf(curriculum).some((p) => MOVING.has(merged(p.entry).status));
    timer = setTimeout(tick, moving ? FAST : SLOW);
  }

  function tick() {
    if (destroyed) return;
    refreshList();
    if (current) refreshDetail();
    else schedule();
  }

  const onVisibility = () => {
    if (document.hidden) clearTimeout(timer);
    else tick();
  };
  document.addEventListener('visibilitychange', onVisibility);
  const unsubscribeQueue = queue.subscribe(() => {
    if (!destroyed) updateDetail();
  });
  const unsubscribeRecipe = recipe.subscribe(() => {
    if (destroyed) return;
    const open = recipeOpen();
    planBtn.hidden = !open;
    if (open !== hadRecipe && mainMode === 'empty' && !main.contains(document.activeElement)) renderEmptyMain();
    hadRecipe = open;
  });
  if (queue.get().phase === 'idle') queue.refresh();

  tick();

  return {
    setCurriculum(id) {
      if (destroyed) return;
      const next = id || null;
      if (next === current) return;
      detailCtl?.abort();
      current = next;
      curriculum = null;
      detailRefs = null;
      listSignature = '';
      // Back to the list: never leave the previous curriculum on screen, even when the list cannot be read.
      if (!next) {
        if (listCount === 0) renderEmptyMain();
        else if (listCount > 0) renderPickMain();
        else {
          mainMode = 'loading';
          mount(main, loading('Reading the curricula…'));
        }
      }
      tick();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      clearTimeout(timer);
      listCtl?.abort();
      detailCtl?.abort();
      document.removeEventListener('visibilitychange', onVisibility);
      unsubscribeQueue();
      unsubscribeRecipe();
      root.remove();
    },
  };
}
