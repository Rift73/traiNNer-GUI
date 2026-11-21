// Run files: a run's recipe, seed, starting point and input places as one JSON object, so the run can be
// repeated or continued on another machine (studio/REPRODUCTION.md). Three dialogs and one shared block:
//
//   openFineTuneDialog({run, artifact})              -> Promise<{yaml, name, label} | null>
//   openRunFileExport({run, artifacts, preselect})   -> Promise<void>
//   openRunFileImport()                              -> Promise<{doc: {yaml, path: null, name}, label, saved: false} | null>
//   renderPrepared(answer, {onOpen}?)                -> HTMLElement
//
// Nothing here starts, saves or queues anything: the last step hands back a candidate recipe.
//
// A run file is kept as the exact text the service (or the person's file) gave. The bundle's digest covers
// its other fields, and a browser that parses and writes it again can change it (1000000.0 comes back as
// 1000000), after which the service refuses it. So a parsed copy is used only to check `format` and to
// display; requests, "Save the file" and "Copy" all use the text.

import { h, mount, uid, fmt } from '../dom.js';
import { api } from '../api.js';
import { openDialog, notice, badge, loading, issueList, copyText } from '../ui.js';
import { renderResume, resumeSentence } from '../recipe/resume.js';
import { highlightedPre } from '../recipe/yaml-highlight.js';
import { pickPath } from '../components/path-picker.js';

const FORMAT = 'trainner-run-recipe';
/** A run file is a few tens of kilobytes; this only keeps a wrong file from freezing the page. */
const MAX_FILE_BYTES = 8 * 1024 * 1024;

const RUN_FILE_WORDS =
  'A run file describes a run so it can be repeated or continued elsewhere: its recipe, its seed, how it started and where its inputs were. It holds no weights and no data; copy those yourself.';

// ───────── words for the service's codes (studio/reproduction.py) ─────────

/** bundle.mode */
const MODE_WORDS = {
  replay: { name: 'Repeat the run', hint: 'The same recipe, seed and starting point.' },
  'weights-only': { name: 'Fine-tune from a checkpoint', hint: 'Starts from those weights with a fresh optimizer, schedule and iteration count.' },
};

/** initialization.kind (_initialization, and 'weights-only' for a chosen weight) */
const START_WORDS = {
  fresh: 'Starts new, at iteration 0',
  pretrained: 'Starts from pretrained weights',
  'full-state': 'Continues from a saved training state',
  'weights-only': 'Starts from the chosen weights, fresh optimizer',
  unverified: 'How it started is not recorded',
};

/** seed.evidence (export_run_recipe) and initialization.seedEvidence ('explicit' is added by prepare) */
const SEED_WORDS = {
  runtime: 'as the trainer reported it',
  invocation: 'from the command that started the trainer',
  runRecord: "from this app's record of the run",
  legacyLog: "read from the run's training log",
  declared: 'as written in the recipe; the trainer did not report it',
  explicit: 'given for this recipe',
  missing: 'not recorded',
};

/** initialization.optimizerState */
const OPTIMIZER_WORDS = {
  fresh: { text: 'Fresh: the optimizer and schedule start new', kind: '' },
  restore: { text: 'Restored from the saved training state', kind: '' },
  unverified: { text: 'Not verified', kind: 'warn' },
};

/** inputs[].kind (_kind) */
const KIND_WORDS = { dataset: 'Dataset folder', weight: 'Weights file', state: 'Training state' };

/** inputs[].status (_compare_input, plus 'declared', 'unresolved' and 'unavailable' set by prepare) */
const INPUT_WORDS = {
  declared: { text: 'Path only; contents not compared', kind: '' },
  fileVersionMatch: { text: 'The same file as recorded', kind: 'ok' },
  sha256Match: { text: 'Contents match the recorded checksum', kind: 'ok' },
  saveIdMatch: { text: 'The same save, by the ID stored in the file', kind: 'ok' },
  launchMetadataMatch: { text: 'Size and date match; contents not compared', kind: 'warn' },
  unverified: { text: 'Not verified', kind: 'warn' },
  sha256Mismatch: { text: 'Contents differ from the recorded checksum', kind: 'err' },
  differentSize: { text: 'A different size than recorded', kind: 'err' },
  saveIdMismatch: { text: 'A different save, by the ID stored in the file', kind: 'err' },
  invalid: { text: 'Not a readable weights file', kind: 'err' },
  unavailable: { text: 'The file was not found or could not be read', kind: 'err' },
  unresolved: { text: 'No place on this machine yet', kind: 'err' },
};

/**
 * inputs[].launchEvidence (_launch_evidence): the launch record against the file seen at export. Said
 * without the words "run file", because the fine-tune dialog shows this block and never names one.
 */
const LAUNCH_WORDS = {
  fileVersionMatch: "When the run's details were collected, this was the file the run had started with.",
  launchMetadataMatch: "When the run's details were collected, its size and date matched the start of the run; its contents were not compared.",
  different: "When the run's details were collected, the file no longer matched the start of the run.",
  unverified: 'Whether this is the file the run started with is not verified.',
};

/** compatibility.*.status */
const COMPARE_WORDS = {
  match: { text: 'The same as for the original run', kind: 'ok' },
  different: { text: 'Differs from the original run', kind: 'warn' },
  unverified: { text: 'Not verified', kind: 'warn' },
  preserved: { text: 'Unchanged from the original run', kind: 'ok' },
  modified: { text: 'Changed to start from the chosen weights', kind: '' },
};

/** compatibility.config.basis (_basis) */
const BASIS_WORDS = {
  effective: 'Taken from the options the trainer ran with.',
  submitted: 'Taken from the recipe as it was submitted; the options the trainer ran with were not readable.',
  archived: "Taken from the recipe saved in the run's folder.",
};

// ───────── small helpers ─────────

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const dotted = (field) => (Array.isArray(field) ? field.map(String).join('.') : String(field ?? ''));
const lastPart = (path) => {
  const parts = String(path ?? '').split(/[\\/]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : String(path ?? '');
};
/** The `name:` a candidate recipe ends up with (the service may have changed a typed name). */
const yamlName = (yaml) => /^name:\s*(.+?)\s*$/m.exec(String(yaml ?? ''))?.[1]?.replace(/^['"]|['"]$/g, '') || '';
/** Characters no file name may hold on Windows are replaced. */
const fileSafe = (name) => String(name ?? '').replace(/[\\/:*?"<>|\u0000-\u001f]/g, '_').trim() || 'run';

/**
 * A status in words. Colour repeats the word; an error also says "Error". The code sits in the title.
 * A status the service did not send is "Not reported"; one this screen has no words for says so.
 */
function state(words, code = null) {
  if (!words && (code === null || code === undefined)) return h('span', { class: 'muted' }, 'Not reported');
  const kind = words?.kind ?? 'warn';
  return h(
    'span',
    { class: `runfile__state${kind ? ` runfile__state--${kind}` : ''}`, title: code ? String(code) : null },
    kind === 'err' ? h('strong', null, 'Error: ') : null,
    words?.text ?? 'A result this screen does not know',
  );
}

const pathText = (path) => h('span', { class: 'runfile__path' }, String(path));
/** A recipe option's dotted name; it may wrap after a dot, not inside a word. */
const optionText = (field) => {
  const parts = dotted(field).split('.');
  return h('span', { class: 'runfile__path runfile__option' }, parts.flatMap((part, i) => (i < parts.length - 1 ? [`${part}.`, h('wbr')] : [part])));
};
const absent = (text) => h('span', { class: 'muted' }, text);
const title = (text) => h('h3', { class: 'runfile__title' }, text);
const kv = (rows) => h('dl', { class: 'kv' }, rows.filter(Boolean).flatMap(([label, value]) => [h('dt', null, label), h('dd', null, value)]));

function seedValue(value, evidence) {
  if (!Number.isInteger(value)) return state({ text: 'Not recorded', kind: 'warn' }, evidence);
  const how = SEED_WORDS[evidence];
  return h('span', { title: evidence ? String(evidence) : null }, how ? `${value}, ${how}` : String(value));
}

function startValue(kind, resume = null) {
  const words = (kind === 'full-state' && resumeSentence(resume).replace(/\.$/, '')) || START_WORDS[kind];
  if (!words) return state(null, kind);
  return kind === 'unverified' ? state({ text: words, kind: 'warn' }, kind) : h('span', { title: String(kind) }, words);
}

// ───────── path groups for the "where are its inputs" rows ─────────

/** The folder a file sits in, as text ('' when there is none above). */
export function folderOf(path) {
  const text = String(path ?? '');
  const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
  if (cut < 0) return '';
  if (cut === 0) return text[0];
  return text.slice(0, cut);
}

/** Component-wise common prefix of paths (either separator); '' when they share nothing. */
export function commonFolder(paths) {
  const list = (Array.isArray(paths) ? paths : []).filter((p) => typeof p === 'string' && p);
  if (!list.length) return '';
  if (list.length === 1) return list[0];
  const separator = list[0].includes('\\') && !list[0].includes('/') ? '\\' : '/';
  const parts = list.map((p) => p.split(/[\\/]/));
  let shared = 0;
  while (shared < parts[0].length && parts.every((own) => shared < own.length && own[shared] === parts[0][shared])) shared += 1;
  const head = parts[0].slice(0, shared).join(separator);
  if (head) return head;
  return shared > 0 ? separator : '';
}

/** A shared folder too shallow to be worth one row: nothing, the root, or a bare drive. */
const tooShallow = (folder) => !folder || /^[\\/]+$/.test(folder) || /^[A-Za-z]:[\\/]?$/.test(folder);

/**
 * The "Was at" places to prefill for a bundle: one for the dataset folders and one for the other inputs
 * (weights and training states, by the folder they sit in; a fine-tune bundle's chosen weight included).
 * @returns {string[]}
 */
export function suggestedPlaces(bundle) {
  const inputs = Array.isArray(bundle?.inputs) ? bundle.inputs.filter(isObject) : [];
  const where = (rows) => rows.map((row) => row.sourcePath).filter((p) => typeof p === 'string' && p);
  const datasets = where(inputs.filter((row) => row.kind === 'dataset'));
  const files = where(inputs.filter((row) => row.kind !== 'dataset'));
  const chosen = bundle?.initialization?.weight?.sourcePath;
  if (typeof chosen === 'string' && chosen) files.push(chosen);
  const group = (paths) => {
    const unique = [...new Set(paths)];
    if (!unique.length) return [];
    const shared = commonFolder(unique);
    return tooShallow(shared) ? unique : [shared];
  };
  return [...new Set([...group(datasets), ...group(files.map(folderOf).filter(Boolean))])];
}

/** Inference weights among a run's artifacts: models/net_g* or models/net_ae*, .safetensors or .pth. */
export function inferenceWeights(artifacts) {
  return (Array.isArray(artifacts) ? artifacts : []).filter(
    (a) => typeof a?.path === 'string' && a.path.startsWith('models/') && /^net_(g|ae)/.test(lastPart(a.path)) && /\.(safetensors|pth)$/.test(a.path),
  );
}

// ───────── what a run file holds (export result, import origin) ─────────

function bundleFacts(bundle) {
  const mode = MODE_WORDS[bundle?.mode];
  const origin = isObject(bundle?.origin) ? bundle.origin : {};
  const init = isObject(bundle?.initialization) ? bundle.initialization : {};
  const weight = isObject(init.weight) ? init.weight : null;
  return kv([
    ['Run', typeof origin.name === 'string' && origin.name ? origin.name : absent('Not named')],
    ['What it is for', mode ? h('span', { title: String(bundle.mode) }, `${mode.name}: ${mode.hint.charAt(0).toLowerCase()}${mode.hint.slice(1)}`) : state(null, bundle?.mode)],
    ['Seed', seedValue(bundle?.seed?.value, bundle?.seed?.evidence)],
    ['How it starts', startValue(init.kind)],
    weight && typeof weight.sourcePath === 'string' ? ['Chosen weights', pathText(weight.sourcePath)] : null,
  ]);
}

function bundleInputs(bundle) {
  const inputs = Array.isArray(bundle?.inputs) ? bundle.inputs.filter(isObject) : [];
  if (!inputs.length) return h('p', { class: 'muted' }, 'The recipe names no input paths.');
  return h(
    'div',
    { class: 'runfile__scroll' },
    h(
      'table',
      { class: 'table runfile__table' },
      h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Recipe option'), h('th', { scope: 'col' }, 'Kind'), h('th', { scope: 'col' }, 'Where it was'))),
      h(
        'tbody',
        null,
        inputs.map((row) =>
          h(
            'tr',
            null,
            h('td', null, optionText(row.field)),
            h('td', null, KIND_WORDS[row.kind] ?? state(null, row.kind)),
            h('td', null, typeof row.sourcePath === 'string' && row.sourcePath ? pathText(row.sourcePath) : row.value !== undefined && row.value !== null ? [pathText(String(row.value)), h('p', { class: 'runfile__note' }, 'As written in the recipe; its full place was not found.')] : absent('Not recorded')),
          ),
        ),
      ),
    ),
  );
}

// ───────── the answer of a check, as a block ─────────

/** Whether a candidate can be opened, and how the button reads. null when there is no candidate. */
function openChoice(answer) {
  if (typeof answer?.yaml !== 'string') return null;
  return answer.ok === true ? { label: 'Open as recipe', primary: true } : { label: 'Open as recipe to fix it', primary: false };
}

function differenceValue(value) {
  if (value === null || value === undefined) return absent('Not set');
  if (value === '') return absent('Empty');
  return pathText(typeof value === 'string' ? value : JSON.stringify(value));
}

function details(summary, ...body) {
  return h('details', { class: 'runfile__details' }, h('summary', { class: 'runfile__summary' }, summary), body);
}

function preparedInputs(inputs) {
  const rows = (Array.isArray(inputs) ? inputs : []).filter(isObject);
  if (!rows.length) return h('p', { class: 'muted' }, 'The recipe names no input paths.');
  return h(
    'div',
    { class: 'runfile__scroll' },
    h(
      'table',
      { class: 'table runfile__table' },
      h(
        'thead',
        null,
        h('tr', null, ['Recipe option', 'Kind', 'Original place', 'Place on this machine', 'Compared with the original'].map((text) => h('th', { scope: 'col' }, text))),
      ),
      h(
        'tbody',
        null,
        rows.map((row) => {
          const original = typeof row.sourcePath === 'string' && row.sourcePath ? row.sourcePath : row.declared;
          const launch = row.launchEvidence;
          return h(
            'tr',
            null,
            h('td', null, optionText(row.field)),
            h('td', null, KIND_WORDS[row.kind] ?? state(null, row.kind)),
            h('td', null, original !== undefined && original !== null ? pathText(String(original)) : absent('Not recorded')),
            h('td', null, typeof row.path === 'string' && row.path ? (row.path === original && row.mapped !== true ? absent('The same place') : [pathText(row.path), row.mapped === true ? [' ', badge('mapped')] : null]) : absent('None yet')),
            h('td', null, state(INPUT_WORDS[row.status], row.status), launch ? h('p', { class: 'runfile__note', title: String(launch) }, LAUNCH_WORDS[launch] ?? 'How it relates to the start of the run is not known here.') : null),
          );
        }),
      ),
    ),
  );
}

function comparison(compatibility, warnings) {
  const c = isObject(compatibility) ? compatibility : {};
  const part = (key) => (isObject(c[key]) ? c[key] : null);
  const environment = part('environment');
  const source = part('source');
  const config = part('config');
  const data = part('data');
  const device = isObject(environment?.device) ? environment.device : null;
  const shown = new Set();
  /** The service's own warning about one part of the comparison, in its words. */
  const warned = (area) =>
    warnings
      .filter((issue) => issue.path === `compatibility.${area}`)
      .map((issue) => {
        shown.add(issue);
        return h('p', { class: 'runfile__warn', title: issue.code ? String(issue.code) : null }, h('strong', null, issue.level === 'error' ? 'Error: ' : 'Warning: '), String(issue.message ?? ''));
      });
  const status = (block) => (block ? state(COMPARE_WORDS[block.status], block.status) : absent('Not reported'));
  const note = (text) => (typeof text === 'string' && text ? h('p', { class: 'runfile__note' }, text) : null);

  const differences = Array.isArray(environment?.differences) ? environment.differences.filter(isObject) : [];
  const environmentBody = [
    status(environment),
    warned('environment'),
    differences.length
      ? details(
          `Show ${fmt.plural(differences.length, 'difference')}`,
          h(
            'div',
            { class: 'runfile__scroll' },
            h(
              'table',
              { class: 'table runfile__table' },
              h('thead', null, h('tr', null, ['Setting', 'For the original run', 'On this machine'].map((text) => h('th', { scope: 'col' }, text)))),
              h('tbody', null, differences.map((d) => h('tr', null, h('td', null, pathText(String(d.path ?? ''))), h('td', null, differenceValue(d.recorded)), h('td', null, differenceValue(d.current))))),
            ),
          ),
        )
      : null,
  ];

  const changes = isObject(source?.differences) ? source.differences : null;
  const groups = changes
    ? [
        ['added', 'Added'],
        ['removed', 'Removed'],
        ['changed', 'Changed'],
      ].map(([key, label]) => ({ label, count: changes[key]?.count, paths: Array.isArray(changes[key]?.paths) ? changes[key].paths : [] }))
    : [];
  const counted = groups.filter((group) => Number.isInteger(group.count));
  const sourceBody = [
    status(source),
    warned('source'),
    note(source?.reason),
    counted.length ? h('p', null, `Files: ${counted.map((group) => `${fmt.int(group.count)} ${group.label.toLowerCase()}`).join(', ')}.`) : null,
    note(typeof source?.recorded?.scope === 'string' ? `Covers: ${source.recorded.scope}.` : ''),
    groups.some((group) => group.paths.length)
      ? details(
          'Show the files',
          groups
            .filter((group) => group.paths.length)
            .map((group) => [
              h('p', { class: 'runfile__group' }, Number.isInteger(group.count) && group.count > group.paths.length ? `${group.label} (the first ${fmt.int(group.paths.length)} of ${fmt.int(group.count)})` : group.label),
              h('ul', { class: 'runfile__list' }, group.paths.map((path) => h('li', null, pathText(String(path))))),
            ]),
        )
      : null,
  ];

  const used = isObject(device?.recorded) ? device.recorded : null;
  const usedText = used ? [used.name, used.type].filter((v) => typeof v === 'string' && v) : [];
  const deviceBody = [
    status(device),
    warned('environment.device'),
    note(usedText.length ? `The original run used ${usedText.length === 2 ? `${usedText[0]} (${usedText[1]})` : usedText[0]}.` : device ? "The original run's device was not recorded." : ''),
  ];

  const block = kv([
    ['Recipe', [status(config), warned('config'), note(BASIS_WORDS[config?.basis] ?? '')]],
    ['Environment', environmentBody],
    ['Training code', sourceBody],
    ['Device', deviceBody],
    ['Data', [status(data), warned('data'), note(data?.reason)]],
  ]);
  const rest = warnings.filter((issue) => !shown.has(issue));
  return [block, rest.map((issue) => h('p', { class: 'runfile__warn', title: issue.code ? String(issue.code) : null }, h('strong', null, 'Warning: '), String(issue.message ?? '')))];
}

/**
 * The answer of api.reproductionPrepare drawn as a block (shared by both flows).
 * With `onOpen`, the block ends with the button that opens the candidate; the dialogs here keep that
 * button in their foot instead.
 * @param {object} answer  {ok, yaml, initialization, compatibility, inputs, issues}
 * @param {{onOpen?: (answer: object) => void}} [options]
 * @returns {HTMLElement}
 */
export function renderPrepared(answer, { onOpen = null } = {}) {
  const a = isObject(answer) ? answer : {};
  const issues = (Array.isArray(a.issues) ? a.issues : []).filter(isObject);
  const aboutComparison = (issue) => typeof issue.path === 'string' && (issue.path === 'compatibility' || issue.path.startsWith('compatibility.')) && issue.level !== 'error';
  const compared = issues.filter(aboutComparison);
  const own = issues.filter((issue) => !aboutComparison(issue));
  const errors = own.filter((issue) => issue.level === 'error');
  const warnings = own.filter((issue) => issue.level !== 'error');
  const hasYaml = typeof a.yaml === 'string';

  let verdict;
  if (a.ok === true) verdict = notice('ok', 'This recipe passes the check on this machine.');
  else if (errors.length) verdict = notice('err', errors.length === 1 ? '1 problem must be solved first.' : `${fmt.int(errors.length)} problems must be solved first.`);
  else verdict = notice('err', 'This recipe does not pass the check on this machine.');

  const init = isObject(a.initialization) ? a.initialization : {};
  const resume = isObject(init.resume) ? init.resume : null;
  const choice = onOpen ? openChoice(a) : null;

  return h(
    'section',
    { class: 'runfile__prepared' },
    verdict,
    errors.length ? issueList(errors) : null,
    warnings.length ? [h('p', { class: 'runfile__note' }, fmt.plural(warnings.length, 'warning')), issueList(warnings)] : null,

    title('Starts from'),
    kv([
      ['How it starts', startValue(init.kind, resume)],
      ['Seed', seedValue(init.seed, init.seedEvidence)],
      ['Optimizer state', init.optimizerState === undefined || init.optimizerState === null ? absent('Not reported') : state(OPTIMIZER_WORDS[init.optimizerState], init.optimizerState)],
    ]),
    init.kind === 'full-state' ? renderResume(resume) : null,

    title('Inputs'),
    preparedInputs(a.inputs),

    title('Compared with the original run'),
    comparison(a.compatibility, compared),

    hasYaml
      ? details('The recipe that would open', highlightedPre(a.yaml, 'yaml__pre runfile__yaml'))
      : h('p', { class: 'runfile__warn' }, 'A path could not be resolved, so no recipe could be made. Add or correct a place above and check again.'),
    choice ? h('div', { class: 'row' }, h('button', { class: `btn${choice.primary ? ' btn--primary' : ''}`, type: 'button', onclick: () => onOpen(a) }, choice.label)) : null,
  );
}

// ───────── dialog pieces ─────────

/**
 * Bring a fresh answer to the top of the dialog when it landed below what the dialog shows: after a
 * long form, the only visible change would otherwise be a button in the foot.
 */
function reveal(el) {
  const box = el.closest('.dialog__body');
  if (!box) return;
  const top = el.getBoundingClientRect().top - box.getBoundingClientRect().top;
  if (top > box.clientHeight / 2) el.scrollIntoView({ block: 'start' });
}

/**
 * Keep a dialog open and its controls still while a request runs (as guardDialog in curricula.js does):
 * close() is held back, Escape is refused, buttons and field sets are disabled.
 * @returns {{set: (on: boolean) => void, dispose: () => void}}
 */
function lockDialog(dialog, buttons, fieldsets) {
  let locked = false;
  let refocus = null;
  const nativeClose = dialog.close.bind(dialog);
  dialog.close = (...args) => {
    if (!locked) nativeClose(...args);
  };
  const onCancel = (event) => {
    if (locked) event.preventDefault();
  };
  dialog.addEventListener('cancel', onCancel);
  return {
    set(on) {
      if (on === locked) return;
      locked = on;
      const sets = fieldsets().filter(Boolean);
      if (on) refocus = dialog.contains(document.activeElement) ? document.activeElement : null;
      for (const button of [dialog.querySelector('.dialog__head button'), ...buttons()]) if (button) button.disabled = on;
      for (const set of sets) set.disabled = on;
      if (on) dialog.setAttribute('aria-busy', 'true');
      else dialog.removeAttribute('aria-busy');
      if (!on && refocus?.isConnected && !refocus.disabled && (!document.activeElement || document.activeElement === document.body)) refocus.focus();
    },
    dispose() {
      dialog.removeEventListener('cancel', onCancel);
    },
  };
}

function textField({ label, help = '', value = '', path = false, numeric = false, onInput = null }) {
  const id = uid('runfile-field');
  const helpId = help ? uid('runfile-help') : null;
  const error = h('p', { class: 'field__error', role: 'alert', hidden: true });
  const input = h('input', {
    class: `input${path ? ' input--path' : ''}`,
    id,
    type: 'text',
    value,
    inputmode: numeric ? 'numeric' : null,
    autocomplete: 'off',
    spellcheck: 'false',
    'aria-describedby': helpId,
    oninput: () => {
      setError('');
      onInput?.();
    },
  });
  function setError(text) {
    error.textContent = text || '';
    error.hidden = !text;
    if (text) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  }
  const el = h('div', { class: 'field' }, h('label', { class: 'field__label', for: id }, label), input, help ? h('p', { class: 'field__help', id: helpId }, help) : null, error);
  return { el, input, setError };
}

/** {ok, value}: an empty field is "no seed given" (undefined). */
function readSeed(field) {
  const text = field.input.value.trim();
  if (!text) return { ok: true, value: undefined };
  if (!/^\d+$/.test(text) || !Number.isSafeInteger(Number(text))) {
    field.setError('Enter a whole number, 0 or more, or leave the field empty.');
    field.input.focus();
    return { ok: false, value: undefined };
  }
  return { ok: true, value: Number(text) };
}

/**
 * The check step two dialogs share: a button that sends, a region for the answer, and the button that
 * opens the candidate. One of the two buttons is the primary one at any time.
 * `prepare()` returns null when a field must be corrected first, otherwise the function that sends.
 */
function createCheck({ label, busyText, refused, prepare, onOpen }) {
  const result = h('div', { class: 'runfile__result' });
  // Only the outcome is announced; the tables below it are read when the person goes to them.
  const said = h('p', { class: 'visually-hidden', role: 'status' });
  const staleNote = h('p', { class: 'runfile__warn', hidden: true }, `The fields changed after this check. Press "${label}" again for the result with the new values.`);
  const checkButton = h('button', { class: 'btn btn--primary', type: 'button', onclick: () => check() }, label);
  const openButton = h('button', { class: 'btn', type: 'button', hidden: true, onclick: () => current() && onOpen(answer) });
  let answer = null;
  let stale = false;
  let busy = false;
  let enabled = true;
  let over = false;
  let lock = null;
  let host = null;
  let onEnter = null;

  const current = () => (answer && !stale && typeof answer.yaml === 'string' ? answer : null);
  function sync() {
    const choice = current() ? openChoice(answer) : null;
    openButton.hidden = !choice;
    if (choice) {
      openButton.textContent = choice.label;
      openButton.className = `btn${choice.primary ? ' btn--primary' : ''}`;
    }
    checkButton.className = `btn${choice?.primary ? '' : ' btn--primary'}`;
    checkButton.disabled = busy || !enabled;
    openButton.disabled = busy;
    staleNote.hidden = !(answer && stale);
    result.classList.toggle('runfile__result--stale', Boolean(answer && stale));
  }
  async function check() {
    if (busy || over || !enabled) return;
    const send = prepare();
    if (!send) return;
    busy = true;
    answer = null;
    stale = false;
    lock?.set(true);
    sync();
    said.textContent = busyText;
    mount(result, loading(busyText));
    try {
      const next = await send();
      if (over) return;
      answer = next;
      const block = renderPrepared(next);
      mount(result, block);
      reveal(result);
      said.textContent = `${block.querySelector('.notice')?.textContent ?? ''} The details are below.`.trim();
    } catch (error) {
      if (over || error?.name === 'AbortError') return;
      // The service's sentence; the fields stay as they are so it can be tried again.
      const sentence = `${refused} ${error?.message ?? 'The trainer service did not answer.'}`;
      mount(result, notice('err', sentence));
      reveal(result);
      said.textContent = sentence;
    } finally {
      busy = false;
      lock?.set(false);
      sync();
    }
  }
  return {
    el: h('div', { class: 'runfile__answer' }, said, staleNote, result),
    checkButton,
    openButton,
    /** A field changed: the answer on screen is for older values. */
    changed() {
      if (!answer || stale) return;
      stale = true;
      sync();
    },
    /** Forget the answer (another run file was read). */
    reset() {
      answer = null;
      stale = false;
      if (!busy) {
        mount(result);
        said.textContent = '';
      }
      sync();
    },
    setEnabled(on) {
      enabled = on;
      sync();
    },
    attach(dialog, closeButton, fieldsets) {
      lock = lockDialog(dialog, () => [closeButton, checkButton, openButton], fieldsets);
      // Enter in a one-line field sends the check, as it would in a form.
      onEnter = (event) => {
        if (event.key !== 'Enter' || event.isComposing || event.shiftKey || event.ctrlKey || event.altKey || event.metaKey) return;
        if (!(event.target instanceof HTMLInputElement) || event.target.type !== 'text') return;
        event.preventDefault();
        check();
      };
      host = dialog;
      dialog.addEventListener('keydown', onEnter);
      sync();
    },
    end() {
      over = true;
      host?.removeEventListener('keydown', onEnter);
      lock?.dispose();
    },
  };
}

// ───────── fine-tune on this machine ─────────

/**
 * Fine-tune on this machine. artifact = {path: 'models/net_g_2.safetensors', name, version?} of `run`.
 * Resolves {yaml, name, label} to open as a recipe, or null when closed.
 */
export function openFineTuneDialog({ run, artifact }) {
  const runName = String(run?.name ?? 'run');
  const weightName = artifact?.name || lastPart(artifact?.path);
  const controller = new AbortController();
  let closeDialog = null;

  const check = createCheck({
    label: 'Prepare the recipe',
    busyText: 'Preparing the recipe…',
    refused: 'The recipe was not prepared:',
    prepare() {
      const typed = name.input.value.trim();
      if (!typed) {
        name.setError('Enter a name for the new recipe.');
        name.input.focus();
        return null;
      }
      const seedRead = readSeed(seed);
      if (!seedRead.ok) return null;
      return async () => {
        const bundleText = await api.reproductionExportText({ id: run.id, mode: 'weights-only', weightPath: artifact.path, weightsVersion: artifact.version }, { signal: controller.signal });
        return api.reproductionPrepareText(bundleText, { name: typed, seed: seedRead.value }, { signal: controller.signal });
      };
    },
    onOpen: (answer) => closeDialog?.({ yaml: answer.yaml, name: yamlName(answer.yaml) || name.input.value.trim() || `${runName}_finetune`, label: `fine-tune of ${runName}` }),
  });

  const name = textField({ label: 'Name of the new recipe', value: `${runName}_finetune`, onInput: check.changed });
  const seed = textField({ label: 'Seed', help: "Optional. Empty keeps the run's recorded seed.", numeric: true, onInput: check.changed });
  const fields = h('fieldset', { class: 'runfile__fields' }, h('legend', { class: 'visually-hidden' }, 'The new recipe'), name.el, seed.el);
  const body = h(
    'div',
    { class: 'runfile' },
    h(
      'p',
      null,
      'This makes a new recipe that starts from the weights in ',
      h('span', { class: 'mono' }, weightName),
      ' of ',
      h('strong', null, runName),
      ', with a fresh optimizer, schedule and iteration count; nothing is saved or started until you do that from the recipe.',
    ),
    fields,
    check.el,
  );

  let closeButton = null;
  const { dialog, close, closed } = openDialog({
    title: 'Fine-tune from a checkpoint',
    body,
    wide: true,
    foot: (closeFn) => {
      closeButton = h('button', { class: 'btn', type: 'button', onclick: () => closeFn(null) }, 'Cancel');
      return [closeButton, check.checkButton, check.openButton];
    },
  });
  closeDialog = close;
  check.attach(dialog, closeButton, () => [fields]);
  name.input.focus();
  name.input.select();
  return closed.then((picked) => {
    check.end();
    controller.abort();
    return picked && typeof picked.yaml === 'string' ? picked : null;
  });
}

// ───────── save a run file for another machine ─────────

/**
 * Save a run file for another machine. artifacts = the run's artifact list [{name, path}];
 * preselect = an artifact path or null. Resolves when closed.
 */
export function openRunFileExport({ run, artifacts = [], preselect = null }) {
  const runName = String(run?.name ?? 'run');
  const weights = inferenceWeights(artifacts);
  const preselected = weights.find((a) => a.path === preselect) ?? null;
  const controller = new AbortController();
  const ids = { replay: uid('runfile-mode'), 'weights-only': uid('runfile-mode'), weight: uid('runfile-weight'), group: uid('runfile-group') };
  let mode = preselected ? 'weights-only' : 'replay';
  let made = null; // {text, data}: the run file as the service wrote it, and a parsed copy to show
  let busy = false;
  let over = false;
  let objectUrl = null;
  let lock = null;

  const result = h('div', { class: 'runfile__result' });
  const said = h('p', { class: 'visually-hidden', role: 'status' });
  const weightSelect = h(
    'select',
    { class: 'select', id: ids.weight, disabled: mode !== 'weights-only', onchange: () => forget() },
    weights.map((a) => h('option', { value: a.path, selected: a.path === (preselected ?? weights[0])?.path }, a.name || lastPart(a.path))),
  );
  const radio = (key, disabled = false) =>
    h('input', {
      type: 'radio',
      name: ids.group,
      id: ids[key],
      value: key,
      checked: mode === key,
      disabled,
      onchange: () => {
        mode = key;
        weightSelect.disabled = mode !== 'weights-only';
        forget();
      },
    });
  const choice = (key, extra = null, disabled = false) =>
    h(
      'div',
      { class: 'runfile__choice' },
      radio(key, disabled),
      h('div', { class: 'runfile__choicebody' }, h('label', { class: 'field__label', for: ids[key] }, MODE_WORDS[key].name), h('p', { class: 'field__help' }, MODE_WORDS[key].hint), extra),
    );
  const fields = h(
    'fieldset',
    { class: 'runfile__fields' },
    h('legend', { class: 'runfile__legend' }, 'What the run file is for'),
    choice('replay'),
    choice(
      'weights-only',
      weights.length
        ? h('div', { class: 'field' }, h('label', { class: 'field__label', for: ids.weight }, 'Checkpoint to start from'), weightSelect)
        : h('p', { class: 'field__help' }, 'This run has no saved generator or autoencoder weights to start from.'),
      !weights.length,
    ),
  );
  const makeButton = h('button', { class: 'btn btn--primary', type: 'button', onclick: () => make() }, 'Make the run file');
  let closeButton = null;

  function sync() {
    makeButton.className = `btn${made ? '' : ' btn--primary'}`;
    makeButton.disabled = busy;
  }
  /** The choice changed: a file made for the earlier choice is no longer what would be saved. */
  function forget() {
    if (!made) return;
    made = null;
    mount(result);
    said.textContent = '';
    sync();
  }

  function showMade() {
    const { text, data } = made;
    const fileName = `${fileSafe(runName)}.run.json`;
    const status = h('span', { class: 'runfile__note', role: 'status' });
    const limitations = (Array.isArray(data.limitations) ? data.limitations : []).filter((row) => isObject(row) && typeof row.message === 'string');
    const save = h(
      'button',
      {
        class: 'btn btn--primary',
        type: 'button',
        onclick: () => {
          if (objectUrl) URL.revokeObjectURL(objectUrl);
          objectUrl = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
          const link = h('a', { href: objectUrl, download: fileName, hidden: true });
          result.appendChild(link);
          link.click();
          link.remove();
          status.textContent = `Handed to the browser as ${fileName}.`;
        },
      },
      'Save the file',
    );
    const copy = h(
      'button',
      {
        class: 'btn',
        type: 'button',
        onclick: async () => {
          const done = await copyText(text);
          status.textContent = done ? 'Copied.' : 'The browser did not allow copying. Save the file instead.';
        },
      },
      'Copy',
    );
    mount(
      result,
      title('What the run file holds'),
      bundleFacts(data),
      title('Inputs'),
      bundleInputs(data),
      limitations.map((row) => notice(row.level === 'error' ? 'err' : 'warn', row.message)),
      h('div', { class: 'row row--wrap' }, save, copy, status),
      h('p', null, 'On the other machine: Recipe, Choose a recipe, Open a run file. Copy the checkpoint and dataset files there yourself.'),
    );
  }

  async function make() {
    if (busy || over) return;
    const chosen = mode === 'weights-only' ? weights.find((a) => a.path === weightSelect.value) : null;
    if (mode === 'weights-only' && !chosen) return;
    busy = true;
    made = null;
    lock?.set(true);
    sync();
    said.textContent = 'Making the run file…';
    mount(result, loading('Making the run file…'));
    try {
      const params = chosen ? { id: run.id, mode, weightPath: chosen.path, weightsVersion: chosen.version } : { id: run.id, mode };
      const text = await api.reproductionExportText(params, { signal: controller.signal });
      if (over) return;
      let data = null;
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
      if (!isObject(data) || data.format !== FORMAT) {
        said.textContent = 'The run file was not made: the trainer service answered with something that is not a run file.';
        mount(result, notice('err', said.textContent));
        return;
      }
      made = { text, data };
      showMade();
      reveal(result);
      said.textContent = 'The run file is made. Save or copy it below.';
    } catch (error) {
      if (over || error?.name === 'AbortError') return;
      said.textContent = `The run file was not made: ${error?.message ?? 'The trainer service did not answer.'}`;
      mount(result, notice('err', said.textContent));
    } finally {
      busy = false;
      lock?.set(false);
      sync();
    }
  }

  const body = h('div', { class: 'runfile' }, h('p', null, RUN_FILE_WORDS), h('p', null, 'Run: ', h('strong', null, runName)), fields, said, result);
  const { dialog, closed } = openDialog({
    title: 'Save a run file',
    body,
    wide: true,
    foot: (closeFn) => {
      closeButton = h('button', { class: 'btn', type: 'button', onclick: () => closeFn(undefined) }, 'Close');
      return [closeButton, makeButton];
    },
  });
  lock = lockDialog(dialog, () => [closeButton, makeButton], () => [fields]);
  // Start on the choice, not on the close button the dialog would otherwise focus.
  fields.querySelector('input[type="radio"]:checked')?.focus();
  return closed.then(() => {
    over = true;
    lock.dispose();
    controller.abort();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  });
}

// ───────── open a run file on this machine ─────────

/**
 * Open a run file on this machine. Resolves {doc: {yaml, path: null, name}, label, saved: false}
 * (the same shape the neosr import returns) or null.
 */
export function openRunFileImport() {
  const fileId = uid('runfile-file');
  const areaId = uid('runfile-text');
  const controller = new AbortController();
  let bundleText = null; // the run file as read, sent as it is
  let bundle = null; // a parsed copy, for display only
  let rows = []; // [{el, from, to}]
  let name = null;
  let seed = null;
  let sourceRoot = null;
  let closeDialog = null;

  const fileError = h('p', { class: 'field__error', role: 'alert', hidden: true });
  const readError = h('p', { class: 'field__error', role: 'alert', hidden: true });
  const fileInput = h('input', { type: 'file', id: fileId, accept: '.json,application/json', class: 'runfile__file' });
  const area = h('textarea', { class: 'textarea mono runfile__paste', id: areaId, rows: '5', spellcheck: 'false', autocomplete: 'off', wrap: 'off' });
  const source = h(
    'fieldset',
    { class: 'runfile__fields' },
    h('legend', { class: 'visually-hidden' }, 'The run file'),
    h('div', { class: 'field' }, h('label', { class: 'field__label', for: fileId }, 'Choose a run file (.json) from this computer'), fileInput, fileError),
    h('div', { class: 'field' }, h('label', { class: 'field__label', for: areaId }, 'Or paste the text of a run file'), area, readError),
  );
  const origin = h('div', { class: 'runfile__origin', hidden: true });
  const places = h('fieldset', { class: 'runfile__fields', hidden: true });
  const extra = h('fieldset', { class: 'runfile__fields', hidden: true });

  const check = createCheck({
    label: 'Check on this machine',
    busyText: 'Checking on this machine…',
    refused: 'The run file was not checked:',
    prepare() {
      if (!bundleText) return null;
      const seedRead = seed ? readSeed(seed) : { ok: true, value: undefined };
      if (!seedRead.ok) return null;
      const mappings = rows.map((row) => ({ from: row.from.value.trim(), to: row.to.value.trim() })).filter((m) => m.from && m.to);
      const text = bundleText;
      const rest = {
        mappings: mappings.length ? mappings : undefined,
        name: name?.input.value.trim() || undefined,
        seed: seedRead.value,
        sourceRoot: sourceRoot?.input.value.trim() || undefined,
      };
      return () => api.reproductionPrepareText(text, rest, { signal: controller.signal });
    },
    onOpen: (answer) => closeDialog?.({ yaml: answer.yaml }),
  });
  check.setEnabled(false);

  function placeRow(from = '', to = '') {
    const fromId = uid('runfile-from');
    const toId = uid('runfile-to');
    const fromInput = h('input', { class: 'input input--path', id: fromId, type: 'text', value: from, autocomplete: 'off', spellcheck: 'false', oninput: check.changed });
    const toInput = h('input', { class: 'input input--path grow', id: toId, type: 'text', value: to, autocomplete: 'off', spellcheck: 'false', oninput: check.changed });
    const browse = h(
      'button',
      {
        class: 'btn',
        type: 'button',
        onclick: async () => {
          const picked = await pickPath({ title: 'Where is it on this machine?', mode: 'folder', start: toInput.value.trim() });
          if (!picked) return;
          toInput.value = picked;
          check.changed();
        },
      },
      'Browse',
    );
    const row = { el: null, from: fromInput, to: toInput };
    const remove = h(
      'button',
      {
        class: 'btn btn--quiet',
        type: 'button',
        onclick: () => {
          rows = rows.filter((other) => other !== row);
          row.el.remove();
          check.changed();
          addButton.focus();
        },
      },
      'Remove',
    );
    row.el = h(
      'li',
      { class: 'runfile__place' },
      h('div', { class: 'field' }, h('label', { class: 'field__label', for: fromId }, 'Was at'), fromInput),
      h('div', { class: 'field' }, h('label', { class: 'field__label', for: toId }, 'Is now at'), h('div', { class: 'row' }, toInput, browse)),
      remove,
    );
    return row;
  }
  const placeList = h('ul', { class: 'runfile__places' });
  const addButton = h(
    'button',
    {
      class: 'btn btn--sm',
      type: 'button',
      onclick: () => {
        const row = placeRow();
        rows.push(row);
        placeList.appendChild(row.el);
        row.from.focus();
      },
    },
    'Add a place',
  );

  /** Build the origin and the fields for a run file that was just read. Runs once per file, never per check. */
  function showBundle() {
    const originName = typeof bundle.origin?.name === 'string' && bundle.origin.name ? bundle.origin.name : '';
    rows = suggestedPlaces(bundle).map((place) => placeRow(place));
    mount(placeList, rows.map((row) => row.el));
    name = textField({ label: 'Name of the new recipe', help: originName ? `Optional. Empty keeps the name in the run file (${originName}).` : 'Optional. Empty keeps the name in the run file.', onInput: check.changed });
    seed = Number.isInteger(bundle.seed?.value)
      ? null
      : textField({ label: 'Seed', help: 'The run file records no seed. A seed given here fixes this run; it does not verify the original.', numeric: true, onInput: check.changed });
    sourceRoot =
      bundle.origin?.source !== 'managed'
        ? textField({
            label: 'Folder the run was started from',
            help: "On the original machine. The recipe's relative paths are read from this folder; needed only when the run file does not record it.",
            path: true,
            onInput: check.changed,
          })
        : null;
    const inputCount = Array.isArray(bundle.inputs) ? bundle.inputs.length : 0;
    mount(origin, title('The run it describes'), bundleFacts(bundle), inputCount ? details(`Show the ${fmt.plural(inputCount, 'input')} it names`, bundleInputs(bundle)) : null);
    mount(
      places,
      h('legend', { class: 'runfile__legend' }, 'Where are its inputs on this machine?'),
      h('p', { class: 'field__help' }, 'An input whose path starts with a "Was at" place is looked for under its "Is now at" place. Leave "Is now at" empty where nothing moved.'),
      placeList,
      h('div', { class: 'row' }, addButton),
    );
    mount(extra, h('legend', { class: 'visually-hidden' }, 'The new recipe'), name.el, seed?.el ?? null, sourceRoot?.el ?? null);
    for (const part of [origin, places, extra]) part.hidden = false;
  }

  function refuse(text) {
    readError.textContent = text;
    readError.hidden = !text;
    if (text) area.setAttribute('aria-invalid', 'true');
    else area.removeAttribute('aria-invalid');
  }
  /** Read the text in the box. The fields are rebuilt only when it is another run file than before. */
  function read() {
    const text = area.value.replace(/^﻿/, '').trim();
    if (text && text === bundleText) {
      refuse('');
      return;
    }
    let data = null;
    let problem = '';
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        problem = 'This is not a run file: the text is not valid JSON.';
      }
      if (!problem && (!isObject(data) || data.format !== FORMAT)) problem = 'This is JSON, but not a run file. A run file is saved from a run with "Save a run file".';
    }
    refuse(problem);
    check.reset();
    if (!text || problem) {
      bundleText = null;
      bundle = null;
      rows = [];
      for (const part of [origin, places, extra]) {
        part.hidden = true;
        mount(part);
      }
      check.setEnabled(false);
      return;
    }
    bundleText = text;
    bundle = data;
    showBundle();
    check.setEnabled(true);
  }

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    fileError.hidden = true;
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
      fileError.textContent = `${file.name} is larger than 8 MB. A run file is far smaller; choose the .run.json file.`;
      fileError.hidden = false;
      return;
    }
    try {
      area.value = await file.text();
      read();
    } catch (error) {
      fileError.textContent = `${file.name} could not be read: ${error.message}`;
      fileError.hidden = false;
    }
  });
  area.addEventListener('input', read);

  const body = h(
    'div',
    { class: 'runfile' },
    h('p', null, RUN_FILE_WORDS),
    h('p', null, 'Nothing is saved or started here: the result opens in the recipe editor, and you save or start it yourself.'),
    source,
    origin,
    places,
    extra,
    check.el,
  );
  let closeButton = null;
  const { dialog, close, closed } = openDialog({
    title: 'Open a run file',
    body,
    wide: true,
    foot: (closeFn) => {
      closeButton = h('button', { class: 'btn', type: 'button', onclick: () => closeFn(null) }, 'Cancel');
      return [closeButton, check.checkButton, check.openButton];
    },
  });
  closeDialog = close;
  check.attach(dialog, closeButton, () => [source, places, extra]);
  fileInput.focus();
  return closed.then((picked) => {
    check.end();
    controller.abort();
    if (!picked || typeof picked.yaml !== 'string') return null;
    const originName = typeof bundle?.origin?.name === 'string' && bundle.origin.name ? bundle.origin.name : 'unnamed run';
    return { doc: { yaml: picked.yaml, path: null, name: yamlName(picked.yaml) || originName }, label: `run file (${originName})`, saved: false };
  });
}
