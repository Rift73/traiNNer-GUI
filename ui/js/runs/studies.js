// Controlled studies (#/runs/studies[/<studyId>]): one base recipe, a few named variations (arms), the same
// seeds for each. Creating a study saves every arm/seed cell as an ordinary queue entry on hold; releasing
// is the normal queue action. Comparing two arms gives one row per seed, then the service's description of
// the matched seeds (seedSummary: mean, median, range, spread), never a significance test.
//
//   const view = mountStudiesView(host, { studyId });  view.setStudy(id);  view.destroy();
//   const study = await openStudyDialog({ yaml, sourceLabel, name });
//   renderComparison(answer, arms) -> nodes;  renderSeedSummary(answer, armName) -> section | null

import { h, mount, fmt, uid } from '../dom.js';
import { api, newRequestId } from '../api.js';
import { queue, entryStatus, ENTRY_ACTIONS } from '../state/queue.js';
import { recipe, parsePath } from '../state/recipe.js';
import { runStatus } from '../state/runs.js';
import { go, studiesHref, queueHref, runHref, compareHref, openAsRecipe } from '../actions.js';
import { toast, openDialog, confirmDialog, badge, notice, loading, emptyState, copyText } from '../ui.js';
import { createCombobox, toChoices } from '../components/combobox.js';

const LIMIT = 128;
const MAX_SEED = 4294967295;
const SLOW = 20000;
const FAST = 3000;
const MOVING = new Set(['pending', 'running', 'stopping']);
const STUDY_TEXT =
  'A study trains the same recipe several times: a few named variations of it, each with the same seeds. Every entry waits on hold until you release it.';
const HOLD_TEXT = 'Every entry is saved on hold. Nothing trains until you release it.';
const PER_SEED_TEXT = 'One row per seed.';

const PAIR_TEXT = {
  unpaired: 'Waiting: both entries need a run',
  unavailable: 'Unavailable: the recorded result could not be read',
  incompatible: 'Not comparable',
  compared: 'Compared',
};

const isAbort = (error) => error?.name === 'AbortError';
const shortDigest = (digest) => (typeof digest === 'string' ? digest.slice(0, 12) : '');

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

function statusBadge(status) {
  const st = entryStatus(status);
  return badge(st.label, st.kind);
}

/** Fixed decimals for a group of numbers: 4 significant digits of the smallest non-zero magnitude, at most 6. */
function placesFor(values) {
  let smallest = Infinity;
  for (const v of values) if (Number.isFinite(v) && v !== 0) smallest = Math.min(smallest, Math.abs(v));
  if (!Number.isFinite(smallest)) return 0;
  return Math.max(0, Math.min(6, 3 - Math.floor(Math.log10(smallest))));
}

const fixed = (n, places) => (Number.isFinite(n) ? n.toFixed(places) : 'Not reported');
function signedFixed(n, places) {
  if (n === 0) return 'No difference';
  if (!Number.isFinite(n)) return 'Not reported';
  return n > 0 ? `+${n.toFixed(places)}` : n.toFixed(places);
}
const exact = (n) => (Number.isFinite(n) ? `Exact value: ${n}` : null);

// ───────── across seeds: the seedSummary of a comparison ─────────

const NO_COHORT_TEXT =
  'A summary across seeds appears when the baseline and the variation both have runs trained with the same seeds and scored at the same iteration, on the same images, with the same metric settings.';
const ONE_SEED_TEXT = 'One seed cannot show spread: the spread needs at least two matched seeds.';
const ACROSS_TEXT = 'A description of the matched seeds, not a significance test.';

/** Why a seed is left out, by the reason codes of studio/study_results.py (the pair status, or its own). */
const EXCLUDED_TEXT = {
  unpaired: 'one of the two entries has no run yet',
  unavailable: 'the recorded result could not be read',
  incompatible: 'the two runs were scored with different settings, or on no image in common',
  partialPopulation: 'the two runs were not scored on exactly the same images',
  nonfinite: 'a score is not a finite number',
};

const AGGREGATION_TEXT = {
  meanPerSample: "Each run's score is the mean of its per-image scores.",
  pooledDataset: "Each run's score is one value measured over the whole dataset.",
};

const SUMMARY_STATS = [
  ['mean', 'Mean'],
  ['median', 'Median'],
  ['min', 'Lowest'],
  ['max', 'Highest'],
];

const isCount = (n) => Number.isInteger(n) && n >= 0;

/**
 * The left, right and difference of one compared pair, read as studio/study_results.py reads it: the
 * whole-dataset (pooled) scores when present, otherwise the means of the per-image scores.
 */
function pairValues(pair) {
  const c = pair?.status === 'compared' ? pair.comparison : null;
  if (!c) return null;
  if (c.pooled && typeof c.pooled === 'object') return { pooled: true, left: c.pooled.left, right: c.pooled.right, delta: c.pooled.delta, summary: null };
  if (c.summary && typeof c.summary === 'object') return { pooled: false, left: c.summary.meanLeft, right: c.summary.meanRight, delta: c.summary.meanDelta, summary: c.summary };
  return null;
}

/**
 * Decimal places of the per-seed table: one for scores, one for differences. The summary across seeds uses
 * the same two (scores for the baseline and variation rows; differences for the difference row and every
 * spread). Without pairs (a summary shown on its own) they come from the cohorts' own numbers.
 */
function comparisonPlaces(pairs, cohorts = []) {
  const values = (Array.isArray(pairs) ? pairs : []).map(pairValues).filter(Boolean);
  if (values.length) return { value: placesFor(values.flatMap((v) => [v.left, v.right])), difference: placesFor(values.map((v) => v.delta)) };
  const stats = (c, key) => SUMMARY_STATS.map(([stat]) => c?.[key]?.[stat]);
  return { value: placesFor(cohorts.flatMap((c) => [...stats(c, 'left'), ...stats(c, 'right')])), difference: placesFor(cohorts.flatMap((c) => stats(c, 'delta'))) };
}

/** One statistic. A difference is coloured, with a word, only when the service gives the direction. */
function statCell(value, { signed = false, better = null, places = 0 } = {}) {
  if (!Number.isFinite(value)) return h('td', { class: 'num studies__muted' }, 'Not reported');
  const text = signed ? signedFixed(value, places) : fixed(value, places);
  if (!signed || !better || value === 0) return h('td', { class: 'num', title: exact(value) }, text);
  const good = better === 'higher' ? value > 0 : value < 0;
  return h('td', { class: `num ${good ? 'studies__better' : 'studies__worse'}`, title: exact(value) }, text, h('span', { class: 'studies__tone-word' }, good ? 'better' : 'worse'));
}

function spreadCell(stats, places) {
  const sd = stats?.sampleSd;
  if (sd === null) return h('td', { class: 'num studies__muted' }, 'Needs two seeds');
  return Number.isFinite(sd) ? h('td', { class: 'num', title: exact(sd) }, fixed(sd, places)) : h('td', { class: 'num studies__muted' }, 'Not reported');
}

/**
 * "Of 2 matched seeds, 2 improved, 0 tied and 0 regressed." from the service's counts. Without a stated
 * direction the service still counts as if lower were better (study_results.py: sign = -1 unless
 * better == "higher"), so its `improved` is then "lower" and its `regressed` "higher".
 */
function countsLine(cohort, n, better) {
  const { improved, tied, regressed } = cohort;
  if (![improved, tied, regressed].every(isCount)) return null;
  const of = `Of ${fmt.plural(n, 'matched seed')}`;
  if (better) return `${of}, ${fmt.int(improved)} improved, ${fmt.int(tied)} tied and ${fmt.int(regressed)} regressed.`;
  return `${of}, the variation scored higher on ${fmt.int(regressed)}, equal on ${fmt.int(tied)} and lower on ${fmt.int(improved)}.`;
}

function cohortBlock(cohort, answer, names, several, places) {
  const better = cohort.better === 'higher' || cohort.better === 'lower' ? cohort.better : null;
  const seeds = Array.isArray(cohort.seeds) ? cohort.seeds : [];
  const n = Number.isInteger(cohort.n) ? cohort.n : seeds.length;
  const { leftStep, rightStep } = cohort;
  let iteration = 'iteration not reported';
  if (Number.isInteger(leftStep) && leftStep === rightStep) iteration = `iteration ${fmt.int(leftStep)}`;
  else if (Number.isInteger(leftStep) && Number.isInteger(rightStep)) iteration = `iteration ${fmt.int(leftStep)} for the baseline and ${fmt.int(rightStep)} for the variation`;
  const where = [answer?.dataset, answer?.metric, iteration].filter(Boolean).join(', ');
  const oneSeed = ['left', 'right', 'delta'].some((key) => cohort[key]?.sampleSd === null);
  const aggregation = typeof cohort.aggregation === 'string' ? AGGREGATION_TEXT[cohort.aggregation] ?? `Scores combined as ${cohort.aggregation}.` : '';
  const title = `${names.right} against ${names.left}`;
  const counts = countsLine(cohort, n, better);
  const row = (label, hint, stats, signed) =>
    h(
      'tr',
      null,
      h('th', { scope: 'row' }, label, h('span', { class: 'studies__hint' }, hint)),
      SUMMARY_STATS.map(([key]) => statCell(stats?.[key], { signed, better, places: signed ? places.difference : places.value })),
      spreadCell(stats, places.difference),
    );
  return h(
    'div',
    { class: 'studies__cohort' },
    h('h5', { class: 'studies__cohort-title' }, title),
    h('p', { class: 'studies__muted' }, `${where}. ${fmt.plural(n, 'matched seed')}${seeds.length ? `: ${seeds.join(', ')}` : ''}.`),
    counts ? h('p', { class: 'studies__muted' }, counts) : null,
    aggregation ? h('p', { class: 'studies__muted' }, aggregation) : null,
    several && typeof cohort.populationId === 'string'
      ? h('p', { class: 'studies__muted' }, 'Scored images fingerprint ', h('span', { class: 'mono', title: cohort.populationId }, shortDigest(cohort.populationId)))
      : null,
    h(
      'p',
      { class: 'studies__muted' },
      better
        ? `${better === 'higher' ? 'Higher' : 'Lower'} is better for this metric, as the service records it.`
        : 'The service does not say whether higher or lower is better for this metric, so the difference is not marked better or worse.',
    ),
    h(
      'div',
      { class: 'studies__scroll' },
      h(
        'table',
        { class: 'table studies__table' },
        h('caption', { class: 'visually-hidden' }, `Across seeds: ${title}`),
        h(
          'thead',
          null,
          h(
            'tr',
            null,
            h('th', { scope: 'col' }, h('span', { class: 'visually-hidden' }, 'Values')),
            SUMMARY_STATS.map(([, label]) => h('th', { scope: 'col', class: 'num' }, label)),
            h('th', { scope: 'col', class: 'num' }, 'Spread (sample SD)'),
          ),
        ),
        h(
          'tbody',
          null,
          row('Baseline', `${names.left}, left arm`, cohort.left, false),
          row('Variation', `${names.right}, right arm`, cohort.right, false),
          row('Difference per seed', 'variation minus baseline', cohort.delta, true),
        ),
      ),
    ),
    oneSeed ? h('p', { class: 'studies__muted' }, ONE_SEED_TEXT) : null,
  );
}

function excludedLine(entry) {
  const seed = entry?.seed ?? 'not reported';
  const reason = entry?.reason;
  if (typeof reason === 'string' && Object.hasOwn(EXCLUDED_TEXT, reason)) return `Seed ${seed}: ${EXCLUDED_TEXT[reason]}.`;
  return `Seed ${seed}: ${reason === undefined || reason === null ? 'no reason reported' : String(reason)}`;
}

/**
 * The "Across seeds" block of a study comparison: one summary per cohort of `answer.seedSummary`, and the
 * seeds left out. Null when the service sent no seed summary.
 * @param {object} answer  the answer of api.studyCompare
 * @param {(armId: string) => string} [armName]
 * @returns {HTMLElement|null}
 */
export function renderSeedSummary(answer, armName = (id) => String(id ?? '')) {
  const summary = answer?.seedSummary;
  if (!summary || typeof summary !== 'object') return null;
  const cohorts = Array.isArray(summary.cohorts) ? summary.cohorts.filter((c) => c && typeof c === 'object') : [];
  const excluded = Array.isArray(summary.excluded) ? summary.excluded : [];
  const head = h('h4', { class: 'studies__subtitle' }, 'Across seeds');
  const leftOut = excluded.length
    ? h('div', { class: 'studies__cohort' }, h('h5', { class: 'studies__cohort-title' }, 'Left out of this summary'), h('ul', { class: 'studies__excluded' }, excluded.map((entry) => h('li', null, excludedLine(entry)))))
    : null;
  if (!cohorts.length) return h('section', { class: 'studies__seeds', 'aria-label': 'Across seeds' }, head, h('p', { class: 'studies__muted' }, NO_COHORT_TEXT), leftOut);
  const names = { left: armName(answer.leftArmId), right: armName(answer.rightArmId) };
  const places = comparisonPlaces(answer.pairs, cohorts);
  return h(
    'section',
    { class: 'studies__seeds', 'aria-label': 'Across seeds' },
    head,
    h('p', { class: 'studies__muted' }, ACROSS_TEXT),
    cohorts.map((cohort) => cohortBlock(cohort, answer, names, cohorts.length > 1, places)),
    leftOut,
  );
}

/** How a value field is read: JSON when it parses, otherwise plain text. */
export function readValue(raw) {
  const text = String(raw ?? '');
  if (text.trim() === '') return { ok: false, words: 'Enter a value, or choose to remove this option.' };
  try {
    const value = JSON.parse(text);
    let words;
    if (typeof value === 'number') words = `number ${text.trim()}`;
    else if (typeof value === 'boolean') words = `true or false: ${value}`;
    else if (value === null) words = 'null (no value)';
    else if (typeof value === 'string') words = `text ${value === '' ? '(empty)' : value}`;
    else if (Array.isArray(value)) words = `list of ${fmt.plural(value.length, 'item')}`;
    else words = `mapping of ${fmt.plural(Object.keys(value).length, 'key')}`;
    return { ok: true, value, words };
  } catch {
    return { ok: true, value: text, words: `text ${text}` };
  }
}

/** "0, 1, 2" -> {seeds: [0, 1, 2]} or {error}. Seeds are distinct whole numbers from 0 through 4294967295. */
export function readSeeds(raw) {
  const parts = String(raw ?? '')
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part !== '');
  if (!parts.length) return { seeds: [], error: 'Enter at least one seed, for example 0, 1, 2.' };
  const seeds = [];
  for (const part of parts) {
    if (!/^\d+$/.test(part)) return { seeds: [], error: `${part} is not a whole number. Seeds are whole numbers from 0 through 4,294,967,295.` };
    const n = Number(part);
    if (n > MAX_SEED) return { seeds: [], error: `${part} is above 4,294,967,295, the largest seed.` };
    if (seeds.includes(n)) return { seeds: [], error: `Seed ${part} is listed twice. Each seed may appear once.` };
    seeds.push(n);
  }
  return { seeds, error: '' };
}

// ───────── create dialog ─────────

/**
 * Create a study from a recipe text. Resolves to the created study, or null when closed without creating.
 * Opens the study after creating it.
 * @param {{yaml?: string, sourceLabel?: string, name?: string}} options
 * @returns {Promise<object|null>}
 */
export function openStudyDialog({ yaml = '', sourceLabel = '', name = '' } = {}) {
  const requestId = newRequestId(); // one per opening, reused on retry
  const nameId = uid('study-name');
  const seedsId = uid('study-seeds');
  const nameInput = h('input', { class: 'input', id: nameId, type: 'text', value: name, autocomplete: 'off', spellcheck: 'false', maxlength: '120' });
  const nameError = h('p', { class: 'field__error', role: 'alert' });
  const seedsInput = h('input', { class: 'input', id: seedsId, type: 'text', value: '', autocomplete: 'off', spellcheck: 'false', placeholder: 'for example 0, 1, 2', 'aria-describedby': `${seedsId}-help` });
  const seedsError = h('p', { class: 'field__error', role: 'alert' });
  const countLine = h('p', { class: 'studies__count', role: 'status', 'aria-live': 'polite' });
  const generalIssues = h('div', { class: 'studies__issues', 'aria-live': 'polite' });
  const armsList = h('ol', { class: 'studies__arm-list' });
  const arms = []; // [{el, nameInput, edits: [...], issues}]
  let busy = false;
  let createdStudy = null;

  function addEdit(arm, initial = {}) {
    const pathId = uid('study-edit-path');
    const valueId = uid('study-edit-value');
    const removeId = uid('study-edit-remove');
    const edit = {};
    edit.path = h('input', { class: 'input mono', id: pathId, type: 'text', value: initial.path ?? '', placeholder: 'train.optim_g.lr', autocomplete: 'off', spellcheck: 'false' });
    edit.value = h('input', { class: 'input mono', id: valueId, type: 'text', value: initial.value ?? '', autocomplete: 'off', spellcheck: 'false', 'aria-describedby': `${valueId}-read` });
    edit.remove = h('input', { type: 'checkbox', id: removeId, onchange: () => update() });
    edit.reading = h('span', { class: 'studies__reading', id: `${valueId}-read` });
    edit.el = h(
      'li',
      { class: 'studies__edit' },
      h('div', { class: 'studies__edit-field' }, h('label', { class: 'field__label', for: pathId }, 'Option path'), edit.path),
      h('div', { class: 'studies__edit-field' }, h('label', { class: 'field__label', for: valueId }, 'New value'), edit.value, edit.reading),
      h('label', { class: 'studies__check', for: removeId }, edit.remove, 'Remove this option'),
      h(
        'button',
        {
          class: 'btn btn--quiet btn--sm',
          type: 'button',
          onclick: () => {
            arm.edits.splice(arm.edits.indexOf(edit), 1);
            edit.el.remove();
            update();
            arm.addEditBtn.focus();
          },
        },
        'Remove edit',
      ),
    );
    edit.path.addEventListener('input', () => update());
    edit.value.addEventListener('input', () => update());
    arm.edits.push(edit);
    arm.editList.appendChild(edit.el);
    return edit;
  }

  function addArm(armName = '') {
    const armNameId = uid('study-arm-name');
    const arm = { edits: [] };
    arm.nameInput = h('input', { class: 'input', id: armNameId, type: 'text', value: armName, autocomplete: 'off', spellcheck: 'false', maxlength: '120', oninput: () => update() });
    arm.editList = h('ul', { class: 'studies__edit-list' });
    arm.issues = h('div', { class: 'studies__issues' });
    arm.none = h('p', { class: 'studies__muted' }, 'No changes: the base recipe.');
    arm.addEditBtn = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => { const e = addEdit(arm); update(); e.path.focus(); } }, 'Add an edit');
    arm.removeBtn = h(
      'button',
      {
        class: 'btn btn--quiet btn--sm',
        type: 'button',
        onclick: () => {
          arms.splice(arms.indexOf(arm), 1);
          arm.el.remove();
          update();
          addArmBtn.focus();
        },
      },
      'Remove arm',
    );
    arm.el = h(
      'li',
      { class: 'studies__arm-edit' },
      h('div', { class: 'studies__edit-field' }, h('label', { class: 'field__label', for: armNameId }, 'Arm name'), arm.nameInput),
      arm.none,
      arm.editList,
      h('div', { class: 'row' }, arm.addEditBtn, arm.removeBtn),
      arm.issues,
    );
    arms.push(arm);
    armsList.appendChild(arm.el);
    return arm;
  }

  const addArmBtn = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => { const a = addArm(''); update(); a.nameInput.focus(); } }, 'Add an arm');

  /** Validate everything; returns the request body or null. Also refreshes the readings and the count. */
  function collect(showErrors) {
    let ok = true;
    const studyName = nameInput.value.trim();
    nameError.textContent = '';
    if (!studyName) {
      ok = false;
      if (showErrors) nameError.textContent = 'Name the study.';
    }
    const { seeds, error } = readSeeds(seedsInput.value);
    seedsError.textContent = showErrors || seedsInput.value.trim() ? error : '';
    if (error) ok = false;
    const seen = new Set();
    const body = [];
    for (const arm of arms) {
      const armName = arm.nameInput.value.trim();
      const problems = [];
      if (!armName) problems.push('Name this arm.');
      else if (seen.has(armName.toLowerCase())) problems.push(`The arm name ${armName} is used twice.`);
      seen.add(armName.toLowerCase());
      arm.none.hidden = arm.edits.length > 0;
      const edits = [];
      for (const edit of arm.edits) {
        const path = parsePath(edit.path.value.trim());
        const removing = edit.remove.checked;
        edit.value.disabled = removing;
        const reading = removing ? { ok: true, words: 'This option is removed from the recipe.' } : readValue(edit.value.value);
        edit.reading.textContent = edit.value.value || removing ? reading.words : '';
        if (!path.length) problems.push('Enter an option path for every edit, for example train.optim_g.lr.');
        else if (path[0] === 'manual_seed') problems.push('Set manual_seed through the study seeds, not as an edit.');
        if (!reading.ok) problems.push(reading.words);
        edits.push(removing ? { path, remove: true } : { path, value: reading.value });
      }
      if (showErrors && problems.length) mount(arm.issues, notice('err', [...new Set(problems)].join(' ')));
      else if (!problems.length) mount(arm.issues);
      if (problems.length) ok = false;
      body.push({ name: armName, edits });
    }
    const a = arms.length;
    const s = seeds.length;
    const n = a * s;
    countLine.textContent = `${fmt.plural(a, 'arm')} × ${fmt.plural(s, 'seed')} = ${fmt.plural(n, 'entry', 'entries')} (limit ${LIMIT})`;
    if (a < 2) {
      ok = false;
      if (showErrors) mount(generalIssues, notice('err', 'A study needs at least two arms.'));
    } else if (n > LIMIT) {
      ok = false;
      mount(generalIssues, notice('err', `That is ${fmt.int(n)} entries. A study can hold at most ${LIMIT}: remove arms or seeds.`));
    } else if (showErrors || generalIssues.firstChild) {
      mount(generalIssues);
    }
    if (!ok || !yaml) return null;
    return { name: studyName, baseYaml: yaml, arms: body, seeds, requestId };
  }

  function update() {
    collect(false);
  }

  seedsInput.addEventListener('input', update);
  nameInput.addEventListener('input', () => {
    if (nameInput.value.trim()) nameError.textContent = '';
  });

  addArm('Baseline');
  addArm('');

  const body = yaml
    ? h(
        'div',
        { class: 'studies__form' },
        h('p', null, h('strong', null, HOLD_TEXT)),
        h('p', { class: 'studies__muted' }, 'Base recipe: ', sourceLabel || 'the recipe text given'),
        h('div', { class: 'field' }, h('label', { class: 'field__label', for: nameId }, 'Study name'), nameInput, nameError),
        h(
          'div',
          { class: 'field' },
          h('label', { class: 'field__label', for: seedsId }, 'Seeds'),
          seedsInput,
          h('p', { class: 'field__help', id: `${seedsId}-help` }, 'Whole numbers separated by commas. Every arm trains once with each seed.'),
          seedsError,
        ),
        h('h3', { class: 'studies__subtitle' }, 'Arms'),
        h(
          'p',
          { class: 'studies__muted' },
          'Each arm is the base recipe with its edits. An option path is dotted, for example train.optim_g.lr. A value is read as JSON when it parses (numbers, true, false, null, lists, quoted text), otherwise as plain text.',
        ),
        armsList,
        addArmBtn,
        countLine,
        generalIssues,
      )
    : notice('err', 'There is no recipe text to start a study from. Open a recipe on the Recipe screen first.');

  update();

  const { closed } = openDialog({
    title: 'New study',
    body,
    wide: true,
    foot: (close) => {
      const createBtn = h('button', { class: 'btn btn--primary', type: 'button' }, 'Create study');
      createBtn.addEventListener('click', async () => {
        if (busy) return;
        const request = collect(true);
        if (!request) return;
        busy = true;
        createBtn.setAttribute('aria-disabled', 'true');
        mount(generalIssues, loading('Creating the study…'));
        for (const arm of arms) mount(arm.issues);
        try {
          const study = await api.createStudy(request);
          createdStudy = study;
          toast(`Created ${study?.name ?? request.name}: ${fmt.plural(study?.entryCount ?? request.arms.length * request.seeds.length, 'entry', 'entries')} on hold.`, { kind: 'ok' });
          queue.refresh();
          close(study);
          if (study?.id) go(studiesHref(study.id));
        } catch (error) {
          if (error?.status === 422 && Array.isArray(error.issues)) {
            const general = [];
            for (const issue of error.issues) {
              const index = typeof issue.armId === 'string' && /^arm-\d+$/.test(issue.armId) ? Number(issue.armId.slice(4)) - 1 : -1;
              const arm = arms[index] ?? arms.find((a) => a.nameInput.value.trim() === issue.armName);
              const words = [issue.seed !== undefined ? `Seed ${issue.seed}` : '', issue.path ? `${issue.path}` : '', issue.message].filter(Boolean).join(': ');
              const line = h('p', { class: 'studies__issue' }, issue.armName ? `${issue.armName}. ` : '', words);
              if (arm) arm.issues.appendChild(notice('err', line));
              else general.push(line);
            }
            mount(generalIssues, notice('err', [h('p', null, 'The study was not created: some arms do not render or do not pass the recipe check. Each problem is shown beside its arm.'), general]));
          } else {
            mount(generalIssues, notice('err', `The study was not created: ${error?.message ?? 'no answer'}`));
          }
        } finally {
          busy = false;
          createBtn.removeAttribute('aria-disabled');
        }
      });
      return [h('button', { class: 'btn', type: 'button', onclick: () => close(null) }, 'Close'), yaml ? createBtn : null];
    },
  });
  return closed.then((value) => value ?? createdStudy ?? null);
}

/**
 * The answer of a study comparison: one row per seed, then the summary across seeds.
 * @param {object} answer  the answer of api.studyCompare
 * @param {Array<{id: string, name: string}>} arms  the study's arms, for their names
 * @returns {Node[]}
 */
export function renderComparison(answer, arms) {
  const name = (id) => arms.find((arm) => arm.id === id)?.name ?? id;
  const pairs = Array.isArray(answer?.pairs) ? answer.pairs : [];
  const values = pairs.map(pairValues);
  const shown = values.filter(Boolean);
  // Whole-dataset (pooled) scores are one value per run, not a mean of per-image scores: the heads say
  // so when every compared pair is pooled, and each pooled row says so when the table mixes both kinds.
  const allPooled = shown.length > 0 && shown.every((v) => v.pooled);
  const mixed = !allPooled && shown.some((v) => v.pooled);
  const { value: scorePlaces, difference: deltaPlaces } = comparisonPlaces(pairs);
  const rows = pairs.map((p, i) => {
    const c = p.comparison;
    const v = values[i];
    let detail = null;
    if (p.status === 'incompatible' && c) detail = c.differences ? `Recorded with different settings: ${Object.keys(c.differences).join(', ')}.` : c.reason || 'No image matched between the two evaluations.';
    const iteration = c && Number.isFinite(c.leftStep) ? (c.leftStep === c.rightStep ? fmt.int(c.leftStep) : `${fmt.int(c.leftStep)} and ${fmt.int(c.rightStep)}`) : '';
    const direction = c?.better === 'higher' ? 'higher is better' : c?.better === 'lower' ? 'lower is better' : '';
    const sum = v?.summary;
    return h(
      'tr',
      null,
      h('th', { scope: 'row', class: 'num' }, fmt.int(p.seed)),
      h('td', null, statusBadge(p.leftStatus)),
      h('td', null, statusBadge(p.rightStatus)),
      h(
        'td',
        null,
        h('span', null, PAIR_TEXT[p.status] ?? String(p.status ?? 'Not reported')),
        detail ? h('span', { class: 'studies__hint' }, detail) : null,
        p.error ? h('span', { class: 'studies__hint' }, p.error) : null,
        mixed && v?.pooled ? h('span', { class: 'studies__hint' }, 'Whole-dataset scores, not means of per-image scores') : null,
      ),
      h('td', { class: 'num' }, iteration),
      h('td', { class: 'num', title: v ? exact(v.left) : null }, v ? fixed(v.left, scorePlaces) : ''),
      h('td', { class: 'num', title: v ? exact(v.right) : null }, v ? fixed(v.right, scorePlaces) : ''),
      h('td', { class: 'num', title: v ? exact(v.delta) : null }, v ? signedFixed(v.delta, deltaPlaces) : '', v && direction ? h('span', { class: 'studies__hint' }, direction) : null),
      h('td', { class: 'num' }, sum ? `${fmt.int(sum.improved)} / ${fmt.int(sum.tied)} / ${fmt.int(sum.regressed)}` : v?.pooled ? h('span', { class: 'studies__hint' }, 'No per-image count') : ''),
      h('td', null, p.leftRunId && p.rightRunId ? h('a', { href: compareHref(p.leftRunId, p.rightRunId) }, 'Open this pair') : ''),
    );
  });
  const heads = allPooled
    ? ['Whole-dataset score, left', 'Whole-dataset score, right', 'Difference (right minus left)']
    : mixed
      ? ['Score, left', 'Score, right', 'Difference (right minus left)']
      : ['Mean, left', 'Mean, right', 'Mean difference (right minus left)'];
  return [
    h('p', { class: 'studies__muted' }, `Left: ${name(answer.leftArmId)}. Right: ${name(answer.rightArmId)}. ${answer.dataset}, ${answer.metric}, ${Number.isInteger(answer.step) ? `iteration ${fmt.int(answer.step)}` : 'latest common iteration of each pair'}.`),
    h(
      'div',
      { class: 'studies__scroll' },
      h(
        'table',
        { class: 'table studies__table' },
        h('caption', { class: 'visually-hidden' }, allPooled ? 'Comparison of the two arms, one row per seed, by whole-dataset scores' : 'Comparison of the two arms, one row per seed'),
        h(
          'thead',
          null,
          h(
            'tr',
            null,
            h('th', { scope: 'col' }, 'Seed'),
            h('th', { scope: 'col' }, 'Left entry'),
            h('th', { scope: 'col' }, 'Right entry'),
            h('th', { scope: 'col' }, 'Pair'),
            h('th', { scope: 'col', class: 'num' }, 'Iteration'),
            heads.map((head) => h('th', { scope: 'col', class: 'num' }, head)),
            h('th', { scope: 'col', class: 'num' }, 'Improved / tied / regressed'),
            h('th', { scope: 'col' }, h('span', { class: 'visually-hidden' }, 'Link')),
          ),
        ),
        h('tbody', null, rows),
      ),
    ),
    h('p', { class: 'studies__muted' }, PER_SEED_TEXT),
    renderSeedSummary(answer, name),
  ];
}

// ───────── the view ─────────

/**
 * @param {HTMLElement} host
 * @param {{studyId?: string|null}} options
 * @returns {{setStudy(id: string|null): void, destroy(): void}}
 */
export function mountStudiesView(host, { studyId = null } = {}) {
  let current = studyId || null;
  let destroyed = false;
  let timer = 0;
  let listCtl = null;
  let detailCtl = null;
  let study = null; // the last detail answer for `current`
  let detailRefs = null; // in-place parts of the rendered detail
  let listSignature = '';

  const newBtn = h('button', { class: 'btn btn--primary btn--sm', type: 'button', onclick: newFromOpenRecipe }, 'New study from the open recipe');
  const newHelp = h('div', { class: 'studies__new-help', 'aria-live': 'polite' });
  const listEl = h('ul', { class: 'studies__list', 'aria-label': 'Studies, newest first' });
  const listStatus = h('div', { class: 'studies__status', role: 'status' }, loading('Reading the studies…'));
  const side = h('nav', { class: 'studies__side', 'aria-label': 'Studies' }, h('div', { class: 'studies__side-head' }, h('h2', { class: 'studies__title' }, 'Studies'), newBtn), newHelp, listStatus, listEl);
  const main = h('div', { class: 'studies__main' });
  const root = h('div', { class: 'studies' }, side, main);
  host.appendChild(root);

  function newFromOpenRecipe() {
    const r = recipe.get();
    if (r.phase === 'open' && typeof r.yaml === 'string' && r.yaml) {
      mount(newHelp);
      openStudyDialog({ yaml: r.yaml, sourceLabel: r.name ? `the open recipe, ${r.name}` : 'the open recipe', name: '' });
    } else {
      mount(newHelp, notice('warn', h('span', null, 'A study starts from one recipe, and no recipe is open. ', h('a', { href: '#/recipe' }, 'Open a recipe on the Recipe screen'), ', then come back here.')));
    }
  }

  // ───────── list ─────────

  async function refreshList() {
    listCtl?.abort();
    listCtl = new AbortController();
    try {
      const list = await api.studies({ signal: listCtl.signal });
      if (destroyed) return;
      renderList(Array.isArray(list) ? list : []);
    } catch (error) {
      if (isAbort(error) || destroyed) return;
      mount(listStatus, notice('err', `Could not read the studies: ${error?.message ?? 'no answer'}`));
    }
  }

  function renderList(list) {
    const signature = JSON.stringify([current, list.map((s) => [s.id, s.name, s.createdAt, s.armCount, s.seedCount, s.entryCount])]);
    if (signature === listSignature) return;
    listSignature = signature;
    mount(listStatus);
    if (!list.length) {
      mount(listEl);
      mount(listStatus, h('p', { class: 'studies__muted' }, 'No studies yet.'));
      if (!current) renderEmptyMain();
      return;
    }
    // Rebuilt only when the list changed; focus on a study link moves to the same study's new link.
    const focusedId = listEl.contains(document.activeElement) ? document.activeElement?.dataset?.study : null;
    mount(
      listEl,
      list.map((s) =>
        h(
          'li',
          { class: 'studies__item' },
          h(
            'a',
            { class: 'studies__link', href: studiesHref(s.id), dataset: { study: s.id }, 'aria-current': s.id === current ? 'page' : null },
            h('span', { class: 'studies__name' }, s.name || s.id),
            h('span', { class: 'studies__meta' }, fmt.dateTime(s.createdAt) || 'Created time not reported'),
            h('span', { class: 'studies__meta' }, `${fmt.plural(s.armCount, 'arm')} × ${fmt.plural(s.seedCount, 'seed')}, ${fmt.plural(s.entryCount, 'entry', 'entries')}`),
          ),
        ),
      ),
    );
    if (focusedId) [...listEl.querySelectorAll('a')].find((a) => a.dataset.study === focusedId)?.focus();
    if (!current) renderPickMain();
  }

  function renderEmptyMain() {
    detailRefs = null;
    mount(main, emptyState({ title: 'No studies yet', text: STUDY_TEXT, actions: [h('button', { class: 'btn btn--primary', type: 'button', onclick: newFromOpenRecipe }, 'New study from the open recipe')] }));
  }

  function renderPickMain() {
    detailRefs = null;
    mount(main, emptyState({ title: 'Choose a study', text: `${STUDY_TEXT} Choose one in the list to see its arms, its entries and how two arms compare.` }));
  }

  // ───────── detail ─────────

  /** A study entry with the fresher state the queue store may already hold. */
  function merged(entry) {
    const fresh = queue.byId(entry.id);
    if (!fresh) return entry;
    return { ...entry, status: fresh.status ?? entry.status, runId: fresh.runId ?? entry.runId, runStatus: fresh.runStatus ?? entry.runStatus };
  }

  const cellsOf = (s) => (s?.arms ?? []).flatMap((arm) => (arm.entries ?? []).map((entry) => ({ arm, entry: merged(entry) })));

  async function refreshDetail() {
    const id = current;
    if (!id) return;
    detailCtl?.abort();
    detailCtl = new AbortController();
    if (!detailRefs || detailRefs.id !== id) mount(main, loading('Reading the study…'));
    try {
      const answer = await api.study(id, { signal: detailCtl.signal });
      if (destroyed || id !== current) return;
      study = answer;
      const shape = JSON.stringify([answer.id, answer.seeds, (answer.arms ?? []).map((a) => [a.id, a.name, a.edits])]);
      if (!detailRefs || detailRefs.id !== id || detailRefs.shape !== shape) buildDetail(answer, shape);
      else updateDetail();
    } catch (error) {
      if (isAbort(error) || destroyed || id !== current) return;
      if (!detailRefs || detailRefs.id !== id) {
        detailRefs = null;
        mount(main, notice('err', `Could not read this study: ${error?.message ?? 'no answer'}`));
      }
    } finally {
      schedule();
    }
  }

  function buildDetail(s, shape) {
    const refs = { id: s.id, shape, cells: new Map(), armButtons: new Map(), seedButtons: new Map() };
    detailRefs = refs;
    const seeds = Array.isArray(s.seeds) ? s.seeds : [];
    const arms = Array.isArray(s.arms) ? s.arms : [];

    const header = h(
      'header',
      { class: 'studies__head' },
      h('h2', { class: 'studies__title' }, s.name || s.id),
      h(
        'dl',
        { class: 'kv studies__kv' },
        h('dt', null, 'Created'), h('dd', null, timeEl(s.createdAt)),
        h('dt', null, 'Base recipe fingerprint'), h('dd', null, s.baseDigest ? h('span', { class: 'mono', title: s.baseDigest }, shortDigest(s.baseDigest)) : 'Not reported'),
        h('dt', null, 'Size'), h('dd', null, `${fmt.plural(s.armCount ?? arms.length, 'arm')} × ${fmt.plural(s.seedCount ?? seeds.length, 'seed')}, ${fmt.plural(s.entryCount ?? arms.length * seeds.length, 'entry', 'entries')}`),
      ),
      h(
        'div',
        { class: 'row' },
        h('button', { class: 'btn btn--sm', type: 'button', onclick: () => openAsRecipe({ yaml: s.baseYaml, name: s.name, label: `Base recipe of the study ${s.name}` }) }, 'Open the base recipe'),
        h(
          'button',
          {
            class: 'btn btn--sm',
            type: 'button',
            'aria-label': 'Copy the base recipe text',
            onclick: async () => toast((await copyText(s.baseYaml ?? '')) ? 'Copied the base recipe text.' : 'The base recipe text could not be copied.', { kind: 'ok' }),
          },
          'Copy',
        ),
      ),
      h('p', { class: 'studies__muted' }, HOLD_TEXT),
    );

    const armsSection = h(
      'section',
      { class: 'studies__section', 'aria-label': 'Arms' },
      h('h3', { class: 'studies__subtitle' }, 'Arms'),
      h(
        'div',
        { class: 'studies__arms' },
        arms.map((arm) =>
          h(
            'div',
            { class: 'studies__arm' },
            h('h4', { class: 'studies__arm-name' }, arm.name),
            Array.isArray(arm.edits) && arm.edits.length
              ? h(
                  'table',
                  { class: 'table studies__table' },
                  h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Option'), h('th', { scope: 'col' }, 'New value'))),
                  h(
                    'tbody',
                    null,
                    arm.edits.map((edit) =>
                      h(
                        'tr',
                        null,
                        h('th', { scope: 'row', class: 'mono studies__value' }, dotted(edit.path)),
                        edit.remove ? h('td', { class: 'studies__muted' }, 'removed') : h('td', { class: 'mono studies__value' }, valueText(edit.value)),
                      ),
                    ),
                  ),
                )
              : h('p', { class: 'studies__muted' }, 'No changes: the base recipe.'),
          ),
        ),
      ),
    );

    refs.counts = h('p', { class: 'studies__counts', role: 'status', 'aria-live': 'polite' });
    refs.releaseStatus = h('div', { class: 'studies__release-status', 'aria-live': 'polite' });
    refs.allButton = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => releaseScope('all') }, 'Release all on hold');

    const headRow = h(
      'tr',
      null,
      h('th', { scope: 'col' }, 'Arm'),
      seeds.map((seed) => {
        const button = h('button', { class: 'btn btn--quiet btn--sm', type: 'button', onclick: () => releaseScope('seed', seed) }, 'Release this seed');
        refs.seedButtons.set(seed, button);
        return h('th', { scope: 'col', class: 'studies__seed-head' }, h('span', { class: 'studies__seed' }, `Seed ${seed}`), button);
      }),
    );
    const bodyRows = arms.map((arm) => {
      const button = h('button', { class: 'btn btn--quiet btn--sm', type: 'button', onclick: () => releaseScope('arm', arm.id) }, 'Release this arm');
      refs.armButtons.set(arm.id, button);
      return h(
        'tr',
        null,
        h('th', { scope: 'row', class: 'studies__arm-head' }, h('span', { class: 'studies__arm-name' }, arm.name), button),
        (arm.entries ?? []).map((entry) => {
          const td = h('td', { class: 'studies__cell' });
          refs.cells.set(entry.id, { td, sig: '', arm, seed: entry.seed });
          return td;
        }),
      );
    });

    const grid = h(
      'section',
      { class: 'studies__section', 'aria-label': 'Entries' },
      h('h3', { class: 'studies__subtitle' }, 'Entries'),
      h('p', { class: 'studies__muted' }, 'One row per arm, one column per seed. Each cell is a queue entry.'),
      refs.counts,
      h('div', { class: 'row row--wrap' }, refs.allButton),
      refs.releaseStatus,
      h('div', { class: 'studies__scroll' }, h('table', { class: 'table studies__grid' }, h('caption', { class: 'visually-hidden' }, `Entries of ${s.name}: arms by seeds`), h('thead', null, headRow), h('tbody', null, bodyRows))),
    );

    mount(main, header, armsSection, grid, comparePart(s));
    updateDetail();
  }

  function cellSignature(entry) {
    return JSON.stringify([entry.status, entry.runId ?? null, entry.runStatus ?? null, entry.error ?? null]);
  }

  function renderCell(entry, arm) {
    const st = entryStatus(entry.status);
    const where = `${arm.name}, seed ${entry.seed}`;
    const parts = [statusBadge(entry.status), h('a', { class: 'studies__cell-link', href: queueHref(entry.id), 'aria-label': `Queue entry for ${where}`, dataset: { focus: 'entry' } }, 'Queue entry')];
    if (entry.runId) {
      const rs = entry.runStatus ? runStatus(entry.runStatus) : null;
      parts.push(h('a', { class: 'studies__cell-link', href: runHref(entry.runId), 'aria-label': `Run for ${where}${rs ? `, ${rs.label}` : ''}`, dataset: { focus: 'run' } }, rs ? `Run: ${rs.label.toLowerCase()}` : 'Run'));
    }
    const action = entry.status === 'held' ? 'release' : entry.status === 'pending' ? 'hold' : null;
    const buttons = [];
    if (action) buttons.push(actionButton(action, entry, where));
    if (ENTRY_ACTIONS.cancel.from.includes(entry.status)) buttons.push(actionButton('cancel', entry, where));
    if (buttons.length) parts.push(h('div', { class: 'studies__cell-actions' }, buttons));
    if (st.kind === 'err' && entry.error) parts.push(h('p', { class: 'studies__cell-error' }, entry.error));
    return parts;
  }

  const busy = new Set();

  function actionButton(action, entry, where) {
    const label = ENTRY_ACTIONS[action].label;
    const key = `${action}|${entry.id}`;
    return h(
      'button',
      {
        class: `btn btn--sm${action === 'cancel' ? ' btn--quiet' : ''}`,
        type: 'button',
        'aria-label': `${label}: ${where}`,
        'aria-disabled': busy.has(key) ? 'true' : null,
        dataset: { focus: action },
        onclick: () => runAction(action, entry, where, key),
      },
      label,
    );
  }

  async function runAction(action, entry, where, key) {
    if (busy.has(key)) return;
    if (action === 'cancel') {
      // Same wording as the queue view's cancel confirmation.
      const ok = await confirmDialog({
        title: `Cancel ${where}?`,
        body: 'Cancelling cannot be undone. The entry will never start; it stays in the finished list, where its recipe text can still be opened.',
        confirmLabel: 'Cancel entry',
        cancelLabel: 'Keep entry',
        danger: true,
      });
      if (!ok) return;
    }
    busy.add(key);
    try {
      await queue[action](entry.id);
      const done = { release: `Released ${where}.`, hold: `Put ${where} on hold.`, cancel: `Cancelled ${where}.` }[action];
      toast(action === 'release' && queue.get().paused ? `${done} The queue is paused, so it waits until you continue the queue.` : done, { kind: 'ok' });
    } catch (error) {
      toast(`${where} was not changed: ${error?.message ?? 'no answer'}`, { kind: 'err' });
    } finally {
      busy.delete(key);
      if (!destroyed) refreshDetail();
    }
  }

  /** Update cells, counts and release buttons in place; a cell is rebuilt only when it changed. */
  function updateDetail() {
    const refs = detailRefs;
    if (!refs || !study || study.id !== refs.id) return;
    const cells = cellsOf(study);
    const active = document.activeElement;
    for (const { arm, entry } of cells) {
      const slot = refs.cells.get(entry.id);
      if (!slot) continue;
      const sig = cellSignature(entry);
      if (slot.sig === sig) continue;
      const hadFocus = slot.td.contains(active);
      const focusKey = hadFocus ? active?.dataset?.focus ?? null : null;
      slot.sig = sig;
      mount(slot.td, renderCell(entry, arm));
      if (hadFocus) {
        // The same control again, or the entry link when that action no longer applies.
        const target = [...slot.td.querySelectorAll('a, button')].find((el) => el.dataset.focus === focusKey) ?? slot.td.querySelector('a');
        target?.focus();
      }
    }
    const count = (pred) => cells.filter(({ entry }) => pred(entry.status)).length;
    const words = [
      ['on hold', count((s) => s === 'held')],
      ['released', count((s) => s === 'pending')],
      ['training', count((s) => s === 'running' || s === 'stopping')],
      ['completed', count((s) => s === 'completed')],
      ['failed', count((s) => s === 'failed')],
      ['cancelled', count((s) => s === 'cancelled')],
    ];
    const stopped = count((s) => s === 'stopped');
    if (stopped) words.push(['stopped', stopped]);
    const text = words.map(([w, n]) => `${fmt.int(n)} ${w}`).join(', ');
    if (refs.counts.textContent !== text) refs.counts.textContent = `${text}.`;
    const held = (pred) => cells.filter(({ arm, entry }) => entry.status === 'held' && pred(arm, entry)).length;
    setButton(refs.allButton, `Release all on hold (${fmt.int(held(() => true))})`, held(() => true));
    for (const [armId, button] of refs.armButtons) setButton(button, `Release this arm (${fmt.int(held((arm) => arm.id === armId))})`, held((arm) => arm.id === armId));
    for (const [seed, button] of refs.seedButtons) setButton(button, `Release this seed (${fmt.int(held((arm, entry) => entry.seed === seed))})`, held((arm, entry) => entry.seed === seed));
  }

  function setButton(button, text, n) {
    if (button.textContent !== text) button.textContent = text;
    if (n > 0 && !releasing) button.removeAttribute('aria-disabled');
    else button.setAttribute('aria-disabled', 'true');
  }

  let releasing = false;

  /** Release the held cells of one scope, one after another, after a confirmation; stop at the first refusal. */
  async function releaseScope(scope, key) {
    if (releasing || !study) return;
    const all = cellsOf(study).filter(({ entry }) => entry.status === 'held');
    const chosen = scope === 'arm' ? all.filter(({ arm }) => arm.id === key) : scope === 'seed' ? all.filter(({ entry }) => entry.seed === key) : all;
    // all: arm by arm, seeds in study order; a seed column: arms in study order (cellsOf keeps that order)
    if (!chosen.length) return;
    const n = chosen.length;
    const scopeWords = scope === 'arm' ? ` of the arm ${chosen[0].arm.name}` : scope === 'seed' ? ` with seed ${key}` : '';
    const ok = await confirmDialog({
      title: `Release ${fmt.plural(n, 'entry', 'entries')}${scopeWords}?`,
      body: h(
        'div',
        null,
        h('p', null, `${fmt.plural(n, 'entry', 'entries')} will be authorised to train. They run one at a time, in the order they are released, whenever the trainer is free.`),
        queue.get().paused ? h('p', null, 'The queue is paused, so none of them starts until you continue the queue.') : null,
      ),
      confirmLabel: `Release ${fmt.plural(n, 'entry', 'entries')}`,
      cancelLabel: 'Keep on hold',
    });
    if (!ok || destroyed) return;
    releasing = true;
    const refs = detailRefs;
    let done = 0;
    try {
      for (const { arm, entry } of chosen) {
        if (refs) mount(refs.releaseStatus, loading(`Releasing ${done + 1} of ${n}: ${arm.name}, seed ${entry.seed}…`));
        try {
          await queue.release(entry.id);
          done += 1;
        } catch (error) {
          if (refs) mount(refs.releaseStatus, notice('err', `Released ${done} of ${n}. ${arm.name}, seed ${entry.seed} was not released: ${error?.message ?? 'no answer'} The rest stay on hold.`));
          return;
        }
      }
      if (refs) mount(refs.releaseStatus, notice('ok', `Released ${fmt.plural(done, 'entry', 'entries')}.`));
    } finally {
      releasing = false;
      if (!destroyed) refreshDetail();
    }
  }

  // ───────── compare two arms ─────────

  function comparePart(s) {
    const arms = Array.isArray(s.arms) ? s.arms : [];
    const ids = { left: uid('study-left'), right: uid('study-right'), dataset: uid('study-dataset'), metric: uid('study-metric'), step: uid('study-step') };
    // The dataset and metric names the linked runs recorded, once read (see below); other names can be typed.
    const known = { datasets: [], metrics: [] };
    const armSelect = (id, index) => {
      const select = h('select', { class: 'select', id }, arms.map((arm) => h('option', { value: arm.id }, arm.name)));
      if (arms[index]) select.value = arms[index].id;
      return select;
    };
    const left = armSelect(ids.left, 0);
    const right = armSelect(ids.right, 1);
    const datasetCombo = createCombobox({ id: ids.dataset, listLabel: 'Datasets the runs were validated on', getChoices: () => toChoices(known.datasets) });
    const metricCombo = createCombobox({ id: ids.metric, listLabel: 'Metrics the runs recorded', getChoices: () => toChoices(known.metrics) });
    const dataset = datasetCombo.input;
    const metric = metricCombo.input;
    const modeName = uid('study-step-mode');
    const latest = h('input', { type: 'radio', name: modeName, id: `${ids.step}-latest`, checked: true, onchange: () => { step.disabled = true; } });
    const chosen = h('input', { type: 'radio', name: modeName, id: `${ids.step}-chosen`, onchange: () => { step.disabled = false; step.focus(); } });
    const step = h('input', { class: 'input input--num', id: ids.step, type: 'number', min: '0', step: '1', disabled: true, 'aria-label': 'Iteration for every pair' });
    const error = h('p', { class: 'field__error', role: 'alert' });
    const result = h('div', { class: 'studies__result', 'aria-live': 'polite' });
    let ctl = null;

    const field = (id, label, control) => h('div', { class: 'studies__field' }, h('label', { class: 'field__label', for: id }, label), control);
    const el = h(
      'section',
      { class: 'studies__section', 'aria-label': 'Compare two arms' },
      h('h3', { class: 'studies__subtitle' }, 'Compare two arms'),
      h('p', { class: 'studies__muted' }, 'Compares the recorded per-image validation scores of the two arms, seed by seed, for the entries that have a run.'),
      h(
        'div',
        { class: 'studies__controls' },
        field(ids.left, 'Left arm', left),
        field(ids.right, 'Right arm', right),
        field(ids.dataset, 'Dataset', datasetCombo.el),
        field(ids.metric, 'Metric', metricCombo.el),
        h(
          'fieldset',
          { class: 'studies__fieldset' },
          h('legend', { class: 'field__label' }, 'Iteration'),
          h('label', { class: 'studies__check', for: `${ids.step}-latest` }, latest, 'Latest common iteration of each pair'),
          h('label', { class: 'studies__check', for: `${ids.step}-chosen` }, chosen, 'This iteration for every pair:'),
          step,
        ),
        h('button', { class: 'btn btn--primary', type: 'button', onclick: run }, 'Compare'),
      ),
      error,
      result,
    );

    // Suggestions from the recorded validation of a linked run (at most three runs are asked).
    (async () => {
      const runIds = [...new Set(cellsOf(s).map(({ entry }) => entry.runId).filter(Boolean))].slice(0, 3);
      const datasets = new Set();
      const metrics = new Set();
      for (const runId of runIds) {
        try {
          const reply = await api.validation(runId, { limit: 5 });
          for (const ev of reply?.evaluations ?? []) {
            if (ev.dataset) datasets.add(ev.dataset);
            for (const key of Object.keys(ev.metrics ?? {})) metrics.add(key);
          }
        } catch {
          // Suggestions only: a run without readable validation offers none.
        }
        if (destroyed || detailRefs?.id !== s.id) return;
        if (datasets.size) break;
      }
      known.datasets = [...datasets];
      known.metrics = [...metrics];
      if (!dataset.value && datasets.size === 1) dataset.value = [...datasets][0];
    })();

    async function run() {
      error.textContent = '';
      const body = { studyId: s.id, leftArmId: left.value, rightArmId: right.value, dataset: dataset.value.trim(), metric: metric.value.trim() };
      if (body.leftArmId === body.rightArmId) {
        error.textContent = 'Choose two different arms.';
        return;
      }
      if (!body.dataset || !body.metric) {
        error.textContent = 'Enter the dataset and the metric to compare.';
        return;
      }
      if (chosen.checked) {
        if (!/^\d+$/.test(step.value.trim())) {
          error.textContent = 'Enter the iteration as a whole number, or choose the latest common iteration.';
          return;
        }
        body.step = Number(step.value.trim());
      }
      ctl?.abort();
      ctl = new AbortController();
      mount(result, loading('Comparing the two arms…'));
      try {
        const answer = await api.studyCompare(body, { signal: ctl.signal });
        if (destroyed) return;
        mount(result, renderComparison(answer, arms));
      } catch (e) {
        if (isAbort(e) || destroyed) return;
        mount(result, notice(e?.status === 400 || e?.status === 404 ? 'warn' : 'err', `The arms were not compared: ${e?.message ?? 'no answer'}`));
      }
    }
    detailRefs.compareAbort = () => ctl?.abort();
    return el;
  }

  // ───────── polling ─────────

  function schedule() {
    clearTimeout(timer);
    if (destroyed || document.hidden) return;
    const moving = current && study?.id === current && cellsOf(study).some(({ entry }) => MOVING.has(entry.status));
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
  // Fresh queue state for the paused line in the release confirmation and for the cells.
  if (queue.get().phase === 'idle') queue.refresh();

  tick();

  return {
    setStudy(id) {
      if (destroyed) return;
      const next = id || null;
      if (next === current) return;
      detailCtl?.abort();
      detailRefs?.compareAbort?.();
      current = next;
      study = null;
      detailRefs = null;
      listSignature = '';
      tick();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      clearTimeout(timer);
      listCtl?.abort();
      detailCtl?.abort();
      detailRefs?.compareAbort?.();
      document.removeEventListener('visibilitychange', onVisibility);
      unsubscribeQueue();
      root.remove();
    },
  };
}
