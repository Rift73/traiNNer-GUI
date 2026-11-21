// "Weigh two scores": choosing a checkpoint by two validation scores of one dataset at once
// (GET /api/run/checkpoint-frontier, studio/checkpoint_frontier.py). The service answers with the measured
// checkpoints that no other checkpoint beats on both scores. Sits under "Which checkpoint measured best?"
// in a run's Files tab (detail.js) and uses its words for scores and for how a file was tied to its score.
// Rules in css/frontier.css.
//
// What the service sends, as its code produces it:
//   observedCount   validation passes of the dataset whose recorded summary holds at least one of the two
//                   scores. Scores known only from the log text are not used.
//   excluded        {missingOrNaN, directionUnknown, weightsUnavailable (only with availableOnly)}: passes
//                   left out, by reason, with counts.
//   cohorts         the passes that can be weighed, grouped so that nothing unlike is compared: one group
//                   per attempt, task, evaluator settings, population and pair of directions.
//   populationId    a digest of the images measured and of the evaluation protocol; null when no complete
//                   record of the images was kept. `populationVerified` is false then, and every pass
//                   without such a record lands in the same group.
//   point.checkpoint  status 'available' (evidence 'recordedSaveId'; artifact {name, path, role, version,
//                   saveId}), 'missing' (reason 'numberedWeightsMissing'), 'invalid' (reason
//                   'invalidWeightContainer') or 'unverified' (reason 'noEvaluationReceipt',
//                   'numberedSaveUnmatched', 'invalidWeightAssociation', 'weightsChanged',
//                   'weightIdentityMismatch', or what the trainer noted while measuring). A file is tied to
//                   a score only by the save the measurement itself recorded, never through the log.
//   point.values    {metric: number, or 'inf' / '-inf'}; points come best first score first.
//   directions      the service orients a score from what the evaluation itself recorded. Where it recorded
//                   nothing, the pass is left out (directionUnknown) unless the request says `xDirection` /
//                   `yDirection` ('higher' | 'lower'); a direction against a recorded one is refused (400,
//                   "The recorded metric direction is higher."). A direction is sent only for a score the
//                   person chose one for, with the buttons this part then offers.
//
// mountCheckpointFrontier(host, {runId, hasFile?, onPick?, selection?, request?}) -> {refresh(), destroy()}

import { h, fmt, uid } from '../dom.js';
import { api } from '../api.js';
import { loading, notice } from '../ui.js';
import { CHECKPOINT_REASONS, scoreText } from './detail.js';

const isNum = Number.isFinite;
const isText = (value) => typeof value === 'string' && value !== '';
const asObject = (value) => (value && typeof value === 'object' && !Array.isArray(value) ? value : null);
const reason = (error) => (error && typeof error.message === 'string' && error.message.trim()) || 'No reason was given.';
const fileName = (path) => String(path ?? '').split(/[\\/]/).pop();
const shortAttempt = (attempt) => (isText(attempt) ? attempt.slice(0, 8) : '');

const DIRECTION_WORDS = { higher: 'higher is better', lower: 'lower is better' };
/** How an available file was tied to its score, in the words of the ranking above (detail.js). */
const EVIDENCE_WORDS = {
  recordedSaveId: 'Matched by the save ID this measurement recorded.',
  uniqueLoggedSave: 'Matched through the training log, the weaker evidence: a file replaced later would go unnoticed.',
};
/** Reasons only this answer gives; every other reason is worded by the ranking (CHECKPOINT_REASONS). */
const RECEIPT_REASONS = { noEvaluationReceipt: 'this measurement did not record which save it used' };

/**
 * The datasets of a run that have two or more recorded scores, from the `series` of
 * GET /api/run/checkpoint-selection ([{dataset, metric, direction}]), in the order given.
 * @param {Array} series
 * @returns {Array<{dataset: string, metrics: string[], directions: Record<string, 'higher' | 'lower' | null>}>}
 *   `directions` holds, per score, the direction the series states, or null where it states none
 */
export function qualifyingDatasets(series) {
  const groups = [];
  for (const s of Array.isArray(series) ? series : []) {
    if (!s || !isText(s.dataset) || !isText(s.metric)) continue;
    let group = groups.find((g) => g.dataset === s.dataset);
    if (!group) groups.push((group = { dataset: s.dataset, metrics: [], directions: {} }));
    if (group.metrics.includes(s.metric)) continue;
    group.metrics.push(s.metric);
    group.directions[s.metric] = Object.hasOwn(DIRECTION_WORDS, s.direction) ? s.direction : null;
  }
  return groups.filter((g) => g.metrics.length >= 2);
}

/** One sentence per reason the service left measurements out, with its count. */
export function excludedSentences(excluded) {
  const out = [];
  for (const [key, n] of Object.entries(asObject(excluded) ?? {})) {
    if (!isNum(n) || n <= 0) continue;
    const one = n === 1;
    const lead = `${fmt.plural(n, 'measurement')} ${one ? 'is' : 'are'} left out`;
    if (key === 'missingOrNaN') out.push(`${lead}: ${one ? 'it lacks' : 'they lack'} one of the two scores, or one of them is NaN (not a number).`);
    else if (key === 'directionUnknown') out.push(`${lead}: the service does not know whether higher or lower is better for one of the two scores.`);
    else if (key === 'weightsUnavailable') out.push(`${lead}: ${one ? 'its' : 'their'} weights file is gone, damaged or could not be matched with certainty.`);
    else out.push(`${lead} for a reason this page has no words for: ${key}.`);
  }
  return out;
}

/** "2 of 12 measured checkpoints are not beaten on both psnr and ssim." for one group of the answer. */
export function cohortSentence(cohort, x, y) {
  const n = cohort?.observedCount;
  const k = cohort?.frontierCount;
  if (n === 1) return 'The one measured checkpoint is the only candidate.';
  if (!isNum(n) || !isNum(k)) return `The measured checkpoints that no other beats on both ${x} and ${y}.`;
  return `${fmt.int(k)} of ${fmt.int(n)} measured checkpoints ${k === 1 ? 'is' : 'are'} not beaten on both ${x} and ${y}.`;
}

/**
 * The title of one group among several. An attempt is named by the start of the id the service gave it, as
 * the ranking and the scores do; an attempt measured in more than one way numbers its ways.
 */
export function cohortTitle(cohort, cohorts) {
  const same = cohorts.filter((c) => c.attempt === cohort.attempt);
  return `Attempt ${shortAttempt(cohort.attempt) || 'not recorded'}${same.length > 1 ? `, way of measuring ${same.indexOf(cohort) + 1} of ${same.length}` : ''}`;
}

/**
 * Mount the part into `host` (appended). Nothing is drawn unless a dataset of the run has two recorded
 * scores, and the service is asked to weigh them only once the part is opened.
 * @param {HTMLElement} host
 * @param {object} options
 * @param {string} options.runId
 * @param {(path: string) => boolean} [options.hasFile] whether the file list below holds this file
 * @param {(path: string) => void} [options.onPick] select that file in the list below (the host moves focus)
 * @param {Function} [options.selection] answers like api.checkpointSelection (the run's series)
 * @param {Function} [options.request] answers like api.checkpointFrontier
 * @returns {{refresh: () => void, destroy: () => void}} `refresh` reads again without disturbing what is in use
 */
export function mountCheckpointFrontier(
  host,
  { runId, hasFile = () => true, onPick = () => {}, selection = (id, params, opts) => api.checkpointSelection(id, params, opts), request = (id, params, opts) => api.checkpointFrontier(id, params, opts) } = {},
) {
  const ids = { dataset: uid('frontier-dataset'), first: uid('frontier-first'), second: uid('frontier-second') };
  const datasetSelect = h('select', { class: 'select frontier__select', id: ids.dataset });
  const firstSelect = h('select', { class: 'select frontier__select', id: ids.first });
  const secondSelect = h('select', { class: 'select frontier__select', id: ids.second });
  const field = (id, label, select) => h('div', { class: 'frontier__field' }, h('label', { class: 'frontier__label', for: id }, label), select);
  const datasetField = field(ids.dataset, 'Dataset', datasetSelect);
  const onlyBox = h('input', { type: 'checkbox' });
  // The answer in one sentence, drawn once and announced; the tables and notes under it are not.
  const said = h('p', { class: 'frontier__line', role: 'status' });
  // "For <score>: Higher is better / Lower is better", one group per score that needs it. Outside the
  // answer, so a pressed button is never rebuilt with it.
  const directionBox = h('div', { class: 'frontier__directions', hidden: true });
  const area = h('div', { class: 'frontier__answer', tabindex: '-1' });
  const details = h(
    'details',
    { class: 'frontier__details' },
    h('summary', { class: 'frontier__summary' }, 'Weigh two scores'),
    h(
      'div',
      { class: 'frontier__body' },
      h('div', { class: 'frontier__controls' }, datasetField, field(ids.first, 'First score', firstSelect), field(ids.second, 'Second score', secondSelect)),
      directionBox,
      said,
      area,
      h('label', { class: 'frontier__check' }, onlyBox, 'Only checkpoints whose file is still there'),
    ),
  );
  const box = h('section', { class: 'frontier', hidden: true, 'aria-label': 'Weigh two scores' }, details);
  host.appendChild(box);

  let groups = []; // the datasets with two or more scores
  let groupsSig = '';
  const chosen = { dataset: null, first: null, second: null };
  let seriesCtrl = null;
  let askCtrl = null;
  let sig = ''; // what is on screen: the answer and which of its files the list below holds
  let waiting = null; // a changed answer of a background read, held while the keyboard is inside the old one
  let unknown = false; // the service has no such route (or no such run): the part stays hidden
  let gone = false;
  const choice = new Map(); // score -> the direction the person chose for it ('higher' | 'lower')
  let shownChoice = new Map(); // the choices the answer on screen was asked with
  let leftUnknown = false; // the answer on screen left measurements out for want of a direction
  const directionGroups = new Map(); // score -> its group of two buttons

  const group = () => groups.find((g) => g.dataset === chosen.dataset) ?? null;
  const metrics = () => group()?.metrics ?? [];
  /** The direction the run's series states for a score of the chosen dataset, or null. */
  const stated = (metric) => group()?.directions?.[metric] ?? null;
  /**
   * Whether a score gets its two buttons: the series states no direction for it, or the person already
   * chose one. A score with a stated direction is offered them only when the answer on screen left
   * measurements out for want of a direction although no score without one was left unanswered: the
   * service does not say which of the two it could not orient, so both are offered then.
   */
  const offered = (metric) => !stated(metric) || choice.has(metric) || (leftUnknown && [chosen.first, chosen.second].every((m) => stated(m) || shownChoice.has(m)));
  const inArea = () => document.activeElement !== area && area.contains(document.activeElement);
  /** Replace the answer. A control inside it that has the keyboard hands it to the answer area first. */
  const show = (...nodes) => {
    if (inArea()) area.focus();
    area.replaceChildren(...nodes);
  };

  function directionGroup(metric) {
    const button = (direction, text) =>
      h(
        'button',
        {
          class: 'btn btn--sm',
          type: 'button',
          'aria-pressed': 'false',
          dataset: { direction },
          onclick: () => {
            // Pressing the chosen one again takes the choice back.
            if (choice.get(metric) === direction) choice.delete(metric);
            else choice.set(metric, direction);
            syncDirections();
            ask();
          },
        },
        text,
      );
    return h(
      'div',
      { class: 'frontier__choice', role: 'group', 'aria-label': `Which direction is better for ${metric}` },
      h('span', { class: 'frontier__label' }, `For ${metric}:`),
      button('higher', 'Higher is better'),
      button('lower', 'Lower is better'),
    );
  }

  /**
   * Bring the direction buttons in line with the chosen scores: a group per score that is offered one, first
   * score first, the chosen direction pressed. A group that holds the keyboard is neither removed nor moved.
   */
  function syncDirections() {
    const wanted = [chosen.first, chosen.second].filter((metric) => isText(metric) && offered(metric));
    for (const [metric, el] of directionGroups) {
      if (wanted.includes(metric) || el.contains(document.activeElement)) continue;
      el.remove();
      directionGroups.delete(metric);
    }
    for (const metric of wanted) if (!directionGroups.has(metric)) directionGroups.set(metric, directionGroup(metric));
    const ordered = [...wanted.map((metric) => directionGroups.get(metric)), ...[...directionGroups].filter(([metric]) => !wanted.includes(metric)).map(([, el]) => el)];
    if (!directionBox.contains(document.activeElement)) directionBox.replaceChildren(...ordered);
    else for (const el of ordered) if (!el.parentNode) directionBox.appendChild(el);
    for (const [metric, el] of directionGroups) {
      for (const button of el.querySelectorAll('button')) button.setAttribute('aria-pressed', String(choice.get(metric) === button.dataset.direction));
    }
    directionBox.hidden = directionGroups.size === 0;
  }

  /** A new pair of scores: choices for scores no longer shown go, and so does what the last answer said. */
  function pairChanged({ all = false } = {}) {
    for (const metric of [...choice.keys()]) if (all || (metric !== chosen.first && metric !== chosen.second)) choice.delete(metric);
    leftUnknown = false;
    syncDirections();
  }

  function fillMetrics() {
    const names = metrics();
    for (const [select, value] of [
      [firstSelect, chosen.first],
      [secondSelect, chosen.second],
    ]) {
      select.replaceChildren(...names.map((name, i) => h('option', { value: String(i), selected: name === value }, name)));
      select.value = String(names.indexOf(value));
    }
  }

  function fill() {
    datasetSelect.replaceChildren(...groups.map((g, i) => h('option', { value: String(i), selected: g.dataset === chosen.dataset }, g.dataset)));
    datasetSelect.value = String(groups.findIndex((g) => g.dataset === chosen.dataset));
    datasetField.hidden = groups.length < 2;
    fillMetrics();
  }

  function setGroups(next) {
    const nextSig = JSON.stringify(next);
    if (nextSig === groupsSig) {
      if (details.open && groups.length) ask({ quiet: true });
      return;
    }
    groups = next;
    groupsSig = nextSig;
    if (!groups.length) {
      askCtrl?.abort();
      box.hidden = true;
      return;
    }
    // A choice survives a re-read while the run still has it.
    const kept = group() && metrics().includes(chosen.first) && metrics().includes(chosen.second);
    if (!kept) {
      chosen.dataset = groups[0].dataset;
      [chosen.first, chosen.second] = groups[0].metrics;
    }
    fill();
    if (kept) syncDirections();
    else pairChanged({ all: true });
    box.hidden = unknown;
    if (details.open) ask({ quiet: kept && Boolean(sig) });
  }

  function readSeries() {
    seriesCtrl?.abort();
    const c = new AbortController();
    seriesCtrl = c;
    // Asked at once; a function that throws instead of answering is caught like a refusal.
    new Promise((resolve) => resolve(selection(runId, {}, { signal: c.signal }))).then(
      (res) => {
        if (gone || c !== seriesCtrl) return;
        setGroups(qualifyingDatasets(res?.series));
      },
      () => {
        // The ranking above reads the same scores and says what went wrong; this part adds nothing to it.
      },
    );
  }

  /** Which files of the answer the list below holds: a "Choose" button appears and goes with its file. */
  const pickKey = (answer) =>
    (Array.isArray(answer?.cohorts) ? answer.cohorts : [])
      .flatMap((cohort) => (Array.isArray(cohort?.points) ? cohort.points : []))
      .map((point) => (point?.checkpoint?.status === 'available' && isText(point.checkpoint.artifact?.path) ? `${point.checkpoint.artifact.path}:${Boolean(hasFile(point.checkpoint.artifact.path))}` : ''))
      .join('|');

  function ask({ quiet = false } = {}) {
    if (unknown || !group()) return;
    askCtrl?.abort();
    const c = new AbortController();
    askCtrl = c;
    const params = { dataset: chosen.dataset, x: chosen.first, y: chosen.second };
    // A direction goes out only for a score the person chose one for.
    const asked = new Map([chosen.first, chosen.second].filter((metric) => choice.has(metric)).map((metric) => [metric, choice.get(metric)]));
    if (asked.has(chosen.first)) params.xDirection = asked.get(chosen.first);
    if (asked.has(chosen.second)) params.yDirection = asked.get(chosen.second);
    if (onlyBox.checked) params.availableOnly = true;
    if (!quiet) {
      sig = '';
      waiting = null;
      said.textContent = '';
      show(loading('Weighing the recorded scores…'));
    }
    new Promise((resolve) => resolve(request(runId, params, { signal: c.signal }))).then(
      (res) => {
        if (gone || c !== askCtrl) return;
        const answer = asObject(res) ?? {};
        const nextSig = `${JSON.stringify(answer)}|${pickKey(answer)}|${JSON.stringify([...asked])}`;
        if (nextSig === sig) return; // unchanged: nothing is rebuilt
        // A background read never rebuilds what the keyboard is in; the new answer waits until it leaves.
        if (quiet && inArea()) waiting = { answer, sig: nextSig, asked };
        else draw(answer, nextSig, asked);
      },
      (error) => {
        if (gone || c !== askCtrl || error?.name === 'AbortError') return;
        if (error?.status === 404 || /unknown api endpoint/i.test(String(error?.message ?? ''))) {
          // A service without this route, or without this run's folder: nothing to offer, nothing to say.
          unknown = true;
          box.hidden = true;
          return;
        }
        if (quiet && sig) return; // a background read that fails leaves what is on screen
        sig = '';
        const refused = error?.status === 400 || error?.status === 422;
        // A refusal of a chosen direction (one against what the evaluation recorded) is shown in the
        // service's words; the way on is to ask again without the choice.
        const withdraw = refused && asked.size > 0;
        const again = h(
          'button',
          {
            class: 'btn btn--sm',
            type: 'button',
            onclick: () => {
              if (withdraw) {
                for (const metric of asked.keys()) choice.delete(metric);
                syncDirections();
              }
              ask();
            },
          },
          withdraw ? 'Ask without a direction' : 'Try again',
        );
        said.textContent = '';
        show(notice('err', refused ? `The service refused: ${reason(error)}` : `The two scores could not be weighed: ${reason(error)}`, again));
      },
    );
  }

  function draw(answer, nextSig, asked) {
    sig = nextSig;
    waiting = null;
    shownChoice = asked;
    leftUnknown = isNum(answer.excluded?.directionUnknown) && answer.excluded.directionUnknown > 0;
    const { sentence, nodes } = renderAnswer(answer);
    show(...nodes);
    syncDirections();
    if (said.textContent !== sentence) said.textContent = sentence;
  }

  function checkpointCells(point) {
    const c = asObject(point.checkpoint) ?? {};
    if (c.status === 'available') {
      const path = c.artifact?.path;
      const name = c.artifact?.name ?? fileName(path);
      const how = EVIDENCE_WORDS[c.evidence];
      const choose =
        isText(path) && hasFile(path)
          ? h('button', { class: 'btn btn--sm', type: 'button', onclick: () => onPick(path) }, 'Choose', h('span', { class: 'visually-hidden' }, ` iteration ${fmt.int(point.step)}`))
          : null;
      return [h('td', { class: 'frontier__file' }, h('span', { class: 'mono' }, name || 'Not named'), how ? h('span', { class: 'frontier__how' }, how) : null), h('td', null, choose)];
    }
    let words = 'Not reported';
    if (c.status === 'missing') words = 'No file is kept for this iteration';
    else if (c.status === 'invalid') words = 'The file is damaged or incomplete';
    else if (c.status === 'unverified') {
      const why = RECEIPT_REASONS[c.reason] ?? CHECKPOINT_REASONS[c.reason];
      words = `Not matched with certainty${why ? `: ${why}` : ''}`;
    }
    return [h('td', { class: 'frontier__file' }, words), h('td', null)];
  }

  function cohortBlock(cohort, title, x, y, leftOut) {
    const points = (Array.isArray(cohort.points) ? cohort.points : []).filter((point) => point && typeof point === 'object');
    const directions = asObject(cohort.directions) ?? {};
    const sentence = cohortSentence(cohort, x, y);
    const n = cohort.observedCount;
    const k = cohort.frontierCount;
    const head = (metric) => {
      const words = DIRECTION_WORDS[directions[metric]];
      const mine = shownChoice.get(metric) === directions[metric] ? ', as you chose' : '';
      return h('th', { scope: 'col', class: 'num' }, metric, words ? h('span', { class: 'frontier__direction' }, ` (${words}${mine})`) : null);
    };
    const value = (point, metric) => {
      const v = asObject(point.values)?.[metric];
      return h('td', { class: 'num' }, scoreText({ value: typeof v === 'number' ? v : null, nonfinite: typeof v === 'string' ? v : null }));
    };
    const notes = [];
    if (isNum(n) && isNum(k) && n > k) notes.push('Every other measured checkpoint is worse than one of these on at least one of the two scores and no better on the other.');
    if (isNum(k) && k > points.length && points.length) notes.push(`The first ${fmt.int(points.length)} of ${fmt.int(k)} are listed.`);
    if (cohort.populationVerified === false && n !== 1) {
      notes.push(
        'The service could not confirm that all of these were measured on the same images: it holds no complete record of the images behind each score. The scores are the values each validation pass recorded.',
      );
    }
    const tableLabel = title ? `${title}: checkpoints not beaten on both scores` : 'Checkpoints not beaten on both scores';
    const el = h(
      'section',
      { class: 'frontier__cohort' },
      title ? h('h4', { class: 'frontier__title' }, title) : null,
      // A group among several has its own line; the only group's sentence is the answer line above.
      title ? h('p', { class: 'frontier__line' }, sentence) : null,
      leftOut,
      points.length
        ? h(
            'div',
            { class: 'frontier__tablewrap', tabindex: '0', role: 'region', 'aria-label': tableLabel },
            h(
              'table',
              { class: 'table frontier__table' },
              h(
                'thead',
                null,
                h('tr', null, h('th', { scope: 'col', class: 'num' }, 'Iteration'), head(x), head(y), h('th', { scope: 'col' }, 'Checkpoint'), h('th', { scope: 'col' }, h('span', { class: 'visually-hidden' }, 'Choose'))),
              ),
              h(
                'tbody',
                null,
                points.map((point) => h('tr', null, h('td', { class: 'num' }, isNum(point.step) ? fmt.int(point.step) : 'Not reported'), value(point, x), value(point, y), checkpointCells(point))),
              ),
            ),
          )
        : null,
      notes.map((text) => h('p', { class: 'frontier__note' }, text)),
    );
    return { el, sentence };
  }

  function renderAnswer(answer) {
    const named = Array.isArray(answer.metrics) && answer.metrics.length === 2 && answer.metrics.every(isText);
    const [x, y] = named ? answer.metrics : [chosen.first, chosen.second];
    const cohorts = (Array.isArray(answer.cohorts) ? answer.cohorts : []).filter((cohort) => cohort && typeof cohort === 'object');
    const leftOut = excludedSentences(answer.excluded).map((text) => h('p', { class: 'frontier__note' }, text));
    // `sentence` goes to the answer line above the area; `nodes` are what follows it.
    if (!cohorts.length) {
      const n = answer.observedCount;
      // Left out for want of a direction: name the score that needs one, which the buttons above take.
      // Where the series states a direction for both, the service does not say which one it lacks.
      const open = [x, y].filter((metric) => !shownChoice.has(metric));
      const lacking = open.some((metric) => !stated(metric)) ? open.filter((metric) => !stated(metric)) : open;
      const sentence =
        leftUnknown && lacking.length
          ? `Say whether higher or lower is better for ${lacking.join(' and ')} to weigh ${lacking.length === 1 ? 'it' : 'them'}.`
          : n === 0
            ? 'No recorded validation pass of this run holds both scores, so there is nothing to weigh. Scores read from the log text are not used here.'
            : !leftOut.length
              ? 'The service listed no checkpoint to weigh.'
              : n === 1
                ? 'The one recorded measurement cannot be weighed.'
                : isNum(n)
                  ? `None of the ${fmt.int(n)} recorded measurements can be weighed.`
                  : 'No recorded measurement can be weighed.';
      return { sentence, nodes: leftOut };
    }
    if (cohorts.length === 1) {
      const block = cohortBlock(cohorts[0], '', x, y, leftOut);
      return { sentence: block.sentence, nodes: [block.el] };
    }
    const sentence = `The measurements fall into ${fmt.int(cohorts.length)} groups that cannot be weighed against each other: a different attempt, or a different way of measuring. Each has its own answer.`;
    return { sentence, nodes: [...leftOut, ...cohorts.map((cohort) => cohortBlock(cohort, cohortTitle(cohort, cohorts), x, y, []).el)] };
  }

  details.addEventListener('toggle', () => {
    // Asked once, when first opened; after that the answer is kept and refresh() reads it again.
    if (details.open && !sig && !area.firstChild) ask();
  });
  datasetSelect.addEventListener('change', () => {
    const next = groups[Number(datasetSelect.value)];
    if (!next) return;
    chosen.dataset = next.dataset;
    [chosen.first, chosen.second] = next.metrics;
    fillMetrics();
    pairChanged({ all: true });
    ask();
  });
  // The two scores are never the same one: choosing the other's score swaps them.
  const choose = (select, mine, other, otherSelect) => () => {
    const names = metrics();
    const next = names[Number(select.value)];
    if (!isText(next)) return;
    if (next === chosen[other]) {
      chosen[other] = chosen[mine];
      otherSelect.value = String(names.indexOf(chosen[other]));
    }
    chosen[mine] = next;
    pairChanged();
    ask();
  };
  firstSelect.addEventListener('change', choose(firstSelect, 'first', 'second', secondSelect));
  secondSelect.addEventListener('change', choose(secondSelect, 'second', 'first', firstSelect));
  onlyBox.addEventListener('change', () => ask());
  area.addEventListener('focusout', (event) => {
    if (!waiting || (event.relatedTarget && area.contains(event.relatedTarget))) return;
    const held = waiting;
    Promise.resolve().then(() => {
      if (gone || waiting !== held || inArea()) return;
      draw(held.answer, held.sig, held.asked);
    });
  });

  readSeries();

  return {
    /** Read the run's scores again and, when the part is open, its answer; an unchanged answer is left alone. */
    refresh() {
      if (!gone && !unknown) readSeries();
    },
    destroy() {
      gone = true;
      seriesCtrl?.abort();
      askCtrl?.abort();
      box.remove();
    },
  };
}
