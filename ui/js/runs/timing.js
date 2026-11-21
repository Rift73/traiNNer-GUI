// "How fast this run trained": the rates, the data wait and the memory a run's trainer log recorded
// (GET /api/run/timing, studio/run_timing.py). One closed block; the log is read only once it is opened.
// Rules in css/timing.css.
//
// What the service sends, as its code produces it (studio/run_timing.py, native/studio/history/history.cpp,
// traiNNer/utils/logger.py):
//   base, against   {runId, sources (log names, newest first), source (the log read), indexedBytes,
//                   sourceBytes, more (the log is not read to its end yet), series}
//   series[key]     {loggedPoints, firstStep, lastStep, minimum, maximum, median, last} over the log lines
//                   that hold the value; negative values are left out.
//     performance/iterations_per_second    "[performance: N it/s]": 1 / the trainer's average time per
//                                          iteration over its recent window (up to 200 iterations)
//     performance/wall_updates_per_second  "[wall: N updates/s]": iterations done since the session's first
//                                          one, divided by all the time passed since then
//     performance/data_wait_percent        "[data wait: N%]": average time waiting for a batch over the
//                                          average time per iteration, capped at 100
//     memory/peak_allocated_GiB            "[peak VRAM: N GB allocated": torch.cuda.max_memory_allocated / 1024^3
//     memory/peak_reserved_GiB             "... N GB reserved]": torch.cuda.max_memory_reserved / 1024^3
//     memory/device_total_GiB              second number of "[ mem: A / T GiB ]"; not described here
//   matchedRecentRates  {loggedStepsInCommon, firstStep, lastStep, medianAgainstOverBase}: at the iterations
//                   where both logs hold performance/iterations_per_second above zero, the median of the
//                   other run's rate divided by this run's; null with fewer than two such iterations.
//   comparisonPendingIndex  true instead of matchedRecentRates while either log is still being read.
//
// mountRunTiming(host, {runId, runs?, request?}) -> {refresh(), destroy()}

import { h, fmt, uid } from '../dom.js';
import { api } from '../api.js';
import { loading, notice } from '../ui.js';

const isNum = Number.isFinite;
const isText = (value) => typeof value === 'string' && value !== '';
const asObject = (value) => (value && typeof value === 'object' && !Array.isArray(value) ? value : null);
const reason = (error) => (error && typeof error.message === 'string' && error.message.trim()) || 'No reason was given.';

const RATE = 'performance/iterations_per_second';
const WALL = 'performance/wall_updates_per_second';
const WAIT = 'performance/data_wait_percent';

/** The series whose meaning the trainer's code confirms. Any other key is shown as it is, in mono. */
export const SERIES_WORDS = {
  [RATE]: { label: 'Rate the trainer logged (iterations per second)' },
  [WALL]: { label: 'Rate counting all the time since the first iteration (iterations per second)' },
  [WAIT]: { label: 'Time spent waiting for data', suffix: '%' },
  'memory/peak_allocated_GiB': { label: 'Most GPU memory PyTorch had allocated (GiB)' },
  'memory/peak_reserved_GiB': { label: 'Most GPU memory PyTorch had reserved (GiB)' },
};
const ORDER = [RATE, WALL, WAIT, 'memory/peak_allocated_GiB', 'memory/peak_reserved_GiB', 'memory/device_total_GiB'];

/** A number with up to four significant digits; a small value that is not zero never reads "0". */
export function num(value) {
  if (!isNum(value)) return 'Not recorded';
  if (value === 0) return '0';
  const rounded = Number(value.toPrecision(4));
  return Number.isInteger(rounded) && Math.abs(rounded) >= 1000 && Math.abs(rounded) < 1e15 ? fmt.int(rounded) : String(rounded);
}

/** "7.73 KiB": a count of bytes in the largest binary unit that keeps it at 1 or more. */
export function bytes(n) {
  if (!isNum(n) || n < 0) return 'an unknown size';
  if (n < 1024) return fmt.plural(n, 'byte');
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let value = n / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${num(value)} ${units[i]}`;
}

const steps = (s) => (isNum(s.firstStep) && isNum(s.lastStep) ? (s.firstStep === s.lastStep ? fmt.int(s.firstStep) : `${fmt.int(s.firstStep)} to ${fmt.int(s.lastStep)}`) : 'Not recorded');
const one = (s) => s.loggedPoints === 1;

/** "(96.53 to 124.1 over 3 log lines, iterations 1 to 3)" or "(one log line, iteration 2)". */
function span(s, suffix = '') {
  if (one(s)) return `(one log line, iteration ${steps(s)})`;
  const lines = isNum(s.loggedPoints) ? ` over ${fmt.plural(s.loggedPoints, 'log line')}` : '';
  return `(${num(s.minimum)}${suffix} to ${num(s.maximum)}${suffix}${lines}, iterations ${steps(s)})`;
}

/** The sentence for the rate the trainer printed on its log lines. */
export function rateSentence(s) {
  return `The trainer logged ${one(s) ? '' : 'a median of '}${num(s.median)} iterations per second ${span(s)}.`;
}

/** The sentence for the rate over the whole session: iterations done over all the time passed. */
export function wallSentence(s) {
  return `Counting all the time since the first iteration, whatever it was spent on, the rate was ${one(s) ? '' : 'a median of '}${num(s.median)} iterations per second ${span(s)}.`;
}

/** The sentence for the share of the time spent waiting for the next batch. */
export function waitSentence(s) {
  if (one(s)) return `Waiting for data took ${num(s.median)}% of the time (one log line, iteration ${steps(s)}).`;
  return `Waiting for data took a median of ${num(s.median)}% of the time (${num(s.minimum)}% to ${num(s.maximum)}%).`;
}

/** 'warn' when half of the time or more went to waiting for data, 'fine' under a tenth, else ''. */
export const waitVerdict = (s) => (!isNum(s?.median) ? '' : s.median >= 50 ? 'warn' : s.median < 10 ? 'fine' : '');
export const WAIT_WARNING = 'More than half of the time went to waiting for data. More data loader workers, or faster storage for the dataset, will help more than a faster GPU.';
export const WAIT_FINE = 'The data loader kept up: little time was spent waiting for data.';

/** The one sentence of a comparison, from `matchedRecentRates`. */
export function matchedSentence(matched, name) {
  const m = asObject(matched) ?? {};
  if (!isNum(m.loggedStepsInCommon) || m.loggedStepsInCommon < 2 || !isNum(m.medianAgainstOverBase)) {
    return 'The two logs share fewer than two logged iterations with a rate, so the runs cannot be compared step by step.';
  }
  const lead = `At the ${fmt.int(m.loggedStepsInCommon)} logged iterations both runs share (${steps(m)}), ${name} ran at `;
  return m.medianAgainstOverBase === 1 ? `${lead}the same rate as this run, going by the median.` : `${lead}a median of ${num(m.medianAgainstOverBase)} times this run’s rate.`;
}

/** The keys of both answers: the known ones in a fixed order, then the rest by name. */
export function seriesKeys(...lists) {
  const all = new Set(lists.flatMap((series) => Object.keys(asObject(series) ?? {})));
  return [...ORDER.filter((key) => all.has(key)), ...[...all].filter((key) => !ORDER.includes(key)).sort()];
}

/**
 * Mount the block into `host` (appended), closed. The service is asked only once it is opened.
 * @param {HTMLElement} host
 * @param {object} options
 * @param {string} options.runId
 * @param {() => Array<{id: string, name: string}>} [options.runs] the runs offered under "Compare with"
 * @param {Function} [options.request] answers like api.runTiming
 * @returns {{refresh: () => void, destroy: () => void}} `refresh` reads again without disturbing what is in use
 */
export function mountRunTiming(host, { runId, runs = () => [], request = (id, params, opts) => api.runTiming(id, params, opts) } = {}) {
  const ids = { log: uid('timing-log'), against: uid('timing-against') };
  const logSelect = h('select', { class: 'select timing__select', id: ids.log });
  const againstSelect = h('select', { class: 'select timing__select', id: ids.against });
  const field = (id, label, select) => h('div', { class: 'timing__field' }, h('label', { class: 'field__label timing__label', for: id }, label), select);
  const logField = field(ids.log, 'Log', logSelect);
  logField.hidden = true;
  const area = h('div', { class: 'timing__answer', tabindex: '-1' });
  const summary = h('summary', { class: 'timing__summary' }, 'How fast this run trained');
  const details = h('details', { class: 'timing' }, summary, h('div', { class: 'timing__body' }, h('div', { class: 'timing__controls' }, logField, field(ids.against, 'Compare with', againstSelect)), area));
  host.appendChild(details);

  const chosen = { source: null, against: null };
  const names = new Map(); // run id -> the name last offered for it
  let logs = []; // the log names the Log choice offers, as the service ordered them
  let againstSig = null;
  let ctrl = null;
  let asked = false; // the block has been opened once
  let sig = ''; // the answer on screen
  let waiting = null; // a changed answer of a background read, held while the keyboard is inside the old one
  let gone = false;

  const inArea = () => document.activeElement !== area && area.contains(document.activeElement);
  /** Whether the keyboard is in the result area, or nowhere because its control was just replaced. */
  const mayFocus = () => document.activeElement === area || area.contains(document.activeElement) || document.activeElement === document.body || !document.activeElement;
  /** Replace the result. A control inside it that has the keyboard hands it to the result area first. */
  const show = (...nodes) => {
    if (inArea()) area.focus();
    area.replaceChildren(...nodes);
  };
  const otherName = () => names.get(chosen.against) || chosen.against || 'the other run';

  /** Fill "Compare with" from runs(), without this run. A choice in use is neither rebuilt nor dropped. */
  function fillAgainst() {
    let list = [];
    try {
      list = runs();
    } catch {
      list = [];
    }
    list = (Array.isArray(list) ? list : []).filter((run) => run && isText(run.id) && run.id !== runId);
    for (const run of list) names.set(run.id, isText(run.name) ? run.name : run.id);
    if (chosen.against && !list.some((run) => run.id === chosen.against)) list = [...list, { id: chosen.against }];
    const nextSig = JSON.stringify(list.map((run) => [run.id, names.get(run.id) ?? run.id]));
    if (nextSig === againstSig || (againstSig !== null && document.activeElement === againstSelect)) return;
    againstSig = nextSig;
    againstSelect.replaceChildren(
      h('option', { value: '', selected: !chosen.against }, 'No other run'),
      ...list.map((run) => h('option', { value: run.id, selected: run.id === chosen.against }, names.get(run.id) ?? run.id)),
    );
    againstSelect.value = chosen.against ?? '';
  }

  /** Bring the Log choice in line with the answer: shown only when the run has more than one log. */
  function fillLogs(base) {
    const next = (Array.isArray(base?.sources) ? base.sources : []).filter(isText);
    const current = isText(base?.source) ? base.source : chosen.source;
    if (JSON.stringify(next) !== JSON.stringify(logs) && document.activeElement !== logSelect) {
      logs = next;
      logSelect.replaceChildren(...logs.map((name, i) => h('option', { value: String(i), selected: name === current }, name)));
    }
    if (document.activeElement !== logSelect && logs.includes(current)) logSelect.value = String(logs.indexOf(current));
    logField.hidden = logs.length < 2;
  }

  /**
   * Ask the service. `quiet` keeps what is on screen until the answer is there; `from` says which button
   * asked ('retry' | 'more'), so the keyboard is put somewhere sensible once its button is gone.
   */
  function ask({ quiet = false, from = '' } = {}) {
    asked = true;
    ctrl?.abort();
    const c = new AbortController();
    ctrl = c;
    const params = {};
    if (chosen.source) params.source = chosen.source;
    if (chosen.against) params.against = chosen.against;
    const against = chosen.against;
    if (!quiet) {
      sig = '';
      waiting = null;
      show(loading('Reading the log…'));
    }
    // Asked at once; a function that throws instead of answering is caught like a refusal.
    new Promise((resolve) => resolve(request(runId, params, { signal: c.signal }))).then(
      (res) => {
        if (gone || c !== ctrl) return;
        const answer = asObject(res) ?? {};
        const nextSig = `${JSON.stringify(answer)}|${against ?? ''}|${against ? otherName() : ''}`;
        if (nextSig === sig && !from) return; // unchanged: nothing is rebuilt
        // A background read never rebuilds what the keyboard is in; the new answer waits until it leaves.
        if (quiet && !from && inArea()) waiting = { answer, sig: nextSig, against };
        else draw(answer, nextSig, against, from);
      },
      (error) => {
        if (gone || c !== ctrl || error?.name === 'AbortError') return;
        if (quiet && !from && sig) return; // a background read that fails leaves what is on screen
        sig = '';
        waiting = null;
        const again = h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => ask({ from: 'retry' }) }, 'Try again');
        const move = from && mayFocus();
        show(notice('err', `The timing could not be read: ${reason(error)}`, again));
        if (move) again.focus();
      },
    );
  }

  function draw(answer, nextSig, against, from = '') {
    const move = from && mayFocus();
    sig = nextSig;
    waiting = null;
    fillLogs(asObject(answer.base));
    const nodes = renderAnswer(answer, against);
    show(...nodes);
    if (!move) return;
    // The button that asked is gone: its successor takes the keyboard, or the block's own heading.
    const next = from === 'more' ? area.querySelector?.('.timing__more') : null;
    (next ?? summary).focus();
  }

  function seriesTable(base, other, name) {
    const keys = seriesKeys(base, other);
    if (!keys.length) return null;
    const cell = (s, field, suffix) => h('td', { class: 'num' }, s && isNum(s[field]) ? `${num(s[field])}${suffix}` : 'Not recorded');
    const rows = keys.map((key) => {
      const words = Object.hasOwn(SERIES_WORDS, key) ? SERIES_WORDS[key] : null;
      const suffix = words?.suffix ?? '';
      const s = asObject(base?.[key]);
      const what = words ? h('th', { scope: 'row', class: 'timing__what', title: key }, words.label) : h('th', { scope: 'row', class: 'timing__what mono' }, key);
      return h(
        'tr',
        null,
        what,
        h('td', { class: 'num' }, s && isNum(s.loggedPoints) ? fmt.int(s.loggedPoints) : 'Not recorded'),
        h('td', { class: 'num' }, s ? steps(s) : 'Not recorded'),
        cell(s, 'minimum', suffix),
        cell(s, 'median', suffix),
        cell(s, 'maximum', suffix),
        cell(s, 'last', suffix),
        other ? cell(asObject(other[key]), 'median', suffix) : null,
      );
    });
    const head = (text, numeric = true) => h('th', { scope: 'col', class: numeric ? 'num' : null }, text);
    return h(
      'div',
      { class: 'timing__tablewrap', tabindex: '0', role: 'region', 'aria-label': 'Everything the log recorded' },
      h(
        'table',
        { class: 'table timing__table' },
        h('thead', null, h('tr', null, head('What', false), head('Log lines'), head('Iterations'), head('Lowest'), head('Median'), head('Highest'), head('Last'), other ? head(`Median, ${name}`) : null)),
        h('tbody', null, rows),
      ),
    );
  }

  function renderAnswer(answer, against) {
    const base = asObject(answer.base) ?? {};
    const series = asObject(base.series) ?? {};
    const other = against ? asObject(answer.against) : null;
    const otherSeries = other ? (asObject(other.series) ?? {}) : null;
    const name = otherName();
    const line = (text) => h('p', { class: 'timing__line' }, text);
    const note = (...parts) => h('p', { class: 'timing__note' }, ...parts);
    const nodes = [];

    // No log at all (the service sends `source: null` and no log names) is not the same as a log without a rate.
    const noLog = !isText(base.source);
    if (noLog) nodes.push(line('This run has no trainer log, so there is nothing to summarise.'));
    else if (!Object.keys(series).length) nodes.push(line('This log holds no rate or data-wait line, so there is nothing to summarise.'));
    if (asObject(series[RATE])) nodes.push(line(rateSentence(series[RATE])));
    if (asObject(series[WALL])) nodes.push(line(wallSentence(series[WALL])));
    if (asObject(series[WAIT])) {
      nodes.push(line(waitSentence(series[WAIT])));
      const verdict = waitVerdict(series[WAIT]);
      if (verdict === 'warn') nodes.push(notice('warn', WAIT_WARNING));
      else if (verdict === 'fine') nodes.push(note(WAIT_FINE));
    }

    const pending = Boolean(base.more) || Boolean(other?.more);
    if (other) {
      if (asObject(answer.matchedRecentRates)) {
        nodes.push(line(matchedSentence(answer.matchedRecentRates, name)));
        nodes.push(note('The comparison uses the rate the trainer logged, at the iterations where both logs hold one above zero.'));
      } else if (answer.comparisonPendingIndex) {
        nodes.push(line('The two runs are compared once both logs are read to the end.'));
      }
    }

    const table = seriesTable(series, otherSeries, name);
    if (table) nodes.push(h('h4', { class: 'timing__title' }, 'Everything the log recorded'), table);

    if (!noLog) nodes.push(note('From ', h('span', { class: 'mono' }, base.source), '.'));
    if (other) nodes.push(isText(other.source) ? note(`The numbers of ${name} are from `, h('span', { class: 'mono' }, other.source), '.') : note(`The service named no log for ${name}.`));
    if (pending) {
      const still = [];
      if (base.more) still.push(`The log is still being read: ${bytes(base.indexedBytes)} of ${bytes(base.sourceBytes)} so far.`);
      if (other?.more) still.push(`The log of ${name} is still being read: ${bytes(other.indexedBytes)} of ${bytes(other.sourceBytes)} so far.`);
      nodes.push(
        h(
          'div',
          { class: 'timing__pending' },
          h('p', { class: 'timing__note' }, still.join(' ')),
          h('button', { class: 'btn btn--sm btn--quiet timing__more', type: 'button', onclick: () => ask({ quiet: true, from: 'more' }) }, 'Read on'),
        ),
      );
    }
    return nodes;
  }

  details.addEventListener('toggle', () => {
    // Asked once, when first opened; after that the answer is kept and refresh() reads it again.
    if (!details.open || asked || gone) return;
    fillAgainst();
    ask();
  });
  logSelect.addEventListener('change', () => {
    const next = logs[Number(logSelect.value)];
    if (!isText(next)) return;
    chosen.source = next;
    ask();
  });
  againstSelect.addEventListener('change', () => {
    chosen.against = isText(againstSelect.value) ? againstSelect.value : null;
    ask();
  });
  area.addEventListener('focusout', (event) => {
    if (!waiting || (event.relatedTarget && area.contains(event.relatedTarget))) return;
    const held = waiting;
    Promise.resolve().then(() => {
      if (gone || waiting !== held || inArea()) return;
      draw(held.answer, held.sig, held.against);
    });
  });

  fillAgainst();

  return {
    /** Read again when the block has been opened; an unchanged answer is left alone. */
    refresh() {
      if (gone) return;
      fillAgainst();
      if (asked) ask({ quiet: Boolean(sig) });
    },
    destroy() {
      gone = true;
      ctrl?.abort();
      details.remove();
    },
  };
}
