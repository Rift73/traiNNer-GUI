// Pair alignment check (analysis kind `pair-registration`): the outcome in one sentence, what was
// measured, the service's outcome counts, how to read a shift, one row per measured pair (brightness and
// two colour-plane displacements, in words, with their evidence), and the caveat. Used by the analysis
// job page and by the Data screen; `job` may be null.
//
// Shapes (studio/pair_audit.py audit_pairs, studio/alignment.py inspect_pair, native/studio/alignment):
//   result = {split, population, sampled, seed, radius, scale, units: 'LQ pixels', counts: {outcome: n},
//     method, pairs: [{gt, lq | gt, lq (databases) + gtKey, lqKey, assessment, estimates: {luma,
//     blueMinusLuma, redMinusLuma: {dx, dy, atLimit, peak, zero, peakMargin, patchCount} | null},
//     evidence: {same keys: {assessment, informativePatches, distinctPeaks?, supportingPatches?}},
//     alphaIgnored} | {…pair, assessment: 'unavailable', message}]}
//   progress = {stage: 'select', completed: 0} | {stage: 'registration', completed, total}
//   Direction (alignment.py): positive dx/dy means the low quality content appears right/down of the
//   area-downsampled ground truth.

import { h, fmt } from '../dom.js';
import { badge } from '../ui.js';

/** The three planes, in the Data screen's words (views/data.js ALIGN_CHANNELS). */
const PLANES = [
  ['luma', 'Brightness'],
  ['blueMinusLuma', 'Blue against brightness'],
  ['redMinusLuma', 'Red against brightness'],
];

/** Per-plane readings (alignment.py _assess), in the Data screen's words. */
const PLANE_WORDS = {
  noClearShift: 'No clear shift',
  possibleShift: 'Possible shift',
  ambiguous: 'Ambiguous: several offsets fit about equally well',
  inconsistentAcrossPatches: 'The patches disagree',
  searchBoundary: 'Best match at the edge of the search range: the real shift may be larger',
  insufficientTexture: 'Not enough texture to measure',
};

/** Per-pair outcomes (pair_audit.py _assessment and `unavailable`), in plain words, in this order. */
const OUTCOMES = [
  ['noClearLumaShift', 'No clear brightness shift'],
  ['possibleLumaShift', 'Possible brightness shift'],
  ['possibleChromaShift', 'Possible colour shift, brightness not shifted'],
  ['differentPlaneShifts', 'Brightness and colour shifted by different amounts'],
  ['localDisagreement', 'The patches of the image disagree'],
  ['inconclusive', 'Inconclusive'],
  ['unavailable', 'Could not be measured'],
];
const OUTCOME_WORDS = Object.fromEntries(OUTCOMES);

const SET_WORDS = { train: 'training data', val: 'validation data', test: 'test data' };

const CAVEAT =
  'This is a diagnostic, not a verdict on the data. Compression of the source images and deliberate shifts of a colour channel also produce a displacement, so a shift here is not by itself a fault.';
const CAPTION = 'Pairs with a possible shift first, then the largest shift. Each shift says where the low quality image sits against its ground truth.';
/** What to do with a flagged pair: the Data screen shows one pair on its stage and measures it alone ("Check alignment"). */
const NEXT_STEP = 'Look at a flagged pair before changing the data: on the Data screen, check the folders, open the pair and use "Check alignment" to see it and measure it on its own.';

/** How to read the table, one line per idea; `scale` is the recipe's scale as the result reports it. */
function legendLines(scale) {
  const perPixel = Number.isFinite(scale) && scale > 1 ? ` At ${fmt.int(scale)}×, 1 low quality pixel is ${fmt.int(scale)} ground truth pixels.` : '';
  return [
    ['Shift', `How far the low quality image sits from its ground truth, in low quality pixels, with the ground truth reduced to the same size.${perPixel}`],
    ['Brightness', 'The structure of the picture. A brightness shift means the whole image is offset against its ground truth.'],
    ['Blue and red', 'The two colour planes, each measured against brightness. A colour plane that moves while brightness stays can come from how the source was compressed or how its colour was stored.'],
    ['Match', 'How well the two images agree at that shift, and with no shift at all; 1.00 is a perfect match. A shift is called possible only where the match is clearly better than with no shift.'],
    ['Patches', 'Up to nine areas of each image are measured. The more of them agree, the firmer the estimate.'],
  ];
}

/** The outcome counts as one sentence: how many of the measured pairs may be shifted, and in what. */
function conclusion(counts, sampled) {
  const n = (key) => (Number.isFinite(counts[key]) ? counts[key] : 0);
  const total = Number.isFinite(sampled) ? sampled : Object.values(counts).reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0);
  if (!total) return '';
  const flagged = n('possibleLumaShift') + n('possibleChromaShift') + n('differentPlaneShifts');
  const unclear = n('localDisagreement') + n('inconclusive');
  const unread = n('unavailable');
  const rest = [unclear ? `${fmt.int(unclear)} gave no clear reading` : '', unread ? `${fmt.int(unread)} could not be measured` : ''].filter(Boolean).join(' and ');
  if (!flagged) {
    if (n('noClearLumaShift') === total) return total === 1 ? 'The measured pair shows no clear shift.' : `None of the ${fmt.int(total)} measured pairs shows a clear shift.`;
    return `No measured pair shows a clear shift${rest ? `; ${rest}` : ''}.`;
  }
  const kinds = [
    n('possibleLumaShift') ? `${fmt.int(n('possibleLumaShift'))} in brightness` : '',
    n('possibleChromaShift') ? `${fmt.int(n('possibleChromaShift'))} in colour only` : '',
    n('differentPlaneShifts') ? `${fmt.int(n('differentPlaneShifts'))} with brightness and colour apart` : '',
  ].filter(Boolean);
  const list = kinds.length <= 1 ? kinds.join('') : `${kinds.slice(0, -1).join(', ')} and ${kinds[kinds.length - 1]}`;
  return `${fmt.int(flagged)} of the ${fmt.plural(total, 'measured pair')} may be shifted: ${list}.${rest ? ` ${rest.charAt(0).toUpperCase()}${rest.slice(1)}.` : ''}`;
}

/**
 * A displacement in words: "2.00 px left, 2.00 px up", "0.25 px right", "No shift". Positive x is right,
 * positive y is down (alignment.py). A component that rounds to 0.00 is left out.
 */
function shiftWords(dx, dy) {
  const part = (value, positive, negative) => (Number(Math.abs(value).toFixed(2)) === 0 ? '' : `${Math.abs(value).toFixed(2)} px ${value > 0 ? positive : negative}`);
  const parts = [part(dx, 'right', 'left'), part(dy, 'down', 'up')].filter(Boolean);
  return parts.length ? parts.join(', ') : 'No shift';
}

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const words = (key) => {
  const text = String(key).replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase();
  return text.charAt(0).toUpperCase() + text.slice(1);
};
const baseName = (path) => {
  const text = String(path ?? '').replace(/[\\/]+$/, '');
  const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
  return cut < 0 ? text : text.slice(cut + 1);
};

/** "the training data", "the validation data (val_2)". */
function setPhrase(split) {
  const key = String(split ?? '');
  const phase = key.split('_')[0];
  const word = SET_WORDS[phase];
  if (!word) return key ? `the dataset ${key}` : 'the data';
  return key === phase ? `the ${word}` : `the ${word} (${key})`;
}

/** A displacement component: two decimals ("-0.00" never shown), the full value in `title`. */
function shiftValue(value) {
  const text = value.toFixed(2);
  return Number(text) === 0 ? '0.00' : text;
}

/** Correlation values: two decimals, so the two numbers of a match line read alike; full value in `title`. */
const scoreText = (value) => (Number.isFinite(value) ? shiftValue(value) : '');

/** Pair outcomes in which the service reports a possible shift: listed first, with a warn badge. */
const POSSIBLE = new Set(['possibleLumaShift', 'possibleChromaShift', 'differentPlaneShifts']);

/** Largest displacement of a pair over its planes; -1 when nothing was measured. */
function largest(pair) {
  let best = -1;
  for (const [key] of PLANES) {
    const e = pair?.estimates?.[key];
    if (isObject(e) && Number.isFinite(e.dx) && Number.isFinite(e.dy)) best = Math.max(best, Math.hypot(e.dx, e.dy));
  }
  return best;
}

/** The pair's name: the low quality file, or its database key; both sides in `title`. */
function pairLabel(pair) {
  const keyed = typeof pair.lqKey === 'string' || typeof pair.gtKey === 'string';
  const name = keyed ? pair.lqKey ?? pair.gtKey : baseName(pair.lq ?? pair.gt);
  const title = keyed
    ? `Ground truth: ${pair.gt ?? ''} key ${pair.gtKey ?? ''}\nLow quality: ${pair.lq ?? ''} key ${pair.lqKey ?? ''}`
    : `Ground truth: ${pair.gt ?? ''}\nLow quality: ${pair.lq ?? ''}`;
  return { name: name || 'Unnamed pair', title };
}

/** One plane of one pair: "x, y" (or "Not measured") over the evidence for it. */
function planeCell(estimate, evidence) {
  const measured = isObject(estimate) && Number.isFinite(estimate.dx) && Number.isFinite(estimate.dy);
  const shift = measured
    ? h('span', { class: 'registration__shift', title: `x ${String(estimate.dx)}, y ${String(estimate.dy)} low quality pixels (positive is right and down)` }, shiftWords(estimate.dx, estimate.dy))
    : h('span', { class: 'registration__shift registration__muted' }, 'Not measured');
  const lines = [];
  if (isObject(evidence) && evidence.assessment !== undefined) lines.push(PLANE_WORDS[evidence.assessment] ?? words(evidence.assessment));
  if (measured && Number.isFinite(estimate.peak)) {
    const atZero = Number.isFinite(estimate.zero) ? `, ${scoreText(estimate.zero)} with no shift` : '';
    lines.push(`Match ${scoreText(estimate.peak)} at this offset${atZero}`);
  }
  if (isObject(evidence) && Number.isFinite(evidence.supportingPatches) && Number.isFinite(evidence.distinctPeaks)) {
    const verb = evidence.distinctPeaks === 1 ? 'agrees' : 'agree';
    lines.push(`${fmt.int(evidence.supportingPatches)} of ${fmt.plural(evidence.distinctPeaks, 'clear patch', 'clear patches')} ${verb}`);
  } else if (isObject(evidence) && Number.isFinite(evidence.informativePatches)) {
    lines.push(fmt.plural(evidence.informativePatches, 'patch with texture', 'patches with texture'));
  }
  const title = measured
    ? [
        Number.isFinite(estimate.peak) ? `Match at the best offset: ${String(estimate.peak)}` : '',
        Number.isFinite(estimate.zero) ? `Match with no shift: ${String(estimate.zero)}` : '',
        Number.isFinite(estimate.peakMargin) ? `Lead over the next best offset: ${String(estimate.peakMargin)}` : '',
        Number.isFinite(estimate.patchCount) ? `Patches with texture: ${String(estimate.patchCount)}` : '',
      ]
        .filter(Boolean)
        .join('\n')
    : null;
  return h('td', { class: 'registration__cell' }, shift, lines.length ? h('span', { class: 'registration__evidence', title: title || null }, lines.join('. ')) : null);
}

/**
 * One plain sentence for a running job's progress; '' when absent or of an unknown stage.
 * @param {object|null} progress
 */
export function pairRegistrationProgressText(progress) {
  if (!isObject(progress)) return '';
  if (progress.stage === 'select') return 'Selecting the pairs to measure';
  if (progress.stage === 'registration') {
    if (Number.isFinite(progress.completed) && Number.isFinite(progress.total)) return `Measured ${fmt.int(progress.completed)} of ${fmt.plural(progress.total, 'pair')}`;
    return 'Measuring the pairs';
  }
  return '';
}

/**
 * Draw a pair alignment result into `host`.
 * @param {HTMLElement} host
 * @param {{job?: object|null, result?: object|null, artifacts?: Array<{path: string, url: string}>}} options
 * @returns {() => void} cleanup
 */
export function renderPairRegistration(host, { job = null, result = null, artifacts = [] } = {}) {
  const r = isObject(result) ? result : {};
  const pairs = Array.isArray(r.pairs) ? r.pairs.filter(isObject) : [];
  const split = r.split ?? job?.params?.split ?? null;
  const seed = Number.isFinite(r.seed) ? r.seed : job?.params?.seed;

  // 1. What was measured.
  const measured =
    Number.isFinite(r.sampled) && Number.isFinite(r.population)
      ? `Measured ${fmt.int(r.sampled)} of ${fmt.plural(r.population, 'pair')} of ${setPhrase(split)}${Number.isFinite(seed) ? `, chosen with seed ${fmt.int(seed)}` : ''}.`
      : `The service did not report how many pairs of ${setPhrase(split)} were measured.`;
  const radius = Number.isFinite(r.radius) ? `Search radius ${fmt.plural(r.radius, 'low quality pixel')}` : '';
  const scale = Number.isFinite(r.scale) ? `scale ${fmt.int(r.scale)}×` : '';
  const setting = [radius, scale].filter(Boolean).join(', ');

  // 2. Outcome counts, in the service's classes.
  const counts = isObject(r.counts) ? r.counts : {};
  const known = OUTCOMES.filter(([key]) => Number.isFinite(counts[key]));
  const unknown = Object.keys(counts).filter((key) => !OUTCOME_WORDS[key] && Number.isFinite(counts[key]));
  const countRows = [...known.map(([key, label]) => [label, counts[key]]), ...unknown.map((key) => [words(key), counts[key]])];
  const summary = conclusion(counts, r.sampled);
  const flagged = [...POSSIBLE].some((key) => Number.isFinite(counts[key]) && counts[key] > 0);

  // 3. Pairs, largest displacement first; the service's order breaks ties (stable sort).
  // Pairs with a possible shift first, then the largest displacement; the service's order breaks ties.
  const ordered = pairs
    .map((pair, index) => ({ pair, index, size: largest(pair), possible: POSSIBLE.has(pair.assessment) ? 0 : 1 }))
    .sort((a, b) => a.possible - b.possible || b.size - a.size || a.index - b.index);

  const table = h(
    'div',
    { class: 'registration__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Measured pairs' },
    h(
      'table',
      { class: 'table registration__table' },
      h('caption', { class: 'registration__caption' }, CAPTION),
      h(
        'thead',
        null,
        h('tr', null, h('th', { scope: 'col' }, 'Pair'), h('th', { scope: 'col' }, 'Outcome'), PLANES.map(([, label]) => h('th', { scope: 'col' }, label))),
      ),
      h(
        'tbody',
        null,
        ordered.map(({ pair }) => {
          const label = pairLabel(pair);
          const outcomeText = pair.assessment === undefined ? 'Not reported' : OUTCOME_WORDS[pair.assessment] ?? words(pair.assessment);
          // The warn badge repeats the words of a possible-shift outcome; every other outcome is plain text.
          const outcome = POSSIBLE.has(pair.assessment) ? badge(outcomeText, 'warn') : outcomeText;
          const nameCell = h(
            'td',
            { class: 'registration__pair' },
            h('span', { class: 'registration__name', title: label.title }, label.name),
            pair.alphaIgnored === true ? h('span', { class: 'registration__evidence' }, 'Transparency ignored') : null,
          );
          if (pair.assessment === 'unavailable' || !isObject(pair.estimates)) {
            return h(
              'tr',
              null,
              nameCell,
              h('td', null, outcome),
              h('td', { colspan: String(PLANES.length), class: 'registration__reason' }, typeof pair.message === 'string' && pair.message ? pair.message : 'The service gave no reason.'),
            );
          }
          return h('tr', null, nameCell, h('td', null, outcome), PLANES.map(([key]) => planeCell(pair.estimates[key], pair.evidence?.[key])));
        }),
      ),
    ),
  );

  const files = Array.isArray(artifacts) ? artifacts.filter((a) => a && typeof a.url === 'string') : [];
  const root = h(
    'div',
    { class: 'registration' },
    summary ? h('p', { class: 'registration__lead' }, summary) : null,
    h('p', { class: summary ? 'registration__note' : 'registration__lead' }, measured, setting ? ` ${setting}.` : ''),
    flagged ? h('p', { class: 'registration__note' }, NEXT_STEP) : null,
    countRows.length
      ? h(
          'dl',
          { class: 'kv registration__counts', 'aria-label': 'Outcomes' },
          countRows.map(([label, n]) => [h('dt', null, label), h('dd', { class: 'num' }, fmt.plural(n, 'pair'))]),
        )
      : h('p', { class: 'registration__muted' }, 'The service reported no outcome counts.'),
    pairs.length
      ? h(
          'details',
          { class: 'registration__legend' },
          h('summary', { class: 'registration__legendhead' }, 'How to read the table'),
          h('dl', { class: 'registration__legendlist' }, legendLines(r.scale).map(([term, text]) => [h('dt', null, term), h('dd', null, text)])),
        )
      : null,
    pairs.length ? table : h('p', { class: 'registration__muted' }, 'No pairs were measured.'),
    typeof r.method === 'string' && r.method ? h('p', { class: 'registration__note' }, `Method, as the service states it: ${r.method}`) : null,
    files.length ? h('ul', { class: 'registration__files' }, files.map((a) => h('li', null, h('a', { href: a.url, download: baseName(a.path) || 'output' }, a.path)))) : null,
    h('p', { class: 'registration__caveat' }, CAVEAT),
  );
  host.appendChild(root);
  return () => {
    root.remove();
  };
}
