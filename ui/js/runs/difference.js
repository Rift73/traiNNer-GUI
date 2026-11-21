// Result of a completed image difference job (analysis kind `image-difference`): two saved PNG images
// compared pixel by pixel, optionally inside a rectangle and optionally against a reference image. The
// producer is studio/image_difference.py (native/studio/image_difference/difference.c). `job` may be null.
//
//   renderImageDifference(host, {job, result, artifacts}) -> cleanup
//   imageDifferenceProgressText(progress) -> one sentence, or ''
//
// Shapes:
//   result = {inputs: {left, right, reference?: {path, version}}, width, height,
//     region: {x, y, width, height} (the whole image when none was asked for), pixels, bitDepth: 8 | 16,
//     channels: ['Y'] | ['R', 'G', 'B'], units: 'stored integer sample levels',
//     leftVsRight: {changedPixels, channels: {name: {mae, rmse, max}}},
//     leftVsReference?, rightVsReference? (same shape),
//     closerPixels?: {left, right, tie, criterion: 'sumSquaredChannelError'}}
//   A pixel counts as changed when any of its channels differs. The saved file `absolute-difference.png`
//   holds |left - right| per channel, in the same sample levels, for the compared area only.
//   progress = {stage: 'comparing', completed, total} (rows)

import { h, mount, fmt, uid } from '../dom.js';
import { notice, copyText, toast } from '../ui.js';

const ARTIFACT = 'absolute-difference.png';
/** A small picture is enlarged by a whole factor up to about this many CSS pixels on its long side. */
const SHOWN_LONG_SIDE = 480;
/** Above this many pixels the picture is not copied into a canvas (4096 × 4096). */
const CANVAS_PIXEL_LIMIT = 16777216;
const KNOWN_UNITS = 'stored integer sample levels';
const KNOWN_CRITERION = 'sumSquaredChannelError';
const NOT_REPORTED = 'Not reported';

const CHANNEL_WORDS = new Map([
  ['Y', 'Brightness (grey)'],
  ['R', 'Red'],
  ['G', 'Green'],
  ['B', 'Blue'],
  ['A', 'Alpha'],
]);

const NO_REFERENCE = 'No reference was given, so this says how different the two images are, not which one is better.';
const READING = 'Black means the two images are equal at that pixel; brighter means more different.';
const READING_COLOUR = 'In a colour image each colour channel shows its own difference: a pixel that differs only in red is shown red.';
const WHERE = 'White marks every pixel that differs at all, whatever the size of the difference; black means equal.';
const MASKS = 'If these are class masks, a level is a class number: any difference means another class, and its size says nothing about how wrong it is.';
const VERSIONS = 'The file versions were checked when the job ran, so these are the files as they were then.';

// ───────── pure helpers (exported for checks) ─────────

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isCount = (v) => Number.isFinite(v) && v >= 0;
const baseName = (path) => {
  const text = String(path ?? '').replace(/[\\/]+$/, '');
  return text.slice(Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\')) + 1);
};

const levelFormat = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });

/**
 * A size in sample levels: three significant digits with trailing zeros dropped ("1", "0.25", "12.4"),
 * one decimal at most from 100 up ("12,345.7"). '' when absent.
 */
export function levelText(value) {
  if (!Number.isFinite(value)) return '';
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs < 0.001) return 'less than 0.001';
  if (abs >= 100) return levelFormat.format(value);
  return String(Number(value.toPrecision(3)));
}

/** "1 level", "41 levels", "0.25 levels". */
const levelsPhrase = (value) => `${levelText(value)} ${value === 1 ? 'level' : 'levels'}`;

/**
 * A share with one decimal. A share that is not nothing never reads "0.0%", and one that is not
 * everything never reads "100.0%". '' when it cannot be computed.
 */
export function shareText(part, whole) {
  if (!isCount(part) || !Number.isFinite(whole) || whole <= 0 || part > whole) return '';
  const percent = (part / whole) * 100;
  if (part > 0 && percent < 0.05) return 'less than 0.1%';
  if (part < whole && percent >= 99.95) return 'more than 99.9%';
  return `${percent.toFixed(1)}%`;
}

/** The highest sample level of a bit depth (255 for 8, 65,535 for 16); null when absent. */
export function maxLevelOf(bitDepth) {
  return Number.isInteger(bitDepth) && bitDepth >= 1 && bitDepth <= 16 ? 2 ** bitDepth - 1 : null;
}

/** The largest `max` over the channels of one comparison; null unless every channel reports one. */
export function largestOf(summary) {
  const entries = isObject(summary?.channels) ? Object.values(summary.channels) : [];
  if (!entries.length || !entries.every((entry) => isObject(entry) && isCount(entry.max))) return null;
  return entries.reduce((best, entry) => Math.max(best, entry.max), 0);
}

/**
 * The compared area. The job reports the whole image as a region too, so a region that covers the image
 * reads "The whole image". `rect` is null for the whole image or when the region is not understood.
 */
export function areaOf(result) {
  const r = isObject(result) ? result : {};
  const g = r.region;
  // No region in an answer that reports the image: the whole image. An answer that reports nothing says nothing.
  if (g === null || g === undefined) {
    return isCount(r.width) || isCount(r.height) || isCount(r.pixels) || isObject(r.leftVsRight) ? { whole: true, text: 'The whole image', rect: null } : { whole: false, text: NOT_REPORTED, rect: null };
  }
  if (!isObject(g) || ![g.x, g.y, g.width, g.height].every(isCount)) return { whole: false, text: NOT_REPORTED, rect: null };
  if (g.x === 0 && g.y === 0 && g.width === r.width && g.height === r.height) return { whole: true, text: 'The whole image', rect: null };
  return { whole: false, text: `${fmt.dims(g.width, g.height)} pixels from ${g.x}, ${g.y} (left, top)`, rect: g };
}

/** "312 of 900 compared pixels differ (34.7%)", without the full stop. */
export function changedPhrase(changed, pixels) {
  const verb = changed === 1 ? 'differs' : 'differ';
  if (!isCount(pixels) || pixels < changed) return `${fmt.plural(changed, 'pixel')} ${verb}`;
  const share = shareText(changed, pixels);
  return `${fmt.int(changed)} of ${fmt.plural(pixels, 'compared pixel')} ${verb}${share ? ` (${share})` : ''}`;
}

/** The first sentence: how many pixels differ and by how much at most. */
export function leadSentence(result) {
  const r = isObject(result) ? result : {};
  const changed = r.leftVsRight?.changedPixels;
  if (!isCount(changed)) return 'The job did not report how many pixels differ.';
  if (changed === 0) return areaOf(r).whole ? 'No pixel differs: the two images are identical.' : 'No pixel differs: the two images are identical in this area.';
  const largest = largestOf(r.leftVsRight);
  const top = maxLevelOf(r.bitDepth);
  const size = largest === null ? '' : `${levelsPhrase(largest)}${top === null ? '' : ` out of ${fmt.int(top)}`}`;
  if (isCount(r.pixels) && changed === r.pixels) {
    const all = r.pixels === 1 ? 'The one compared pixel differs' : `All ${fmt.int(r.pixels)} compared pixels differ`;
    return size ? `${all}, by ${size} at most.` : `${all}.`;
  }
  const counted = changedPhrase(changed, r.pixels);
  return size ? `${counted}; the largest difference is ${size}.` : `${counted}.`;
}

/** The three counts of `closerPixels` with their sum; null unless all three are reported. */
export function closerCounts(result) {
  const c = result?.closerPixels;
  if (!isObject(c) || !isCount(c.left) || !isCount(c.right) || !isCount(c.tie)) return null;
  return { left: c.left, right: c.right, tie: c.tie, total: c.left + c.right + c.tie };
}

/** Which image is closer to the reference, in words, with every count written out. */
export function closerSentence(closer) {
  if (!closer) return 'The job did not report which image is closer to the reference at each pixel.';
  const { left, right, tie, total } = closer;
  if (total === 0) return 'No pixel was compared against the reference.';
  const everywhere = total === 1 ? 'at the one compared pixel' : `at all ${fmt.int(total)} compared pixels`;
  if (right === total) return `The second image is closer to the reference ${everywhere}.`;
  if (left === total) return `The first image is closer to the reference ${everywhere}.`;
  if (tie === total) return `The two images are equally close to the reference ${everywhere}.`;
  const share = (n) => shareText(n, total);
  const ties = tie === 0 ? 'none are equally close' : `${fmt.int(tie)} ${tie === 1 ? 'is' : 'are'} equally close (${share(tie)})`;
  if (left === right) return `Each image is closer to the reference at ${fmt.plural(left, 'pixel')} (${share(left)}); ${ties}.`;
  const [ahead, behind] = right > left ? [['second', right], ['first', left]] : [['first', left], ['second', right]];
  return `The ${ahead[0]} image is closer to the reference at ${fmt.plural(ahead[1], 'pixel')} (${share(ahead[1])}), the ${behind[0]} at ${fmt.int(behind[1])} (${share(behind[1])}); ${ties}.`;
}

/**
 * How the saved picture can be shown: `black` (nothing differs), `brighten` by a whole factor that puts
 * the largest difference at or just under full brightness, or `saved` with the reason in words.
 */
export function brightenPlan({ largest, bitDepth }) {
  const first = 'first shows where the images differ. "As saved" shows how much.';
  if (!isCount(largest)) {
    return {
      kind: 'saved',
      reason: 'The job did not report the largest difference, so the picture is shown as saved.',
      short: `The job did not report the largest difference, so the picture cannot be brightened and ${first}`,
    };
  }
  if (largest === 0) return { kind: 'black' };
  const top = maxLevelOf(bitDepth);
  if (top === null) {
    return {
      kind: 'saved',
      reason: 'The job did not report the bit depth, so the picture cannot be brightened here and is shown as saved.',
      short: `The job did not report the bit depth, so the picture cannot be brightened and ${first}`,
    };
  }
  if (bitDepth !== 8) {
    return {
      kind: 'saved',
      reason: `The file stores ${fmt.int(bitDepth)} bits per channel and the browser reduces it to 8 to draw it, which loses small differences. So it is not brightened here and is shown as saved; download the file to read its real values.`,
      short: `The file stores ${fmt.int(bitDepth)} bits per channel, which cannot be brightened here, so the picture first shows where the images differ. Download the file to read its real values.`,
    };
  }
  const factor = Math.floor(top / largest);
  if (factor < 2) {
    const size = `The largest difference (${levelsPhrase(largest)} out of ${fmt.int(top)})`;
    return {
      kind: 'saved',
      reason: `${size} is already more than half of full brightness, so the picture is shown as saved.`,
      short: `${size} is too large to brighten, so the picture ${first}`,
    };
  }
  return { kind: 'brighten', factor, exact: factor * largest === top };
}

/** The whole factor by which a small picture is enlarged on screen; 1 for a picture that is large enough. */
export function displayScale(width, height) {
  const long = Math.max(width, height);
  return long > 0 ? Math.max(1, Math.floor(SHOWN_LONG_SIDE / long)) : 1;
}

/**
 * A short name for each path: the file name, with as many parent folders as it takes to tell two
 * different files of the same name apart ("masks/0.png").
 */
export function shortNames(paths) {
  const parts = paths.map((path) => (path ? String(path).split(/[\\/]+/).filter(Boolean) : []));
  const depth = paths.map(() => 1);
  const name = (i) => parts[i].slice(-depth[i]).join('/');
  for (let round = 0; round < 64; round += 1) {
    const longer = new Set();
    for (let i = 0; i < paths.length; i += 1) {
      for (let j = 0; j < paths.length; j += 1) {
        if (i !== j && paths[i] && paths[j] && paths[i] !== paths[j] && name(i) === name(j) && depth[i] < parts[i].length) longer.add(i);
      }
    }
    if (!longer.size) break;
    for (const i of longer) depth[i] += 1;
  }
  return paths.map((path, i) => (path ? name(i) : ''));
}

/**
 * One plain sentence for a running job's progress; '' when absent or of an unknown stage.
 * @param {object|null} progress
 */
export function imageDifferenceProgressText(progress) {
  if (!isObject(progress) || progress.stage !== 'comparing') return '';
  if (isCount(progress.completed) && isCount(progress.total)) return `Compared ${fmt.int(progress.completed)} of ${fmt.plural(progress.total, 'row')} of pixels`;
  return 'Comparing the two images';
}

// ───────── renderer ─────────

const channelWord = (name) => CHANNEL_WORDS.get(name) ?? String(name);

/** Channel names in the job's order, then any the comparison reports beyond that list. */
function channelNames(r, ...summaries) {
  const names = Array.isArray(r.channels) ? r.channels.filter((name) => typeof name === 'string' && name) : [];
  for (const summary of summaries) {
    if (!isObject(summary?.channels)) continue;
    for (const name of Object.keys(summary.channels)) if (!names.includes(name)) names.push(name);
  }
  return names;
}

const colHead = (text, numeric = false) => h('th', { scope: 'col', class: numeric ? 'num' : null }, text);

/** A measure in levels; the unrounded value in `title`. */
function levelCell(value) {
  const text = levelText(value);
  return h('td', { class: text ? 'num' : 'num difference__absent', title: text ? String(value) : null }, text || NOT_REPORTED);
}

function countCell(changed, pixels, span) {
  const share = isCount(changed) ? shareText(changed, pixels) : '';
  const text = isCount(changed) ? `${fmt.int(changed)}${share ? ` (${share})` : ''}` : '';
  return h('td', { class: text ? 'num' : 'num difference__absent', rowspan: span > 1 ? String(span) : null }, text || NOT_REPORTED);
}

function unitSentence(units, top) {
  if (units === KNOWN_UNITS) return top === null ? 'In stored sample levels.' : `In stored sample levels, 0 to ${fmt.int(top)}.`;
  if (typeof units === 'string' && units) return `In the unit the job states: ${units}.`;
  return 'The job did not report the unit of these numbers.';
}

function channelTable(names, summary, units, top, labelId) {
  const channels = isObject(summary?.channels) ? summary.channels : {};
  return h(
    'div',
    { class: 'difference__table-wrap', tabindex: '0', role: 'region', 'aria-labelledby': labelId },
    h(
      'table',
      { class: 'table difference__table' },
      h(
        'caption',
        { class: 'difference__caption' },
        `${unitSentence(units, top)} The mean difference is the typical size of the difference over the compared pixels; the root mean square counts large differences more heavily; the largest is the worst single pixel.`,
      ),
      h('thead', null, h('tr', null, colHead('Channel'), colHead('Mean difference', true), colHead('Root mean square', true), colHead('Largest', true))),
      h(
        'tbody',
        null,
        names.map((name) => {
          const entry = Object.hasOwn(channels, name) && isObject(channels[name]) ? channels[name] : {};
          return h('tr', null, h('th', { scope: 'row', title: channelWord(name) === name ? null : `Channel ${name}` }, channelWord(name)), levelCell(entry.mae), levelCell(entry.rmse), levelCell(entry.max));
        }),
      ),
    ),
  );
}

/** "Each image against the reference": one row per image, or one row per image and channel. */
function referenceTable(names, sides, pixels, units, top, labelId) {
  const single = names.length <= 1;
  const entryOf = (summary, name) => (name !== undefined && isObject(summary?.channels) && Object.hasOwn(summary.channels, name) && isObject(summary.channels[name]) ? summary.channels[name] : {});
  const groups = sides.map(([label, summary]) => {
    const s = isObject(summary) ? summary : {};
    if (single) {
      const entry = entryOf(s, names[0]);
      return h('tbody', null, h('tr', null, h('th', { scope: 'row' }, label), countCell(s.changedPixels, pixels, 1), levelCell(entry.mae), levelCell(entry.max)));
    }
    return h(
      'tbody',
      null,
      names.map((name, index) => {
        const entry = entryOf(s, name);
        return h(
          'tr',
          null,
          index === 0 ? [h('th', { scope: 'rowgroup', rowspan: String(names.length) }, label), countCell(s.changedPixels, pixels, names.length)] : null,
          h('td', null, channelWord(name)),
          levelCell(entry.mae),
          levelCell(entry.max),
        );
      }),
    );
  });
  return h(
    'div',
    { class: 'difference__table-wrap', tabindex: '0', role: 'region', 'aria-labelledby': labelId },
    h(
      'table',
      { class: 'table difference__table' },
      h('caption', { class: 'difference__caption', id: labelId }, `Each image against the reference. ${unitSentence(units, top)}`),
      h(
        'thead',
        null,
        h('tr', null, colHead('Image'), colHead('Pixels that differ from the reference', true), single ? null : colHead('Channel'), colHead('Mean difference', true), colHead('Largest', true)),
      ),
      groups,
    ),
  );
}

/** The three-part bar (first, equally close, second) with every number written beside it. */
function closerBar(closer) {
  const parts = [
    ['first', 'First image closer', closer.left],
    ['tie', 'Equally close', closer.tie],
    ['second', 'Second image closer', closer.right],
  ];
  const width = (n) => `${((n / closer.total) * 100).toFixed(3)}%`;
  return [
    h(
      'div',
      { class: 'difference__bar', 'aria-hidden': 'true' },
      parts.map(([key, , n]) => (n > 0 ? h('span', { class: `difference__bar-part difference__bar-part--${key}`, style: { width: width(n) } }) : null)),
    ),
    h(
      'ul',
      { class: 'difference__legend' },
      parts.map(([key, label, n]) =>
        h(
          'li',
          { class: 'difference__legend-item' },
          h('span', { class: `difference__swatch difference__swatch--${key}`, 'aria-hidden': 'true' }),
          `${label}: `,
          h('span', { class: 'num' }, `${fmt.plural(n, 'pixel')} (${shareText(n, closer.total)})`),
        ),
      ),
    ),
  ];
}

function ruleLine(criterion) {
  if (criterion === KNOWN_CRITERION) {
    return h('p', { class: 'difference__note' }, 'Closer means a smaller sum of squared channel differences from the reference at that pixel. Equally close means the two sums are the same.');
  }
  if (typeof criterion === 'string' && criterion) {
    return h('p', { class: 'difference__note' }, 'Which image is closer at a pixel was decided ', h('span', { class: 'difference__code', title: criterion }, 'by a rule this screen does not know'), '.');
  }
  return h('p', { class: 'difference__note' }, 'The job did not report the rule it used to decide which image is closer.');
}

/**
 * The saved difference picture: enlarged when small, and redrawn on a canvas in up to two derived views,
 * each offered only when the canvas hands back values that agree with the job's numbers: "Brightened"
 * (every value multiplied) and "Where they differ" (white for any difference at all). Returns the nodes
 * and a release function for the image and the canvas.
 */
function picturePart({ artifact, plan, largest, top, bitDepth, changed, colour, labelId, signal }) {
  if (plan.kind === 'black') {
    return { nodes: [h('p', { class: 'difference__explain' }, 'The difference image is black everywhere.')], release: () => {} };
  }
  const download = h('a', { class: 'btn btn--sm', href: artifact.url, download: ARTIFACT }, 'Download the difference image (PNG)');
  const reading = h('p', { class: 'difference__note' }, colour ? `${READING} ${READING_COLOUR}` : READING);

  const explainId = uid('difference-explain');
  const img = h('img', { class: 'difference__image', alt: '', 'aria-describedby': explainId });
  const canvas = h('canvas', { class: 'difference__image', role: 'img', 'aria-describedby': explainId });
  const frame = h('div', { class: 'stage difference__frame' });
  const explain = h('p', { class: 'difference__explain', id: explainId, role: 'status' }, 'Loading the difference image…');
  const sizeNote = h('p', { class: 'difference__note', hidden: true });
  /** The views that can be shown, in button order; decided once, when the picture has loaded. */
  let offered = ['saved'];
  let view = 'saved';
  let loaded = false;
  let what = '';
  // Why the picture is not brightened: the full sentence when it is shown as saved, the short one when it
  // first shows where the images differ.
  let reason = plan.kind === 'brighten' ? '' : plan.reason;
  let shortReason = plan.kind === 'brighten' ? '' : plan.short;

  const largestWords = largest === null ? '' : `${levelsPhrase(largest)}${top === null ? '' : ` out of ${fmt.int(top)}`}`;
  const sentence = () => {
    if (view === 'bright') {
      return `Shown ${fmt.int(plan.factor)} times brighter so small differences can be seen; ${plan.exact && !colour ? 'white' : 'the brightest point'} is the largest difference (${levelsPhrase(largest)}). The file itself holds the real values.`;
    }
    if (view === 'where') return shortReason ? `${WHERE} ${shortReason}` : WHERE;
    if (reason && !offered.includes('where')) return reason;
    if (Number.isInteger(bitDepth) && bitDepth !== 8) {
      return `Shown as the browser draws the file: it reduces ${fmt.int(bitDepth)} bits per channel to 8, which loses small differences. Download the file to read its real values.`;
    }
    return `Shown ${bitDepth === 8 ? "with the file's real values" : 'as saved'}: brighter means a larger difference, and a small one may be too dark to see.${largestWords ? ` The largest difference is ${largestWords}.` : ''}`;
  };

  // The buttons are made once and only their pressed state changes, so a focused button keeps the focus.
  const buttons = new Map(
    [
      ['bright', 'Brightened'],
      ['where', 'Where they differ'],
      ['saved', 'As saved'],
    ].map(([key, label]) => [key, h('button', { class: 'btn btn--sm', type: 'button', 'aria-pressed': 'false', onclick: () => choose(key) }, label)]),
  );
  const views = h('div', { class: 'seg difference__views', role: 'group', 'aria-labelledby': labelId, hidden: true });

  /** Draw the loaded picture into the canvas and hand back its pixels; throws with the reason in words. */
  const read = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('The browser gave no drawing surface, so the picture is shown as saved.');
    context.drawImage(img, 0, 0);
    try {
      return { context, pixels: context.getImageData(0, 0, canvas.width, canvas.height) };
    } catch {
      throw new Error('The browser did not allow the picture to be read, so it is shown as saved.');
    }
  };

  /** Redraw the canvas for one derived view (pixel data, not interface colour). */
  const paint = (kind) => {
    const { context, pixels } = read();
    const data = pixels.data;
    if (kind === 'bright') {
      for (let i = 0; i < data.length; i += 4) {
        data[i] *= plan.factor;
        data[i + 1] *= plan.factor;
        data[i + 2] *= plan.factor;
      }
    } else {
      for (let i = 0; i < data.length; i += 4) {
        const mark = data[i] || data[i + 1] || data[i + 2] ? 255 : 0;
        data[i] = mark;
        data[i + 1] = mark;
        data[i + 2] = mark;
      }
    }
    context.putImageData(pixels, 0, 0);
  };

  const draw = () => {
    if (view !== 'saved') {
      try {
        paint(view);
      } catch {
        view = 'saved';
      }
    }
    for (const [key, button] of buttons) button.setAttribute('aria-pressed', String(key === view));
    if (view === 'bright') canvas.setAttribute('aria-label', `${what}, shown ${fmt.int(plan.factor)} times brighter: black where they are equal, brighter where they differ more.`);
    if (view === 'where') canvas.setAttribute('aria-label', `${what}, as a map: white where the two images differ at all, black where they are equal.`);
    const shown = view === 'saved' ? img : canvas;
    if (frame.firstChild !== shown) mount(frame, shown);
    reading.hidden = view === 'where';
    mount(explain, sentence());
  };
  function choose(next) {
    if (signal.aborted || !loaded || next === view || !offered.includes(next)) return;
    view = next;
    draw();
  }

  const onLoad = () => {
    if (signal.aborted) return;
    const width = img.naturalWidth;
    const height = img.naturalHeight;
    const scale = displayScale(width, height);
    const dims = fmt.dims(width, height);
    what = `Absolute difference between the first and the second image${dims ? `, ${dims} pixels` : ''}`;
    img.alt = `${what}, as saved: black where they are equal, brighter where they differ more.`;
    for (const el of [img, canvas]) {
      el.style.width = scale > 1 ? `${width * scale}px` : '';
      el.classList.toggle('difference__image--pixels', scale > 1);
    }
    let canBrighten = false;
    let canMark = false;
    try {
      if (width * height > CANVAS_PIXEL_LIMIT) throw new Error(`The picture is too large to redraw here (${fmt.dims(width, height)} pixels), so it is shown as saved.`);
      const data = read().pixels.data;
      let seen = 0;
      let marked = 0;
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] > seen) seen = data[i];
        if (data[i + 1] > seen) seen = data[i + 1];
        if (data[i + 2] > seen) seen = data[i + 2];
        if (data[i] || data[i + 1] || data[i + 2]) marked += 1;
      }
      // A derived view is offered only when the canvas agrees with the job's numbers: its brightest value
      // with the largest difference (brightening), its count of non-zero pixels with the changed pixels
      // (the map). A file of more than 8 bits is reduced by the browser, so its map needs that count.
      const exact = largest === null || seen === largest;
      if (plan.kind === 'brighten') {
        if (exact) {
          canBrighten = true;
        } else {
          const mismatch = `The browser did not hand back the file's exact values (its brightest value reads ${fmt.int(seen)} where the job measured ${fmt.int(largest)})`;
          reason = `${mismatch}, so the picture is shown as saved.`;
          shortReason = `${mismatch}, so the picture is not brightened and first shows where the images differ.`;
        }
      }
      canMark = isCount(changed) ? marked === changed : bitDepth === 8 && exact;
    } catch (error) {
      if (plan.kind === 'brighten') reason = error?.message || 'The picture could not be redrawn, so it is shown as saved.';
    }
    if (!canBrighten && !canMark) {
      canvas.width = 0;
      canvas.height = 0;
    }
    offered = [canBrighten ? 'bright' : '', canMark ? 'where' : '', 'saved'].filter(Boolean);
    view = offered[0];
    if (offered.length > 1) {
      mount(views, offered.map((key) => buttons.get(key)));
      views.hidden = false;
    }
    loaded = true;
    if (scale > 1) mount(sizeNote, `Enlarged ${fmt.int(scale)} times so single pixels can be seen: each square is one pixel of the ${dims} file.`);
    else if (Math.max(width, height) > SHOWN_LONG_SIDE) mount(sizeNote, `The file is ${dims} pixels and is fitted to the width available here; download it to see every pixel.`);
    sizeNote.hidden = !sizeNote.firstChild;
    draw();
  };
  const onError = () => {
    if (signal.aborted) return;
    views.hidden = true;
    explain.hidden = true;
    mount(frame, notice('warn', 'The difference image could not be loaded here. The link below still asks the service for the file.'));
    frame.className = 'difference__frame-failed';
  };
  img.addEventListener('load', onLoad, { signal });
  img.addEventListener('error', onError, { signal });
  img.src = artifact.url;

  const release = () => {
    img.removeEventListener('load', onLoad);
    img.removeEventListener('error', onError);
    img.removeAttribute('src');
    canvas.width = 0;
    canvas.height = 0;
  };
  return { nodes: [views, frame, explain, sizeNote, reading, h('div', { class: 'difference__actions' }, download)], release };
}

/**
 * Draw an image difference result into `host`.
 * @param {HTMLElement} host
 * @param {{job?: object|null, result?: object|null, artifacts?: Array<{path: string, available?: boolean, version?: string, url?: string}>}} options
 * @returns {() => void} cleanup
 */
export function renderImageDifference(host, { job = null, result = null, artifacts = [] } = {}) {
  const r = isObject(result) ? result : {};
  const params = isObject(job?.params) ? job.params : {};
  const controller = new AbortController();
  const { signal } = controller;
  const top = maxLevelOf(r.bitDepth);
  const pair = isObject(r.leftVsRight) ? r.leftVsRight : {};
  const largest = largestOf(pair);
  const names = channelNames(r, pair);
  const colour = names.length > 1;
  const area = areaOf(r);

  // 2. What was compared.
  const inputs = isObject(r.inputs) ? r.inputs : {};
  const pathOf = (side) => {
    const reported = inputs[side]?.path;
    if (typeof reported === 'string' && reported) return reported;
    return typeof params[side] === 'string' && params[side] ? params[side] : '';
  };
  const paths = { left: pathOf('left'), right: pathOf('right'), reference: pathOf('reference') };
  const hasReference = Boolean(paths.reference) || isObject(r.leftVsReference) || isObject(r.rightVsReference) || isObject(r.closerPixels);
  const [leftName, rightName, referenceName] = shortNames([paths.left, paths.right, paths.reference]);
  const versioned = ['left', 'right', 'reference'].some((side) => typeof inputs[side]?.version === 'string' && inputs[side].version);

  const copyButton = (path, whose) =>
    h(
      'button',
      {
        class: 'btn btn--sm',
        type: 'button',
        'aria-label': `Copy the path of ${whose}`,
        onclick: async () => {
          const ok = await copyText(path);
          if (signal.aborted) return;
          toast(ok ? 'Copied the path.' : `Copying failed. The path is ${path}`, { kind: ok ? 'ok' : 'err' });
        },
      },
      'Copy path',
    );
  const fileValue = (path, name, whose, same) =>
    path
      ? h('span', { class: 'difference__file' }, h('span', { class: 'mono difference__name', title: path }, name || path), copyButton(path, whose), same ? h('span', { class: 'difference__same' }, same) : null)
      : h('span', { class: 'difference__absent' }, NOT_REPORTED);
  const sameAs = (path, others) => {
    const match = path ? others.find(([other]) => other === path) : null;
    return match ? `The same file as ${match[1]}` : '';
  };

  const facts = [
    ['First image', fileValue(paths.left, leftName, 'the first image', '')],
    ['Second image', fileValue(paths.right, rightName, 'the second image', sameAs(paths.right, [[paths.left, 'the first image']]))],
    hasReference
      ? ['Reference', fileValue(paths.reference, referenceName, 'the reference', sameAs(paths.reference, [[paths.left, 'the first image'], [paths.right, 'the second image']]))]
      : null,
    ['Image size', isCount(r.width) && isCount(r.height) ? `${fmt.dims(r.width, r.height)} pixels` : null],
    ['Area compared', area.text],
    ['Bit depth', top === null ? null : `${fmt.int(r.bitDepth)} bits per channel (levels 0 to ${fmt.int(top)})`],
  ].filter(Boolean);

  // 3. Difference per channel.
  const changed = pair.changedPixels;
  const countLine = isCount(changed)
    ? `${changedPhrase(changed, r.pixels)}.${colour ? ' A pixel counts as different when any of its channels differs.' : ''}`
    : 'The job did not report how many pixels differ.';
  const channelsId = uid('difference-channels');

  // 4. The difference image.
  const pictureId = uid('difference-picture');
  const files = Array.isArray(artifacts) ? artifacts.filter(isObject) : [];
  const artifact = files.find((a) => a.path === ARTIFACT) ?? files.find((a) => baseName(a.path) === ARTIFACT) ?? null;
  const usable = artifact && artifact.available !== false && typeof artifact.url === 'string' && artifact.url;
  let picture = { nodes: [], release: () => {} };
  if (usable) {
    picture = picturePart({ artifact, plan: brightenPlan({ largest, bitDepth: r.bitDepth }), largest, top, bitDepth: r.bitDepth, changed, colour, labelId: pictureId, signal });
  } else {
    picture.nodes = [
      notice(
        'warn',
        artifact
          ? 'The difference image is no longer available: the job saved it, but the file cannot be read now. The numbers above are still what the job measured.'
          : 'The job reported no difference image. The numbers above are still what the job measured.',
      ),
    ];
  }

  // 5. Against the reference.
  const closer = closerCounts(r);
  const referenceId = uid('difference-reference');
  const referencePart = hasReference
    ? [
        h('h4', { class: 'difference__title' }, 'Against the reference'),
        h('p', { class: 'difference__verdict' }, closerSentence(closer)),
        closer && closer.total > 0 ? closerBar(closer) : null,
        closer ? ruleLine(r.closerPixels.criterion) : null,
        referenceTable(
          channelNames(r, r.leftVsReference, r.rightVsReference),
          [
            ['First image', r.leftVsReference],
            ['Second image', r.rightVsReference],
          ],
          r.pixels,
          r.units,
          top,
          referenceId,
        ),
      ]
    : null;

  const root = h(
    'div',
    { class: 'difference' },
    h('p', { class: 'difference__lead' }, leadSentence(r)),
    h('h4', { class: 'difference__title' }, 'What was compared'),
    h(
      'dl',
      { class: 'kv difference__facts' },
      facts.map(([label, value]) => [h('dt', null, label), h('dd', { class: value === null ? 'difference__absent' : null }, value === null ? NOT_REPORTED : value)]),
    ),
    versioned ? h('p', { class: 'difference__note' }, VERSIONS) : null,
    h('h4', { class: 'difference__title', id: channelsId }, 'Difference per channel'),
    h('p', { class: 'difference__count' }, countLine),
    names.length ? channelTable(names, pair, r.units, top, channelsId) : h('p', { class: 'difference__absent' }, 'The job reported no difference per channel.'),
    hasReference ? null : h('p', { class: 'difference__note' }, NO_REFERENCE),
    names.length === 1 && names[0] === 'Y' ? h('p', { class: 'difference__note' }, MASKS) : null,
    h('h4', { class: 'difference__title', id: pictureId }, 'Difference image'),
    picture.nodes,
    referencePart,
    isObject(result) && Object.keys(result).length
      ? h('details', { class: 'difference__raw' }, h('summary', null, 'Everything the job reported'), h('pre', { class: 'difference__json', tabindex: '0' }, JSON.stringify(result, null, 2)))
      : null,
  );
  host.appendChild(root);
  return () => {
    controller.abort();
    picture.release();
    root.remove();
  };
}
