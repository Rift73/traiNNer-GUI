// Result of a completed patch extraction job (analysis kind `extract-patches`): one dataset of a
// restoration recipe cut into paired PNG patches. The producer is studio/dataset_patches.py. Sizes are in
// low quality pixels; each ground truth rectangle is the same rectangle multiplied by the recipe scale.
// The job writes into its own folder and does not change the source images.
//
//   renderExtractPatches(host, {job, result, artifacts}) -> cleanup
//   extractPatchesProgressText(progress) -> one sentence, or ''
//
// Shapes:
//   job.params = {task, split, scale, cropSize, stride}
//   result = {scale, cropSize, stride, sourcePairs, patchPairs, sourceRoots: {gt: [path], lq: [path]},
//     sourceMetaInfo: path|null, sourceFilenameTemplate, dataset: {name, type, io_backend, filename_tmpl,
//     dataroot_gt, dataroot_lq, meta_info}, coordinates, destination, gtPath, lqPath, metaInfoPath,
//     manifestPath, reportPath, outputCount}
//   artifacts = [{path, available, version, url}]: patches/dataset.json, patches/patches.jsonl and
//     patches/meta_info.txt (the PNG patches are not among them)
//   a line of patches.jsonl = {key, source: {row, rootId, lq: {path, width, height, …}, gt: {…}},
//     lqBox: [left, top, width, height], gtBox: [...]}
//   progress = {stage: 'selecting'|'extracting'|'lq'|'gt'|'verifying', completed, total, sourceRow,
//     completedPatchPairs}; `verifying` counts source files checked for changes

import { h, mount, fmt, uid } from '../dom.js';
import { notice } from '../ui.js';
import { api } from '../api.js';
import { recipe } from '../state/recipe.js';
import { openAsRecipe } from '../actions.js';
import { highlightedPre } from '../recipe/yaml-highlight.js';
import { isObject, isText, baseName, toYaml, sameRoots, readNames, remainder, copyButton, absent, pathValue, downloadLink, lazyDetails, namesBlock } from './split.js';

const P = 'patches';
/** At most this many lines of the coordinates file are drawn as table rows. */
const ROWS_MAX = 100;

const WHERE = "The patches are PNG files in this job's folder on the trainer machine. The source images were not changed.";
const LINES = "These lines replace the same options of the recipe's dataset; its other options (crop size, batch size, augmentation) stay as they are.";
const NOT_SAVED = 'Nothing is saved or started. The recipe opens unsaved, with the dataset pointing at the patches.';
const EDGE = 'The last patch of a row or a column ends at the edge of the image, so it can overlap its neighbour by more.';

// ───────── pure helpers (exported for checks) ─────────

const whole = (v) => Number.isInteger(v) && v > 0;
const clean = (v) => (typeof v === 'string' ? v.trim() : '');

/**
 * One plain sentence for a running job's progress; '' when absent or of an unknown stage.
 * @param {object|null} progress
 */
export function extractPatchesProgressText(progress) {
  if (!isObject(progress)) return '';
  const { stage, completed, total } = progress;
  if (stage === 'selecting') {
    if (Number.isFinite(total)) return `Found ${fmt.plural(total, 'source pair')}.`;
    return Number.isFinite(completed) && completed > 0 ? `Reading the pairs: ${fmt.int(completed)} found so far.` : 'Reading the pairs.';
  }
  if (stage === 'extracting' || stage === 'lq' || stage === 'gt') {
    const parts = [];
    if (Number.isFinite(progress.sourceRow) && progress.sourceRow > 0) parts.push(`source image ${fmt.int(progress.sourceRow)}`);
    if (Number.isFinite(progress.completedPatchPairs)) parts.push(`${fmt.plural(progress.completedPatchPairs, 'patch pair')} so far`);
    return parts.length ? `Cutting patches: ${parts.join(', ')}.` : 'Cutting patches.';
  }
  if (stage === 'verifying') {
    const counted = Number.isFinite(completed) && Number.isFinite(total);
    return counted ? `Checking that the source images did not change: ${fmt.int(completed)} of ${fmt.plural(total, 'file')}.` : 'Checking that the source images did not change.';
  }
  return '';
}

/** "245 patch pairs cut from 5 source pairs." */
export function leadSentence(result) {
  const r = isObject(result) ? result : {};
  if (!Number.isFinite(r.patchPairs)) {
    return Number.isFinite(r.sourcePairs) ? `Patches were cut from ${fmt.plural(r.sourcePairs, 'source pair')}. The job did not report how many.` : 'The job did not report how many patch pairs it cut.';
  }
  const from = Number.isFinite(r.sourcePairs) ? ` cut from ${fmt.plural(r.sourcePairs, 'source pair')}` : ' cut';
  return `${fmt.plural(r.patchPairs, 'patch pair')}${from}.`;
}

/** "24 × 24 low quality pixels, 96 × 96 ground truth pixels (4×)"; '' when the size is absent. */
export function sizeText(cropSize, scale) {
  if (!whole(cropSize)) return '';
  const lq = `${fmt.dims(cropSize, cropSize)} low quality pixels`;
  if (!whole(scale)) return lq;
  return `${lq}, ${fmt.dims(cropSize * scale, cropSize * scale)} ground truth pixels (${fmt.int(scale)}×)`;
}

/** "20 low quality pixels: neighbouring patches overlap by 4"; '' when the step is absent. */
export function stepText(stride, cropSize) {
  if (!whole(stride)) return '';
  const step = fmt.plural(stride, 'low quality pixel');
  if (!whole(cropSize) || stride > cropSize) return step;
  return stride === cropSize ? `${step}: no overlap` : `${step}: neighbouring patches overlap by ${fmt.int(cropSize - stride)}`;
}

/**
 * Where the open recipe stands against these patches.
 * @param {object} state `recipe.get()`
 * @param {string|null} split the dataset key the patches were cut from
 * @param {object} result the job's result
 * @returns {{kind: 'none'|'checking'|'unchecked'|'unread'|'applied'|'nodataset'|'other'|'match', name: string, reason?: string, lqSize?: number|null}}
 *   `applied`: the recipe's dataset already reads the list of these patches. `lqSize`: the recipe's
 *   lq_size when it is a whole number larger than a patch (given for `match` and `applied`).
 */
export function patchesStanding(state, split, result) {
  const s = isObject(state) ? state : {};
  const r = isObject(result) ? result : {};
  const name = isText(s.name) ? s.name : '';
  if (s.phase !== 'open') return { kind: 'none', name };
  // The dataset is read from the service's parsed copy; one that belongs to older text says nothing.
  if (!isObject(s.config) || s.configCurrent === false) {
    if (s.check === 'failed') return { kind: 'unchecked', name, reason: isText(s.checkError) ? s.checkError : '' };
    if (s.check === 'done') return { kind: 'unread', name };
    return { kind: 'checking', name };
  }
  const mine = isObject(s.config.datasets) && isText(split) ? s.config.datasets[split] : null;
  if (!isObject(mine)) return { kind: 'nodataset', name };
  const lqSize = Number.isInteger(mine.lq_size) && whole(r.cropSize) && mine.lq_size > r.cropSize ? mine.lq_size : null;
  const list = clean(r.dataset?.meta_info);
  if (list && clean(mine.meta_info) === list) return { kind: 'applied', name, lqSize };
  const roots = isObject(r.sourceRoots) ? r.sourceRoots : {};
  if (!sameRoots(mine.dataroot_gt, roots.gt) || !sameRoots(mine.dataroot_lq, roots.lq)) {
    const has = (value) => isText(value) || (Array.isArray(value) && value.length > 0);
    return { kind: has(mine.dataroot_gt) && has(mine.dataroot_lq) ? 'other' : 'nodataset', name };
  }
  return { kind: 'match', name, lqSize };
}

/** The sentence shown in place of the button; '' when the button is offered. */
export function standingSentence(standing, split) {
  const which = standing.name ? `The open recipe (${standing.name})` : 'The open recipe';
  const instead = 'Open the recipe they were cut from, or copy the lines above.';
  switch (standing.kind) {
    case 'none':
      return 'No recipe is open.';
    case 'checking':
      return `${which} is being checked. Whether these patches belong to it shows here in a moment.`;
    case 'unchecked':
      return `${which} could not be checked${standing.reason ? ` (${standing.reason})` : ''}, so it is not known whether these patches belong to it. Check it again on the Recipe screen, or copy the lines above.`;
    case 'unread':
      return `${which} could not be read as it is written, so it is not known whether these patches belong to it. Solve its errors on the Recipe screen, or copy the lines above.`;
    case 'applied':
      return `${which} already uses these patches.`;
    case 'nodataset':
      return `${which} has no dataset ${split} with a ground truth and a low quality folder, so these patches do not belong to it. ${instead}`;
    case 'other':
      return `${which} ${split === 'train' ? 'trains on other folders' : `has other folders in its dataset ${split}`}, so these patches do not belong to it. ${instead}`;
    default:
      return '';
  }
}

/** The crop warning for a recipe whose lq_size is larger than a patch. */
export function cropWarning(lqSize, cropSize) {
  return `The recipe crops ${fmt.int(lqSize)} low quality pixels for training (lq_size), more than a patch holds (${fmt.int(cropSize)}). Lower lq_size to ${fmt.int(cropSize)} or less before training.`;
}

/** The first lines of the coordinates file as rows, and how many lines there are. */
export function readPatchRows(text, limit = ROWS_MAX) {
  const { names: lines, total } = readNames(text, limit);
  const rows = lines.map((line) => {
    let record = null;
    try {
      record = JSON.parse(line);
    } catch {
      // Shown as a line that could not be read.
    }
    if (!isObject(record)) return { unread: true };
    const source = isObject(record.source) ? record.source : {};
    const lq = isObject(source.lq) && isText(source.lq.path) ? source.lq.path : '';
    const gt = isObject(source.gt) && isText(source.gt.path) ? source.gt.path : '';
    const box = Array.isArray(record.lqBox) ? record.lqBox : [];
    return { key: isText(record.key) ? record.key : '', lq, gt, left: box[0], top: box[1] };
  });
  return { rows, total };
}

// ───────── pieces ─────────

/** One file or folder of the job: its name, an optional download link, and where it is. */
function fileBlock({ name, link = null, note = '', path, more = null }) {
  return h(
    'div',
    { class: `${P}__file`, role: 'group', 'aria-label': name },
    h('div', { class: `${P}__file-head` }, h('span', { class: `${P}__file-name` }, name), link),
    note ? h('p', { class: `${P}__note` }, note) : null,
    h('dl', { class: `kv ${P}__file-facts` }, h('dt', null, 'Path on the trainer machine'), h('dd', null, pathValue(path, P))),
    more,
  );
}

/** "Show the first patches": the first lines of the coordinates file as a table. */
function patchesTable(artifact, signal) {
  const draw = (text) => {
    const { rows, total } = readPatchRows(text, ROWS_MAX);
    if (!total) return { nodes: [h('p', { class: `${P}__note` }, 'The file holds no patches.')], focus: null };
    const more = total - rows.length;
    const cell = (value) => (Number.isFinite(value) ? fmt.int(value) : 'Not reported');
    const wrap = h(
      'div',
      { class: `${P}__table-wrap`, tabindex: '0', role: 'region', 'aria-label': 'First patches and where they came from' },
      h(
        'table',
        { class: `table ${P}__table` },
        h('caption', { class: `${P}__caption` }, 'Left and top are the corner of the patch in its source image, in low quality pixels.'),
        h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Patch'), h('th', { scope: 'col' }, 'Source image'), h('th', { scope: 'col', class: 'num' }, 'Left'), h('th', { scope: 'col', class: 'num' }, 'Top'))),
        h(
          'tbody',
          null,
          rows.map((row) =>
            row.unread
              ? h('tr', null, h('td', { colspan: '4', class: `${P}__muted` }, 'This line could not be read.'))
              : h(
                  'tr',
                  null,
                  h('td', { class: `${P}__cell` }, row.key || 'Not reported'),
                  h('td', { class: `${P}__cell`, title: row.lq || row.gt ? `Low quality: ${row.lq || 'not reported'}\nGround truth: ${row.gt || 'not reported'}` : null }, row.lq || row.gt || 'Not reported'),
                  h('td', { class: 'num' }, cell(row.left)),
                  h('td', { class: 'num' }, cell(row.top)),
                ),
          ),
        ),
      ),
    );
    return {
      nodes: [
        h('p', { class: `${P}__note` }, more > 0 ? `The first ${fmt.int(rows.length)} of ${fmt.plural(total, 'patch', 'patches')}, in the order of the file.` : `${fmt.plural(total, 'patch', 'patches')}, in the order of the file.`),
        wrap,
        more > 0 ? h('p', { class: `${P}__note` }, `and ${fmt.int(more)} more. The downloaded file has all of them.`) : null,
      ],
      focus: wrap,
    };
  };
  return lazyDetails({ artifact, signal, summary: 'Show the first patches', draw, what: 'file', prefix: P });
}

// ───────── renderer ─────────

/**
 * Draw a patch extraction result into `host`.
 * @param {HTMLElement} host
 * @param {{job?: object|null, result?: object|null, artifacts?: Array<{path: string, available?: boolean, version?: string, url?: string}>}} options
 * @returns {() => void} cleanup
 */
export function renderExtractPatches(host, { job = null, result = null, artifacts = [] } = {}) {
  const controller = new AbortController();
  const { signal } = controller;
  const r = isObject(result) ? result : {};
  const params = isObject(job?.params) ? job.params : {};
  const files = Array.isArray(artifacts) ? artifacts.filter(isObject) : [];
  const fileNamed = (name) => files.find((a) => baseName(a.path) === name) ?? null;
  const readable = (a) => Boolean(a && isText(a.url) && a.available !== false);
  const split = isText(params.split) ? params.split : null;
  /** Paths of `result` that are drawn below; everything else goes into the closing block. */
  const shown = [];
  const use = (key, fallback) => {
    if (r[key] !== undefined && r[key] !== null) {
      shown.push([key]);
      return r[key];
    }
    return fallback;
  };

  // 1 and 2. What was made, and from what.
  for (const key of ['patchPairs', 'sourcePairs']) if (Number.isFinite(r[key])) shown.push([key]);
  const cropSize = use('cropSize', params.cropSize);
  const scale = use('scale', params.scale);
  const stride = use('stride', params.stride);
  const roots = isObject(r.sourceRoots) ? r.sourceRoots : {};
  const rootLines = (key, label) => {
    const list = Array.isArray(roots[key]) ? roots[key].filter(isText) : isText(roots[key]) ? [roots[key]] : [];
    if (list.length) shown.push(['sourceRoots', key]);
    return list.length ? list.map((path) => h('div', { class: `${P}__root` }, h('span', { class: `${P}__root-label` }, label), pathValue(path, P))) : h('div', { class: `${P}__root` }, h('span', { class: `${P}__root-label` }, label), absent(undefined, P));
  };
  let selection = null;
  if (isText(r.sourceMetaInfo)) selection = h('div', { class: `${P}__root` }, 'The pairs listed in ', h('span', { class: `${P}__path` }, r.sourceMetaInfo));
  else if (r.sourceMetaInfo === null) selection = h('div', { class: `${P}__root` }, 'Every pair of the folders');
  if (selection) shown.push(['sourceMetaInfo']);
  const step = stepText(stride, cropSize);
  const facts = [
    ['Patch size', sizeText(cropSize, scale) || absent(undefined, P)],
    ['Step', step ? (whole(cropSize) && stride <= cropSize ? [`${step}. `, h('span', { class: `${P}__aside` }, EDGE)] : step) : absent(undefined, P)],
    ['Which data', split ? h('span', { class: 'mono' }, split) : absent(undefined, P)],
    ['Cut from', h('div', { class: `${P}__roots` }, rootLines('gt', 'Ground truth'), rootLines('lq', 'Low quality'), selection)],
  ];

  // 3. Where the patches are.
  const used = new Set();
  const take = (name) => {
    const artifact = fileNamed(name);
    if (artifact) used.add(artifact);
    return artifact;
  };
  const listFile = take('meta_info.txt');
  const coordinates = take('patches.jsonl');
  const record = take('dataset.json');
  for (const key of ['gtPath', 'lqPath', 'metaInfoPath', 'manifestPath', 'reportPath']) if (isText(r[key])) shown.push([key]);
  const others = files.filter((a) => !used.has(a) && readable(a));

  // 4. Recipe lines.
  const dataset = isObject(r.dataset) && Object.keys(r.dataset).length ? r.dataset : null;
  if (dataset) shown.push(['dataset']);
  const yamlText = dataset ? toYaml(split ? { datasets: { [split]: dataset } } : dataset) : '';
  const linesId = uid('patches-lines');

  // 5. The open recipe with these patches.
  const canApply = Boolean(dataset && split);
  const openButton = h('button', { class: 'btn btn--primary', type: 'button', hidden: true }, 'Open the recipe with these patches');
  // Focusable by script only: it takes the focus of the button when the button goes away under it.
  const why = h('p', { class: `${P}__why`, tabindex: '-1', hidden: true });
  const print = h('p', { class: `${P}__print`, hidden: true }, NOT_SAVED);
  const warnSlot = h('div', { class: `${P}__warn` });
  const errorSlot = h('div', { class: `${P}__error` });
  let busy = false;
  let gone = false;
  let drawn = '';
  let warned = '';
  const standingNow = () => patchesStanding(recipe.get(), split, r);
  /** Follows the recipe store by changing these nodes in place: nothing here is rebuilt. */
  const update = () => {
    if (gone || !canApply) return;
    const standing = standingNow();
    const sig = JSON.stringify([standing, busy]);
    if (sig === drawn) return;
    drawn = sig;
    const offered = standing.kind === 'match';
    const hadFocus = document.activeElement === openButton;
    openButton.hidden = !offered;
    openButton.disabled = busy;
    openButton.textContent = busy ? 'Preparing…' : 'Open the recipe with these patches';
    print.hidden = !offered;
    why.hidden = offered;
    // "Already uses these patches" is a quiet remark; every other sentence says why the button is missing.
    why.className = standing.kind === 'applied' ? `${P}__why ${P}__note` : `${P}__why`;
    why.textContent = standingSentence(standing, split);
    const warning = whole(standing.lqSize) ? cropWarning(standing.lqSize, cropSize) : '';
    if (warning !== warned) {
      warned = warning;
      mount(warnSlot, warning ? notice('warn', warning) : null);
    }
    if (!offered) mount(errorSlot, null);
    // The button that had the focus is gone: hand the focus to the sentence that took its place.
    if (!offered && hadFocus) why.focus();
  };
  const openWithPatches = async () => {
    if (busy || gone) return;
    if (standingNow().kind !== 'match') {
      update();
      return;
    }
    const before = recipe.get();
    const edits = Object.entries(dataset).map(([key, value]) => ({ path: ['datasets', split, key], value }));
    busy = true;
    mount(errorSlot, null);
    update();
    try {
      const answer = await api.render(before.yaml, edits, { signal });
      if (gone) return;
      if (!answer || typeof answer.yaml !== 'string') throw new Error('The service returned no recipe text for these patches.');
      if (recipe.get().yaml !== before.yaml) throw new Error('The open recipe changed while the patches were being put in. Nothing was opened. Try again.');
      // Made from the open recipe's text as it is now (checked just above): its unsaved changes are kept.
      await openAsRecipe({ yaml: answer.yaml, name: isText(before.name) ? before.name : '', label: 'training patches', derived: true });
    } catch (error) {
      if (error?.name === 'AbortError' || gone) return;
      const issues = Array.isArray(error?.issues) ? error.issues.filter((issue) => isObject(issue) && isText(issue.message)) : [];
      mount(
        errorSlot,
        notice('err', [
          `The recipe was not opened: ${error?.message || 'the service gave no reason.'}`,
          issues.length ? h('ul', { class: `${P}__notes` }, issues.map((issue) => h('li', null, isText(issue.path) ? `${issue.path}: ${issue.message}` : issue.message))) : null,
        ]),
      );
    } finally {
      busy = false;
      update();
    }
  };
  openButton.addEventListener('click', openWithPatches);
  const noAction = dataset
    ? 'The job did not say which dataset of the recipe the patches were cut from, so they cannot be put into a recipe from here. Copy the lines above.'
    : 'The job did not report the dataset options for the patches, so there is nothing to put into a recipe.';

  // 6. Whatever the job reported that is not drawn above.
  const rest = remainder(r, shown);

  const root = h(
    'div',
    { class: P },
    h('p', { class: `${P}__lead` }, leadSentence(r)),
    h('dl', { class: `kv ${P}__facts` }, facts.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])),

    h('h4', { class: `${P}__title` }, 'Where the patches are'),
    h('p', { class: `${P}__note` }, WHERE),
    fileBlock({ name: 'Ground truth patches', path: r.gtPath }),
    fileBlock({ name: 'Low quality patches', path: r.lqPath }),
    fileBlock({
      name: 'List of patches',
      link: downloadLink(listFile, 'Download the list of patches', 'meta_info.txt', P),
      note: 'A text file with one patch name per line; the recipe lines below point at it.',
      path: r.metaInfoPath,
      more: readable(listFile) ? namesBlock(listFile, { region: 'Names in the list of patches' }, signal, P) : null,
    }),
    fileBlock({
      name: 'Where each patch came from',
      link: downloadLink(coordinates, 'Download the coordinates', 'patches.jsonl', P),
      note: 'One JSON line per patch: its source image and the rectangle it was cut from, in both images.',
      path: r.manifestPath,
      more: readable(coordinates) ? patchesTable(coordinates, signal) : null,
    }),
    fileBlock({ name: 'Record', link: downloadLink(record, 'Download the record', 'dataset.json', P), note: 'A JSON file with the settings of this job and the dataset options.', path: r.reportPath }),
    others.length
      ? h('ul', { class: `${P}__others`, 'aria-label': 'Other files of the job' }, others.map((a) => h('li', null, h('a', { class: `${P}__download`, href: a.url, download: baseName(a.path) || 'output' }, String(a.path ?? 'output')))))
      : null,

    h('div', { class: `${P}__head` }, h('h4', { class: `${P}__title`, id: linesId }, 'Recipe lines'), yamlText ? copyButton('Copy', yamlText, 'Copied the recipe lines.') : null),
    yamlText
      ? [
          h('p', { class: `${P}__note` }, LINES),
          // Shown without the final line break (it would draw an empty last line); "Copy" keeps it.
          h('div', { class: `${P}__yaml`, tabindex: '0', role: 'region', 'aria-labelledby': linesId }, highlightedPre(yamlText.trimEnd())),
          h('p', { class: `${P}__note` }, "Written as YAML from the job's answer. The paths are absolute: when the patches are moved, change them in the recipe too."),
        ]
      : h('p', { class: `${P}__note` }, 'The job did not report the dataset options, so there are no recipe lines.'),

    h('div', { class: `${P}__action` }, canApply ? [warnSlot, openButton, why, print, errorSlot] : h('p', { class: `${P}__why` }, noAction)),

    rest
      ? h(
          'details',
          { class: `${P}__raw` },
          h('summary', { class: `${P}__raw-summary` }, 'Everything else the job reported'),
          h('pre', { class: `${P}__json`, tabindex: '0' }, JSON.stringify(rest, null, 2)),
        )
      : null,
  );
  host.appendChild(root);
  update();
  const unsubscribe = canApply ? recipe.subscribe(update) : null;

  return () => {
    gone = true;
    controller.abort();
    if (typeof unsubscribe === 'function') unsubscribe();
    openButton.removeEventListener('click', openWithPatches);
    root.remove();
  };
}
