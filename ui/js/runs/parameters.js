// Result of a completed network parameters job (analysis kind `network-parameters`): the named
// parameters and buffers of the network a recipe describes, with their shapes and sizes, and, when the
// job was asked to, what the recipe's transfer of starting weights would copy. The producer is
// studio/network_parameters.py (described in studio/API.md). `job` may be null.
//
//   renderNetworkParameters(host, {job, result, artifacts}) -> cleanup
//   networkParametersProgressText(progress, job) -> '' (the service reports no progress for this kind)
//
// Shapes:
//   result = {task, role, architecture, inspection: 'unwrapped_constructor'|'initial_generator_transfer',
//     parameterTensors, parameters, trainableParameters, matchedTrainableParameters, parameterBytes,
//     bufferElements, bufferBytes, preview: [row, …], outputCount, transfer?}
//   `matchedTrainableParameters` are the trainable parameters whose tensor names pass the recipe's
//   generator optimizer filter (train.optim_include_params_g / optim_exclude_params_g).
//   row = {name, kind: 'parameter'|'buffer', shape: [n, …], dtype, elements, bytes} and, for a parameter,
//     {requiresGrad, selectedByName} and, with a transfer, {initialization: 'transferred'|'constructor'}.
//   `preview` holds the first rows of the list file; the result does not say how many rows the file has.
//   transfer = {source, counts: {source, target, loaded, parameters, buffers, skipped}, labelPolicy,
//     freshSelectedElements, freshExcludedElements, loadedSelectedElements}
//   artifacts = [{path, absolutePath, available, version, url}]: `network-parameters.jsonl` (one row per
//   line, every named tensor) and, with a transfer, `transfer-receipt.json` =
//     {source, paramKey, labelPolicy, counts, loaded: [name, …], skipped: [{key, reason}, …]}

import { h, mount, fmt, uid } from '../dom.js';
import { notice, copyText, toast } from '../ui.js';
import { catalog } from '../state/catalog.js';
import { bytesText, typeWords } from './convert.js';
import { isObject, isText, baseName, remainder, pathValue, downloadLink, lazyDetails, refusal } from './split.js';

/** The list file is parsed whole to filter it: above either limit it is only offered as a download. */
export const ROWS_MAX = 20000;
const FILE_MAX_BYTES = 32 * 1024 * 1024;
/** At most this many rows of the list are drawn at once; the filter finds the others. */
export const ROWS_DRAWN = 500;

const LIST_FILE = 'network-parameters.jsonl';
const RECEIPT_FILE = 'transfer-receipt.json';

/** How the job read the network, per `inspection`. */
const READ = {
  unwrapped_constructor: 'Built as the recipe describes, without loading weights',
  initial_generator_transfer: "Built, then the recipe's transfer applied on the CPU",
};
/** Where a parameter's starting values come from, per `initialization`. */
const START = { transferred: 'copied', constructor: 'starts fresh' };
const COPY_FAILED = 'Copying failed. Select the name and copy it by hand.';

// ───────── pure helpers (exported for checks) ─────────

const isCount = (v) => Number.isFinite(v) && v >= 0;

/**
 * One plain sentence for a running job's progress. The job reports no progress, so this is always ''.
 * @returns {string}
 */
export function networkParametersProgressText() {
  return '';
}

/** The first sentence: how many parameters, in how many tensors, of what size. */
export function leadText(result) {
  const r = isObject(result) ? result : {};
  const named = isText(r.architecture);
  if (!isCount(r.parameters)) return `The job did not report how many parameters ${named ? r.architecture : 'the network'} has.`;
  const tensors = isCount(r.parameterTensors) ? ` in ${fmt.plural(r.parameterTensors, 'tensor')}` : '';
  const size = bytesText(r.parameterBytes);
  return `${named ? r.architecture : 'The network'} has ${fmt.plural(r.parameters, 'parameter')}${tensors}${size ? ` (${size})` : ''}.`;
}

/**
 * The second sentence: how many of the parameters are trainable and how many the recipe's optimizer filter
 * selects. Nothing is said about the filter when the job did not report it; '' when there is nothing to say.
 */
export function trainingText(result) {
  const r = isObject(result) ? result : {};
  const all = r.parameters;
  const trainable = r.trainableParameters;
  const selected = r.matchedTrainableParameters;
  if (!isCount(all) || all === 0) return '';
  const filter = (of) => `The recipe's optimizer filter selects ${selected === 0 ? 'none' : fmt.int(selected)} of ${of} for training.`;
  if (!isCount(trainable)) return isCount(selected) ? filter('them') : '';
  if (trainable === all) {
    if (!isCount(selected)) return 'All of them are trainable.';
    return selected === all ? 'All of them are trained.' : filter('them');
  }
  if (trainable === 0) return 'None of them are trainable.';
  const first = `${fmt.int(trainable)} of them ${trainable === 1 ? 'is' : 'are'} trainable.`;
  if (!isCount(selected)) return first;
  return `${first} ${selected === trainable ? 'All of the trainable ones are trained.' : filter('those')}`;
}

/**
 * The transfer sentence in three parts, [before the file name, the file name, after it]:
 * "242 of 244 tensors are copied from " + name + "; 2 are skipped."
 */
export function transferParts(transfer) {
  const t = isObject(transfer) ? transfer : {};
  const c = isObject(t.counts) ? t.counts : {};
  const name = isText(t.source) ? baseName(t.source) || t.source : 'the starting weights';
  if (!isCount(c.loaded)) return ['The job did not report how many tensors are copied from ', name, '.'];
  const of = isCount(c.target) ? ` of the network's ${fmt.int(c.target)} tensors` : c.loaded === 1 ? ' tensor' : ' tensors';
  const verb = c.loaded === 1 ? 'is' : 'are';
  // What the network keeps is its own count (target minus loaded). `skipped` counts names from the file
  // and from the network together, so it is said with the list of those names, not here.
  let after = '.';
  const kept = isCount(c.target) && c.target >= c.loaded ? c.target - c.loaded : null;
  if (kept === 0) after = '; none keep their starting values.';
  else if (kept !== null) after = `; ${fmt.int(kept)} ${kept === 1 ? 'keeps its starting values' : 'keep their starting values'}.`;
  return [`${fmt.int(c.loaded)}${of} ${verb} copied from `, name, after];
}

/** The transfer sentence as one string. */
export const transferText = (transfer) => transferParts(transfer).join('');

/** The warning for values that start fresh outside the optimizer filter (`freshExcludedElements` above zero). */
export function leftOutText(n) {
  return n === 1
    ? '1 value starts fresh and is left out of training by the optimizer filter. It would keep its starting value.'
    : `${fmt.int(n)} values start fresh and are left out of training by the optimizer filter. They would keep their starting values.`;
}

/** A shape as "16 × 3 × 3 × 3"; a tensor without dimensions holds a single value. */
export function shapeText(shape) {
  if (!Array.isArray(shape)) return '';
  return shape.length ? shape.map(String).join(' × ') : 'Single value';
}

/** What is said about a row besides its numbers: buffer, not trained, selected, copied or starts fresh. */
export function flagWords(row) {
  const words = [];
  if (row.kind === 'buffer') words.push('buffer');
  else if (isText(row.kind) && row.kind !== 'parameter') words.push(row.kind);
  if (row.requiresGrad === false) words.push('not trained');
  if (row.selectedByName === true) words.push('selected');
  if (isText(row.initialization)) words.push(START[row.initialization] ?? row.initialization);
  return words;
}

/** Whether a value is a row of the list: an object with a name. */
const isRow = (row) => isObject(row) && isText(row.name);

/**
 * The rows of the list file (one JSON object per line). `unread` counts lines that are not a row;
 * `over` is true when the file has more than `limit` rows, and reading stopped there.
 */
export function readRows(text, limit = ROWS_MAX) {
  const source = String(text ?? '').replace(/^﻿/, '');
  const rows = [];
  let unread = 0;
  let start = 0;
  while (start < source.length) {
    let end = source.indexOf('\n', start);
    if (end < 0) end = source.length;
    const line = source.slice(start, end).trim();
    start = end + 1;
    if (!line) continue;
    if (rows.length >= limit) return { rows, unread, over: true };
    try {
      const row = JSON.parse(line);
      if (isRow(row)) rows.push(row);
      else unread += 1;
    } catch {
      unread += 1;
    }
  }
  return { rows, unread, over: false };
}

/** The rows whose name contains `query` (plain text, upper and lower case alike); all of them for an empty query. */
export function matchingRows(rows, query) {
  const wanted = String(query ?? '').trim().toLowerCase();
  return wanted ? rows.filter((row) => row.name.toLowerCase().includes(wanted)) : rows;
}

/** The line above the table. `total` is null until the list file is read; `hasFile`: the list file can be read. */
export function shownText({ total = null, matching = 0, filtered = false, hasFile = true, reported = null }) {
  if (total === null) {
    const where = hasFile ? ' The list file has all of them.' : '';
    // `reported`: the number of named tensors the job stated (parameters and buffers), when it did.
    const of = isCount(reported) && reported > matching ? ` of ${fmt.int(reported)}` : '';
    return matching ? `Showing the first ${fmt.int(matching)}${of} named tensors.${where}` : `The job reported no preview of the named tensors.${where}`;
  }
  if (matching > ROWS_DRAWN) {
    return filtered
      ? `${fmt.int(matching)} of ${fmt.int(total)} match. The first ${fmt.int(ROWS_DRAWN)} are shown.`
      : `The first ${fmt.int(ROWS_DRAWN)} of ${fmt.int(total)} shown. Filter by name to find the others.`;
  }
  return `${fmt.int(matching)} of ${fmt.int(total)} shown.`;
}

/**
 * The skipped tensors of a transfer receipt: [{key, reason}] with '' for a reason the receipt does not give.
 * Throws (with the end of a sentence as its message) when the text is not such a receipt.
 */
export function readSkipped(text) {
  let receipt;
  try {
    receipt = JSON.parse(String(text ?? '').replace(/^﻿/, ''));
  } catch {
    throw new Error('it is not JSON.');
  }
  if (!isObject(receipt) || !Array.isArray(receipt.skipped)) throw new Error('it has no list of skipped tensors.');
  return receipt.skipped.map((entry) => {
    if (isObject(entry)) return { key: isText(entry.key) ? entry.key : JSON.stringify(entry), reason: isText(entry.reason) ? entry.reason : '' };
    return { key: typeof entry === 'string' ? entry : JSON.stringify(entry), reason: '' };
  });
}

// ───────── renderer ─────────

/**
 * Draw a network parameters result into `host`.
 * @param {HTMLElement} host
 * @param {{job?: object|null, result?: object|null, artifacts?: Array<{path: string, absolutePath?: string, available?: boolean, version?: string, url?: string}>}} options
 * @returns {() => void} cleanup
 */
export function renderNetworkParameters(host, { result = null, artifacts = [] } = {}) {
  const controller = new AbortController();
  const { signal } = controller;
  const r = isObject(result) ? result : {};
  const files = Array.isArray(artifacts) ? artifacts.filter(isObject) : [];
  const fileNamed = (name) => files.find((a) => baseName(a.path) === name) ?? null;
  const readable = (artifact) => Boolean(artifact && isText(artifact.url) && artifact.available !== false);
  const absent = () => h('span', { class: 'parameters__muted' }, 'Not reported');
  /** Paths of `result` that are drawn below; everything else goes into the closing block. */
  const shown = [];
  const drew = (...keys) => keys.forEach((key) => shown.push(Array.isArray(key) ? key : [key]));

  // 1. The outcome.
  const training = trainingText(r);
  if (isCount(r.parameters)) drew('parameters');
  if (isCount(r.parameters) && isCount(r.parameterTensors)) drew('parameterTensors');
  if (isCount(r.parameters) && isCount(r.parameterBytes)) drew('parameterBytes');
  if (training && isCount(r.trainableParameters)) drew('trainableParameters');
  if (training && isCount(r.matchedTrainableParameters)) drew('matchedTrainableParameters');

  // 2. Facts.
  let buffers = absent();
  if (isCount(r.bufferElements)) {
    drew('bufferElements');
    const size = bytesText(r.bufferBytes);
    if (size) drew('bufferBytes');
    buffers =
      r.bufferElements === 0
        ? 'None'
        : [`${fmt.plural(r.bufferElements, 'value')}${size ? `, ${size}` : ''}`, h('span', { class: 'parameters__aside' }, 'Stored with the weights, not trained')];
  }
  if (isText(r.architecture)) drew('architecture');
  if (isText(r.task)) drew('task');
  if (isText(r.inspection)) drew('inspection');
  const facts = [
    ['Network', isText(r.architecture) ? h('span', { class: 'mono' }, r.architecture) : absent()],
    ['Task', isText(r.task) ? catalog.taskLabel(r.task) || r.task : absent()],
    ['Buffers', buffers],
    ['How it was read', isText(r.inspection) ? (READ[r.inspection] ?? r.inspection) : absent()],
  ];

  // 3. What the transfer does.
  const t = isObject(r.transfer) ? r.transfer : null;
  const receipt = fileNamed(RECEIPT_FILE);
  let transferPart = null;
  if (t) {
    const counts = isObject(t.counts) ? t.counts : {};
    if (isText(t.source)) drew(['transfer', 'source']);
    if (isCount(counts.loaded)) {
      drew(['transfer', 'counts', 'loaded']);
      if (isCount(counts.target)) drew(['transfer', 'counts', 'target']);
      if (isCount(counts.skipped)) drew(['transfer', 'counts', 'skipped']);
    }
    const [before, name, after] = transferParts(t);
    const lines = [];
    if (isCount(t.freshSelectedElements)) {
      drew(['transfer', 'freshSelectedElements']);
      lines.push(h('li', null, `Selected for training and starting fresh: ${fmt.plural(t.freshSelectedElements, 'value')}.`));
    }
    if (isCount(t.loadedSelectedElements)) {
      drew(['transfer', 'loadedSelectedElements']);
      lines.push(h('li', null, `Selected for training and copied: ${fmt.plural(t.loadedSelectedElements, 'value')}.`));
    }
    let leftOut = null;
    if (isCount(t.freshExcludedElements)) {
      drew(['transfer', 'freshExcludedElements']);
      if (t.freshExcludedElements > 0) leftOut = notice('warn', leftOutText(t.freshExcludedElements));
      else lines.push(h('li', null, `Starting fresh but not selected for training: ${fmt.plural(0, 'value')}.`));
    }
    let policy = null;
    if (isText(t.labelPolicy)) {
      drew(['transfer', 'labelPolicy']);
      policy =
        t.labelPolicy === 'noRemap'
          ? h('p', { class: 'parameters__note' }, 'Class rows are copied by name, without remapping.')
          : h('p', { class: 'parameters__note' }, 'Label policy: ', h('span', { class: 'mono' }, t.labelPolicy));
    }
    /** The skipped keys of the receipt, each with the reason the receipt gives. */
    const drawSkipped = (text) => {
      const skipped = readSkipped(text);
      if (!skipped.length) return { nodes: [h('p', { class: 'parameters__note' }, 'The receipt lists no skipped tensor.')], focus: null };
      const drawn = skipped.slice(0, ROWS_DRAWN);
      const more = skipped.length - drawn.length;
      const box = h(
        'div',
        { class: 'parameters__names-box', tabindex: '0', role: 'region', 'aria-label': 'Names left out of the copy' },
        h(
          'ul',
          { class: 'parameters__skipped' },
          drawn.map((entry) => h('li', null, h('span', { class: 'parameters__key' }, entry.key), entry.reason ? `: ${entry.reason}` : ': no reason given')),
        ),
      );
      return {
        nodes: [
          h('p', { class: 'parameters__note' }, `${fmt.plural(skipped.length, 'name')} from the weights file or the network, each with the reason the receipt gives.`),
          box,
          more > 0 ? h('p', { class: 'parameters__note' }, `and ${fmt.int(more)} more. The downloaded receipt has all of them.`) : null,
        ],
        focus: box,
      };
    };
    transferPart = [
      h('h4', { class: 'parameters__title' }, 'What the transfer does'),
      h('p', { class: 'parameters__text' }, before, isText(t.source) ? h('span', { class: 'parameters__source', title: t.source }, name) : name, after),
      lines.length ? h('ul', { class: 'parameters__counts' }, lines) : null,
      leftOut,
      policy,
      readable(receipt) && counts.skipped !== 0
        ? lazyDetails({ artifact: receipt, signal, summary: 'Names left out of the copy', draw: drawSkipped, what: 'receipt', prefix: 'parameters', maxBytes: FILE_MAX_BYTES })
        : null,
    ];
  }

  // 4. Named tensors: the preview, then every row of the list file once it is asked for.
  const preview = Array.isArray(r.preview) ? r.preview.filter(isRow) : [];
  if (Array.isArray(r.preview)) drew('preview');
  const listFile = fileNamed(LIST_FILE);
  /** Every row of the list file, once it is read. */
  let all = null;
  let state = 'idle';
  /** The rows now in the table, in order. */
  let drawn = [];
  const tableId = uid('parameters-table');
  const filterId = uid('parameters-filter');
  const status = h('p', { class: 'parameters__note', role: 'status' });
  const showAll = h('button', { class: 'btn btn--sm', type: 'button' }, 'Show all');
  const filterInput = h('input', { class: 'input parameters__filter-input', id: filterId, type: 'search', autocomplete: 'off', spellcheck: 'false' });
  const filterField = h('div', { class: 'parameters__filter', hidden: true }, h('label', { class: 'field__label', for: filterId }, 'Filter by name'), filterInput);
  /** What the reading of the list says: that it failed, or that the file is too large to show. */
  const slot = h('div', { class: 'parameters__load' });
  const tbody = h('tbody');
  const wrap = h(
    'div',
    { class: 'parameters__table-wrap', tabindex: '0', role: 'region', 'aria-labelledby': tableId },
    h(
      'table',
      { class: 'table parameters__table' },
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', { scope: 'col' }, 'Name'),
          h('th', { scope: 'col' }, 'Shape'),
          h('th', { scope: 'col', class: 'num' }, 'Values'),
          h('th', { scope: 'col', class: 'num' }, 'Size'),
          h('th', { scope: 'col' }, 'Number type'),
          h('th', { scope: 'col' }, 'Notes'),
          h('th', { scope: 'col' }, h('span', { class: 'visually-hidden' }, 'Copy')),
        ),
      ),
      tbody,
    ),
  );
  const rowNode = (row) => {
    const words = typeWords(row.dtype);
    return h(
      'tr',
      null,
      // A plain cell: the shared `.table th` is the sticky column header.
      h('td', { class: 'parameters__name' }, row.name),
      h('td', { class: 'parameters__shape' }, shapeText(row.shape)),
      h('td', { class: 'num' }, isCount(row.elements) ? fmt.int(row.elements) : ''),
      h('td', { class: 'num', title: isCount(row.bytes) ? fmt.plural(row.bytes, 'byte') : null }, bytesText(row.bytes)),
      h('td', null, words ? h('span', { title: row.dtype }, words) : isText(row.dtype) ? h('span', { class: 'mono' }, row.dtype) : ''),
      h('td', null, flagWords(row).join(', ')),
      h(
        'td',
        { class: 'parameters__copy' },
        h('button', { class: 'btn btn--sm btn--quiet', type: 'button', dataset: { name: row.name } }, 'Copy name', h('span', { class: 'visually-hidden' }, ` ${row.name}`)),
      ),
    );
  };
  const drawRows = (list) => {
    const active = document.activeElement;
    const held = Boolean(active && tbody.contains(active));
    if (held && drawn.length <= list.length && drawn.every((row, index) => row.name === list[index].name)) {
      // A "Copy name" button of the table has the focus: the rows that are there stay, the others follow them.
      for (const row of list.slice(drawn.length)) tbody.appendChild(rowNode(row));
    } else {
      mount(tbody, list.map(rowNode));
      // The button that had the focus is gone: its table takes the focus.
      if (held) wrap.focus();
    }
    drawn = list;
  };
  const redraw = () => {
    const query = all ? filterInput.value : '';
    const matching = matchingRows(all ?? preview, query);
    const list = matching.length > ROWS_DRAWN ? matching.slice(0, ROWS_DRAWN) : matching;
    drawRows(list);
    wrap.hidden = list.length === 0;
    status.textContent = shownText({ total: all ? all.length : null, matching: matching.length, filtered: query.trim() !== '', hasFile: readable(listFile), reported: isCount(r.tensorCount) ? r.tensorCount : null });
  };
  const tooLarge = (text) => {
    state = 'large';
    const hadFocus = document.activeElement === showAll;
    // Focusable by script only: it takes the focus of the button it replaces.
    const note = h('p', { class: 'parameters__note', tabindex: '-1' }, text);
    showAll.hidden = true;
    mount(slot, note);
    if (hadFocus) note.focus();
  };
  const loadAll = async () => {
    if (state === 'loading' || state === 'done' || state === 'large' || signal.aborted) return;
    state = 'loading';
    // Not `disabled`: a disabled button would drop the keyboard focus it holds.
    showAll.setAttribute('aria-disabled', 'true');
    showAll.textContent = 'Reading the list…';
    let failure = '';
    let read = null;
    let large = '';
    try {
      let response;
      try {
        response = await fetch(listFile.url, { signal });
      } catch (cause) {
        if (cause?.name === 'AbortError') throw cause;
        throw new Error('The trainer service did not answer.');
      }
      if (!response.ok) throw new Error(await refusal(response));
      const sizeText = (bytes) => `The list is ${fmt.int(Math.round(bytes / 1048576))} MB, too large to show here. The download has all of it.`;
      const announced = Number(response.headers?.get?.('content-length'));
      if (Number.isFinite(announced) && announced > FILE_MAX_BYTES) {
        large = sizeText(announced);
      } else {
        const text = await response.text();
        if (text.length > FILE_MAX_BYTES) {
          large = sizeText(text.length);
        } else {
          read = readRows(text, ROWS_MAX);
          if (read.over) large = `The list has more than ${fmt.int(ROWS_MAX)} tensors, too many to show here. The download has all of them.`;
        }
      }
    } catch (error) {
      if (error?.name === 'AbortError') return;
      failure = error?.message || 'No reason was given.';
    }
    if (signal.aborted) return;
    showAll.removeAttribute('aria-disabled');
    showAll.textContent = 'Show all';
    if (failure) {
      state = 'failed';
      mount(slot, notice('err', `The list could not be read: ${failure}`));
      return;
    }
    if (large) {
      tooLarge(large);
      return;
    }
    state = 'done';
    all = read.rows;
    const hadFocus = document.activeElement === showAll;
    mount(slot, read.unread ? h('p', { class: 'parameters__note' }, `${fmt.plural(read.unread, 'line')} of the list could not be read and ${read.unread === 1 ? 'is' : 'are'} left out.`) : null);
    filterField.hidden = false;
    showAll.hidden = true;
    redraw();
    // The button that asked is gone: hand its focus to the field that replaced it.
    if (hadFocus) filterInput.focus();
  };
  const onCopy = async (event) => {
    const button = event.target?.closest?.('button[data-name]');
    const name = button?.dataset?.name;
    if (!isText(name)) return;
    const ok = await copyText(name);
    if (signal.aborted) return;
    toast(ok ? 'Copied the name.' : COPY_FAILED, { kind: ok ? 'ok' : 'err' });
  };
  showAll.addEventListener('click', loadAll);
  filterInput.addEventListener('input', redraw);
  tbody.addEventListener('click', onCopy);
  redraw();

  // 5. Files.
  const used = new Set();
  const fileBlock = (artifact, title, label, fallbackName, about) => {
    if (artifact) used.add(artifact);
    return h(
      'div',
      { class: 'parameters__file', role: 'group', 'aria-label': title },
      h('div', { class: 'parameters__file-head' }, h('span', { class: 'parameters__file-name' }, title), downloadLink(artifact, label, fallbackName, 'parameters')),
      h('p', { class: 'parameters__note' }, about),
      h('dl', { class: 'kv parameters__file-facts' }, h('dt', null, 'Path on the trainer machine'), h('dd', null, pathValue(readable(artifact) ? artifact.absolutePath : null, 'parameters'))),
    );
  };
  const fileBlocks = [
    fileBlock(listFile, 'List of every named tensor', 'Download the list', LIST_FILE, 'A text file with one JSON line per tensor: its name, shape, number type, number of values and size.'),
    t || receipt ? fileBlock(receipt, 'Transfer receipt', 'Download the receipt', RECEIPT_FILE, 'A JSON file with the names of the tensors that are copied, and the skipped ones with the reason.') : null,
  ];
  const others = files.filter((a) => !used.has(a) && readable(a));

  // 6. Whatever the job reported that is not drawn above.
  const rest = remainder(r, shown);

  const root = h(
    'div',
    { class: 'parameters' },
    h('p', { class: 'parameters__lead' }, leadText(r), training ? ` ${training}` : null),
    h('dl', { class: 'kv parameters__facts' }, facts.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])),
    transferPart,

    h('h4', { class: 'parameters__title', id: tableId }, 'Named tensors'),
    h('div', { class: 'parameters__controls' }, status, readable(listFile) ? [showAll, filterField] : null),
    slot,
    wrap,
    h(
      'p',
      { class: 'parameters__note' },
      'These names are what ',
      h('code', null, 'train.optim_include_params_g'),
      ' and ',
      h('code', null, 'optim_exclude_params_g'),
      ' match (as regular expressions).',
    ),

    h('h4', { class: 'parameters__title' }, 'Files'),
    fileBlocks,
    others.length
      ? h('ul', { class: 'parameters__others', 'aria-label': 'Other files of the job' }, others.map((a) => h('li', null, h('a', { class: 'parameters__download', href: a.url, download: baseName(a.path) || 'output' }, String(a.path ?? 'output')))))
      : null,

    rest
      ? h(
          'details',
          { class: 'parameters__raw' },
          h('summary', { class: 'parameters__raw-summary' }, 'Everything else the job reported'),
          h('pre', { class: 'parameters__json', tabindex: '0' }, JSON.stringify(rest, null, 2)),
        )
      : null,
  );
  host.appendChild(root);

  return () => {
    controller.abort();
    showAll.removeEventListener('click', loadAll);
    filterInput.removeEventListener('input', redraw);
    tbody.removeEventListener('click', onCopy);
    root.remove();
  };
}
