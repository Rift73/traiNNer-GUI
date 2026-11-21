// Result of a completed weights conversion job (analysis kind `convert-weights`): the weights of one
// PyTorch .pth file saved as a safetensors file. The producer is studio/weight_convert.py. `job` may be null.
//
//   renderConvertWeights(host, {job, result, artifacts}) -> cleanup
//   convertWeightsProgressText(progress) -> '' (the service reports no progress for this kind)
//
// Shapes:
//   result = {source, payload: name | null, tensorCount, dtypes: {'torch.float32': n, …}, tensorBytes,
//     metadataKeys: [name, …], format: 'safetensors'}
//   `payload` is the weights dictionary that was saved: the one asked for, or the file's only one when
//   none was asked for (the job refuses a file with several and no choice). It is null when the .pth holds
//   its tensors directly, with no dictionary inside; such a file has no network metadata either.
//   `tensorBytes` is the size of the tensor values, not of the saved file (which also has a header).
//   `metadataKeys` are the entries of the file's network metadata, kept as JSON in the safetensors header.
//   artifacts = [{path: '<source name>.safetensors', absolutePath, available, version, url}]; `absolutePath`
//   is the saved file on the trainer machine (absent from an older service). The result has no output path.

import { h, fmt } from '../dom.js';
import { notice, copyText, toast } from '../ui.js';

const NOT_REPORTED = 'Not reported';
const SAME = 'Tensor names and number types are the same as in the .pth file. The .pth file was not changed.';
/** Said only when the service did not report where the file is. */
const WHERE = "The file is in this job's folder on the trainer machine.";
const USE = 'This file can be used wherever a checkpoint is asked for.';

/** PyTorch number types in plain words, by the name after "torch.". */
const TYPE_WORDS = new Map([
  ['float16', '16-bit floats'],
  ['bfloat16', '16-bit floats (bfloat16)'],
  ['float32', '32-bit floats'],
  ['float64', '64-bit floats'],
  ['int8', '8-bit integers'],
  ['int16', '16-bit integers'],
  ['int32', '32-bit integers'],
  ['int64', '64-bit integers'],
  ['uint8', '8-bit unsigned integers'],
  ['uint16', '16-bit unsigned integers'],
  ['uint32', '32-bit unsigned integers'],
  ['uint64', '64-bit unsigned integers'],
  ['bool', 'true or false values'],
]);

// ───────── pure helpers (exported for checks) ─────────

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isCount = (v) => Number.isFinite(v) && v >= 0;
const baseName = (path) => {
  const text = String(path ?? '').replace(/[\\/]+$/, '');
  return text.slice(Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\')) + 1);
};

/** "6.13 MB" in decimal units, as the analysis job page writes sizes; '' when absent. */
export function bytesText(n) {
  if (!isCount(n)) return '';
  const units = [['TB', 1e12], ['GB', 1e9], ['MB', 1e6], ['kB', 1e3]];
  const found = units.find(([, size]) => n >= size);
  return found ? `${(n / found[1]).toFixed(2)} ${found[0]}` : fmt.plural(n, 'byte');
}

/** The plain words for a number type ("torch.float32" -> "32-bit floats"), or null for one this screen does not know. */
export function typeWords(name) {
  const key = String(name ?? '').replace(/^torch\./, '');
  return TYPE_WORDS.get(key) ?? null;
}

/** Number types with their counts, in the service's order: [{name, count, words | null}]. */
export function typeRows(dtypes) {
  if (!isObject(dtypes)) return [];
  return Object.entries(dtypes)
    .filter(([, count]) => isCount(count))
    .map(([name, count]) => ({ name, count, words: typeWords(name) }));
}

/**
 * The first sentence in three parts, [before the size, the size, after it]: tensors, source name, saved
 * name and size of the weights, each left out when the job did not report it.
 */
export function leadParts({ tensorCount, sourceName, savedName, format, tensorBytes }) {
  const counted = isCount(tensorCount);
  const what = counted ? fmt.plural(tensorCount, 'tensor') : 'The weights';
  const from = sourceName ? `${counted ? ' from' : ' of'} ${sourceName}` : '';
  const as = savedName ? ` as ${savedName}` : typeof format === 'string' && format ? ` as a ${format} file` : '';
  const size = bytesText(tensorBytes);
  if (!counted && !from && !as && !size) return ['The job did not report what it saved.', '', ''];
  const head = `${what}${from}${counted ? '' : ' were'} saved${as}`;
  return size ? [`${head} (`, size, ' of weights).'] : [`${head}.`, '', ''];
}

/** The first sentence as one string. */
export const leadText = (facts) => leadParts(facts).join('');

/**
 * One plain sentence for a running job's progress. The conversion reports no progress, so this is
 * always ''.
 * @returns {string}
 */
export function convertWeightsProgressText() {
  return '';
}

// ───────── renderer ─────────

/** "210 tensors of 32-bit floats, 34 of 64-bit integers"; an unknown type keeps its own name, in mono. */
function typesValue(rows) {
  return rows.map((row, index) => [
    index ? ', ' : null,
    `${index ? fmt.int(row.count) : fmt.plural(row.count, 'tensor')} of `,
    row.words ? h('span', { title: row.name }, row.words) : h('span', { class: 'mono convert__type' }, row.name),
  ]);
}

/**
 * Draw a weights conversion result into `host`.
 * @param {HTMLElement} host
 * @param {{job?: object|null, result?: object|null, artifacts?: Array<{path: string, available?: boolean, version?: string, url?: string}>}} options
 * @returns {() => void} cleanup
 */
export function renderConvertWeights(host, { job = null, result = null, artifacts = [] } = {}) {
  const r = isObject(result) ? result : {};
  /** The job answered with something: only then is anything said about what it did. */
  const reported = isObject(result) && Object.keys(result).length > 0;
  const params = isObject(job?.params) ? job.params : null;
  const controller = new AbortController();
  const { signal } = controller;

  const source = typeof r.source === 'string' && r.source ? r.source : typeof params?.weights === 'string' ? params.weights : '';
  const files = Array.isArray(artifacts) ? artifacts.filter((a) => isObject(a) && typeof a.path === 'string' && a.path) : [];
  const artifact = files.find((a) => /\.safetensors$/i.test(a.path)) ?? (files.length === 1 ? files[0] : null);
  const savedName = artifact ? baseName(artifact.path) : '';
  const usable = artifact && artifact.available !== false && typeof artifact.url === 'string' && artifact.url;

  // 1. Lead; the exact number of bytes is in the `title` of the size, as on the analysis job page.
  const [leadHead, leadSize, leadTail] = leadParts({ tensorCount: r.tensorCount, sourceName: baseName(source), savedName, format: r.format, tensorBytes: r.tensorBytes });
  const lead = [leadHead, leadSize ? h('span', { title: `${fmt.plural(r.tensorBytes, 'byte')} of tensor values` }, leadSize) : null, leadTail];

  // 2. Facts.
  const copyButton = h(
    'button',
    {
      class: 'btn btn--sm',
      type: 'button',
      'aria-label': 'Copy the path of the source file',
      onclick: async () => {
        const ok = await copyText(source);
        if (signal.aborted) return;
        toast(ok ? 'Copied the path.' : `Copying failed. The path is ${source}`, { kind: ok ? 'ok' : 'err' });
      },
    },
    'Copy path',
  );
  const absent = () => h('span', { class: 'convert__absent' }, NOT_REPORTED);
  const sourceValue = source ? h('span', { class: 'convert__file' }, h('span', { class: 'mono convert__name', title: source }, baseName(source) || source), copyButton) : absent();

  let dictionary;
  if (typeof r.payload === 'string' && r.payload) {
    // Asked for by name, or chosen by the service because the file has exactly one.
    const unasked = params !== null && (params.payload === null || params.payload === undefined);
    dictionary = [h('span', { class: 'mono' }, r.payload), unasked ? h('span', { class: 'convert__aside' }, "The file's only one; none was asked for") : null];
  } else if (r.payload === null) {
    dictionary = 'None: the .pth file holds its tensors directly, not inside a named dictionary';
  } else {
    dictionary = absent();
  }

  const rows = typeRows(r.dtypes);
  const keys = Array.isArray(r.metadataKeys) ? r.metadataKeys.filter((key) => typeof key === 'string') : null;
  let kept;
  if (keys === null) kept = absent();
  else if (!keys.length) kept = 'Nothing besides the tensors';
  else kept = h('span', { title: 'Entries of the network metadata, saved with the tensors' }, keys.map((key, index) => [index ? ', ' : null, h('span', { class: 'mono' }, key)]));

  const facts = [
    ['Source file', sourceValue],
    ['Weights dictionary', h('span', { class: 'convert__file' }, dictionary)],
    ['Number types', rows.length ? typesValue(rows) : absent()],
    ['Kept with the file', kept],
  ];

  // 3. The file: the download, where it is on the trainer machine (when the service says so), and what
  // can be done with it next.
  const size = artifact && isCount(artifact.size) ? bytesText(artifact.size) : '';
  const savedPath = usable && typeof artifact.absolutePath === 'string' && artifact.absolutePath ? artifact.absolutePath : '';
  const copySavedButton = h(
    'button',
    {
      class: 'btn btn--sm',
      type: 'button',
      onclick: async () => {
        const ok = await copyText(savedPath);
        if (signal.aborted) return;
        toast(ok ? 'Copied the path.' : 'Copying failed. Select the path and copy it by hand.', { kind: ok ? 'ok' : 'err' });
      },
    },
    'Copy path',
    h('span', { class: 'visually-hidden' }, ` of ${savedName || 'the safetensors file'}`),
  );
  /** Open the start dialog of another job with this file as its weights. Loaded on the click: analysis.js loads this module. */
  const useButton = (label, kind) =>
    h(
      'button',
      {
        class: 'btn btn--sm btn--quiet',
        type: 'button',
        onclick: async () => {
          try {
            const { openAnalysisDialog } = await import('./analysis.js');
            if (signal.aborted) return;
            await openAnalysisDialog({ kind, weights: savedPath });
          } catch (error) {
            if (!signal.aborted) toast(`The dialog could not be opened: ${error.message}`, { kind: 'err' });
          }
        },
      },
      label,
    );
  const filePart = usable
    ? [
        h('div', { class: 'convert__actions' }, h('a', { class: 'btn btn--sm', href: artifact.url, download: savedName || 'weights.safetensors' }, `Download ${savedName || 'the file'}${size ? ` (${size})` : ''}`)),
        savedPath
          ? [
              h(
                'dl',
                { class: 'kv convert__facts' },
                h('dt', null, 'Path on the trainer machine'),
                h('dd', null, h('span', { class: 'convert__file' }, h('span', { class: 'mono convert__name' }, savedPath), copySavedButton)),
              ),
              h('p', { class: 'convert__note' }, USE),
              h('div', { class: 'convert__actions' }, useButton('Try on an image…', 'predict'), useButton('Export to ONNX…', 'export')),
            ]
          : h('p', { class: 'convert__note' }, WHERE),
      ]
    : notice(
        'warn',
        artifact
          ? 'The safetensors file is no longer available: the job saved it, but the file cannot be read now. The facts above are still what the job reported.'
          : 'The job reported no output file. The facts above are still what the job reported.',
      );

  const root = h(
    'div',
    { class: 'convert' },
    h('p', { class: 'convert__lead' }, lead),
    h('dl', { class: 'kv convert__facts' }, facts.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])),
    reported ? h('p', { class: 'convert__note' }, SAME) : null,
    h('h4', { class: 'convert__title' }, 'The safetensors file'),
    filePart,
    reported
      ? h('details', { class: 'convert__raw' }, h('summary', null, 'Everything the job reported'), h('pre', { class: 'convert__json', tabindex: '0' }, JSON.stringify(result, null, 2)))
      : null,
  );
  host.appendChild(root);
  return () => {
    controller.abort();
    root.remove();
  };
}
