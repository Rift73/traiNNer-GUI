// Results of two analysis kinds that look at held-out data:
//   'evaluate'       one checkpoint measured on the recipe's validation or test data (studio/evaluate.py,
//                    traiNNer/evaluation.py): scores per dataset and what kind of score each is, results by
//                    class where the evaluation kept class counts (classes.js), the decision for each
//                    image where a classifier's evaluation kept them, the change between frames where a
//                    video evaluation kept it, a detector's operating points where it kept those, saved
//                    images, recorded files, and the way to the compare view with this job on the left.
//   'split-leakage'  files with identical encoded bytes in the training and the validation data
//                    (studio/split_audit.py). Also used by the Data screen.
// Everything shown comes from the job and its result; absent fields stay absent.
//
//   renderEvaluation(host, {job, result, artifacts}) -> cleanup
//   evaluationProgressText(progress) -> one sentence, or ''
//   renderSplitLeakage(host, {job, result, artifacts}) -> cleanup
//   splitLeakageProgressText(progress) -> one sentence, or ''

import { h, fmt, uid } from '../dom.js';
import { api, sizedUrl, THUMB_SIZE } from '../api.js';
import { badge, notice } from '../ui.js';
import { go, compareHref } from '../actions.js';
import { runs } from '../state/runs.js';
import { mountClassReport } from './classes.js';

const isNum = Number.isFinite;
const isText = (value) => typeof value === 'string' && value !== '';
const asObject = (value) => (value && typeof value === 'object' && !Array.isArray(value) ? value : null);
const reason = (error) => (error && typeof error.message === 'string' && error.message.trim()) || 'No reason was given.';

const baseName = (path) => {
  const text = String(path ?? '');
  return text.slice(Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\')) + 1) || text;
};

/** "A", "A and B", "A, B and C". */
const listText = (items) => (items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`);

// ════════════════════════ evaluation ════════════════════════

/** Tasks whose evaluator reports its own metrics and never saves images (studio/evaluate.py refuses saveImages). */
const PROVIDER_TASKS = new Set(['image_classification', 'semantic_segmentation', 'object_detection']);
/** Detection metrics that are counts (traiNNer/tasks/object_detection/metrics.py), shown as whole numbers. */
const DETECTION_COUNTS = new Set(['targets', 'false_positives50']);
/** Tasks whose evaluator can keep a confusion matrix (the recipe option val.record_samples). */
const CLASS_TASKS = new Set(['image_classification', 'semantic_segmentation']);
/** What the `aggregation` word of a recorded score means (GET /api/analysis/evaluations). */
const AGGREGATION_TEXT = {
  meanPerSample: 'Each score is the mean of the per-image scores.',
  pooledDataset: 'Each score is computed over the whole dataset by the evaluator, not as a mean of per-image scores.',
};
const IMAGE_PAGE = 48;
/** The service's largest page of measured summaries; a job has one summary per dataset. */
const SUMMARY_PAGE = 250;
const FILE_PAGE = 50;
const PNG_RE = /\.png$/i;
/** A decisions file above this size (20 MB) is offered as a download only: it is not read into a table. */
const DECISION_MAX_BYTES = 20 * 1000 * 1000;
const DECISION_PAGE = 100;

/**
 * A measured score with enough digits to tell two checkpoints apart: 4 decimals from 1 up,
 * 4 significant digits below 1 ("31.2047", "0.9132", "0.01733"). '' when not a finite number.
 */
export function scoreText(value) {
  if (!isNum(value)) return '';
  return Math.abs(value) >= 1 ? value.toFixed(4) : value.toPrecision(4);
}

/** The cell word for a non-finite marker of the result ('nan', 'inf', '-inf' from Python's str()). */
export function nonfiniteWord(marker) {
  const key = String(marker ?? '').trim().toLowerCase();
  if (key === 'nan') return 'NaN';
  if (key === 'inf' || key === '+inf' || key === 'infinity') return 'Infinite';
  if (key === '-inf' || key === '-infinity') return 'Negative infinite';
  return key ? String(marker) : 'NaN';
}

/** The same marker inside a sentence. */
function nonfiniteSentenceWord(marker) {
  const key = String(marker ?? '').trim().toLowerCase();
  if (key === 'nan' || !key) return 'NaN (not a number)';
  if (key === 'inf' || key === '+inf' || key === 'infinity') return 'infinite';
  if (key === '-inf' || key === '-infinity') return 'negative infinite';
  return String(marker);
}

/**
 * Every metric name of every dataset, finite or not: whole-dataset scores first, then the per-class
 * ones ("IoU/road", "AP50/car"), each group in the order the service listed them.
 */
export function metricNames(datasets) {
  const seen = [];
  for (const row of Array.isArray(datasets) ? datasets : []) {
    for (const name of [...Object.keys(asObject(row?.metrics) ?? {}), ...Object.keys(asObject(row?.nonfinite) ?? {})]) {
      if (!seen.includes(name)) seen.push(name);
    }
  }
  return [...seen.filter((name) => !name.includes('/')), ...seen.filter((name) => name.includes('/'))];
}

/** "the validation data", "the test data", "the validation and test data", from dataset keys. */
export function holdoutWords(splits, split) {
  const keys = Array.isArray(splits) && splits.length ? splits.map(String) : isText(split) ? [split] : [];
  if (!keys.length) return 'every validation and test dataset';
  const phases = new Set(keys.map((key) => key.split('_')[0]));
  const val = phases.has('val');
  const test = phases.has('test');
  if (val && test) return 'the validation and test data';
  if (val) return 'the validation data';
  if (test) return 'the test data';
  return `the dataset${keys.length === 1 ? '' : 's'} ${listText(keys)}`;
}

/**
 * "Measured on the CPU with 32-bit floats." from device and precision; only what the result states.
 * For the GPU, `extra` may name the card and its number: "Measured on the GPU <name> (CUDA device <n>) with 32-bit floats."
 */
export function measuredHowText(device, precision, extra) {
  const key = isText(device) ? device.toLowerCase() : '';
  const card = isText(extra?.deviceName) ? ` ${extra.deviceName.trim()}` : '';
  const index = Number.isInteger(extra?.deviceIndex) ? ` (CUDA device ${extra.deviceIndex})` : '';
  const where = !key ? '' : key === 'cpu' ? 'on the CPU' : key === 'cuda' ? `on the GPU${card}${index}` : `on the device ${device}`;
  const WORDS = { float32: '32-bit floats', float16: '16-bit floats', bfloat16: 'bfloat16 numbers', float64: '64-bit floats' };
  const how = !isText(precision) ? '' : `with ${WORDS[precision.toLowerCase()] ?? precision}`;
  const parts = [where, how].filter(Boolean);
  return parts.length ? `Measured ${parts.join(' ')}.` : '';
}

/** One plain sentence about a running evaluation ({stage: 'evaluating', completed, total, dataset?}), or ''. */
export function evaluationProgressText(progress) {
  if (!progress || typeof progress !== 'object' || progress.stage !== 'evaluating') return '';
  const done = progress.completed;
  const total = progress.total;
  if (!isNum(done) || !isNum(total) || total < 1) return '';
  if (done <= 0) return total === 1 ? 'Measuring the dataset.' : `Measuring the first of ${fmt.int(total)} datasets.`;
  const last = isText(progress.dataset) ? ` Last finished: ${progress.dataset}.` : '';
  return `${fmt.int(done)} of ${fmt.plural(total, 'dataset')} measured.${last}`;
}

/** What a recorded file of an evaluation is, in plain words; '' when not known. */
export function evaluationFileText(path) {
  const name = baseName(path);
  if (name === 'evaluation.yml') return 'The recipe as it was evaluated';
  if (name === 'validation-summary.sqlite3') return 'Recorded scores per dataset (database)';
  if (name === 'validation-samples.sqlite3') return 'Recorded scores per image (database)';
  if (name === 'validation-images.sqlite3') return 'Index of the saved images (database)';
  if (/\.sqlite3$/i.test(name)) return 'Recorded scores (database)';
  if (/(^|[\\/])classifications[\\/][^\\/]+\.jsonl$/i.test(String(path ?? ''))) return "Each image's predicted class (JSON lines)";
  if (/(^|[\\/])temporal-errors[\\/][^\\/]+\.jsonl$/i.test(String(path ?? ''))) return 'Each scored change between two frames (JSON lines)';
  if (/(^|[\\/])operating-points[\\/][^\\/]+\.jsonl$/i.test(String(path ?? ''))) return "The detector's results at each confidence (JSON lines)";
  if (/(^|[\\/])detections[\\/][^\\/]+\.json$/i.test(String(path ?? ''))) return 'The detected boxes, in COCO results form (JSON)';
  return '';
}

function scoresTable(datasets, task, titleId) {
  const names = metricNames(datasets);
  const counts = task === 'object_detection';
  const cell = (row, name) => {
    const nonfinite = asObject(row.nonfinite);
    if (nonfinite && name in nonfinite) {
      return h('td', { class: 'num holdout__nonfinite', title: String(nonfinite[name]) }, nonfiniteWord(nonfinite[name]));
    }
    const metrics = asObject(row.metrics);
    if (!metrics || !(name in metrics)) return h('td', { class: 'num holdout__absent' }, 'Not measured');
    const value = metrics[name];
    if (!isNum(value)) return h('td', { class: 'num holdout__absent' }, 'Not reported');
    const text = counts && DETECTION_COUNTS.has(name) && Number.isInteger(value) ? fmt.int(value) : scoreText(value);
    return h('td', { class: 'num', title: String(value) }, text);
  };
  return h(
    'div',
    { class: 'holdout__tablewrap', tabindex: '0', role: 'region', 'aria-labelledby': titleId },
    h(
      'table',
      { class: 'table holdout__table' },
      h(
        'thead',
        null,
        h('tr', null, h('th', { scope: 'col' }, 'Dataset'), h('th', { scope: 'col', class: 'num' }, 'Samples'), names.map((name) => h('th', { scope: 'col', class: 'num' }, name))),
      ),
      h(
        'tbody',
        null,
        datasets.map((row) =>
          h(
            'tr',
            null,
            // A td, not a th: `.table th` is sticky to the top of the scrolling box.
            h('td', { class: 'holdout__rowhead' }, isText(row.dataset) ? row.dataset : h('span', { class: 'holdout__absent' }, 'No name')),
            h('td', { class: isNum(row.samples) ? 'num' : 'num holdout__absent' }, isNum(row.samples) ? fmt.int(row.samples) : 'Not reported'),
            names.map((name) => cell(row, name)),
          ),
        ),
      ),
    ),
  );
}

function nonfiniteNotice(datasets) {
  const lines = [];
  for (const row of datasets) {
    const nonfinite = asObject(row.nonfinite);
    if (!nonfinite) continue;
    for (const [name, marker] of Object.entries(nonfinite)) {
      lines.push(`${name} on ${isText(row.dataset) ? row.dataset : 'a dataset without a name'} is ${nonfiniteSentenceWord(marker)}.`);
    }
  }
  if (!lines.length) return null;
  return notice(
    'warn',
    h(
      'div',
      { class: 'holdout__noticebody' },
      h('p', { class: 'holdout__noticeline' }, lines.length === 1 ? 'One score is not a finite number:' : `${fmt.int(lines.length)} scores are not finite numbers:`),
      h('ul', { class: 'holdout__noticelist' }, lines.map((text) => h('li', null, text))),
    ),
  );
}

/**
 * What kind of score the table holds: one sentence per `aggregation` word found in the rows of
 * GET /api/analysis/evaluations ([{dataset, metric, aggregation}]). A kind that does not cover every
 * score of the table names the metrics it covers; scores no row speaks of are left unexplained.
 * @param {Array} rows
 * @param {Array} datasets the rows of the scores table ([{dataset, metrics, nonfinite}])
 * @returns {string[]}
 */
export function aggregationSentences(rows, datasets) {
  const shown = new Map(); // metric -> the datasets whose row holds it
  let cells = 0;
  (Array.isArray(datasets) ? datasets : []).forEach((row, index) => {
    if (!row || typeof row !== 'object') return;
    // A dataset without a name cannot be matched to a recorded row: its scores stay uncovered.
    const dataset = isText(row.dataset) ? row.dataset : index;
    for (const name of new Set([...Object.keys(asObject(row.metrics) ?? {}), ...Object.keys(asObject(row.nonfinite) ?? {})])) {
      if (!shown.has(name)) shown.set(name, new Set());
      if (!shown.get(name).has(dataset)) cells += 1;
      shown.get(name).add(dataset);
    }
  });
  const covered = new Map(); // aggregation -> metric -> datasets
  for (const row of Array.isArray(rows) ? rows : []) {
    if (!row || typeof row !== 'object' || !isText(row.metric) || !isText(row.dataset)) continue;
    if (!isText(row.aggregation) || !Object.hasOwn(AGGREGATION_TEXT, row.aggregation)) continue;
    if (!shown.get(row.metric)?.has(row.dataset)) continue;
    if (!covered.has(row.aggregation)) covered.set(row.aggregation, new Map());
    const metrics = covered.get(row.aggregation);
    if (!metrics.has(row.metric)) metrics.set(row.metric, new Set());
    metrics.get(row.metric).add(row.dataset);
  }
  const kinds = Object.keys(AGGREGATION_TEXT).filter((kind) => covered.has(kind));
  const count = (metrics) => [...metrics.values()].reduce((sum, set) => sum + set.size, 0);
  if (kinds.length === 1 && count(covered.get(kinds[0])) === cells) return [AGGREGATION_TEXT[kinds[0]]];
  return kinds.map((kind) => {
    const names = [...covered.get(kind)].map(([metric, sets]) => (sets.size === shown.get(metric).size ? metric : `${metric} on ${listText([...sets])}`));
    const text = AGGREGATION_TEXT[kind];
    return `For ${listText(names)}: ${text[0].toLowerCase()}${text.slice(1)}`;
  });
}

/** The sentences above, under the scores table once the rows are read; nothing when they cannot be. */
function kindsLine(job, datasets, signal) {
  const el = h('div', { class: 'holdout__kinds', hidden: true });
  api.analysisEvaluations(job.id, { signal }).then(
    (rows) => {
      if (signal.aborted) return;
      const sentences = aggregationSentences(rows, datasets);
      el.replaceChildren(...sentences.map((text) => h('p', { class: 'holdout__quiet' }, text)));
      el.hidden = !sentences.length;
    },
    () => {
      // Not knowing the kind is no error of the report: the scores stand as measured.
    },
  );
  return el;
}

/**
 * "Results by class": one row per evaluation of the job that kept class counts (GET /api/analysis/validation),
 * each opening the class report of classes.js under the list. Hidden until the answer says there is
 * something to show: nothing for a task without classes, one quiet sentence for a classification or
 * segmentation that kept none. `order` is the dataset names in the order of the scores table.
 * @returns {{el: HTMLElement, destroy: () => void}}
 */
function classesPart(job, task, order, signal) {
  const titleId = uid('holdout-classes');
  const title = h('h4', { class: 'holdout__title', id: titleId, tabindex: '-1' }, 'Results by class');
  const body = h('div', { class: 'holdout__classbody' });
  const reportHost = h('div', { class: 'holdout__classes', hidden: true });
  const part = h('section', { class: 'holdout__part', hidden: true, 'aria-labelledby': titleId }, title, body, reportHost);
  let report = null;

  const close = (button) => {
    report?.destroy();
    report = null;
    reportHost.hidden = true;
    // "Hide classes" is gone with the report: back to the button that opened it.
    button.focus();
  };
  const show = (row, button) => {
    report?.destroy();
    reportHost.hidden = false;
    report = mountClassReport(reportHost, {
      evaluation: row,
      request: (params, opts) => api.analysisClasses(job.id, params, opts),
      headingText: `Class results: ${isText(row.dataset) ? row.dataset : 'dataset not reported'}`,
      onClose: () => close(button),
      retry: true,
      headingLevel: 5,
    });
  };
  const rowItem = (row) => {
    const button = h('button', { class: 'btn btn--sm', type: 'button' }, 'Show classes', isText(row.dataset) ? h('span', { class: 'visually-hidden' }, ` of ${row.dataset}`) : null);
    button.addEventListener('click', () => show(row, button));
    return h('li', { class: 'holdout__classrow' }, h('span', { class: 'holdout__classname' }, isText(row.dataset) ? row.dataset : h('span', { class: 'holdout__absent' }, 'No name')), button);
  };

  const load = (asked = false) => {
    if (asked) {
      // "Try again" is replaced by the answer: keep the keyboard in this part meanwhile.
      body.replaceChildren(h('p', { class: 'holdout__quiet', role: 'status' }, 'Reading the results by class…'));
      title.focus();
    }
    api.analysisValidation(job.id, { limit: SUMMARY_PAGE }, { signal }).then(
      (page) => {
        if (signal.aborted) return;
        const all = (Array.isArray(page?.evaluations) ? page.evaluations : []).filter((row) => row && typeof row === 'object');
        const place = (row) => (order.includes(row.dataset) ? order.indexOf(row.dataset) : order.length);
        const rows = all.filter((row) => Number.isInteger(row.id) && row.hasClassCounts === true).sort((a, b) => place(a) - place(b));
        const kind = isText(task) ? task : all.map((row) => row.task).find(isText);
        if (rows.length) {
          body.replaceChildren(h('ul', { class: 'holdout__classrows', 'aria-labelledby': titleId }, rows.map(rowItem)));
        } else if (CLASS_TASKS.has(kind)) {
          body.replaceChildren(
            h('p', { class: 'holdout__quiet' }, 'This evaluation kept no per-class counts; the recipe option ', h('span', { class: 'mono' }, 'val.record_samples'), ' keeps them.'),
          );
        } else {
          body.replaceChildren();
        }
        part.hidden = !body.firstChild;
      },
      (error) => {
        if (signal.aborted) return;
        const again = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => load(true) }, 'Try again');
        body.replaceChildren(notice('err', `Whether this evaluation kept results by class could not be read: ${reason(error)}`, again));
        part.hidden = false;
        // Only when the person is still here: focus is never pulled back from somewhere else.
        if (asked && document.activeElement === title) again.focus();
      },
    );
  };
  load();

  return {
    el: part,
    destroy() {
      report?.destroy();
      report = null;
    },
  };
}

// ──────── decisions per image ────────

/**
 * The rows of a decisions file (traiNNer/tasks/image_classification/results.py): JSON lines, one object
 * per evaluated image ({input, targetId, target, correct, top: [{id, label, probability}]}, `top` most
 * likely first), in the order of the evaluation. Empty lines are skipped; any other line that is not a
 * JSON object is counted and left out.
 * @param {string} text
 * @returns {{rows: object[], unreadable: number}}
 */
export function parseDecisions(text) {
  const rows = [];
  let unreadable = 0;
  for (const line of String(text ?? '').split('\n')) {
    if (!line.trim()) continue;
    let row = null;
    try {
      row = asObject(JSON.parse(line));
    } catch {
      // Not JSON: counted with the lines that are not objects.
    }
    if (row) rows.push(row);
    else unreadable += 1;
  }
  return { rows, unreadable };
}

/**
 * What the rows read say in sum: "1 of 2 images classified correctly (50.0%)." Computed here from the
 * rows, over the ones that say whether they were correct; then what was left out, each in a sentence.
 * @param {object[]} rows
 * @param {number} unreadable lines of the file that could not be read
 * @returns {string[]}
 */
export function decisionSummary(rows, unreadable) {
  const judged = rows.filter((row) => typeof row.correct === 'boolean').length;
  const right = rows.filter((row) => row.correct === true).length;
  const open = rows.length - judged;
  const out = [];
  if (!rows.length) out.push('The file holds no decisions.');
  if (judged) out.push(`${fmt.int(right)} of ${fmt.plural(judged, 'image')} classified correctly (${fmt.percent(right, judged)}).`);
  if (open) out.push(`For ${fmt.plural(open, 'image')} the file does not say whether the decision was correct.`);
  if (unreadable > 0) out.push(`${fmt.plural(unreadable, 'line')} could not be read and ${unreadable === 1 ? 'is' : 'are'} left out.`);
  return out;
}

/** A probability as a percentage with one decimal ("75.0%"); '' when not a finite number. */
const percentText = (value) => (isNum(value) ? `${(value * 100).toFixed(1)}%` : '');

function decisionRow(row) {
  const top = Array.isArray(row.top) ? row.top.map(asObject) : [];
  const first = top[0] ?? null;
  const name = (item) => (isText(item?.label) ? item.label : Number.isInteger(item?.id) ? `Class ${item.id}` : '');
  const absent = (text, cls = '') => h('td', { class: `${cls} holdout__absent`.trim() }, text);
  const others = top.slice(1).map((item) => [name(item) || 'Class not reported', percentText(item?.probability)].filter(Boolean).join(' '));
  return h(
    'tr',
    null,
    isText(row.input) ? h('td', null, h('span', { class: 'mono', title: row.input }, baseName(row.input))) : absent('Not reported'),
    isText(row.target) ? h('td', null, row.target) : absent('Not labelled'),
    name(first) ? h('td', null, name(first)) : absent('Not reported'),
    row.correct === true ? h('td', { class: 'holdout__decisionok' }, 'Correct') : row.correct === false ? h('td', { class: 'holdout__decisionwrong' }, 'Wrong') : absent('Not reported'),
    isNum(first?.probability) ? h('td', { class: 'num', title: String(first.probability) }, percentText(first.probability)) : absent('Not reported', 'num'),
    !top.length ? absent('Not reported') : others.length ? h('td', { class: 'holdout__decisionothers' }, others.join(', ')) : absent('None'),
  );
}

/**
 * The filter, the table and its paging for the rows of one file, in file order (the evaluation's order).
 * At most DECISION_PAGE rows are drawn at a time; rows are added to the table, never redrawn, so the
 * control that was used keeps the keyboard.
 * @param {object[]} rows at least one
 * @param {string} labelId the id of the heading that names the table
 * @returns {Node[]}
 */
function decisionsTable(rows, labelId) {
  const wrong = rows.filter((row) => row.correct === false);
  const tbody = h('tbody');
  const wrap = h(
    'div',
    { class: 'holdout__tablewrap', tabindex: '0', role: 'region', 'aria-labelledby': labelId },
    h(
      'table',
      { class: 'table holdout__table holdout__decisiontable' },
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          ['Image', 'True class', 'Predicted class', 'Result'].map((text) => h('th', { scope: 'col' }, text)),
          h('th', { scope: 'col', class: 'num' }, 'Probability'),
          h('th', { scope: 'col' }, 'Other candidates'),
        ),
      ),
      tbody,
    ),
  );
  const more = h('button', { class: 'btn btn--sm', type: 'button' });
  const count = h('p', { class: 'holdout__quiet', role: 'status' });
  let list = rows;
  let shown = 0;
  const draw = () => {
    const next = list.slice(shown, shown + DECISION_PAGE);
    tbody.append(...next.map(decisionRow));
    shown += next.length;
    const left = list.length - shown;
    if (left > 0) more.textContent = `Show ${fmt.int(Math.min(DECISION_PAGE, left))} more`;
    // The button leaves with the last rows: the keyboard goes to the table it filled.
    else if (document.activeElement === more) wrap.focus();
    more.hidden = left <= 0;
    count.textContent = `Showing ${fmt.int(shown)} of ${list === rows ? fmt.int(list.length) : fmt.plural(list.length, 'wrong decision')}.`;
  };
  more.addEventListener('click', draw);
  let filter;
  if (wrong.length) {
    const box = h('input', { type: 'checkbox' });
    box.addEventListener('change', () => {
      list = box.checked ? wrong : rows;
      shown = 0;
      tbody.replaceChildren();
      wrap.scrollTop = 0;
      draw();
    });
    filter = h('label', { class: 'holdout__decisioncheck' }, box, 'Only wrong decisions');
  } else {
    filter = h('p', { class: 'holdout__quiet' }, 'No wrong decision.');
  }
  draw();
  return [filter, wrap, h('div', { class: 'holdout__more' }, more, count)];
}

/**
 * The text of a decisions file, or only its size when it is too large to show: the size the service
 * states (Content-Length) where it states one, in which case the file is not downloaded, else the length
 * of the text read. Rejects with the reason in words.
 * @returns {Promise<{text: string} | {tooLarge: true, bytes: number}>}
 */
async function readDecisions(url, signal) {
  let response;
  try {
    response = await fetch(url, { signal });
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    throw new Error('The trainer service did not answer.');
  }
  if (!response.ok) {
    let message = `The service answered ${`${response.status} ${response.statusText ?? ''}`.trim()}.`;
    try {
      const payload = await response.json();
      if (isText(payload?.error)) message = payload.error;
    } catch {
      // Not JSON: the HTTP status line is the reason.
    }
    throw new Error(message);
  }
  const stated = response.headers.get('Content-Length');
  const bytes = stated === null || stated === '' ? NaN : Number(stated);
  if (isNum(bytes) && bytes > DECISION_MAX_BYTES) {
    Promise.resolve(response.body?.cancel()).catch(() => {});
    return { tooLarge: true, bytes };
  }
  let text;
  try {
    text = await response.text();
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    throw new Error('The download stopped before the end of the file.');
  }
  return text.length > DECISION_MAX_BYTES ? { tooLarge: true, bytes: text.length } : { text };
}

/**
 * "Decisions per image": the files a classifier's evaluation wrote when it was started with 'Each
 * image's predicted class' (`result.classificationResults`, dataset name -> artifact path). One row per
 * dataset in the order of the scores table, with "Show decisions" (the file read as text and drawn as a
 * table under the rows, one dataset at a time) and the download. Without files: one quiet sentence for
 * an evaluation that was asked to keep them or is a classification, nothing (null) for any other task.
 * @param {object} job
 * @param {object} r the result
 * @param {string[]} order dataset names in the order of the scores table
 * @param {Array<{path: string, available?: boolean, url?: string}>} artifacts the files listed so far
 * @param {AbortSignal} signal the report's; reading stops with it
 * @returns {{el: HTMLElement, destroy: () => void} | null}
 */
function decisionsPart(job, r, order, artifacts, signal) {
  const entries = Object.entries(asObject(r.classificationResults) ?? {}).filter(([, path]) => isText(path));
  const asked = job?.params?.saveClassifications === true;
  if (!entries.length && !asked && r.task !== 'image_classification') return null;
  const titleId = uid('holdout-decisions');
  const part = h('section', { class: 'holdout__part', 'aria-labelledby': titleId }, h('h4', { class: 'holdout__title', id: titleId }, 'Decisions per image'));
  if (!entries.length) {
    part.appendChild(
      h(
        'p',
        { class: 'holdout__quiet' },
        asked
          ? "The evaluation was asked to keep each image's decision, but recorded none."
          : "The decisions per image were not kept. Start the evaluation with 'Each image's predicted class' to keep them.",
      ),
    );
    return { el: part, destroy() {} };
  }

  const status = h('p', { class: 'holdout__decisionsummary', role: 'status' });
  const view = h('div', { class: 'holdout__decisions', hidden: true });
  let reading = null; // the controller of the file that is open

  const close = (button) => {
    reading?.abort();
    reading = null;
    view.replaceChildren();
    view.hidden = true;
    // "Hide decisions" is gone with the table: back to the button that opened it.
    button.focus();
  };
  const show = (name, file, button) => {
    reading?.abort();
    const ctrl = new AbortController();
    reading = ctrl;
    const headingId = uid('holdout-decisions-of');
    const heading = h('h5', { class: 'holdout__decisiontitle', id: headingId, tabindex: '-1' }, `Decisions: ${name || 'dataset not reported'}`);
    const body = h('div', { class: 'holdout__decisionbody' });
    const hide = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => close(button) }, 'Hide decisions');
    view.replaceChildren(h('div', { class: 'holdout__decisionhead' }, heading, hide), status, body);
    view.hidden = false;
    const load = (again = false) => {
      body.replaceChildren();
      status.textContent = 'Reading the decisions…';
      // The button that asked ("Show decisions" above, or "Try again", which is replaced by the answer)
      // hands the keyboard to the heading of what it opened.
      heading.focus();
      readDecisions(file.url, ctrl.signal).then(
        (answer) => {
          if (ctrl.signal.aborted) return;
          if (answer.tooLarge) {
            status.textContent = `This file is too large to show here (${bytesText(answer.bytes)}). Download it to read it.`;
            return;
          }
          const { rows, unreadable } = parseDecisions(answer.text);
          status.textContent = decisionSummary(rows, unreadable).join(' ');
          if (rows.length) body.replaceChildren(...decisionsTable(rows, headingId));
        },
        (error) => {
          if (ctrl.signal.aborted) return;
          const retry = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => load(true) }, 'Try again');
          status.textContent = '';
          body.replaceChildren(notice('err', `The decisions could not be read: ${reason(error)}`, retry));
          // Only when the person is still here: focus is never pulled back from somewhere else.
          if (again && document.activeElement === heading) retry.focus();
        },
      );
    };
    load();
  };

  const slashes = (path) => String(path).replace(/\\/g, '/');
  const more = isNum(r.outputCount) && r.outputCount > artifacts.length;
  const rowItem = ([name, path]) => {
    const of = () => (name ? h('span', { class: 'visually-hidden' }, ` of ${name}`) : null);
    const file = artifacts.find((a) => a && isText(a.path) && slashes(a.path) === slashes(path)) ?? null;
    let offer;
    if (file && file.available !== false && isText(file.url)) {
      const button = h('button', { class: 'btn btn--sm', type: 'button' }, 'Show decisions', of());
      button.addEventListener('click', () => show(name, file, button));
      offer = [button, h('a', { class: 'holdout__decisionlink', href: file.url, download: baseName(path) || null }, 'Download (JSON lines)', of())];
    } else {
      // A file that is not among the files listed so far is not known to be gone.
      offer = h('span', { class: 'holdout__quiet' }, !file && more ? "The file is not among the files listed so far. 'Show more files' under Files lists the rest." : 'The file is no longer available.');
    }
    return h('li', { class: 'holdout__classrow' }, h('span', { class: 'holdout__classname' }, name || h('span', { class: 'holdout__absent' }, 'No name')), offer);
  };
  const place = ([name]) => (order.includes(name) ? order.indexOf(name) : order.length);
  part.append(h('ul', { class: 'holdout__classrows', 'aria-labelledby': titleId }, [...entries].sort((a, b) => place(a) - place(b)).map(rowItem)), view);
  signal.addEventListener('abort', () => reading?.abort(), { once: true });

  return {
    el: part,
    destroy() {
      reading?.abort();
      reading = null;
    },
  };
}

// ──────── what the two sections below share ────────

/** The dataset names of the given maps: in the order of the scores table, then the rest as listed. */
function datasetNames(order, ...maps) {
  const names = [];
  for (const map of maps) {
    for (const name of Object.keys(asObject(map) ?? {})) if (!names.includes(name)) names.push(name);
  }
  const place = (name) => (order.includes(name) ? order.indexOf(name) : order.length);
  return names.sort((a, b) => place(a) - place(b));
}

/** The fields of `value` that are not among `known`, or null when there are none. */
function unknownFields(value, known) {
  const out = {};
  for (const [key, field] of Object.entries(asObject(value) ?? {})) if (!known.includes(key)) out[key] = field;
  return Object.keys(out).length ? out : null;
}

/**
 * The download of a dataset's JSON lines file, or the sentence for a file that cannot be offered.
 * @returns {{file: object | null, el: HTMLElement} | null} null when the result names no file
 */
function jsonlOffer(path, artifacts, r, name) {
  if (!isText(path)) return null;
  const same = (a) => a && isText(a.path) && a.path.replace(/\\/g, '/') === path.replace(/\\/g, '/');
  const file = artifacts.find(same) ?? null;
  if (file && file.available !== false && isText(file.url)) {
    const of = name ? h('span', { class: 'visually-hidden' }, ` of ${name}`) : null;
    return { file, el: h('a', { class: 'holdout__decisionlink', href: file.url, download: baseName(path) || null }, 'Download (JSON lines)', of) };
  }
  // A file that is not among the files listed so far is not known to be gone.
  const more = isNum(r.outputCount) && r.outputCount > artifacts.length;
  return { file: null, el: h('span', { class: 'holdout__quiet' }, !file && more ? "The file is not among the files listed so far. 'Show more files' under Files lists the rest." : 'The file is no longer available.') };
}

const setTitle = (name, extra = '') =>
  h('h5', { class: 'holdout__settitle' }, name || h('span', { class: 'holdout__absent' }, 'No name'), extra ? h('span', { class: 'holdout__setcount' }, `, ${extra}`) : null);

// ──────── change between frames ────────

/** Why a change between two frames was not scored (traiNNer/tasks/temporal_results.py). */
const EXCLUSION_WORDS = {
  discontinuousWindow: 'where the input window was not continuous',
  nonconsecutiveCentres: 'where the frames were not consecutive',
  sizeChange: 'where the image size changed',
};
const TEMPORAL_FIELDS = ['frames', 'scoredTransitions', 'excludedTransitions', 'exclusions', 'meanAbsoluteDeltaError', 'largest'];
const TEMPORAL_ROW_FIELDS = ['previous', 'current', 'previousTarget', 'currentTarget', 'meanAbsoluteDeltaError', 'elements'];
/** The scored changes drawn at most; the file holds them all. */
const TEMPORAL_ROW_CAP = 500;

/**
 * What a dataset's summary says: how many changes were scored and their mean error, then what was left
 * out and why. A mean the summary gives as null (no change was scored) is "not defined", never 0.
 * @param {object} summary {scoredTransitions, excludedTransitions, exclusions, meanAbsoluteDeltaError}
 * @returns {string[]}
 */
export function temporalSentences(summary) {
  const s = asObject(summary) ?? {};
  const count = (value) => (Number.isInteger(value) && value >= 0 ? value : null);
  const scored = count(s.scoredTransitions);
  const excluded = count(s.excludedTransitions);
  const mean = s.meanAbsoluteDeltaError;
  const lead = [
    scored === null ? '' : `${fmt.plural(scored, 'frame-to-frame change')} scored${excluded === null ? '' : ` out of ${fmt.int(scored + excluded)}`}`,
    isNum(mean) ? `mean error ${mean === 0 ? '0' : scoreText(mean)} (0 means the output changed exactly as the reference did)` : mean === null ? 'the mean error is not defined' : '',
  ]
    .filter(Boolean)
    .join('; ');
  const out = lead ? [`${lead[0].toUpperCase()}${lead.slice(1)}.`] : [];
  if (excluded) {
    const reasons = Object.entries(asObject(s.exclusions) ?? {})
      .filter(([, n]) => isNum(n) && n > 0)
      .map(([key, n]) => `${fmt.int(n)} ${Object.hasOwn(EXCLUSION_WORDS, key) ? EXCLUSION_WORDS[key] : `recorded as ${key}`}`);
    out.push(`${fmt.int(excluded)} left out${reasons.length ? `: ${reasons.join(', ')}` : ''}.`);
  }
  return out;
}

/** A frame by its file name; the full path, and the reference frame's, in the tooltip. */
const frameName = (path, target) => h('span', { class: 'mono', title: [path, isText(target) ? `Reference: ${target}` : ''].filter(Boolean).join('\n') }, baseName(path));

function largestLine(largest) {
  const row = asObject(largest);
  if (!row) return null;
  const value = row.meanAbsoluteDeltaError;
  const both = isText(row.previous) && isText(row.current);
  if (!isNum(value) && !both) return null;
  const title = [String(value), isNum(row.elements) ? `${fmt.int(row.elements)} values compared` : ''].filter(Boolean).join('\n');
  return h(
    'p',
    { class: 'holdout__line' },
    'Largest error',
    isNum(value) ? [': ', h('span', { title }, scoreText(value))] : null,
    both ? [isNum(value) ? ', from ' : ': from ', frameName(row.previous, row.previousTarget), ' to ', frameName(row.current, row.currentTarget)] : null,
    '.',
  );
}

function temporalTable(rows, labelId) {
  const absent = (cls = '') => h('td', { class: `${cls} holdout__absent`.trim() }, 'Not reported');
  const frame = (path, target) => (isText(path) ? h('td', null, frameName(path, target)) : absent());
  return h(
    'div',
    { class: 'holdout__tablewrap', tabindex: '0', role: 'region', 'aria-labelledby': labelId },
    h(
      'table',
      { class: 'table holdout__table holdout__decisiontable' },
      h(
        'thead',
        null,
        h('tr', null, h('th', { scope: 'col' }, 'From frame'), h('th', { scope: 'col' }, 'To frame'), h('th', { scope: 'col', class: 'num' }, 'Error'), h('th', { scope: 'col', class: 'num' }, 'Values compared')),
      ),
      h(
        'tbody',
        null,
        rows.map((row) =>
          h(
            'tr',
            null,
            frame(row.previous, row.previousTarget),
            frame(row.current, row.currentTarget),
            isNum(row.meanAbsoluteDeltaError) ? h('td', { class: 'num', title: String(row.meanAbsoluteDeltaError) }, scoreText(row.meanAbsoluteDeltaError)) : absent('num'),
            isNum(row.elements) ? h('td', { class: 'num' }, fmt.int(row.elements)) : absent('num'),
          ),
        ),
      ),
    ),
  );
}

/** What was read of a file of scored changes, in a sentence or two. */
export function temporalRowsText(count, unreadable) {
  const out = [];
  if (!count) out.push('The file holds no scored changes.');
  else if (count > TEMPORAL_ROW_CAP) out.push(`Showing the first ${fmt.int(TEMPORAL_ROW_CAP)} of ${fmt.plural(count, 'scored change')}. Download the file for the rest.`);
  else out.push(`${fmt.plural(count, 'scored change')}, in the order of the evaluation.`);
  if (unreadable > 0) out.push(`${fmt.plural(unreadable, 'line')} could not be read and ${unreadable === 1 ? 'is' : 'are'} left out.`);
  return out.join(' ');
}

/**
 * "Show the scored changes": closed until asked; the file is read when it is first opened and drawn
 * under the summary line, which keeps the keyboard (only the part under it is replaced).
 */
function scoredChangesFold(file, name, signal) {
  const labelId = uid('holdout-changes');
  const summary = h('summary', { class: 'holdout__foldsummary', id: labelId }, 'Show the scored changes', name ? h('span', { class: 'visually-hidden' }, ` of ${name}`) : null);
  const status = h('p', { class: 'holdout__quiet', role: 'status' });
  const body = h('div', { class: 'holdout__foldbody' });
  const fold = h('details', { class: 'holdout__fold' }, summary, status, body);
  let asked = false;
  const load = (again = false) => {
    asked = true;
    // "Try again" is replaced by the answer: the keyboard goes to the summary line first.
    if (again) summary.focus();
    body.replaceChildren();
    status.textContent = 'Reading the scored changes…';
    readDecisions(file.url, signal).then(
      (answer) => {
        if (signal.aborted) return;
        if (answer.tooLarge) {
          status.textContent = `This file is too large to show here (${bytesText(answer.bytes)}). Download it to read it.`;
          return;
        }
        const { rows, unreadable } = parseDecisions(answer.text);
        status.textContent = temporalRowsText(rows.length, unreadable);
        if (rows.length) body.replaceChildren(temporalTable(rows.slice(0, TEMPORAL_ROW_CAP), labelId));
      },
      (error) => {
        if (signal.aborted) return;
        const retry = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => load(true) }, 'Try again');
        status.textContent = '';
        body.replaceChildren(notice('err', `The scored changes could not be read: ${reason(error)}`, retry));
        // Only when the person is still here: focus is never pulled back from somewhere else.
        if (again && document.activeElement === summary) retry.focus();
      },
    );
  };
  fold.addEventListener('toggle', () => {
    if (fold.open && !asked) load();
  });
  return fold;
}

/**
 * "Change between frames": what an evaluation started with the temporal errors option kept
 * (`result.temporalSummary`, dataset name -> summary, and `result.temporalResults`, dataset name ->
 * artifact path of the JSON lines, one row per scored change). Null when neither names a dataset.
 * Fields this does not draw are put into `rest`.
 */
function temporalPart(r, order, artifacts, signal, rest) {
  const summaries = asObject(r.temporalSummary) ?? {};
  const results = asObject(r.temporalResults) ?? {};
  const names = datasetNames(order, summaries, results);
  if (!names.length) return null;
  const titleId = uid('holdout-temporal');
  const sets = names.map((name) => {
    const summary = asObject(summaries[name]);
    const left = summary ? unknownFields(summary, TEMPORAL_FIELDS) : summaries[name];
    const largestLeft = unknownFields(summary?.largest, TEMPORAL_ROW_FIELDS);
    if (left || largestLeft) rest.temporalSummary = { ...rest.temporalSummary, [name]: { ...(asObject(left) ?? (left ? { value: left } : {})), ...(largestLeft ? { largest: largestLeft } : {}) } };
    const offer = jsonlOffer(results[name], artifacts, r, name);
    return h(
      'div',
      { class: 'holdout__set' },
      setTitle(name, isNum(summary?.frames) ? fmt.plural(summary.frames, 'frame') : ''),
      temporalSentences(summary).map((text) => h('p', { class: 'holdout__line' }, text)),
      largestLine(summary?.largest),
      offer ? h('div', { class: 'holdout__more' }, offer.el) : null,
      offer?.file ? scoredChangesFold(offer.file, name, signal) : null,
    );
  });
  return h(
    'section',
    { class: 'holdout__part', 'aria-labelledby': titleId },
    h('h4', { class: 'holdout__title', id: titleId }, 'Change between frames'),
    h('p', { class: 'holdout__quiet' }, 'How far the change from one frame to the next differs from the same change in the reference. It does not measure the image error itself: read it together with the scores.'),
    sets,
  );
}

// ──────── operating points ────────

/** The counts of a summary group (traiNNer/tasks/object_detection/operating_points.py), with every scored detection kept. */
const GROUP_COUNTS = [
  ['support', 'Reference boxes'],
  ['tp', 'Correct detections'],
  ['fp', 'False detections'],
  ['fn', 'Missed boxes'],
  ['scoredDetections', 'Detections scored'],
  ['pointCount', 'Distinct confidences'],
];
const GROUP_FIELDS = [...GROUP_COUNTS.map(([key]) => key), 'bestF1', 'scope', 'classId', 'categoryId', 'name'];
const POINT_FIELDS = ['threshold', 'support', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'];
const OPERATING_FIELDS = ['iouThreshold', 'scoreComparison', 'detectorLimits', 'categories', 'global'];
const LIMIT_FIELDS = ['scoreThresholdExclusive', 'nmsIou', 'maxDetectionsPerImage'];

/**
 * The detector's own limits in words ({scoreThresholdExclusive, nmsIou, maxDetectionsPerImage}; a limit
 * the service does not know is null and is named as not known, never given a value).
 * @returns {string[]}
 */
export function detectorLimitSentences(limits) {
  const l = asObject(limits);
  if (!l) return [];
  const known = [];
  const unknown = [];
  const take = (key, words, missing) => {
    if (isNum(l[key])) known.push(words(fmt.num(l[key])));
    else if (l[key] === null) unknown.push(missing);
  };
  take('scoreThresholdExclusive', (n) => `it reports only detections with a confidence above ${n}`, 'the lowest confidence it reports');
  take('nmsIou', (n) => `it removes overlapping boxes of the same class at an IoU of ${n}`, 'the overlap at which it removes boxes of the same class');
  take('maxDetectionsPerImage', (n) => `it keeps at most ${n} detections per image`, 'the most detections it keeps per image');
  const out = [];
  if (known.length) out.push(`The detector's own limits: ${known.join('; ')}. Only the detections it reported are counted here.`);
  if (unknown.length) out.push(`Not known: ${listText(unknown)}.`);
  return out;
}

/** The best F1 point of a group in words, or why there is none; '' when the group does not say. */
function bestPointText(group) {
  if (group.bestF1 === null) {
    if (group.support === 0) return 'Not defined: there is no reference box.';
    if (group.pointCount === 0) return 'Not defined: no detection was scored.';
    return 'Not defined';
  }
  const best = asObject(group.bestF1);
  if (!best || !isNum(best.f1)) return '';
  const parts = [isNum(best.precision) ? `precision ${scoreText(best.precision)}` : '', isNum(best.recall) ? `recall ${scoreText(best.recall)}` : ''].filter(Boolean);
  return `${scoreText(best.f1)}${isNum(best.threshold) ? ` at the confidence threshold ${scoreText(best.threshold)}` : ''}${parts.length ? ` (${parts.join(', ')})` : ''}`;
}

function categoriesTable(categories, labelId) {
  const num = (value, text) => (isNum(value) ? h('td', { class: 'num', title: String(value) }, text(value)) : h('td', { class: 'num holdout__absent' }, value === null ? 'Not defined' : 'Not reported'));
  return h(
    'div',
    { class: 'holdout__tablewrap', tabindex: '0', role: 'region', 'aria-labelledby': labelId },
    h(
      'table',
      { class: 'table holdout__table' },
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', { scope: 'col' }, 'Category'),
          [...GROUP_COUNTS.map(([, words]) => words), 'Best F1', 'Its confidence threshold', 'Its precision', 'Its recall'].map((text) => h('th', { scope: 'col', class: 'num' }, text)),
        ),
      ),
      h(
        'tbody',
        null,
        categories.map((row) => {
          // A group without a best point (null) has no threshold, precision or recall either.
          const best = row.bestF1 === null ? { f1: null, threshold: null, precision: null, recall: null } : (asObject(row.bestF1) ?? {});
          const ids = [isNum(row.categoryId) ? `Category id ${row.categoryId}` : '', isNum(row.classId) ? `class number ${row.classId} of the model` : ''].filter(Boolean).join(', ');
          return h(
            'tr',
            null,
            h('td', { class: 'holdout__rowhead', title: ids || null }, isText(row.name) ? row.name : isNum(row.categoryId) ? `Category ${row.categoryId}` : h('span', { class: 'holdout__absent' }, 'No name')),
            GROUP_COUNTS.map(([key]) => num(row[key], fmt.int)),
            num(best.f1, scoreText),
            num(best.threshold, scoreText),
            num(best.precision, scoreText),
            num(best.recall, scoreText),
          );
        }),
      ),
    ),
  );
}

/**
 * "Operating points": what a detection evaluation started with the operating points option kept
 * (`result.detectionOperatingSummary`, dataset name -> {iouThreshold, scoreComparison, detectorLimits,
 * categories, global}, and `result.detectionOperatingPoints`, dataset name -> artifact path of the JSON
 * lines, one row per distinct confidence). Null when neither names a dataset. Fields this does not draw
 * are put into `rest`.
 */
function operatingPart(r, order, artifacts, rest) {
  const summaries = asObject(r.detectionOperatingSummary) ?? {};
  const results = asObject(r.detectionOperatingPoints) ?? {};
  const names = datasetNames(order, summaries, results);
  if (!names.length) return null;
  const titleId = uid('holdout-operating');
  const sets = names.map((name) => {
    const summary = asObject(summaries[name]) ?? {};
    const pooled = asObject(summary.global);
    const categories = (Array.isArray(summary.categories) ? summary.categories : []).filter((row) => row && typeof row === 'object');
    const atOrAbove = summary.scoreComparison === '>=';
    const left = {
      ...(asObject(summaries[name]) || summaries[name] === undefined ? unknownFields(summary, OPERATING_FIELDS) : { value: summaries[name] }),
      ...('scoreComparison' in summary && !atOrAbove ? { scoreComparison: summary.scoreComparison } : {}),
    };
    const add = (key, value) => {
      if (value) left[key] = value;
    };
    add('detectorLimits', asObject(summary.detectorLimits) ? unknownFields(summary.detectorLimits, LIMIT_FIELDS) : summary.detectorLimits);
    add('global', pooled ? unknownFields({ ...unknownFields(pooled, GROUP_FIELDS), ...(unknownFields(pooled.bestF1, POINT_FIELDS) ? { bestF1: unknownFields(pooled.bestF1, POINT_FIELDS) } : {}) }, []) : summary.global);
    add('categories', Array.isArray(summary.categories) ? (categories.some((row) => unknownFields(row, GROUP_FIELDS) || unknownFields(row.bestF1, POINT_FIELDS)) ? categories.map((row) => ({ ...unknownFields(row, GROUP_FIELDS), ...(unknownFields(row.bestF1, POINT_FIELDS) ? { bestF1: unknownFields(row.bestF1, POINT_FIELDS) } : {}) })) : null) : summary.categories);
    if (Object.keys(left).length) rest.detectionOperatingSummary = { ...rest.detectionOperatingSummary, [name]: left };

    const facts = pooled
      ? [...GROUP_COUNTS.filter(([key]) => isNum(pooled[key])).map(([key, words]) => [words, fmt.int(pooled[key])]), ...(bestPointText(pooled) ? [['Best F1', bestPointText(pooled)]] : [])]
      : [];
    const tableId = uid('holdout-operating-of');
    const offer = jsonlOffer(results[name], artifacts, r, name);
    return h(
      'div',
      { class: 'holdout__set' },
      setTitle(name),
      isNum(summary.iouThreshold)
        ? h('p', { class: 'holdout__line' }, `A detection counts as correct when its box overlaps a reference box of the same category with an IoU of at least ${fmt.num(summary.iouThreshold)}.`, atOrAbove ? ' A confidence threshold keeps the detections with a confidence at or above it.' : '')
        : null,
      facts.length
        ? [
            h('p', { class: 'holdout__quiet' }, 'All categories together, with every scored detection kept:'),
            h('dl', { class: 'holdout__facts' }, facts.map(([term, value]) => [h('dt', { class: 'holdout__term' }, term), h('dd', { class: 'holdout__value' }, value)])),
          ]
        : null,
      categories.length ? [h('p', { class: 'holdout__quiet', id: tableId }, 'By category, with every scored detection kept:'), categoriesTable(categories, tableId)] : null,
      detectorLimitSentences(summary.detectorLimits).map((text) => h('p', { class: 'holdout__quiet' }, text)),
      offer ? h('div', { class: 'holdout__more' }, offer.el, offer.file ? h('span', { class: 'holdout__quiet' }, 'One row per distinct confidence.') : null) : null,
    );
  });
  return h('section', { class: 'holdout__part', 'aria-labelledby': titleId }, h('h4', { class: 'holdout__title', id: titleId }, 'Operating points'), sets);
}

/** What the sections above did not draw, as the job reported it; null when there is nothing. */
function restPart(rest) {
  if (!Object.keys(rest).length) return null;
  return h(
    'details',
    { class: 'holdout__raw' },
    h('summary', { class: 'holdout__rawsummary' }, 'Everything else the job reported'),
    h('pre', { class: 'holdout__json', tabindex: '0' }, JSON.stringify(rest, null, 2)),
  );
}

/**
 * The run a checkpoint belongs to: a run started by this service keeps its files under a folder named
 * by its id (".../runs/<runId>/experiments/<name>/models/..."). Answers with the id only when a folder
 * name of the path is exactly the id of one run of the list; never by name, and '' when two runs match.
 * @param {string} path the checkpoint's path on the trainer machine
 * @param {Array<{id: string}>} list the shared run list (`runs.get().list`)
 * @returns {string}
 */
export function runOfWeights(path, list) {
  if (!isText(path)) return '';
  const folders = new Set(path.split(/[\\/]+/).filter(Boolean));
  const found = new Set();
  for (const run of Array.isArray(list) ? list : []) {
    if (run && isText(run.id) && folders.has(run.id)) found.add(run.id);
  }
  return found.size === 1 ? [...found][0] : '';
}

/**
 * The way to the compare view with this job on the left. When the checkpoint belongs to a run of the
 * shared list, the button offers that run and a link keeps the open choice. It follows the list (which
 * can arrive after the report) by changing the button's words in place, so focus stays where it is.
 * @returns {{el: HTMLElement, destroy: () => void}}
 */
function comparePart(job, weights) {
  const token = `job:${job.id}`;
  let runId = '';
  const button = h('button', { class: 'btn', type: 'button', onclick: () => go(runId ? compareHref(token, runId) : compareHref(token)) });
  const other = h('a', { class: 'holdout__comparelink', href: compareHref(token), hidden: true }, 'Compare with something else');
  const sync = () => {
    runId = runOfWeights(weights, runs.get().list);
    const text = runId ? 'Compare with its run' : 'Compare with…';
    if (button.textContent !== text) button.textContent = text;
    other.hidden = !runId;
  };
  sync();
  return { el: h('div', { class: 'holdout__compare' }, button, other), destroy: runs.subscribe(sync) };
}

function leadLine(r, job) {
  const weights = isText(r.weights) ? r.weights : isText(job?.params?.weights) ? job.params.weights : '';
  const datasets = Array.isArray(r.datasets) ? r.datasets : [];
  const names = datasets.map((row) => (isText(row?.dataset) ? row.dataset : '')).filter(Boolean);
  const how = measuredHowText(r.device, r.precision, { deviceName: r.deviceName, deviceIndex: r.deviceIndex });
  const tiled = tilesText(r.tiles);
  // Said only when the output is not ordinary colour: the scores were then measured on those planes.
  const FORMAT_WORDS = { y: "The model's output is grayscale (the Y plane only).", uv: "The model's output is the two colour planes only (UV)." };
  const format = isText(r.outputPixelFormat) ? FORMAT_WORDS[r.outputPixelFormat.toLowerCase()] ?? '' : '';
  return h(
    'p',
    { class: 'holdout__lead' },
    'Checkpoint ',
    weights ? h('span', { class: 'mono holdout__file', title: weights }, baseName(weights)) : h('span', { class: 'holdout__absent' }, 'not reported'),
    ` on ${holdoutWords(r.splits, job?.params?.split)}`,
    names.length ? `, ${names.length === 1 ? 'dataset' : 'datasets'} ${listText(names)}` : '',
    '.',
    how ? ` ${how}` : '',
    tiled ? ` ${tiled}` : '',
    format ? ` ${format}` : '',
  );
}

/**
 * "In tiles of 20 pixels with an overlap of 6 (tile mode context), put together on the CPU." from the
 * result's `tiles`; only the parts the result states, nothing when it was not tiled.
 */
export function tilesText(tiles) {
  if (!tiles || typeof tiles !== 'object' || !Number.isFinite(tiles.size) || tiles.size <= 0) return '';
  const overlap = Number.isFinite(tiles.overlap) ? ` with an overlap of ${tiles.overlap}` : '';
  const mode = isText(tiles.mode) ? ` (tile mode ${tiles.mode})` : '';
  const stitch = !isText(tiles.stitchDevice) ? '' : `, put together on the ${tiles.stitchDevice.toLowerCase() === 'cpu' ? 'CPU' : tiles.stitchDevice.toLowerCase() === 'cuda' ? 'GPU' : `device ${tiles.stitchDevice}`}`;
  return `In tiles of ${tiles.size} pixels${overlap}${mode}${stitch}.`;
}

/** Saved output images, from GET /api/analysis/images, a page at a time. */
function imagesPart(job, signal) {
  const titleId = uid('holdout-images');
  const status = h('p', { class: 'holdout__quiet', role: 'status' }, 'Listing the saved images…');
  const grid = h('ul', { class: 'holdout__grid', hidden: true, 'aria-labelledby': titleId });
  const more = h('button', { class: 'btn btn--sm', type: 'button', hidden: true }, 'Show more images');
  const moreStatus = h('p', { class: 'holdout__quiet', role: 'status' });
  const part = h('section', { class: 'holdout__part', 'aria-labelledby': titleId }, h('h4', { class: 'holdout__title', id: titleId }, 'Saved images'), status, grid, h('div', { class: 'holdout__more' }, more, moreStatus));
  const saveAsked = job?.params?.saveImages === true;
  let next = null;
  let shown = 0;
  const datasets = new Set();
  const items = [];

  const thumb = (image) => {
    const name = String(image?.image ?? image?.name ?? '');
    const dims = fmt.dims(image?.width, image?.height);
    return h(
      'li',
      { class: 'holdout__thumb' },
      h(
        'a',
        { class: 'holdout__thumblink', href: image.url, target: '_blank', rel: 'noopener', title: [name, dims].filter(Boolean).join(', ') },
        h('img', { class: 'holdout__thumbimg', src: sizedUrl(image.url, THUMB_SIZE), alt: '', loading: 'lazy', decoding: 'async', draggable: 'false' }),
        h('span', { class: 'holdout__thumbname' }, name || baseName(image.path) || 'Image'),
        h('span', { class: 'visually-hidden' }, ' (opens the full image in a new tab)'),
      ),
      h('span', { class: 'holdout__thumbset' }, isText(image.dataset) ? image.dataset : ''),
    );
  };

  const add = (page) => {
    const images = (Array.isArray(page?.images) ? page.images : []).filter((image) => image && isText(image.url));
    for (const image of images) {
      if (isText(image.dataset)) datasets.add(image.dataset);
      const el = thumb(image);
      items.push(el);
      grid.appendChild(el);
    }
    shown += images.length;
    next = page?.next ?? null;
    // The dataset under each name only when the images come from more than one.
    for (const el of items) {
      const set = el.querySelector('.holdout__thumbset');
      if (set) set.hidden = datasets.size < 2;
    }
    grid.hidden = shown === 0;
    more.hidden = next === null || next === undefined;
    if (shown === 0) {
      status.textContent = saveAsked
        ? "No images were recorded, although the evaluation was started with 'Save output images'."
        : "No images were saved. Start the evaluation with 'Save output images' to keep them.";
    } else {
      status.textContent = more.hidden
        ? `${fmt.plural(shown, 'output image')}. Each opens the full image in a new tab.`
        : `The first ${fmt.plural(shown, 'output image')}. Each opens the full image in a new tab.`;
    }
  };

  api.analysisImages(job.id, { limit: IMAGE_PAGE }, { signal }).then(
    (page) => {
      if (!signal.aborted) add(page);
    },
    (error) => {
      if (signal.aborted) return;
      status.textContent = `The saved images could not be listed: ${reason(error)}`;
    },
  );
  more.addEventListener('click', () => {
    if (next === null || next === undefined) return;
    more.disabled = true;
    moreStatus.textContent = '';
    api.analysisImages(job.id, { limit: IMAGE_PAGE, after: next }, { signal }).then(
      (page) => {
        if (signal.aborted) return;
        more.disabled = false;
        add(page);
      },
      (error) => {
        if (signal.aborted) return;
        more.disabled = false;
        moreStatus.textContent = `More images could not be listed: ${reason(error)}`;
      },
    );
  });
  return part;
}

/** Every recorded file as a download link; PNG outputs folded into one group. */
function filesPart(job, artifacts, outputCount, signal) {
  const titleId = uid('holdout-files');
  const loaded = artifacts.filter((a) => a && isText(a.url)).slice();
  const list = h('div', { class: 'holdout__fileswrap' });
  const status = h('p', { class: 'holdout__quiet', role: 'status' });
  const more = h('button', { class: 'btn btn--sm', type: 'button', hidden: true }, 'Show more files');
  const link = (a) =>
    h(
      'li',
      { class: 'holdout__fileitem' },
      h('a', { class: 'mono holdout__filelink', href: a.url, download: baseName(a.path) || null }, String(a.path ?? a.url)),
      evaluationFileText(a.path) ? h('span', { class: 'holdout__filedesc' }, evaluationFileText(a.path)) : null,
    );
  let groupOpen = false;
  const draw = () => {
    const pngs = loaded.filter((a) => PNG_RE.test(String(a.path ?? '')));
    const others = loaded.filter((a) => !PNG_RE.test(String(a.path ?? '')));
    const previous = list.querySelector('details');
    if (previous) groupOpen = previous.open;
    list.replaceChildren(
      loaded.length
        ? h(
            'ul',
            { class: 'holdout__files', 'aria-labelledby': titleId },
            others.map(link),
            pngs.length
              ? h(
                  'li',
                  { class: 'holdout__fileitem' },
                  h(
                    'details',
                    { class: 'holdout__filegroup', open: groupOpen },
                    h('summary', { class: 'holdout__filesummary' }, fmt.plural(pngs.length, 'output image')),
                    h('ul', { class: 'holdout__grouplist' }, pngs.map(link)),
                  ),
                )
              : null,
          )
        : h('p', { class: 'holdout__quiet' }, 'This job saved no files.'),
    );
    const remaining = isNum(outputCount) && outputCount > loaded.length && isText(job?.id);
    more.hidden = !remaining;
    status.textContent = remaining ? `Listing ${fmt.int(loaded.length)} of ${fmt.plural(outputCount, 'file')}.` : '';
  };
  more.addEventListener('click', () => {
    more.disabled = true;
    api.analysisResult(job.id, { offset: loaded.length, limit: FILE_PAGE }, { signal }).then(
      (page) => {
        if (signal.aborted) return;
        more.disabled = false;
        const add = (Array.isArray(page?.artifacts) ? page.artifacts : []).filter((a) => a && isText(a.url));
        loaded.push(...add);
        if (!add.length || page?.next === null || page?.next === undefined) outputCount = loaded.length;
        draw();
      },
      (error) => {
        if (signal.aborted) return;
        more.disabled = false;
        status.textContent = `More files could not be listed: ${reason(error)}`;
      },
    );
  });
  draw();
  return h('section', { class: 'holdout__part', 'aria-labelledby': titleId }, h('h4', { class: 'holdout__title', id: titleId }, 'Files'), list, h('div', { class: 'holdout__more' }, more, status));
}

/**
 * Render a completed evaluation job into `host` (appended).
 * @param {HTMLElement} host
 * @param {{job?: object, result: object, artifacts?: Array<{path: string, url: string}>}} answer
 *   `result` and `artifacts` as GET /api/analysis/result gives them; `job` the job record
 * @returns {() => void} cleanup
 */
export function renderEvaluation(host, { job = null, result = null, artifacts = [] } = {}) {
  const controller = new AbortController();
  let classes = null;
  let compare = null;
  let decisions = null;
  let r = asObject(result);
  if (r && !Array.isArray(r.datasets) && asObject(r.result) && Array.isArray(r.result.datasets)) r = r.result;
  const files = Array.isArray(artifacts) ? artifacts : [];
  const order = r && Array.isArray(r.datasets) ? r.datasets.filter((row) => row && typeof row === 'object').map((row) => row.dataset) : [];
  // What the two sections below do not draw of their summaries; shown as reported at the end.
  const rest = {};
  const temporal = r ? temporalPart(r, order, files, controller.signal, rest) : null;
  const operating = r ? operatingPart(r, order, files, rest) : null;
  let root;
  if (!r || (!Array.isArray(r.datasets) && !temporal && !operating)) {
    root = h('div', { class: 'holdout' }, notice('', 'The job finished without measured scores to show.'));
  } else {
    const datasets = (Array.isArray(r.datasets) ? r.datasets : []).filter((row) => row && typeof row === 'object');
    const titleId = uid('holdout-scores');
    // An evaluation can keep only what the sections below show: no score is then no failure.
    const unscored = datasets.length > 0 && metricNames(datasets).length === 0;
    const images = isText(job?.id) && (!PROVIDER_TASKS.has(r.task) || job?.params?.saveImages === true);
    // What the service keeps of a finished evaluation (the kind of each score, class counts) can be
    // read, and the job compared, only once it has completed with something measured.
    const recorded = isText(job?.id) && job.status === 'completed' && datasets.length > 0;
    if (recorded) {
      classes = classesPart(job, r.task, datasets.map((row) => row.dataset), controller.signal);
      compare = comparePart(job, isText(job.params?.weights) ? job.params.weights : r.weights);
    }
    decisions = decisionsPart(job, r, datasets.map((row) => row.dataset), files, controller.signal);
    root = h(
      'div',
      { class: 'holdout' },
      h('div', { class: 'holdout__head' }, leadLine(r, job), compare ? compare.el : null),
      h(
        'section',
        { class: 'holdout__part', 'aria-labelledby': titleId },
        h('h4', { class: 'holdout__title', id: titleId }, 'Measured scores'),
        datasets.length ? scoresTable(datasets, r.task, titleId) : h('p', { class: 'holdout__quiet' }, 'No dataset was measured.'),
        unscored ? h('p', { class: 'holdout__quiet' }, `The evaluation reported no score for ${datasets.length === 1 ? 'this dataset' : 'these datasets'}.`) : null,
        recorded ? kindsLine(job, datasets, controller.signal) : null,
        nonfiniteNotice(datasets),
        h('p', { class: 'holdout__note' }, "One measurement of this checkpoint. It is not part of any run's validation history."),
      ),
      classes ? classes.el : null,
      decisions ? decisions.el : null,
      temporal,
      operating,
      images ? imagesPart(job, controller.signal) : null,
      filesPart(job, files, r.outputCount, controller.signal),
      restPart(rest),
    );
  }
  host.appendChild(root);
  return () => {
    controller.abort();
    classes?.destroy();
    compare?.destroy();
    decisions?.destroy();
    root.remove();
  };
}

// ════════════════════════ split leakage ════════════════════════

const ROLE_WORDS = { gt: 'Ground truth', lq: 'Low quality', pairedGt: 'Paired ground truth', targetGt: 'Target ground truth' };
const ISSUE_WORDS = {
  missing: 'Missing',
  unreadable: 'Could not be read',
  changed: 'Changed while being read',
  selectionError: 'Could not be listed',
  selectionChanged: 'File list changed',
};
const STATE_WORDS = { missing: 'missing', unreadable: 'could not be read', changed: 'changed while being read' };

const isTrain = (id) => String(id ?? '') === 'train';

/** "the training data", "the validation data", "the validation data val_2". */
function dataWords(id, single = true) {
  const key = String(id ?? '');
  if (key === 'train') return 'the training data';
  if (key.split('_')[0] === 'val') return single && key === 'val' ? 'the validation data' : `the validation data ${key}`;
  return key ? `the dataset ${key}` : 'the data';
}

/** "Text: 3 of 16 files." / "Text: 3 files so far." / "Text." */
function counted(text, done, total, unit) {
  if (isNum(done) && isNum(total)) return `${text}: ${fmt.int(done)} of ${fmt.plural(total, unit)}.`;
  if (isNum(done) && done > 0) return `${text}: ${fmt.plural(done, unit)} so far.`;
  return `${text}.`;
}

/** One plain sentence about a running leakage check ({stage, completed, total, splitId}), or ''. */
export function splitLeakageProgressText(progress) {
  if (!progress || typeof progress !== 'object') return '';
  const { stage, completed: done, total } = progress;
  const data = dataWords(progress.splitId);
  switch (stage) {
    case 'selecting':
      return isNum(total) ? `Listed ${fmt.plural(total, 'sample')} of ${data}.` : counted(`Listing the files of ${data}`, done, null, 'sample');
    case 'verifying-selection':
      return counted(`Checking that the file list of ${data} has not changed`, done, total, 'sample');
    case 'disk-metadata':
      return counted('Reading file sizes', done, total, 'file');
    case 'disk-hashing':
      return counted('Reading the files that share a size with another file, to compare their content', done, total, 'file');
    case 'disk-verifying':
      return counted('Checking that no file changed while it was read', done, total, 'file');
    case 'lmdb-metadata':
      return `Reading record sizes in the LMDB databases${isNum(done) && done > 0 ? `: database ${fmt.int(done)}` : ''}.`;
    case 'lmdb-hashing':
      return `Reading the LMDB records that share a size with another record${isNum(done) && done > 0 ? `: database ${fmt.int(done)}` : ''}.`;
    case 'lmdb-verifying':
      return `Checking that no LMDB database changed while it was read${isNum(done) && done > 0 ? `: database ${fmt.int(done)}` : ''}.`;
    case 'reporting':
      return 'Writing the report.';
    default:
      return '';
  }
}

/**
 * What the verdict covers, per dataset, from each split's counts:
 * "Checked all 8 training files and 6 of 8 validation files." plus one sentence per dataset that was
 * not compared at all.
 */
export function coverageSentences(splits) {
  const list = (Array.isArray(splits) ? splits : []).filter((split) => split && typeof split === 'object');
  const validation = list.filter((split) => !isTrain(split.id));
  const single = validation.length === 1;
  const phrases = [];
  const left = [];
  for (const split of list) {
    const counts = asObject(split.counts);
    const side = isTrain(split.id) ? 'training files' : single ? 'validation files' : `validation files of ${split.id}`;
    if (!counts || !isNum(counts.inputs)) {
      left.push(`${dataWords(split.id, single).replace(/^t/, 'T')} could not be listed, so none of its files were compared.`);
      continue;
    }
    if (split.selectionComplete === false) {
      left.push(`The file list of ${dataWords(split.id, single)} changed during the check, so none of its files were compared.`);
      continue;
    }
    const ready = asObject(counts.inputStates)?.ready;
    const read = isNum(ready) ? ready : 0;
    if (read >= counts.inputs) phrases.push(counts.inputs === 1 ? `the one ${side.replace('files', 'file')}` : `all ${fmt.int(counts.inputs)} ${side}`);
    else phrases.push(`${fmt.int(read)} of ${fmt.int(counts.inputs)} ${side}`);
  }
  return [...(phrases.length ? [`Checked ${listText(phrases)}.`] : []), ...left];
}

/** Why a check is partial, from the counts of files that were not checked and the datasets not compared. */
export function partialReasons(result) {
  const states = asObject(result?.counts?.sourceStates) ?? {};
  const out = [];
  for (const [state, n] of Object.entries(states)) {
    if (state === 'ready' || !isNum(n) || n <= 0) continue;
    const files = fmt.plural(n, 'file');
    if (state === 'missing') out.push(`${files} ${n === 1 ? 'is' : 'are'} missing`);
    else if (state in STATE_WORDS) out.push(`${files} ${STATE_WORDS[state]}`);
    else out.push(`${files} ${n === 1 ? 'has' : 'have'} the status ${state}`);
  }
  return out;
}

function bytesText(bytes) {
  if (!isNum(bytes)) return '';
  if (bytes < 1000) return fmt.plural(bytes, 'byte');
  const units = ['kB', 'MB', 'GB', 'TB'];
  let value = bytes / 1000;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}

function verdictPart(r) {
  const counts = asObject(r.counts) ?? {};
  const overlap = r.hasTrainValidationOverlap;
  const shared = isNum(counts.trainValidationGroups) ? counts.trainValidationGroups : null;
  const complete = r.complete === true;
  const ready = asObject(counts.sourceStates)?.ready;
  const sources = counts.sources;
  const coverage = coverageSentences(r.splits);
  const reasons = partialReasons(r);
  const readText = isNum(ready) ? (isNum(sources) && sources > ready ? `${fmt.int(ready)} of ${fmt.plural(sources, 'file')}` : fmt.plural(ready, 'file')) : 'files';
  const partialLine = complete
    ? null
    : h(
        'p',
        { class: 'holdout__noticeline' },
        `This check is partial${reasons.length ? `: ${listText(reasons)}` : ''}. Identical files among the ones that were not checked would not be found.`,
      );
  const lines = (verdict, extra) =>
    h(
      'div',
      { class: 'holdout__noticebody' },
      h('p', { class: 'holdout__verdict' }, verdict),
      coverage.map((text) => h('p', { class: 'holdout__noticeline' }, text)),
      partialLine,
      extra,
    );

  if (overlap === true || (shared !== null && shared > 0)) {
    const n = shared ?? 0;
    const verdict = n
      ? `${fmt.plural(n, 'file')} ${n === 1 ? 'appears' : 'appear'} in both the training and the validation data${complete ? '.' : `, in the ${readText} that could be read. There may be more.`}`
      : `Files appear in both the training and the validation data${complete ? '.' : `, in the ${readText} that could be read. There may be more.`}`;
    return notice('err', lines(verdict, h('p', { class: 'holdout__noticeline' }, 'A model validated on images it trained on reports scores that are too good.')));
  }
  if (overlap === false && complete) {
    return notice('ok', lines('No identical files were found between training and validation.', null));
  }
  if (overlap === false || overlap === null || overlap === undefined) {
    return notice('warn', lines(`No identical files were found between training and validation in the ${readText} that could be read.`, null));
  }
  return notice('', 'The check did not say whether training and validation share files.');
}

function exampleItem(example, showSplit) {
  const role = ROLE_WORDS[example.role] ?? (isText(example.role) ? example.role : '');
  const lmdb = example.storage === 'lmdb';
  return h(
    'li',
    { class: 'holdout__pathitem', title: isText(example.sampleId) ? `Sample ${example.sampleId}` : null },
    h('span', { class: 'holdout__role' }, [role, showSplit && isText(example.splitId) ? example.splitId : ''].filter(Boolean).join(', ')),
    h('span', { class: 'mono holdout__path' }, String(example.path ?? ''), lmdb && isText(example.key) ? h('span', { class: 'holdout__key' }, ` key ${example.key}`) : null),
  );
}

function sideColumn(title, examples, bySplit, ids, showSplit) {
  const counts = bySplit.filter((item) => ids(item.splitId));
  const total = counts.reduce((sum, item) => (isNum(item.inputs) ? sum + item.inputs : sum), 0);
  const unlisted = counts
    .map((item) => {
      const listed = examples.filter((example) => example.splitId === item.splitId).length;
      return isNum(item.inputs) && item.inputs > listed ? { id: item.splitId, n: item.inputs - listed } : null;
    })
    .filter(Boolean);
  return h(
    'div',
    { class: 'holdout__side' },
    h('h5', { class: 'holdout__sidetitle' }, title, counts.length ? h('span', { class: 'holdout__sidecount' }, `, ${fmt.plural(total, 'file')}`) : null),
    examples.length
      ? h('ul', { class: 'holdout__paths' }, examples.map((example) => exampleItem(example, showSplit)))
      : h('p', { class: 'holdout__quiet' }, counts.length ? 'None listed.' : 'None.'),
    unlisted.map(({ id, n }) => h('p', { class: 'holdout__quiet' }, `${fmt.plural(n, 'more file')}${showSplit ? ` in ${id}` : ''} not listed.`)),
  );
}

function groupItem(group, multiVal) {
  const examples = (Array.isArray(group.examples) ? group.examples : []).filter((example) => example && typeof example === 'object');
  const bySplit = (Array.isArray(group.bySplit) ? group.bySplit : []).filter((item) => item && typeof item === 'object');
  const flags = [
    group.trainValidation === true ? badge('In training and validation', 'err') : null,
    group.crossValidation === true ? badge('In two validation datasets', 'warn') : null,
    group.withinSplit === true ? badge('Repeated within one dataset', 'warn') : null,
    group.sameSampleRoles === true ? badge('Ground truth and low quality are the same file', 'warn') : null,
  ];
  const size = bytesText(group.encodedBytes);
  const copies = isNum(group.inputs) ? fmt.plural(group.inputs, 'place') : '';
  return h(
    'li',
    { class: 'holdout__group' },
    h(
      'div',
      { class: 'holdout__grouphead' },
      flags,
      h(
        'span',
        { class: 'holdout__groupsize', title: [isText(group.sha256) ? `SHA-256 ${group.sha256}` : '', isNum(group.encodedBytes) ? fmt.plural(group.encodedBytes, 'byte') : ''].filter(Boolean).join('\n') || null },
        [size ? `${size} each` : '', copies ? `found in ${copies}` : ''].filter(Boolean).join(', '),
      ),
    ),
    h(
      'div',
      { class: 'holdout__sides' },
      sideColumn('Training', examples.filter((example) => isTrain(example.splitId)), bySplit, isTrain, false),
      sideColumn('Validation', examples.filter((example) => !isTrain(example.splitId)), bySplit, (id) => !isTrain(id), multiVal),
    ),
  );
}

function groupsPart(r) {
  const counts = asObject(r.counts) ?? {};
  const groups = (Array.isArray(r.groups) ? r.groups : []).filter((group) => group && typeof group === 'object');
  const total = counts.groups;
  if (!groups.length && !(isNum(total) && total > 0)) return null;
  const titleId = uid('holdout-groups');
  const multiVal = (Array.isArray(r.splits) ? r.splits : []).filter((split) => split && !isTrain(split.id)).length > 1;
  const limit = r.limits?.groups;
  const kinds = [
    ['trainValidationGroups', 'in both training and validation'],
    ['crossValidationGroups', 'in two validation datasets'],
    ['withinSplitGroups', 'repeated within one dataset'],
    ['sameSampleRoleGroups', 'where the ground truth and the low-quality input of one sample are the same file'],
  ]
    .filter(([key]) => isNum(counts[key]) && counts[key] > 0)
    .map(([key, words]) => `${fmt.int(counts[key])} ${words}`);
  const unlisted = isNum(total) ? total - groups.length : 0;
  const limitText = isNum(limit) ? ` The check lists at most ${fmt.int(limit)} (Examples to list).` : '';
  return h(
    'section',
    { class: 'holdout__part', 'aria-labelledby': titleId },
    h('h4', { class: 'holdout__title', id: titleId }, 'Identical files'),
    isNum(total)
      ? h(
          'p',
          { class: 'holdout__line' },
          `${fmt.plural(total, 'group')} of identical files${kinds.length ? `: ${listText(kinds)}` : ''}.`,
          kinds.length > 1 ? ' A group can be counted more than once.' : '',
          ' Each group is one file content and every place it was found.',
        )
      : null,
    groups.length ? h('ul', { class: 'holdout__groups', 'aria-labelledby': titleId }, groups.map((group) => groupItem(group, multiVal))) : null,
    unlisted > 0 || r.groupsTruncated === true
      ? h(
          'p',
          { class: 'holdout__quiet' },
          groups.length
            ? `${fmt.plural(Math.max(0, unlisted), 'more group')} not listed.${limitText}`
            : `None of the groups are listed.${limit === 0 ? ' The check was started with Examples to list set to 0.' : limitText}`,
        )
      : null,
  );
}

function issuesPart(r) {
  const states = asObject(r.counts?.sourceStates) ?? {};
  const issues = (Array.isArray(r.issues) ? r.issues : []).filter((issue) => issue && typeof issue === 'object');
  const bad = Object.entries(states).filter(([state, n]) => state !== 'ready' && isNum(n) && n > 0);
  if (!issues.length && !bad.length) return null;
  const titleId = uid('holdout-issues');
  const unlisted = isNum(r.issueTotal) ? r.issueTotal - issues.length : 0;
  return h(
    'section',
    { class: 'holdout__part', 'aria-labelledby': titleId },
    h('h4', { class: 'holdout__title', id: titleId }, 'Files and datasets that could not be checked'),
    bad.length
      ? h(
          'ul',
          { class: 'holdout__counts' },
          bad.map(([state, n]) => h('li', { class: 'holdout__count' }, `${fmt.plural(n, 'file')} ${STATE_WORDS[state] ?? `with the status ${state}`}`)),
        )
      : null,
    issues.length
      ? h(
          'ul',
          { class: 'holdout__issues', 'aria-labelledby': titleId },
          issues.map((issue) =>
            h(
              'li',
              { class: 'holdout__issue' },
              badge(ISSUE_WORDS[issue.code] ?? (isText(issue.code) ? issue.code : 'Error'), issue.code === 'changed' || issue.code === 'selectionChanged' ? 'warn' : 'err'),
              h(
                'span',
                { class: 'mono holdout__path' },
                String(issue.path ?? ''),
                isText(issue.key) ? h('span', { class: 'holdout__key' }, ` key ${issue.key}`) : null,
              ),
              h('span', { class: 'holdout__issuemsg' }, String(issue.message ?? '')),
            ),
          ),
        )
      : null,
    unlisted > 0 || r.issuesTruncated === true
      ? h(
          'p',
          { class: 'holdout__quiet' },
          `${fmt.int(Math.max(0, unlisted))} more not listed.${isNum(r.limits?.issues) ? ` The check lists at most ${fmt.int(r.limits.issues)}.` : ''}`,
        )
      : null,
  );
}

function sizesLine(counts) {
  const { sources, hashedSources: hashed, sizeOnlySources: sizeOnly } = counts;
  if (!isNum(sources) || !isNum(hashed) || !isNum(sizeOnly)) return null;
  const parts = [];
  if (hashed > 0) parts.push(`${fmt.int(hashed)} of the ${fmt.plural(sources, 'file')} share their size with another file and were read in full to compare their content`);
  if (sizeOnly > 0) parts.push(`${fmt.int(sizeOnly)} ${sizeOnly === 1 ? 'has a size' : 'have sizes'} no other file has, so ${sizeOnly === 1 ? 'it cannot' : 'they cannot'} be identical to another`);
  return parts.length ? h('p', { class: 'holdout__quiet' }, `${parts.join('; ')}.`.replace(/^./, (c) => c.toUpperCase())) : null;
}

/**
 * Render a completed split leakage check into `host` (appended). `job` is optional (the Data screen).
 * @param {HTMLElement} host
 * @param {{job?: object, result: object, artifacts?: Array}} answer `result` as GET /api/analysis/result gives it
 * @returns {() => void} cleanup
 */
export function renderSplitLeakage(host, { job = null, result = null, artifacts = [] } = {}) {
  void job;
  void artifacts;
  let r = asObject(result);
  if (r && !asObject(r.counts) && asObject(r.result) && asObject(r.result.counts)) r = r.result;
  let root;
  if (!r || !asObject(r.counts) || !Array.isArray(r.splits)) {
    root = h('div', { class: 'holdout' }, notice('', 'The job finished without a leakage report to show.'));
  } else {
    root = h(
      'div',
      { class: 'holdout' },
      verdictPart(r),
      h(
        'p',
        { class: 'holdout__line' },
        'Identical means the same encoded bytes. The same picture saved again or resized is not detected.',
        isText(r.algorithm) && r.algorithm !== 'sha256-encoded' ? ` Method: ${r.algorithm}.` : '',
      ),
      sizesLine(r.counts),
      groupsPart(r),
      issuesPart(r),
    );
  }
  host.appendChild(root);
  return () => root.remove();
}
