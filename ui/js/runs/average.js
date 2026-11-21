// Result of a completed checkpoint averaging job (analysis kind `average-checkpoints`): the weights of
// several saved checkpoints of one run, averaged into one safetensors file. The producer is
// studio/checkpoint_average.py. `job` may be null.
//
//   renderAverageCheckpoints(host, {job, result, artifacts}) -> cleanup
//   averageCheckpointsProgressText(progress, job) -> 'Averaging: 7 of 11 tensors.'
//
// Shapes:
//   result = {schemaVersion, method: 'uniform', runId, role: 'net_g' | …, task, inputs: [{path, step,
//     version}], output, version, tensorCount, tensorBytes, averagedTensors, identicalBufferTensors,
//     outputCount}
//   `inputs[].path` is relative to the run's folder; `output` is the saved file on the trainer machine.
//   `tensorBytes` is the size of the tensor values, not of the saved file (which also has a header).
//   `identicalBufferTensors` counts the tensors that were the same in every checkpoint and were copied.
//   artifacts = [{path: 'averaged.safetensors', absolutePath, available, version, url}]
//   progress = {stage: 'averaging', completedTensors, totalTensors, payloadBytesRead}
//
// The run is read once (api.run) for its name and its recipe: the name becomes the text of the link to
// the run, and the recipe goes with the file into the follow-up jobs.

import { h, fmt } from '../dom.js';
import { notice, toast } from '../ui.js';
import { api } from '../api.js';
import { runHref } from '../actions.js';
import { bytesText } from './convert.js';
import { isObject, isText, baseName, remainder, pathValue, downloadLink } from './split.js';

const P = 'average';
/** At most this many iterations are named in the first sentence. */
const STEPS_MAX = 5;
const USE = "It holds weights only. It can be used wherever a checkpoint is asked for, but it cannot continue the run's training state.";
const USE_DISCRIMINATOR = "It holds the discriminator's weights only. It can be given as a discriminator's starting weights, but it cannot be tried, evaluated or exported on its own, and it cannot continue the run's training state.";
const NO_RECIPE = "The run's recipe could not be read, so the follow-up jobs are not offered here. The file can still be chosen by path in a new job.";

const ROLE_WORDS = new Map([
  ['net_g', 'The network'],
  ['net_g_ema', 'The EMA network'],
  ['net_ae', 'The autoencoder'],
  ['net_ae_ema', 'The EMA autoencoder'],
  ['net_d', 'The discriminator'],
]);
const METHOD_WORDS = new Map([['uniform', 'Each checkpoint counts the same']]);
/** The follow-up jobs, as [button label, analysis kind]. */
const NEXT = [
  ['Try on an image…', 'predict'],
  ['Evaluate…', 'evaluate'],
  ['Export to ONNX…', 'export'],
];

// ───────── pure helpers (exported for checks) ─────────

const isCount = (v) => Number.isFinite(v) && v >= 0;

/** "iterations 1 and 2"; up to five are named, then "and N more". '' for an empty list. */
export function stepsText(steps) {
  const all = Array.isArray(steps) ? steps.filter(isCount) : [];
  if (!all.length) return '';
  const named = all.slice(0, STEPS_MAX).map((step) => fmt.int(step));
  if (all.length === 1) return `iteration ${named[0]}`;
  const more = all.length - named.length;
  if (more > 0) return `iterations ${named.join(', ')} and ${fmt.int(more)} more`;
  return `iterations ${named.slice(0, -1).join(', ')} and ${named[named.length - 1]}`;
}

/**
 * The first sentence. `count` is the number of checkpoints (null when the job did not list them),
 * `steps` their iterations (given only when every checkpoint has one), `savedName` the file's name.
 */
export function leadText({ count = null, steps = [], savedName = '' } = {}) {
  if (!isCount(count) && !savedName) return 'The job did not report what it saved.';
  const named = stepsText(steps);
  const what = isCount(count) ? `The average of ${fmt.plural(count, 'checkpoint')}${named ? ` (${named})` : ''}` : 'The average of the checkpoints';
  return `${what} was saved${savedName ? ` as ${savedName}` : ''}.`;
}

/** The "Tensors" value in parts: [text before the size, the size, text after it]; all '' when nothing was reported. */
export function tensorParts({ tensorCount, tensorBytes, averagedTensors, identicalBufferTensors } = {}) {
  const size = bytesText(tensorBytes);
  const after = [];
  if (isCount(averagedTensors) && averagedTensors !== tensorCount) after.push(`${fmt.int(averagedTensors)} averaged`);
  if (isCount(identicalBufferTensors) && identicalBufferTensors > 0) {
    after.push(`${fmt.int(identicalBufferTensors)} copied unchanged because they were identical in every checkpoint`);
  }
  const head = isCount(tensorCount) ? fmt.int(tensorCount) : '';
  const tail = after.join(', ');
  return [head && size ? `${head}, ` : head, size ? `${size} of weights` : '', tail && (head || size) ? `, ${tail}` : tail];
}

/**
 * One plain sentence for a running job's progress.
 * @param {{stage?: string, completedTensors?: number, totalTensors?: number, payloadBytesRead?: number}|null} progress
 * @param {object|null} [job]
 * @returns {string}
 */
export function averageCheckpointsProgressText(progress, job = null) {
  if (!isObject(progress) || progress.stage !== 'averaging') return '';
  const done = isCount(progress.completedTensors) ? progress.completedTensors : progress.completed;
  const total = isCount(progress.totalTensors) ? progress.totalTensors : progress.total;
  if (isCount(done) && isCount(total)) return `Averaging: ${fmt.int(done)} of ${fmt.plural(total, 'tensor')}.`;
  const read = bytesText(progress.payloadBytesRead);
  return read ? `Averaging: ${read} read.` : 'Averaging the checkpoints.';
}

// ───────── renderer ─────────

/**
 * Draw a checkpoint averaging result into `host`.
 * @param {HTMLElement} host
 * @param {{job?: object|null, result?: object|null, artifacts?: Array<{path: string, absolutePath?: string, available?: boolean, version?: string, url?: string}>}} options
 * @returns {() => void} cleanup
 */
export function renderAverageCheckpoints(host, { job = null, result = null, artifacts = [] } = {}) {
  const r = isObject(result) ? result : {};
  const reported = isObject(result) && Object.keys(result).length > 0;
  const controller = new AbortController();
  const { signal } = controller;

  const files = Array.isArray(artifacts) ? artifacts.filter((a) => isObject(a) && isText(a.path)) : [];
  const artifact = files.find((a) => /\.safetensors$/i.test(a.path)) ?? (files.length === 1 ? files[0] : null);
  const usable = Boolean(artifact) && artifact.available !== false && isText(artifact.url);
  const savedName = artifact ? baseName(artifact.path) : baseName(isText(r.output) ? r.output : '');
  const savedPath = usable && isText(artifact.absolutePath) ? artifact.absolutePath : '';

  // 1. Outcome.
  const inputs = Array.isArray(r.inputs) ? r.inputs.filter(isObject) : null;
  const everyStep = inputs !== null && inputs.length > 0 && inputs.every((input) => isCount(input.step));
  const lead = leadText({ count: inputs ? inputs.length : null, steps: everyStep ? inputs.map((input) => input.step) : [], savedName });

  // 2. Facts; a value the job did not report has no row.
  const facts = [];
  if (isText(r.role)) facts.push(['Network', ROLE_WORDS.get(r.role) ?? h('span', { class: 'mono' }, r.role)]);
  if (isText(r.method)) facts.push(['How', METHOD_WORDS.get(r.method) ?? r.method]);
  const [tensorHead, tensorSize, tensorTail] = tensorParts(r);
  if (tensorHead || tensorSize || tensorTail) {
    facts.push(['Tensors', [tensorHead, tensorSize ? h('span', { title: `${fmt.plural(r.tensorBytes, 'byte')} of tensor values` }, tensorSize) : null, tensorTail]]);
  }
  // Said only for a job that asked for it: the statistics come from one checkpoint and are not averaged.
  const batchNormAsked = r.batchNorm === 'fromLatest';
  if (batchNormAsked) {
    const copied = r.copiedBatchNormTensors;
    const from = isCount(r.batchNormSourceStep) ? `iteration ${fmt.int(r.batchNormSourceStep)}` : 'the latest checkpoint';
    facts.push([
      'Batch-norm statistics',
      isCount(copied) && copied > 0 ? `${fmt.plural(copied, 'tensor')} taken from ${from}, not averaged` : isCount(copied) ? 'None found in this network, so nothing was taken from one checkpoint' : `Taken from ${from}`,
    ]);
  }
  // The link's text becomes the run's name once the run was read; the link itself is never replaced.
  const runLink = isText(r.runId) ? h('a', { class: `${P}__run`, href: runHref(r.runId) }, 'the run') : null;
  if (runLink) facts.push(['From run', runLink]);

  // 3. The checkpoints that were averaged.
  const inputsPart =
    inputs && inputs.length
      ? [
          h('h4', { class: `${P}__title` }, 'Averaged checkpoints'),
          h(
            'div',
            { class: `${P}__table-wrap` },
            h(
              'table',
              { class: `table ${P}__table` },
              h('thead', null, h('tr', null, h('th', { scope: 'col', class: 'num' }, 'Iteration'), h('th', { scope: 'col' }, 'File'))),
              h(
                'tbody',
                null,
                inputs.map((input) =>
                  h(
                    'tr',
                    null,
                    h('td', { class: 'num' }, isCount(input.step) ? fmt.int(input.step) : h('span', { class: `${P}__muted` }, 'Not reported')),
                    h('td', null, isText(input.path) ? h('span', { class: `${P}__path` }, input.path) : h('span', { class: `${P}__muted` }, 'Not reported')),
                  ),
                ),
              ),
            ),
          ),
        ]
      : null;

  // 4 and 5. The file, and the jobs that can use it. The buttons are added once, after the run's recipe
  // was read; until then the place for them is empty.
  const next = usable && savedPath ? h('div', { class: `${P}__next` }) : null;
  const useButton = (label, kind, yaml, runName) =>
    h(
      'button',
      {
        class: 'btn btn--sm btn--quiet',
        type: 'button',
        onclick: async () => {
          try {
            // Loaded on the click: analysis.js loads this module.
            const mod = await import('./analysis.js');
            if (signal.aborted) return;
            await mod.openAnalysisDialog({
              kind,
              yaml,
              weights: savedPath,
              weightsVersion: isText(artifact.version) ? artifact.version : null,
              sourceLabel: runName ? `run ${runName}` : 'the run',
            });
          } catch (error) {
            if (!signal.aborted) toast(`The dialog could not be opened: ${error.message}`, { kind: 'err' });
          }
        },
      },
      label,
    );
  let offered = false;
  const discriminator = r.role === 'net_d';
  const offer = (yaml, runName) => {
    // A discriminator has no output to look at: the three jobs are for a network that makes one.
    if (!next || offered || discriminator) return;
    offered = true;
    next.appendChild(
      yaml
        ? h('div', { class: `${P}__actions`, role: 'group', 'aria-label': 'Use the averaged file' }, NEXT.map(([label, kind]) => useButton(label, kind, yaml, runName)))
        : h('p', { class: `${P}__note`, role: 'status' }, NO_RECIPE),
    );
  };

  const unavailable = [
    artifact ? 'The averaged file is no longer available: the job saved it, but the file cannot be read now.' : 'The job reported no output file.',
    facts.length ? ' The facts above are still what the job reported.' : null,
  ];
  const filePart = usable
    ? [
        h('div', { class: `${P}__actions` }, downloadLink(artifact, `Download ${savedName || 'the file'}`, 'averaged.safetensors', P)),
        savedPath ? h('dl', { class: `kv ${P}__facts` }, h('dt', null, 'Path on the trainer machine'), h('dd', null, pathValue(savedPath, P))) : null,
        h('p', { class: `${P}__note` }, discriminator ? USE_DISCRIMINATOR : USE),
        next,
      ]
    : notice('warn', unavailable);

  // 6. What is left of the answer. The checkpoints stay in it while they hold more than the table shows.
  const shown = [['method'], ['runId'], ['role'], ['tensorCount'], ['tensorBytes'], ['averagedTensors'], ['identicalBufferTensors']];
  if (inputs && inputs.length === r.inputs.length && inputs.every((input) => Object.keys(input).every((key) => key === 'path' || key === 'step'))) shown.push(['inputs']);
  if (savedPath && r.output === savedPath) shown.push(['output']);
  if (batchNormAsked) shown.push(['batchNorm'], ['batchNormSourceStep'], ['copiedBatchNormTensors']);
  const rest = reported ? remainder(result, shown) : null;

  const root = h(
    'div',
    { class: P },
    h('p', { class: `${P}__lead` }, lead),
    facts.length ? h('dl', { class: `kv ${P}__facts` }, facts.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])) : null,
    inputsPart,
    h('h4', { class: `${P}__title` }, 'The averaged file'),
    filePart,
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

  // One read of the run serves the link's text and the follow-up jobs.
  if (runLink) {
    api.run(r.runId, { signal }).then(
      (run) => {
        if (signal.aborted) return;
        const runName = isText(run?.name) ? run.name : '';
        if (runName) runLink.textContent = runName;
        offer(isText(run?.yaml) ? run.yaml : '', runName);
      },
      () => {
        if (!signal.aborted) offer('', '');
      },
    );
  } else {
    offer('', '');
  }

  return () => {
    controller.abort();
    root.remove();
  };
}
