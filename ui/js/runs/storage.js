// Disk space for a run's checkpoints: what the saves take now, what the remaining ones need and whether
// that fits on the disk. The producer is studio/checkpoint_storage.py (GET /api/run/checkpoint-storage,
// read-only). Sizes are bytes; an estimate the service could not make is null and reads "Not known".
//
//   mountCheckpointStorage(host, {runId, getRun, request}) -> {update(run), destroy()}
//   storageVerdict(report) -> {kind: 'fits' | 'tight' | 'full' | 'done' | 'unknown', sentence}
//
// Shapes (api.checkpointStorage):
//   report = {runId, policy: {source: 'managedAttempt' | 'experimentYaml' | 'neosrToml' | null, saveEvery,
//     targetStep, keepRegularSets, retention: 'disabled' | 'completeRegularSets' | 'unsupported' |
//     'separateNeoSRPolicies' | 'unknown', format: 'safetensors' | 'pth' | null, keepExports?, keepStates?},
//     observed: {bytes, fileCount, complete, numberedBytes, latestBytes, otherBytes, temporaryBytes,
//     recentCompleteSets: [{step, bytes, onCadence, files: [{role, path, bytes, validationSave?}]}]},
//     projection: {sampleSteps, numberedSaveBytes, atomicTemporaryFileBytes, newNumberedSaveHeadroomBytes,
//     fromStep, fromStepSource, runStatus, scenario: 'completedAttempt' | 'ifContinuedToTarget',
//     remainingRegularSaves, additionalFinalNumberedSaves, remainingNumberedWriteBytes,
//     peakAdditionalNumberedBytes, futureRetainedNumberedBytes, regularRetentionWindowBytes, assumptions},
//     filesystem: {availableBytes (null unless every checkpoint folder is on one disk), volumes:
//     [{directories, availableBytes}], wouldFitOneNewSet, wouldFitProjectedNumberedPeak}, limitations}
//   The forecast is "if the run goes on to its last iteration"; for a run that is not training it says
//   what continuing would need. It takes no credit for deleting files that are already on disk.

import { h, mount, fmt, uid } from '../dom.js';
import { notice } from '../ui.js';
import { api } from '../api.js';

/** While a run is active the report is read at most this often. */
const ACTIVE_GAP_MS = 60000;
const FINISHED = new Set(['completed', 'failed', 'stopped', 'cancelled']);
const NOT_KNOWN = 'Not known';
const UNKNOWN_SENTENCE = 'How much the remaining saves need is not known.';

/** Recipe fields of the files of one save, in plain words. */
const ROLE_WORDS = new Map([
  ['pretrain_network_g', 'Network weights'],
  ['pretrain_network_g_ema', 'EMA weights'],
  ['pretrain_network_d', 'Discriminator weights'],
  ['pretrain_network_ae', 'Autoencoder weights'],
  ['pretrain_network_ae_ema', 'Autoencoder EMA weights'],
  ['resume_state', 'Training state'],
]);
const SOURCE_WORDS = new Map([
  ['managedAttempt', 'Read from the recipe this run was started with.'],
  ['experimentYaml', "Read from the recipe in the run's experiment folder."],
  ['neosrToml', "Read from the NeoSR recipe in the run's experiment folder."],
]);
const FROM_WORDS = new Map([
  ['completionReceipt', "the run's completion record"],
  ['reportedIteration', 'the iteration the run last reported'],
  ['mostRecentlyWrittenCompleteSet', 'the newest complete save on disk'],
]);

// ───────── pure helpers (exported for checks) ─────────

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isCount = (v) => Number.isFinite(v) && v >= 0;
const texts = (list) => (Array.isArray(list) ? list.filter((item) => typeof item === 'string' && item) : []);

/** "6.13 MB" in decimal units with two decimals, as the analysis job page writes sizes; '' when absent. */
export function bytesText(n) {
  if (!isCount(n)) return '';
  const units = [['TB', 1e12], ['GB', 1e9], ['MB', 1e6], ['kB', 1e3]];
  const found = units.find(([, size]) => n >= size);
  return found ? `${(n / found[1]).toFixed(2)} ${found[0]}` : fmt.plural(n, 'byte');
}

/** Saves still to come (regular ones plus one at the last iteration); null when the service does not know. */
export function remainingSaves(report) {
  const p = report?.projection;
  if (!isObject(p) || !isCount(p.remainingRegularSaves) || !isCount(p.additionalFinalNumberedSaves)) return null;
  return p.remainingRegularSaves + p.additionalFinalNumberedSaves;
}

/** What is on disk now, as one sentence; '' when the service did not report it. */
function onDiskSentence(observed) {
  if (!isObject(observed) || !isCount(observed.fileCount)) return '';
  if (observed.fileCount === 0) return 'This run has no checkpoint files on disk.';
  if (!isCount(observed.bytes)) return '';
  const partial = observed.complete === false ? 'at least ' : '';
  return `This run's checkpoints take ${partial}${bytesText(observed.bytes)} in ${fmt.plural(observed.fileCount, 'file')}.`;
}

/** Free space, as one sentence; several disks are said as such, never added up. */
function freeSentence(filesystem) {
  if (!isObject(filesystem)) return '';
  if (isCount(filesystem.availableBytes)) return `${bytesText(filesystem.availableBytes)} is free on that disk.`;
  const disks = Array.isArray(filesystem.volumes) ? filesystem.volumes.length : 0;
  return disks > 1 ? `The checkpoint folders are on ${fmt.int(disks)} disks; the free space of each is listed below.` : '';
}

/**
 * The answer to "will the saves fit?", from the service's own comparison.
 *   done: nothing more is scheduled. fits: the remaining saves fit. tight: the next save fits, the
 *   remaining ones do not. full: the next save does not fit. unknown: the service could not tell.
 * `help` lists facts of the report that explain a shortage. Pure.
 * @param {object|null} report the answer of api.checkpointStorage
 * @returns {{kind: 'fits'|'tight'|'full'|'done'|'unknown', sentence: string, help: string[]}}
 */
export function storageVerdict(report) {
  const r = isObject(report) ? report : {};
  const p = isObject(r.projection) ? r.projection : {};
  const f = isObject(r.filesystem) ? r.filesystem : {};
  const o = isObject(r.observed) ? r.observed : {};
  const policy = isObject(r.policy) ? r.policy : {};
  const join = (...parts) => parts.filter(Boolean).join(' ');
  const count = remainingSaves(r);
  const free = isCount(f.availableBytes) ? bytesText(f.availableBytes) : '';
  // The forecast of a run that is not training says what continuing it would need.
  const resting = FINISHED.has(String(p.runStatus)) || p.runStatus === 'imported';
  const target = isCount(policy.targetStep) ? ` to iteration ${fmt.int(policy.targetStep)}` : '';
  const one = count === 1;

  const help = [];
  if (policy.retention === 'disabled') help.push('The recipe keeps every numbered save (no pruning).');
  else if (policy.retention === 'completeRegularSets' && isCount(policy.keepRegularSets)) help.push(`The recipe keeps the last ${fmt.plural(policy.keepRegularSets, 'regular save')} and removes older ones.`);
  if (isCount(o.numberedBytes) && o.numberedBytes > 0) help.push(`The numbered saves already on disk take ${bytesText(o.numberedBytes)}; the forecast counts on none of them being removed.`);
  if (isCount(o.temporaryBytes) && o.temporaryBytes > 0) help.push(`Temporary files take ${bytesText(o.temporaryBytes)}.`);

  if (f.wouldFitOneNewSet === false && count !== 0) {
    const needs = isCount(p.newNumberedSaveHeadroomBytes) ? `it needs about ${bytesText(p.newNumberedSaveHeadroomBytes)} and ${free} is free` : 'there is too little free space';
    return { kind: 'full', sentence: `The next save does not fit: ${needs}.`, help };
  }
  if (f.wouldFitProjectedNumberedPeak === false && count !== 0) {
    const verb = resting ? 'would need' : 'need';
    const needs = isCount(p.peakAdditionalNumberedBytes) ? `they ${verb} about ${bytesText(p.peakAdditionalNumberedBytes)} at their peak and ${free} is free` : 'there is too little free space';
    return { kind: 'tight', sentence: `Not enough room for the remaining saves${resting ? ` if this run is continued${target}` : ''}: ${needs}.`, help };
  }
  if (count === 0) {
    return { kind: 'done', sentence: join(onDiskSentence(o) || 'No more saves are scheduled for this run.', freeSentence(f)), help: [] };
  }
  if (count === null) {
    return { kind: 'unknown', sentence: join(UNKNOWN_SENTENCE, onDiskSentence(o), freeSentence(f)), help: [] };
  }
  const each = isCount(p.numberedSaveBytes) ? `, about ${bytesText(p.numberedSaveBytes)}${one ? '' : ' each'}` : '';
  const expected = resting
    ? `If this run is continued${target}, ${fmt.plural(count, 'more save')} ${one ? 'is' : 'are'} expected${each}.`
    : `${fmt.plural(count, 'more save')} ${one ? 'is' : 'are'} expected${each}.`;
  if (!isCount(p.peakAdditionalNumberedBytes)) {
    return { kind: 'unknown', sentence: join(expected, `How much ${one ? 'it needs' : 'they need'} is not known.`, freeSentence(f)), help: [] };
  }
  const need = `${one ? 'It' : 'They'} ${resting ? 'would need' : one ? 'needs' : 'need'} about ${bytesText(p.peakAdditionalNumberedBytes)} at ${one ? 'its' : 'their'} peak`;
  if (f.wouldFitProjectedNumberedPeak === true && free) return { kind: 'fits', sentence: `${expected} ${need} and ${free} is free: enough.`, help: [] };
  return { kind: 'unknown', sentence: join(expected, `${need}.`, freeSentence(f) || 'How much is free on the disk is not known.'), help: [] };
}

/** The saving plan of the recipe, as sentences. */
export function planSentences(policy) {
  const p = isObject(policy) ? policy : {};
  const out = [];
  const every = isCount(p.saveEvery) && p.saveEvery > 0 ? `A numbered save every ${p.saveEvery === 1 ? 'iteration' : fmt.plural(p.saveEvery, 'iteration')}` : '';
  const target = isCount(p.targetStep) ? `iteration ${fmt.int(p.targetStep)}` : '';
  if (every) out.push(target ? `${every}, up to ${target}.` : `${every}; the last iteration is not known.`);
  else out.push(target ? `How often the run saves is not known; its last iteration is ${fmt.int(p.targetStep)}.` : 'How often the run saves and its last iteration are not known.');
  if (p.retention === 'disabled') out.push('Every numbered save is kept (no pruning).');
  else if (p.retention === 'completeRegularSets') out.push(isCount(p.keepRegularSets) ? `The last ${fmt.plural(p.keepRegularSets, 'regular save')} ${p.keepRegularSets === 1 ? 'is' : 'are'} kept; older ones are removed.` : 'Only the most recent regular saves are kept; how many is not known.');
  else if (p.retention === 'unsupported') out.push('The recipe asks to keep only the most recent saves, which the trainer does not do for this network, so every save is counted as kept.');
  else if (p.retention === 'separateNeoSRPolicies') {
    const kept = (n) => (isCount(n) ? fmt.int(n) : 'not stated');
    out.push(`A NeoSR recipe keeps exported weights and training states by separate rules (exports kept: ${kept(p.keepExports)}, states kept: ${kept(p.keepStates)}).`);
  } else out.push('Which saves are kept is not known.');
  if (p.format === 'safetensors') out.push('Weights are saved as safetensors files.');
  else if (p.format === 'pth') out.push('Weights are saved as .pth files.');
  else out.push('The file format of the weights is not known.');
  out.push(SOURCE_WORDS.get(p.source) ?? 'No recipe could be read for this run.');
  return out;
}

/** The lines of `limitations` that say why the file list is not complete. */
export function inventoryReasons(limitations) {
  return texts(limitations).filter((line) => /^Could not inspect |^Some entries changed/.test(line));
}

// ───────── drawing ─────────

/** A size with its exact bytes in `title`; "Not known" for an estimate the service could not make. */
function size(n) {
  return isCount(n) ? h('span', { class: 'num', title: fmt.plural(n, 'byte') }, bytesText(n)) : h('span', { class: 'storage__absent' }, NOT_KNOWN);
}
const kv = (rows, label) => h('dl', { class: 'kv storage__facts', 'aria-label': label }, rows.filter(Boolean).map(([name, value]) => [h('dt', null, name), h('dd', null, value)]));
const title = (text) => h('h4', { class: 'storage__subtitle' }, text);
const list = (lines) => h('ul', { class: 'storage__list' }, lines.map((line) => h('li', null, line)));
const fitWord = (value) => (value === true ? 'Yes' : value === false ? 'No' : h('span', { class: 'storage__absent' }, NOT_KNOWN));

function setsTable(sets) {
  const fileLine = (file) => {
    const role = ROLE_WORDS.get(file.role);
    return h(
      'li',
      { class: 'storage__file' },
      role ? h('span', null, role) : h('span', { class: 'mono' }, String(file.role ?? 'File')),
      ', ',
      size(file.bytes),
      typeof file.path === 'string' && file.path ? h('span', { class: 'mono storage__path' }, file.path) : null,
      isObject(file.validationSave) ? h('span', { class: 'storage__aside' }, 'A validation was measured with this file') : null,
    );
  };
  return h(
    'div',
    { class: 'storage__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Recent complete saves' },
    h(
      'table',
      { class: 'table storage__table' },
      h('thead', null, h('tr', null, h('th', { scope: 'col', class: 'num' }, 'Iteration'), h('th', { scope: 'col', class: 'num' }, 'Size'), h('th', { scope: 'col' }, 'Files'))),
      h(
        'tbody',
        null,
        sets.map((set) =>
          h(
            'tr',
            null,
            h('td', { class: 'num' }, isCount(set.step) ? fmt.int(set.step) : NOT_KNOWN, set.onCadence === false ? h('span', { class: 'storage__aside' }, 'Outside the regular schedule') : null),
            h('td', { class: 'num' }, size(set.bytes)),
            h('td', null, h('ul', { class: 'storage__files' }, (Array.isArray(set.files) ? set.files.filter(isObject) : []).map(fileLine))),
          ),
        ),
      ),
    ),
  );
}

/** Everything behind the sentence: plan, files on disk, measured saves, forecast, disks, the service's notes. */
function detailsBody(report) {
  const r = isObject(report) ? report : {};
  const o = isObject(r.observed) ? r.observed : {};
  const p = isObject(r.projection) ? r.projection : {};
  const f = isObject(r.filesystem) ? r.filesystem : {};
  const sets = Array.isArray(o.recentCompleteSets) ? o.recentCompleteSets.filter(isObject) : [];
  const volumes = Array.isArray(f.volumes) ? f.volumes.filter(isObject) : [];
  const assumptions = texts(p.assumptions);
  const limitations = texts(r.limitations);
  const reasons = inventoryReasons(limitations);
  const count = remainingSaves(r);
  const steps = Array.isArray(p.sampleSteps) ? p.sampleSteps.filter(isCount) : [];

  let completeLine = 'The service did not say whether this list is complete.';
  if (o.complete === true) completeLine = 'The list is complete.';
  else if (o.complete === false) completeLine = `The list is not complete. ${reasons.length ? reasons.join(' ') : 'The service gave no reason.'}`;

  let toCome = h('span', { class: 'storage__absent' }, NOT_KNOWN);
  if (count !== null) {
    const last = p.additionalFinalNumberedSaves > 0 ? `, ${fmt.int(p.additionalFinalNumberedSaves)} of them at the last iteration, outside the regular schedule` : '';
    toCome = `${fmt.int(count)}${last}`;
  }
  const from = isCount(p.fromStep) ? `Iteration ${fmt.int(p.fromStep)}${FROM_WORDS.has(p.fromStepSource) ? ` (${FROM_WORDS.get(p.fromStepSource)})` : ''}` : h('span', { class: 'storage__absent' }, NOT_KNOWN);
  const scenario = p.scenario === 'completedAttempt' ? 'The run is complete: nothing more is scheduled.' : p.scenario === 'ifContinuedToTarget' ? 'What is needed if the run goes on to its last iteration.' : null;

  return [
    title('Saving plan'),
    h('p', { class: 'storage__text' }, planSentences(r.policy).join(' ')),

    title('On disk now'),
    kv(
      [
        ['All checkpoint files', isCount(o.bytes) ? [size(o.bytes), isCount(o.fileCount) ? ` in ${fmt.plural(o.fileCount, 'file')}` : null] : size(null)],
        ['Numbered saves', size(o.numberedBytes)],
        ['Latest copies', size(o.latestBytes)],
        ['Other files', size(o.otherBytes)],
        ['Temporary files', size(o.temporaryBytes)],
      ],
      'On disk now',
    ),
    h('p', { class: 'storage__note' }, completeLine),

    title('Recent complete saves'),
    sets.length ? setsTable(sets) : h('p', { class: 'storage__note' }, 'No complete save could be measured, so the size of a save is not known.'),

    title('Forecast'),
    scenario ? h('p', { class: 'storage__note' }, scenario) : null,
    kv(
      [
        ['Counted from', from],
        ['Saves still to come', toCome],
        ['Size of one save', [size(p.numberedSaveBytes), steps.length ? h('span', { class: 'storage__aside' }, `The largest of the saves measured at ${steps.length === 1 ? 'iteration' : 'iterations'} ${steps.map((step) => fmt.int(step)).join(', ')}`) : null]],
        ['Room one new save needs while it is written', size(p.newNumberedSaveHeadroomBytes)],
        isCount(p.atomicTemporaryFileBytes) ? ['Largest single file of a save', size(p.atomicTemporaryFileBytes)] : null,
        ['Still to be written', size(p.remainingNumberedWriteBytes)],
        ['Most extra room needed at once', size(p.peakAdditionalNumberedBytes)],
        ['Kept at the end', size(p.futureRetainedNumberedBytes)],
        isCount(p.regularRetentionWindowBytes) ? ['A full window of kept regular saves', size(p.regularRetentionWindowBytes)] : null,
      ],
      'Forecast',
    ),

    title(volumes.length > 1 ? 'Disks' : 'Disk'),
    volumes.length > 1
      ? [
          h('p', { class: 'storage__note' }, `The checkpoint folders are on ${fmt.int(volumes.length)} disks, so the forecast is not compared with one free-space figure.`),
          h(
            'div',
            { class: 'storage__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Disks' },
            h(
              'table',
              { class: 'table storage__table' },
              h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Folders'), h('th', { scope: 'col', class: 'num' }, 'Free'))),
              h('tbody', null, volumes.map((volume) => h('tr', null, h('td', { class: 'mono' }, texts(volume.directories).join(', ') || NOT_KNOWN), h('td', { class: 'num' }, size(volume.availableBytes))))),
            ),
          ),
        ]
      : null,
    kv(
      [
        volumes.length > 1 ? null : ['Free space', size(f.availableBytes)],
        volumes.length === 1 && texts(volumes[0].directories).length ? ['Folders measured', h('span', { class: 'mono' }, texts(volumes[0].directories).join(', '))] : null,
        ['One new save fits', fitWord(f.wouldFitOneNewSet)],
        ['The remaining saves fit at their peak', fitWord(f.wouldFitProjectedNumberedPeak)],
      ],
      'Disk',
    ),

    title('Assumptions'),
    assumptions.length ? list(assumptions) : h('p', { class: 'storage__note' }, 'The service stated none.'),
    title('What is missing'),
    limitations.length ? list(limitations) : h('p', { class: 'storage__note' }, 'The service reported nothing missing.'),
  ];
}

/**
 * Mount the "Disk space for checkpoints" part of a run. It reads once when mounted and again when
 * `update(run)` brings a changed status or iteration: at most once a minute while the run is active, and
 * never while the page is hidden. A service without this report (404) leaves the part hidden.
 * @param {HTMLElement} host
 * @param {{runId: string, getRun?: () => object|null, request?: (id: string, opts: {signal: AbortSignal}) => Promise<object>}} options
 * @returns {{update: (run: object|null) => void, destroy: () => void}}
 */
export function mountCheckpointStorage(host, { runId, getRun = () => null, request = (id, opts) => api.checkpointStorage(id, opts) } = {}) {
  const titleId = uid('storage-title');
  const lead = h('div', { class: 'storage__lead' }, h('p', { class: 'storage__sentence storage__absent' }, 'Reading the disk space…'));
  const stale = h('p', { class: 'storage__note', hidden: true });
  const body = h('div', { class: 'storage__body' });
  const details = h('details', { class: 'storage__details', hidden: true }, h('summary', null, 'How this was worked out'), body);
  const root = h('section', { class: 'storage', 'aria-labelledby': titleId }, h('h3', { class: 'storage__title', id: titleId }, 'Disk space for checkpoints'), lead, stale, details);
  host.appendChild(root);

  const keyOf = (run) => (run ? `${run.status ?? ''}|${run.iteration ?? ''}` : '');
  let run = getRun() ?? null;
  let lastKey = keyOf(run);
  let lastReadAt = 0;
  let timer = 0;
  let wanted = false;
  let controller = null;
  let destroyed = false;
  let gone = false;
  let shownLead = '';
  let waitingBody = null;

  const drawBody = (report) => {
    waitingBody = null;
    mount(body, detailsBody(report));
  };
  // A background update never rebuilds what holds the keyboard focus: the body is redrawn when focus leaves it.
  const onFocusOut = (event) => {
    if (waitingBody && !body.contains(event.relatedTarget)) drawBody(waitingBody);
  };
  body.addEventListener('focusout', onFocusOut);

  const render = (report) => {
    const verdict = storageVerdict(report);
    const limitations = texts(report?.limitations);
    const signature = JSON.stringify([verdict, verdict.kind === 'unknown' ? limitations : null]);
    if (signature !== shownLead) {
      shownLead = signature;
      const short = verdict.kind === 'full' || verdict.kind === 'tight';
      mount(
        lead,
        short
          ? notice('warn', [h('p', { class: 'storage__sentence' }, verdict.sentence), verdict.help.length ? list(verdict.help) : null])
          : h('p', { class: 'storage__sentence' }, verdict.sentence),
        verdict.kind === 'unknown' && limitations.length ? [h('p', { class: 'storage__note' }, 'What the service says is missing:'), list(limitations)] : null,
      );
    }
    stale.hidden = true;
    details.hidden = false;
    if (body.contains(document.activeElement)) waitingBody = report;
    else drawBody(report);
  };

  const fail = (error) => {
    // An older service has no such report: the part disappears without a word.
    if (error?.status === 404 || /unknown api endpoint/i.test(String(error?.message ?? ''))) {
      gone = true;
      root.hidden = true;
      return;
    }
    const why = `The disk space could not be read: ${error?.message || String(error)}`;
    if (shownLead) {
      mount(stale, `${why} The figures shown are from the last successful reading.`);
      stale.hidden = false;
    } else {
      mount(lead, notice('warn', why, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => read() }, 'Read again')));
    }
  };

  function read() {
    if (destroyed || gone) return;
    controller?.abort();
    const mine = new AbortController();
    controller = mine;
    wanted = false;
    lastReadAt = Date.now();
    let answer;
    try {
      answer = Promise.resolve(request(runId, { signal: mine.signal }));
    } catch (error) {
      answer = Promise.reject(error);
    }
    answer.then(
      (report) => {
        if (!mine.signal.aborted && !destroyed) render(report);
      },
      (error) => {
        if (!mine.signal.aborted && !destroyed && error?.name !== 'AbortError') fail(error);
      },
    );
  }

  /** Read now, or as soon as the page is visible and the minute since the last reading of an active run is over. */
  function due() {
    if (destroyed || gone) return;
    wanted = true;
    if (document.hidden) return;
    const active = Boolean(run) && !FINISHED.has(String(run.status));
    const wait = active && lastReadAt ? lastReadAt + ACTIVE_GAP_MS - Date.now() : 0;
    if (wait > 0) {
      if (!timer) {
        timer = setTimeout(() => {
          timer = 0;
          if (wanted) due();
        }, wait);
      }
      return;
    }
    read();
  }
  const onVisible = () => {
    if (!document.hidden && wanted) due();
  };
  document.addEventListener('visibilitychange', onVisible);
  due();

  return {
    update(next) {
      if (destroyed) return;
      run = next ?? null;
      const key = keyOf(run);
      if (key === lastKey) return;
      lastKey = key;
      due();
    },
    destroy() {
      destroyed = true;
      controller?.abort();
      clearTimeout(timer);
      timer = 0;
      document.removeEventListener('visibilitychange', onVisible);
      body.removeEventListener('focusout', onFocusOut);
      root.remove();
    },
  };
}
