// Provenance tab of a run: GET /api/run/provenance, read once when the tab is first opened.
// Per attempt: the launch record the service wrote (recipe fingerprint, declared seed, inputs as seen,
// source fingerprint, environment) and the trainer's effective record (`runtime`). Both shapes come from
// code, so every field is rendered defensively and unknown keys are shown, never dropped.

import { h, mount as mountNodes, fmt } from '../dom.js';
import { api } from '../api.js';
import { badge, emptyState, notice, loading } from '../ui.js';

const LABELS = {
  version: 'Record version',
  recipeSha256: 'Recipe fingerprint',
  declaredSeed: 'Declared seed',
  seed: 'Seed',
  initialStep: 'Initial step',
  device: 'Device',
  type: 'Type',
  name: 'Name',
  computeCapability: 'Compute capability',
  memoryBytes: 'Memory',
  task: 'Task',
  modelClass: 'Model class',
  pytorch: 'PyTorch',
  cudaBuild: 'CUDA build',
  deterministic: 'Deterministic algorithms',
  deterministicWarnOnly: 'Deterministic, warn only',
  float32MatmulPrecision: 'Float32 matmul precision',
  compileRequested: 'Compile requested',
  ampRequested: 'Mixed precision requested',
  ampBf16Requested: 'bfloat16 mixed precision requested',
  worldSize: 'World size',
  rank: 'Rank',
  trainingData: 'Data loader',
  samples: 'Samples',
  microbatchesPerEpoch: 'Micro-batches per epoch',
  batchSize: 'Batch size',
  workers: 'Workers',
  accumulation: 'Gradient accumulation',
  optimizers: 'Optimizer classes',
  schedulers: 'Scheduler classes',
  schedulerBindings: 'Scheduler bindings',
  python: 'Python',
  interpreter: 'Interpreter',
  platform: 'Platform',
  kernel: 'Kernel',
  machine: 'Machine',
  logicalCpus: 'Logical CPUs',
  checkout: 'Checkout',
  state: 'State folder',
  diskFreeBytes: 'Free disk space',
  diskTotalBytes: 'Disk size',
  llvm: 'LLVM compiler',
  environment: 'Tuning variables',
  sha256: 'Fingerprint',
  files: 'Files',
  gitHead: 'Git commit',
  scope: 'Scope',
  path: 'Path',
  size: 'Size',
  mtimeNs: 'Modified',
  error: 'Error',
};

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
/** "someKeyName" -> "Some key name" for keys without a label. */
const humanize = (key) => {
  const words = String(key).replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ').trim().toLowerCase();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : String(key);
};
const labelOf = (key) => LABELS[key] ?? humanize(key);

/** A long hex fingerprint shortened, with the full value in `title`. */
function hash(value) {
  const text = String(value);
  return h('span', { class: 'mono', title: text }, text.length > 16 ? `${text.slice(0, 12)}…` : text);
}

function bytes(n) {
  return Number.isFinite(n) ? `${fmt.int(n)} bytes` : '';
}

function fromNs(ns) {
  const n = typeof ns === 'string' ? Number(ns) : ns;
  if (!Number.isFinite(n)) return '';
  return fmt.dateTime(new Date(n / 1e6).toISOString());
}

const SEPARATORS = '/\\._-:';

/**
 * A path or a class name in the monospaced face. On the phone edition a line may break after each
 * separator, so a narrow column breaks "torch.optim.adamw." before "AdamW" instead of inside a word. The
 * <wbr> marks change the text flow, so the desktop page, which has room, keeps the plain text.
 */
function mono(value) {
  return h('span', { class: 'mono' }, breakable(String(value)));
}

/** The text, with a <wbr> after each separator on the phone edition only (see mono). */
function breakable(text) {
  if (document.documentElement.dataset.edition !== 'mobile') return text;
  const parts = [];
  let from = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (!SEPARATORS.includes(text[i])) continue;
    parts.push(text.slice(from, i + 1), h('wbr'));
    from = i + 1;
  }
  parts.push(text.slice(from));
  return parts;
}

/** Any value as display nodes: primitives as text, arrays as lists, objects as nested key/value lists. */
function valueNode(value, key = '') {
  if (value === null || value === undefined) return h('span', { class: 'muted' }, 'Not reported');
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') {
    if (/Bytes$/.test(key)) return bytes(value);
    return fmt.int(value) && Number.isInteger(value) ? (/seed|Step|rank|size/i.test(key) ? String(value) : fmt.int(value)) : fmt.num(value, 6);
  }
  if (typeof value === 'string') {
    if (/sha256$/i.test(key) || key === 'gitHead') return hash(value);
    if (/class|path|interpreter|checkout|state|llvm|scope/i.test(key) || value.includes('/')) return mono(value);
    // One long word in the text face ("torch.optim.adamw.AdamW" in a list of classes) breaks the same way.
    return /\s/.test(value) ? value : breakable(value);
  }
  if (Array.isArray(value)) {
    if (!value.length) return h('span', { class: 'muted' }, 'None');
    if (value.every((v) => !isObject(v) && !Array.isArray(v))) {
      return key === 'computeCapability' ? value.join('.') : h('ul', { class: 'runs__prov-list' }, value.map((v) => h('li', null, valueNode(v, key))));
    }
    return h('ol', { class: 'runs__prov-list' }, value.map((v) => h('li', null, valueNode(v))));
  }
  if (isObject(value)) return kvList(value);
  return String(value);
}

const LONG = 28;

/**
 * How an entry sits on the phone (css/mobile-screens.css; no desktop stylesheet reads these classes):
 * a group (a nested list) and a long value go under their label at the full width, a number never breaks.
 */
function entryClass(value, key) {
  const long = (v) => typeof v === 'string' && v.length > LONG && !(/sha256$/i.test(key) || key === 'gitHead');
  if (isObject(value)) return Object.keys(value).length ? 'runs__prov-group' : null;
  if (Array.isArray(value)) {
    if (value.some((v) => isObject(v) || Array.isArray(v))) return 'runs__prov-group';
    return value.some(long) ? 'runs__prov-long' : null;
  }
  if (typeof value === 'number') return 'runs__prov-num';
  return long(value) ? 'runs__prov-long' : null;
}

function kvList(obj, skip = []) {
  const keys = Object.keys(obj).filter((k) => !skip.includes(k));
  if (!keys.length) return h('span', { class: 'muted' }, 'None');
  return h(
    'dl',
    { class: 'kv runs__prov-kv' },
    keys.map((k) => {
      const kind = entryClass(obj[k], k);
      return [h('dt', { class: kind }, labelOf(k)), h('dd', { class: kind }, valueNode(obj[k], k))];
    }),
  );
}

function inputsTable(inputs) {
  const entries = Object.entries(inputs);
  if (!entries.length) return h('p', { class: 'muted' }, 'No input paths were recorded.');
  return h(
    'div',
    { class: 'runs__table-wrap' },
    h(
      'table',
      // runs__prov-inputs and data-label: on the phone each row is a record with its figures labelled
      // (css/mobile-screens.css). No desktop stylesheet reads them.
      { class: 'table runs__table runs__prov-inputs' },
      h('thead', null, h('tr', null, ['Option', 'Path', 'Size', 'Modified'].map((t, i) => h('th', { scope: 'col', class: i >= 2 ? 'num' : null }, t)))),
      h(
        'tbody',
        null,
        entries.map(([option, rec]) => {
          const r = isObject(rec) ? rec : { error: String(rec) };
          const path = typeof r.path === 'string' ? mono(r.path) : h('span', { class: 'muted' }, 'Not reported');
          const cells = typeof r.error === 'string' && r.error
            ? [h('td', { colspan: '2', class: 'runs__prov-err' }, r.error)]
            : [h('td', { class: 'num', 'data-label': 'Size' }, bytes(r.size) || 'Not reported'), h('td', { class: 'num', 'data-label': 'Modified' }, fromNs(r.mtimeNs) || 'Not reported')];
          return h('tr', null, h('th', { scope: 'row', class: 'mono runs__metric-name' }, option), h('td', { class: 'runs__prov-path' }, path), cells);
        }),
      ),
    ),
  );
}

function environmentBlock(env) {
  if (!isObject(env)) return valueNode(env);
  const main = {};
  for (const k of ['python', 'platform', 'kernel', 'machine', 'interpreter']) if (env[k] !== undefined) main[k] = env[k];
  const rest = Object.keys(env).filter((k) => !(k in main) && k !== 'packages');
  const packages = isObject(env.packages) ? Object.entries(env.packages) : [];
  return h(
    'div',
    { class: 'stack' },
    Object.keys(main).length ? kvList(main) : null,
    packages.length
      ? h(
          'details',
          { class: 'runs__prov-details' },
          h('summary', null, `Installed packages (${fmt.int(packages.length)})`),
          h('dl', { class: 'kv runs__prov-kv runs__prov-packages' }, packages.map(([name, version]) => [h('dt', { class: 'mono' }, name), h('dd', { class: 'mono' }, String(version))])),
        )
      : null,
    rest.length
      ? h('details', { class: 'runs__prov-details' }, h('summary', null, 'More about the machine'), kvList(Object.fromEntries(rest.map((k) => [k, env[k]]))))
      : null,
  );
}

function section(title, ...body) {
  return h('section', { class: 'runs__prov-section' }, h('h4', { class: 'runs__prov-title' }, title), ...body);
}

function launchBlock(launch) {
  if (!isObject(launch)) return section('Launch record', valueNode(launch));
  const head = {};
  for (const k of ['recipeSha256', 'declaredSeed', 'version']) if (launch[k] !== undefined) head[k] = launch[k];
  const known = new Set(['recipeSha256', 'declaredSeed', 'version', 'inputs', 'source', 'environment']);
  const other = Object.keys(launch).filter((k) => !known.has(k));
  const src = launch.source;
  return h(
    'div',
    { class: 'stack' },
    section('Launch record', h('p', { class: 'runs__hint muted' }, 'Written by this service when the attempt started.'), kvList(head)),
    launch.inputs !== undefined
      ? section(
          'Inputs',
          h('p', { class: 'runs__hint muted' }, 'Size and modification time describe each file as the service saw it at launch. They are not content hashes: a file with the same size and time may still differ.'),
          isObject(launch.inputs) ? inputsTable(launch.inputs) : valueNode(launch.inputs),
        )
      : null,
    src !== undefined
      ? section(
          'Training source fingerprint',
          isObject(src) && typeof src.error === 'string' ? notice('warn', `The source fingerprint was not recorded: ${src.error}`) : valueNode(src),
        )
      : null,
    launch.environment !== undefined ? section('Environment', environmentBlock(launch.environment)) : null,
    other.length ? section('Other launch fields', kvList(Object.fromEntries(other.map((k) => [k, launch[k]])))) : null,
  );
}

function attemptBlock(attempt, index) {
  const a = isObject(attempt) ? attempt : {};
  const label = a.attempt !== undefined && a.attempt !== null ? `Attempt ${a.attempt}` : `Attempt ${index + 1}`;
  const other = Object.keys(a).filter((k) => !['attempt', 'current', 'launch', 'runtime'].includes(k));
  return h(
    'section',
    { class: 'runs__prov-attempt' },
    h('h3', { class: 'runs__section-title runs__prov-head' }, h('span', { class: 'mono' }, label), a.current ? badge('Current attempt', 'ok') : null),
    a.launch !== undefined ? launchBlock(a.launch) : h('p', { class: 'muted' }, 'No launch record was written for this attempt.'),
    a.runtime !== undefined
      ? section('What the trainer used', h('p', { class: 'runs__hint muted' }, "The trainer's own record of its effective settings."), isObject(a.runtime) ? kvList(a.runtime) : valueNode(a.runtime))
      : h('p', { class: 'muted' }, 'The trainer wrote no record of its effective settings for this attempt.'),
    other.length ? kvList(Object.fromEntries(other.map((k) => [k, a[k]]))) : null,
  );
}

/**
 * @param {HTMLElement} host
 * @param {{run: object}} options
 * @returns {{setRun(run: object): void, destroy(): void}}
 */
export function mountProvenance(host, { run }) {
  const id = run.id;
  let destroyed = false;
  let controller = null;
  const body = h('div', { class: 'runs__prov', 'aria-live': 'polite' });
  host.appendChild(body);
  load();

  async function load() {
    controller?.abort();
    const ctrl = new AbortController();
    controller = ctrl;
    mountNodes(body, h('div', { class: 'runs__pad' }, loading('Reading the provenance record…')));
    try {
      const res = await api.provenance(id, { signal: ctrl.signal });
      if (destroyed || ctrl.signal.aborted) return;
      const attempts = Array.isArray(res?.attempts) ? res.attempts : [];
      if (res?.recorded === false || !attempts.length) {
        mountNodes(
          body,
          h(
            'div',
            { class: 'runs__pad' },
            emptyState({
              title: 'No provenance recorded',
              text: 'No provenance was recorded for this run. Runs started by this service record it when they launch; imported runs have none.',
            }),
          ),
        );
        return;
      }
      mountNodes(body, h('div', { class: 'runs__pad stack' }, attempts.map(attemptBlock)));
    } catch (error) {
      if (error?.name === 'AbortError' || destroyed) return;
      mountNodes(
        body,
        h('div', { class: 'runs__pad' }, notice('err', `The provenance of this run could not be read: ${error.message}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: load }, 'Try again'))),
      );
    }
  }

  let attemptId = run.attemptId;
  return {
    // The record is written at launch; read it again only when a new attempt started (a resume).
    setRun(next) {
      if (next?.attemptId !== undefined && next.attemptId !== attemptId) {
        attemptId = next.attemptId;
        load();
      }
    },
    destroy() {
      destroyed = true;
      controller?.abort();
      host.replaceChildren();
    },
  };
}
