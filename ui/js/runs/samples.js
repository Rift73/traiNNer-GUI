// "Training samples" tab: what the model received. GET /api/run/training-samples lists the captures the
// recipe asked for (logger.training_sample_steps): the first sample fed for each requested update, saved
// after the batch was installed on the model and before the optimizer step. The fields per task come from
// traiNNer/training/sample_capture.py; nothing here is shown that the service does not send.
//
//   mountTrainingSamples(host, {runId, active}) -> {setActive(bool), setShown(bool), destroy()}
//   readSafetensors(buffer) -> {name: {dtype, shape, values}} (exported for checks)

import { h, mount, fmt } from '../dom.js';
import { api } from '../api.js';
import { catalog, folderWords } from '../state/catalog.js';
import { notice, loading, emptyState, roleChip } from '../ui.js';

const POLL_MS = 10000;
const PREVIEW_BOX = 256; // long side of a preview at its whole-number zoom, in CSS pixels
const RESTORATION = new Set(['image_restoration', 'recurrent_restoration']);
const CAPTURE_STAGE = 'afterFeedBeforeOptimize';

// ───────── pure helpers ─────────

/** "3 × 16 × 16"; a tensor with no dimensions is a scalar. */
export function shapeText(shape) {
  if (!Array.isArray(shape)) return '';
  return shape.length ? shape.join(' × ') : 'Scalar';
}

/** Height and width of the preview PNG from the tensor shape ([C,H,W], [T,C,H,W] or a [H,W] mask). */
export function previewSize(shape) {
  if (!Array.isArray(shape) || shape.length < 2) return null;
  const [height, width] = shape.slice(-2);
  return Number.isInteger(height) && Number.isInteger(width) && height > 0 && width > 0 ? { width, height } : null;
}

/** The whole-number zoom that fits the long side into the preview box, never below 1. */
export function previewZoom(size) {
  if (!size) return 1;
  return Math.max(1, Math.floor(PREVIEW_BOX / Math.max(size.width, size.height)));
}

/** What a tensor is, in the app's words for the task: {text, role?} where role is a roleChip role. */
export function tensorRole(task, name) {
  const words = folderWords(task);
  if (RESTORATION.has(task)) {
    if (name === 'input') return { text: words.lq.name, role: 'lq' };
    if (name === 'target') return { text: words.gt.name, role: 'gt' };
  }
  if (task === 'autoencoder') {
    if (name === 'input') return { text: 'Input', role: 'lq' };
    if (name === 'target') return { text: 'Image to reconstruct', role: 'gt' };
  }
  if (task === 'semantic_segmentation') {
    if (name === 'input') return { text: words.lq.short, role: 'lq' };
    if (name === 'mask') return { text: words.gt.short, role: 'gt' };
  }
  if (name === 'input') return { text: 'Image', role: 'lq' };
  const plain = { label: 'Class label', boxes: 'Boxes', labels: 'Box labels', shotIds: 'Shot IDs' };
  return { text: plain[name] ?? name };
}

/** The facts line of one capture, from the fields present only. */
export function factsText(sample, { withAttempt = false } = {}) {
  const parts = [];
  if (Number.isInteger(sample.epoch)) parts.push(`epoch ${fmt.int(sample.epoch)}`);
  if (Number.isInteger(sample.batchInEpoch)) parts.push(`batch ${fmt.int(sample.batchInEpoch)} of the epoch`);
  // sampleIndex is zero-based; it is said one-based, like the batch.
  if (Number.isInteger(sample.sampleIndex)) parts.push(`sample ${fmt.int(sample.sampleIndex + 1)} of the batch`);
  if (typeof sample.task === 'string' && sample.task) parts.push(catalog.taskLabel(sample.task));
  if (withAttempt && typeof sample.attempt === 'string' && sample.attempt) parts.push(`attempt ${sample.attempt.slice(0, 8)}`);
  const text = parts.join(', ');
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : '';
}

const DTYPES = {
  F64: [8, (b) => new Float64Array(b)],
  F32: [4, (b) => new Float32Array(b)],
  I64: [8, (b) => Array.from(new BigInt64Array(b), Number)],
  I32: [4, (b) => new Int32Array(b)],
  I16: [2, (b) => new Int16Array(b)],
  I8: [1, (b) => new Int8Array(b)],
  U8: [1, (b) => new Uint8Array(b)],
  BOOL: [1, (b) => Array.from(new Uint8Array(b), (v) => v !== 0)],
};

/**
 * Read a .safetensors file (8-byte little-endian header size, JSON header, raw little-endian data).
 * Types this reader does not know come back with `values: null`.
 * @param {ArrayBuffer} buffer
 */
export function readSafetensors(buffer) {
  const view = new DataView(buffer);
  const size = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, size)));
  const start = 8 + size;
  const out = {};
  for (const [name, t] of Object.entries(header)) {
    if (name === '__metadata__' || !t || !Array.isArray(t.data_offsets)) continue;
    const [a, b] = t.data_offsets;
    const reader = DTYPES[t.dtype];
    out[name] = { dtype: t.dtype, shape: t.shape, values: reader ? Array.from(reader[1](buffer.slice(start + a, start + b))) : null };
  }
  return out;
}

// ───────── one capture ─────────

function previewFigure(name, tensor, role) {
  const size = previewSize(tensor.shape);
  const zoom = previewZoom(size);
  const img = h('img', {
    class: 'samples__img',
    src: tensor.previewUrl,
    alt: `${role.text} preview${size ? `, ${size.width} × ${size.height} pixels` : ''}`,
    width: size ? size.width * zoom : null,
    height: size ? size.height * zoom : null,
    decoding: 'async',
    draggable: 'false',
  });
  return h(
    'figure',
    { class: 'samples__figure' },
    h('div', { class: 'samples__frame' }, h('a', { class: 'samples__imglink', href: tensor.previewUrl, target: '_blank', rel: 'noopener' }, img, h('span', { class: 'visually-hidden' }, ' (opens the full image in a new tab)'))),
    h('figcaption', { class: 'samples__caption' }, size ? `${size.width} × ${size.height} pixels, shown at ${zoom}×` : `Shown at ${zoom}×`),
  );
}

function tensorBlock(sample, name, tensor) {
  const role = tensorRole(sample.task, name);
  const notes = [];
  // Temporal restoration: the PNG is one frame of the window (sample_capture.py writes previewFrame).
  if (Number.isInteger(tensor.previewFrame) && Array.isArray(tensor.shape) && tensor.shape.length === 4) {
    notes.push(`The preview is frame ${fmt.int(tensor.previewFrame + 1)} of ${fmt.int(tensor.shape[0])}, the labelled centre of the window. The raw tensors keep every frame.`);
  }
  if (name === 'mask' && tensor.previewUrl) {
    notes.push('The mask keeps its class IDs: the grey level of each pixel is its class number, so the preview can look almost black.');
    if (Number.isInteger(sample.ignoreLabel)) notes.push(`Pixels with the value ${sample.ignoreLabel} are ignored in training.`);
  }
  return h(
    'div',
    { class: 'samples__tensor' },
    h(
      'p',
      { class: 'samples__tensor-head' },
      role.role ? roleChip(role.role, role.text) : h('span', { class: 'samples__tensor-name' }, role.text),
      h('span', { class: 'samples__shape' }, [shapeText(tensor.shape), tensor.dtype].filter(Boolean).join(', ')),
    ),
    tensor.previewUrl ? previewFigure(name, tensor, role) : null,
    notes.map((text) => h('p', { class: 'samples__note' }, text)),
  );
}

/** Detection: the boxes and labels kept for training live in the raw tensors; read them on request. */
function boxesBlock(sample) {
  const count = sample.tensors?.boxes?.shape?.[0];
  const slot = h('div', { class: 'samples__boxes', 'aria-live': 'polite' });
  const facts = [];
  if (Number.isInteger(count)) facts.push(`${fmt.plural(count, 'box', 'boxes')} kept for training`);
  if (Number.isInteger(sample.ignoredAnnotations) && sample.ignoredAnnotations > 0) facts.push(`${fmt.plural(sample.ignoredAnnotations, 'annotation')} marked to be ignored left out`);
  const names = Array.isArray(sample.classNames) && sample.classNames.length ? `Categories in the recipe: ${sample.classNames.join(', ')}.` : '';
  let busy = false;
  const button = h(
    'button',
    {
      class: 'btn btn--sm',
      type: 'button',
      onclick: async () => {
        if (busy) return;
        busy = true;
        button.disabled = true;
        mount(slot, loading('Reading the boxes from the raw tensors…'));
        try {
          const res = await fetch(sample.rawUrl);
          if (!res.ok) throw new Error(`The service answered ${res.status}.`);
          const tensors = readSafetensors(await res.arrayBuffer());
          const boxes = tensors.boxes?.values ?? [];
          const labels = tensors.labels?.values ?? [];
          const rows = [];
          for (let i = 0; i * 4 + 3 < boxes.length; i += 1) rows.push([i + 1, labels[i], ...boxes.slice(i * 4, i * 4 + 4)]);
          button.remove();
          mount(
            slot,
            rows.length
              ? h(
                  'div',
                  { class: 'samples__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Boxes kept for training' },
                  h(
                    'table',
                    { class: 'table samples__table' },
                    h('caption', { class: 'samples__caption' }, `Corners as the service records them (${typeof sample.boxFormat === 'string' ? sample.boxFormat : 'format not reported'}).`),
                    h('thead', null, h('tr', null, ['Box', 'Label', 'x1', 'y1', 'x2', 'y2'].map((t) => h('th', { scope: 'col', class: 'num' }, t)))),
                    h('tbody', null, rows.map((r) => h('tr', null, r.map((v) => h('td', { class: 'num' }, typeof v === 'number' ? String(Number(v.toFixed(2))) : String(v ?? '')))))),
                  ),
                )
              : h('p', { class: 'samples__note' }, 'No box was kept for training in this sample.'),
          );
        } catch (error) {
          busy = false;
          button.disabled = false;
          mount(slot, notice('err', `The boxes could not be read: ${error.message}`));
        }
      },
    },
    'Show the boxes',
  );
  return h('div', { class: 'samples__detection' }, facts.length ? h('p', { class: 'samples__note' }, `${facts.join('; ')}.`) : null, names ? h('p', { class: 'samples__note' }, names) : null, sample.rawUrl && count ? button : null, slot);
}

function sampleCard(sample) {
  const facts = h('p', { class: 'samples__facts' });
  const tensors = sample.tensors && typeof sample.tensors === 'object' ? Object.entries(sample.tensors) : [];
  const task = sample.task;
  const shown = tensors.filter(([name]) => !(task === 'object_detection' && (name === 'boxes' || name === 'labels')));
  const sources = [
    typeof sample.lq_path === 'string' && sample.lq_path ? [RESTORATION.has(task) ? 'Low quality file' : 'Input file', sample.lq_path] : null,
    typeof sample.gt_path === 'string' && sample.gt_path ? [RESTORATION.has(task) ? 'Ground truth file' : 'Target file', sample.gt_path] : null,
  ].filter(Boolean);
  const className = Array.isArray(sample.classNames) && Number.isInteger(sample.classId) ? sample.classNames[sample.classId] : undefined;
  const card = h(
    'li',
    { class: 'samples__card', dataset: { sample: String(sample.id ?? '') } },
    h('h4', { class: 'samples__title' }, Number.isInteger(sample.candidateStep) ? `Update ${fmt.int(sample.candidateStep)}` : 'Update not reported'),
    facts,
    sample.stage && sample.stage !== CAPTURE_STAGE ? h('p', { class: 'samples__note' }, `Captured at the stage ${sample.stage}.`) : null,
    task === 'image_classification' && Number.isInteger(sample.classId) ? h('p', { class: 'samples__class' }, 'Target class: ', className ? h('strong', null, className) : null, `${className ? ' ' : ''}(class ID ${sample.classId})`) : null,
    task === 'autoencoder' && sample.reconstructionSource === 'target' ? h('p', { class: 'samples__note' }, 'The autoencoder reconstructs the target image.') : null,
    h('div', { class: 'samples__tensors' }, shown.map(([name, t]) => tensorBlock(sample, name, t && typeof t === 'object' ? t : {}))),
    task === 'object_detection' ? boxesBlock(sample) : null,
    sources.length ? h('dl', { class: 'kv samples__sources' }, sources.map(([label, path]) => [h('dt', null, label), h('dd', { class: 'mono' }, path)])) : null,
    sample.rawUrl ? h('p', { class: 'samples__raw' }, h('a', { href: sample.rawUrl, download: `training-sample-${sample.candidateStep ?? 'capture'}.safetensors` }, 'Download the raw tensors (safetensors)')) : null,
  );
  return {
    card,
    /** Facts are text in place: the card is never rebuilt, so a focused link inside it survives. */
    setFacts(withAttempt) {
      const text = factsText(sample, { withAttempt });
      if (facts.textContent !== text) facts.textContent = text;
      facts.hidden = !text;
    },
  };
}

// ───────── the tab ─────────

/**
 * @param {HTMLElement} host
 * @param {{runId: string, active?: boolean, request?: (id: string, params: object, opts: object) => Promise<object>}} options
 * @returns {{setActive(active: boolean): void, setShown(shown: boolean): void, destroy(): void}}
 */
export function mountTrainingSamples(host, { runId, active = false, request = (id, params, opts) => api.trainingSamples(id, params, opts) } = {}) {
  const lead = h(
    'p',
    { class: 'samples__lead', hidden: true },
    'Each block is the first sample fed to the model for an update the recipe asked to capture. It was saved after the batch was fed and before the optimizer step, so the update it belongs to may later have been rejected.',
  );
  const status = h('div', { class: 'samples__status' });
  const list = h('ol', { class: 'samples__list', 'aria-label': 'Captured training samples' });
  const more = h('button', { class: 'btn btn--sm', type: 'button', hidden: true, onclick: () => loadMore() }, 'Show more');
  const root = h('section', { class: 'samples', 'aria-label': 'Training samples' }, h('h3', { class: 'samples__heading' }, 'What the model received'), lead, status, list, h('div', { class: 'samples__more' }, more));
  host.appendChild(root);

  const cards = new Map(); // sample id -> {card, setFacts}
  const attempts = new Set();
  let nextCursor = null;
  let loaded = false;
  let isActive = Boolean(active);
  let shown = true;
  let timer = 0;
  let ctrl = null;
  let moreCtrl = null;
  let gone = false;

  const isAbort = (error) => error?.name === 'AbortError';

  function emptyView() {
    return emptyState({
      title: 'No training samples',
      text: 'This run captured no training samples. To capture them, set logger.training_sample_steps in the recipe before training, for example [1, 500]: the first sample fed for each of those updates is saved. Capture is off by default because it writes the samples to disk and pauses the trainer briefly at those updates.',
      actions: [h('a', { class: 'btn', href: '#/recipe' }, 'Open the Recipe screen')],
    });
  }

  /** Add captures: `where` 'end' (older, from Show more or the first read) or 'start' (newer, from a poll). */
  function add(samples, where) {
    const fresh = samples.filter((s) => s && typeof s === 'object' && s.id !== undefined && !cards.has(String(s.id)));
    const anchor = where === 'start' ? list.firstChild : null;
    for (const s of fresh) {
      const entry = sampleCard(s);
      cards.set(String(s.id), entry);
      if (typeof s.attempt === 'string' && s.attempt) attempts.add(s.attempt);
      list.insertBefore(entry.card, anchor);
    }
    const withAttempt = attempts.size > 1;
    for (const entry of cards.values()) entry.setFacts(withAttempt);
    lead.hidden = cards.size === 0;
    return fresh.length;
  }

  function showState() {
    if (!loaded) return;
    if (cards.size) {
      if (status.dataset.state !== 'list') {
        status.dataset.state = 'list';
        status.replaceChildren();
      }
    } else if (status.dataset.state !== 'empty') {
      status.dataset.state = 'empty';
      mount(status, emptyView());
    }
    more.hidden = nextCursor === null || nextCursor === undefined;
  }

  async function first() {
    ctrl?.abort();
    const c = new AbortController();
    ctrl = c;
    status.dataset.state = 'loading';
    mount(status, loading('Reading the training samples…'));
    try {
      const page = await request(runId, {}, { signal: c.signal });
      if (gone || c !== ctrl) return;
      loaded = true;
      add(Array.isArray(page?.samples) ? page.samples : [], 'end');
      nextCursor = page?.next ?? null;
      showState();
    } catch (error) {
      if (gone || c !== ctrl || isAbort(error)) return;
      if (error?.status === 404) {
        // No experiment folder (a launch that failed early, or one still starting): nothing was captured.
        loaded = true;
        showState();
      } else {
        status.dataset.state = 'error';
        mount(status, notice('err', `The training samples could not be read: ${error.message}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => first() }, 'Try again')));
      }
    } finally {
      schedule();
    }
  }

  async function loadMore() {
    if (nextCursor === null || nextCursor === undefined || moreCtrl) return;
    const c = new AbortController();
    moreCtrl = c;
    more.disabled = true;
    try {
      const page = await request(runId, { after: nextCursor }, { signal: c.signal });
      if (gone) return;
      add(Array.isArray(page?.samples) ? page.samples : [], 'end');
      nextCursor = page?.next ?? null;
      showState();
    } catch (error) {
      if (gone || isAbort(error)) return;
      status.dataset.state = 'more-error';
      mount(status, notice('err', `More training samples could not be read: ${error.message}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => loadMore() }, 'Try again')));
    } finally {
      if (moreCtrl === c) moreCtrl = null;
      if (!gone) more.disabled = false;
    }
  }

  /** While the run trains: read the newest page and put captures not yet shown above the others. */
  async function poll() {
    timer = 0;
    if (gone || !isActive || !shown || !loaded) return;
    if (typeof document !== 'undefined' && document.hidden) {
      schedule();
      return;
    }
    const c = new AbortController();
    ctrl = c;
    try {
      const page = await request(runId, {}, { signal: c.signal });
      if (gone || c !== ctrl) return;
      const hadNone = cards.size === 0;
      add(Array.isArray(page?.samples) ? page.samples : [], hadNone ? 'end' : 'start');
      if (hadNone) nextCursor = page?.next ?? null;
      showState();
    } catch {
      // A failed poll is tried again at the next tick; what is on screen stays.
    } finally {
      if (ctrl === c) schedule();
    }
  }

  function schedule() {
    clearTimeout(timer);
    timer = 0;
    if (!gone && isActive && shown && loaded) timer = setTimeout(poll, POLL_MS);
  }

  first();

  return {
    setActive(next) {
      const was = isActive;
      isActive = Boolean(next);
      if (isActive !== was) schedule();
    },
    setShown(next) {
      shown = Boolean(next);
      schedule();
    },
    destroy() {
      gone = true;
      clearTimeout(timer);
      ctrl?.abort();
      moreCtrl?.abort();
      root.remove();
    },
  };
}
