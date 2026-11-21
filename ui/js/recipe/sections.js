// Where each option of a recipe is shown. Data plus pure helpers, no DOM.
// Membership of the degradation groups is derived from the real schema property names by prefix and
// suffix, so a new trainer option with a known prefix lands in its group without an edit here.

import { hasAt, isPlainObject } from './schema.js';

export const SECTIONS = [
  { id: 'basics', title: 'Basics', text: 'What the recipe is called, its task and scale, how many GPUs it uses and how it seeds randomness.' },
  { id: 'speed', title: 'Speed and memory', text: 'Precision, memory layout and compilation settings that trade training speed against memory and exactness.' },
  { id: 'datasets', title: 'Datasets', text: 'The folders of ground truth (high-quality) and low quality images used for training and validation, and how they are cropped and batched.' },
  { id: 'degradations', title: 'Degradations', text: 'How the ground truth images are degraded on the fly to make the low quality training inputs.' },
  { id: 'network', title: 'Network', text: 'The architecture being trained, and the discriminator that judges its output when the recipe uses a GAN loss.' },
  { id: 'pretrained', title: 'Pretrained weights and resume', short: 'Pretrained weights', text: 'Weights to start from, how strictly to load them, and a saved training state to resume.' },
  { id: 'training', title: 'Training', text: 'How long to train and how the weights are updated: optimizers, learning rate schedule, EMA and gradient clipping.' },
  { id: 'losses', title: 'Losses', text: 'What training minimizes. Each loss compares the output with the ground truth and is scaled by its weight.' },
  { id: 'validation', title: 'Validation', text: 'How often the model is checked on the validation images, and which metrics are computed.' },
  { id: 'logging', title: 'Logging and checkpoints', text: 'How often progress is logged and checkpoints are saved, and where logs are sent.' },
  { id: 'other', title: 'Other options', text: 'Options that do not belong to a section above. Everything the recipe sets is shown somewhere.' },
];

export const SECTION_IDS = SECTIONS.map((s) => s.id);
export const sectionById = (id) => SECTIONS.find((s) => s.id === id) ?? SECTIONS[0];

export const BASICS = ['name', 'task', 'task_options', 'scale', 'num_gpu', 'manual_seed', 'deterministic', 'deterministic_warn_only', 'checkpoint_rng', 'input_pixel_format', 'output_pixel_format'];
export const SPEED = ['use_amp', 'amp_bf16', 'use_channels_last', 'fast_matmul', 'use_compile', 'compile_mode', 'compile_discriminator', 'compile_ldl_ema', 'detect_anomaly', 'clip_keep'];
export const NETWORK_KEYS = ['network_g', 'network_d', 'network_g_teacher', 'network_ae'];

/** Root keys the trainer fills in itself; hidden unless a recipe sets them. */
export const INTERNAL = new Set([
  'dist', 'launcher', 'rank', 'world_size', 'auto_resume', 'watch', 'start_iter', 'is_train', 'root_path',
  'switch_iter_per_epoch', 'dist_params', 'contents', 'find_unused_parameters',
]);

export const DATASET_FIRST = ['name', 'type', 'dataroot_gt', 'dataroot_lq', 'lq_size', 'gt_size', 'batch_size_per_gpu', 'accum_iter', 'num_worker_per_gpu', 'use_hflip', 'use_rot'];
export const PATH_FIRST = ['pretrain_network_g', 'param_key_g', 'strict_load_g', 'pretrain_network_d', 'strict_load_d', 'resume_state'];
export const TRAIN_FIRST = ['total_iter', 'warmup_iter', 'ema_decay', 'grad_clip', 'grad_clip_max_norm', 'ema_switch_iter', 'ema_update_after_step', 'ema_power'];
/** train.* objects shown as their own sub-headings with a generic editor */
export const TRAIN_OBJECTS = ['optim_g', 'optim_d', 'scheduler'];
export const isLegacyLossKey = (key) => /_opt$/.test(key);

// Degradation groups in pipeline order. `match` is applied in ASSIGN_ORDER below, first match wins.
const FIRST_PASS = ['thicklines_prob', 'blur_prob', 'resize_prob', 'resize_mode_list', 'resize_mode_prob', 'resize_range', 'gaussian_noise_prob', 'noise_range', 'poisson_scale_range', 'gray_noise_prob'];
const PREFIX_GROUPS = [
  ['ntsc', 'NTSC', 'ntsc_'],
  ['rainbow', 'Rainbow', 'rainbow_'],
  ['ghosting', 'Ghosting', 'ghosting_'],
  ['interlace', 'Interlace', 'interlace_'],
  ['scanline', 'Scanlines', 'scanline_'],
  ['lowpass', 'Lowpass', 'lowpass_'],
  ['overshoot', 'Overshoot', 'overshoot_'],
  ['shift', 'Channel shift', 'shift_'],
  ['subsampling', 'Chroma subsampling', 'subsampling_'],
  ['hfnoise', 'High-frequency noise', 'otf_hf_noise_'],
  ['dithering', 'Dithering', 'dithering_'],
  ['dvd', 'DVD', 'dvd_'],
];
const COMBOS = ['otf_degradation_combos', 'otf_degradation_combo_weights', 'otf_no_combo_weight', 'otf_global_degradations'];
const GROUP_ORDER = ['sharpen', 'first', 'second', 'final', 'ntsc', 'rainbow', 'ghosting', 'interlace', 'scanline', 'lowpass', 'overshoot', 'shift', 'subsampling', 'compress1', 'compress2', 'hfnoise', 'dithering', 'dvd', 'combos', 'queue'];
const GROUP_TITLES = {
  sharpen: 'Sharpen input',
  first: 'First pass',
  second: 'Second pass',
  final: 'Final resize',
  compress1: 'Compression, stage 1',
  compress2: 'Compression, stage 2',
  combos: 'Combinations',
  queue: 'Queue and debug',
  ...Object.fromEntries(PREFIX_GROUPS.map(([id, title]) => [id, title])),
};
export const DEGRADATION_SWITCH = 'high_order_degradation';

/** Which degradation group a root key belongs to, or null. `names` is the set of all root names. */
function degradationGroupOf(key, names) {
  if (key.startsWith('lq_usm')) return 'sharpen';
  if (key.startsWith('compress_')) return /2$/.test(key) ? 'compress2' : 'compress1';
  for (const [id, , prefix] of PREFIX_GROUPS) if (key.startsWith(prefix)) return id;
  if (COMBOS.includes(key)) return 'combos';
  if (key === 'queue_size' || key === 'dataroot_lq_prob' || key.startsWith('high_order_degradations_debug')) return 'queue';
  const base = key.replace(/[23]$/, '');
  // Second pass / final resize: the same names as the first pass with a 2 or 3 suffix.
  if (base !== key && (FIRST_PASS.includes(base) || names.has(base))) return key.endsWith('2') ? 'second' : 'final';
  // First pass: the listed names, plus any unsuffixed name whose 2/3 variant exists.
  if (FIRST_PASS.includes(key) || names.has(`${key}2`) || names.has(`${key}3`)) return 'first';
  return null;
}

/**
 * Lay out the root options of the schema.
 * @param {string[]} rootNames ReduxOptions property names
 * @returns {{sectionOf: Map<string, string>, groups: Array<{id, title, keys: string[], probKey: string|null}>}}
 */
export function buildLayout(rootNames) {
  const names = new Set(rootNames);
  const sectionOf = new Map();
  const groupKeys = new Map(GROUP_ORDER.map((id) => [id, []]));
  for (const key of rootNames) {
    let section = null;
    if (BASICS.includes(key)) section = 'basics';
    else if (SPEED.includes(key)) section = 'speed';
    else if (key === 'datasets') section = 'datasets';
    else if (NETWORK_KEYS.includes(key)) section = 'network';
    else if (key === 'path') section = 'pretrained';
    else if (key === 'train') section = 'training';
    else if (key === 'val') section = 'validation';
    else if (key === 'logger') section = 'logging';
    else if (key === DEGRADATION_SWITCH) section = 'degradations';
    else {
      const group = degradationGroupOf(key, names);
      if (group) {
        section = 'degradations';
        groupKeys.get(group).push(key);
      }
    }
    sectionOf.set(key, section ?? 'other');
  }
  const groups = GROUP_ORDER.map((id) => {
    const keys = groupKeys.get(id);
    // A group's summary shows its probability when it has exactly one <prefix>_prob key.
    // Groups without a prefix (first pass, queue) have several or unrelated probabilities: no summary.
    const prefixed = PREFIX_GROUPS.find(([gid]) => gid === id);
    let probKey = null;
    if (prefixed && keys.includes(`${prefixed[2]}prob`)) probKey = `${prefixed[2]}prob`;
    else if (id === 'compress1' && keys.includes('compress_prob')) probKey = 'compress_prob';
    else if (id === 'compress2' && keys.includes('compress_prob2')) probKey = 'compress_prob2';
    return { id, title: GROUP_TITLES[id], keys, probKey };
  }).filter((g) => g.keys.length);
  return { sectionOf, groups };
}

/**
 * Section of a config path (issue paths, focus requests). Root keys the schema does not know go to
 * "Other options"; train.losses is its own section.
 */
export function sectionForPath(path, layout) {
  if (!path.length) return 'basics';
  const [first, second] = path;
  if (first === 'train' && second === 'losses') return 'losses';
  // task and task_options are shown together in Basics even if a schema places them elsewhere.
  if (first === 'task' || first === 'task_options') return 'basics';
  return layout.sectionOf.get(String(first)) ?? 'other';
}

/**
 * Where an issue belongs: its path, plus the missing key when the message reports a missing
 * required field at the parent (`logger`: "Object missing required field `save_checkpoint_freq`").
 * @returns {{path: Array<string|number>, missing: string|null}}
 */
const MISSING_FIELD = /missing required (?:field|key|option)s?\s*[`'"]?([A-Za-z0-9_]+)/i;
const UNKNOWN_FIELD = /contains unknown field\s*[`'"]([A-Za-z0-9_]+)/i;
// The task check reports task_options errors at `task`, naming the spot as "at `$.class_names`".
const TASK_OPTION_AT = /task_options[\s\S]*\bat `\$\.([^`]+)`/;
export function issueTarget(path, message) {
  const text = String(message ?? '');
  const at = path.length === 1 && path[0] === 'task' ? TASK_OPTION_AT.exec(text) : null;
  const base = at ? ['task_options', ...at[1].split(/[.[\]]/).filter(Boolean).map((p) => (/^\d+$/.test(p) ? Number(p) : p))] : path;
  const m = MISSING_FIELD.exec(text);
  if (m) return { path: [...base, m[1]], missing: m[1] };
  // "Object contains unknown field `neosr_review`" at the parent: the key itself has a row.
  const u = UNKNOWN_FIELD.exec(text);
  return u ? { path: [...base, u[1]], missing: null } : { path: base, missing: null };
}

/** Root keys shown in "Other options": schema keys not placed elsewhere plus set unknown keys. */
export function otherKeys(layout, config) {
  const keys = [];
  for (const [key, section] of layout.sectionOf) {
    if (section !== 'other') continue;
    if (INTERNAL.has(key) && !(config && key in config)) continue;
    keys.push(key);
  }
  for (const key of Object.keys(config ?? {})) if (!layout.sectionOf.has(key)) keys.push(key);
  return keys;
}

/** Order `names`: the `first` list (only names that exist) then the rest in schema order. */
export function orderKeys(names, first) {
  const set = new Set(names);
  const head = first.filter((k) => set.has(k));
  return [...head, ...names.filter((k) => !head.includes(k))];
}

// ───────── helpers the form uses to decide what to build ─────────

/** Shape of a config subtree: keys and value types, not values. Its change means rows change. */
export function shape(v, depth = 0) {
  if (Array.isArray(v)) return 'a';
  if (isPlainObject(v)) {
    if (depth > 3) return 'o';
    return `{${Object.keys(v).sort().map((k) => `${k}:${shape(v[k], depth + 1)}`).join(',')}}`;
  }
  return v === null ? 'n' : (typeof v)[0];
}

/**
 * What decides the set of rows of a section. Sections of plain root options (basics, speed,
 * degradations) always show the same rows, so only values change there.
 */
export function sectionSignature(id, c, layout, hasCatalog) {
  switch (id) {
    case 'datasets':
      return shape(c.datasets);
    case 'network':
      return JSON.stringify([c.network_g?.type, c.network_ae?.type, shape(c.network_g), shape(c.network_d), shape(c.network_g_teacher), shape(c.network_ae), hasCatalog]);
    case 'basics':
      return JSON.stringify([typeof c.task === 'string' ? c.task : null, shape(c.task_options), hasCatalog]);
    case 'pretrained':
      return shape(c.path);
    case 'training':
      return shape(isPlainObject(c.train) ? { ...c.train, losses: null } : c.train);
    case 'losses': {
      const list = c.train?.losses;
      return JSON.stringify([Array.isArray(list) ? list.map((l) => [l?.type, shape(l)]) : null, hasCatalog]);
    }
    case 'validation':
      return shape(c.val);
    case 'logging':
      return shape(c.logger);
    case 'other':
      return JSON.stringify(otherKeys(layout, c).map((k) => [k, shape(c[k])]));
    default:
      return '';
  }
}

export const countText = (n) => (n ? `${n} set` : 'All default');
export const setCount = (base, keys, c) => keys.filter((k) => hasAt(c, [...base, k])).length;

/** Summary text of a degradation group: its probability when it has one, else how many keys are set. */
export function degradationState(g, c, defaultOf) {
  if (g.probKey) {
    const set = hasAt(c, [g.probKey]);
    const v = set ? c[g.probKey] : defaultOf(g.probKey);
    if (typeof v === 'number') {
      const text = v === 0 ? 'Off' : `probability ${v}`;
      return set ? text : `${text} (default)`;
    }
  }
  return countText(setCount([], g.keys, c));
}

/**
 * Everything "Find an option" can find: [{section, base, key, prop}] for the options the sections
 * would show for this config. env: {schema, layout, rootProps, netKind, lossKind}.
 */
export function searchEntries(env, c) {
  const { schema, layout, rootProps, netKind, lossKind } = env;
  const out = [];
  const addProps = (sec, base, prop, skip = () => false) => {
    for (const [k, p] of Object.entries(schema.props(prop))) if (!skip(k)) out.push({ section: sec, base, key: k, prop: p });
  };
  for (const [key, sec] of layout.sectionOf) {
    if (['basics', 'speed', 'degradations'].includes(sec)) out.push({ section: sec, base: [], key, prop: rootProps[key] });
  }
  for (const key of otherKeys(layout, c)) out.push({ section: 'other', base: [], key, prop: rootProps[key] ?? null });
  const dsKind = schema.kind(rootProps.datasets);
  for (const ds of Object.keys(isPlainObject(c.datasets) ? c.datasets : {})) addProps('datasets', ['datasets', ds], dsKind.item);
  const def = schema.variantFor(netKind, c.network_g?.type);
  if (def) addProps('network', ['network_g'], def, (k) => k === 'type');
  for (const key of ['network_d', 'network_g_teacher', 'network_ae']) if (key in c) out.push({ section: 'network', base: [], key, prop: rootProps[key] });
  addProps('pretrained', ['path'], rootProps.path);
  const train = isPlainObject(c.train) ? c.train : {};
  addProps('training', ['train'], rootProps.train, (k) => k === 'losses' || (isLegacyLossKey(k) && !(k in train)));
  const losses = Array.isArray(train.losses) ? train.losses : [];
  losses.forEach((loss, i) => {
    const lossDef = schema.variantFor(lossKind, loss?.type);
    if (lossDef) addProps('losses', ['train', 'losses', i], lossDef, (k) => k === 'type');
  });
  addProps('validation', ['val'], rootProps.val);
  addProps('logging', ['logger'], rootProps.logger);
  return out;
}
