// Options form of the Recipe screen: a view of recipe.get().config laid over the JSON Schema.
// The YAML text stays the single source of truth; every edit goes through recipe.applyEdits(), and
// "Undo removal" puts back the text a removal found (recipe.setYaml).
// Only the open section (or the search results) is built. Background updates refresh values in
// place; rows are rebuilt only when the set of rows changes, and focus/caret are put back after.

import { h, mount, debounce, uid } from '../dom.js';
import { persist } from '../store.js';
import { api } from '../api.js';
import { recipe, parsePath, getAt } from '../state/recipe.js';
import { data } from '../state/data.js';
import { catalog as tasks, folderWords } from '../state/catalog.js';
import { go } from '../actions.js';
import { toast, badge, notice, loading, issueList, confirmDialog, promptDialog, countIssues } from '../ui.js';
import { loadSchema, makeSchema, hasAt, isPlainObject, kindFromValue, variantTags, withServiceOptions } from './schema.js';
import {
  SECTIONS, SECTION_IDS, sectionById, buildLayout, sectionForPath, otherKeys, orderKeys, issueTarget,
  BASICS, SPEED, DATASET_FIRST, PATH_FIRST, TRAIN_FIRST, TRAIN_OBJECTS, NETWORK_KEYS, isLegacyLossKey, DEGRADATION_SWITCH,
  shape, sectionSignature, degradationState, searchEntries, countText, setCount,
} from './sections.js';
import { createField, genericEditor, addOptionRow, pathKey, typeField } from './fields.js';
import { createCombobox, matchChoice, toChoices } from '../components/combobox.js';
import { VIDEO_DATASET } from '../state/video-prep.js';
import { mountVideoPrep } from './video-prep.js';

const SEARCH_LIMIT = 60;
const toList = (v) => (Array.isArray(v) ? v : typeof v === 'string' && v ? [v] : []).filter((x) => typeof x === 'string' && x);
const NOT_LISTED = 'Not listed for this task';
/**
 * The registered discriminators whose id does not say so, each read in the trainer's sources: dunet is the
 * network_d of the shipped GAN templates; lqd and mcd extend it and are handled by class where network_d
 * is built; sed is handled there by name; metagan, metagan2 and metagan3 are the MetaGAN family.
 */
const DISCRIMINATOR_IDS = new Set(['dunet', 'lqd', 'mcd', 'metagan', 'metagan2', 'metagan3', 'sed']);
/** The one dataset type that reads ZIP archives: the trainer refuses ZIP storage on every other. */
const ZIP_DATASET = 'RealESRGANDataset';
/** The check's message for a dataset folder or file that does not exist. */
const MISSING_PATH = /^Path was not found\b/;
const IR_WORDS = folderWords('image_restoration');
const restorationWords = (words) => words.gt.name === IR_WORDS.gt.name && words.lq.name === IR_WORDS.lq.name;
const sentence = (text) => {
  const t = String(text ?? '').trim();
  return t ? `${t.charAt(0).toUpperCase()}${t.slice(1)}${/[.!?]$/.test(t) ? '' : '.'}` : '';
};
/** A folder word for running text: "mask", "image"; abbreviations (GT, LQ) give way to the full name. */
const folderNoun = (side) => (/^[A-Z]{2,}$/.test(side.short) ? side.name : side.short).toLowerCase();

/** Lead sentence of the Datasets section in the words of the recipe's task. */
function datasetsLede(task, fallback, wholeImages = false) {
  const words = folderWords(task);
  // A table of records per split: no folder and no image at all.
  if (words.records) return 'The CSV files of numbers used for training and validation, and how they are batched. Their columns are named in Basics, under the task options.';
  // A folder plus a file per split (detection: images and a COCO annotation file).
  // The folder noun is used as an adjective here: "image folders", not "images folders".
  if (words.file) return `The ${folderNoun(words.lq).replace(/s$/, '')} folders and ${words.file.name.toLowerCase()}s used for training and validation, and how they are batched.`;
  // One folder per split (classification): there is no second folder to name.
  if (words.paired === false) return 'The labelled image folders used for training and validation, and how they are batched.';
  // Tasks that read whole images use no crop sizes, so the sentence does not mention cropping.
  const how = wholeImages ? 'how they are batched' : 'how they are cropped and batched';
  if (restorationWords(words)) return wholeImages ? fallback.replace('how they are cropped and batched', how) : fallback;
  return `The ${folderNoun(words.lq)} and ${folderNoun(words.gt)} folders used for training and validation, and ${how}.`;
}
const REPLACE_LIST_NOTE = 'Adding an item replaces the list and removes comments inside it.';
const NEUFLOW_RDUAL = "NeuFlow v2 weights for RDUAL's frozen motion estimator. Optional: when the generator checkpoint does not include them and this is empty, the first training downloads the official weights once and reuses them. A path set here must exist.";
/** The three starts of an RDUAL training: [case, ...rest], a string in `` is an option name. */
const RDUAL_CASES = [
  ['A complete RDUAL checkpoint:', ' choose it as ', '`pretrain_network_g`', '. It includes the NeuFlow v2 weights, so nothing is downloaded.'],
  ['A DUAL Light checkpoint:', ' choose it as ', '`pretrain_network_g`', ' and turn ', '`strict_load_g`', ' off.'],
  ['From scratch:', ' leave ', '`pretrain_network_g`', ' empty.'],
];
const RDUAL_NOTE = 'In the last two cases the first training downloads the official NeuFlow v2 weights once and reuses them afterwards. To use your own file instead, choose it below: it must exist.';
/** The words for a block that is added by choosing its type, by the catalog list that names its types. */
const TYPED_WORDS = {
  optimizers: { word: 'optimizer', choose: 'Choose an optimizer' },
  schedulers: { word: 'schedule', choose: 'Choose a schedule' },
  metrics: { word: 'metric', choose: 'Choose a metric' },
};
/**
 * The last removal made in the form: {path, before, after}, the recipe text as the removal found it and
 * left it. "Undo removal" is offered while the text is still `after`, so it puts back exactly what was
 * there. Kept outside the form so it lasts while another screen is open.
 */
let lastRemoval = null;

export function mountForm(el) {
  const root = h('div', { class: 'form' });
  el.appendChild(root);

  let disposed = false;
  let schema = null;
  let layout = null;
  let rootProps = {};
  let netKind = null;
  let lossKind = null;
  let catalog = null; // catalog state once loaded: {tasks, architectures, losses}

  let section = persist.get('recipe-section', 'basics');
  if (!SECTION_IDS.includes(section)) section = 'basics';
  let query = '';
  let fields = new Map(); // pathKey -> field (rows of the current build only)
  let updaters = []; // (config) => void, for summaries that follow values
  let builtSig = null;
  let issuesBox = null;
  const openGroups = new Set(); // <details> the person opened, kept across rebuilds
  const groupSlots = new WeakMap(); // <details> -> issue badge slot in its summary
  let lastSeen = {};
  let broken = false;
  let brokenTimer = 0;
  let flashTimer = 0;

  let blocks = new Map(); // pathKey -> block issue target (dataset, loss, object group), current build only

  const config = () => recipe.get().config ?? {};
  const ctx = {
    // floatLiteral: a number as typed in scientific notation (1e-4); the backend keeps that spelling.
    apply: (path, value, floatLiteral) => recipe.applyEdits([typeof floatLiteral === 'string' ? { path, value, floatLiteral } : { path, value }]),
    // Deleting the key (not writing null or the default) is how an option goes back to its default.
    remove: (path) => {
      noteFocusForRemoval(path);
      return removeAt(path).catch((error) => {
        focusAfter = null;
        throw error;
      });
    },
    edit: (edits) => recipe.applyEdits(edits),
    // A new key: the row that asked for it is redrawn, so once the checked recipe arrives focus goes to
    // the row of the key that was added (unless the person has moved on to another control).
    add: (path, value, floatLiteral) => {
      const active = document.activeElement;
      // A new object (a schedule for one optimizer) has no row of its own: its type, else its first option.
      const inside = isPlainObject(value) ? [...('type' in value ? ['type'] : []), ...Object.keys(value).filter((k) => k !== 'type')].map((k) => pathKey([...path, k])) : [];
      if (!active || active === document.body || content.contains(active)) focusAfter = { config: recipe.get().config, keys: [pathKey(path), ...inside], at: Date.now() };
      return recipe.applyEdits([typeof floatLiteral === 'string' ? { path, value, floatLiteral } : { path, value }]).catch((error) => {
        focusAfter = null;
        throw error;
      });
    },
  };

  // ───────── focus after a removal ─────────
  // The button that removed a key hides or goes away with its row. Once the checked recipe arrives,
  // focus goes to the same row's control, else the next row's, else the section heading.

  // {config, keys: pathKeys in order of preference, at, marks?: `data-mark` values of buttons tried before
  // the keys, from?: the button that asked (focus still on it counts as "has not moved on"),
  // under?: a pathKey whose first row is tried after the keys (a block that came back)}
  let focusAfter = null;

  // ───────── undo of a removal ─────────
  // A removal can be undone while nothing else has changed the recipe: the text goes back to what the
  // removal found, comments and layout included. That is a different thing from adding: a block that is
  // added is new, with the type the person chooses and no options set (addTyped).

  let lastChange = null; // {before, after} of the latest change of the recipe text, whoever made it

  /** Remove `path` through the service; the text around the removal is kept for "Undo removal". */
  function removeAt(path) {
    // With no other edit under way, the next change of the text is this removal's own. A removal that
    // waits behind another edit cannot tell the two apart, so it is not offered for undo.
    const alone = recipe.get().pending === 0;
    const seen = lastChange;
    return recipe.applyEdits([{ path, remove: true }]).then(() => {
      if (alone && lastChange !== seen) lastRemoval = { path, ...lastChange };
    });
  }

  /**
   * "Undo removal" for the last thing removed at or under `base` (a path, or a test of the removed path).
   * Hidden unless there is one and the recipe text is still what that removal left.
   */
  function undoOffer(base) {
    const said = h('span', { class: 'mono' });
    const mine = typeof base === 'function' ? base : (path) => base.every((part, i) => String(path[i]) === String(part));
    const applies = () => Boolean(lastRemoval) && lastRemoval.after === recipe.get().yaml && mine(lastRemoval.path);
    const button = h(
      'button',
      {
        class: 'btn btn--sm',
        type: 'button',
        onclick: () => {
          if (!applies()) return paint();
          const { path, before } = lastRemoval;
          lastRemoval = null;
          // What comes back has rows again: focus goes to its own row, else its type, else its first row.
          focusAfter = { config: recipe.get().config, keys: [pathKey(path), pathKey([...path, 'type'])], under: pathKey(path), at: Date.now() };
          recipe.setYaml(before, { immediate: true });
        },
      },
      'Undo removal',
    );
    const el = h('div', { class: 'form__actions form__undo', role: 'status', hidden: true }, h('span', null, said, ' was removed.'), button);
    const paint = () => {
      el.hidden = !applies();
      if (!el.hidden) said.textContent = pathKey(lastRemoval.path);
    };
    paint();
    updaters.push(paint);
    return el;
  }

  function noteFocusForRemoval(path) {
    // A confirm dialog may have left focus on the page body: that still counts as "in the form".
    const active = document.activeElement;
    if (active && active !== document.body && !content.contains(active)) {
      focusAfter = null;
      return;
    }
    const key = pathKey(path);
    const inside = (k) => k === key || k.startsWith(`${key}.`);
    const keys = [...fields.keys()];
    const at = keys.findIndex(inside);
    const next = at >= 0 ? keys.slice(at).filter((k) => !inside(k)) : [];
    // A block that is asked for by its type once it is gone (addTyped): the field that asks comes first.
    focusAfter = { config: recipe.get().config, keys: [...(fields.has(key) ? [key] : []), ...next], marks: [`${key}:add`], at: Date.now() };
  }

  const reachable = (el) => {
    if (!el || !el.isConnected || !content.contains(el)) return false;
    for (let n = el; n && n !== content; n = n.parentElement) {
      if (n.hidden) return false;
      if (n !== el && n.tagName === 'DETAILS' && !n.open) return false;
    }
    return true;
  };

  function restoreFocusAfterRemoval() {
    if (!focusAfter) return;
    const s = recipe.get();
    if (Date.now() - focusAfter.at > 15000) {
      focusAfter = null;
      return;
    }
    if (s.config === focusAfter.config || !s.configCurrent) return; // the removal has not reached the form yet
    const goal = focusAfter;
    focusAfter = null;
    const a = document.activeElement;
    const onAsker = Boolean(a) && a === goal.from && reachable(a);
    if (a && a !== document.body && reachable(a) && !onAsker) return; // focus is somewhere the person can see
    // A button named by its mark: the same action on the block that moved, or on its neighbour.
    for (const mark of goal.marks ?? []) {
      const el = [...content.querySelectorAll('[data-mark]')].find((x) => x.dataset.mark === mark && !x.disabled && reachable(x));
      if (el) {
        el.focus({ preventScroll: true });
        return;
      }
    }
    const within = goal.under ? [...fields.keys()].filter((k) => k.startsWith(`${goal.under}.`)) : [];
    for (const k of [...goal.keys, ...within]) {
      const field = fields.get(k);
      if (field && reachable(field.el)) {
        field.focus();
        if (content.contains(document.activeElement)) return;
      }
    }
    if (onAsker) return;
    content.querySelector('.form__title')?.focus({ preventScroll: true });
  }
  const refusedToast = (what) => (error) => toast(`${what}: ${error.message}`, { kind: 'err' });

  // ───────── task ─────────

  /** The recipe's own task, or null when it leaves the choice to the trainer. */
  const explicitTask = (c = config()) => (typeof c.task === 'string' && c.task && c.task !== 'auto' ? c.task : null);
  /** The task the check decided (the recipe's task, or the one inferred from the network). */
  const checkedTask = () => {
    const t = recipe.get().summary?.task;
    return typeof t === 'string' && t ? t : null;
  };
  const currentTask = (c = config()) => explicitTask(c) ?? checkedTask();
  const taskEntry = (id) => (catalog && id ? tasks.task(id) : null);
  const netField = (c = config()) => {
    const field = taskEntry(currentTask(c))?.networkField;
    return field === 'network_ae' || field === 'network_g' ? field : 'network_g';
  };

  /** A dataset that trains on video clips (VideoClipDataset): its own panel and words. */
  const isVideoDataset = (base, c = config()) => String(getAt(c, base)?.type ?? '').toLowerCase() === VIDEO_DATASET;
  const isRdual = (c = config()) => String(c.network_g?.type ?? '').toLowerCase() === 'rdual';
  /** A dataset of the type that can read ZIP archives, and one that does (ZIP storage chosen: dataroot_gt names files). */
  const readsZip = (base, c = config()) => String(getAt(c, base)?.type ?? '').toLowerCase() === ZIP_DATASET.toLowerCase();
  const isZipDataset = (base, c = config()) => readsZip(base, c) && getAt(c, [...base, 'io_backend', 'type']) === 'zip';
  /** A recipe that sets up distillation or names teacher weights: the Network section offers to add the teacher. */
  const wantsTeacher = (c = config()) => isPlainObject(c.train?.distillation) || (typeof c.path?.pretrain_network_g_teacher === 'string' && c.path.pretrain_network_g_teacher !== '');

  /** Issue slot for a block that has no single field (a dataset, a loss, an object group). */
  function blockTarget(path) {
    const slot = h('div', { class: 'fld__issues' });
    const target = {
      el: slot,
      pathKey: pathKey(path),
      setIssues: (list) => slot.replaceChildren(...list.map((i) => h('p', { class: i.level === 'error' ? 'field__error' : 'fld__warn' }, i.level === 'error' ? 'Error: ' : 'Warning: ', i.message))),
    };
    blocks.set(target.pathKey, target);
    return slot;
  }

  // ───────── shell ─────────

  const searchInput = h('input', {
    class: 'input form__search',
    type: 'search',
    placeholder: 'Find an option…',
    'aria-label': 'Find an option',
    autocomplete: 'off',
    spellcheck: 'false',
  });
  const picker = h('select', { class: 'select form__picker', 'aria-label': 'Section' });
  const nav = h('nav', { class: 'form__nav', 'aria-label': 'Option sections' });
  const navButtons = new Map();
  const navBadges = new Map();
  const pickerOptions = new Map();
  for (const sec of SECTIONS) {
    const slot = h('span', { class: 'form__navbadge' });
    const button = h(
      'button',
      { class: 'form__navitem', type: 'button', onclick: () => openSection(sec.id) },
      h('span', { class: 'truncate' }, sec.short ?? sec.title),
      slot,
    );
    navButtons.set(sec.id, button);
    navBadges.set(sec.id, slot);
    nav.appendChild(button);
    const option = h('option', { value: sec.id }, sec.short ?? sec.title);
    pickerOptions.set(sec.id, option);
    picker.appendChild(option);
  }
  picker.addEventListener('change', () => openSection(picker.value));

  const statusEl = h('div', { class: 'form__status' });
  const content = h('div', { class: 'form__content' });
  // A rebuild that waited for focus to leave the form (see render) happens when it leaves.
  let focusTimer = 0;
  content.addEventListener('focusout', () => {
    clearTimeout(focusTimer);
    focusTimer = setTimeout(() => {
      if (!disposed && !content.contains(document.activeElement)) render(true);
    }, 0);
  });
  // A disabled <fieldset> disables every control inside it at once (YAML syntax error state).
  const fieldset = h('fieldset', { class: 'form__fieldset' }, content);
  const reviewEl = h('div', { class: 'form__status' });
  const settings = h('div', { class: 'form__settings' }, statusEl, reviewEl, fieldset);

  // ───────── neosr import review ─────────
  // A recipe imported from neosr keeps what could not be translated under neosr_review; the trainer
  // refuses that key, so the form says so at the top until the section is gone.

  let reviewCount = null;
  const reviewSize = (v) => (isPlainObject(v) ? Object.keys(v).length : Array.isArray(v) ? v.length : 1);
  function paintReview(c) {
    const has = Boolean(c) && Object.prototype.hasOwnProperty.call(c, 'neosr_review');
    const n = has ? reviewSize(c.neosr_review) : null;
    if (n === reviewCount) return;
    reviewCount = n;
    if (!has) {
      mount(reviewEl);
      return;
    }
    const show = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => recipe.requestFocus('neosr_review') }, 'Show the section');
    const remove = h(
      'button',
      {
        class: 'btn btn--sm',
        type: 'button',
        onclick: async () => {
          const count = reviewSize(config().neosr_review);
          const ok = await confirmDialog({
            title: 'Remove the neosr_review section?',
            body: `It holds ${count === 1 ? '1 entry' : `${count} entries`} the import could not translate. Removing the section drops ${count === 1 ? 'it' : 'them'} from the recipe; Changes can bring the section back.`,
            confirmLabel: 'Remove the section',
            danger: true,
          });
          if (!ok) return;
          // The notice goes away with the section: focus moves to the section heading first.
          content.querySelector('.form__title')?.focus({ preventScroll: true });
          recipe.applyEdits([{ path: ['neosr_review'], remove: true }]).catch(refusedToast('The neosr_review section was not removed'));
        },
      },
      'Remove the section',
    );
    mount(
      reviewEl,
      notice(
        'warn',
        'This recipe was imported from neosr and still has entries to review. Resolve them, then remove the neosr_review section to enable training.',
        h('div', { class: 'row' }, show, remove),
      ),
    );
  }
  const shell = h(
    'div',
    { class: 'form__layout' },
    h('div', { class: 'form__outline' }, searchInput, picker, nav),
    settings,
  );

  const runSearch = debounce(() => {
    const q = searchInput.value.trim().toLowerCase();
    if (q === query) return;
    query = q;
    builtSig = null;
    settings.scrollTop = 0;
    render(true);
  }, 150);
  searchInput.addEventListener('input', runSearch);
  searchInput.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && searchInput.value) {
      searchInput.value = '';
      runSearch.flush();
    }
  });

  function markNav() {
    for (const [id, button] of navButtons) {
      if (id === section && !query) button.setAttribute('aria-current', 'true');
      else button.removeAttribute('aria-current');
    }
    if (picker.value !== section) picker.value = section;
  }

  function openSection(id) {
    if (!SECTION_IDS.includes(id)) return;
    section = id;
    persist.set('recipe-section', id);
    if (query || searchInput.value) {
      searchInput.value = '';
      runSearch.cancel();
      query = '';
    }
    builtSig = null;
    markNav();
    render(true);
    settings.scrollTop = 0;
  }

  // ───────── row builders ─────────

  const register = (field) => {
    fields.set(field.pathKey, field);
    return field;
  };

  function leaf(path, key, kind, unknown = false, extra = {}) {
    const c = config();
    return register(createField({ path, key, kind, value: getAt(c, path), isSet: hasAt(c, path), unknown, ...extra, ...fieldExtras(path, key, kind) }, ctx)).el;
  }

  /** Task-dependent wording and notes for a row: folder words, dataset types, the class count hint. */
  function fieldExtras(path, key, kind) {
    const out = {};
    const task = currentTask();
    if (path.length === 3 && path[0] === 'datasets') {
      const words = folderWords(task);
      // A task that reads only dataroot_lq (paired: false) does not relabel dataroot_gt: the schema's
      // label stays, and the check says whether the option is allowed there.
      const side = key === 'dataroot_gt' ? (words.paired === false ? null : words.gt) : key === 'dataroot_lq' ? words.lq : null;
      if (words.file && key === words.file.key) {
        // A file per split (detection's COCO JSON): caption, the task's description, a file picker.
        out.caption = words.file.name;
        out.lead = sentence(words.file.what);
        out.browse = { mode: 'file', extensions: ['.json'], title: `Choose the ${words.file.name.toLowerCase()}` };
      } else if (side) {
        out.caption = side.name;
        // "Choose the class folders folder" reads badly: a plural name is a folder of folders.
        out.pickTitle = /folders$/i.test(side.name) ? `Choose the folder that holds the ${side.name.toLowerCase()}` : `Choose the ${side.name.toLowerCase()} folder`;
        // A task with its own words describes the folder in them; the schema's text (written for
        // high and low resolution images) stays available behind "More".
        if (!restorationWords(words)) out.lead = sentence(side.what);
      } else if (key === 'type' && (kind.kind === 'enum' || kind.kind === 'string')) {
        // The schema names the dataset types it knows. A dataset class added to the trainer has another
        // name, so the field takes any id: the known ones are listed, and the check says what runs.
        const entry = taskEntry(task);
        const listed = Array.isArray(entry?.datasetTypes) ? entry.datasetTypes : [];
        const known = () => {
          const named = (kind.options ?? kind.suggestions ?? []).filter((t) => typeof t === 'string');
          const all = [...new Set([...named, ...listedIn('datasets').map((item) => item.id)])];
          const first = listed.filter((t) => all.includes(t));
          return [...first, ...all.filter((t) => !first.includes(t))];
        };
        out.kind = { ...kind, kind: 'string', options: undefined, suggestions: undefined };
        out.choices = {
          listLabel: 'Dataset types',
          getChoices: () => toChoices(known().map((t) => ({ value: t, note: listed.length && !listed.includes(t) ? 'not listed for this task' : null }))),
        };
        out.note = (value, isSet) => {
          if (!isSet || typeof value !== 'string' || !value) return null;
          if (listed.length && !listed.includes(value)) return `${NOT_LISTED}: ${tasks.taskLabel(task)} lists ${listed.join(', ')}.`;
          const all = known();
          return all.length && !all.some((t) => t.toLowerCase() === value.toLowerCase()) ? `Not one of the ${all.length} dataset types the installed trainer lists.` : null;
        };
      }
      // Clip training reads video files, and makes its low quality clips itself.
      if (isVideoDataset(path.slice(0, 2))) {
        if (key === 'dataroot_gt') {
          out.caption = 'Videos';
          out.lead = 'Folder that holds the training videos, or single video files. Several entries are separated by commas.';
          out.pickTitle = 'Folder with the training videos';
        } else if (key === 'dataroot_lq') out.lead = 'Not used: the low quality clips are made on the fly.';
      } else if (key === 'dataroot_gt') {
        const base = path.slice(0, 2);
        if (isZipDataset(base)) {
          // ZIP storage reads archive files: the row asks for them, not for a folder.
          out.lead = 'The .zip files that hold the images, in place of folders.';
          out.browse = { mode: 'file', extensions: ['.zip'] };
          out.pickTitle = 'Choose a ZIP archive of images';
        } else {
          // A .zip file where archives are not read: the row says what reads them.
          out.note = (value, isSet) =>
            isSet && toList(value).some((p) => /\.zip$/i.test(p))
              ? { text: readsZip(base) ? 'A .zip file is read once io_backend.type is zip.' : `ZIP archives are read by ${ZIP_DATASET} only, with io_backend.type set to zip.`, warn: true }
              : null;
        }
      }
    }
    if (path.length === 4 && path[0] === 'datasets' && path[2] === 'io_backend' && key === 'type' && (kind.kind === 'string' || kind.kind === 'enum')) {
      // The storage types the schema names; a dataset class added to the trainer may read another, so any
      // text can be typed. ZIP is read by one dataset type: it is listed for that type alone, and the row
      // says so when another type has it.
      const base = path.slice(0, 2);
      const named = (kind.options ?? kind.suggestions ?? []).filter((t) => typeof t === 'string');
      out.kind = { ...kind, kind: 'string', options: undefined, suggestions: undefined };
      out.choices = {
        listLabel: 'Storage types',
        getChoices: () => toChoices(named.filter((t) => t !== 'zip' || readsZip(base)).map((t) => ({ value: t, note: t === 'zip' ? '.zip archives' : null }))),
      };
      out.note = (value, isSet) => {
        if (!isSet || value !== 'zip') return null;
        return readsZip(base) ? 'dataroot_gt names the .zip files. ZIP storage takes no other io_backend option.' : { text: `ZIP storage is read by ${ZIP_DATASET} only.`, warn: true };
      };
    }
    if (path.length === 2 && path[0] === 'path' && key === 'neuflow_weights' && isRdual()) out.lead = NEUFLOW_RDUAL;
    if (isBestWeights(path)) {
      out.caption = BEST_CAPTION;
      out.lead = BEST_HELP;
      out.choices = { listLabel: 'Metrics of this recipe', unsetText: BEST_OFF, getChoices: bestChoices };
      out.note = bestNote;
    }
    const nf = netField();
    if ((path.length === 2 && path[0] === nf && key === 'num_classes') || (path.length === 2 && path[0] === 'task_options' && (key === 'class_names' || key === 'categories'))) {
      out.note = () => classCountNote();
    }
    return out;
  }

  /**
   * Derived comparison of two recipe values: num_classes against class_names (segmentation,
   * classification) or against categories (detection, where the schema says label 0 is background
   * and the categories are labels 1, 2, …). Stated as facts, shown only when they do not line up.
   */
  function classCountNote() {
    const c = config();
    const nf = netField(c);
    const n = getAt(c, [nf, 'num_classes']);
    if (typeof n !== 'number') return null;
    const categories = getAt(c, ['task_options', 'categories']);
    if (Array.isArray(categories)) {
      const k = categories.length;
      if (n === k + 1) return null;
      return `Two recipe values: ${nf}.num_classes is ${n}, and task_options.categories lists ${k} ${k === 1 ? 'category' : 'categories'}, which the task numbers from label 1 after background label 0.`;
    }
    const names = getAt(c, ['task_options', 'class_names']);
    if (!Array.isArray(names) || n === names.length) return null;
    return `These two recipe values differ: ${nf}.num_classes is ${n}, and task_options.class_names lists ${names.length} ${names.length === 1 ? 'name' : 'names'}.`;
  }

  // ───────── val.save_best ─────────
  // The trainer accepts one of the recipe's own metrics here: a key under val.metrics that holds a type
  // (trainner_config/metrics.py), or one of the names the recipe check lists. The row offers those names
  // and "Do not keep them" (which removes the key); another name can be typed, and the check judges it.
  // The three tasks that score under their own names (top1, mIoU, AP50, …) list names only once the
  // check does.

  const BEST_CAPTION = 'Keep the best weights by';
  const BEST_OFF = 'Do not keep them';
  const BEST_HELP =
    'Saves the weights of the validation that scored best on this metric, one file per validation dataset, even when it falls between two numbered saves. The numbered saves and the training state are not changed.';
  const OWN_METRIC_TASKS = ['image_classification', 'semantic_segmentation', 'object_detection'];
  const isBestWeights = (path) => path.length === 2 && path[0] === 'val' && path[1] === 'save_best';
  /**
   * The metric names the latest recipe check accepts for val.save_best, each with its direction
   * ('higher' or 'lower'); null when the check lists none, and then the rules above apply.
   */
  const snapshotMetrics = () => {
    const listed = recipe.get().summary?.snapshotMetrics;
    return isPlainObject(listed) && Object.keys(listed).length ? listed : null;
  };
  const directionWords = (direction) => (direction === 'higher' ? 'higher is better' : direction === 'lower' ? 'lower is better' : null);
  /** The names offered: those the check lists (they are what the trainer accepts), else the recipe's own metrics. */
  const bestNames = () => {
    const listed = snapshotMetrics();
    if (listed) return Object.keys(listed);
    if (OWN_METRIC_TASKS.includes(currentTask())) return [];
    const metrics = getAt(config(), ['val', 'metrics']);
    return isPlainObject(metrics) ? Object.keys(metrics).filter((k) => isPlainObject(metrics[k]) && typeof metrics[k].type === 'string') : [];
  };
  const bestChoices = () => {
    const listed = snapshotMetrics();
    return toChoices([{ unset: true, text: BEST_OFF }, ...bestNames().map((name) => ({ value: name, note: listed ? directionWords(listed[name]) : null }))]);
  };
  /** Under the row: a name that is not one of the recipe's metrics, and which datasets the check keeps weights for. */
  function bestNote(value, isSet) {
    if (!isSet || typeof value !== 'string' || !value) return null;
    const names = bestNames();
    const said = names.length && !names.includes(value) ? ['Not a metric of this recipe.'] : [];
    const kept = recipe.get().summary?.bestValidationDatasets;
    if (Array.isArray(kept) && !kept.length) return { text: [...said, 'No validation dataset of this recipe is scored, so no best weights would be kept.'].join(' '), warn: true };
    if (Array.isArray(kept)) said.push(`Best weights are kept for: ${kept.join(', ')}.`);
    return said.length ? said.join(' ') : null;
  }

  function subHeading(path, key, description, extra = null) {
    return h(
      'div',
      { class: 'form__subhead' },
      h('h3', { class: 'form__subtitle mono', dataset: { path: pathKey(path) } }, key, extra),
      description ? h('p', { class: 'fld__desc' }, description) : null,
    );
  }

  function genericBlock(path, key, description = '', unknown = false) {
    // A block the recipe has can be taken out as a whole: the shared schedule, to give each optimizer its
    // own; a discriminator that is no longer wanted. It asks first, and Changes can bring the block back.
    const remove = hasAt(config(), path)
      ? h(
          'div',
          { class: 'form__actions' },
          h(
            'button',
            {
              class: 'btn btn--sm btn--quiet',
              type: 'button',
              onclick: async () => {
                const ok = await confirmDialog({ title: `Remove ${key}?`, body: `${pathKey(path)} and every option in it are taken out of the recipe. Changes can bring it back.`, confirmLabel: `Remove ${key}`, danger: true });
                if (!ok) return;
                ctx.remove(path).catch(refusedToast(`${key} was not removed`));
              },
            },
            `Remove ${key}`,
          ),
        )
      : null;
    const value = getAt(config(), path);
    // An optimizer, a schedule or a metric the recipe does not have cannot take options yet: its place
    // asks for the type. A value that is no block of options is said as it is, since choosing replaces it.
    const absent = Boolean(blockListName(path)) && !isPlainObject(value);
    const odd = absent && value !== null && value !== undefined;
    return h(
      'div',
      { class: 'form__sub' },
      subHeading(path, key, description, unknown ? h('span', { class: 'fld__tag fld__tag--unknown' }, 'Not in the schema') : null),
      blockTarget(path),
      undoOffer(path),
      absent
        ? [h('p', { class: 'form__empty' }, odd ? `${key} is ${JSON.stringify(value)} in the recipe, which is not a block of options. Choosing a type replaces it.` : `${key} is not set. Choose its type to add it.`), addTyped(path)]
        : genericEditor({ path, value, ctx, register, firstKeys: ['type'], rowExtras: knownRow, knownOptions: knownOptionsAt, special: blockSpecial }),
      remove,
    );
  }

  // ───────── adding a block by its type ─────────
  // An optimizer, a schedule and a metric take the options of their type, so one the recipe does not have
  // is added by choosing that type: the installed ones are listed, any other id can be typed, and nothing
  // is written before the button is pressed. The new block holds its type alone. The options that type
  // takes are then rows of the block, each at its default until it is changed (unsetParameters).

  const moreId = (path) => `more:${pathKey(path)}`;

  /**
   * The field and button that add a typed block at `base`, or under `base` by a name of its own.
   * @param {Array} base the block's path; with `keyed`, the path of the mapping that holds such blocks
   * @param {{label: string, placeholder: string, choices?: () => Array, suggest?: (type: string) => string}|null} keyed
   *   the name a block goes under (a schedule's optimizer, a metric's name): `choices` lists the usual
   *   names, `suggest` gives the name a picked type starts with. Any other name can be typed.
   */
  function addTyped(base, keyed = null) {
    const known = knownList([...base, ...(keyed ? [''] : []), 'type']);
    const { word, choose } = TYPED_WORDS[known.name];
    const typeId = uid('addtype');
    const keyId = uid('addkey');
    const error = h('p', { class: 'field__error', role: 'alert', hidden: true });
    const say = (message) => {
      error.textContent = message ?? '';
      error.hidden = !message;
    };
    let suggested = '';
    const add = () => {
      const typed = picker.input.value.trim();
      const key = keyed ? keyInput.value.trim() : '';
      const held = getAt(config(), base);
      const problem = !typed
        ? `${choose} from the list, or type its id.`
        : !keyed
          ? null
          : !key
            ? `Enter the ${keyed.label.toLowerCase()}.`
            : /[.\s[\]/]/.test(key)
              ? `The ${keyed.label.toLowerCase()} cannot contain dots, slashes, brackets or spaces.`
              : isPlainObject(held) && key in held
                ? `${pathKey(base)} already has ${key}.`
                : null;
      say(problem);
      if (problem) return (typed ? keyInput : picker.input).focus();
      // A listed type is written as it is listed; any other id is the person's own and is written as typed.
      const hit = listedIn(known.name).find((item) => namedBy(item, typed));
      const path = keyed ? [...base, key] : base;
      picker.input.value = '';
      if (keyed) keyInput.value = '';
      suggested = '';
      // The new block opens with the options of its type in view; focus goes to its type, its first row.
      openGroups.add(moreId(path));
      focusAfter = { config: recipe.get().config, keys: [pathKey([...path, 'type'])], at: Date.now() };
      return recipe.applyEdits([{ path, value: { type: hit ? shownName(hit) : typed } }]).catch((failure) => {
        focusAfter = null;
        toast(`The ${word} was not added: ${failure.message}`, { kind: 'err' });
      });
    };
    const picker = createCombobox({
      id: typeId,
      inputClass: 'input--path',
      placeholder: choose,
      closed: false,
      listLabel: known.label,
      // Where focus goes when the block is removed: the place to choose it again.
      attrs: { 'data-mark': `${pathKey(base)}:add` },
      getChoices: () => typeChoices(known),
      onPick: (choice) => {
        say(null);
        if (!keyed) return;
        // A name the person has not typed themselves follows the type that is picked.
        if (keyed.suggest && (!keyInput.value.trim() || keyInput.value === suggested)) {
          suggested = keyed.suggest(String(choice.value));
          keyInput.value = suggested;
        }
        keyInput.focus();
      },
      onEnter: add,
      onEscape: () => {
        picker.input.value = '';
        say(null);
      },
    });
    const keyCombo = keyed?.choices
      ? createCombobox({ id: keyId, inputClass: 'input--path', placeholder: keyed.placeholder, closed: false, listLabel: keyed.label, getChoices: () => toChoices(keyed.choices()), onPick: () => say(null), onEnter: add })
      : null;
    const keyInput = keyCombo ? keyCombo.input : keyed ? h('input', { class: 'input input--path', id: keyId, type: 'text', autocomplete: 'off', spellcheck: 'false', placeholder: keyed.placeholder }) : null;
    if (keyInput && !keyCombo) {
      keyInput.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        add();
      });
    }
    return h(
      'div',
      { class: `form__actions${keyed ? ' form__addkeyed' : ''}` },
      h('label', { class: 'visually-hidden', for: typeId }, `Type of the ${word} to add`),
      picker.el,
      keyed ? [h('label', { class: 'visually-hidden', for: keyId }, keyed.label), keyCombo ? keyCombo.el : keyInput] : null,
      h('button', { class: 'btn btn--sm', type: 'button', onclick: add }, `Add ${word}`),
      error,
    );
  }

  // ───────── known values of free-form options ─────────
  // optim_g, scheduler, schedulers, network_d and val.metrics are free-form in the schema: their rows come
  // from the recipe's own keys. The catalog names what the installed trainer offers for their `type`, the
  // options an optimizer, a schedule or a metric takes, and the values some of those options take. The
  // rows list them. The lists are read when a list opens, so a catalog that arrives later is used as soon
  // as it is there. Any other spelling can be typed: the check stays the judge.

  const isEntry = (item) => item && typeof item.id === 'string' && item.id;
  /**
   * The same lists as the schema carries them (optimizers, schedulers, metrics). A service that was
   * started before it listed something still serves the current schema, so the form reads it from there.
   */
  const packagedList = (name) => {
    const list = schema?.root?.['x-trainner-catalog']?.[name];
    return Array.isArray(list) ? list.filter(isEntry) : [];
  };
  const listedIn = (name) => {
    const list = tasks.get()[name];
    const own = Array.isArray(list) ? list.filter(isEntry) : [];
    return own.length ? own : packagedList(name);
  };
  /** The name an entry is listed and written by: its display name where the catalog gives one (AdamW), else its id. */
  const shownName = (item) => (typeof item.displayName === 'string' && item.displayName ? item.displayName : item.id);
  /** Whether `text` names the entry: its id or its display name, without regard to case. */
  const namedBy = (item, text) => {
    const t = String(text).toLowerCase();
    return item.id.toLowerCase() === t || shownName(item).toLowerCase() === t;
  };
  /**
   * Whether `path` is a schedule: the shared one (train.scheduler), one optimizer's (train.schedulers.g,
   * .d, .ae), or one inside another (… .schedulers[i], to any depth).
   */
  function isSchedulePath(path) {
    if (String(path[0]) !== 'train') return false;
    const rest = String(path[1]) === 'scheduler' ? path.slice(2) : String(path[1]) === 'schedulers' && path.length >= 3 ? path.slice(3) : null;
    if (!rest || rest.length % 2) return false;
    return rest.every((part, i) => (i % 2 ? /^\d+$/.test(String(part)) : String(part) === 'schedulers'));
  }
  /** The schedule that holds the one at `path` (… .schedulers[i]); null for a schedule that stands alone. */
  const scheduleHolder = (path) => (path.length >= 4 && isSchedulePath(path) && isSchedulePath(path.slice(0, -2)) ? path.slice(0, -2) : null);
  /** The catalog entry of the schedule type written at `path`; null when it is not a listed one. */
  const scheduleEntry = (path) => {
    const type = getAt(config(), [...path, 'type']);
    return typeof type === 'string' && type ? (listedIn('schedulers').find((item) => namedBy(item, type)) ?? null) : null;
  };
  /** The listed schedules the catalog allows inside `entry` (their `allowedParents` name it); none for most. */
  const partsOf = (entry) => (entry ? listedIn('schedulers').filter((item) => Array.isArray(item.allowedParents) && item.allowedParents.some((parent) => namedBy(entry, parent))) : []);

  /** Which catalog list names the values of the option at `path`, with the words for it; null for most paths. */
  function knownList(path) {
    const [a, b, c, d] = path.map(String);
    if (path.length === 3 && a === 'train' && /^optim_/.test(b) && c === 'type') return { name: 'optimizers', many: 'optimizers', label: 'Optimizers' };
    if (String(path[path.length - 1]) === 'type' && isSchedulePath(path.slice(0, -1))) {
      const known = { name: 'schedulers', many: 'schedulers', label: 'Schedulers' };
      const holder = scheduleHolder(path.slice(0, -1));
      if (!holder) return known;
      // Inside another schedule, the list holds the types the catalog allows there. Any other can be typed.
      return {
        ...known,
        only: (item) => {
          const entry = scheduleEntry(holder);
          return !entry || partsOf(entry).includes(item);
        },
        misfit: (item) => `The installed trainer does not list ${shownName(item)} for use inside ${shownName(scheduleEntry(holder))}.`,
      };
    }
    if (path.length === 4 && a === 'val' && b === 'metrics' && d === 'type') {
      // The one thing the catalog says about a metric besides its name.
      return { name: 'metrics', many: 'metrics', label: 'Metrics', note: (item) => (item.requiresReference === false ? 'needs no reference image' : null) };
    }
    if (path.length === 2 && /^network_/.test(a) && b === 'type') {
      // Every installed network type; for the discriminator, those that fit one come first.
      return { name: 'architectures', many: 'network types', label: 'Network types', order: a === 'network_d' ? discriminatorsFirst : null };
    }
    return null;
  }

  // network_d, network_g_teacher and network_ae are free-form in the schema, yet the trainer builds each
  // from a registered network type, and the schema describes those types for network_g. The recipe check
  // holds every one of these blocks to that description, so the form reads the option names, the values
  // they start from and the values they list from it too. A type the schema does not describe (a network
  // added to the trainer) or one that does not close its options takes any name: those are typed.

  /** A network block of the recipe as a whole: network_g, network_d, network_g_teacher, network_ae. */
  const isNetworkBlock = (path) => path.length === 1 && NETWORK_KEYS.includes(String(path[0]));
  /** The schema's description of a network type; null for a type it does not describe. */
  const networkDef = (type) => (typeof type === 'string' && type ? schema.variantFor(netKind, type) : null);
  /**
   * Whether a listed network type is a discriminator: its name says so, or it is one of the registered
   * ones known by id. Nothing is read from a type's options. This orders and fills the lists; any other
   * type can still be typed.
   */
  const fitsDiscriminator = (item) => /discriminator/i.test(item.id) || DISCRIMINATOR_IDS.has(item.id.toLowerCase());
  /** `items` with the discriminators first: the ones named as one, then the ones known by id. */
  const discriminatorsFirst = (items) => {
    const fit = items.filter(fitsDiscriminator);
    const named = fit.filter((item) => /discriminator/i.test(item.id));
    return [...named, ...fit.filter((item) => !named.includes(item)), ...items.filter((item) => !fit.includes(item))];
  };
  /**
   * The options the schema lists for the network type written at `path`, as the catalog lists parameters.
   * `scale` is left out except for the discriminator: every other network gets the recipe's own scale,
   * over whatever its block says.
   */
  function networkParameters(path) {
    const def = networkDef(getAt(config(), [...path, 'type']));
    if (!def) return [];
    return Object.entries(schema.props(def))
      .filter(([name]) => name !== 'type' && !(name === 'scale' && String(path[0]) !== 'network_d'))
      .map(([name, prop]) => {
        const kind = schema.kind(prop);
        const choices = kind.kind === 'enum' ? kind.options : kind.suggestions;
        return { name, ...(kind.hasDefault ? { default: kind.default } : {}), ...(Array.isArray(choices) && choices.length ? { choices } : {}) };
      });
  }

  /**
   * The catalog list that describes the free-form object at `path`: an optimizer, a schedule (the shared
   * one, one optimizer's, one inside another), a metric. null for others.
   */
  function blockListName(path) {
    const [a, b] = path.map(String);
    if (path.length === 2 && a === 'train' && /^optim_/.test(b)) return 'optimizers';
    if (isSchedulePath(path)) return 'schedulers';
    if (path.length === 3 && a === 'val' && b === 'metrics') return 'metrics';
    return null;
  }
  /** The parameters the catalog lists for the object at `path`, by the type the recipe gives it now. */
  function blockParameters(path) {
    if (isNetworkBlock(path)) return networkParameters(path);
    const name = blockListName(path);
    const type = name ? getAt(config(), [...path, 'type']) : null;
    if (typeof type !== 'string') return [];
    const named = (item) => namedBy(item, type);
    const own = listedIn(name).find(named);
    // A service that names the type without its parameters (it started before it listed them): the
    // schema's copy of the same entry has them.
    const entry = own && !Array.isArray(own.parameters) ? (packagedList(name).find(named) ?? own) : own;
    return Array.isArray(entry?.parameters) ? entry.parameters.filter((p) => p && typeof p.name === 'string') : [];
  }
  /**
   * The types of the recipe's optimizers, schedules and metrics: a change of one changes which rows list
   * values and what "Add option" offers, so the section is drawn again.
   */
  const blockTypes = (c) => {
    const train = isPlainObject(c.train) ? c.train : {};
    const metrics = isPlainObject(c.val?.metrics) ? c.val.metrics : {};
    const each = isPlainObject(train.schedulers) ? train.schedulers : {};
    return JSON.stringify([
      Object.keys(train).filter((k) => /^optim_/.test(k)).map((k) => train[k]?.type ?? null),
      Object.values(metrics).map((m) => m?.type ?? null),
      scheduleShape(train.scheduler),
      Object.keys(each).map((k) => [k, scheduleShape(each[k])]),
      // The same for the networks (a type decides the options offered), for whether a teacher is asked
      // for, and for each dataset's type and storage (ZIP storage names files where the others name folders).
      NETWORK_KEYS.map((k) => c[k]?.type ?? null),
      wantsTeacher(c),
      Object.values(isPlainObject(c.datasets) ? c.datasets : {}).map((d) => [d?.type ?? null, d?.io_backend?.type ?? null]),
    ]);
  };
  /**
   * What decides the rows of a schedule: its type, its keys, and the same for each schedule in its list,
   * in order (the section's own signature sees a list only as "a list").
   */
  function scheduleShape(node) {
    if (!isPlainObject(node)) return [shape(node)];
    return [node.type ?? null, shape({ ...node, schedulers: null }), Array.isArray(node.schedulers) ? node.schedulers.map(scheduleShape) : null];
  }

  /** The types a `knownList` answer names, as the choices of a list: each by the name it is written by. */
  const typeChoices = (known) => {
    const all = listedIn(known.name).filter((item) => !known.only || known.only(item));
    return toChoices((known.order ? known.order(all) : all).map((item) => ({ value: shownName(item), note: known.note ? known.note(item) : null })));
  };

  /** More of a generic row's spec: the list of known values and a note when the value is not among them. */
  function knownRow(path, key, value) {
    if (typeof value !== 'string') return null;
    const known = knownList(path);
    if (!known) {
      // An option of an optimizer, a schedule or a metric whose values the catalog lists (data_format,
      // mode, net).
      const block = path.slice(0, -1);
      const listed = () => {
        const parameter = blockParameters(block).find((item) => item.name === key);
        return Array.isArray(parameter?.choices) ? parameter.choices : [];
      };
      if (!listed().length) return null;
      return {
        choices: { listLabel: `Values of ${key}`, getChoices: () => toChoices(listed()) },
        note: (now, isSet) => {
          const values = listed();
          return isSet && typeof now === 'string' && values.length && !values.includes(now) ? `Not one of the values listed for ${key}: ${values.join(', ')}.` : null;
        },
      };
    }
    return {
      choices: { listLabel: known.label, getChoices: () => typeChoices(known) },
      // Compared without regard to case: recipes write AdamW where the trainer's id is adamw.
      note: (now, isSet) => {
        const all = listedIn(known.name);
        if (!isSet || typeof now !== 'string' || !now || !all.length) return null;
        const hit = all.find((item) => namedBy(item, now));
        if (!hit) return `Not one of the ${all.length} ${known.many} the installed trainer lists.`;
        // A listed type in a place the catalog does not list it for (a schedule inside another).
        return known.only && !known.only(hit) ? known.misfit(hit) : null;
      },
    };
  }

  // ───────── a schedule made of schedules ─────────
  // SequentialLR and ChainedScheduler hold a list of schedules (`schedulers`). Each is a schedule like any
  // other: a type, options, and for some a list of its own. The catalog says which types may stand inside
  // which (`allowedParents`). The list is drawn as one block per schedule, in order, with the rows of a
  // schedule that stands alone. A change inside one schedule writes that key only, so the others, the
  // milestones and keys the catalog does not know stay as they are. Adding and moving write the list as a
  // whole (the service has no "insert at" edit). An entry that is not a mapping keeps its value and is
  // shown as it is; a list under a type the catalog does not list as holding schedules stays one row.

  /** What the form draws in a free-form block beyond the recipe's own keys (fields.js genericEditor `special`). */
  const blockSpecial = {
    // A block that lacks its type asks for it first: nothing else about it is known without one.
    first: (path, object) => (typedBlock(path) && !('type' in object) ? [leaf([...path, 'type'], 'type', kindFromValue(''), false, knownRow([...path, 'type'], 'type', '') ?? {})] : []),
    row: (path, key, value) => {
      const holder = path.slice(0, -1);
      return key === 'schedulers' && Array.isArray(value) && isSchedulePath(holder) && partsOf(scheduleEntry(holder)).length ? scheduleParts(path, value, true) : null;
    },
    more: (path, object) => [
      ...unsetParameters(path, object),
      // A type that holds schedules and has no list yet: the place to add the first one.
      ...(!('schedulers' in object) && isSchedulePath(path) && partsOf(scheduleEntry(path)).length ? [scheduleParts([...path, 'schedulers'], [], false)] : []),
    ],
    // A schedule of one optimizer and a metric go under a name of their own, and each needs its type: the
    // two are asked for together, where "Add option" would take a name and a value typed as JSON.
    adder: (path, object) => {
      const key = pathKey(path);
      if (key === 'train.schedulers') {
        const roles = Object.keys(schema.props(schema.props(rootProps.train).schedulers));
        return [
          Object.keys(object).length ? null : h('p', { class: 'form__empty' }, 'No optimizer has a schedule of its own.'),
          addTyped(path, {
            label: 'Optimizer',
            placeholder: roles.length ? `for optimizer: ${roles.join(', ')}` : 'for optimizer',
            choices: roles.length ? () => roles.filter((role) => !(role in object)).map((role) => ({ value: role, note: `optim_${role}` })) : null,
          }),
        ];
      }
      if (key === 'val.metrics') {
        return [
          Object.keys(object).length ? null : h('p', { class: 'form__empty' }, 'No metric is listed.'),
          addTyped(path, { label: 'Metric name', placeholder: 'name, for example psnr', suggest: (type) => type.replace(/^calculate_/, '') }),
        ];
      }
      return null;
    },
  };

  // ───────── the options a type takes ─────────
  // The catalog lists the parameters of each optimizer, schedule and metric, most with the value they start
  // from. Those the recipe does not set are rows too: drawn at that value and written only once the person
  // changes one. A parameter the type cannot do without stands in the open; the others wait in a group.

  /** The kind of a listed parameter: read from the value it starts from; without one, any value is taken. */
  function parameterKind(parameter) {
    const hasDefault = 'default' in parameter && parameter.default !== undefined;
    const listed = Array.isArray(parameter.choices) && parameter.choices.some((choice) => typeof choice === 'string');
    const base = listed ? kindFromValue('') : hasDefault && parameter.default !== null ? kindFromValue(parameter.default) : { kind: 'loose', nullable: hasDefault, description: '' };
    return { ...base, hasDefault, default: parameter.default };
  }

  /** Rows for the parameters the type at `path` lists and `object` (the block as the recipe has it) does not set. */
  function unsetParameters(path, object) {
    if (!blockListName(path)) return [];
    // A type that holds schedules has its `schedulers` drawn as that list, not as a row.
    const holds = partsOf(scheduleEntry(path)).length > 0;
    const unset = blockParameters(path).filter((p) => p.name !== 'type' && !(p.name in object) && !(holds && p.name === 'schedulers'));
    const type = getAt(config(), [...path, 'type']);
    const row = (p, lead) => leaf([...path, p.name], p.name, parameterKind(p), false, { ...(knownRow([...path, p.name], p.name, '') ?? {}), ...(lead ? { lead } : {}) });
    const needed = unset.filter((p) => p.required === true && !('default' in p));
    const rest = unset.filter((p) => !needed.includes(p));
    return [
      ...needed.map((p) => row(p, `${type} needs this option: it has no default.`)),
      ...(rest.length ? [group(moreId(path), 'More options', rest.map((p) => row(p, null)), `${rest.length} not set`).el] : []),
    ];
  }

  /**
   * The schedules inside the schedule that holds `listPath` (… .schedulers).
   * @param {Array} listPath path of the list
   * @param {Array} list its entries as the recipe has them
   * @param {boolean} exists whether the recipe has the key (false: only the place to add the first one)
   */
  function scheduleParts(listPath, list, exists) {
    const holder = listPath.slice(0, -1);
    const holderName = shownName(scheduleEntry(holder));
    const listKey = pathKey(listPath);
    const mark = (i, act) => `${listKey}.${i}:${act}`;
    const addMark = `${listKey}:add`;
    const addId = uid('addsched');
    /** Write the list as a whole, then send focus to what `goal` names once the checked recipe is back. */
    const writeList = (next, goal, failed) => {
      focusAfter = { config: recipe.get().config, keys: [], ...goal, at: Date.now() };
      return recipe.applyEdits([{ path: listPath, value: next }]).catch((error) => {
        focusAfter = null;
        toast(`${failed}: ${error.message}`, { kind: 'err' });
      });
    };
    /**
     * The list as the checked recipe has it, once the edits under way have landed. Adding and moving write
     * the whole list, and `list` is only what these rows were drawn from: a value changed inside one
     * schedule since then is not in it, and one still on its way (the field that lost focus to the button)
     * is not in the parsed recipe yet either. Writing from either would put the old value back.
     */
    const currentList = () =>
      new Promise((resolve) => {
        const read = () => {
          const s = recipe.get();
          if (s.pending || s.check === 'checking') return false;
          const now = getAt(s.config ?? {}, listPath);
          resolve(Array.isArray(now) ? now : []);
          return true;
        };
        if (read()) return;
        const off = recipe.subscribe(() => {
          if (read()) off();
        });
      });
    const parts = list.map((child, i) => {
      const base = [...listPath, i];
      const label = `schedulers[${i}]`;
      const button = (act, text, disabled, onclick) =>
        h('button', { class: 'btn btn--sm btn--quiet', type: 'button', disabled, dataset: { mark: mark(i, act) }, 'aria-label': `${text} ${label}`, onclick }, text);
      const move = (to, act) => async (event) => {
        const from = event.currentTarget;
        const next = [...(await currentList())];
        // The schedule is no longer there (the list changed meanwhile): nothing to move.
        if (i >= next.length) return;
        const [moved] = next.splice(i, 1);
        next.splice(to, 0, moved);
        // Focus follows the schedule that moved: the same button on it, else its other move button.
        writeList(next, { marks: [mark(to, act), mark(to, act === 'up' ? 'down' : 'up')], from }, 'The schedule was not moved');
      };
      const remove = async (event) => {
        const from = event.currentTarget;
        const named = isPlainObject(child) && typeof child.type === 'string' && child.type ? `${label} (${child.type})` : label;
        const ok = await confirmDialog({
          title: `Remove ${label}?`,
          body: `${named} and its options are taken out of ${listKey}. The other schedules stay as they are.`,
          confirmLabel: 'Remove schedule',
          danger: true,
        });
        if (!ok) return;
        // The same button on the schedule that takes this place, else the last one's, else the box to add one.
        const left = Math.min(i, list.length - 2);
        focusAfter = { config: recipe.get().config, keys: [], marks: [...(left >= 0 ? [mark(left, 'remove')] : []), addMark], from, at: Date.now() };
        removeAt(base).catch((error) => {
          focusAfter = null;
          toast(`The schedule was not removed: ${error.message}`, { kind: 'err' });
        });
      };
      // The type comes first even when the recipe left it out (blockSpecial.first).
      const body = isPlainObject(child)
        ? genericEditor({ path: base, value: child, ctx, register, depth: 1, firstKeys: ['type'], rowExtras: knownRow, knownOptions: knownOptionsAt, special: blockSpecial })
        : leaf(base, label, kindFromValue(child));
      return h(
        'div',
        { class: 'form__nested' },
        h(
          'div',
          { class: 'form__nestedhead' },
          h('p', { class: 'form__nestedtitle mono', dataset: { path: pathKey(base) } }, label),
          h(
            'span',
            { class: 'fld__itemacts' },
            button('up', 'Move up', i === 0, move(i - 1, 'up')),
            button('down', 'Move down', i === list.length - 1, move(i + 1, 'down')),
            button('remove', 'Remove', false, remove),
          ),
        ),
        blockTarget(base),
        body,
      );
    });

    const add = async () => {
      const typed = picker.input.value.trim();
      if (!typed) return picker.input.focus();
      // A listed schedule is written as it is listed; any other id is the person's own and is written as typed.
      const hit = listedIn('schedulers').find((item) => namedBy(item, typed));
      picker.input.value = '';
      const now = await currentList();
      // The section is redrawn with the new schedule in it: focus goes to its type, its first row.
      return writeList([...now, { type: hit ? shownName(hit) : typed }], { keys: [pathKey([...listPath, now.length, 'type'])] }, 'The schedule was not added');
    };
    const picker = createCombobox({
      id: addId,
      inputClass: 'input--path',
      placeholder: 'Choose a schedule',
      closed: false,
      listLabel: `Schedulers for use inside ${holderName}`,
      attrs: { 'data-mark': addMark },
      getChoices: () => toChoices(partsOf(scheduleEntry(holder)).map((item) => shownName(item))),
      onEnter: add,
      onEscape: () => {
        picker.input.value = '';
      },
    });
    const removeList = exists
      ? h(
          'button',
          {
            class: 'btn btn--sm btn--quiet',
            type: 'button',
            'aria-label': 'Remove schedulers',
            onclick: async () => {
              if (list.length) {
                const ok = await confirmDialog({ title: 'Remove schedulers?', body: `${listKey} and every schedule in it are taken out of the recipe. Changes can bring it back.`, confirmLabel: 'Remove schedulers', danger: true });
                if (!ok) return;
              }
              ctx.remove(listPath).catch(refusedToast('schedulers was not removed'));
            },
          },
          'Remove',
        )
      : null;
    return h(
      'div',
      { class: 'form__nested' },
      h('div', { class: 'form__nestedhead' }, h('p', { class: 'form__nestedtitle mono', dataset: { path: listKey } }, 'schedulers'), removeList),
      blockTarget(listPath),
      list.length ? parts : h('p', { class: 'form__empty' }, `No schedule is listed inside ${holderName} yet.`),
      h(
        'div',
        { class: 'form__actions' },
        h('label', { class: 'visually-hidden', for: addId }, 'Schedule to add'),
        picker.el,
        h('button', { class: 'btn btn--sm', type: 'button', onclick: add }, 'Add schedule'),
      ),
      list.length ? h('p', { class: 'fld__desc form__help' }, 'Adding or moving a schedule replaces the list and removes comments inside it.') : null,
    );
  }

  /** A block whose options come with its type: an optimizer, a schedule, a metric, a network. */
  const typedBlock = (path) => Boolean(blockListName(path)) || isNetworkBlock(path);

  /**
   * What "Add option" offers for the network block at `path`: the options the schema lists for its type,
   * each with the value it starts from and the values it takes. null: nothing known. An optimizer, a
   * schedule and a metric show their options as rows instead (unsetParameters), so their "Add option"
   * takes the names the catalog does not list.
   */
  function knownOptionsAt(path) {
    if (!isNetworkBlock(path)) return null;
    return {
      options: blockParameters(path).map((p) => ({ name: p.name, hasDefault: 'default' in p && p.default !== undefined, default: p.default, choices: Array.isArray(p.choices) ? p.choices : null })),
    };
  }
  // The editor shows the name list only where one can exist.
  knownOptionsAt.applies = isNetworkBlock;

  function group(id, title, rows, stateText = '', { mono = false } = {}) {
    const stateEl = h('span', { class: 'form__groupstate' }, stateText);
    const issueSlot = h('span', { class: 'form__groupissues' });
    const details = h(
      'details',
      {
        class: 'form__group',
        open: openGroups.has(id),
        ontoggle: () => (details.open ? openGroups.add(id) : openGroups.delete(id)),
      },
      h('summary', { class: 'form__summary' }, h('span', { class: `form__grouptitle${mono ? ' mono' : ''}` }, title), issueSlot, stateEl),
      h('div', { class: 'form__groupbody' }, rows),
    );
    groupSlots.set(details, issueSlot);
    return { el: details, stateEl };
  }


  /** One schema property: a field row, a group of sub-fields, or a generic editor. */
  function renderProp(base, key, prop, depth = 0) {
    const path = [...base, key];
    const kind = schema.kind(prop);
    // A block whose options come with its type (an optimizer, a schedule, the schedules per optimizer, a
    // network): the schema names its `type` at most, and the lists above name the rest. It is edited as
    // the free-form block it is, whatever shape the schema writes that object in.
    if ((kind.kind === 'object' || kind.kind === 'json') && (typedBlock(path) || pathKey(path) === 'train.schedulers')) return genericBlock(path, key, kind.description);
    if (kind.kind === 'object' && depth < 2) {
      const props = schema.props(kind.node);
      const names = Object.keys(props);
      const value = getAt(config(), path);
      const unknown = isPlainObject(value) ? Object.keys(value).filter((k) => !(k in props)) : [];
      const id = `obj:${pathKey(path)}`;
      // An object the schema writes out in place without closing it (io_backend) takes keys the schema
      // does not name: those the recipe has are rows, and "Add option" adds one.
      const inPlace = !prop.$ref && ![...(Array.isArray(prop.anyOf) ? prop.anyOf : []), ...(Array.isArray(prop.oneOf) ? prop.oneOf : [])].some((m) => m && m.$ref);
      const adder = inPlace && kind.node.additionalProperties !== false ? addOptionRow(path, ctx, null, () => Object.keys(isPlainObject(getAt(config(), path)) ? getAt(config(), path) : {})) : null;
      const g = group(
        id,
        key,
        [kind.description ? h('p', { class: 'fld__desc form__groupdesc' }, kind.description) : null, blockTarget(path), names.map((k) => renderProp(path, k, props[k], depth + 1)), unknown.map((k) => unknownRow([...path, k], k)), adder],
        countText(setCount(path, names, config())),
        { mono: true },
      );
      updaters.push((c) => {
        g.stateEl.textContent = countText(setCount(path, names, c));
      });
      return g.el;
    }
    if (['object', 'freeObject', 'map', 'union'].includes(kind.kind)) return genericBlock(path, key, kind.description);
    return leaf(path, key, kind);
  }

  /** A key the schema does not know: control from its value. */
  function unknownRow(path, key) {
    const value = getAt(config(), path);
    if (isPlainObject(value)) return genericBlock(path, key, '', true);
    return leaf(path, key, kindFromValue(value), true);
  }

  /** Rows for the properties of an object option (path.*, val.*, logger.*), unknown keys last. */
  function objectRows(base, prop, first = [], exclude = () => false) {
    const props = schema.props(prop);
    const names = orderKeys(Object.keys(props).filter((k) => !exclude(k)), first);
    const value = getAt(config(), base);
    const unknown = isPlainObject(value) ? Object.keys(value).filter((k) => !(k in props)) : [];
    return [...names.map((k) => renderProp(base, k, props[k])), ...unknown.map((k) => unknownRow([...base, k], k))];
  }

  /**
   * "Starting weights for RDUAL": the three ways a RDUAL training starts, and the two files they can name. It
   * does not say which case applies: nothing the service answers tells a complete checkpoint from another.
   */
  function rdualGuide() {
    const pick = (key, title) => async () => {
      const { pickPath } = await import('../components/path-picker.js');
      const current = getAt(config(), ['path', key]);
      const picked = await pickPath({ title, mode: 'file', start: typeof current === 'string' ? current : '', extensions: ['.safetensors', '.pth'], confirmLabel: 'Use this file' });
      if (typeof picked !== 'string' || !picked) return;
      ctx.apply(['path', key], picked).catch(refusedToast(`${key} was not changed`));
    };
    return h(
      'div',
      { class: 'notice form__rdual' },
      h(
        'div',
        { class: 'notice__body' },
        h('p', { class: 'form__rdualtitle' }, h('strong', null, 'Starting weights for RDUAL')),
        h(
          'ul',
          { class: 'form__rduallist' },
          RDUAL_CASES.map(([head, ...rest]) =>
            h('li', { class: 'form__rdualcase' }, h('span', { class: 'form__rdualhead' }, head), rest.map((part) => (/^`.*`$/.test(part) ? h('code', { class: 'mono' }, part.slice(1, -1)) : part))),
          ),
        ),
        h('p', { class: 'form__rdualnote' }, RDUAL_NOTE),
        h(
          'div',
          { class: 'row form__rdualacts' },
          h('button', { class: 'btn btn--sm', type: 'button', onclick: pick('pretrain_network_g', 'Choose the generator checkpoint') }, 'Choose generator checkpoint…'),
          h('button', { class: 'btn btn--sm', type: 'button', onclick: pick('neuflow_weights', 'Choose the NeuFlow v2 weights') }, 'Choose NeuFlow weights…'),
        ),
      ),
    );
  }

  const defaultOf = (key) => (rootProps[key] ? schema.kind(rootProps[key]).default : undefined);
  const rootKeysOf = (id, order = []) => orderKeys([...layout.sectionOf].filter(([, s]) => s === id).map(([k]) => k), order);

  // ───────── sections ─────────

  const builders = {
    basics: buildBasics,
    speed: () => rootKeysOf('speed', SPEED).map((k) => renderProp([], k, rootProps[k])),
    degradations: buildDegradations,
    datasets: buildDatasets,
    network: buildNetwork,
    pretrained: () => [isRdual() ? rdualGuide() : null, ...objectRows(['path'], rootProps.path, PATH_FIRST)],
    training: buildTraining,
    losses: buildLosses,
    validation: () => objectRows(['val'], rootProps.val),
    logging: () => objectRows(['logger'], rootProps.logger),
    other: buildOther,
  };

  function buildBasics() {
    const rows = [];
    const keys = rootKeysOf('basics', BASICS);
    if (!keys.includes('task')) rows.push(taskRow(), ...taskOptionsRows());
    for (const k of keys) {
      if (k === 'task_options') continue;
      if (k === 'task') rows.push(taskRow(), ...taskOptionsRows());
      else rows.push(renderProp([], k, rootProps[k]));
    }
    return rows;
  }

  /** Warnings from comparing the recipe with the catalog entry of its task (not check results). */
  function taskWarnings(c) {
    const id = currentTask(c);
    const entry = taskEntry(id);
    if (!entry) return [];
    const out = [];
    const nf = netField(c);
    const type = getAt(c, [nf, 'type']);
    if (typeof type === 'string' && type) {
      const { ids } = tasks.networksFor(id);
      if (!ids.some((x) => x.toLowerCase() === type.toLowerCase())) out.push(`The network ${type} is not listed for ${entry.label}.`);
    }
    const listed = Array.isArray(entry.datasetTypes) ? entry.datasetTypes : [];
    if (listed.length && isPlainObject(c.datasets)) {
      for (const [key, ds] of Object.entries(c.datasets)) {
        const t = isPlainObject(ds) ? ds.type : undefined;
        if (typeof t === 'string' && !listed.includes(t)) out.push(`Dataset ${key} has the type ${t}, which is not listed for ${entry.label} (it lists ${listed.join(', ')}).`);
      }
    }
    return out;
  }

  /**
   * task: the catalog's tasks in a list, or "Decided from the network" (no task key; the trainer infers
   * it). Another task id can be typed, as the YAML would take it; the check says whether the trainer has it.
   */
  function taskRow() {
    const c = config();
    const explicit = typeof c.task === 'string' ? c.task : null;
    const id = 'form-task';
    const INFER = 'Decided from the network';
    const list = catalog ? catalog.tasks.filter((t) => t && typeof t.id === 'string') : [];
    let shownText = explicitTask(c) ?? '';
    const tag = h('span', { class: 'fld__tag' }, explicit !== null ? 'Set' : 'Not set');
    const decided = h('p', { class: 'fld__note', role: 'status' });
    const only = h('p', { class: 'fld__note', hidden: true });
    const unlisted = h('p', { class: 'fld__note', hidden: true });
    const warnings = h('div', { class: 'fld__issues' });
    const issueSlot = h('div', { class: 'fld__issues' });
    const show = () => {
      shownText = explicitTask(config()) ?? '';
      input.value = shownText;
    };
    // Enter, then the blur caused by a confirm dialog, both fire: one change at a time.
    let busy = false;
    const commit = async (next) => {
      if (busy) return;
      const now = config();
      if (next === (explicitTask(now) ?? '')) return show();
      busy = true;
      try {
        let edits;
        if (!next) {
          edits = [{ path: ['task'], remove: true }];
          if ('task_options' in now) {
            const ok = await confirmDialog({
              title: 'Remove the task options too?',
              body: 'The trainer refuses task_options without an explicit task, so leaving the task to the network also removes task_options from the recipe.',
              confirmLabel: 'Remove task and task options',
              danger: true,
            });
            if (!ok) return show();
            edits.push({ path: ['task_options'], remove: true });
          }
        } else edits = [{ path: ['task'], value: next }];
        shownText = next;
        input.value = next;
        await recipe.applyEdits(edits).catch((error) => {
          toast(`The task was not changed: ${error.message}`, { kind: 'err' });
          show();
        });
      } finally {
        busy = false;
      }
    };
    /** What the typed text asks for: '' for no task, a listed id as it is listed, else the text itself. */
    const typed = () => {
      const text = input.value.trim();
      if (!text || text === INFER) return '';
      return list.find((t) => t.id.toLowerCase() === text.toLowerCase())?.id ?? text;
    };
    const combo = createCombobox({
      id,
      inputClass: 'input--path',
      placeholder: INFER,
      closed: false,
      listLabel: 'Tasks',
      getChoices: () => toChoices([{ unset: true, text: INFER }, ...list.map((t) => ({ value: t.id, note: typeof t.label === 'string' ? t.label : null }))]),
      current: () => shownText,
      attrs: { 'data-fld': 'task' },
      onPick: (choice) => commit(choice.unset ? '' : choice.value),
      onEnter: () => commit(typed()),
      onEscape: () => {
        input.value = shownText;
      },
    });
    const { input } = combo;
    input.value = shownText;
    input.addEventListener('blur', () => commit(typed()));
    const paintInfo = () => {
      const now = config();
      const task = explicitTask(now);
      if (task) decided.textContent = '';
      else {
        const t = checkedTask();
        decided.textContent = t ? `Decided from the network: ${tasks.taskLabel(t)} (${t}).` : 'The check has not decided a task yet.';
      }
      decided.hidden = !decided.textContent;
      const entry = taskEntry(currentTask(now));
      only.hidden = !entry?.trainingOnly;
      only.textContent = entry?.trainingOnly ? `The catalog marks ${entry.label} as training only.` : '';
      // A task the installed trainer does not list is still the recipe's: said here, judged by the check.
      const strange = Boolean(catalog && task && list.length && !list.some((t) => t.id === task));
      unlisted.textContent = strange ? `Not one of the ${list.length} tasks the installed trainer lists.` : '';
      unlisted.hidden = !strange;
      warnings.replaceChildren(...taskWarnings(now).map((text) => h('p', { class: 'fld__warn' }, 'Compared with the catalog: ', text)));
    };
    const desc = rootProps.task ? schema.kind(rootProps.task).description : '';
    const el = h(
      'div',
      { class: 'fld', dataset: { path: 'task' } },
      h('div', { class: 'fld__label' }, h('div', { class: 'fld__head' }, h('label', { class: 'fld__key', for: id }, 'task'), tag), desc ? h('p', { class: 'fld__desc' }, desc) : null),
      h('div', { class: 'fld__control' }, combo.el, decided, only, unlisted, warnings, catalog ? null : catalogState(input), issueSlot),
    );
    paintInfo();
    updaters.push(paintInfo);
    register({
      el,
      path: ['task'],
      key: 'task',
      pathKey: 'task',
      update: () => {
        if (document.activeElement !== input && !busy) show();
      },
      setIssues: (issues) => {
        issueSlot.replaceChildren(...issues.map((i) => h('p', { class: i.level === 'error' ? 'field__error' : 'fld__warn' }, i.level === 'error' ? 'Error: ' : 'Warning: ', i.message)));
        if (issues.some((i) => i.level === 'error')) input.setAttribute('aria-invalid', 'true');
        else input.removeAttribute('aria-invalid');
      },
      focus: () => input.focus({ preventScroll: true }),
    });
    return el;
  }

  /** Under the task field while the catalog is not there: loading, or why it failed and "Try again". */
  function catalogState(select) {
    const state = tasks.get();
    if (state.phase !== 'error') return h('p', { class: 'fld__note', role: 'status' }, 'The task list appears when the catalog has loaded.');
    return h(
      'div',
      { class: 'fld__line' },
      h('p', { class: 'fld__warn', role: 'status' }, `The task list could not be loaded: ${String(state.error ?? 'no reason was given').replace(/[.\s]+$/, '')}.`),
      h(
        'button',
        {
          class: 'btn btn--sm',
          type: 'button',
          onclick: () => {
            // The field survives the rebuild that follows (it has data-fld), so focus goes there.
            select.focus({ preventScroll: true });
            loadCatalog();
          },
        },
        'Try again',
      ),
    );
  }

  /** task_options: drawn from the selected task's own schema; flagged when the task has none. */
  function taskOptionsRows() {
    const c = config();
    const id = explicitTask(c);
    const entry = taskEntry(id);
    const value = c.task_options;
    const has = 'task_options' in c;
    const title = (extra) =>
      h('div', { class: 'form__blockhead' }, h('h3', { class: 'form__blocktitle', dataset: { path: 'task_options' } }, 'Task options', h('span', { class: 'mono muted' }, 'task_options'), extra));
    if (entry?.taskOptions && typeof entry.taskOptions === 'object') {
      const ts = makeSchema(entry.taskOptions);
      const props = ts.props(ts.rootNode);
      const rows = [];
      for (const [k, prop] of Object.entries(props)) {
        const kind = ts.kind(prop);
        const path = ['task_options', k];
        if (kind.kind === 'list' && kind.itemType === 'string') {
          const itemLabel = k === 'class_names' ? (i) => `ID ${i}` : null;
          rows.push(register(createField({ path, key: k, kind, value: getAt(c, path), isSet: hasAt(c, path), listEditor: true, itemLabel, ...fieldExtras(path, k, kind) }, ctx)).el);
        } else if (['object', 'freeObject', 'map'].includes(kind.kind)) rows.push(genericBlock(path, k, kind.description));
        else rows.push(leaf(path, k, kind));
      }
      for (const k of isPlainObject(value) ? Object.keys(value).filter((x) => !(x in props)) : []) rows.push(unknownRow(['task_options', k], k));
      return [h('div', { class: 'form__block' }, title(null), h('p', { class: 'fld__desc form__groupdesc' }, `Options of ${entry.label}, from the task's own schema in the catalog.`), blockTarget(['task_options']), rows)];
    }
    if (!has) return [];
    const reason = !catalog ? null : !id ? 'The trainer refuses task_options without an explicit task.' : entry ? `${entry.label} has no task options.` : `The catalog does not describe task options for ${id}.`;
    const remove = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => ctx.remove(['task_options']).catch(refusedToast('task_options was not removed')) }, 'Remove task options');
    return [
      h(
        'div',
        { class: 'form__block' },
        title(reason ? h('span', { class: 'fld__tag fld__tag--unknown' }, 'Not used by this task') : null),
        reason ? h('p', { class: 'fld__warn' }, reason) : null,
        h('div', { class: 'form__actions' }, remove),
        blockTarget(['task_options']),
        isPlainObject(value) ? genericEditor({ path: ['task_options'], value, ctx, register }) : leaf(['task_options'], 'task_options', kindFromValue(value), false, { removable: true }),
      ),
    ];
  }

  function buildDegradations() {
    const rows = [];
    if (rootProps[DEGRADATION_SWITCH]) rows.push(renderProp([], DEGRADATION_SWITCH, rootProps[DEGRADATION_SWITCH]));
    for (const g of layout.groups) {
      const built = group(`deg:${g.id}`, g.title, g.keys.map((k) => renderProp([], k, rootProps[k])), degradationState(g, config(), defaultOf));
      updaters.push((c) => {
        built.stateEl.textContent = degradationState(g, c, defaultOf);
      });
      rows.push(built.el);
    }
    return rows;
  }

  /**
   * True when the recipe started from a template ("From template …") and the check reports one of its
   * dataset folders or files (dataroot_gt, dataroot_lq, the task's file option) as not found.
   */
  function templatePathsMissing() {
    const s = recipe.get();
    if (!/^template /.test(s.baseLabel ?? '')) return false;
    const fileKey = folderWords(currentTask()).file?.key;
    return (Array.isArray(s.issues) ? s.issues : []).some((issue) => {
      if (issue?.level !== 'error' || !MISSING_PATH.test(String(issue.message ?? ''))) return false;
      const p = parsePath(issue.path);
      return p[0] === 'datasets' && (p[2] === 'dataroot_gt' || p[2] === 'dataroot_lq' || (Boolean(fileKey) && p[2] === fileKey));
    });
  }

  function buildDatasets() {
    const datasets = config().datasets;
    const dsKind = schema.kind(rootProps.datasets);
    const dsProps = dsKind.kind === 'map' ? schema.props(dsKind.item) : {};
    const blocks = [];
    // One line for the whole section, not one per field: the errors themselves stay on their fields.
    const pointer = h(
      'p',
      { class: 'form__pointer', role: 'status', hidden: true },
      "These paths are the template's examples. ",
      h('a', { href: '#/data' }, 'Choose your own data on the Data screen'),
      '.',
    );
    const paintPointer = () => {
      pointer.hidden = !templatePathsMissing();
    };
    paintPointer();
    updaters.push(paintPointer);
    for (const [key, value] of Object.entries(isPlainObject(datasets) ? datasets : {})) {
      const base = ['datasets', key];
      const v = isPlainObject(value) ? value : {};
      // A task that reads only dataroot_lq (paired: false) does not ask for a second folder: an unset
      // dataroot_gt waits with the other unset options under "More dataset options".
      const words = folderWords(currentTask());
      const single = words.paired === false;
      // A task's file option (detection's annotation_file) comes right after the image folder.
      const taskOrder = words.file ? DATASET_FIRST.flatMap((k) => (k === 'dataroot_lq' ? [k, words.file.key] : [k])) : DATASET_FIRST;
      // The dataset type that reads folders or ZIP archives: its storage is asked for right after its folders.
      const order = readsZip(base) ? taskOrder.flatMap((k) => (k === 'dataroot_lq' ? [k, 'io_backend'] : [k])) : taskOrder;
      const first = order.filter((k) => (k in dsProps || k in v) && !(single && k === 'dataroot_gt' && !(k in v)));
      const setRest = Object.keys(v).filter((k) => !first.includes(k));
      const more = Object.keys(dsProps).filter((k) => !first.includes(k) && !(k in v));
      const row = (k) => (k in dsProps ? renderProp(base, k, dsProps[k]) : unknownRow([...base, k], k));
      // Video clips: the panel checks and prepares them; the paired image check does not apply.
      const video = isVideoDataset(base, config());
      let videoHost = null;
      if (video && key === 'train') {
        videoHost = h('div', { class: 'form__video' });
        mountVideoPrep(videoHost);
      }
      // The Data screen checks folders: archives are left to the recipe check.
      const check = !video && !isZipDataset(base) && toList(v.dataroot_gt).length
        ? h(
            'button',
            {
              class: 'btn btn--sm',
              type: 'button',
              onclick: () => {
                // Read the folders now: they may have been edited since this block was built.
                const ds = getAt(config(), base) ?? {};
                const gt = toList(ds.dataroot_gt);
                const lq = toList(ds.dataroot_lq);
                if (!gt.length) return;
                const scale = config().scale;
                data.setInputs({ gt: gt[0], lq: lq[0] ?? '', scale: Number.isInteger(scale) ? scale : null });
                data.scan();
                go('#/data');
              },
            },
            'Check these folders',
          )
        : null;
      blocks.push(
        h(
          'div',
          { class: 'form__block' },
          h(
            'div',
            { class: 'form__blockhead' },
            h('h3', { class: 'form__blocktitle', dataset: { path: pathKey(base) } }, h('span', { class: 'mono' }, key), typeof v.name === 'string' && v.name !== key ? h('span', { class: 'muted' }, v.name) : null),
            check,
            h(
              'button',
              {
                class: 'btn btn--sm btn--quiet',
                type: 'button',
                onclick: async () => {
                  const ok = await confirmDialog({ title: `Remove the dataset ${key}?`, body: `datasets.${key} and every option in it are taken out of the recipe. Changes can bring it back.`, confirmLabel: 'Remove dataset', danger: true });
                  if (!ok) return;
                  ctx.remove(base).catch(refusedToast('The dataset was not removed'));
                },
              },
              'Remove dataset',
            ),
          ),
          blockTarget(base),
          videoHost,
          first.map(row),
          setRest.map(row),
          more.length ? group(`ds:${key}:more`, 'More dataset options', more.map(row), `${more.length} at their defaults`).el : null,
        ),
      );
    }
    if (!blocks.length) blocks.push(h('p', { class: 'form__empty' }, 'This recipe has no datasets yet. Add one for training, then point it at your image folders.'));
    const add = h(
      'div',
      { class: 'form__actions' },
      h(
        'button',
        {
          class: 'btn btn--sm',
          type: 'button',
          onclick: async () => {
            const existing = isPlainObject(config().datasets) ? Object.keys(config().datasets) : [];
            const key = await promptDialog({
              title: 'Add dataset',
              label: 'Dataset key',
              value: existing.includes('train') ? (existing.includes('val') ? '' : 'val') : 'train',
              help: 'The key under datasets in the YAML, for example train or val.',
              confirmLabel: 'Add dataset',
              validate: (text) => (!text ? 'Enter a key.' : /[\s.[\]/]/.test(text) ? 'A key cannot contain spaces, dots, slashes or brackets.' : existing.includes(text) ? 'This recipe already has a dataset with that key.' : null),
            });
            if (!key) return;
            // The task's first listed dataset type when it lists any (imagemaskdataset for segmentation).
            const listed = taskEntry(currentTask())?.datasetTypes;
            const type = Array.isArray(listed) && typeof listed[0] === 'string' ? listed[0] : 'pairedimagedataset';
            recipe.applyEdits([{ path: ['datasets', key], value: { name: key, type } }]).catch((error) => toast(`The dataset was not added: ${error.message}`, { kind: 'err' }));
          },
        },
        'Add dataset',
      ),
    );
    return [pointer, ...blocks, undoOffer(['datasets']), add];
  }

  // ───────── network ─────────

  function archIds() {
    const fromCatalog = Array.isArray(catalog?.architectures) ? catalog.architectures.map((a) => a?.id).filter((id) => typeof id === 'string') : [];
    if (fromCatalog.length) return fromCatalog;
    return netKind?.refs ? netKind.refs.flatMap((ref) => variantTags(schema.lookup(ref), netKind.propertyName)).filter((t) => typeof t === 'string') : [];
  }

  /** Networks offered for the current task, recommended first; `all` is every installed network. */
  function networkChoices() {
    const all = archIds();
    if (!catalog || !catalog.tasks.length) return { ids: all, recommended: new Set(), all, task: null };
    const task = currentTask();
    const { ids, recommended } = tasks.networksFor(task);
    return { ids, recommended, all: [...new Set([...ids, ...all])], task: taskEntry(task) };
  }

  /**
   * The network type picker. A type the schema describes replaces the block, after a confirm when options
   * the new type does not have would drop. A type it does not describe (a network added to the trainer,
   * an id typed by hand) changes the type alone: nothing is known about its options, so all of them stay.
   */
  function typePicker(nf) {
    const { ids, recommended, all, task } = networkChoices();
    const listed = (v) => ids.some((x) => x.toLowerCase() === String(v).toLowerCase());
    const onPick = async (match) => {
      const had = isPlainObject(config()[nf]);
      const old = had ? config()[nf] : {};
      const def = schema.variantFor(netKind, match);
      const props = def ? schema.props(def) : {};
      const kept = {};
      const dropped = [];
      for (const [k, val] of Object.entries(old)) {
        if (k === 'type') continue;
        // Without a description of the new type no option can be called foreign to it.
        if (!def || k in props) kept[k] = val;
        else dropped.push(k);
      }
      if (dropped.length) {
        const ok = await confirmDialog({
          title: `Change the network to ${match}?`,
          body: h(
            'div',
            { class: 'stack' },
            h('p', null, `${match} does not have ${dropped.length === 1 ? 'this option' : 'these options'}, so ${dropped.length === 1 ? 'it is' : 'they are'} removed from the recipe:`),
            h('p', { class: 'mono' }, dropped.join(', ')),
            h('p', null, 'Options both networks have keep their values.'),
          ),
          confirmLabel: 'Change network',
        });
        if (!ok) return false;
      }
      // With nothing to drop only the type line changes, so the block keeps its layout and comments.
      const edit = had && !dropped.length ? { path: [nf, 'type'], value: match } : { path: [nf], value: { type: match, ...kept } };
      return recipe.applyEdits([edit]).then(
        () => true,
        (err) => {
          toast(`The network was not changed: ${err.message}`, { kind: 'err' });
          return false;
        },
      );
    };
    const field = typeField({
      id: 'form-network-type',
      path: [nf, 'type'],
      ids,
      accept: all,
      listLabel: 'Network types',
      restNote: task ? 'not listed for this task' : null,
      getValue: () => getAt(config(), [nf, 'type']),
      description: !ids.length ? '' : task ? `${ids.length} network types are available for ${task.label}${recommended.size ? `, ${recommended.size} of them recommended` : ''}.` : `${ids.length} network types are available.`,
      onPick,
      optionLabel: (x) => (recommended.has(x) ? 'recommended for this task' : null),
      note: (v) => {
        if (typeof v !== 'string' || !v) return null;
        if (all.length && !all.some((x) => x.toLowerCase() === v.toLowerCase())) return `Not one of the ${all.length} network types the installed trainer lists.`;
        if (!task) return null;
        if (!listed(v)) return `${NOT_LISTED}.`;
        return [...recommended].some((x) => x.toLowerCase() === v.toLowerCase()) ? 'Recommended for this task.' : null;
      },
    });
    return register(field).el;
  }

  function buildNetwork() {
    const c = config();
    const nf = netField(c);
    const ng = isPlainObject(c[nf]) ? c[nf] : {};
    const def = typeof ng.type === 'string' ? schema.variantFor(netKind, ng.type) : null;
    const rows = [subHeading([nf], nf, rootProps[nf] ? schema.kind(rootProps[nf]).description : ''), blockTarget([nf]), typePicker(nf)];
    if (def) {
      const props = schema.props(def);
      for (const k of Object.keys(props)) if (k !== 'type') rows.push(renderProp([nf], k, props[k]));
      for (const k of Object.keys(ng)) if (k !== 'type' && !(k in props)) rows.push(unknownRow([nf, k], k));
    } else if (typeof ng.type === 'string') {
      rows.push(h('p', { class: 'form__empty' }, `The schema does not describe the network type ${ng.type}, so its options are shown as they are in the recipe.`));
      rows.push(genericEditor({ path: [nf], value: Object.fromEntries(Object.entries(ng).filter(([k]) => k !== 'type')), ctx, register }));
    }
    if (typeof ng.type === 'string' && ng.type) rows.push(listParameters());

    const ndDesc = rootProps.network_d ? schema.kind(rootProps.network_d).description : '';
    // A discriminator without a type asks for it first, like every typed block (blockSpecial.first).
    if (isPlainObject(c.network_d)) rows.push(genericBlock(['network_d'], 'network_d', ndDesc));
    else {
      rows.push(
        addNetwork('network_d', {
          noun: 'discriminator',
          empty: 'No discriminator. Recipes that use a GAN loss need one.',
          listed: 'the installed types that fit a discriminator',
          example: 'unetdiscriminatorsn',
          names: () => discriminatorsFirst(listedIn('architectures')).filter(fitsDiscriminator).map((item) => item.id),
        }),
      );
    }
    for (const key of ['network_g', 'network_g_teacher', 'network_ae']) {
      if (key !== nf && key in c) rows.push(genericBlock([key], key, rootProps[key] ? schema.kind(rootProps[key]).description : ''));
      else if (key === 'network_g_teacher' && nf === 'network_g' && wantsTeacher(c)) {
        rows.push(
          addNetwork(key, {
            noun: 'teacher network',
            empty: 'No teacher network. Distillation needs one, with its weights in path.pretrain_network_g_teacher.',
            listed: 'the installed types for this task',
            // Without the discriminators: every other installed type for the task stays listed.
            names: () => networkChoices().ids.filter((id) => !fitsDiscriminator({ id })),
          }),
        );
      } else if (key !== nf) {
        // A network that leaves no place behind when it is removed can still be put back.
        rows.push(undoOffer([key]));
      }
    }
    return rows;
  }

  /**
   * The place of a network the recipe does not have yet (network_d, network_g_teacher): a line on what it
   * is for and a button that asks for its type. `names()` are the installed types listed for it (`listed`
   * says which those are); any other type can be typed.
   */
  function addNetwork(key, { noun, empty, listed, names, example = '' }) {
    const add = async () => {
      const ids = names();
      const typed = await promptDialog({
        title: `Add ${noun}`,
        label: `${noun.charAt(0).toUpperCase()}${noun.slice(1)} type`,
        help: `The network type of the ${noun}, as the trainer names it${ids.length ? `. The list holds ${listed}; another type can be typed.` : example ? `, for example ${example}.` : '.'}`,
        confirmLabel: `Add ${noun}`,
        choices: ids.length ? ids : null,
      });
      if (!typed) return;
      // A listed type is written as it is listed; any other is the person's own and is written as typed.
      const hit = listedIn('architectures').find((item) => namedBy(item, typed));
      // This button goes away once the network exists: focus goes to its type.
      focusAfter = { config: recipe.get().config, keys: [pathKey([key, 'type'])], at: Date.now() };
      recipe.applyEdits([{ path: [key], value: { type: hit ? hit.id : typed } }]).catch((error) => {
        focusAfter = null;
        toast(`The ${noun} was not added: ${error.message}`, { kind: 'err' });
      });
    };
    return h(
      'div',
      { class: 'form__sub' },
      subHeading([key], key, rootProps[key] ? schema.kind(rootProps[key]).description : ''),
      undoOffer([key]),
      h('p', { class: 'form__empty' }, empty),
      // The mark: where focus goes when the network is removed.
      h('div', { class: 'form__actions' }, h('button', { class: 'btn btn--sm', type: 'button', dataset: { mark: `${key}:add` }, onclick: add }, `Add ${noun}`)),
    );
  }

  // ───────── training and losses ─────────

  function buildTraining() {
    const trainProps = schema.props(rootProps.train);
    const train = isPlainObject(config().train) ? config().train : {};
    const skip = (k) => k === 'losses' || TRAIN_OBJECTS.includes(k) || isLegacyLossKey(k);
    const rows = objectRows(['train'], rootProps.train, TRAIN_FIRST, skip);
    for (const key of TRAIN_OBJECTS) {
      if (key in trainProps || key in train) rows.push(genericBlock(['train', key], key, trainProps[key] ? schema.kind(trainProps[key]).description : ''));
    }
    const legacy = Object.keys(trainProps).filter((k) => isLegacyLossKey(k) && k in train);
    if (legacy.length) {
      rows.push(h('h3', { class: 'form__subtitle' }, 'Older loss options'));
      rows.push(h('p', { class: 'fld__desc form__groupdesc' }, 'Loss settings in the older format. The losses list in the Losses section is the current way to set them.'));
      for (const k of legacy) rows.push(renderProp(['train'], k, trainProps[k]));
    }
    return rows;
  }

  function lossIds() {
    const fromCatalog = Array.isArray(catalog?.losses) ? catalog.losses.map((l) => l?.id).filter((id) => typeof id === 'string') : [];
    if (fromCatalog.length) return fromCatalog;
    return lossKind?.refs ? lossKind.refs.flatMap((ref) => variantTags(schema.lookup(ref), lossKind.propertyName)).filter((t) => typeof t === 'string') : [];
  }

  function buildLosses() {
    const list = getAt(config(), ['train', 'losses']);
    const losses = Array.isArray(list) ? list : [];
    const rows = [];
    losses.forEach((loss, i) => {
      const base = ['train', 'losses', i];
      const v = isPlainObject(loss) ? loss : {};
      const type = typeof v.type === 'string' ? v.type : '';
      const def = type ? schema.variantFor(lossKind, type) : null;
      const props = def ? schema.props(def) : {};
      const off = h('span', { class: 'badge', hidden: v.loss_weight !== 0 }, 'Off (weight 0)');
      updaters.push((c) => {
        off.hidden = getAt(c, [...base, 'loss_weight']) !== 0;
      });
      // Managed VOSR assets need no paths. Keep explicit overrides editable in older recipes.
      const others = Object.keys(props).filter((k) =>
        k !== 'type' && k !== 'loss_weight' &&
        (type !== 'vosrloss' || (k !== 'engine_dir' && k !== 'plugin') || k in v));
      const unknown = Object.keys(v).filter((k) => k !== 'type' && k !== 'loss_weight' && !(k in props));
      const weightKind = props.loss_weight ? schema.kind(props.loss_weight) : { ...kindFromValue(v.loss_weight ?? 0), kind: 'number' };
      const remove = h(
        'button',
        {
          class: 'btn btn--sm btn--quiet',
          type: 'button',
          onclick: async () => {
            const ok = await confirmDialog({ title: `Remove ${type || 'this loss'}?`, body: 'The loss and its options are taken out of the losses list. The other losses and their comments stay as they are.', confirmLabel: 'Remove loss', danger: true });
            if (!ok) return;
            // Deletes only this item's lines; the rest of the list keeps its layout.
            ctx.remove(base).catch(refusedToast('The loss was not removed'));
          },
        },
        'Remove',
      );
      // A loss the schema does not describe (one added to the trainer, an id typed by hand): its options
      // are the recipe's own keys, each editable, and "Add option" adds one. Nothing is hidden or dropped.
      const custom = Boolean(type) && !def;
      const opts = custom
        ? null
        : group(
            `loss:${i}:${type}`,
            'Options',
            [...others.map((k) => renderProp(base, k, props[k])), ...unknown.map((k) => unknownRow([...base, k], k))],
            countText(setCount(base, others, config())),
          );
      if (opts) {
        updaters.push((c) => {
          opts.stateEl.textContent = countText(setCount(base, others, c));
        });
      }
      const own = custom
        ? [
            h('p', { class: 'form__empty' }, `The schema does not describe the loss type ${type}, so its options are shown as they are in the recipe.`),
            genericEditor({ path: base, value: Object.fromEntries(Object.entries(v).filter(([k]) => k !== 'type' && k !== 'loss_weight')), ctx, register }),
          ]
        : null;
      rows.push(
        h(
          'div',
          { class: 'form__block' },
          h(
            'div',
            { class: 'form__blockhead' },
            h('h3', { class: 'form__blocktitle', dataset: { path: pathKey(base) } }, h('span', { class: 'mono' }, type || 'No type'), off, def || !type ? null : h('span', { class: 'fld__tag fld__tag--unknown' }, 'Not in the schema')),
            remove,
          ),
          blockTarget(base),
          type === 'vosrloss'
            ? h('p', { class: 'fld__desc' }, 'On an RTX 5090 with TensorRT 11.4.0.106, the engines and plugin download automatically on first use.')
            : null,
          leaf([...base, 'loss_weight'], 'loss_weight', weightKind),
          own ?? (others.length || unknown.length ? opts.el : null),
        ),
      );
    });
    if (!rows.length) rows.push(h('p', { class: 'form__empty' }, 'This recipe has no losses. Add at least one so training has something to minimize.'));

    const selectId = 'form-add-loss';
    const addError = h('p', { class: 'field__error', role: 'alert', hidden: true });
    const sayAddError = (message) => {
      addError.textContent = message ?? '';
      addError.hidden = !message;
    };
    const addLoss = () => {
      const typed = picker.input.value.trim();
      if (!typed) {
        sayAddError(null);
        return picker.input.focus();
      }
      // A listed loss is written as it is listed; any other id is the person's own and is written as typed.
      const hit = matchChoice(toChoices(lossIds()), typed);
      sayAddError(null);
      const type = hit ? hit.value : typed;
      picker.input.value = '';
      const def = schema.variantFor(lossKind, type);
      const weightProp = def ? schema.props(def).loss_weight : null;
      const w = weightProp ? schema.kind(weightProp) : null;
      const weight = w && w.hasDefault && typeof w.default === 'number' ? w.default : 1;
      // The backend cannot append by index: the whole list is set, with the new loss at the end.
      const current = getAt(config(), ['train', 'losses']);
      const existing = Array.isArray(current) ? current : [];
      // The section is redrawn with the new loss in it: focus goes to that loss's weight, its first field.
      focusAfter = { config: recipe.get().config, keys: [pathKey(['train', 'losses', existing.length, 'loss_weight'])], at: Date.now() };
      return recipe.applyEdits([{ path: ['train', 'losses'], value: [...existing, { type, loss_weight: weight }] }]).catch((error) => {
        focusAfter = null;
        toast(`The loss was not added: ${error.message}`, { kind: 'err' });
      });
    };
    // The installed losses, in a list that narrows while typing. Another id can be typed: the check says
    // whether the trainer has it.
    const picker = createCombobox({
      id: selectId,
      inputClass: 'input--path',
      placeholder: 'Choose a loss',
      closed: false,
      listLabel: 'Losses',
      getChoices: () => toChoices(lossIds()),
      onPick: () => sayAddError(null),
      onEnter: addLoss,
      onEscape: () => {
        picker.input.value = '';
        sayAddError(null);
      },
    });
    const add = h('button', { class: 'btn btn--sm', type: 'button', onclick: addLoss }, 'Add a loss');
    rows.push(
      undoOffer(['train', 'losses']),
      h(
        'div',
        { class: 'form__actions' },
        h('label', { class: 'visually-hidden', for: selectId }, 'Loss to add'),
        picker.el,
        add,
        addError,
      ),
      losses.length ? h('p', { class: 'fld__desc form__help' }, REPLACE_LIST_NOTE) : null,
      losses.length ? measureLosses() : null,
    );
    return rows;
  }

  /** Open the job dialog (runs/analysis.js) on the open recipe as it is now; every failure ends in one toast. */
  async function openRecipeJob(kind, what) {
    try {
      const mod = await import('../runs/analysis.js');
      if (typeof mod?.openAnalysisDialog !== 'function') throw new Error('its dialog is missing from this version of the app.');
      // The dialog asks before anything starts, and opens the job itself when one is created.
      await mod.openAnalysisDialog({ kind, yaml: recipe.get().yaml, sourceLabel: 'the open recipe' });
    } catch (error) {
      toast(`${what} could not be opened: ${error?.message ?? error}`, { kind: 'err' });
    }
  }

  /**
   * "List this network's parameters": a job that builds the network without weights or data and names every
   * tensor. The service takes a recipe that is not complete yet, so it does not wait for the check.
   */
  function listParameters() {
    const helpId = uid('form-parameters-help');
    return h(
      'div',
      { class: 'form__sub' },
      h(
        'div',
        { class: 'form__actions' },
        h('button', { class: 'btn btn--sm btn--quiet', type: 'button', 'aria-describedby': helpId, onclick: () => openRecipeJob('network-parameters', 'The parameter list') }, "List this network's parameters"),
      ),
      h(
        'p',
        { class: 'fld__desc form__help', id: helpId },
        "Builds the network on the trainer machine's CPU, without weights or data, and lists every named parameter with its shape and size. It also counts what the optimizer filter selects, and can show what a weight transfer would copy. Training is not started.",
      ),
    );
  }

  /** "Measure these losses": a loss report analysis job, set up in its own dialog (runs/analysis.js). */
  function measureLosses() {
    const helpId = uid('form-measure-help');
    const reasonId = uid('form-measure-reason');
    const reason = h('p', { class: 'fld__note', id: reasonId, role: 'status', hidden: true });
    const button = h(
      'button',
      {
        class: 'btn btn--sm btn--quiet',
        type: 'button',
        'aria-describedby': `${helpId} ${reasonId}`,
        onclick: () => openRecipeJob('loss-report', 'The loss report'),
      },
      'Measure these losses',
    );
    // The report runs the recipe as it is, so it waits until the check finds no errors.
    const paint = () => {
      const { errors } = countIssues(recipe.get().issues ?? []);
      button.disabled = errors > 0;
      reason.hidden = !errors;
      reason.textContent = !errors ? '' : errors === 1 ? 'The check reports an error in this recipe. Fix it before measuring.' : `The check reports ${errors} errors in this recipe. Fix them before measuring.`;
    };
    paint();
    updaters.push(paint);
    return h(
      'div',
      { class: 'form__sub' },
      h('div', { class: 'form__actions' }, button),
      h('p', { class: 'fld__desc form__help', id: helpId }, "Runs the recipe's loss terms on a few batches on the trainer machine's CPU and reports each term's gradient size. Training is not started."),
      reason,
    );
  }

  function buildOther() {
    const c = config();
    const keys = otherKeys(layout, c);
    const rows = [];
    for (const key of keys) {
      if (!(key in rootProps)) rows.push(unknownRow([key], key));
      else if (key === 'onnx') {
        if (key in c) rows.push(genericBlock([key], key, schema.kind(rootProps[key]).description));
      } else rows.push(renderProp([], key, rootProps[key]));
    }
    if (!rows.length) rows.push(h('p', { class: 'form__empty' }, 'Every option of this recipe is shown in a section above.'));
    // A key the schema does not know leaves no row behind when it is removed: it can still be put back.
    rows.push(undoOffer((path) => sectionForPath(path, layout) === 'other'));
    return rows;
  }

  // ───────── search ─────────

  function buildSearch(q) {
    const matches = searchEntries({ schema, layout, rootProps, netKind, lossKind }, config()).filter((e) => {
      if (e.key.toLowerCase().includes(q)) return true;
      const d = e.prop ? schema.kind(e.prop).description : '';
      return d.toLowerCase().includes(q);
    });
    const shown = matches.slice(0, SEARCH_LIMIT);
    const parts = [h('h2', { class: 'form__title', tabindex: '-1' }, `Options matching "${q}"`)];
    parts.push(h('p', { class: 'form__lede', role: 'status' }, matches.length ? (matches.length > shown.length ? `Showing the first ${shown.length} of ${matches.length} matches. Type more to narrow the search.` : `${matches.length} ${matches.length === 1 ? 'match' : 'matches'}.`) : 'No option matches. Try part of the option name as it appears in the YAML.'));
    for (const sec of SECTIONS) {
      const mine = shown.filter((e) => e.section === sec.id);
      if (!mine.length) continue;
      parts.push(h('h3', { class: 'form__resulthead' }, sec.title));
      for (const e of mine) {
        if (!e.base.length && e.key === 'task') parts.push(taskRow());
        else if (!e.base.length && e.key === 'task_options') parts.push(...taskOptionsRows());
        else if (e.prop) {
          const label = e.base.length ? `${e.base.map(String).join('.')}.` : '';
          // Keep the full path visible: the same key exists in several places (datasets, losses).
          parts.push(label ? h('div', { class: 'form__result' }, h('p', { class: 'form__resultpath mono' }, label), renderProp(e.base, e.key, e.prop)) : renderProp(e.base, e.key, e.prop));
        } else parts.push(unknownRow([...e.base, e.key], e.key));
      }
    }
    return parts;
  }

  // ───────── render ─────────

  function rebuild() {
    // Remember the focused control and its caret so a rebuild does not take them away.
    const active = document.activeElement;
    let restore = null;
    if (active && content.contains(active) && active.dataset?.fld) {
      const same = [...content.querySelectorAll('[data-fld]')].filter((x) => x.dataset.fld === active.dataset.fld);
      restore = { key: active.dataset.fld, index: same.indexOf(active), value: active.value, start: active.selectionStart, end: active.selectionEnd };
    }
    const scroll = settings.scrollTop;
    fields = new Map();
    blocks = new Map();
    updaters = [];
    issuesBox = null;
    if (query) mount(content, buildSearch(query));
    else {
      const sec = sectionById(section);
      issuesBox = h('div', { class: 'form__sectionissues' });
      const lede = section === 'datasets' ? datasetsLede(currentTask(), sec.text, taskEntry(currentTask())?.datasetGeometry === 'whole_images') : sec.text;
      // tabindex -1: focus can land on the heading when the row that held it is gone.
      mount(content, h('h2', { class: 'form__title', tabindex: '-1' }, sec.title), h('p', { class: 'form__lede' }, lede), issuesBox, h('div', { class: 'form__rows' }, builders[section]()));
    }
    settings.scrollTop = scroll;
    if (restore) {
      const target = [...content.querySelectorAll('[data-fld]')].filter((x) => x.dataset.fld === restore.key)[restore.index];
      if (target) {
        if (typeof restore.value === 'string' && target.value !== restore.value && target.type !== 'checkbox') target.value = restore.value;
        target.focus({ preventScroll: true });
        try {
          if (restore.start !== null && restore.start !== undefined) target.setSelectionRange(restore.start, restore.end);
        } catch {
          // selection is not supported on this control type
        }
      }
    }
  }

  function applyIssues(s) {
    const issues = Array.isArray(s.issues) ? s.issues : [];
    const per = new Map(SECTION_IDS.map((id) => [id, []]));
    const byField = new Map();
    const unmatched = [];
    for (const issue of issues) {
      // Both path forms (`train.losses[1]`, `datasets.train.dataroot_gt.0`) parse to the same segments;
      // a missing required field reported at its parent is placed at the missing key.
      const { path: p } = issueTarget(parsePath(issue.path), issue.message);
      const sec = sectionForPath(p, layout);
      per.get(sec)?.push(issue);
      const target = targetFor(p);
      if (target) {
        if (!byField.has(target)) byField.set(target, []);
        byField.get(target).push(issue);
      } else if (!query && sec === section) unmatched.push(issue);
    }
    for (const field of fields.values()) field.setIssues(byField.get(field) ?? []);
    for (const block of blocks.values()) block.setIssues(byField.get(block) ?? []);
    // A closed group still says that it holds issues, so outline badges always lead somewhere visible.
    const tally = new Map();
    for (const [field, list] of byField) {
      for (let d = field.el.parentElement; d && d !== content; d = d.parentElement) {
        if (d.tagName !== 'DETAILS') continue;
        const t = tally.get(d) ?? { errors: 0, warnings: 0 };
        for (const issue of list) issue.level === 'error' ? (t.errors += 1) : (t.warnings += 1);
        tally.set(d, t);
      }
    }
    for (const d of content.querySelectorAll('details')) {
      const slot = groupSlots.get(d);
      if (!slot) continue;
      const t = tally.get(d);
      mount(slot, t?.errors ? badge(t.errors === 1 ? '1 error' : `${t.errors} errors`, 'err') : t?.warnings ? badge(t.warnings === 1 ? '1 warning' : `${t.warnings} warnings`, 'warn') : null);
    }
    if (issuesBox) mount(issuesBox, unmatched.length ? issueList(unmatched, { onSelect: (issue) => recipe.requestFocus(issue.path) }) : null);
    for (const sec of SECTIONS) {
      const { errors, warnings } = countIssues(per.get(sec.id));
      const slot = navBadges.get(sec.id);
      const label = sec.short ?? sec.title;
      if (errors) mount(slot, badge(errors === 1 ? '1 error' : `${errors} errors`, 'err'));
      else if (warnings) mount(slot, badge(warnings === 1 ? '1 warning' : `${warnings} warnings`, 'warn'));
      else mount(slot);
      const text = errors ? `${label} (${errors} ${errors === 1 ? 'error' : 'errors'})` : warnings ? `${label} (${warnings} ${warnings === 1 ? 'warning' : 'warnings'})` : label;
      const option = pickerOptions.get(sec.id);
      if (option.textContent !== text) option.textContent = text;
    }
  }

  /** The deepest field or block of the current build on the path (a field wins over a block). */
  function targetFor(p) {
    for (let n = p.length; n > 0; n -= 1) {
      const key = pathKey(p.slice(0, n));
      const hit = fields.get(key) ?? blocks.get(key);
      if (hit) return hit;
    }
    return null;
  }

  function setBroken(next) {
    if (next === broken) return;
    broken = next;
    fieldset.disabled = broken;
    mount(statusEl, broken ? notice('warn', 'The YAML has a syntax error, so options cannot be edited here. Fix it in the YAML pane, or use Changes to revert.') : null);
  }

  function render(force = false) {
    if (disposed || !schema) return;
    const s = recipe.get();
    if (!force && lastSeen.config === s.config && lastSeen.issues === s.issues && lastSeen.configCurrent === s.configCurrent && lastSeen.check === s.check) return;
    lastSeen = { config: s.config, issues: s.issues, configCurrent: s.configCurrent, check: s.check };

    if (!s.config) {
      builtSig = null;
      if (s.check === 'failed') mount(content, notice('err', `The recipe could not be checked, so its options cannot be shown: ${s.checkError ?? 'no reason given'}.`));
      else if (s.check === 'done') mount(content, notice('warn', 'The YAML has a syntax error, so options cannot be edited here. Fix it in the YAML pane, or use Changes to revert.'));
      else mount(content, loading('Reading the recipe…'));
      return;
    }

    // The broken state shows only when it lasts: while someone types in the YAML pane, config is
    // briefly not current before the debounced check starts, and the notice would flicker.
    const wantBroken = !s.configCurrent && s.check === 'done';
    clearTimeout(brokenTimer);
    if (!wantBroken) setBroken(false);
    else if (!broken) {
      brokenTimer = setTimeout(() => {
        const now = recipe.get();
        setBroken(!now.configCurrent && now.check === 'done');
      }, 700);
    }

    const c = s.config;
    // The task decides wording, choices and the network block, so a task change rebuilds the rows.
    const base = `${query ? `q:${query}:${shape(c)}` : `s:${section}:${sectionSignature(section, c, layout, Boolean(catalog))}`}|${explicitTask(c) ?? ''}|${catalog ? 'ready' : tasks.get().phase}|${blockTypes(c)}`;
    // The task the check inferred is a background result: while a control in the form holds focus,
    // a change of it alone waits, so the field being typed in is not rebuilt under the person.
    const sig = `${base}|${explicitTask(c) ? '' : checkedTask() ?? ''}`;
    const onlyCheckMoved = builtSig !== null && builtSig.startsWith(`${base}|`) && content.contains(document.activeElement);
    if (sig !== builtSig && !onlyCheckMoved) {
      builtSig = sig;
      rebuild();
    } else {
      for (const field of fields.values()) field.update(getAt(c, field.path), hasAt(c, field.path));
    }
    for (const fn of updaters) fn(c);
    applyIssues(s);
    paintReview(c);
    restoreFocusAfterRemoval();
  }

  // ───────── focus requests ─────────

  function focusPath(text) {
    if (!schema || !layout || !recipe.get().config) return;
    const p = parsePath(text);
    if (!p.length) return;
    const sec = sectionForPath(p, layout);
    if (query || sec !== section) openSection(sec);
    const hit = targetFor(p);
    // Blocks have no control to focus: scroll to their heading instead.
    const target = hit && typeof hit.focus === 'function' ? hit : null;
    let node = target?.el ?? null;
    if (!node) {
      // No field for the path (a dataset block, a whole loss): use its heading when there is one.
      for (let n = p.length; n > 0 && !node; n -= 1) node = content.querySelector(`[data-path="${CSS.escape(pathKey(p.slice(0, n)))}"]`);
    }
    if (!node) return;
    // A whole dataset that has the training videos panel: the panel to the top, its title focused.
    const panel = p.length === 2 && p[0] === 'datasets' && !target ? node.closest('.form__block')?.querySelector('.vprep') : null;
    if (panel) {
      panel.scrollIntoView({ block: 'start' });
      panel.querySelector('.vprep__title')?.focus({ preventScroll: true });
      return;
    }
    for (let d = node.parentElement; d && d !== content; d = d.parentElement) if (d.tagName === 'DETAILS') d.open = true;
    node.scrollIntoView({ block: 'center' });
    if (target) target.focus();
    const row = target ? target.el : node;
    row.classList.add('form__flash');
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => row.classList.remove('form__flash'), 1200);
  }

  // ───────── start ─────────

  async function start() {
    mount(root, loading('Reading the recipe…'));
    try {
      const loaded = await loadSchema();
      if (disposed) return;
      schema = loaded;
    } catch (error) {
      if (disposed) return;
      mount(
        root,
        h(
          'div',
          { class: 'form__pad' },
          notice(
            'err',
            `The options schema did not load, so the form cannot be shown: ${error.message}`,
            h('button', { class: 'btn btn--sm', type: 'button', onclick: () => { api.schema.reset(); start(); } }, 'Try again'),
          ),
        ),
      );
      return;
    }
    // Options the service checks but the schema does not list yet are offered too (schema.js).
    rootProps = withServiceOptions(schema.props(schema.rootNode));
    layout = buildLayout(Object.keys(rootProps));
    netKind = rootProps.network_g ? schema.kind(rootProps.network_g) : null;
    const lossesProp = schema.props(rootProps.train).losses;
    lossKind = lossesProp ? schema.kind(schema.deref(lossesProp).items ?? {}) : null;
    mount(root, shell);
    markNav();
    lastSeen = {};
    render(true);
    loadCatalog();
  }

  function loadCatalog() {
    const job = tasks.load();
    render(true); // shows "loading" again after a failed attempt
    job.then(
      (result) => {
        if (disposed) return;
        catalog = { tasks: result.tasks ?? [], architectures: result.architectures ?? [], losses: result.losses ?? [] };
        render(true);
      },
      () => {
        // Without the catalog the pickers fall back to the ids in the schema; the task field says why.
        if (!disposed) render(true);
      },
    );
  }

  const unsubscribe = recipe.subscribe((s, prev) => {
    if (s.yaml !== prev.yaml) lastChange = { before: prev.yaml, after: s.yaml };
    render();
  });
  const unfocus = recipe.onFocusRequest(focusPath);
  start();

  return () => {
    disposed = true;
    unsubscribe();
    unfocus();
    runSearch.cancel();
    clearTimeout(brokenTimer);
    clearTimeout(flashTimer);
    clearTimeout(focusTimer);
    root.remove();
  };
}
