// Result of a completed dataset split job (analysis kind `split-dataset`): a restoration recipe's paired
// training images divided into a training list and a validation list. The producer is
// studio/dataset_split.py (described in studio/DATASET_SPLIT.md). The job writes three files and copies
// or moves no image: each list holds ground truth file names, one per line.
//
// The same kind also splits labelled data (classification, segmentation, detection:
// studio/supervised_split.py). The lists are then JSON lines ({root, image, mask?} per line) or, for
// detection, COCO files; the result counts `samples`, and has `task`, `imageIdentity`, `sourceManifest`,
// `classes` ([{id, name, train, val}], classification) and `recipePath` (the job's recipe written again
// with its `datasets` section replaced by the two datasets).
//
// The same kind also splits paired video by whole scene (studio/video_split.py; dataset type
// PairedVideoDataset). The result then counts complete `windows` (a window is `clipSize` consecutive
// frames around one target frame) and `scenes`; scenes that share a file on disk are joined into one
// group and stay on one side (`physicalIdentity`). Each list is a scene manifest (JSON:
// {format, clipSize, rootCount, scenes: [{root, numbered, key, membershipSha256}]}), read by the dataset
// option `video_scene_manifest`, and the result has `recipePath` like the labelled case.
//
//   renderSplitDataset(host, {job, result, artifacts}) -> cleanup
//   splitDatasetProgressText(progress, job) -> one sentence, or ''
//
// Shapes:
//   result = {algorithm, seed, groupBy: 'pair'|'parent'|'scene', validationFraction,
//     achievedValidationFraction, counts: {pairs, groups, targetVal, train, val, trainGroups, valGroups},
//     sourceMetaInfo: path|null, destination, manifests: {train, val}, manifestSha256: {train, val},
//     datasets: {train: {…dataset options, meta_info}, val: {…}}, reportPath, notes: [string], outputCount}
//   artifacts = [{path, available, version, url}] (no url when the file is no longer there)
//   progress = {stage: 'selecting'|'assigning'|'writing'|'verifying', completed, total} with total null
//     while the pairs are still being found

import { h, mount, fmt, uid } from '../dom.js';
import { notice, copyText, confirmDialog, toast, loading } from '../ui.js';
import { api } from '../api.js';
import { recipe } from '../state/recipe.js';
import { openAsRecipe } from '../actions.js';
import { highlightedPre } from '../recipe/yaml-highlight.js';

/** At most this many names of a list are drawn; the downloaded file has all of them. */
const NAMES_MAX = 200;

/** What stays together, per `groupBy` (DATASET_SPLIT.md, "Assignment"). */
const GROUPING = {
  pair: { kept: 'Each pair by itself', whole: 'Whole pairs go', exact: true },
  parent: { kept: 'All pairs in the same folder (one group per folder)', whole: 'Whole folders go', exact: false },
  scene: { kept: 'Everything under the same top-level folder (one group per top-level folder)', whole: 'Whole top-level folders go', exact: false },
};

/**
 * The same for labelled data (studio/supervised_split.py `_groups`). A group is named by the image's path
 * inside its data folder, for classification inside its class folder: equal subfolder names in different
 * class folders are therefore one group.
 */
const GROUPING_IMAGES = {
  sample: { kept: 'Each image by itself', whole: 'Whole images go' },
  parent: { kept: 'All images in the same folder (one group per folder)', whole: 'Whole folders go' },
  scene: { kept: 'Everything under the same top-level folder (one group per top-level folder)', whole: 'Whole top-level folders go' },
};
const GROUPING_CLASSES = {
  parent: { kept: 'All images in the same subfolder of a class folder; subfolders with the same name in different class folders are one group', whole: 'Whole subfolders go' },
  scene: { kept: 'Everything under the same first subfolder of a class folder; subfolders with the same name in different class folders are one group', whole: 'Whole subfolders go' },
};
const LABELLED = new Set(['image_classification', 'semantic_segmentation', 'object_detection']);
/** A COCO file is parsed whole to list its images; above this size it is only offered as a download. */
const COCO_MAX_BYTES = 32 * 1024 * 1024;

const REPEATABLE =
  'The same seed, grouping and file names give the same split again on any machine. It does not depend on the order in which the folder is listed.';
const GENERATED = "The service wrote the job's recipe again with its training and validation data set to the two lists.";
const GENERATED_ONLY = 'Its datasets section holds these two datasets only: any other dataset of the recipe the job was given is not in it.';
/**
 * What the generated recipe's datasets section holds. A job that names the keys it wrote
 * (`recipeDatasetKeys: {train, val}`) changed only those and kept the recipe's other datasets.
 */
export function generatedKeysText(keys) {
  const train = isObject(keys) && isText(keys.train) ? keys.train : '';
  const val = isObject(keys) && isText(keys.val) ? keys.val : '';
  if (!train || !val) return GENERATED_ONLY;
  return `Only its datasets ${train} and ${val} were written; the recipe's other validation and test datasets are kept as they were.`;
}
const GENERATED_PRINT = 'Nothing is saved or started. The recipe opens unsaved.';
const NOT_SAVED =
  'Nothing is saved and no training is started. The recipe opens on the Recipe screen as text that is not saved yet, with meta_info of its training dataset set to the training list and datasets.val set to the validation dataset.';
const COPY_FAILED = 'Copying failed. Select the text and copy it by hand.';

const SIDES = [
  { key: 'train', name: 'Training', file: 'train.txt', download: 'Download the training list', region: 'Names in the training list', scenes: 'Scenes in the training list' },
  { key: 'val', name: 'Validation', file: 'val.txt', download: 'Download the validation list', region: 'Names in the validation list', scenes: 'Scenes in the validation list' },
];

/** Paired video split by whole scene (studio/video_split.py): windows are counted, scenes stay whole. */
const VIDEO_TYPE = 'pairedvideodataset';
const GROUPING_VIDEO = { kept: 'Whole scenes', whole: 'Whole scenes go', exact: false };
/** The scene manifest this view can list (trainner_data/frame_sequences.py). */
const SCENE_FORMAT = 'paired-video-scenes-v1';
/** `physicalIdentity.method` 'device-and-inode': two paths are one file when the disk says so. */
const IDENTITY_SCOPE = 'Compared by file identity on disk, not by content: a copied or re-encoded frame is not recognised as the same.';
const FRAME_ROLES = { gt: 'ground truth', lq: 'low quality' };
const SCENE_NAMES =
  'A scene is named by the start of its frame names and by its folder; folder 1 is the first folder of the dataset. The list does not say how many windows each scene has.';
const SCENES_UNKNOWN = 'This list is not written in a form this page knows, so its scenes are not shown here. The download has all of it.';

// ───────── pure helpers (exported for checks) ─────────

export const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
export const isText = (v) => typeof v === 'string' && v !== '';
export const baseName = (path) => {
  const text = String(path ?? '');
  return text.slice(Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\')) + 1);
};

/**
 * What is kept together, in words, for a grouping and a task; null for a grouping this view does not know.
 * @param {string|null} groupBy
 * @param {string|null} task
 * @param {boolean} [noCopies] labelled data only: the job found no identical images. With copies, or with
 *   classes to keep on both sides, the share is never promised as "the nearest that fits".
 */
export function groupingWords(groupBy, task, noCopies = false) {
  if (!LABELLED.has(task)) return GROUPING[groupBy] ?? null;
  const key = groupBy === 'pair' ? 'sample' : groupBy;
  const words = (task === 'image_classification' && GROUPING_CLASSES[key]) || GROUPING_IMAGES[key];
  return words ? { ...words, exact: key === 'sample' && task !== 'image_classification' && noCopies } : null;
}

/**
 * One plain sentence for a running job's progress; '' when absent or of an unknown stage.
 * @param {object|null} progress
 * @param {object|null} [job] when given and its task is a labelled one, the shared stages say "images"
 */
export function splitDatasetProgressText(progress, job = null) {
  if (!isObject(progress)) return '';
  if (isVideoType(job?.params?.datasetType)) return videoProgressText(progress);
  const unit =LABELLED.has(job?.params?.task) ? 'image' : 'pair';
  const counted = Number.isFinite(progress.completed) && Number.isFinite(progress.total);
  const of = counted ? `${fmt.int(progress.completed)} of ${fmt.plural(progress.total, unit)}` : '';
  if (progress.stage === 'selecting') return counted ? `Read ${of} from the folders` : `Finding the ${unit}s`;
  if (progress.stage === 'assigning') return counted ? `Assigned ${of} to a side` : `Assigning the ${unit}s to a side`;
  if (progress.stage === 'writing') return counted ? `Wrote ${of} to the lists` : 'Writing the two lists';
  if (progress.stage === 'verifying') return counted ? `Checked ${of} against the written lists` : 'Checking the written lists';
  // Stages only the labelled-data splitter has (studio/supervised_split.py).
  const images = counted ? `${fmt.int(progress.completed)} of ${fmt.plural(progress.total, 'image')}` : '';
  if (progress.stage === 'identifying') return counted ? `Compared ${images} for identical copies` : 'Looking for identical images';
  const side = /^(writing|verifying)-(train|val)$/.exec(String(progress.stage ?? ''));
  if (side) {
    const list = side[2] === 'train' ? 'training list' : 'validation list';
    if (side[1] === 'writing') return counted ? `Wrote ${images} to the ${list}` : `Writing the ${list}`;
    return counted ? `Checked ${images} of the ${list}` : `Checking the ${list}`;
  }
  return '';
}

/** Whether a dataset type (as the job or a dataset names it) is paired video. */
export const isVideoType = (type) => typeof type === 'string' && type.toLowerCase() === VIDEO_TYPE;

/**
 * The same for a paired video split (studio/video_split.py `tick`): scenes are found, their frame files
 * compared by identity on disk, whole groups of scenes assigned, each list written, every frame file
 * checked for a change made meanwhile, and the files put in place.
 */
function videoProgressText(progress) {
  const done = Number.isFinite(progress.completed) ? progress.completed : null;
  const counted = done !== null && Number.isFinite(progress.total);
  const of = (unit) => `${fmt.int(done)} of ${fmt.plural(progress.total, unit)}`;
  const stage = String(progress.stage ?? '');
  if (stage === 'selecting') return 'Finding the scenes and their windows';
  if (stage === 'identifying') {
    return done ? `Looking for files that more than one scene uses: ${fmt.plural(done, 'frame')} looked at` : 'Looking for files that more than one scene uses';
  }
  if (stage === 'assigning') {
    if (counted && done > 0) return `Assigned ${of('window')} to a side`;
    return counted ? `Assigning ${fmt.plural(progress.total, 'window')} to a side, whole scenes at a time` : 'Assigning whole scenes to a side';
  }
  const side = /^writing-(train|val)$/.exec(stage);
  if (side) {
    const list = side[1] === 'train' ? 'training list' : 'validation list';
    if (counted && done > 0) return `Wrote ${of('scene')} to the ${list}`;
    return counted ? `Writing ${fmt.plural(progress.total, 'scene')} to the ${list}` : `Writing the ${list}`;
  }
  if (stage === 'verifying') return counted ? `Checked ${of('frame file')} for a change made during the split` : 'Checking that no frame file changed during the split';
  if (stage === 'publishing') return 'Putting the lists, the record and the recipe in place';
  return '';
}

/**
 * The lead sentence of a paired video split: complete windows on each side, and the scenes they come from.
 * @param {object|null} counts `{windows, scenes, train, val, trainScenes, valScenes}`
 */
export function videoLeadSentence(counts) {
  const c = isObject(counts) ? counts : {};
  const lead = leadSentence({ train: c.train, val: c.val, pairs: c.windows }, 'window');
  if (!Number.isFinite(c.train) || !Number.isFinite(c.val) || !Number.isFinite(c.scenes)) return lead;
  const sides = Number.isFinite(c.trainScenes) && Number.isFinite(c.valScenes) ? ` (${fmt.int(c.trainScenes)} for training, ${fmt.int(c.valScenes)} for validation)` : '';
  return `${lead.slice(0, -1)}, from ${fmt.plural(c.scenes, 'scene')}${sides}.`;
}

/** How many files on disk more than one scene uses, in a sentence; '' when the job did not report it. */
export function sharedFilesSentence(shared, counts) {
  if (!Number.isFinite(shared)) return '';
  if (shared === 0) return 'No file on disk is used by more than one scene.';
  const c = isObject(counts) ? counts : {};
  const joined = Number.isFinite(c.scenes) && Number.isFinite(c.groups) ? ` (${fmt.plural(c.scenes, 'scene')}, ${fmt.plural(c.groups, 'group')})` : '';
  return `${fmt.plural(shared, 'file on disk is', 'files on disk are')} used by more than one scene; those scenes were joined and stay on the same side${joined}.`;
}

/**
 * One example of a shared file in words: the frames ([{root, role: 'gt'|'lq', frame}]) that are one file
 * on disk. '' when the example does not name at least two frames.
 */
export function sharedExampleSentence(example) {
  const parts = (Array.isArray(example) ? example : [])
    .filter((entry) => isObject(entry) && isText(entry.frame))
    .map((entry) => {
      const where = [FRAME_ROLES[entry.role], Number.isInteger(entry.root) ? `folder ${entry.root}` : ''].filter(Boolean).join(', ');
      return where ? `${entry.frame} (${where})` : entry.frame;
    });
  if (parts.length < 2) return '';
  return `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]} are the same file.`;
}

/** A scene of a scene manifest in words: the start of its frame names (`key`) and its folder (`root`). */
export function sceneName(scene) {
  const folder = Number.isInteger(scene?.root) ? ` (folder ${scene.root})` : '';
  if (isText(scene?.key)) return `${scene.key}${folder}`;
  return `${scene?.numbered === true ? 'Frames named by number only' : 'Scene without a name'}${folder}`;
}

/**
 * The scenes of a scene manifest: the first `limit` in words and how many there are; null when the text
 * is not a manifest of the format this view knows.
 */
export function readSceneManifest(text, limit = NAMES_MAX) {
  let file;
  try {
    file = JSON.parse(String(text ?? '').replace(/^﻿/, ''));
  } catch {
    return null;
  }
  if (!isObject(file) || file.format !== SCENE_FORMAT || !Array.isArray(file.scenes)) return null;
  if (!file.scenes.every((scene) => isObject(scene) && typeof scene.key === 'string')) return null;
  return { names: file.scenes.slice(0, limit).map(sceneName), total: file.scenes.length };
}

/** The lead sentence: how many pairs (or images, for labelled data) went where. The total is `pairs`, else `samples`. */
export function leadSentence(counts, unit = 'pair') {
  const c = isObject(counts) ? counts : {};
  const all = Number.isFinite(c.pairs) ? c.pairs : c.samples;
  if (!Number.isFinite(c.train) || !Number.isFinite(c.val)) {
    return Number.isFinite(all)
      ? `${fmt.plural(all, unit)} ${all === 1 ? 'was' : 'were'} divided. The job did not report how many went to each side.`
      : `The job did not report how many ${unit}s went to each side.`;
  }
  const total = Number.isFinite(all) ? `, out of ${fmt.int(all)}` : '';
  return `${fmt.plural(c.train, unit)} for training and ${fmt.int(c.val)} for validation${total}.`;
}

/**
 * Two shares as percentages with the fewest decimals that keep a difference visible: whole percent,
 * one decimal when both would read the same, more only when one decimal still hides it.
 * @returns {[string, string, boolean]} the two texts and whether the shares differ
 */
export function shareTexts(achieved, asked) {
  const text = (value, digits) => `${(value * 100).toFixed(digits)}%`;
  if (!Number.isFinite(achieved) || !Number.isFinite(asked)) {
    return [Number.isFinite(achieved) ? text(achieved, 0) : '', Number.isFinite(asked) ? text(asked, 0) : '', false];
  }
  if (achieved === asked) return [text(achieved, 0), text(asked, 0), false];
  for (let digits = 0; digits <= 4; digits += 1) {
    if (text(achieved, digits) !== text(asked, digits)) return [text(achieved, digits), text(asked, digits), true];
  }
  return [text(achieved, 0), text(asked, 0), false];
}

/** The sentences after the lead about the validation share; '' when the achieved share is absent. */
export function shareSentence(achieved, asked, groupBy, words = GROUPING[groupBy]) {
  const [got, wanted, differ] = shareTexts(achieved, asked);
  if (!got) return '';
  if (!wanted) return `That is ${got} for validation.`;
  if (!differ) return `That is ${got} for validation, as asked for.`;
  const grouping = words;
  const whole = grouping ? grouping.whole : 'Whole groups go';
  const why = grouping?.exact ? 'so the share is the nearest that fits' : 'so the share can only come close to what was asked for';
  return `That is ${got} for validation; ${wanted} was asked for. ${whole} to one side, ${why}.`;
}

/** A root option as a list of strings: a string, or a list of strings; null when it is anything else. */
function rootList(value) {
  if (isText(value)) return [value];
  if (Array.isArray(value) && value.length && value.every(isText)) return value;
  return null;
}

/** Forward slashes, no trailing slash, no leading "./". */
function cleanPath(path) {
  let text = String(path).replace(/\\/g, '/');
  while (text.startsWith('./')) text = text.slice(2);
  while (text.length > 1 && text.endsWith('/')) text = text.slice(0, -1);
  return text;
}

/**
 * Whether a root written in a recipe names the folder the result gives as an absolute path: the same
 * text, or a relative root that the absolute one ends with at a folder boundary.
 */
export function sameRoot(recipeRoot, resultRoot) {
  const mine = cleanPath(recipeRoot);
  const theirs = cleanPath(resultRoot);
  if (mine === theirs) return true;
  const absolute = mine.startsWith('/') || /^[A-Za-z]:\//.test(mine);
  if (absolute || mine === '' || mine === '.' || mine.split('/').includes('..')) return false;
  return theirs.endsWith(`/${mine}`);
}

/** Two root options (string or list) name the same folders, in the same order. */
export function sameRoots(recipeValue, resultValue) {
  const mine = rootList(recipeValue);
  const theirs = rootList(resultValue);
  if (!mine || !theirs || mine.length !== theirs.length) return false;
  return mine.every((root, index) => sameRoot(root, theirs[index]));
}

/**
 * Whether a recipe already reads both lists of this split: its training dataset has one of `lists.train`
 * as meta_info, and one of its validation datasets (`val`, or a key whose part before the first "_" is
 * "val") has one of `lists.val`. Paths are compared as text, trimmed and nothing more. `option` is the
 * dataset option that carries the list: meta_info, or annotation_file for detection.
 * @param {object|null} datasets the recipe's `datasets`
 * @param {{train?: string[], val?: string[]}|null} lists
 */
export function usesSplit(datasets, lists, option = 'meta_info') {
  if (!isObject(datasets) || !isObject(lists)) return false;
  const clean = (value) => (typeof value === 'string' ? value.trim() : '');
  const among = (value, wanted) => clean(value) !== '' && Array.isArray(wanted) && wanted.some((path) => clean(path) === clean(value));
  if (!isObject(datasets.train) || !among(datasets.train[option], lists.train)) return false;
  return Object.entries(datasets).some(([key, dataset]) => key.split('_')[0] === 'val' && isObject(dataset) && among(dataset[option], lists.val));
}

/**
 * Where the open recipe stands against this split.
 * @param {object} state `recipe.get()`
 * @param {object|null} train `result.datasets.train`
 * @param {{train?: string[], val?: string[]}|null} [lists] the paths of this split's two lists (see `usesSplit`)
 * @returns {{kind: 'none'|'checking'|'unchecked'|'unread'|'applied'|'notrain'|'other'|'match', name: string, hasVal?: boolean, reason?: string}}
 *   `applied`: the recipe already reads both lists of this split, so there is nothing to put into it.
 *   `unchecked`: the check itself could not run; `unread`: it answered without a parsed recipe.
 */
export function recipeStanding(state, train, lists = null, option = 'meta_info') {
  const s = isObject(state) ? state : {};
  const name = isText(s.name) ? s.name : '';
  if (s.phase !== 'open') return { kind: 'none', name };
  // The folders are read from the service's parsed copy; one that belongs to older text says nothing.
  if (!isObject(s.config) || s.configCurrent === false) {
    if (s.check === 'failed') return { kind: 'unchecked', name, reason: isText(s.checkError) ? s.checkError : '' };
    if (s.check === 'done') return { kind: 'unread', name };
    return { kind: 'checking', name };
  }
  if (usesSplit(s.config.datasets, lists, option)) return { kind: 'applied', name };
  const mine = isObject(s.config.datasets) ? s.config.datasets.train : null;
  if (!isObject(mine) || !rootList(mine.dataroot_gt) || !rootList(mine.dataroot_lq)) return { kind: 'notrain', name };
  const same = isObject(train) && sameRoots(mine.dataroot_gt, train.dataroot_gt) && sameRoots(mine.dataroot_lq, train.dataroot_lq);
  if (!same) return { kind: 'other', name };
  const val = s.config.datasets.val;
  return { kind: 'match', name, hasVal: val !== undefined && val !== null };
}

/** The sentence shown in place of the button; '' when the button is offered. */
export function standingSentence(standing) {
  const which = standing.name ? `The open recipe (${standing.name})` : 'The open recipe';
  const instead = 'Open the recipe the split was made from, or copy the recipe lines below.';
  switch (standing.kind) {
    case 'none':
      return 'No recipe is open.';
    case 'checking':
      return `${which} is being checked. Whether this split belongs to it shows here in a moment.`;
    case 'unchecked':
      return `${which} could not be checked${standing.reason ? ` (${standing.reason})` : ''}, so it is not known whether this split belongs to it. Check it again on the Recipe screen, or copy the recipe lines below.`;
    case 'unread':
      return `${which} could not be read as it is written, so it is not known whether this split belongs to it. Solve its errors on the Recipe screen, or copy the recipe lines below.`;
    case 'notrain':
      return `${which} has no training dataset with a ground truth and a low quality folder, so this split does not belong to it. ${instead}`;
    case 'applied':
      return `${which} already uses this split.`;
    case 'other':
      return `${which} trains on other folders, so this split does not belong to it. ${instead}`;
    default:
      return '';
  }
}

/**
 * A value as block YAML. Strings that are not plain-safe are written JSON-quoted (valid YAML); lists as
 * "- item" lines; null as `null`.
 */
export function toYaml(value) {
  const reserved = /^(?:true|false|yes|no|on|off|y|n|null|~|[-+]?\.(?:inf|nan))$/i;
  // A plain scalar here starts with a letter, "_", "/", "~" or "." (never a digit or a sign, so it
  // cannot be read as a number or a date) and holds only characters with no meaning in YAML.
  const plain = (s) => /^[A-Za-z_./~][A-Za-z0-9_./~-]*$/.test(s) && !/^\.\d/.test(s) && !reserved.test(s);
  const text = (s) => (plain(s) ? s : JSON.stringify(s));
  const scalar = (v) => (typeof v === 'string' ? text(v) : v === null || v === undefined ? 'null' : typeof v === 'number' || typeof v === 'boolean' ? String(v) : JSON.stringify(String(v)));
  const empty = (v) => (Array.isArray(v) ? (v.length ? '' : '[]') : isObject(v) ? (Object.keys(v).length ? '' : '{}') : scalar(v));
  const lines = [];
  /** Write `value` as the content that follows `first` (a "key:" or a "-"), children indented by `pad`. */
  const put = (first, value, pad) => {
    const inline = empty(value);
    if (inline) {
      lines.push(`${first} ${inline}`);
    } else if (Array.isArray(value)) {
      lines.push(first);
      for (const item of value) put(`${pad}  -`, item, `${pad}    `);
    } else {
      lines.push(first);
      for (const [key, item] of Object.entries(value)) put(`${pad}  ${text(key)}:`, item, `${pad}  `);
    }
  };
  if (!isObject(value) || !Object.keys(value).length) return `${empty(value)}\n`;
  for (const [key, item] of Object.entries(value)) put(`${text(key)}:`, item, '');
  return `${lines.join('\n')}\n`;
}

/** The names of a list file: the first `limit` and how many there are (empty lines are not names). */
export function readNames(text, limit = NAMES_MAX) {
  const source = String(text ?? '').replace(/^﻿/, '');
  const names = [];
  let total = 0;
  let start = 0;
  while (start < source.length) {
    let end = source.indexOf('\n', start);
    if (end < 0) end = source.length;
    const line = source.charCodeAt(end - 1) === 13 ? source.slice(start, end - 1) : source.slice(start, end);
    if (line) {
      total += 1;
      if (names.length < limit) names.push(line);
    }
    start = end + 1;
  }
  return { names, total };
}

/**
 * The images of a JSON-lines list (one `{root, image, mask?}` per line, as supervised_split.py writes it):
 * the first `limit` and how many lines there are. A line that is not such an entry is shown as written.
 */
export function readJsonLineImages(text, limit = NAMES_MAX) {
  const { names: lines, total } = readNames(text, limit);
  const names = lines.map((line) => {
    try {
      const entry = JSON.parse(line);
      if (isObject(entry) && isText(entry.image)) return Number.isInteger(entry.root) && entry.root > 1 ? `${entry.image} (root ${entry.root})` : entry.image;
    } catch {
      // Shown as written.
    }
    return line;
  });
  return { names, total };
}

/**
 * The images of a COCO file: `file_name` of the first `limit` images and how many images there are.
 * Throws (with the end of a sentence as its message) when the text is not such a file.
 */
export function readCocoImages(text, limit = NAMES_MAX) {
  let coco;
  try {
    coco = JSON.parse(String(text ?? '').replace(/^\uFEFF/, ''));
  } catch {
    throw new Error('it is not JSON.');
  }
  if (!isObject(coco) || !Array.isArray(coco.images)) throw new Error('it has no list of images.');
  const names = coco.images.slice(0, limit).map((image) => (isObject(image) && isText(image.file_name) ? image.file_name : 'Image without a file name'));
  return { names, total: coco.images.length };
}

/** How a list file is read here, by its name: 'names' (.txt), 'lines' (.jsonl), 'coco' (.json), or null. */
export function listFormat(name) {
  const text = String(name ?? '').toLowerCase();
  if (text.endsWith('.txt')) return 'names';
  if (text.endsWith('.jsonl')) return 'lines';
  if (text.endsWith('.json')) return 'coco';
  return null;
}

/** `result` without the parts this view draws; null when nothing is left. */
export function remainder(result, shown) {
  const strip = (value, paths) => {
    if (!isObject(value)) return value;
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      const below = paths.filter((path) => path[0] === key);
      if (below.some((path) => path.length === 1)) continue;
      const rest = strip(item, below.map((path) => path.slice(1)));
      if (isObject(item) && below.length && !Object.keys(rest).length) continue;
      out[key] = rest;
    }
    return out;
  };
  const rest = strip(result, shown);
  return Object.keys(rest).length ? rest : null;
}

// ───────── pieces (shared with runs/patches.js, which passes its own class prefix) ─────────

export function copyButton(label, value, done) {
  return h(
    'button',
    {
      class: 'btn btn--sm',
      type: 'button',
      onclick: async () => {
        const ok = await copyText(value);
        toast(ok ? done : COPY_FAILED, { kind: ok ? 'ok' : 'err' });
      },
    },
    label,
  );
}

export const absent = (text = 'Not reported', prefix = 'split') => h('span', { class: `${prefix}__muted` }, text);

/** A path on the trainer machine with "Copy path", or "Not reported". */
export function pathValue(path, prefix = 'split') {
  if (!isText(path)) return absent(undefined, prefix);
  return h('span', { class: `${prefix}__path-row` }, h('span', { class: `${prefix}__path` }, path), copyButton('Copy path', path, 'Copied the path.'));
}

/** The download link of a file, or the sentence that says why there is none. */
export function downloadLink(artifact, label, fallbackName, prefix = 'split') {
  if (!artifact) return absent('The job did not list this file for download.', prefix);
  if (!isText(artifact.url) || artifact.available === false) return absent("This file is no longer in the job's folder, so it cannot be downloaded.", prefix);
  return h('a', { class: `${prefix}__download`, href: artifact.url, download: baseName(artifact.path) || fallbackName }, label);
}

function checksumValue(sum) {
  if (!isText(sum)) return absent();
  if (!/^[0-9a-f]{64}$/i.test(sum)) return h('span', { class: 'split__sum' }, sum);
  return h('span', { class: 'split__sum', title: sum }, `${sum.slice(0, 12)}…`);
}

/** Why a download answered with an error, in the service's words when it gave any. */
export async function refusal(response) {
  try {
    const payload = await response.json();
    if (payload && isText(payload.error)) return payload.error;
  } catch {
    // Not JSON: the status line is all there is.
  }
  return `${response.status} ${response.statusText}`.trim();
}

/**
 * A closed <details> that reads a file of the job on first opening and draws it with `draw(text)`, which
 * returns `{nodes, focus}` (`focus`: the node that takes the focus after "Try again", or null). A read
 * that fails shows the reason and "Try again". `what` names the file in the sentences ("list"). With
 * `maxBytes`, a larger file is not drawn: the block says its size and leaves it to the download.
 */
export function lazyDetails({ artifact, signal, summary, draw, what = 'list', prefix = 'split', maxBytes = 0 }) {
  const body = h('div', { class: `${prefix}__names-body` });
  let state = 'idle';
  const load = async (retry = false) => {
    if (state === 'loading' || signal.aborted) return;
    state = 'loading';
    mount(body, loading(`Reading the ${what}…`));
    let failure = '';
    let drawn = null;
    try {
      let response;
      try {
        response = await fetch(artifact.url, { signal });
      } catch (cause) {
        if (cause?.name === 'AbortError') throw cause;
        throw new Error('The trainer service did not answer.');
      }
      if (!response.ok) throw new Error(await refusal(response));
      const tooLarge = (bytes) => ({
        nodes: [h('p', { class: `${prefix}__note` }, `This ${what} is ${fmt.int(Math.round(bytes / 1048576))} MB, too large to show here. The download has all of it.`)],
        focus: null,
      });
      const announced = Number(response.headers?.get?.('content-length'));
      if (maxBytes && Number.isFinite(announced) && announced > maxBytes) {
        drawn = tooLarge(announced);
      } else {
        const text = await response.text();
        drawn = maxBytes && text.length > maxBytes ? tooLarge(text.length) : draw(text);
      }
    } catch (error) {
      if (error?.name === 'AbortError') return;
      failure = error?.message || 'No reason was given.';
    }
    if (signal.aborted) return;
    if (drawn) {
      state = 'done';
      mount(body, drawn.nodes);
      // The button that asked is gone: hand its focus to what replaced it.
      if (retry) (drawn.focus ?? body.closest('details')?.querySelector('summary'))?.focus();
      return;
    }
    state = 'failed';
    const again = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => load(true) }, 'Try again');
    mount(body, notice('err', `The ${what} could not be read: ${failure}`, again));
    if (retry) again.focus();
  };
  return h(
    'details',
    {
      class: `${prefix}__names`,
      ontoggle: (event) => {
        if (event.currentTarget.open && state === 'idle') load();
      },
    },
    h('summary', { class: `${prefix}__names-summary` }, summary),
    body,
  );
}

/**
 * "Show the names": reads the list on first opening, draws at most NAMES_MAX names. `side.region` labels
 * the scroll box. `format` (see `listFormat`): file names per line, the image of each JSON line, or the
 * file_name of each image of a COCO file.
 */
export function namesBlock(artifact, side, signal, prefix = 'split', format = 'names') {
  const draw = (text) => {
    const read = format === 'coco' ? readCocoImages(text, NAMES_MAX) : format === 'lines' ? readJsonLineImages(text, NAMES_MAX) : readNames(text, NAMES_MAX);
    const more = read.total - read.names.length;
    const box = read.total
      ? h(
          'div',
          { class: `${prefix}__names-box`, tabindex: '0', role: 'region', 'aria-label': side.region },
          h('ol', { class: `${prefix}__names-list` }, read.names.map((name) => h('li', { class: `${prefix}__name` }, name))),
        )
      : h('p', { class: `${prefix}__note` }, 'The list holds no names.');
    return {
      nodes: [
        h('p', { class: `${prefix}__note` }, more > 0 ? `The first ${fmt.int(read.names.length)} of ${fmt.plural(read.total, 'name')}, in the order of the list.` : `${fmt.plural(read.total, 'name')}, in the order of the list.`),
        box,
        more > 0 ? h('p', { class: `${prefix}__note` }, `and ${fmt.int(more)} more. The downloaded list has all of them.`) : null,
      ],
      focus: read.total ? box : null,
    };
  };
  return lazyDetails({ artifact, signal, summary: 'Show the names', draw, what: 'list', prefix, maxBytes: format === 'coco' ? COCO_MAX_BYTES : 0 });
}

/**
 * "Show the scenes": reads a scene manifest on first opening and lists at most NAMES_MAX scenes. A file
 * of another form is not listed: the block says so and leaves it to the download.
 */
function scenesBlock(artifact, side, signal) {
  const draw = (text) => {
    const read = readSceneManifest(text, NAMES_MAX);
    if (!read) return { nodes: [h('p', { class: 'split__note' }, SCENES_UNKNOWN)], focus: null };
    const more = read.total - read.names.length;
    const box = read.total
      ? h(
          'div',
          { class: 'split__names-box', tabindex: '0', role: 'region', 'aria-label': side.scenes },
          h('ol', { class: 'split__names-list' }, read.names.map((name) => h('li', { class: 'split__name' }, name))),
        )
      : h('p', { class: 'split__note' }, 'The list holds no scenes.');
    const count = more > 0 ? `The first ${fmt.int(read.names.length)} of ${fmt.plural(read.total, 'scene')}` : fmt.plural(read.total, 'scene');
    return {
      nodes: [
        h('p', { class: 'split__note' }, `${count}, in the order of the list. ${SCENE_NAMES}`),
        box,
        more > 0 ? h('p', { class: 'split__note' }, `and ${fmt.int(more)} more. The downloaded list has all of them.`) : null,
      ],
      focus: read.total ? box : null,
    };
  };
  return lazyDetails({ artifact, signal, summary: 'Show the scenes', draw, what: 'list', prefix: 'split', maxBytes: COCO_MAX_BYTES });
}

/** "Images per class": one row per class; a side without an image of the class says "none". */
function classesTable(classes) {
  const cell = (n) => (!Number.isFinite(n) ? absent() : n === 0 ? 'none' : fmt.int(n));
  return h(
    'div',
    { class: 'split__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Images per class' },
    h(
      'table',
      { class: 'table split__table' },
      h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Class'), h('th', { scope: 'col', class: 'num' }, 'Training'), h('th', { scope: 'col', class: 'num' }, 'Validation'))),
      h(
        'tbody',
        null,
        classes.map((c) =>
          h(
            'tr',
            null,
            // A plain cell: the shared `.table th` is the sticky column header.
            h('td', { class: 'split__class', title: Number.isFinite(c.id) ? `Class ID ${c.id}` : null }, isText(c.name) ? c.name : absent()),
            h('td', { class: 'num' }, cell(c.train)),
            h('td', { class: 'num' }, cell(c.val)),
          ),
        ),
      ),
    ),
  );
}

// ───────── renderer ─────────

/**
 * Draw a dataset split result into `host`.
 * @param {HTMLElement} host
 * @param {{job?: object|null, result?: object|null, artifacts?: Array<{path: string, available?: boolean, version?: string, url?: string}>}} options
 * @returns {() => void} cleanup
 */
export function renderSplitDataset(host, { job = null, result = null, artifacts = [] } = {}) {
  const controller = new AbortController();
  const { signal } = controller;
  const r = isObject(result) ? result : {};
  const params = isObject(job?.params) ? job.params : {};
  const counts = isObject(r.counts) ? r.counts : {};
  const files = Array.isArray(artifacts) ? artifacts.filter(isObject) : [];
  const fileNamed = (name) => files.find((a) => baseName(a.path) === name) ?? null;
  /** Paths of `result` that are drawn below; everything else goes into the closing block. */
  const shown = [];

  // 1. What was made.
  const groupBy = isText(r.groupBy) ? r.groupBy : isText(params.groupBy) ? params.groupBy : null;
  const asked = Number.isFinite(r.validationFraction) ? r.validationFraction : params.validationFraction;
  const task = isText(r.task) ? r.task : isText(params.task) ? params.task : null;
  const labelled = LABELLED.has(task);
  /** Paired video split by whole scene: the job or a dataset names the type, or the result counts windows. */
  const video =
    isVideoType(params.datasetType) || (isObject(r.datasets) && (isVideoType(r.datasets.train?.type) || isVideoType(r.datasets.val?.type))) || Number.isFinite(counts.windows);
  /** What is counted: pairs for restoration, images for labelled data (which reports `samples`), windows for video. */
  const unit = video ? 'window' : Number.isFinite(counts.pairs) ? 'pair' : Number.isFinite(counts.samples) || labelled ? 'image' : 'pair';
  /** The dataset option that carries a list: detection reads its COCO file as annotation_file, video its scene manifest. */
  const option = video ? 'video_scene_manifest' : task === 'object_detection' ? 'annotation_file' : 'meta_info';
  const identity = isObject(r.imageIdentity) ? r.imageIdentity : null;
  // Video is split by scene only; another grouping is shown as the job names it.
  const grouping = video ? (groupBy === 'scene' ? GROUPING_VIDEO : null) : groupingWords(groupBy, task, identity?.duplicateRows === 0);
  const share = shareSentence(r.achievedValidationFraction, asked, groupBy, grouping);
  const lead = video ? videoLeadSentence(counts) : leadSentence(counts, unit);
  const sidesCounted = Number.isFinite(counts.train) && Number.isFinite(counts.val);
  for (const key of ['train', 'val']) if (Number.isFinite(counts[key])) shown.push(['counts', key]);
  if (video) {
    if (Number.isFinite(counts.windows)) shown.push(['counts', 'windows']);
    if (sidesCounted && Number.isFinite(counts.scenes)) {
      shown.push(['counts', 'scenes']);
      if (Number.isFinite(counts.trainScenes) && Number.isFinite(counts.valScenes)) shown.push(['counts', 'trainScenes'], ['counts', 'valScenes']);
    }
  } else if (Number.isFinite(counts.pairs)) shown.push(['counts', 'pairs']);
  else if (Number.isFinite(counts.samples)) shown.push(['counts', 'samples']);
  if (share) {
    shown.push(['achievedValidationFraction']);
    if (Number.isFinite(r.validationFraction)) shown.push(['validationFraction']);
  }
  const notes = Array.isArray(r.notes) ? r.notes : [];
  if (notes.length) shown.push(['notes']);

  // 2. Facts.
  const seed = Number.isFinite(r.seed) ? r.seed : params.seed;
  if ('groupBy' in r && isText(r.groupBy)) shown.push(['groupBy']);
  if (Number.isFinite(r.seed)) shown.push(['seed']);
  const facts = [['Kept together', groupBy ? h('span', { title: `groupBy: ${groupBy}` }, grouping ? grouping.kept : groupBy) : absent()]];
  if (video) {
    if (Number.isFinite(r.clipSize)) shown.push(['clipSize']);
    facts.push(['Frames per window', Number.isFinite(r.clipSize) ? fmt.int(r.clipSize) : absent()]);
  } else if (groupBy !== 'pair' && groupBy !== 'sample') {
    const sides = Number.isFinite(counts.trainGroups) && Number.isFinite(counts.valGroups);
    if (sides) shown.push(['counts', 'trainGroups'], ['counts', 'valGroups']);
    if (sides && Number.isFinite(counts.groups)) shown.push(['counts', 'groups']);
    facts.push([
      'Groups on each side',
      sides
        ? `${fmt.int(counts.trainGroups)} for training and ${fmt.int(counts.valGroups)} for validation${Number.isFinite(counts.groups) ? `, out of ${fmt.int(counts.groups)}` : ''}`
        : absent(),
    ]);
  }
  // A seed is typed back as it is written, so it is shown without thousands separators.
  facts.push(['Seed', Number.isFinite(seed) ? h('span', { class: 'num' }, String(seed)) : absent()]);
  if (identity && Number.isFinite(identity.duplicateRows)) {
    // supervised_split.py `_groups`: files with the same size and SHA-256 join one group, so they share a side.
    shown.push(['imageIdentity', 'duplicateRows']);
    facts.push([
      'Identical images',
      identity.duplicateRows === 0
        ? 'No image has an identical copy in this data.'
        : `${fmt.plural(identity.duplicateRows, 'image is', 'images are')} an identical copy of another image; an image and its copies stay on the same side.`,
    ]);
  }
  const physical = video && isObject(r.physicalIdentity) ? r.physicalIdentity : null;
  const sharedText = physical ? sharedFilesSentence(physical.sharedAcrossScenes, counts) : '';
  if (sharedText) {
    // video_split.py `_audit`: scenes that reach one file on disk (same device and inode) join one group.
    shown.push(['physicalIdentity', 'sharedAcrossScenes']);
    const some = physical.sharedAcrossScenes > 0;
    if (some && Number.isFinite(counts.scenes) && Number.isFinite(counts.groups)) shown.push(['counts', 'scenes'], ['counts', 'groups']);
    const given = some && Array.isArray(physical.examples) ? physical.examples : [];
    const examples = given.map(sharedExampleSentence);
    // Examples are drawn only when every one of them reads as a sentence; otherwise they stay in the closing block.
    const drawnExamples = examples.length > 0 && examples.every(Boolean);
    if (drawnExamples) shown.push(['physicalIdentity', 'examples']);
    const truncated = drawnExamples && physical.examplesTruncated === true;
    if (drawnExamples && typeof physical.examplesTruncated === 'boolean') shown.push(['physicalIdentity', 'examplesTruncated']);
    const byIdentity = physical.method === 'device-and-inode';
    if (byIdentity) shown.push(['physicalIdentity', 'method']);
    facts.push([
      'Shared files',
      h(
        'div',
        { class: 'split__shared' },
        h('span', null, sharedText),
        drawnExamples
          ? h('ul', { class: 'split__shared-list', 'aria-label': 'Files used by more than one scene' }, examples.map((text) => h('li', null, text)), truncated ? h('li', null, 'and more') : null)
          : null,
        byIdentity ? h('span', { class: 'split__muted' }, IDENTITY_SCOPE) : null,
      ),
    ]);
  }
  if (isText(r.sourceMetaInfo)) {
    shown.push(['sourceMetaInfo']);
    facts.push(['Made from', h('span', null, 'The list the recipe already used: ', h('span', { class: 'split__path' }, r.sourceMetaInfo))]);
  } else if (isText(r.sourceManifest)) {
    shown.push(['sourceManifest']);
    facts.push(['Made from', h('span', null, task === 'object_detection' ? "The recipe's annotation file: " : 'The list the recipe already used: ', h('span', { class: 'split__path' }, r.sourceManifest))]);
  }

  // Classification: how many images of each class went to each side.
  const classes = Array.isArray(r.classes) ? r.classes.filter(isObject) : [];
  if (classes.length) shown.push(['classes']);
  const withoutVal = classes.filter((c) => c.val === 0).length;

  // 3. The two lists and the record.
  const manifests = isObject(r.manifests) ? r.manifests : {};
  const sums = isObject(r.manifestSha256) ? r.manifestSha256 : {};
  const used = new Set();
  /** The file of a side: the one the result names; without that, the names the service is known to write. */
  const listFile = (side) => {
    const named = isText(manifests[side.key]) ? baseName(manifests[side.key]) : '';
    const candidates = named ? [named] : [side.file, `${side.key}.jsonl`, `${side.key}.json`];
    const artifact = candidates.map(fileNamed).find(Boolean) ?? null;
    return { artifact, name: baseName(artifact?.path) || named || side.file };
  };
  const listFormats = [];
  const sideBlock = (side) => {
    const { artifact, name: listName } = listFile(side);
    // A video list is a scene manifest: a .json name does not mean a COCO file there.
    const format = video ? (listName.toLowerCase().endsWith('.json') ? 'scenes' : null) : listFormat(listName);
    listFormats.push(format);
    if (artifact) used.add(artifact);
    if (isText(manifests[side.key])) shown.push(['manifests', side.key]);
    if (isText(sums[side.key])) shown.push(['manifestSha256', side.key]);
    const n = counts[side.key];
    const readable = artifact && isText(artifact.url) && artifact.available !== false;
    return h(
      'div',
      { class: 'split__file', role: 'group', 'aria-label': `${side.name} list` },
      h(
        'div',
        { class: 'split__file-head' },
        h('span', { class: 'split__file-name' }, side.name),
        h('span', { class: 'split__file-count' }, Number.isFinite(n) ? fmt.plural(n, unit) : `Number of ${unit}s not reported`),
        downloadLink(artifact, side.download, listName),
      ),
      h(
        'dl',
        { class: 'kv split__file-facts' },
        h('dt', null, 'Path on the trainer machine'),
        h('dd', null, pathValue(manifests[side.key])),
        h('dt', null, 'SHA-256 of the list'),
        h('dd', null, checksumValue(sums[side.key])),
      ),
      // A format this view cannot read is offered as a download only.
      readable && format ? (format === 'scenes' ? scenesBlock(artifact, side, signal) : namesBlock(artifact, side, signal, 'split', format)) : null,
    );
  };
  const sideBlocks = SIDES.map(sideBlock);
  const listsNote = listFormats.every((f) => f === 'scenes')
    ? 'Each list is a JSON file that names the scenes of its side and pins their frames.'
    : listFormats.every((f) => f === 'names')
    ? 'Each list is a text file with one ground truth file name per line.'
    : listFormats.every((f) => f === 'lines')
      ? `Each list is a text file with one JSON line per image: where the image${task === 'semantic_segmentation' ? ' and its mask are' : ' is'} inside the data folders.`
      : listFormats.every((f) => f === 'coco')
        ? 'Each list is a COCO annotation file that keeps the images of its side and their annotations.'
        : '';
  const record = fileNamed('split.json');
  if (record) used.add(record);
  if (isText(r.reportPath)) shown.push(['reportPath']);
  const recordBlock = h(
    'div',
    { class: 'split__file', role: 'group', 'aria-label': 'Record of this split' },
    h('div', { class: 'split__file-head' }, h('span', { class: 'split__file-name' }, 'Record of this split'), downloadLink(record, 'Download the record', 'split.json')),
    h('p', { class: 'split__note' }, 'A JSON file with what this page shows: seed, grouping, counts and the checksum of each list.'),
    h('dl', { class: 'kv split__file-facts' }, h('dt', null, 'Path on the trainer machine'), h('dd', null, pathValue(r.reportPath))),
  );
  // The recipe the service wrote for labelled data and for paired video (`recipe.yml`, `recipePath`).
  const generated = isText(r.recipePath);
  const recipeFile = generated ? fileNamed(baseName(r.recipePath)) : null;
  if (recipeFile) used.add(recipeFile);
  if (generated) shown.push(['recipePath']);
  const recipeReadable = Boolean(recipeFile && isText(recipeFile.url) && recipeFile.available !== false);
  const others = files.filter((a) => !used.has(a) && isText(a.url) && a.available !== false);

  // 4. Recipe lines.
  const datasets = isObject(r.datasets) ? r.datasets : {};
  const train = isObject(datasets.train) ? datasets.train : null;
  const val = isObject(datasets.val) ? datasets.val : null;
  const fragment = {};
  if (train) fragment.train = train;
  if (val) fragment.val = val;
  if (train) shown.push(['datasets', 'train']);
  if (val) shown.push(['datasets', 'val']);
  const yamlText = train || val ? toYaml({ datasets: fragment }) : '';
  const rootKeys = train ? Object.keys(train).filter((key) => key.startsWith('dataroot')) : [];
  const sameFolders = Boolean(train && val && rootKeys.length && rootKeys.every((key) => sameRoots(train[key], val[key])));
  const linesId = uid('split-lines');
  const linesText =
    train && val
      ? `These are the recipe's training dataset and a validation dataset${sameFolders ? ' on the same folders' : ''}, each limited to its list by ${option}; nothing was copied or moved.`
      : `The job reported only one of the two datasets. It is limited to its list by ${option}; nothing was copied or moved.`;

  // 5. The one primary action. A result with a generated recipe opens that file; a restoration result
  // without one puts the split into the open recipe (through the render route); otherwise there is none.
  const mode = generated ? (recipeReadable ? 'generated' : 'none') : !labelled && !video && train && val && isText(train.meta_info) ? 'render' : 'none';
  const canApply = mode !== 'none';
  const buttonText = mode === 'generated' ? 'Open the generated recipe' : 'Open the recipe with this split';
  const noAction = generated
    ? `The generated recipe cannot be read from here, so it cannot be opened.${yamlText ? ' Copy the recipe lines below into your recipe.' : ''}`
    : labelled || video
      ? `The job wrote no recipe for this split.${yamlText ? ' Copy the recipe lines below into your recipe.' : ''}`
      : 'The job did not report both datasets with their lists, so there is nothing to put into a recipe.';
  const openButton = h('button', { class: 'btn btn--primary', type: 'button', hidden: true }, buttonText);
  // Focusable by script only: it takes the focus of the button when the button goes away under it.
  const why = h('p', { class: 'split__why', tabindex: '-1', hidden: true });
  const print = h('p', { class: 'split__print', hidden: true });
  const errorSlot = h('div', { class: 'split__error' });
  let busy = false;
  let gone = false;
  let drawn = '';
  /** Where each list of this split is, as the result names it (the list itself, and the dataset that reads it). */
  const lists = {
    train: [manifests.train, train?.[option]].filter(isText),
    val: [manifests.val, val?.[option]].filter(isText),
  };
  const standingNow = () => {
    if (mode !== 'generated') return recipeStanding(recipe.get(), train, lists, option);
    // The generated recipe can always be opened; it only gives way when the open recipe reads these lists already.
    const s = recipe.get();
    const current = isObject(s) && s.phase === 'open' && isObject(s.config) && s.configCurrent !== false;
    return current && usesSplit(s.config.datasets, lists, option) ? { kind: 'applied', name: isText(s.name) ? s.name : '' } : { kind: 'match', name: '' };
  };
  /** Follows the recipe store by changing the three nodes in place: nothing here is rebuilt. */
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
    openButton.textContent = busy ? (mode === 'generated' ? 'Reading the recipe…' : 'Preparing…') : buttonText;
    print.hidden = !offered;
    print.textContent = !offered ? '' : mode === 'generated' ? GENERATED_PRINT : `${standing.name ? `The open recipe is ${standing.name}. ` : ''}${NOT_SAVED}`;
    why.hidden = offered;
    // "Already uses this split" is a quiet remark; every other sentence says why the button is missing.
    why.className = standing.kind === 'applied' ? 'split__why split__note' : 'split__why';
    why.textContent = standingSentence(standing);
    if (!offered) mount(errorSlot, null);
    // The button that had the focus is gone: hand the focus to the sentence that took its place.
    if (!offered && hadFocus) why.focus();
  };
  const openWithSplit = async () => {
    if (busy || gone) return;
    const standing = standingNow();
    if (standing.kind !== 'match') {
      update();
      return;
    }
    if (standing.hasVal) {
      const ok = await confirmDialog({
        title: 'Replace the validation dataset?',
        body: `${standing.name ? `The recipe ${standing.name}` : 'The open recipe'} already has a validation dataset (datasets.val). Opening it with this split replaces that dataset with the validation dataset of this split. Nothing is saved until you save the recipe.`,
        confirmLabel: 'Replace and open',
      });
      if (!ok || gone || busy) return;
      if (standingNow().kind !== 'match') {
        update();
        return;
      }
    }
    const before = recipe.get();
    const edits = [
      { path: ['datasets', 'train', 'meta_info'], value: train.meta_info },
      { path: ['datasets', 'val'], value: val },
    ];
    busy = true;
    mount(errorSlot, null);
    update();
    try {
      const answer = await api.render(before.yaml, edits, { signal });
      if (gone) return;
      if (!answer || typeof answer.yaml !== 'string') throw new Error('The service returned no recipe text for this split.');
      if (recipe.get().yaml !== before.yaml) throw new Error('The open recipe changed while the split was being prepared. Nothing was opened. Try again.');
      // Made from the open recipe's text as it is now (checked just above): its unsaved changes are kept.
      await openAsRecipe({ yaml: answer.yaml, name: isText(before.name) ? before.name : '', label: 'validation split', derived: true });
    } catch (error) {
      if (error?.name === 'AbortError' || gone) return;
      const issues = Array.isArray(error?.issues) ? error.issues.filter((issue) => isObject(issue) && isText(issue.message)) : [];
      mount(
        errorSlot,
        notice('err', [
          `The recipe was not opened: ${error?.message || 'the service gave no reason.'}`,
          issues.length ? h('ul', { class: 'split__notes' }, issues.map((issue) => h('li', null, isText(issue.path) ? `${issue.path}: ${issue.message}` : issue.message))) : null,
        ]),
      );
    } finally {
      busy = false;
      update();
    }
  };
  /** Reads the generated recipe and opens it in place of whatever recipe is open. */
  const openGenerated = async (retry = false) => {
    if (busy || gone) return;
    busy = true;
    // A retry keeps its notice (and the focus on "Try again") until the new answer is there.
    if (!retry) mount(errorSlot, null);
    update();
    let failure = '';
    try {
      let response;
      try {
        response = await fetch(recipeFile.url, { signal });
      } catch (cause) {
        if (cause?.name === 'AbortError') throw cause;
        throw new Error('The trainer service did not answer.');
      }
      if (!response.ok) throw new Error(await refusal(response));
      const text = await response.text();
      if (gone) return;
      if (!text.trim()) throw new Error('The file is empty.');
      // Not `derived`: this text replaces the open recipe, so the app asks before unsaved changes are
      // discarded. No name is passed: the file has its own `name:` line, which the Recipe screen shows.
      await openAsRecipe({ yaml: text, name: '', label: 'validation split' });
    } catch (error) {
      if (error?.name === 'AbortError' || gone) return;
      failure = error?.message || 'No reason was given.';
    } finally {
      busy = false;
      update();
    }
    if (gone) return;
    if (failure) {
      const again = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => openGenerated(true) }, 'Try again');
      mount(errorSlot, notice('err', `The generated recipe could not be read: ${failure}`, again));
      if (retry) again.focus();
      return;
    }
    mount(errorSlot, null);
    if (retry && !openButton.hidden) openButton.focus();
  };
  const onOpen = () => (mode === 'generated' ? openGenerated() : openWithSplit());
  openButton.addEventListener('click', onOpen);

  // 6. Whatever the job reported that is not drawn above.
  const rest = remainder(r, shown);

  const root = h(
    'div',
    { class: 'split' },
    h('p', { class: 'split__lead' }, lead),
    share ? h('p', { class: 'split__note' }, share) : null,
    notes.length ? h('ul', { class: 'split__notes', 'aria-label': 'Notes from the service' }, notes.map((note) => h('li', null, typeof note === 'string' ? note : JSON.stringify(note)))) : null,
    h('dl', { class: 'kv split__facts' }, facts.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])),
    // Not said for video: its groups come from file identity on disk, which another machine need not share.
    video ? null : h('p', { class: 'split__note' }, REPEATABLE),
    classes.length
      ? [
          h('h4', { class: 'split__title' }, 'Images per class'),
          withoutVal ? h('p', { class: 'split__note' }, `${fmt.plural(withoutVal, 'class has', 'classes have')} no validation image.`) : null,
          classesTable(classes),
        ]
      : null,

    // What to do with the split comes before its evidence: the one primary button, then the files.
    h('h4', { class: 'split__title' }, generated ? 'The recipe with this split' : 'Use this split'),
    generated ? h('p', { class: 'split__note' }, `${GENERATED} ${generatedKeysText(r.recipeDatasetKeys)}`) : null,
    h('div', { class: 'split__action' }, canApply ? [openButton, why, print, errorSlot] : h('p', { class: 'split__why' }, noAction)),
    generated
      ? h(
          'div',
          { class: 'split__file', role: 'group', 'aria-label': 'Generated recipe' },
          h('div', { class: 'split__file-head' }, h('span', { class: 'split__file-name' }, 'Generated recipe'), downloadLink(recipeFile, 'Download the recipe', 'recipe.yml')),
          h('dl', { class: 'kv split__file-facts' }, h('dt', null, 'Path on the trainer machine'), h('dd', null, pathValue(r.recipePath))),
        )
      : null,

    h('h4', { class: 'split__title' }, 'The two lists'),
    listsNote ? h('p', { class: 'split__note' }, listsNote) : null,
    sideBlocks,
    recordBlock,
    others.length
      ? h('ul', { class: 'split__others', 'aria-label': 'Other files of the job' }, others.map((a) => h('li', null, h('a', { class: 'split__download', href: a.url, download: baseName(a.path) || 'output' }, String(a.path ?? 'output')))))
      : null,

    h('div', { class: 'split__head' }, h('h4', { class: 'split__title', id: linesId }, 'Recipe lines'), yamlText ? copyButton('Copy', yamlText, 'Copied the recipe lines.') : null),
    yamlText
      ? [
          h('p', { class: 'split__note' }, linesText),
          // Shown without the final line break (it would draw an empty last line); "Copy" keeps it.
          h('div', { class: 'split__yaml', tabindex: '0', role: 'region', 'aria-labelledby': linesId }, highlightedPre(yamlText.trimEnd())),
          h('p', { class: 'split__note' }, "Written as YAML from the job's answer. The paths are absolute: when the folders or the lists are moved, change them in the recipe too."),
        ]
      : h('p', { class: 'split__note' }, 'The job did not report the two datasets, so there are no recipe lines.'),

    rest
      ? h(
          'details',
          { class: 'split__raw' },
          h('summary', { class: 'split__raw-summary' }, 'Everything else the job reported'),
          h('pre', { class: 'split__json', tabindex: '0' }, JSON.stringify(rest, null, 2)),
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
    openButton.removeEventListener('click', onOpen);
    root.remove();
  };
}
