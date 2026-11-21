// Result of a data audit ("task-data" analysis job): what the service counted in full for each dataset of a
// classification, segmentation or detection recipe, kept apart from what it checked by opening a sample of
// the files. Everything shown comes from the result; absent fields stay absent.

import { h, fmt } from '../dom.js';
import { api } from '../api.js';
import { badge, issueList, notice } from '../ui.js';

/** Longest edge of an example's preview, in px (the service accepts 16 to 1536). */
const THUMB_EDGE = 96;

const ROLE = { train: 'Training set', val: 'Validation set', test: 'Test set' };
const SEGMENTATION = 'semantic_segmentation';
const CLASSIFICATION = 'image_classification';
const DETECTION = 'object_detection';

/** What one audited row is, per task: segmentation opens an image and its mask together. */
const UNITS = {
  [SEGMENTATION]: { one: 'image and mask pair', many: 'image and mask pairs' },
  [CLASSIFICATION]: { one: 'image', many: 'images' },
  [DETECTION]: { one: 'image', many: 'images' },
};
const FILES = { one: 'file', many: 'files' };
const unitFor = (task) => UNITS[task] ?? FILES;

const isNum = Number.isFinite;
const plural = (n, unit) => fmt.plural(n, unit.one, unit.many);

/** "Training set", "Validation set", "Test set" or "Dataset", from the dataset key (phase before the first "_"). */
export function datasetRole(id) {
  return ROLE[String(id ?? '').split('_')[0]] ?? 'Dataset';
}

/** "the training set", or "the validation set val_2" when the key is more than the phase. */
function roleInSentence(id) {
  const key = String(id ?? '');
  const role = datasetRole(key).toLowerCase();
  return key && key !== key.split('_')[0] ? `the ${role} ${key}` : role === 'dataset' && key ? `the dataset ${key}` : `the ${role}`;
}

/**
 * One plain sentence for a running job's progress ({stage, splitId, completed, total}), e.g.
 * "Checking the training set: 40 of 100 files"; '' when absent or of an unknown stage.
 * @param {object|null} progress
 * @param {string} [task] the recipe's task, when known, for the unit ("image and mask pairs")
 */
export function taskDataProgressText(progress, task) {
  if (!progress || typeof progress !== 'object') return '';
  const set = progress.splitId ? roleInSentence(progress.splitId) : 'the data';
  const unit = unitFor(task);
  const done = progress.completed;
  const total = progress.total;
  if (progress.stage === 'inventory') {
    return isNum(done) && done > 0 ? `Listing ${set}: ${plural(done, unit)} found so far` : `Listing ${set}`;
  }
  if (progress.stage === 'auditing') {
    if (isNum(done) && isNum(total)) return `Checking ${set}: ${fmt.int(done)} of ${plural(total, unit)}`;
    if (isNum(done) && done > 0) return `Checking ${set}: ${plural(done, unit)} opened`;
    return `Checking ${set}`;
  }
  return '';
}

/**
 * How much of a dataset was opened, from the audit block {eligible, attempted} only:
 * "Opened 100 of 2,340 images." / "Opened every image (2,340)." / "No files were opened."
 */
export function coverageText(audit, task) {
  if (!audit || typeof audit !== 'object') return '';
  const unit = unitFor(task);
  const { eligible, attempted } = audit;
  if (eligible === 0) return `There were no ${unit.many} to open.`;
  if (!isNum(attempted) || attempted === 0) return 'No files were opened.';
  if (isNum(eligible) && attempted >= eligible) return `Opened every ${unit.one} (${fmt.int(eligible)}).`;
  if (isNum(eligible)) return `Opened ${fmt.int(attempted)} of ${plural(eligible, unit)}.`;
  return `Opened ${plural(attempted, unit)}.`;
}

/** The audit method in plain words; an unknown method is named as the service gave it. */
export function methodText(method, task) {
  if (method === 'image-headers-and-mask-pixels') {
    return "Each opened image's header was read for its size, and its mask was decoded to count the class value of every pixel.";
  }
  if (method === 'still-image-headers') {
    return task === DETECTION
      ? "Each opened image's header was read for its size, which was compared with the size in the annotation file. Pixels were not decoded."
      : "Each opened image's header was read for its size and frame count. Pixels were not decoded.";
  }
  return typeof method === 'string' && method ? `Method: ${method}.` : '';
}

/** "Opened every file" means a full count; anything less is a sample. */
const openedAll = (audit) => isNum(audit?.eligible) && isNum(audit?.attempted) && audit.attempted >= audit.eligible;

const sizeText = (size) => (Array.isArray(size) && size.length === 2 ? fmt.dims(size[0], size[1]) : '');
const sameSize = (a, b) => Array.isArray(a) && Array.isArray(b) && a[0] === b[0] && a[1] === b[1];
const baseName = (path) => {
  const text = String(path ?? '');
  return text.slice(Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\')) + 1) || text;
};

// ───────── small builders ─────────

/** Definition rows; a value that is not a number is left out (absent stays absent). */
function figures(rows) {
  const items = rows.filter(([, value]) => isNum(value));
  if (!items.length) return null;
  return h(
    'dl',
    { class: 'kv taskdata__figures' },
    items.map(([label, value]) => [h('dt', null, label), h('dd', { class: 'num' }, fmt.int(value))]),
  );
}

/** A thin neutral bar and the percentage, for a part of a stated total. */
function shareCell(part, total) {
  const pct = fmt.percent(part, total);
  if (!pct) return h('td', { class: 'taskdata__share' });
  const width = Math.max(0, Math.min(100, (part / total) * 100));
  return h(
    'td',
    { class: 'taskdata__share' },
    h('span', { class: 'taskdata__bar', 'aria-hidden': 'true' }, h('span', { class: 'taskdata__barfill', style: { width: `${width}%` } })),
    h('span', { class: 'taskdata__pct' }, pct),
  );
}

const numCell = (value) => h('td', { class: 'num' }, isNum(value) ? fmt.int(value) : h('span', { class: 'muted' }, 'Not reported'));
const flagsCell = (flags) => h('td', { class: 'taskdata__flags' }, flags.filter(Boolean).map((flag) => badge(flag, 'warn')));

function table(caption, heads, rows) {
  return h(
    'div',
    { class: 'taskdata__tablewrap' },
    h(
      'table',
      { class: 'table taskdata__table' },
      h('caption', { class: 'taskdata__caption' }, caption),
      h('thead', null, h('tr', null, heads.map(([text, numeric]) => h('th', { scope: 'col', class: numeric ? 'num' : null }, text)))),
      h('tbody', null, rows),
    ),
  );
}

const group = (title, ...children) => h('div', { class: 'taskdata__group' }, h('h4', { class: 'taskdata__grouptitle' }, title), children);
const line = (text, cls = 'taskdata__line') => (text ? h('p', { class: cls }, text) : null);

// ───────── per dataset ─────────

function sourceLines(split, task) {
  const rows = [];
  const path = (label, value) => {
    if (typeof value === 'string' && value) rows.push(h('dt', null, label), h('dd', { class: 'mono taskdata__path' }, value));
  };
  const roots = Array.isArray(split.roots) ? split.roots : [];
  if (task === SEGMENTATION) {
    for (const root of roots) {
      const suffix = roots.length > 1 ? ` ${root.id}` : '';
      path(`Images${suffix}`, root.images);
      path(`Masks${suffix}`, root.masks);
    }
  } else if (task === DETECTION) {
    path('Image folder', split.imageRoot);
    path('Annotation file', split.annotationFile);
  } else {
    for (const root of roots) path(roots.length > 1 ? `Class folders ${root.id}` : 'Class folders', root.path);
  }
  return rows.length ? h('dl', { class: 'kv taskdata__sources' }, rows) : null;
}

function problems(listed, { truncated, sampled }) {
  const errors = listed.filter((issue) => issue.level === 'error').length;
  const warnings = listed.length - errors;
  if (!listed.length) {
    if (truncated) return line('No problems of this dataset are among those listed.', 'taskdata__quiet');
    return line(sampled ? 'No problems found in the files that were opened.' : 'No problems found.', 'taskdata__quiet');
  }
  return h(
    'div',
    { class: 'taskdata__problems' },
    h(
      'p',
      { class: 'taskdata__problemcount' },
      errors ? badge(fmt.plural(errors, 'error'), 'err') : null,
      warnings ? badge(fmt.plural(warnings, 'warning'), 'warn') : null,
      truncated ? h('span', { class: 'taskdata__quiet' }, 'listed for this dataset') : null,
    ),
    h('div', { class: 'taskdata__issues' }, issueList(listed)),
  );
}

function incompleteNote(split, task) {
  if (split.inventoryComplete !== false) return null;
  if (task === DETECTION && split.counts === null) return notice('err', 'The annotation file could not be read, so nothing was counted. The problems above say why.');
  if (task === DETECTION) return notice('warn', 'The image folder could not be read, so the image paths were not checked. The problems above say why.');
  return notice('warn', 'Some folders could not be read, so these counts are incomplete. The problems above say which.');
}

function classificationCounted(split) {
  const counts = split.counts ?? {};
  const classes = Array.isArray(split.classes) ? split.classes : [];
  const unknown = Array.isArray(split.unknownClasses) ? split.unknownClasses : null;
  const hasIds = classes.some((record) => isNum(record.id));
  const total = counts.images;
  const rows = classes.map((record) => {
    const dirs = Array.isArray(record.directories) ? record.directories : [];
    return h(
      'tr',
      null,
      h('td', { class: 'mono taskdata__name' }, String(record.name ?? '')),
      hasIds ? h('td', { class: 'num' }, isNum(record.id) ? String(record.id) : h('span', { class: 'muted' }, 'None')) : null,
      numCell(record.images),
      shareCell(record.images, total),
      flagsCell([
        !dirs.length ? 'No folder' : record.images === 0 ? 'No images' : '',
        dirs.some((dir) => dir.inventoryComplete === false) ? 'Folder not fully read' : '',
        unknown?.includes(record.name) ? "Not in the recipe's class list" : '',
      ]),
    );
  });
  const heads = [['Class'], ...(hasIds ? [['ID', true]] : []), ['Images', true], ['Share of images'], ['Notes']];
  const comparison = split.classComparison;
  let compared = null;
  if (comparison && typeof comparison === 'object') {
    const missing = Array.isArray(comparison.missingClasses) ? comparison.missingClasses : [];
    const extra = Array.isArray(comparison.extraClasses) ? comparison.extraClasses : [];
    const against = roleInSentence(comparison.comparedTo);
    compared =
      missing.length || extra.length
        ? notice('warn', [`Compared with ${against}:`, missing.length ? ` missing ${missing.join(', ')}.` : '', extra.length ? ` Not in it: ${extra.join(', ')}.` : ''].join(''))
        : line(`Same class folders as ${against}.`, 'taskdata__quiet');
  }
  return [
    figures([
      ['Images', counts.images],
      ['Class folders', counts.classFolders],
      ['Empty class folders', counts.emptyFolders],
      ['Images outside a class folder', counts.unlabeledImages],
    ]),
    rows.length ? table('Images per class, counted in full', heads, rows) : line('No class folders were found.', 'taskdata__quiet'),
    compared,
  ];
}

function segmentationCounted(split) {
  const counts = split.counts ?? {};
  const roots = Array.isArray(split.roots) ? split.roots : [];
  const perRoot =
    roots.length > 1
      ? table(
          'Per folder pair',
          [['Pair'], ['Images', true], ['Masks', true], ['Pairs', true], ['Images without a mask', true], ['Masks without an image', true], ['Notes']],
          roots.map((root) =>
            h(
              'tr',
              null,
              h('td', null, String(root.id ?? '')),
              numCell(root.counts?.images),
              numCell(root.counts?.masks),
              numCell(root.counts?.pairs),
              numCell(root.counts?.missingMasks),
              numCell(root.counts?.missingImages),
              flagsCell([root.inventoryComplete === false ? 'Not fully read' : '']),
            ),
          ),
        )
      : null;
  return [
    figures([
      ['Images', counts.images],
      ['Masks', counts.masks],
      ['Image and mask pairs', counts.pairs],
      ['Images without a mask', counts.missingMasks],
      ['Masks without an image', counts.missingImages],
      ['Names that match more than one file', counts.duplicateStems],
    ]),
    perRoot,
  ];
}

function categoryFlags(category, configured) {
  return [
    category.declared === false ? 'Not declared in the annotation file' : '',
    category.declared !== false && category.boxes === 0 ? 'Declared but unused' : '',
    configured && category.declared !== false && (category.configuredName === null || category.configuredName === undefined) ? "Not in the recipe's categories" : '',
    typeof category.configuredName === 'string' && typeof category.name === 'string' && category.configuredName !== category.name ? `Named ${category.configuredName} in the recipe` : '',
  ];
}

function detectionCounted(split) {
  if (!split.counts || typeof split.counts !== 'object') return [];
  const counts = split.counts;
  const categories = Array.isArray(split.categories) ? split.categories : [];
  const configured = categories.some((category) => typeof category.configuredName === 'string');
  const hasLabels = categories.some((category) => isNum(category.label));
  const rows = categories.map((category) =>
    h(
      'tr',
      null,
      h('td', { class: 'taskdata__name' }, category.name ?? h('span', { class: 'muted' }, 'No name')),
      h('td', { class: 'num' }, isNum(category.id) ? String(category.id) : ''),
      hasLabels ? h('td', { class: 'num' }, isNum(category.label) ? String(category.label) : h('span', { class: 'muted' }, 'None')) : null,
      numCell(category.boxes),
      numCell(category.ignoredBoxes),
      shareCell(category.boxes, counts.boxes),
      flagsCell(categoryFlags(category, configured)),
    ),
  );
  const heads = [['Category'], ['ID', true], ...(hasLabels ? [['Training label', true]] : []), ['Boxes', true], ['Ignored boxes', true], ['Share of boxes'], ['Notes']];
  return [
    figures([
      ['Image records', counts.images],
      ['Distinct images', counts.distinctImages],
      ['Categories declared', counts.categories],
      ['Annotations', counts.annotations],
      ['Boxes', counts.boxes],
      ['Usable boxes', counts.usableBoxes],
      ['Ignored boxes', counts.ignoredBoxes],
      ['Crowd boxes', counts.crowdBoxes],
      ['Boxes marked ignore', counts.explicitIgnoreBoxes],
      ['Boxes clipped to the image', counts.clippedBoxes],
      ['Invalid annotations', counts.invalidAnnotations],
      ['Images without annotations', counts.emptyImages],
      ['Images with only ignored boxes', counts.ignoredOnlyImages],
      ['Images without usable boxes', counts.imagesWithoutUsableBoxes],
      ['Image paths checked', counts.pathsChecked],
      ['Image path errors', counts.pathErrors],
    ]),
    rows.length ? table('Boxes per category, counted in full', heads, rows) : line('The annotation file declares no categories.', 'taskdata__quiet'),
  ];
}

function classificationChecked(split) {
  const classes = (Array.isArray(split.classes) ? split.classes : []).filter((record) => isNum(record.headersRead) && (record.directories?.length ?? 0) > 0);
  if (!classes.length) return null;
  return table(
    'Headers read in the opened images, per class',
    [['Class'], ['Headers read', true]],
    classes.map((record) => h('tr', null, h('td', { class: 'mono taskdata__name' }, String(record.name ?? '')), numCell(record.headersRead))),
  );
}

function segmentationChecked(split) {
  const parts = [];
  const g = split.geometry;
  if (g && typeof g === 'object') {
    parts.push(
      figures([
        ['Pairs with sizes compared', g.checked],
        ['Image and mask the same size', g.matched],
        ['Image and mask of different sizes', g.mismatched],
        ['Smaller than the training crop', g.tooSmall],
      ]),
    );
  }
  const m = split.maskPixels;
  if (m && typeof m === 'object') {
    parts.push(figures([['Masks decoded', m.auditedMasks], ['Pixels in the decoded masks', m.pixels]]));
    const total = m.pixels;
    const classes = Array.isArray(m.classes) ? m.classes : [];
    const rows = classes.map((record) =>
      h(
        'tr',
        null,
        h('td', { class: 'mono taskdata__name' }, String(record.name ?? '')),
        h('td', { class: 'num' }, isNum(record.id) ? String(record.id) : ''),
        numCell(record.pixels),
        shareCell(record.pixels, total),
        flagsCell([record.pixels === 0 ? 'None in the decoded masks' : '']),
      ),
    );
    if (m.ignored && typeof m.ignored === 'object') {
      rows.push(
        h(
          'tr',
          null,
          h('td', { class: 'taskdata__name' }, 'Ignored'),
          h('td', { class: 'num' }, isNum(m.ignored.id) ? String(m.ignored.id) : ''),
          numCell(m.ignored.pixels),
          shareCell(m.ignored.pixels, total),
          flagsCell([]),
        ),
      );
    }
    const other = m.unknown ?? m.unmapped;
    const isUnknown = Boolean(m.unknown);
    const values = Array.isArray(other?.values) ? other.values : [];
    for (const value of values) {
      rows.push(
        h(
          'tr',
          null,
          h('td', { class: 'taskdata__name' }, isUnknown ? 'Unknown value' : 'Value'),
          h('td', { class: 'num' }, isNum(value.id) ? String(value.id) : ''),
          numCell(value.pixels),
          shareCell(value.pixels, total),
          flagsCell([isUnknown ? 'Not a class of the recipe' : '']),
        ),
      );
    }
    if (rows.length) parts.push(table('Pixels per class value in the decoded masks', [['Class'], ['Value', true], ['Pixels', true], ['Share of pixels'], ['Notes']], rows));
    if (!isUnknown && m.unmapped) parts.push(line('The recipe names no classes, so the values are listed as found.', 'taskdata__quiet'));
    if (other?.valuesTruncated === true || (isNum(other?.unlistedPixels) && other.unlistedPixels > 0)) {
      parts.push(notice('warn', `${fmt.plural(other.unlistedPixels, 'pixel')} have values that are not listed: the service lists a limited number of distinct values.`));
    }
  }
  return parts;
}

function sampleItem(sample, task, thumbs) {
  const flags = [];
  let name;
  let title;
  let facts = [];
  if (task === SEGMENTATION) {
    name = baseName(sample.image);
    title = [sample.image, sample.mask].filter(Boolean).join('\n');
    const image = sizeText(sample.imageSize);
    const mask = sizeText(sample.maskSize);
    if (image) facts.push(image);
    if (mask && !sameSize(sample.imageSize, sample.maskSize)) facts.push(`mask ${mask}`);
    if (typeof sample.maskMode === 'string' && sample.maskMode) facts.push(`mask mode ${sample.maskMode}`);
    if (isNum(sample.ignoredPixels) && sample.ignoredPixels > 0) facts.push(`${fmt.plural(sample.ignoredPixels, 'ignored pixel')}`);
    if (sample.geometryMatches === false) flags.push('Geometry differs');
    if (isNum(sample.unknownPixels) && sample.unknownPixels > 0) flags.push(fmt.plural(sample.unknownPixels, 'unknown pixel'));
    if (sample.imageSize === null) flags.push('Image header not read');
    if (sample.ignoredPixels === null) flags.push('Mask not decoded');
  } else if (task === DETECTION) {
    name = typeof sample.fileName === 'string' && sample.fileName ? sample.fileName : baseName(sample.path);
    title = sample.path ?? null;
    const size = sizeText(sample.size);
    if (size) facts.push(size);
    if (isNum(sample.boxes)) facts.push(fmt.plural(sample.boxes, 'box', 'boxes'));
    if (isNum(sample.ignoredBoxes) && sample.ignoredBoxes > 0) facts.push(`${fmt.int(sample.ignoredBoxes)} ignored`);
    if (isNum(sample.clippedBoxes) && sample.clippedBoxes > 0) flags.push(sample.clippedBoxes === 1 ? 'Clipped box' : `${fmt.int(sample.clippedBoxes)} clipped boxes`);
    if (sample.empty === true) flags.push('No annotations');
    if (sample.size === null) flags.push('Header not read');
    else if (Array.isArray(sample.size) && Array.isArray(sample.declaredSize) && !sameSize(sample.size, sample.declaredSize)) {
      flags.push(`Annotation file says ${sizeText(sample.declaredSize)}`);
    }
  } else {
    name = baseName(sample.path);
    title = sample.path ?? null;
    const size = sizeText(sample.size);
    if (typeof sample.className === 'string') facts.push(sample.className);
    if (size) facts.push(size);
    if (sample.size === null) flags.push('Header not read');
  }
  // A small preview, loaded when the row scrolls into view; the row says everything without it.
  const path = task === SEGMENTATION ? sample.image : sample.path;
  const readable = task === SEGMENTATION ? Array.isArray(sample.imageSize) : Array.isArray(sample.size);
  let thumb = null;
  if (thumbs && typeof path === 'string' && path && readable) {
    thumb = h('span', { class: 'taskdata__thumb', 'aria-hidden': 'true' });
    thumbs.push({ el: thumb, path });
  }
  return h(
    'li',
    { class: 'taskdata__sample' },
    thumb,
    h('span', { class: 'mono taskdata__samplename', title }, name),
    h('span', { class: 'taskdata__samplefacts' }, facts.join(', ')),
    flags.length ? h('span', { class: 'taskdata__flags' }, flags.map((flag) => badge(flag, 'warn'))) : null,
  );
}

function examples(split, task, context) {
  const limits = context.limits;
  const samples = Array.isArray(split.samples) ? split.samples : [];
  const attempted = split.audit?.attempted;
  if (!samples.length) return isNum(attempted) && attempted > 0 ? line('No examples were kept.', 'taskdata__quiet') : null;
  const kept = isNum(limits?.samplesPerSplit) ? ` The service keeps at most ${fmt.plural(limits.samplesPerSplit, 'example')} per dataset.` : '';
  return h(
    'div',
    { class: 'taskdata__examples' },
    h('h5', { class: 'taskdata__subtitle' }, 'Examples of opened files'),
    split.samplesTruncated === true
      ? line(isNum(attempted) ? `Listing ${fmt.int(samples.length)} of the ${fmt.int(attempted)} opened.${kept}` : `More files were opened than are listed.${kept}`, 'taskdata__quiet')
      : null,
    h('ul', { class: 'taskdata__samples' }, samples.map((sample) => sampleItem(sample, task, context.thumbs))),
  );
}

/** True for a result about tables of records (a split with named columns), not image files. */
const ofRecords = (result) => Array.isArray(result?.splits) && result.splits.some((split) => Array.isArray(split?.columns));

/** A dataset of numeric records (one CSV file): its named columns with their ranges, and the records read. */
function recordsSection(split, listed, context, { role, key, name, sampled }) {
  const audit = split.audit ?? null;
  const columns = split.columns.filter((column) => column && typeof column.name === 'string');
  const ROLE = { feature: 'Feature', target: 'Target' };
  const num = (value) => (isNum(value) ? String(value) : '');
  const columnRows = columns.map((column) =>
    h('tr', null, h('td', { class: 'mono taskdata__name' }, column.name), h('td', null, ROLE[column.role] ?? String(column.role ?? '')), h('td', { class: 'num' }, num(column.min)), h('td', { class: 'num' }, num(column.max))),
  );
  const samples = Array.isArray(split.samples) ? split.samples.filter((sample) => sample && sample.values && typeof sample.values === 'object') : [];
  const sampleRows = samples.map((sample) => h('tr', null, h('td', { class: 'num' }, num(sample.record)), columns.map((column) => h('td', { class: 'num' }, num(sample.values[column.name])))));
  const read =
    isNum(audit?.completed) && isNum(audit?.eligible)
      ? audit.completed === audit.eligible
        ? `Read every record (${fmt.int(audit.completed)}).`
        : `Read ${fmt.int(audit.completed)} of ${fmt.int(audit.eligible)} records.`
      : isNum(split.counts?.recordsInspected)
        ? `Read ${fmt.int(split.counts.recordsInspected)} records.`
        : '';
  return h(
    'section',
    { class: 'taskdata__set', 'aria-label': `${role} ${key}` },
    h(
      'header',
      { class: 'taskdata__sethead' },
      h('h3', { class: 'taskdata__settitle' }, role, ' ', h('span', { class: 'mono taskdata__key' }, key), name ? h('span', { class: 'taskdata__setname' }, name) : null),
      typeof split.source === 'string' && split.source ? h('p', { class: 'mono taskdata__quiet' }, split.source) : null,
    ),
    problems(listed, { truncated: context.truncated, sampled }),
    line(read, 'taskdata__coverage'),
    columnRows.length ? table('Columns, in the order the recipe names them', [['Column'], ['Role'], ['Smallest', true], ['Largest', true]], columnRows) : null,
    sampleRows.length ? table(split.samplesTruncated === true ? 'The first records read' : 'The records read', [['Record', true], ...columns.map((column) => [column.name, true])], sampleRows) : null,
  );
}

function datasetSection(split, task, listed, context) {
  const audit = split.audit ?? null;
  const sampled = !openedAll(audit) || audit?.complete === false;
  const role = datasetRole(split.id);
  const key = String(split.id ?? '');
  const name = typeof split.name === 'string' && split.name && split.name !== key ? split.name : '';
  // A table of numeric records has columns and rows, not class folders or opened image files.
  if (Array.isArray(split.columns)) return recordsSection(split, listed, context, { role, key, name, sampled });
  const counted =
    task === SEGMENTATION ? segmentationCounted(split) : task === DETECTION ? detectionCounted(split) : classificationCounted(split);
  const checked = task === SEGMENTATION ? segmentationChecked(split) : task === CLASSIFICATION ? classificationChecked(split) : null;
  const readLine =
    isNum(audit?.attempted) && isNum(audit?.completed) && audit.completed < audit.attempted
      ? `Read ${fmt.int(audit.completed)} of the ${fmt.int(audit.attempted)} opened; the problems above say what could not be read.`
      : '';
  return h(
    'section',
    { class: 'taskdata__set', 'aria-label': `${role} ${key}` },
    h(
      'header',
      { class: 'taskdata__sethead' },
      h('h3', { class: 'taskdata__settitle' }, role, ' ', h('span', { class: 'mono taskdata__key' }, key), name ? h('span', { class: 'taskdata__setname' }, name) : null),
      sourceLines(split, task),
    ),
    problems(listed, { truncated: context.truncated, sampled }),
    group('Counted in full', incompleteNote(split, task), counted),
    group('Checked by opening files', line(coverageText(audit, task), 'taskdata__coverage'), line(methodText(audit?.method, task), 'taskdata__quiet'), line(readLine, 'taskdata__quiet'), checked, examples(split, task, context)),
  );
}

// ───────── whole result ─────────

function summary(result, outside) {
  const totals = result.totals ?? {};
  const issues = Array.isArray(result.issues) ? result.issues : [];
  const errors = isNum(totals.error) ? totals.error : issues.filter((issue) => issue.level === 'error').length;
  const warnings = isNum(totals.warning) ? totals.warning : issues.filter((issue) => issue.level !== 'error').length;
  const parts = [];
  if (errors || warnings) {
    parts.push(
      h(
        'p',
        { class: 'taskdata__verdict' },
        errors ? badge(fmt.plural(errors, 'error'), 'err') : null,
        errors && warnings ? ' ' : null,
        warnings ? badge(fmt.plural(warnings, 'warning'), 'warn') : null,
        ' found in all datasets.',
      ),
    );
  } else {
    parts.push(h('p', { class: 'taskdata__verdict' }, badge('No problems', 'ok'), result.complete === true ? (ofRecords(result) ? ' found. Every record was read.' : ' found. Every file was counted and opened.') : ofRecords(result) ? ' found in the records that were read.' : ' found in the files that were opened.'));
  }
  if (result.issuesTruncated === true && isNum(totals.issues)) {
    const limit = isNum(result.limits?.issues) ? ` The service lists at most ${fmt.int(result.limits.issues)}.` : '';
    parts.push(line(`${fmt.int(Math.max(0, totals.issues - issues.length))} more not listed.${limit}`, 'taskdata__line'));
  }
  const limits = result.limits;
  if (limits && 'auditPerSplit' in limits) {
    parts.push(
      line(
        limits.auditPerSplit === null
          ? ofRecords(result)
            ? 'Asked to read every record of each dataset.'
            : 'Asked to open every file in each dataset.'
          : isNum(limits.auditPerSplit)
            ? ofRecords(result)
              ? `Asked to read up to ${fmt.plural(limits.auditPerSplit, 'record')} of each dataset.`
              : `Asked to open up to ${fmt.plural(limits.auditPerSplit, 'file')} in each dataset.`
            : '',
        'taskdata__quiet',
      ),
    );
  }
  if (outside.length) parts.push(h('div', { class: 'taskdata__problems' }, h('p', { class: 'taskdata__problemcount' }, 'Not tied to one dataset'), h('div', { class: 'taskdata__issues' }, issueList(outside))));
  return h('div', { class: 'taskdata__summary' }, parts);
}

/**
 * Render a task-data audit result into `host` (appended). Returns the cleanup function.
 * @param {HTMLElement} host
 * @param {{job?: object, result: object, artifacts?: Array}} input `result` is the job's result
 *   (GET /api/analysis/result → `.result`)
 */
export function renderTaskData(host, { job = null, result = null, artifacts = [] } = {}) {
  void job;
  void artifacts;
  let data = result;
  if (data && !Array.isArray(data.splits) && data.result && Array.isArray(data.result.splits)) data = data.result;
  let root;
  /** {el, path} of each example row's preview box */
  const thumbs = [];
  if (!data || !Array.isArray(data.splits)) {
    root = h('div', { class: 'taskdata' }, notice('', 'The job finished without a data audit to show.'));
  } else {
    const task = data.task;
    const issues = Array.isArray(data.issues) ? data.issues : [];
    const ids = new Set(data.splits.map((split) => String(split.id)));
    const outside = issues.filter((issue) => !ids.has(String(issue.splitId)));
    const context = { truncated: data.issuesTruncated === true, limits: data.limits ?? null, thumbs };
    root = h(
      'div',
      { class: 'taskdata' },
      summary(data, outside),
      data.splits.length
        ? data.splits.map((split) =>
            datasetSection(
              split,
              task,
              issues.filter((issue) => String(issue.splitId) === String(split.id)),
              context,
            ),
          )
        : line('The recipe has no datasets to check.', 'taskdata__quiet'),
    );
  }
  host.appendChild(root);

  // Previews only through POST /api/data/preview at a small size, only for rows that come into view.
  let alive = true;
  const urls = [];
  let observer = null;
  if (thumbs.length && typeof IntersectionObserver === 'function') {
    const byEl = new Map(thumbs.map((thumb) => [thumb.el, thumb.path]));
    observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          observer?.unobserve(entry.target);
          const el = entry.target;
          api.previewBlob(byEl.get(el), { maxSize: THUMB_EDGE }).then(
            (blob) => {
              if (!alive) return;
              const url = URL.createObjectURL(blob);
              urls.push(url);
              const img = h('img', { class: 'taskdata__thumbimg', alt: '', draggable: 'false', decoding: 'async' });
              img.addEventListener('error', () => img.remove(), { once: true });
              img.src = url;
              el.appendChild(img);
            },
            () => {},
          );
        }
      },
      { rootMargin: '160px 0px' },
    );
    for (const thumb of thumbs) observer.observe(thumb.el);
  }

  return () => {
    alive = false;
    observer?.disconnect();
    observer = null;
    for (const url of urls) URL.revokeObjectURL(url);
    urls.length = 0;
    root.remove();
  };
}
