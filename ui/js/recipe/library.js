// The recipe chooser: a list of templates / saved recipes (or a paste box) on the left and an
// inspection pane on the right that shows what the selected recipe really sets, read from the
// backend's check, before the person commits to it. Used full screen when nothing is open
// (mountLibrary) and inside a dialog (openLibraryDialog).

import { h, mount, debounce, uid, fmt } from '../dom.js';
import { api } from '../api.js';
import { getAt } from '../state/recipe.js';
import { catalog, folderWords } from '../state/catalog.js';
import { NO_SCALE_TASKS } from '../state/data.js';
import { openDialog, badge, emptyState, notice, loading, issueList, toast, copyText, countIssues } from '../ui.js';
import { highlightHtml } from './yaml-highlight.js';

/** Rows drawn per step: the list is filtered first, then drawn this many at a time. */
const CHUNK = 120;
/** The phone edition (mobile.html marks the document): one pane at a time instead of two side by side. */
const PHONE = document.documentElement.dataset.edition === 'mobile';

const SOURCES = [
  { id: 'templates', label: 'Templates', noun: ['template', 'templates'] },
  { id: 'recipes', label: 'Saved recipes', noun: ['saved recipe', 'saved recipes'] },
  { id: 'paste', label: 'Paste YAML' },
];

// Templates do not change while the app runs; saved recipes do, so only templates are cached.
let templatesPromise = null;
function loadEntries(source) {
  if (source === 'templates') {
    if (!templatesPromise) {
      templatesPromise = api.templates().catch((error) => {
        templatesPromise = null;
        throw error;
      });
    }
    return templatesPromise;
  }
  return api.recipes();
}

const familyOf = (path) => {
  const parts = String(path ?? '').split('/');
  return parts.length > 1 ? parts[parts.length - 2] : 'Other';
};

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const present = (v) => v !== undefined && v !== null && v !== '';
// String(2e-4) is "0.0002": exact, and short enough for learning rates and weights
const show = (v) => String(v);
const toList = (v) => (Array.isArray(v) ? v : present(v) ? [v] : []).filter(present).map(String);

/** "What it sets": only lines whose values exist in the parsed config. */
function settingsList(config, task) {
  const words = folderWords(task);
  const rows = [];
  const add = (label, value, mono = false) => rows.push(h('dt', null, label), h('dd', { class: mono ? 'mono' : null }, value));
  const total = getAt(config, ['train', 'total_iter']);
  if (isNum(total)) add('Training length', `${fmt.int(total)} iterations`);
  const batch = getAt(config, ['datasets', 'train', 'batch_size_per_gpu']);
  const accum = getAt(config, ['datasets', 'train', 'accum_iter']);
  if (present(batch)) add('Batch', `${show(batch)} per GPU${isNum(accum) && accum > 1 ? `, gradient accumulation over ${fmt.int(accum)} steps` : ''}`);
  const lq = getAt(config, ['datasets', 'train', 'lq_size']);
  const gt = getAt(config, ['datasets', 'train', 'gt_size']);
  // A task with one folder per split (paired: false) has no second side to name.
  const whole = catalog.task(task)?.datasetGeometry === 'whole_images';
  const sideText = (side) => (words.paired === false || whole ? '' : `, ${side.name.toLowerCase()} side`);
  if (present(lq)) add('Training crop', `${show(lq)} px${sideText(words.lq)}`);
  else if (present(gt)) add('Training crop', `${show(gt)} px${sideText(words.gt)}`);
  const otf = config.high_order_degradation;
  if (typeof otf === 'boolean') add('On-the-fly degradation', otf ? 'On' : 'Off');
  const losses = getAt(config, ['train', 'losses']);
  if (Array.isArray(losses) && losses.length) {
    add(
      'Losses',
      h(
        'ul',
        { class: 'lib__losses' },
        losses
          .filter((loss) => loss && typeof loss === 'object')
          .map((loss) => h('li', { class: 'mono' }, String(loss.type ?? 'untyped'), present(loss.loss_weight) ? h('span', { class: 'muted' }, ` weight ${show(loss.loss_weight)}`) : null)),
      ),
    );
  }
  const optim = getAt(config, ['train', 'optim_g', 'type']);
  const lr = getAt(config, ['train', 'optim_g', 'lr']);
  if (present(optim) || present(lr)) add('Optimizer', [present(optim) ? String(optim) : null, present(lr) ? `learning rate ${show(lr)}` : null].filter(Boolean).join(', '));
  const disc = getAt(config, ['network_d', 'type']);
  if (present(disc)) add('Discriminator', String(disc), true);
  if (typeof config.use_amp === 'boolean') add('Mixed precision', config.use_amp ? 'On' : 'Off');
  const datasets = config.datasets;
  if (datasets && typeof datasets === 'object') {
    for (const [key, ds] of Object.entries(datasets)) {
      if (!ds || typeof ds !== 'object') continue;
      const gtDirs = toList(ds.dataroot_gt);
      const lqDirs = toList(ds.dataroot_lq);
      // Without a second folder in the task, dataroot_gt is named by its option key only.
      if (gtDirs.length) add(words.paired === false ? `${key}: dataroot_gt` : `${key}: ${words.gt.name.toLowerCase()} (dataroot_gt)`, gtDirs.join('\n'), true);
      if (lqDirs.length) add(`${key}: ${words.lq.name.toLowerCase()} (dataroot_lq)`, lqDirs.join('\n'), true);
    }
  }
  const pretrain = getAt(config, ['path', 'pretrain_network_g']);
  if (present(pretrain)) add('Pretrained weights', String(pretrain), true);
  return rows.length ? h('dl', { class: 'kv lib__kv' }, rows) : h('p', { class: 'muted' }, 'The check returned no options from this list.');
}

function verdictBadges(check) {
  const { errors, warnings } = countIssues(check.issues);
  if (check.ok && !errors) {
    return [badge('Passes check', 'ok'), warnings ? badge(fmt.plural(warnings, 'warning'), 'warn') : null];
  }
  return [badge(errors ? fmt.plural(errors, 'error') : 'Does not pass check', 'err'), warnings ? badge(fmt.plural(warnings, 'warning'), 'warn') : null];
}

/** A scale is a fact of a recipe that upscales: classification, segmentation and detection do not show one. */
const showsScale = (task, scale) => isNum(scale) && !NO_SCALE_TASKS.includes(String(task ?? ''));

/** Facts, verdict, issues and settings of a check result. */
function checkSummary(check) {
  const facts = [];
  const s = check.summary ?? {};
  if (present(s.task)) facts.push(h('span', null, h('span', { class: 'muted' }, 'Task '), catalog.taskLabel(String(s.task))));
  if (present(s.architecture)) facts.push(h('span', null, h('span', { class: 'muted' }, 'Network '), String(s.architecture)));
  if (showsScale(s.task, s.scale)) facts.push(h('span', null, h('span', { class: 'muted' }, 'Scale '), `${s.scale}×`));
  const issues = Array.isArray(check.issues) ? check.issues : [];
  return [
    facts.length ? h('div', { class: 'lib__facts' }, facts) : null,
    h('div', { class: 'row row--wrap' }, verdictBadges({ ...check, issues })),
    issues.length
      ? h('details', { class: 'lib__details' }, h('summary', null, `Show ${fmt.plural(issues.length, 'issue')}`), issueList(issues))
      : null,
    h('h3', { class: 'lib__subtitle' }, 'What it sets'),
    check.config && typeof check.config === 'object' ? settingsList(check.config, s.task) :h('p', { class: 'muted' }, 'The check returned no parsed options.'),
  ];
}

// ───────── neosr import ─────────

/** The backend refuses more than 2 MiB of TOML (studio/neosr.py), counted in UTF-8 bytes. */
const NEOSR_MAX_BYTES = 2 * 1024 * 1024;
const byteLength = (text) => new TextEncoder().encode(text).length;
const valueText = (v) => (typeof v === 'string' ? v : JSON.stringify(v));

/**
 * Translate a neosr TOML into a native recipe draft (POST /api/recipes/import-neosr, writes no file).
 * Resolves to a picked recipe in the chooser's shape, or null when closed.
 */
function openNeosrImport() {
  const fileId = uid('neosr-file');
  const areaId = uid('neosr-text');
  const helpId = uid('neosr-help');
  let sourceName = null; // file name, or null for pasted text
  let result = null; // the last translation of the text as it is now
  let busy = false;
  const fileError = h('p', { class: 'field__error', role: 'alert', hidden: true });
  const fileInput = h('input', { type: 'file', id: fileId, accept: '.toml,text/plain,application/toml', class: 'lib__file' });
  const area = h('textarea', { class: 'textarea mono lib__paste', id: areaId, rows: '12', spellcheck: 'false', autocomplete: 'off', wrap: 'off', 'aria-describedby': helpId });
  const answer = h('div', { class: 'lib__importresult', role: 'status' });
  const translate = h('button', { class: 'btn', type: 'button' }, 'Translate');
  let openButton = null;

  const setResult = (next) => {
    result = next;
    if (openButton) openButton.disabled = !result;
  };
  fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    fileError.hidden = true;
    if (!file) return;
    if (file.size > NEOSR_MAX_BYTES) {
      fileError.textContent = `${file.name} is larger than 2 MB, the most the import accepts.`;
      fileError.hidden = false;
      return;
    }
    try {
      area.value = await file.text();
      sourceName = file.name;
      setResult(null);
      mount(answer);
    } catch (error) {
      fileError.textContent = `${file.name} could not be read: ${error.message}`;
      fileError.hidden = false;
    }
  });
  area.addEventListener('input', () => {
    // A translation no longer matches edited text; emptied text no longer comes from the file.
    if (!area.value.trim()) sourceName = null;
    setResult(null);
    mount(answer);
  });

  translate.addEventListener('click', async () => {
    if (busy) return;
    const text = area.value;
    if (!text.trim()) {
      mount(answer, notice('warn', 'Choose a file or paste the TOML first.'));
      return;
    }
    if (byteLength(text) > NEOSR_MAX_BYTES) {
      mount(answer, notice('err', 'The text is larger than 2 MB, the most the import accepts.'));
      return;
    }
    busy = true;
    translate.disabled = true;
    mount(answer, loading('Translating…'));
    try {
      const r = await api.importNeosr(text);
      if (area.value !== text) return; // edited meanwhile: the answer is for older text
      setResult(r);
      renderAnswer(r);
    } catch (error) {
      // The backend's sentence; the text stays so it can be fixed and sent again.
      mount(answer, notice('err', `The import was refused: ${error.message}`));
    } finally {
      busy = false;
      translate.disabled = false;
    }
  });

  function renderAnswer(r) {
    const notes = Array.isArray(r.notes) ? r.notes.filter((n) => typeof n === 'string' && n) : [];
    const review = r.review && typeof r.review === 'object' ? Object.entries(r.review) : [];
    const sha = typeof r.sourceSha256 === 'string' ? r.sourceSha256 : '';
    const pre = h('pre', { class: 'yaml__pre lib__yaml', tabindex: '0', role: 'region', 'aria-label': 'Translated YAML' });
    pre.innerHTML = highlightHtml(typeof r.yaml === 'string' ? r.yaml : '');
    mount(
      answer,
      r.needsReview
        ? notice(
            'warn',
            h(
              'div',
              { class: 'stack' },
              h('p', null, `${fmt.plural(review.length, 'neosr entry', 'neosr entries')} could not be translated. The recipe keeps them in a neosr_review section, which must be resolved and then removed before training.`),
              h(
                'dl',
                { class: 'lib__review' },
                review.map(([key, entry]) => [
                  h('dt', { class: 'mono lib__reviewkey' }, key, ' = ', valueText(entry?.value)),
                  h('dd', { class: 'lib__reviewwhy' }, String(entry?.reason ?? '')),
                ]),
              ),
            ),
          )
        : notice('ok', 'Every neosr entry was translated.'),
      notes.length ? h('div', null, h('h3', { class: 'lib__subtitle' }, 'Notes from the import'), h('ul', { class: 'lib__notes' }, notes.map((n) => h('li', null, n)))) : null,
      h('div', { class: 'lib__yamlhead' }, h('h3', { class: 'lib__subtitle' }, 'Translated YAML'), sha ? h('span', { class: 'mono muted', title: `SHA-256 of the source: ${sha}` }, `Source fingerprint ${sha.slice(0, 12)}`) : null),
      pre,
    );
  }

  const body = h(
    'div',
    { class: 'stack lib__import' },
    h('p', { id: helpId }, 'The text is sent to the trainer service, which translates it into a recipe draft. Nothing is saved: the draft opens in the editor, and you save it yourself.'),
    h('div', { class: 'field' }, h('label', { class: 'field__label', for: fileId }, 'Choose a .toml file from this computer'), fileInput, h('p', { class: 'field__help' }, 'Up to 2 MB.'), fileError),
    h('div', { class: 'field' }, h('label', { class: 'field__label', for: areaId }, 'Or paste the neosr TOML'), area),
    h('div', { class: 'row' }, translate),
    answer,
  );
  const dialog = openDialog({
    title: 'Import a neosr config',
    body,
    wide: true,
    foot: (close) => {
      openButton = h('button', { class: 'btn btn--primary', type: 'button', disabled: true, onclick: () => result && close(result) }, 'Open as a draft');
      return [h('button', { class: 'btn', type: 'button', onclick: () => close(undefined) }, 'Cancel'), openButton];
    },
  });
  area.focus();
  return dialog.closed.then((r) => {
    if (!r || typeof r.yaml !== 'string') return null;
    const named = /^name:\s*(.+?)\s*$/m.exec(r.yaml)?.[1]?.replace(/^['"]|['"]$/g, '');
    const name = named || (sourceName ? sourceName.replace(/\.toml$/i, '') : 'neosr import');
    return { doc: { yaml: r.yaml, path: null, name }, label: `neosr import (${sourceName ?? 'pasted text'})`, saved: false };
  });
}

/**
 * @param {{onPick: (picked: {doc: {path: string|null, name: string, yaml: string}, label: string, saved: boolean}) => void, intro?: boolean}} options
 * @returns {{el: HTMLElement, destroy: () => void, focus: () => void}}
 */
export function createLibrary({ onPick, intro = false, pickLabel = null }) {
  let destroyed = false;
  let source = 'templates';
  let task = 'all';
  let family = 'all';
  let query = '';
  const lists = new Map(); // source -> {status, entries, error, box, none, more, options, limit}
  const docs = new Map(); // path -> {doc, check, checkError}
  let selected = null; // entry
  let inspectAbort = null;
  let pasteCheck = null; // {yaml, check}

  const countLine = h('p', { class: 'lib__count', role: 'status' });
  const listHost = h('div', { class: 'lib__listhost' });
  const inspect = h('div', { class: 'lib__inspect', role: 'region', 'aria-label': 'Selected recipe', tabindex: '-1' });
  const searchId = uid('lib-search');
  const search = h('input', {
    class: 'input',
    id: searchId,
    type: 'search',
    placeholder: 'Find by name or network',
    autocomplete: 'off',
    spellcheck: 'false',
    oninput: (event) => {
      query = event.currentTarget.value.trim().toLowerCase();
      resetLimit();
      applyFilter();
    },
  });
  const taskSeg = h('div', { class: 'seg lib__tasks', role: 'group', 'aria-label': 'Task' });
  const familyId = uid('lib-family');
  const familySelect = h('select', {
    class: 'select lib__family',
    id: familyId,
    onchange: () => {
      family = familySelect.value;
      resetLimit();
      applyFilter();
    },
  });
  const familyField = h('div', { class: 'lib__familyfield' }, h('label', { class: 'visually-hidden', for: familyId }, 'Family'), familySelect);
  const filters = h('div', { class: 'lib__filters' }, taskSeg, familyField, h('label', { class: 'visually-hidden', for: searchId }, 'Find by name or network'), search);
  const sourceSeg = h(
    'div',
    { class: 'seg', role: 'group', 'aria-label': 'Source' },
    SOURCES.map((s) =>
      h(
        'button',
        {
          class: 'btn btn--sm',
          type: 'button',
          dataset: { source: s.id },
          'aria-pressed': String(s.id === source),
          onclick: () => setSource(s.id),
        },
        s.label,
      ),
    ),
  );
  const importButton = h(
    'button',
    {
      class: 'btn btn--sm btn--quiet lib__importbtn',
      type: 'button',
      onclick: async () => {
        const picked = await openNeosrImport();
        if (picked && !destroyed) onPick(picked);
      },
    },
    'Import a neosr config',
  );
  // A run file from another machine (or this one): mapped to this machine's folders, checked, then opened
  // as a recipe like any other pick. Nothing is saved or started by it.
  const runFileButton = h(
    'button',
    {
      class: 'btn btn--sm btn--quiet lib__importbtn',
      type: 'button',
      title: 'A run file describes a run made elsewhere: its recipe, seed, starting point and inputs.',
      onclick: async () => {
        try {
          const mod = await import('../runs/runfile.js');
          const picked = await mod.openRunFileImport();
          if (picked && !destroyed) onPick(picked);
        } catch (error) {
          toast(`The run file dialog could not be opened: ${error?.message ?? error}`, { kind: 'err' });
        }
      },
    },
    'Open a run file',
  );
  const column = h('div', { class: 'lib__col' }, h('div', { class: 'lib__controls' }, sourceSeg, filters, h('div', { class: 'lib__imports' }, importButton, runFileButton), countLine), listHost);
  // The phone edition (mobile.html) shows the list and the chosen recipe one at a time: data-pane says
  // which (css/mobile-screens.css lays it out), and this button leads back from the recipe to the list.
  // The desktop page shows both side by side and gets neither.
  const backLabel = h('span');
  const backButton = PHONE ? h('button', { class: 'lib__back', type: 'button', onclick: () => showPane('list') }, backLabel) : null;
  const el = h(
    'div',
    { class: `lib${intro ? ' lib--full' : ' lib--dialog'}` },
    intro
      ? h(
          'header',
          { class: 'lib__intro' },
          h('h1', null, 'Start with a recipe'),
          h('p', null, 'A recipe is the YAML file that holds every training option. Pick a template for your network, or open a recipe you saved.'),
        )
      : null,
    h('div', { class: 'lib__body' }, column, inspect),
  );
  let listScroll = 0;
  if (PHONE) {
    el.dataset.pane = 'list';
    el.prepend(backButton);
  }

  /** Phone edition only: the list, or the recipe chosen in it. Each keeps its own place on the page. */
  function showPane(pane) {
    if (!PHONE || el.dataset.pane === pane) return;
    if (pane === 'inspect') listScroll = el.scrollTop;
    el.dataset.pane = pane;
    if (pane === 'inspect') {
      el.scrollTop = 0;
      inspect.focus({ preventScroll: true });
      return;
    }
    el.scrollTop = listScroll;
    lists
      .get(source)
      ?.options.find((o) => o.getAttribute('aria-selected') === 'true')
      ?.focus({ preventScroll: true });
  }

  // ───────── tasks ─────────

  const resetLimit = () => {
    const list = lists.get(source);
    if (list) list.limit = CHUNK;
  };

  /** Task and family filters, built from the values that occur in the current list. */
  function renderFilters() {
    const list = lists.get(source);
    const entries = list?.status === 'ready' ? list.entries : [];
    const counts = new Map();
    for (const e of entries) if (present(e.task)) counts.set(String(e.task), (counts.get(String(e.task)) ?? 0) + 1);
    const ids = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a) || a.localeCompare(b));
    if (task !== 'all' && !counts.has(task)) task = 'all';
    taskSeg.hidden = ids.length < 2;
    mount(
      taskSeg,
      ['all', ...ids].map((id) =>
        h(
          'button',
          {
            class: 'btn btn--sm',
            type: 'button',
            'aria-pressed': String(id === task),
            onclick: () => {
              task = id;
              resetLimit();
              renderFilters();
              applyFilter();
            },
          },
          id === 'all' ? 'All tasks' : `${catalog.taskLabel(id)} (${fmt.int(counts.get(id))})`,
        ),
      ),
    );
    const families = [...new Set(entries.filter((e) => task === 'all' || e.task === task).map((e) => e._family))].sort((a, b) => a.localeCompare(b));
    if (family !== 'all' && !families.includes(family)) family = 'all';
    familyField.hidden = families.length < 2;
    mount(familySelect, h('option', { value: 'all' }, 'All families'), families.map((f) => h('option', { value: f }, f)));
    familySelect.value = family;
  }

  // Task labels come from the catalog; until it answers, the ids are shown in plain words.
  catalog.load().then(
    () => {
      if (destroyed) return;
      renderFilters();
      applyFilter();
    },
    () => {},
  );

  // ───────── the list ─────────

  function setSource(id) {
    source = id;
    for (const b of sourceSeg.children) b.setAttribute('aria-pressed', String(b.dataset.source === id));
    if (PHONE) {
      const label = SOURCES.find((s) => s.id === id)?.label ?? 'List';
      backLabel.textContent = label;
      backButton.setAttribute('aria-label', `Back to ${label}`);
      showPane('list');
    }
    filters.hidden = id === 'paste';
    selected = null;
    if (inspectAbort) inspectAbort.abort();
    if (id === 'paste') {
      showPaste();
      return;
    }
    showList();
  }

  function showList() {
    let list = lists.get(source);
    if (!list) {
      list = { status: 'loading', entries: [], error: null, box: null, none: null, more: null, options: [], limit: CHUNK };
      lists.set(source, list);
      fetchList(source, list);
    }
    renderFilters();
    renderListState(list);
    renderInspectEmpty();
  }

  function fetchList(src, list) {
    list.status = 'loading';
    loadEntries(src)
      .then((entries) => {
        const isTemplate = src === 'templates';
        list.entries = (Array.isArray(entries) ? entries.filter((e) => e && typeof e.path === 'string') : []).map((e) => ({
          ...e,
          _family: present(e.family) ? String(e.family) : isTemplate ? familyOf(e.path) : 'Saved recipes',
          _haystack: `${e.name ?? ''} ${e.path} ${e.architecture ?? ''} ${e.family ?? ''}`.toLowerCase(),
        }));
        list.entries.sort((a, b) => a._family.localeCompare(b._family) || String(a.name).localeCompare(String(b.name)));
        list.status = 'ready';
        buildList(list);
        if (!destroyed && source === src) renderFilters();
      })
      .catch((error) => {
        list.status = 'error';
        list.error = error;
      })
      .finally(() => {
        if (!destroyed && source === src) renderListState(list);
      });
  }

  function renderListState(list) {
    if (list.status === 'loading') {
      countLine.textContent = '';
      mount(listHost, loading(source === 'templates' ? 'Loading templates…' : 'Loading saved recipes…'));
      return;
    }
    if (list.status === 'error') {
      countLine.textContent = '';
      mount(
        listHost,
        h(
          'div',
          { class: 'lib__pad' },
          notice('err', `The list could not be loaded: ${list.error?.message ?? 'no reason was given'}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => { fetchList(source, list); renderListState(list); } }, 'Try again')),
        ),
      );
      return;
    }
    mount(listHost, list.box, list.more, list.none);
    applyFilter();
  }

  /** The list's containers; the rows are drawn by applyFilter for the entries that match. */
  function buildList(list) {
    const box = h('div', { class: 'lib__list', role: 'listbox', 'aria-label': source === 'templates' ? 'Templates' : 'Saved recipes' });
    box.addEventListener('keydown', (event) => onListKey(list, event));
    list.box = box;
    list.options = [];
    list.none = h('p', { class: 'lib__none', hidden: true });
    list.more = h(
      'div',
      { class: 'lib__more', hidden: true },
      h(
        'button',
        {
          class: 'btn btn--sm',
          type: 'button',
          onclick: () => {
            const first = list.limit;
            list.limit += CHUNK;
            applyFilter();
            // Focus the first newly drawn row so the keyboard carries on where the list grew.
            const next = list.options[first];
            if (next) select(list, next, { immediate: false });
          },
        },
        'Show more',
      ),
    );
  }

  /** One row: name, then architecture, scale and task straight from the list. */
  function itemRow(list, entry) {
    const broken = present(entry.error);
    const meta = broken
      ? [h('span', { class: 'lib__error' }, 'Unreadable: ', String(entry.error))]
      : [
          present(entry.architecture) ? h('span', { class: 'mono' }, String(entry.architecture)) : null,
          showsScale(entry.task, entry.scale) ? h('span', null, `${entry.scale}×`) : null,
          present(entry.task) ? h('span', null, catalog.taskLabel(String(entry.task))) : null,
        ];
    const opt = h(
      'div',
      {
        class: `lib__item${broken ? ' lib__item--broken' : ''}`,
        role: 'option',
        id: uid('lib-opt'),
        tabindex: '-1',
        'aria-selected': String(selected?.path === entry.path),
        'aria-disabled': broken ? 'true' : null,
        title: entry.path,
        onclick: () => {
          select(list, opt);
          showPane('inspect');
        },
        ondblclick: () => {
          if (!broken) pickEntry(entry);
        },
      },
      h('span', { class: 'lib__name truncate' }, String(entry.name ?? entry.path)),
      h('span', { class: 'lib__meta' }, meta),
    );
    opt._entry = entry;
    return opt;
  }

  const visibleOptions = (list) => list.options;

  /** Filter first, then draw at most `limit` rows, grouped by family. */
  function applyFilter() {
    const list = lists.get(source);
    if (!list || list.status !== 'ready' || source === 'paste') return;
    const inTask = list.entries.filter((e) => task === 'all' || e.task === task);
    const matches = inTask.filter((e) => (family === 'all' || e._family === family) && (!query || e._haystack.includes(query)));
    const drawn = matches.slice(0, list.limit);
    const hadFocus = list.box.contains(document.activeElement);
    const groups = new Map();
    for (const e of drawn) {
      if (!groups.has(e._family)) groups.set(e._family, []);
      groups.get(e._family).push(e);
    }
    const total = new Map();
    for (const e of matches) total.set(e._family, (total.get(e._family) ?? 0) + 1);
    list.options = [];
    const nodes = [];
    for (const [fam, entries] of groups) {
      const headId = uid('lib-group');
      const options = entries.map((e) => itemRow(list, e));
      list.options.push(...options);
      nodes.push(h('div', { class: 'lib__group', role: 'group', 'aria-labelledby': headId }, h('div', { class: 'lib__grouphead', id: headId }, fam, ' ', h('span', { class: 'lib__groupcount' }, String(total.get(fam) ?? options.length))), options));
    }
    list.box.replaceChildren(...nodes);
    const [one, many] = SOURCES.find((s) => s.id === source).noun;
    const shown = matches.length;
    const of = inTask.length;
    countLine.textContent = `${shown === of ? fmt.plural(of, one, many) : `${fmt.int(shown)} of ${fmt.plural(of, one, many)}`}${drawn.length < shown ? `, first ${fmt.int(drawn.length)} shown` : ''}`;
    list.more.hidden = drawn.length >= shown;
    list.none.hidden = shown > 0;
    list.none.textContent = source === 'templates' ? 'No templates match.' : 'No saved recipes match.';
    // keep one reachable tab stop: the selected option if drawn, else the first one
    const stop = list.options.find((o) => o.getAttribute('aria-selected') === 'true') ?? list.options[0];
    if (stop) {
      stop.tabIndex = 0;
      if (hadFocus) stop.focus({ preventScroll: true });
    }
  }

  function select(list, opt, { focus = true, immediate = true } = {}) {
    for (const o of list.options) {
      if (o.getAttribute('aria-selected') === 'true') o.setAttribute('aria-selected', 'false');
      o.tabIndex = -1;
    }
    opt.setAttribute('aria-selected', 'true');
    opt.tabIndex = 0;
    if (focus) opt.focus();
    opt.scrollIntoView({ block: 'nearest' });
    selected = opt._entry;
    if (immediate) inspectSoon.flush(selected);
    else inspectSoon(selected);
  }

  function onListKey(list, event) {
    const visible = visibleOptions(list);
    if (!visible.length) return;
    const index = visible.indexOf(document.activeElement);
    let next = null;
    if (event.key === 'ArrowDown') next = visible[Math.min(visible.length - 1, index + 1)];
    else if (event.key === 'ArrowUp') next = visible[Math.max(0, index - 1)];
    else if (event.key === 'Home') next = visible[0];
    else if (event.key === 'End') next = visible[visible.length - 1];
    else if (event.key === 'Enter' && index >= 0) {
      event.preventDefault();
      if (!present(visible[index]._entry.error)) pickEntry(visible[index]._entry);
      return;
    } else return;
    event.preventDefault();
    // arrow-key runs are debounced so holding a key does not fire a request per row
    if (next) select(list, next, { immediate: false });
  }

  // ───────── inspection ─────────

  function renderInspectEmpty() {
    mount(inspect, emptyState({ title: source === 'recipes' ? 'Select a recipe' : 'Select a template', text: 'Its options and check result appear here.' }));
  }

  async function loadDoc(entry, signal) {
    const cached = docs.get(entry.path);
    if (cached?.doc && (cached.check || cached.checkError)) return cached;
    const doc = cached?.doc ?? (await api.recipe(entry.path, { signal }));
    const record = { doc, check: null, checkError: null };
    try {
      record.check = await api.preflight(doc.yaml, { signal });
    } catch (error) {
      if (error?.name === 'AbortError') throw error;
      record.checkError = error;
    }
    docs.set(entry.path, record);
    return record;
  }

  const inspectSoon = debounce((entry) => inspectEntry(entry), 120);

  /** What the list already says about an entry, shown at once while the file is read and checked. */
  function listFacts(entry) {
    const facts = [];
    if (present(entry.task)) facts.push(h('span', null, h('span', { class: 'muted' }, 'Task '), catalog.taskLabel(String(entry.task))));
    if (present(entry.architecture)) facts.push(h('span', null, h('span', { class: 'muted' }, 'Network '), String(entry.architecture)));
    if (showsScale(entry.task, entry.scale)) facts.push(h('span', null, h('span', { class: 'muted' }, 'Scale '), `${entry.scale}×`));
    return facts.length ? h('div', { class: 'lib__facts' }, facts) : null;
  }

  async function inspectEntry(entry) {
    if (inspectAbort) inspectAbort.abort();
    if (present(entry.error)) {
      // An unreadable file is described, not offered.
      inspectAbort = null;
      mount(inspect, h('h2', { class: 'lib__title truncate', title: entry.path }, String(entry.name ?? entry.path)), h('p', { class: 'lib__path mono muted truncate', title: entry.path }, entry.path), notice('err', `This file cannot be used: ${entry.error}`));
      return;
    }
    const control = new AbortController();
    inspectAbort = control;
    if (!docs.get(entry.path)?.check) mount(inspect, h('h2', { class: 'lib__title' }, String(entry.name)), listFacts(entry), loading('Reading and checking…'));
    try {
      const record = await loadDoc(entry, control.signal);
      if (control.signal.aborted || destroyed || selected !== entry) return;
      renderInspect(entry, record);
    } catch (error) {
      if (error?.name === 'AbortError' || control.signal.aborted || destroyed) return;
      mount(
        inspect,
        h('h2', { class: 'lib__title' }, String(entry.name)),
        notice('err', `${entry.name} could not be read: ${error.message}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => inspectEntry(entry) }, 'Try again')),
      );
    }
  }

  function pickedFor(entry, doc) {
    const name = String(doc.name ?? entry.name ?? '');
    const isTemplate = source === 'templates';
    return {
      // The whole document goes through (revision included), so saving can detect a changed file.
      doc: { ...doc, path: doc.path ?? entry.path, name, yaml: doc.yaml },
      label: isTemplate ? `template ${name}` : `saved recipe ${name}`,
      saved: !isTemplate,
    };
  }

  async function pickEntry(entry) {
    if (present(entry.error)) return;
    const cached = docs.get(entry.path);
    if (cached?.doc) {
      onPick(pickedFor(entry, cached.doc));
      return;
    }
    try {
      const doc = await api.recipe(entry.path);
      docs.set(entry.path, { ...(docs.get(entry.path) ?? {}), doc });
      if (!destroyed) onPick(pickedFor(entry, doc));
    } catch (error) {
      toast(`${entry.name} could not be read: ${error.message}`, { kind: 'err' });
    }
  }

  function renderInspect(entry, { doc, check, checkError }) {
    const isTemplate = source === 'templates';
    const yamlPre = h('pre', { class: 'yaml__pre lib__yaml', tabindex: '0', role: 'region', 'aria-label': `YAML of ${entry.name}` });
    yamlPre.innerHTML = highlightHtml(doc.yaml);
    mount(
      inspect,
      h(
        'div',
        { class: 'lib__head' },
        h('h2', { class: 'lib__title truncate', title: entry.path }, String(entry.name)),
        h('button', { class: 'btn btn--primary', type: 'button', onclick: () => onPick(pickedFor(entry, doc)) }, pickLabel ?? (isTemplate ? 'Use this template' : 'Open this recipe')),
      ),
      h('p', { class: 'lib__path mono muted truncate', title: entry.path }, entry.path),
      check ? checkSummary(check) : notice('err', `The check could not run: ${checkError?.message ?? 'no reason was given'}`),
      h(
        'div',
        { class: 'lib__yamlhead' },
        h('h3', { class: 'lib__subtitle' }, 'YAML'),
        h(
          'button',
          {
            class: 'btn btn--sm',
            type: 'button',
            onclick: async () => {
              const ok = await copyText(doc.yaml);
              toast(ok ? 'Copied the YAML.' : 'The browser did not allow copying. Select the text and copy it instead.', { kind: ok ? 'info' : 'err' });
            },
          },
          'Copy',
        ),
      ),
      yamlPre,
    );
  }

  // ───────── paste ─────────

  function showPaste() {
    countLine.textContent = '';
    const areaId = uid('lib-paste');
    const area = h('textarea', { class: 'textarea mono lib__paste', id: areaId, rows: '16', spellcheck: 'false', autocomplete: 'off', wrap: 'off' });
    const useButton = h('button', { class: 'btn btn--primary', type: 'button', disabled: true, onclick: () => usePasted(area.value) }, pickLabel ?? 'Use this YAML');
    const checkButton = h('button', { class: 'btn', type: 'button', disabled: true, onclick: () => checkPasted(area.value) }, 'Check');
    area.addEventListener('input', () => {
      const empty = area.value.trim() === '';
      useButton.disabled = empty;
      checkButton.disabled = empty;
    });
    mount(
      listHost,
      h('div', { class: 'lib__pad stack' }, h('div', { class: 'field' }, h('label', { class: 'field__label', for: areaId }, 'Recipe YAML'), area), h('div', { class: 'row' }, checkButton, useButton)),
    );
    mount(inspect, emptyState({ title: 'Paste a recipe', text: 'Paste YAML on the left, then check it to see its options and check result here.' }));
  }

  async function checkPasted(yaml) {
    if (inspectAbort) inspectAbort.abort();
    const control = new AbortController();
    inspectAbort = control;
    mount(inspect, loading('Checking the pasted YAML…'));
    try {
      const check = await api.preflight(yaml, { signal: control.signal });
      if (control.signal.aborted || destroyed) return;
      pasteCheck = { yaml, check };
      mount(inspect, h('h2', { class: 'lib__title' }, check.summary?.name ? String(check.summary.name) : 'Pasted YAML'), checkSummary(check));
    } catch (error) {
      if (error?.name === 'AbortError' || control.signal.aborted || destroyed) return;
      mount(inspect, notice('err', `The pasted YAML could not be checked: ${error.message}`));
    }
  }

  async function usePasted(yaml) {
    if (!yaml.trim()) return;
    let name = pasteCheck?.yaml === yaml ? pasteCheck.check?.summary?.name : null;
    if (!name && pasteCheck?.yaml !== yaml) {
      // one quick check to name the document; a failed check still lets the text open
      try {
        name = (await api.preflight(yaml))?.summary?.name ?? null;
      } catch {
        name = null;
      }
    }
    if (destroyed) return;
    onPick({ doc: { path: null, name: name ? String(name) : 'pasted', yaml }, label: 'pasted YAML', saved: false });
  }

  setSource('templates');

  return {
    el,
    focus: () => search.focus(),
    destroy() {
      destroyed = true;
      inspectSoon.cancel();
      if (inspectAbort) inspectAbort.abort();
      el.remove();
    },
  };
}

/** Full-screen chooser shown when no recipe is open. */
export function mountLibrary(el, { onPick }) {
  const lib = createLibrary({ onPick, intro: true });
  el.appendChild(lib.el);
  return () => lib.destroy();
}

/**
 * The chooser in a dialog. Resolves with the picked recipe, or null when closed.
 * pickLabel renames the primary button when picking means something other than opening
 * (for example "Compare with this recipe").
 */
export async function openLibraryDialog({ title = 'Choose a recipe', pickLabel = null } = {}) {
  let dialog = null;
  const lib = createLibrary({ onPick: (picked) => dialog?.close(picked), intro: false, pickLabel });
  dialog = openDialog({ title, body: lib.el, wide: true, flush: true });
  lib.focus();
  const result = await dialog.closed;
  lib.destroy();
  return result && result.doc ? result : null;
}
