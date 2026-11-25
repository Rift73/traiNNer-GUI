// Validation tab of one run: the images the trainer saved while validating, per iteration and per
// sample, and the comparison stage for one sample against the ground truth and input the run recorded,
// or against the same sample at another iteration. Everything shown comes from GET /api/run/images.
//
// Contract (DESIGN.md, Runs screen parts): mountValidation(host, {run}) -> {setRun(run), destroy()}.

import { h, mount, fmt, debounce, uid } from '../dom.js';
import { api, sizedUrl, THUMB_SIZE, PREVIEW_MAX } from '../api.js';
import { notice, emptyState, toast } from '../ui.js';
import { runStatus } from '../state/runs.js';
import { persist } from '../store.js';
import { createCompare, reasonOf } from '../components/compare.js';

const VIEW_KEY = 'validationView';

/** Rows per page of the gallery (the API allows 1..500). */
const PAGE = 100;
const MAX_PAGE = 500;
/** Longest edge of the filmstrip thumbnails under the stage. */
const FILM_SIZE = 96;
/** One sample's history: pages of 500, at most this many pages. */
const HISTORY_PAGES = 10;

const isIter = (n) => Number.isInteger(n) && n >= 0;
const keyOf = (row) => `${row.iteration}|${row.dataset ?? ''}|${row.image}`;
const sampleKeyOf = (row) => `${row.dataset ?? ''}|${row.image}`;
const iterText = (n) => (isIter(n) ? `Iteration ${fmt.int(n)}` : 'Iteration not reported');
/** The sample id exactly as the API gave it (recorded "0/0007.png", filename mode "0007"). */
const sampleText = (row) => String(row?.image ?? row?.name ?? '');
/** Last path segment, for labels; the full id goes in a title. */
const lastSegment = (text) => String(text ?? '').split('/').filter(Boolean).pop() ?? '';
const sampleLabel = (row) => lastSegment(sampleText(row)) || sampleText(row);
const code = (text) => h('code', { class: 'mono' }, text);

const PROVENANCE = {
  recorded: "From the run's image record.",
  filename:
    "Read from file names in the run's visualization folder. Dataset names and ground truth were not recorded for this run.",
};

/**
 * @param {HTMLElement} host
 * @param {{run: {id: string, name?: string, status?: string}}} options
 * @returns {{setRun(run: object): void, destroy(): void}}
 */
export function mountValidation(host, { run } = {}) {
  let current = run ?? null;
  let destroyed = false;

  // ───────── state ─────────
  const st = {
    gen: 0,
    /** 'idle' | 'loading' | 'ready' | 'error' */
    phase: 'idle',
    error: '',
    rows: [],
    keys: new Set(),
    next: null,
    /** 'recorded' | 'filename' | null (page-level value from the API) */
    provenance: null,
    iterations: new Set(),
    datasets: new Set(),
    filter: { iteration: '', dataset: '', text: '' },
    canReveal: false,
    more: false,
    moreError: '',
    /** the load in flight is a "Check for new images" */
    checking: false,
    /** the run was training when the list was last loaded */
    loadedActive: false,
    /** rows that a "Check for new images" found; null when the last load was not a check */
    fresh: null,
  };
  let listAbort = null;
  let moreAbort = null;

  /** the selected sample on the stage */
  const jd = {
    open: false,
    row: null,
    /** rows of this sample, newest first; null while loading */
    history: null,
    historyError: '',
    historyCapped: false,
    historyGen: 0,
    /** 'gt' | 'lq' | 'out' | 'other' | '' per side */
    pick: { a: '', b: '' },
    /** set once the person chose a side, so stepping through samples keeps the pairing */
    userPicked: false,
    /** iteration of "Output at another iteration" */
    other: null,
  };
  let historyAbort = null;
  let compare = null;
  let sourceKey = '';

  // ───────── filter bar ─────────
  const ids = { iter: uid('validation-iter'), dataset: uid('validation-dataset'), name: uid('validation-name'), a: uid('validation-a'), b: uid('validation-b'), strip: uid('validation-strip') };
  const field = (id, label, control, extra = {}) =>
    h('div', { class: 'validation__field', ...extra }, h('label', { class: 'validation__label', for: id }, label), control);

  const iterSelect = h('select', { class: 'select validation__select', id: ids.iter, onchange: () => setFilter({ iteration: iterSelect.value }) });
  const datasetSelect = h('select', { class: 'select validation__select', id: ids.dataset, onchange: () => setFilter({ dataset: datasetSelect.value }) });
  const nameInput = h('input', {
    class: 'input validation__search',
    id: ids.name,
    type: 'search',
    autocomplete: 'off',
    spellcheck: 'false',
    placeholder: 'Part of a sample name',
  });
  const applyText = debounce(() => {
    st.filter.text = nameInput.value;
    renderList();
  }, 150);
  nameInput.addEventListener('input', applyText);
  const datasetField = field(ids.dataset, 'Dataset', datasetSelect, { hidden: true });
  const checkButton = h('button', { class: 'btn', type: 'button', hidden: true, onclick: () => checkForNew() }, 'Check for new images');
  const bar = h(
    'div',
    { class: 'validation__bar' },
    field(ids.iter, 'Iteration', iterSelect),
    datasetField,
    field(ids.name, 'Sample name', nameInput),
    h('div', { class: 'validation__barend' }, checkButton),
  );
  const provEl = h('p', { class: 'validation__prov', hidden: true });
  const errorEl = h('div', { class: 'validation__error', hidden: true });

  // ───────── judging one sample ─────────
  const titleEl = h('h3', { class: 'validation__title', tabindex: '-1' });
  const metaEl = h('p', { class: 'validation__meta' });
  const prevButton = h('button', { class: 'btn btn--sm', type: 'button', 'aria-keyshortcuts': 'P', title: 'Previous sample (P)', onclick: () => step(-1) }, 'Previous sample');
  const nextButton = h('button', { class: 'btn btn--sm', type: 'button', 'aria-keyshortcuts': 'N', title: 'Next sample (N)', onclick: () => step(1) }, 'Next sample');
  const closeButton = h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => closeJudge(true) }, 'Close comparison');
  const head = h(
    'div',
    { class: 'validation__head' },
    h('div', { class: 'validation__heading' }, titleEl, metaEl),
    h('div', { class: 'validation__nav' }, prevButton, nextButton, closeButton),
  );

  const pickSelect = { a: null, b: null };
  for (const side of ['a', 'b']) {
    pickSelect[side] = h('select', {
      class: 'select validation__select validation__pick',
      id: ids[side],
      onchange: () => {
        jd.pick[side] = pickSelect[side].value;
        jd.userPicked = true;
        if (jd.pick[side] === 'other' && !historyRow(jd.other)?.url) jd.other = otherDefault();
        renderStrip();
        applySources();
      },
    });
  }
  const revealButtons = {};
  const pickers = h('div', { class: 'validation__pickers' }, ['a', 'b'].map((side) => {
    const label = `Image ${side.toUpperCase()}`;
    const button = h('button', {
      class: 'btn btn--sm', type: 'button', hidden: true,
      'aria-label': `Open ${label} in Explorer`, title: 'Select this file in Explorer on the training PC',
      onclick: async () => {
        const url = sourceFor(jd.pick[side])?.url;
        if (!url) return;
        button.disabled = true;
        try {
          await api.revealImage(Object.fromEntries(new URL(url, location.href).searchParams));
          toast('Opened in Explorer on the training PC.');
        } catch (error) {
          toast(reasonOf(error), { kind: 'err' });
        } finally {
          button.disabled = false;
        }
      },
    }, 'Open in Explorer');
    revealButtons[side] = button;
    return field(ids[side], label, h('div', { class: 'row row--wrap' }, pickSelect[side], button));
  }));

  const strip = h('div', { class: 'validation__strip', role: 'group', 'aria-labelledby': ids.strip });
  const stripField = h(
    'div',
    { class: 'validation__stripfield', hidden: true },
    h(
      'div',
      { class: 'validation__striphead' },
      h('span', { class: 'validation__label', id: ids.strip }, 'Output at another iteration'),
      h('span', { class: 'validation__hint' }, 'Newest first. Arrow keys step through the iterations.'),
    ),
    strip,
  );
  const historyEl = h('p', { class: 'validation__history', role: 'status' });
  // Filmstrip: the loaded samples of the open iteration, so another sample is one click away
  // while the stage fills the panel and the grid is out of sight.
  const filmLabelId = uid('validation-film');
  // On the stage booth like the grid (neutral surround, the booth's lighter focus colour).
  const film = h('ul', { class: 'stage validation__film', 'aria-labelledby': filmLabelId, onkeydown: onFilmKey });
  const filmField = h(
    'div',
    { class: 'validation__filmfield', hidden: true },
    h('span', { class: 'validation__label', id: filmLabelId }, 'Samples at this iteration'),
    film,
  );
  const stageHost = h('div', { class: 'validation__stage' });
  const filesEl = h('dl', { class: 'kv validation__files' });
  // "Measure the difference…": the two images on the stage as an image-difference job (analysis.js).
  const measureButton = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => openMeasure() }, 'Measure the difference…');
  const measureHelp = h('span', { class: 'validation__hint' });
  const measureEl = h('div', { class: 'validation__measure', hidden: true }, measureButton, measureHelp);
  let measureKey = '';
  let measuring = false;
  // The strip and its status sit under the stage, so the stage never moves while they come and go.
  const judge = h(
    'section',
    { class: 'validation__judge', 'aria-label': 'Compare one sample', hidden: true, onkeydown: onJudgeKey },
    head,
    pickers,
    stageHost,
    stripField,
    historyEl,
    filmField,
    filesEl,
    measureEl,
  );

  // ───────── gallery ─────────
  const statusEl = h('div', { class: 'validation__status', role: 'status' });
  const emptyEl = h('div', { class: 'validation__empty', hidden: true });
  const grid = h('ul', { class: 'validation__grid', 'aria-label': 'Validation images', onkeydown: onGridKey });
  const sheet = h('div', { class: 'stage validation__sheet', hidden: true }, grid);
  const moreButton = h('button', { class: 'btn', type: 'button', hidden: true, onclick: () => loadMore() }, 'Retry');
  const moreError = h('p', { class: 'validation__moreerror', role: 'status' });
  const moreEl = h('div', { class: 'validation__more', hidden: true }, moreButton, moreError);

  // ───────── Images / Scores switch ─────────
  let view = persist.get(VIEW_KEY, 'images') === 'scores' ? 'scores' : 'images';
  /** true / false once GET /api/run/validation answered for this run; null before */
  let measured = null;
  let autoChosen = false;
  let scores = null;
  let scoresLoading = false;
  const viewButtons = {
    images: h('button', { class: 'btn btn--sm', type: 'button', onclick: () => showView('images', true) }, 'Images'),
    scores: h('button', { class: 'btn btn--sm', type: 'button', onclick: () => showView('scores', true) }, 'Scores'),
  };
  const viewBar = h('div', { class: 'val__switch seg', role: 'group', 'aria-label': 'Validation view' }, viewButtons.images, viewButtons.scores);
  const imagesPane = h('div', { class: 'validation__pane' }, bar, provEl, errorEl, judge, statusEl, emptyEl, sheet, moreEl);
  const scoresPane = h('div', { class: 'validation__pane', hidden: true });

  // Sticky at the top of the panel, so the switch stays in reach on a long Scores page.
  const root = h('div', { class: 'validation' }, h('div', { class: 'val__viewbar' }, viewBar), imagesPane, scoresPane);
  host.appendChild(root);

  const pageObserver = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting && !destroyed && view === 'images' && !st.moreError) loadMore();
  }, { rootMargin: '0px 0px 240px 0px' });

  function showView(next, remember) {
    view = next;
    if (remember) persist.set(VIEW_KEY, next);
    imagesPane.hidden = next !== 'images';
    scoresPane.hidden = next !== 'scores';
    for (const [key, button] of Object.entries(viewButtons)) button.setAttribute('aria-pressed', String(key === next));
    if (next === 'scores' && !scores && !scoresLoading && current) {
      scoresLoading = true;
      mount(scoresPane, h('p', { class: 'validation__status', role: 'status' }, 'Loading validation results…'));
      import('./scores.js')
        .then((mod) => {
          scoresLoading = false;
          if (destroyed) return;
          scoresPane.replaceChildren();
          scores = mod.mountScores(scoresPane, {
            run: current,
            galleryHas: (step) => st.iterations.has(step),
            onShowImages: (step) => showImagesAt(step),
          });
        })
        .catch((error) => {
          scoresLoading = false;
          if (!destroyed) mount(scoresPane, notice('err', `The scores could not be shown: ${reasonOf(error)}`));
        });
    }
  }

  function showImagesAt(step) {
    showView('images', true);
    st.filter.iteration = String(step);
    st.filter.dataset = '';
    optionsKey = '';
    loadList();
  }

  /** No gallery images but measured rows: open on Scores (once per run, not remembered). */
  function decideView() {
    if (autoChosen || measured === null || st.phase !== 'ready') return;
    if (st.filter.iteration !== '' || st.filter.dataset !== '') return;
    autoChosen = true;
    if (!st.rows.length && measured === true) showView('scores', false);
    renderEmpty();
  }

  async function probeMeasured(runId) {
    try {
      const page = await api.validation(runId, { limit: 1 });
      if (destroyed || current?.id !== runId) return;
      measured = Array.isArray(page?.evaluations) && page.evaluations.length > 0;
    } catch {
      // Unknown, not "none": the empty state then says nothing about numbers, and Scores shows the error.
      if (destroyed || current?.id !== runId) return;
      measured = 'unknown';
    }
    decideView();
  }

  // ───────── loading the list ─────────
  function listParams(limit, after) {
    const params = { limit };
    if (st.filter.iteration !== '') params.iteration = st.filter.iteration;
    if (st.filter.dataset !== '' && st.provenance === 'recorded') params.dataset = st.filter.dataset;
    if (after) params.after = after;
    return params;
  }

  function resetRows() {
    st.rows = [];
    st.keys = new Set();
    st.next = null;
  }

  /** Add one API page. Rows without a usable key are kept out (nothing to select or label). */
  function accept(page) {
    st.canReveal = page?.canReveal === true;
    const rows = Array.isArray(page?.images) ? page.images : [];
    if (page?.provenance === 'recorded' || page?.provenance === 'filename') st.provenance = page.provenance;
    st.next = typeof page?.next === 'string' && page.next ? page.next : null;
    for (const row of rows) {
      if (!row || typeof row !== 'object' || row.image === undefined || row.image === null) continue;
      const key = keyOf(row);
      if (st.keys.has(key)) continue;
      st.keys.add(key);
      st.rows.push(row);
      if (isIter(row.iteration)) st.iterations.add(row.iteration);
      if (typeof row.dataset === 'string') st.datasets.add(row.dataset);
    }
  }

  /** check: a "Check for new images" reload that keeps as many rows as were loaded and says what is new. */
  async function loadList({ check = false } = {}) {
    if (!current || current.id === undefined || current.id === null) return;
    listAbort?.abort();
    moreAbort?.abort();
    const ctrl = new AbortController();
    listAbort = ctrl;
    st.gen += 1;
    const gen = st.gen;
    const before = check ? new Set(st.keys) : null;
    const limit = check ? Math.min(MAX_PAGE, Math.max(PAGE, st.rows.length)) : PAGE;
    st.checking = check;
    st.phase = 'loading';
    st.error = '';
    st.more = false;
    st.moreError = '';
    st.fresh = null;
    renderStatus();
    renderCheck();
    try {
      const page = await api.images(current.id, listParams(limit), { signal: ctrl.signal });
      if (gen !== st.gen || destroyed) return;
      resetRows();
      accept(page);
      st.phase = 'ready';
      st.loadedActive = runStatus(current.status).active;
      if (before) st.fresh = st.rows.filter((row) => !before.has(keyOf(row))).length;
      // Keep the open sample in step with the fresh rows (sizes and source addresses may be new).
      if (jd.row) {
        const fresh = st.rows.find((row) => keyOf(row) === keyOf(jd.row));
        if (fresh) jd.row = fresh;
      }
    } catch (error) {
      if (error?.name === 'AbortError' || gen !== st.gen || destroyed) return;
      st.phase = 'error';
      st.error = reasonOf(error);
    } finally {
      if (listAbort === ctrl) listAbort = null;
    }
    renderAll();
    scores?.galleryChanged();
    decideView();
  }

  async function loadMore() {
    if (!st.next || st.more || st.phase !== 'ready' || !current) return;
    moreAbort?.abort();
    const ctrl = new AbortController();
    moreAbort = ctrl;
    const gen = st.gen;
    const firstNew = st.rows.length;
    st.more = true;
    st.moreError = '';
    renderMore();
    try {
      const page = await api.images(current.id, listParams(PAGE, st.next), { signal: ctrl.signal });
      if (gen !== st.gen || destroyed) return;
      const hadFocus = document.activeElement === moreButton;
      accept(page);
      st.more = false;
      renderAll();
      // The button disappears with the last page: hand focus to the first image it brought.
      if (hadFocus && moreEl.hidden) {
        const row = st.rows.slice(firstNew).find((r) => tiles.has(keyOf(r)));
        if (row) focusTile(keyOf(row));
      }
    } catch (error) {
      if (error?.name === 'AbortError' || gen !== st.gen || destroyed) return;
      st.more = false;
      st.moreError = `More images could not be loaded: ${reasonOf(error)}`;
      renderMore();
    } finally {
      if (moreAbort === ctrl) moreAbort = null;
    }
  }

  function setFilter(patch) {
    Object.assign(st.filter, patch);
    loadList();
  }

  function checkForNew() {
    loadList({ check: true });
    if (jd.open && jd.row) loadHistory(jd.row, { keep: true });
  }

  // ───────── rendering the list ─────────
  /** rows after the client-side name filter, in API order */
  let visible = [];
  /** key -> tile button */
  const tiles = new Map();
  let shownKeys = [];
  let rovingKey = null;
  let optionsKey = '';

  function renderAll() {
    renderFilters();
    renderProv();
    renderList();
    renderMore();
    renderCheck();
  }

  function renderFilters() {
    const iterations = [...st.iterations].sort((x, y) => y - x);
    const datasets = [...st.datasets].sort();
    const key = JSON.stringify([iterations, datasets, st.filter.iteration, st.filter.dataset]);
    if (key === optionsKey) return;
    optionsKey = key;
    mount(
      iterSelect,
      h('option', { value: '' }, 'All iterations'),
      iterations.map((n) => h('option', { value: String(n) }, fmt.int(n))),
    );
    iterSelect.value = st.filter.iteration;
    mount(
      datasetSelect,
      h('option', { value: '' }, 'All datasets'),
      datasets.map((name) => h('option', { value: name }, name)),
    );
    datasetSelect.value = st.filter.dataset;
    datasetField.hidden = !(st.provenance === 'recorded' && (datasets.length > 1 || st.filter.dataset !== ''));
  }

  function renderProv() {
    const text = PROVENANCE[st.provenance] ?? '';
    provEl.textContent = text;
    provEl.hidden = !text;
  }

  function renderCheck() {
    const active = current ? runStatus(current.status).active : false;
    // While training, and once more after it ends (the last validation may have saved new images).
    checkButton.hidden = !(active || st.loadedActive);
    checkButton.disabled = st.phase === 'loading';
  }

  function filterSuffix() {
    const parts = [];
    if (st.filter.iteration !== '') parts.push(` at iteration ${fmt.int(Number(st.filter.iteration))}`);
    if (st.filter.dataset !== '') parts.push(` in ${st.filter.dataset}`);
    return parts.join('');
  }

  function countLine() {
    const n = st.rows.length;
    if (!n) return '';
    const more = st.next ? ', more are available' : '';
    const text = st.filter.text.trim();
    let line;
    if (text) line = `${fmt.int(visible.length)} of ${fmt.plural(n, 'loaded image')} match “${text}”${more}.`;
    else if (st.next) line = `Showing ${fmt.plural(n, 'image')}${filterSuffix()}, more are available.`;
    else line = `Showing ${n === 1 ? '' : 'all '}${fmt.plural(n, 'image')}${filterSuffix()}.`;
    if (st.fresh !== null) line += st.fresh ? ` ${fmt.plural(st.fresh, 'new image')} since the last check.` : ' No new images since the last check.';
    return line;
  }

  function renderStatus() {
    if (st.phase === 'loading') {
      mount(statusEl, h('span', { class: 'spinner', 'aria-hidden': 'true' }), st.checking ? 'Checking for new images…' : 'Loading validation images…');
    } else if (st.phase === 'ready') {
      mount(statusEl, countLine());
    } else {
      mount(statusEl);
    }
    statusEl.hidden = !statusEl.firstChild;
    const showError = st.phase === 'error';
    const errorKey = showError ? st.error : '';
    if (errorEl.dataset.key !== errorKey) {
      errorEl.dataset.key = errorKey;
      mount(
        errorEl,
        showError
          ? notice('err', `The validation images could not be loaded: ${st.error}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => loadList() }, 'Try again'))
          : null,
      );
    }
    errorEl.hidden = !showError;
  }

  function renderEmpty() {
    let node = null;
    if (st.phase === 'ready' && !st.rows.length) {
      const filtered = st.filter.iteration !== '' || st.filter.dataset !== '';
      if (filtered) {
        node = emptyState({
          title: 'No images match these filters',
          actions: [h('button', { class: 'btn', type: 'button', onclick: () => showAll() }, 'Show all images')],
        });
      } else {
        const active = current ? runStatus(current.status).active : false;
        node = emptyState({
          title: measured === false ? 'This run saved no validation images and recorded no validation numbers' : 'This run saved no validation images',
          actions: measured === true ? [h('button', { class: 'btn', type: 'button', onclick: () => showView('scores', true) }, 'Show scores')] : [],
          text: [
            measured === true ? 'It did record validation numbers: see Scores. ' : '',
            'The trainer saves them when the recipe turns on ',
            code('val.val_enabled'),
            ' and ',
            code('val.save_img'),
            '.',
            active ? ' While it trains, new images appear after each validation: use Check for new images.' : '',
          ],
        });
      }
    } else if (st.phase === 'ready' && !visible.length) {
      node = emptyState({
        title: `No loaded image has a sample name containing “${st.filter.text.trim()}”`,
        text: st.next ? 'Searching the remaining images…' : '',
      });
    }
    mount(emptyEl, node);
    emptyEl.hidden = !node;
  }

  function showAll() {
    st.filter.iteration = '';
    st.filter.dataset = '';
    optionsKey = '';
    loadList();
  }

  function matches(row, text) {
    return sampleText(row).toLowerCase().includes(text);
  }

  function renderList() {
    const text = st.filter.text.trim().toLowerCase();
    visible = text ? st.rows.filter((row) => matches(row, text)) : st.rows.slice();
    const keys = visible.map(keyOf);
    // Appending (Load more) keeps the tiles already there, and the focus with them.
    const isPrefix = shownKeys.length <= keys.length && shownKeys.every((key, i) => key === keys[i]);
    if (!isPrefix) {
      grid.replaceChildren();
      tiles.clear();
      shownKeys = [];
    }
    const frag = document.createDocumentFragment();
    for (let i = shownKeys.length; i < visible.length; i += 1) frag.appendChild(tileFor(visible[i]));
    grid.appendChild(frag);
    shownKeys = keys;
    sheet.hidden = !visible.length;
    updateRoving();
    markSelected();
    renderStatus();
    renderEmpty();
    renderPosition();
  }

  function tileFor(row) {
    const key = keyOf(row);
    const thumb = h('span', { class: 'validation__thumb' });
    const src = sizedUrl(row.url, THUMB_SIZE);
    if (src) {
      const img = h('img', { class: 'validation__thumbimg', alt: '', loading: 'lazy', decoding: 'async', draggable: 'false' });
      img.addEventListener('error', () => mount(thumb, h('span', { class: 'validation__nothumb' }, 'No preview')), { once: true });
      img.src = src;
      thumb.appendChild(img);
    } else {
      thumb.appendChild(h('span', { class: 'validation__nothumb' }, 'No preview'));
    }
    const showDataset = typeof row.dataset === 'string' && st.datasets.size > 1;
    const button = h(
      'button',
      { class: 'validation__tile', type: 'button', tabindex: '-1', dataset: { key }, onclick: () => openJudge(row, { scroll: true }) },
      thumb,
      h('span', { class: 'validation__tilename', title: sampleText(row) }, sampleLabel(row)),
      h('span', { class: 'validation__tilemeta' }, iterText(row.iteration)),
      showDataset ? h('span', { class: 'validation__tilemeta', title: row.dataset }, row.dataset) : null,
    );
    tiles.set(key, button);
    return h('li', { class: 'validation__item' }, button);
  }

  /** One tile in the Tab order: the open sample when it is in the grid, else the first. */
  function updateRoving(preferred) {
    const want = preferred ?? (jd.row && tiles.has(keyOf(jd.row)) ? keyOf(jd.row) : tiles.has(rovingKey) ? rovingKey : shownKeys[0] ?? null);
    if (rovingKey && rovingKey !== want) {
      const old = tiles.get(rovingKey);
      if (old) old.tabIndex = -1;
    }
    rovingKey = want;
    const tile = want ? tiles.get(want) : null;
    if (tile) tile.tabIndex = 0;
  }

  function focusTile(key) {
    const tile = tiles.get(key);
    if (!tile) return;
    updateRoving(key);
    tile.focus();
  }

  function markSelected() {
    const selected = jd.open && jd.row ? keyOf(jd.row) : null;
    for (const [key, tile] of tiles) {
      if (key === selected) tile.setAttribute('aria-current', 'true');
      else if (tile.hasAttribute('aria-current')) tile.removeAttribute('aria-current');
    }
  }

  function columns() {
    const first = grid.firstElementChild;
    if (!first) return 1;
    const top = first.offsetTop;
    let n = 0;
    for (const item of grid.children) {
      if (item.offsetTop !== top) break;
      n += 1;
    }
    return Math.max(1, n);
  }

  function onGridKey(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const key = event.target?.dataset?.key;
    const index = key === undefined ? -1 : shownKeys.indexOf(key);
    if (index < 0) return;
    const cols = columns();
    const last = shownKeys.length - 1;
    const target = { ArrowRight: index + 1, ArrowLeft: index - 1, ArrowDown: index + cols, ArrowUp: index - cols, Home: 0, End: last }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    focusTile(shownKeys[Math.max(0, Math.min(last, target))]);
  }

  function renderMore() {
    moreEl.hidden = !(st.phase === 'ready' && st.next);
    moreButton.hidden = !st.moreError;
    moreError.textContent = st.moreError || (st.more ? 'Loading more images…' : '');
    moreError.hidden = !moreError.textContent;
    pageObserver.unobserve(moreEl);
    if (!moreEl.hidden && !st.more && !st.moreError) pageObserver.observe(moreEl);
  }

  // ───────── one sample on the stage ─────────
  function ensureCompare() {
    if (!compare) compare = createCompare(stageHost, { emptyText: 'Choose what image A and image B show.' });
  }

  function openJudge(row, { scroll = false } = {}) {
    const wasOpen = jd.open;
    const sameSample = jd.row && sampleKeyOf(jd.row) === sampleKeyOf(row);
    jd.row = row;
    jd.open = true;
    judge.hidden = false;
    ensureCompare();
    if (!sameSample) {
      jd.history = null;
      loadHistory(row);
    }
    resolvePicks();
    renderJudge();
    applySources();
    markSelected();
    updateRoving();
    if (scroll) {
      // Opened from a tile: bring the stage into view when its top is not, and move focus to the
      // sample's heading so N / P work at once and the focus is never left on a tile out of sight.
      const top = judge.getBoundingClientRect().top;
      if (!wasOpen || top < 0 || top > window.innerHeight - 80) judge.scrollIntoView({ block: 'start' });
      titleEl.focus({ preventScroll: true });
    }
  }

  function closeJudge(focus) {
    const key = jd.row ? keyOf(jd.row) : null;
    historyAbort?.abort();
    historyAbort = null;
    jd.historyGen += 1;
    jd.open = false;
    jd.row = null;
    jd.history = null;
    judge.hidden = true;
    compare?.setSources({ a: null, b: null });
    sourceKey = '';
    markSelected();
    if (focus && key) {
      if (tiles.has(key)) focusTile(key);
      else (tiles.get(rovingKey) ?? nameInput).focus();
    }
  }

  function step(dir) {
    if (!jd.row || !visible.length) return;
    const index = visible.findIndex((row) => keyOf(row) === keyOf(jd.row));
    const target = index < 0 ? (dir > 0 ? 0 : visible.length - 1) : index + dir;
    if (target < 0 || target >= visible.length) return;
    const focused = document.activeElement;
    const inside = judge.contains(focused);
    openJudge(visible[target]);
    // A control that held focus may have been rebuilt (the strip) or disabled (at either end).
    const now = document.activeElement;
    if (inside && (!judge.contains(now) || now.disabled)) {
      const order = dir > 0 ? [nextButton, prevButton] : [prevButton, nextButton];
      (order.find((b) => !b.disabled) ?? titleEl).focus({ preventScroll: true });
    }
  }

  function onJudgeKey(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const t = event.target;
    if (t instanceof HTMLInputElement || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) return;
    const key = event.key.toLowerCase();
    if (key === 'n') step(1);
    else if (key === 'p') step(-1);
    else return;
    event.preventDefault();
  }

  /** keep: a refresh of the same sample keeps the strip it has until the new history arrives */
  async function loadHistory(row, { keep = false } = {}) {
    historyAbort?.abort();
    const ctrl = new AbortController();
    historyAbort = ctrl;
    jd.historyGen += 1;
    const gen = jd.historyGen;
    if (!keep) {
      jd.history = null;
      jd.historyError = '';
      jd.historyCapped = false;
      renderHistory();
    }
    const params = { image: row.image, limit: MAX_PAGE };
    // Filename-mode rows have dataset null; passing a dataset there returns nothing.
    if (typeof row.dataset === 'string') params.dataset = row.dataset;
    const found = new Map();
    let after = null;
    try {
      for (let page = 0; page < HISTORY_PAGES; page += 1) {
        const result = await api.images(current.id, after ? { ...params, after } : params, { signal: ctrl.signal });
        if (gen !== jd.historyGen || destroyed) return;
        for (const item of Array.isArray(result?.images) ? result.images : []) {
          if (!item || item.image !== row.image || (item.dataset ?? null) !== (row.dataset ?? null) || !isIter(item.iteration)) continue;
          if (!found.has(item.iteration)) found.set(item.iteration, item);
        }
        after = typeof result?.next === 'string' && result.next ? result.next : null;
        if (!after) break;
      }
      if (isIter(row.iteration) && !found.has(row.iteration)) found.set(row.iteration, row);
      jd.history = [...found.values()].sort((x, y) => y.iteration - x.iteration);
      jd.historyCapped = Boolean(after);
      jd.historyError = '';
    } catch (error) {
      if (error?.name === 'AbortError' || gen !== jd.historyGen || destroyed) return;
      jd.historyError = reasonOf(error);
      jd.history = isIter(row.iteration) ? [row] : [];
    } finally {
      if (historyAbort === ctrl) historyAbort = null;
    }
    resolvePicks();
    renderJudge();
    applySources();
  }

  function historyRow(iteration) {
    return iteration === null || iteration === undefined ? null : (jd.history ?? []).find((row) => row.iteration === iteration) ?? null;
  }

  /**
   * Iteration for "Output at another iteration": the previous one (newest saved before the open one),
   * else the nearest later one when the open iteration is the first saved; null when there is none.
   */
  function otherDefault() {
    const n = jd.row?.iteration;
    if (!isIter(n)) return null;
    const rows = (jd.history ?? []).filter((row) => row.url && row.iteration !== n);
    const older = rows.find((row) => row.iteration < n);
    if (older) return older.iteration;
    const newer = rows.filter((row) => row.iteration > n).pop();
    return newer ? newer.iteration : null;
  }

  /** Which choices really exist for the open sample now. */
  function available(choice) {
    const row = jd.row;
    if (!row) return false;
    if (choice === 'gt') return Boolean(row.gtUrl);
    if (choice === 'lq') return Boolean(row.lqUrl);
    if (choice === 'out') return Boolean(row.url);
    if (choice === 'other') return jd.history === null || (jd.history.length > 1 && jd.history.some((r) => r.url));
    return choice === '';
  }

  /** Default pair: A = ground truth, else input, else the previous iteration's output; B = the output. */
  function defaultPick(side) {
    if (side === 'b') return available('out') ? 'out' : '';
    if (available('gt')) return 'gt';
    if (available('lq')) return 'lq';
    return 'other';
  }

  function resolvePicks() {
    for (const side of ['a', 'b']) {
      if (!jd.userPicked || !available(jd.pick[side])) jd.pick[side] = defaultPick(side);
    }
    const wantsOther = jd.pick.a === 'other' || jd.pick.b === 'other';
    // Until the history arrives "other" waits; then it needs a real iteration of this sample.
    // The default pairing always means the previous iteration of the sample now open.
    if (wantsOther && jd.history !== null && (!jd.userPicked || !historyRow(jd.other)?.url)) jd.other = otherDefault();
    if (jd.history !== null) {
      for (const side of ['a', 'b']) {
        if (jd.pick[side] === 'other' && !historyRow(jd.other)?.url) jd.pick[side] = side === 'b' && available('out') ? 'out' : '';
      }
    }
  }

  /** {source, url} for one choice, or null when it does not exist for the open sample. */
  function sourceFor(choice) {
    const row = jd.row;
    if (!row) return null;
    const plain = (url, source) => ({ url, source: { ...source, load: () => Promise.resolve(url) } });
    if (choice === 'gt' && row.gtUrl) {
      return plain(row.gtUrl, { label: 'Ground truth', role: 'gt', note: `Preview of the recorded source, at most ${PREVIEW_MAX} px` });
    }
    if (choice === 'lq' && row.lqUrl) {
      return plain(row.lqUrl, {
        label: 'Low-quality input',
        role: 'lq',
        // Filename mode: lqUrl is the <sample>_lr.png the trainer saved, served as it is (not a preview).
        note: isRecorded(row) ? `Preview of the recorded input, at most ${PREVIEW_MAX} px` : 'Input image the trainer saved beside the output',
      });
    }
    const out = (r) => {
      const sized = Number.isInteger(r.width) && Number.isInteger(r.height);
      return plain(r.url, {
        label: `Output at iteration ${fmt.int(r.iteration)}`,
        role: 'out',
        note: sized ? `Original output, ${fmt.dims(r.width, r.height)}` : 'Original output file',
      });
    };
    if (choice === 'out' && row.url) return out(row);
    if (choice === 'other') {
      const other = historyRow(jd.other);
      if (other?.url) return out(other);
    }
    return null;
  }

  /**
   * What the image-difference job can take for one choice of the stage: a saved output pinned to the
   * version its link carries ({run, path, version}), or the path of the recorded ground truth. Only PNG
   * files; the low quality input is left out (at another size than the output it cannot be compared).
   */
  function measureSide(choice) {
    const isPng = (value) => typeof value === 'string' && /\.png$/i.test(value);
    if (choice === 'gt') return jd.row && isRecorded(jd.row) && isPng(jd.row.gt) ? { value: jd.row.gt, words: 'the ground truth' } : null;
    const row = choice === 'out' ? jd.row : choice === 'other' ? historyRow(jd.other) : null;
    if (!row?.url || !isPng(row.path) || !current) return null;
    let version = '';
    try {
      version = new URL(row.url, window.location.href).searchParams.get('version') ?? '';
    } catch {
      version = '';
    }
    if (!version) return null;
    const words = `the output at iteration ${fmt.int(row.iteration)}`;
    return { value: { run: current.id, path: row.path, version, label: `${sampleLabel(row)}, output at iteration ${fmt.int(row.iteration)}` }, words };
  }

  /** The request the button would open, or null when the stage does not show two images the job can take. */
  function measurePlan() {
    if (!jd.row || jd.pick.a === jd.pick.b) return null;
    const left = measureSide(jd.pick.a);
    const right = measureSide(jd.pick.b);
    if (!left || !right) return null;
    // Two outputs: the recorded ground truth is the reference both are judged against.
    const reference = jd.pick.a !== 'gt' && jd.pick.b !== 'gt' ? measureSide('gt') : null;
    return { left, right, reference };
  }

  function renderMeasure() {
    const plan = measurePlan();
    const key = plan ? JSON.stringify(plan) : '';
    if (key === measureKey) return;
    measureKey = key;
    // Never taken away from under the keyboard: the button stays while it has focus.
    if (!plan && document.activeElement === measureButton) return;
    measureEl.hidden = !plan;
    if (!plan) return;
    measureHelp.textContent = `Counts, pixel by pixel, how far ${plan.left.words} is from ${plan.right.words}${plan.reference ? ', and which of the two is closer to the ground truth' : ''}. It runs as an analysis job on the CPU.`;
  }

  async function openMeasure() {
    const plan = measurePlan();
    if (!plan || measuring) return;
    measuring = true;
    try {
      const mod = await import('./analysis.js');
      if (typeof mod.openAnalysisDialog !== 'function') throw new Error('The analysis part has no openAnalysisDialog.');
      await mod.openAnalysisDialog({ kind: 'image-difference', left: plan.left.value, right: plan.right.value, reference: plan.reference?.value ?? '' });
    } catch (error) {
      toast(`The image difference dialog could not be opened: ${error?.message ?? error}`, { kind: 'err' });
    } finally {
      measuring = false;
    }
  }

  /** Hand the stage new sources only when what it shows really changes (no reload under the person). */
  function applySources() {
    renderMeasure();
    if (!compare || !jd.row) return;
    const a = sourceFor(jd.pick.a);
    const b = sourceFor(jd.pick.b);
    revealButtons.a.hidden = !st.canReveal || !a;
    revealButtons.b.hidden = !st.canReveal || !b;
    const key = JSON.stringify([a?.url, a?.source.label, a?.source.note, b?.url, b?.source.label, b?.source.note]);
    if (key === sourceKey) return;
    sourceKey = key;
    compare.setSources({ a: a?.source ?? null, b: b?.source ?? null });
  }

  function renderJudge() {
    if (!jd.row) return;
    titleEl.textContent = sampleLabel(jd.row);
    titleEl.title = sampleText(jd.row);
    renderPickers();
    renderStrip();
    renderHistory();
    renderFiles();
    renderPosition();
  }

  /** Rows of the filmstrip: the listed (name-filtered) rows at the open iteration, in list order. */
  function filmRows() {
    const n = jd.row?.iteration;
    return isIter(n) ? visible.filter((row) => row.iteration === n) : [];
  }

  function renderFilm() {
    if (!jd.open || !jd.row) return;
    const rows = filmRows();
    const key = rows.map(keyOf).join('\n');
    // Rebuilt only when the set changes, so stepping through samples keeps focus and scroll.
    if (film.dataset.key !== key) {
      film.dataset.key = key;
      mount(
        film,
        rows.map((row) => {
          const thumb = h('span', { class: 'validation__filmthumb' });
          const src = sizedUrl(row.url, FILM_SIZE);
          if (src) {
            const img = h('img', { class: 'validation__thumbimg', alt: '', loading: 'lazy', decoding: 'async', draggable: 'false' });
            img.addEventListener('error', () => mount(thumb, h('span', { class: 'validation__nothumb' }, 'No preview')), { once: true });
            img.src = src;
            thumb.appendChild(img);
          } else {
            thumb.appendChild(h('span', { class: 'validation__nothumb' }, 'No preview'));
          }
          return h(
            'li',
            { class: 'validation__filmitem' },
            h(
              'button',
              { class: 'validation__filmtile', type: 'button', tabindex: '-1', title: sampleText(row), dataset: { key: keyOf(row) }, onclick: () => openJudge(row) },
              thumb,
              h('span', { class: 'validation__filmname' }, sampleLabel(row)),
            ),
          );
        }),
      );
    }
    filmField.hidden = rows.length < 2;
    const current = keyOf(jd.row);
    let roving = null;
    for (const item of film.children) {
      const button = item.firstElementChild;
      const on = button.dataset.key === current;
      if (on) {
        button.setAttribute('aria-current', 'true');
        roving = button;
      } else if (button.hasAttribute('aria-current')) button.removeAttribute('aria-current');
      button.tabIndex = -1;
    }
    const stop = roving ?? film.firstElementChild?.firstElementChild;
    if (stop) stop.tabIndex = 0;
    // Keep the current one in view horizontally only (scrollIntoView would also scroll the panel).
    if (roving && !filmField.hidden) {
      const item = roving.parentNode;
      if (item.offsetLeft < film.scrollLeft) film.scrollLeft = item.offsetLeft;
      else if (item.offsetLeft + item.offsetWidth > film.scrollLeft + film.clientWidth) film.scrollLeft = item.offsetLeft + item.offsetWidth - film.clientWidth;
    }
  }

  function onFilmKey(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const buttons = [...film.children].map((item) => item.firstElementChild);
    const index = buttons.indexOf(event.target);
    if (index < 0) return;
    const target = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: buttons.length - 1 }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    const button = buttons[Math.max(0, Math.min(buttons.length - 1, target))];
    const row = filmRows().find((r) => keyOf(r) === button.dataset.key);
    if (row) openJudge(row);
    button.focus();
  }

  function renderPosition() {
    if (!jd.open || !jd.row) return;
    renderFilm();
    const index = visible.findIndex((row) => keyOf(row) === keyOf(jd.row));
    const parts = [];
    if (typeof jd.row.dataset === 'string') parts.push(h('span', null, `Dataset ${jd.row.dataset}`));
    parts.push(h('span', { class: 'num' }, iterText(jd.row.iteration)));
    parts.push(
      h('span', { class: 'num' }, index < 0 ? 'Not in the filtered list' : `Image ${fmt.int(index + 1)} of ${fmt.int(visible.length)}${st.next ? ' loaded' : ''}`),
    );
    mount(metaEl, parts);
    prevButton.disabled = index === 0 || !visible.length;
    nextButton.disabled = (index >= 0 && index === visible.length - 1) || !visible.length;
  }

  function renderPickers() {
    const row = jd.row;
    const otherLabel =
      jd.history === null ? 'Output at another iteration (loading)' : 'Output at another iteration';
    const options = [
      available('gt') ? ['gt', 'Ground truth'] : null,
      available('lq') ? ['lq', 'Low-quality input'] : null,
      available('out') ? ['out', `Output at iteration ${fmt.int(row.iteration)}`] : null,
      available('other') ? ['other', otherLabel] : null,
      ['', 'No image'],
    ].filter(Boolean);
    for (const side of ['a', 'b']) {
      const select = pickSelect[side];
      const key = JSON.stringify(options);
      if (select.dataset.key !== key) {
        select.dataset.key = key;
        mount(
          select,
          options.map(([value, label]) => h('option', { value, disabled: value === 'other' && jd.history === null }, label)),
        );
      }
      select.value = jd.pick[side];
    }
  }

  function renderStrip() {
    const history = jd.history ?? [];
    const show = history.length > 1;
    stripField.hidden = !show;
    if (!show) {
      strip.replaceChildren();
      strip.dataset.key = '';
      return;
    }
    const key = history.map((row) => row.iteration).join(',');
    if (strip.dataset.key !== key) {
      strip.dataset.key = key;
      mount(
        strip,
        history.map((row) =>
          h(
            'button',
            {
              class: 'btn btn--sm validation__stripbtn',
              type: 'button',
              tabindex: '-1',
              dataset: { iteration: String(row.iteration) },
              'aria-label': `Output at iteration ${fmt.int(row.iteration)}`,
              disabled: !row.url,
              onclick: () => chooseOther(row.iteration),
              onkeydown: onStripKey,
            },
            fmt.int(row.iteration),
          ),
        ),
      );
    }
    const shown = jd.pick.a === 'other' || jd.pick.b === 'other';
    const focusIter = shown && historyRow(jd.other) ? jd.other : jd.row?.iteration;
    let roving = null;
    for (const button of strip.children) {
      const n = Number(button.dataset.iteration);
      button.setAttribute('aria-pressed', String(shown && n === jd.other));
      button.tabIndex = -1;
      if (n === focusIter) roving = button;
    }
    (roving ?? strip.firstElementChild).tabIndex = 0;
  }

  function chooseOther(iteration, focus = false) {
    if (!historyRow(iteration)?.url) return;
    jd.other = iteration;
    if (jd.pick.a !== 'other' && jd.pick.b !== 'other') jd.pick.b = 'other';
    jd.userPicked = true;
    renderPickers();
    renderStrip();
    applySources();
    if (focus) strip.querySelector(`[data-iteration="${iteration}"]`)?.focus();
  }

  function onStripKey(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const buttons = [...strip.children].filter((b) => !b.disabled);
    const index = buttons.indexOf(event.currentTarget);
    if (index < 0) return;
    const target = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: buttons.length - 1 }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    const button = buttons[Math.max(0, Math.min(buttons.length - 1, target))];
    chooseOther(Number(button.dataset.iteration), true);
  }

  function renderHistory() {
    let text = '';
    if (!jd.open) text = '';
    else if (jd.history === null) text = 'Loading the other iterations of this sample…';
    else if (jd.historyError) text = `The other iterations of this sample could not be loaded: ${jd.historyError}`;
    else if (jd.history.length <= 1) text = 'This sample was saved at one iteration only.';
    else text = `This sample was saved at ${fmt.plural(jd.history.length, 'iteration')}${jd.historyCapped ? ' (the newest shown; more exist)' : ''}.`;
    if (historyEl.textContent !== text) historyEl.textContent = text;
    historyEl.hidden = !text;
  }

  /** File names as text, full paths in a title (recorded sources are absolute paths on the trainer machine). */
  function renderFiles() {
    const row = jd.row;
    const items = [];
    const path = (value) =>
      typeof value === 'string' && value
        ? h('dd', { class: 'mono truncate', title: value }, lastSegment(value) || value)
        : h('dd', { class: 'muted' }, 'Not reported');
    items.push(h('dt', null, 'Output file'), path(row.path));
    if (row.gtUrl) items.push(h('dt', null, 'Ground truth source'), path(row.gt));
    if (row.lqUrl && isRecorded(row)) items.push(h('dt', null, 'Low-quality source'), path(row.lq));
    mount(filesEl, items);
  }

  function isRecorded(row) {
    return row?.provenance ? row.provenance === 'recorded' : st.provenance === 'recorded';
  }

  // ───────── lifecycle ─────────
  function reset() {
    listAbort?.abort();
    moreAbort?.abort();
    applyText.cancel();
    closeJudge(false);
    jd.userPicked = false;
    jd.other = null;
    jd.pick = { a: '', b: '' };
    st.phase = 'idle';
    st.error = '';
    st.provenance = null;
    st.iterations = new Set();
    st.datasets = new Set();
    st.filter = { iteration: '', dataset: '', text: '' };
    st.loadedActive = false;
    st.fresh = null;
    nameInput.value = '';
    optionsKey = '';
    measured = null;
    autoChosen = false;
    resetRows();
    renderAll();
    showView(persist.get(VIEW_KEY, 'images') === 'scores' ? 'scores' : 'images', false);
    loadList();
    if (current) probeMeasured(current.id);
  }

  if (current) reset();
  else mount(statusEl, 'Select a run to see its validation images.');

  return {
    setRun(next) {
      if (destroyed || !next) return;
      const before = current?.id;
      const wasActive = current ? runStatus(current.status).active : false;
      current = next;
      scores?.setRun(next);
      if (next.id !== before) {
        reset();
        return;
      }
      // Same run: never reload under the person; offer the check instead.
      renderCheck();
      if (runStatus(next.status).active !== wasActive) renderEmpty();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      pageObserver.disconnect();
      listAbort?.abort();
      moreAbort?.abort();
      historyAbort?.abort();
      applyText.cancel();
      compare?.destroy();
      compare = null;
      scores?.destroy();
      scores = null;
      tiles.clear();
      root.remove();
    },
  };
}
