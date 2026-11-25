// Folder and file picker on POST /api/fs/list. Everything listed comes from the trainer machine; the
// picker never guesses whether a path exists.
//
//   pickPath({title, mode: 'folder'|'file', start, extensions, confirmLabel}) -> Promise<string|null>
//   pickPath({mode: 'file', multiple: true, selected, confirmLabel: string | (count) => string, …})
//     -> Promise<string[]|null>: several files ticked in one visit, in natural order by path (never empty)
//
// Ticking several files (multiple, file mode only) is opt-in; without it the picker is exactly the one
// above. The ticked files are a set of absolute paths kept for the whole visit, so files of two folders
// can be chosen; `selected` seeds it.
//
// Two ways of browsing, one component:
//   - the desktop page: a click highlights a row, a double click or Enter opens it, and the button
//     returns the highlighted folder (or the folder on screen) or the highlighted file;
//   - the phone edition (mobile.html marks the document): there is no double click on glass, so one tap
//     opens a folder and one tap chooses a file. The button returns what the foot of the sheet names:
//     the folder on screen, or the chosen file. Back, Up and the path so far stay at the top of the
//     sheet, and the typed path has a Go button that can be reached while the keyboard is up.
//     css/path-picker.css lays that sheet out.

import { h, mount as fill, fmt, debounce, uid } from '../dom.js';
import { api } from '../api.js';
import { conn } from '../state/conn.js';
import { openDialog } from '../ui.js';

const PAGE = 200;
const FILTER_DELAY = 250;
/** Phone: taps on rows are ignored for this long after a folder appears (the second tap of a double tap). */
const TAP_GUARD = 350;
/** Phone: how many folders Back remembers. */
const TRAIL = 64;

// ───────── pure helpers (exported for checks) ─────────

/** Absolute child path of a listed folder. */
export function joinPath(folder, name) {
  const base = String(folder ?? '').replace(/\/+$/, '');
  return `${base}/${name}`;
}

/** Breadcrumb segments of an absolute POSIX path: [{label, path}], the first one being '/'. */
export function crumbs(path) {
  const parts = String(path ?? '').split('/').filter(Boolean);
  const out = [{ label: '/', path: '/' }];
  let acc = '';
  for (const part of parts) {
    acc += `/${part}`;
    out.push({ label: part, path: acc });
  }
  return out;
}

/**
 * The folder above a path as the person typed it (absolute, relative to the trainer folder, or Windows).
 * '' means the trainer folder; null means there is nothing above.
 */
export function parentOf(path) {
  const text = String(path ?? '').trim();
  if (!text || text === '/') return null;
  const trimmed = text.replace(/[\\/]+$/, '');
  if (!trimmed) return null; // "///"
  if (/^[A-Za-z]:$/.test(trimmed)) return '/mnt'; // a Windows drive lives under /mnt
  const cut = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
  if (cut < 0) return ''; // one relative segment: the trainer folder
  if (cut === 0) return '/';
  const head = trimmed.slice(0, cut);
  if (/^[A-Za-z]:$/.test(head)) return `${head}\\`;
  return head;
}

/** Last segment of a typed path ('' when there is none). */
export function baseName(path) {
  const trimmed = String(path ?? '').trim().replace(/[\\/]+$/, '');
  const cut = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
  return cut < 0 ? trimmed : trimmed.slice(cut + 1);
}

/** Whether a file name ends with one of the extensions (case-insensitive); true when there are none. */
export function matchesExtensions(name, extensions) {
  if (!Array.isArray(extensions) || !extensions.length) return true;
  const lower = String(name ?? '').toLowerCase();
  return extensions.some((ext) => lower.endsWith(String(ext).toLowerCase()));
}

/**
 * A typed or pasted path, ready to send: trimmed, and without the pair of quotes that "Copy as path"
 * on Windows (or a shell) puts around it.
 */
export function cleanTyped(text) {
  const value = String(text ?? '').trim();
  const quoted = /^"(.*)"$/.exec(value) ?? /^'(.*)'$/.exec(value);
  return quoted ? quoted[1].trim() : value;
}

const isAbort = (error) => error?.name === 'AbortError';

/** Paths in natural order (E2 before E10), the order the several-files mode returns. */
const naturalCompare = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' }).compare;
export function naturalSort(paths) {
  return [...paths].sort((a, b) => naturalCompare(a, b) || (a < b ? -1 : a > b ? 1 : 0));
}

/** The folder part of an absolute path ('' when there is none). */
export function folderOfPath(path) {
  const text = String(path ?? '').replace(/[\\/]+$/, '');
  const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
  return cut < 0 ? '' : cut === 0 ? text.slice(0, 1) : text.slice(0, cut);
}

// ───────── dialog ─────────

/**
 * @param {{title?: string, mode?: 'folder'|'file', start?: string, extensions?: string[]|null, showFilesInitially?: boolean, confirmLabel?: string}} options
 * @returns {Promise<string|null>} absolute path on the trainer machine, or null when cancelled
 */
export function pickPath({ title, mode = 'folder', start = '', extensions = null, confirmLabel, showFilesInitially = false, multiple = false, selected = [] } = {}) {
  const fileMode = mode === 'file';
  /** Several files ticked in one visit (file mode only); every other mode is untouched by it. */
  const multi = fileMode && multiple === true;
  /** The ticked files: absolute paths, kept while folders change. */
  const picked = new Set(multi && Array.isArray(selected) ? selected.filter((p) => typeof p === 'string' && p.trim()).map((p) => p.trim()) : []);
  /** The row ticked or unticked last in the folder on screen (the start of a Shift range). */
  let anchor = null;
  const phone = document.documentElement.dataset.edition === 'mobile';
  const ids = {
    path: uid('picker-path'),
    pathHelp: uid('picker-path-help'),
    filter: uid('picker-filter'),
    list: uid('picker-list'),
    status: uid('picker-status'),
    chosen: uid('picker-chosen'),
  };
  const labelFor = (count) => (typeof confirmLabel === 'function' ? confirmLabel(count) : confirmLabel || (multi ? 'Use these files' : fileMode ? 'Use this file' : 'Use this folder'));
  const confirmText = labelFor(picked.size);

  /** {path, parent, entries, next} of the folder on screen, or null before the first answer */
  let listing = null;
  /** highlighted entry {kind, name} in the folder on screen, or null */
  let active = null;
  let controller = null;
  let busy = false;
  let closed = false;
  let typed = '';
  let typedTimer = 0;
  /** "kind\0name" -> row element of the folder on screen */
  let rowByKey = new Map();
  /** phone: the folders shown before this one, oldest first (what Back returns to) */
  const trail = [];
  /** phone: when the folder on screen appeared, and the row whose folder is being opened */
  let shownAt = 0;
  let openingRow = null;
  /**
   * phone: how many times the path field was edited by hand, and that count when the field last showed
   * the folder on screen. They differ while the field holds text that has not been sent yet. A listing
   * asked for before an edit must not write over it (show): over a slow link the answer can arrive while
   * a path is being typed or pasted, and what Go sends has to be that path.
   */
  let pathEdits = 0;
  let pathShown = 0;

  // ── controls ──
  const crumbBar = h('nav', { class: 'picker__crumbs', 'aria-label': 'Current folder' });
  const backButton = phone ? h('button', { class: 'btn btn--sm', type: 'button', disabled: true, 'aria-label': 'Back to the folder shown before', onclick: () => goBack() }, 'Back') : null;
  const upButton = h('button', { class: 'btn btn--sm', type: 'button', disabled: true, 'aria-label': phone ? 'Up to the folder above' : null, onclick: () => goUp() }, 'Up');
  const bar = h('div', { class: 'picker__bar' }, backButton, upButton, crumbBar);
  const pathInput = h('input', {
    class: 'input input--path picker__pathinput',
    id: ids.path,
    type: 'text',
    autocomplete: 'off',
    autocapitalize: 'off',
    autocorrect: 'off',
    spellcheck: 'false',
    enterkeyhint: 'go',
    'aria-describedby': ids.pathHelp,
    onkeydown: (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      submitPath();
    },
    // Phone: the field holds the folder on screen, so a pasted path has to replace it. Touching the
    // field selects what is there (as an address bar does); the phone then offers Paste over it.
    onfocus: phone ? () => setTimeout(() => document.activeElement === pathInput && pathInput.setSelectionRange(0, pathInput.value.length), 0) : null,
    // Phone: typing, pasting, cutting and dictation all count as an edit (setting the value from here does not).
    oninput: phone
      ? () => {
          pathEdits += 1;
        }
      : null,
  });
  // Phone: a button for what Enter does, within reach while the keyboard is up. The keyboard is put away
  // once the folder is on screen (submitPath).
  const goButton = phone ? h('button', { class: 'btn picker__gobtn', type: 'button', 'aria-label': 'Go to this path', onclick: () => submitPath() }, 'Go') : null;
  const root = conn.get().info?.trainerRoot;
  const shortcut = (label, path) => h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => goTo(path) }, label);
  const shortcuts = h(
    'div',
    { class: 'picker__shortcuts', role: 'group', 'aria-label': 'Go to' },
    typeof root === 'string' && root ? shortcut('Trainer folder', root) : null,
    shortcut('Root (/)', '/'),
    shortcut('Windows drives (/mnt)', '/mnt'),
  );

  const filterLabel = 'Name contains (case-sensitive)';
  const filterInput = h('input', {
    class: 'input picker__filter',
    id: ids.filter,
    type: 'search',
    autocomplete: 'off',
    autocapitalize: 'off', // the filter is case-sensitive: a phone must not capitalise its first letter
    autocorrect: 'off',
    spellcheck: 'false',
    enterkeyhint: 'search',
    placeholder: phone ? filterLabel : null, // the phone sheet shows the label in the field itself
    oninput: () => reloadFiltered(),
    onkeydown: (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        reloadFiltered.flush();
        if (phone) filterInput.blur(); // the keyboard gives the list its room back
      }
    },
  });
  const reloadFiltered = debounce(() => {
    if (listing) load(listing.path);
  }, FILTER_DELAY);

  const toggle = (label, onchange) => {
    const input = h('input', { type: 'checkbox', role: 'switch', onchange });
    return { input, el: h('label', { class: 'switch' }, input, h('span', { class: 'switch__track', 'aria-hidden': 'true' }), label) };
  };
  const hidden = toggle('Show hidden', () => listing && load(listing.path));
  const showFiles = fileMode ? null : toggle('Show files', () => listing && load(listing.path, { keepActive: true }));
  if (showFiles) showFiles.input.checked = showFilesInitially;

  const noteBox = h('div', { class: 'picker__note' });
  const errorBox = h('p', { class: 'picker__error', role: 'alert' });
  const list = h('div', {
    class: 'picker__list',
    id: ids.list,
    role: 'listbox',
    'aria-multiselectable': multi ? 'true' : null,
    tabindex: '0',
    'aria-label': fileMode ? 'Folders and files' : 'Folders',
    onkeydown: onListKey,
  });
  const statusLine = h('p', { class: 'picker__status', id: ids.status, role: 'status' });
  const moreButton = h('button', { class: 'btn btn--sm', type: 'button', hidden: true, onclick: () => listing && load(listing.path, { append: true }) }, 'Load more');
  const chosenLine = h('p', { class: 'picker__chosen', id: ids.chosen, role: multi ? 'status' : null });
  let confirmButton;
  // Several files: two actions on the set, between the name filter and the list.
  const selectAllButton = multi ? h('button', { class: 'btn btn--sm btn--quiet picker__multibtn', type: 'button', onclick: () => selectAllShown() }, 'Select all shown') : null;
  const clearButton = multi ? h('button', { class: 'btn btn--sm btn--quiet picker__multibtn', type: 'button', onclick: () => clearSelection() }, 'Clear selection') : null;
  // The range gesture is not discoverable by itself; a phone has no Shift key, so it is not named there.
  const rangeHint = multi && !phone ? h('span', { class: 'picker__multihint' }, 'Shift+click ticks a range.') : null;
  const multiTools = multi ? h('div', { class: 'picker__multitools', role: 'group', 'aria-label': 'Selection' }, selectAllButton, clearButton, rangeHint) : null;

  const pathHelp = phone
    ? 'Paste or type a Windows or WSL path, then tap Go.'
    : 'Type or paste a path and press Enter. Windows paths such as C:\\data are accepted; relative paths start in the trainer folder.';

  const body = h(
    'div',
    { class: 'picker' },
    bar,
    h(
      'div',
      { class: 'picker__goto' },
      h(
        'div',
        { class: 'field picker__pathfield' },
        h('label', { class: 'field__label', for: ids.path }, fileMode ? 'Go to a folder or file' : 'Go to a folder'),
        goButton ? h('div', { class: 'picker__pathrow' }, pathInput, goButton) : pathInput,
        h('p', { class: 'field__help', id: ids.pathHelp }, pathHelp),
      ),
      shortcuts,
    ),
    h(
      'div',
      { class: 'picker__tools' },
      h('div', { class: 'field picker__filterfield' }, h('label', { class: 'field__label', for: ids.filter }, filterLabel), filterInput),
      showFiles?.el,
      hidden.el,
    ),
    multiTools,
    noteBox,
    errorBox,
    list,
    h('div', { class: 'picker__more' }, statusLine, moreButton),
  );
  noteBox.hidden = true;
  errorBox.hidden = true;

  const dialogHandle = openDialog({
    title: title || (fileMode ? 'Choose a file' : 'Choose a folder'),
    wide: true,
    body,
    foot: (close) => {
      // The button is described by the line that names what it will return.
      confirmButton = h('button', { class: 'btn btn--primary', type: 'button', disabled: true, 'aria-describedby': ids.chosen, onclick: () => confirm(close) }, confirmText);
      return [chosenLine, h('button', { class: 'btn', type: 'button', onclick: () => close(null) }, 'Cancel'), confirmButton];
    },
  });
  // The hook css/path-picker.css lays the phone sheet out by.
  dialogHandle.dialog.classList.add('picker__dialog');
  if (phone) {
    // Safari on a phone shows a pressed state (:active) only where something listens for touches.
    dialogHandle.dialog.addEventListener('touchstart', () => {}, { passive: true });
    // While one of the two fields is typed in, a touch on anything else must not take the focus in the
    // middle of the tap. The phone edition gives a sheet its full height back the moment focus is no
    // longer in a field (js/mobile.js), and the list is focusable: the sheet would move between the
    // finger going down and coming up, and the tap would land on something else or on nothing. So the
    // field keeps the focus through the tap, and what was tapped puts the keyboard away (dropKeyboard).
    dialogHandle.dialog.addEventListener('mousedown', (event) => {
      const field = document.activeElement;
      if (field !== pathInput && field !== filterInput) return;
      if (event.target instanceof Element && event.target.closest('input, textarea, select')) return;
      event.preventDefault();
    });
  }

  /** Phone: typing is over (a folder is opening, a file was chosen): the keyboard gives the list its room back. */
  function dropKeyboard() {
    const field = document.activeElement;
    if (phone && (field === pathInput || field === filterInput)) field.blur();
  }

  // ── what Confirm would return ──
  function chosenPath() {
    if (!listing) return null;
    if (fileMode) return active?.kind === 'file' && selectable(active) ? joinPath(listing.path, active.name) : null;
    return active?.kind === 'folder' ? joinPath(listing.path, active.name) : listing.path;
  }

  /** Several files: what the foot says, and the button that hands the set back. */
  function renderPicked() {
    const n = picked.size;
    if (confirmButton) {
      confirmButton.disabled = !n || busy;
      const label = labelFor(n);
      if (confirmButton.textContent !== label) confirmButton.textContent = label;
    }
    const here = listing ? [...picked].filter((p) => folderOfPath(p) === listing.path.replace(/\/+$/, '') || (listing.path === '/' && folderOfPath(p) === '/')).length : n;
    const elsewhere = n - here;
    const text = !n ? 'Tick the files to use.' : `${fmt.plural(n, 'file')} selected${elsewhere ? `, ${fmt.int(elsewhere)} of them in other folders` : ''}`;
    fill(chosenLine, h('span', { class: n ? 'picker__chosenlabel picker__picked' : 'muted' }, text));
    // The two actions say when they would do nothing (aria-disabled keeps them in the tab order with their reason).
    const shown = shownPickable();
    const allTicked = shown.length > 0 && shown.every((entry) => picked.has(pathOf(entry)));
    setAction(selectAllButton, shown.length && !allTicked ? '' : shown.length ? 'Every file shown is ticked.' : 'No file shown here can be ticked.');
    setAction(clearButton, n ? '' : 'No file is ticked.');
  }

  function setAction(button, reason) {
    if (!button) return;
    if (reason) {
      button.setAttribute('aria-disabled', 'true');
      button.title = reason;
    } else {
      button.removeAttribute('aria-disabled');
      button.removeAttribute('title');
    }
  }

  function renderChosen() {
    if (multi) {
      renderPicked();
      return;
    }
    const path = chosenPath();
    if (confirmButton) confirmButton.disabled = !path || busy;
    if (phone) {
      // In full above the two buttons: which kind of thing the button hands back, and which one.
      const hint = fileMode ? 'No file chosen yet. Tap a file to choose it; tap a folder to open it.' : '';
      fill(chosenLine, path ? [h('span', { class: 'picker__chosenlabel' }, fileMode ? 'This file' : 'This folder'), h('span', { class: 'picker__chosenpath' }, path)] : h('span', { class: 'picker__chosenhint' }, hint));
      return;
    }
    fill(chosenLine, path ? [h('span', { class: 'picker__chosenlabel' }, fileMode ? 'File ' : 'Folder '), h('span', { class: 'mono', title: path }, path)] : h('span', { class: 'muted' }, fileMode ? 'Choose a file in the list.' : ''));
  }

  function confirm(close) {
    if (multi) {
      if (picked.size) close(naturalSort(picked));
      return;
    }
    const path = chosenPath();
    if (path) close(path);
  }

  // ── requests ──
  function begin() {
    controller?.abort();
    controller = new AbortController();
    return controller;
  }

  function listParams(path, { after, filter = filterInput.value } = {}) {
    const wantFiles = fileMode || showFiles?.input.checked;
    return {
      path: path === '' ? undefined : path,
      mode: wantFiles ? 'all' : 'folders',
      filter: filter || undefined,
      hidden: hidden.input.checked,
      limit: PAGE,
      after: after ?? undefined,
    };
  }

  function setBusy(value) {
    busy = value;
    list.setAttribute('aria-busy', String(value));
    if (value) statusLine.textContent = 'Loading…';
    else if (openingRow) {
      // The folder did not open (or the request was replaced): the row stops saying it is opening.
      openingRow.classList.remove('picker__row--opening');
      openingRow = null;
    }
    renderChosen();
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = !message;
    // Phone: the message may be above or below what the keyboard leaves in view.
    if (phone && message) errorBox.scrollIntoView({ block: 'nearest' });
  }

  function showNote(message) {
    noteBox.hidden = !message;
    fill(noteBox, message || null);
  }

  /** Reload or page the folder on screen with the current filter and switches. */
  async function load(path, { append = false, keepActive = false } = {}) {
    if (append && !listing?.next) return; // nothing more to fetch; never re-append the first page
    const ctrl = begin();
    setBusy(true);
    try {
      const result = await api.fsList(listParams(path, { after: append ? listing?.next : undefined }), { signal: ctrl.signal });
      if (ctrl !== controller || closed) return;
      showError('');
      if (append && listing) {
        listing = { ...result, entries: [...listing.entries, ...(result.entries ?? [])] };
        appendRows(result.entries ?? []);
        renderStatus();
      } else {
        // The same folder again (a filter, a switch): the path field is rewritten only if it still shows that folder.
        show(result, { active: keepActive ? active : null, edits: pathShown });
      }
    } catch (error) {
      if (isAbort(error) || ctrl !== controller || closed) return;
      showError(error.message);
    } finally {
      if (ctrl === controller && !closed) {
        setBusy(false);
        renderStatus();
      }
    }
  }

  /**
   * Open a path the person typed or chose. On "not found": in file mode, a file is opened as its folder
   * with the file highlighted; with `walk` (the start path) the nearest existing folder above is shown.
   * `back` (phone): the folder is one Back returned to, so the folder left is not remembered.
   */
  async function openPath(target, { walk, back = false }) {
    // What the path field holds now is what this opening replaces (or sends); an edit made after this
    // point is newer than the answer and is kept.
    const edits = pathEdits;
    const ctrl = begin();
    setBusy(true);
    filterInput.value = '';
    reloadFiltered.cancel();
    let path = target;
    let missing = null;
    try {
      for (let step = 0; step < 64; step += 1) {
        let result;
        try {
          result = await api.fsList(listParams(path, { filter: '' }), { signal: ctrl.signal });
        } catch (error) {
          if (isAbort(error) || ctrl !== controller || closed) return;
          if (error.status !== 404) {
            showError(error.message);
            return;
          }
          if (missing === null) missing = error.message;
          if (!walk && !(fileMode && step === 0)) {
            showError(missing);
            return;
          }
          const up = parentOf(path);
          if (up === null) {
            showError(missing);
            return;
          }
          path = up;
          continue;
        }
        if (ctrl !== controller || closed) return;
        let highlight = null;
        let note = '';
        if (missing !== null) {
          const name = baseName(target);
          const isFile = fileMode && step === 1 && name && (await fileExists(result.path, name, ctrl));
          if (ctrl !== controller || closed) return;
          if (isFile) highlight = { kind: 'file', name };
          else if (!walk) {
            showError(missing);
            return;
          } else note = `${target} was not found. Showing ${result.path}, the nearest folder above it that exists.`;
        }
        showError('');
        showNote(note);
        show(result, { active: highlight, back, edits });
        return;
      }
      showError(missing ?? 'The folder could not be opened.');
    } finally {
      if (ctrl === controller && !closed) {
        setBusy(false);
        renderStatus();
      }
    }
  }

  async function fileExists(folder, name, ctrl) {
    try {
      const result = await api.fsList({ path: folder, mode: 'files', filter: name, hidden: name.startsWith('.'), limit: 1024 }, { signal: ctrl.signal });
      return (result.entries ?? []).some((entry) => entry.kind === 'file' && entry.name === name);
    } catch {
      return false;
    }
  }

  /** Open what the path field holds (Enter, or Go on the phone). */
  function submitPath() {
    // Phone: a pasted path may arrive in quotes ("Copy as path"). The desktop page sends what was typed, trimmed.
    const value = phone ? cleanTyped(pathInput.value) : pathInput.value.trim();
    if (!value) return;
    const edits = pathEdits;
    openPath(value, { walk: false }).then(() => {
      // Phone: the keyboard covers the list, so it is put away once the folder is on screen. After a
      // refusal the field keeps the keyboard, with the message under it, for the path to be corrected;
      // so does a field that was edited again while the answer was on its way.
      if (phone && !closed && errorBox.hidden && edits === pathEdits && document.activeElement === pathInput) pathInput.blur();
    });
  }

  /** Open a folder named by a button: a place, or a part of the path so far. */
  function goTo(path) {
    dropKeyboard();
    openPath(path, { walk: false });
  }

  function goUp() {
    if (listing?.parent) goTo(listing.parent);
  }

  /** Phone: return to the folder shown before this one (or to the nearest folder above it that still exists). */
  function goBack() {
    const previous = trail.pop();
    if (backButton) backButton.disabled = !trail.length;
    if (previous === undefined) return;
    dropKeyboard();
    openPath(previous, { walk: true, back: true });
  }

  // ── rendering ──
  /**
   * Put a listing on screen. `edits` (phone): the count of path-field edits this listing answers for. If
   * the field was edited since, the listing is shown but the field keeps what the person put there.
   */
  function show(result, { active: highlight = null, back = false, edits = pathEdits } = {}) {
    const moved = Boolean(listing) && listing.path !== result.path;
    if (phone && moved && !back) {
      trail.push(listing.path);
      if (trail.length > TRAIL) trail.shift();
    }
    listing = { path: result.path, parent: result.parent ?? null, entries: Array.isArray(result.entries) ? result.entries : [], next: result.next ?? null };
    const editing = phone && edits !== pathEdits;
    if (!editing) {
      pathInput.value = listing.path;
      pathShown = pathEdits;
    }
    upButton.disabled = !listing.parent;
    if (backButton) backButton.disabled = !trail.length;
    fill(
      crumbBar,
      crumbs(listing.path).map((crumb, index, all) => [
        index > 1 ? h('span', { class: 'picker__sep', 'aria-hidden': 'true' }, '/') : null,
        index === all.length - 1
          ? h('span', { class: 'picker__crumb--here', 'aria-current': 'location' }, crumb.label)
          : h('button', { class: 'picker__crumb', type: 'button', 'aria-label': crumb.path === '/' ? 'Root (/)' : crumb.path, onclick: () => goTo(crumb.path) }, crumb.label),
      ]),
    );
    rowByKey = new Map();
    active = null;
    openingRow = null;
    anchor = null;
    fill(list);
    appendRows(listing.entries);
    if (phone) {
      // The path so far is one line that scrolls sideways: its end, the folder on screen, is what shows.
      crumbBar.scrollLeft = crumbBar.scrollWidth;
      shownAt = performance.now();
      if (moved) listToTop();
    }
    const listed = highlight && listing.entries.some((e) => e.kind === highlight.kind && e.name === highlight.name);
    // A file checked by fileExists() stays chosen even when it is beyond the first page. (While the path
    // field is being edited, the sheet is not moved to the chosen row: the field stays where it is.)
    setActive(listed || (fileMode && highlight?.kind === 'file') ? highlight : null, !editing);
    renderStatus();
  }

  /**
   * Phone: the sheet's body scrolls as one, with the bar held at its top. A folder opened from deep in
   * a list starts at its first row; the fields above the list stay out of the way if they already were.
   */
  function listToTop() {
    const scroller = body.parentElement;
    if (!scroller) return;
    const over = scroller.getBoundingClientRect().top + bar.offsetHeight - list.getBoundingClientRect().top;
    if (over > 0) scroller.scrollTop -= over;
  }

  const keyOf = (entry) => `${entry.kind}\u0000${entry.name}`;

  function selectable(entry) {
    // Several files: a file the extensions leave out cannot be ticked.
    if (multi && entry.kind === 'file') return matchesExtensions(entry.name, extensions);
    return entry.kind === 'folder' || fileMode;
  }

  // ── several files ──
  const pathOf = (entry) => joinPath(listing?.path ?? '', entry.name);
  const pickable = (entry) => multi && entry?.kind === 'file' && selectable(entry);
  const shownPickable = () => (listing?.entries ?? []).filter(pickable);

  function paintPicked(entry) {
    const row = rowByKey.get(keyOf(entry));
    if (!row) return;
    const on = picked.has(pathOf(entry));
    row.setAttribute('aria-selected', String(on));
    row.classList.toggle('picker__row--picked', on);
  }

  /** Tick or untick one file; `range` (Shift) ticks every file from the anchor to it. */
  function toggleEntry(entry, { range = false } = {}) {
    if (!pickable(entry)) return;
    const entries = listing?.entries ?? [];
    const from = range && anchor ? entries.findIndex((e) => keyOf(e) === anchor) : -1;
    const to = entries.findIndex((e) => keyOf(e) === keyOf(entry));
    if (from >= 0 && to >= 0) {
      for (let i = Math.min(from, to); i <= Math.max(from, to); i += 1) {
        if (!pickable(entries[i])) continue;
        picked.add(pathOf(entries[i]));
        paintPicked(entries[i]);
      }
    } else {
      const path = pathOf(entry);
      if (picked.has(path)) picked.delete(path);
      else picked.add(path);
      paintPicked(entry);
    }
    anchor = keyOf(entry);
    renderPicked();
  }

  function selectAllShown() {
    if (selectAllButton?.getAttribute('aria-disabled') === 'true') return;
    for (const entry of shownPickable()) {
      picked.add(pathOf(entry));
      paintPicked(entry);
    }
    renderPicked();
  }

  function clearSelection() {
    if (clearButton?.getAttribute('aria-disabled') === 'true') return;
    picked.clear();
    for (const entry of listing?.entries ?? []) if (pickable(entry)) paintPicked(entry);
    anchor = null;
    renderPicked();
  }

  /** Desktop, several files: a click ticks a file (Shift: the range) and highlights a folder, which a double click opens. */
  function clickEntry(entry, event) {
    if (!selectable(entry)) return;
    setActive(entry, false);
    if (entry.kind === 'file') toggleEntry(entry, { range: Boolean(event?.shiftKey) });
  }

  function appendRows(entries) {
    const frag = document.createDocumentFragment();
    for (const entry of entries) {
      const dim = entry.kind === 'file' && (!selectable(entry) || !matchesExtensions(entry.name, extensions));
      const tick = pickable(entry);
      const row = h(
        'div',
        {
          class: `picker__row picker__row--${entry.kind}${dim ? ' picker__row--dim' : ''}${multi ? ' picker__row--multi' : ''}`,
          id: uid('picker-row'),
          role: 'option',
          'aria-selected': multi && tick && picked.has(pathOf(entry)) ? 'true' : 'false',
          'aria-disabled': selectable(entry) ? null : 'true',
          onclick: phone ? () => tapEntry(entry, row) : multi ? (event) => clickEntry(entry, event) : () => setActive(entry),
          ondblclick: phone ? null : multi ? () => entry.kind === 'folder' && openEntry(entry) : () => openEntry(entry),
        },
        // Several files: a drawn box at the start of every row (empty for a row that cannot be ticked).
        multi ? h('span', { class: tick ? 'picker__check' : 'picker__check picker__check--none', 'aria-hidden': 'true' }) : null,
        h('span', { class: `picker__kind picker__kind--${entry.kind}` }, entry.kind === 'folder' ? 'Folder' : 'File'),
        h('span', { class: 'picker__name' }, entry.name),
        // Phone: what a tap does, at the end of the row. A folder opens (an arrow); a file is chosen (a ring).
        // Several files: the box says it for a file, so only a folder keeps its arrow.
        phone && selectable(entry) && !(multi && entry.kind === 'file') ? h('span', { class: 'picker__mark', 'aria-hidden': 'true' }) : phone && multi ? h('span', { 'aria-hidden': 'true' }) : null,
      );
      if (multi && tick && picked.has(pathOf(entry))) row.classList.add('picker__row--picked');
      rowByKey.set(keyOf(entry), row);
      frag.appendChild(row);
    }
    list.appendChild(frag);
    if (multi) renderPicked();
  }

  function setActive(entry, scroll = true) {
    const old = active ? rowByKey.get(keyOf(active)) : null;
    // Several files: aria-selected says ticked; the row with the keyboard is only highlighted.
    if (old) {
      if (!multi) old.setAttribute('aria-selected', 'false');
      old.classList.remove('picker__row--active');
    }
    active = entry ? { kind: entry.kind, name: entry.name } : null;
    const now = active ? rowByKey.get(keyOf(active)) : null;
    if (now) {
      if (!multi) now.setAttribute('aria-selected', 'true');
      now.classList.add('picker__row--active');
      list.setAttribute('aria-activedescendant', now.id);
      if (scroll) now.scrollIntoView({ block: 'nearest' });
    } else {
      list.removeAttribute('aria-activedescendant');
    }
    renderChosen();
  }

  function openEntry(entry) {
    if (!listing || !entry) return;
    if (entry.kind === 'folder') openPath(joinPath(listing.path, entry.name), { walk: false });
    // Several files: Enter on a file ticks it; only the button at the foot hands the set back.
    else if (multi) toggleEntry(entry);
    else if (fileMode && selectable(entry)) dialogHandle.close(joinPath(listing.path, entry.name));
  }

  /**
   * Phone: one tap. A folder opens at once; a file is chosen, and the button at the foot confirms it
   * (a tap never closes the sheet). A file in a folder picker is listed for orientation only.
   */
  function tapEntry(entry, row) {
    if (!listing || !selectable(entry)) return;
    // The second tap of a double tap would land on whatever the opened folder lists in that spot.
    if (performance.now() - shownAt < TAP_GUARD) return;
    dropKeyboard();
    if (entry.kind === 'folder') {
      openingRow?.classList.remove('picker__row--opening');
      openPath(joinPath(listing.path, entry.name), { walk: false });
      openingRow = row;
      row.classList.add('picker__row--opening');
    } else if (multi) {
      setActive(entry, false);
      toggleEntry(entry);
    } else {
      setActive(entry, false);
    }
  }

  function renderStatus() {
    if (!listing) {
      if (!busy) statusLine.textContent = '';
      moreButton.hidden = true;
      return;
    }
    const n = listing.entries.length;
    const what = fileMode || showFiles?.input.checked ? 'entry' : 'folder';
    const many = what === 'entry' ? 'entries' : 'folders';
    let text;
    if (!n) text = filterInput.value ? `No ${many} contain “${filterInput.value}”.` : `This folder has no ${many}${hidden.input.checked ? '' : ' that are not hidden'}.`;
    else text = `${fmt.plural(n, what, many)} shown${listing.next ? '. This folder has more.' : '.'}`;
    if (busy) text = 'Loading…';
    statusLine.textContent = text;
    moreButton.hidden = !listing.next;
    moreButton.disabled = busy;
  }

  // ── keyboard ──
  function onListKey(event) {
    const entries = listing?.entries ?? [];
    const index = active ? entries.findIndex((e) => e.kind === active.kind && e.name === active.name) : -1;
    const last = entries.length - 1;
    const page = 10;
    const target = {
      ArrowDown: index + 1,
      ArrowUp: index < 0 ? 0 : index - 1,
      Home: 0,
      End: last,
      PageDown: index + page,
      PageUp: index - page,
    }[event.key];
    if (multi && event.key === ' ') {
      // Space ticks or unticks the highlighted file.
      event.preventDefault();
      if (index >= 0) toggleEntry(entries[index]);
      return;
    }
    if (multi && (event.ctrlKey || event.metaKey) && !event.altKey && event.key.toLowerCase() === 'a') {
      event.preventDefault();
      selectAllShown();
      return;
    }
    if (target !== undefined) {
      event.preventDefault();
      if (!entries.length) return;
      const next = entries[Math.max(0, Math.min(last, target))];
      // Several files: Shift with an arrow ticks from the row ticked last (or the one left) to the new one.
      if (multi && event.shiftKey) {
        const from = index >= 0 ? entries[index] : null;
        if (!anchor && pickable(from)) anchor = keyOf(from);
        setActive(next);
        toggleEntry(next, { range: true });
        return;
      }
      setActive(next);
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      if (index >= 0) openEntry(entries[index]);
      return;
    }
    if (event.key === 'Backspace') {
      event.preventDefault();
      goUp();
      return;
    }
    // Type-ahead: jump to the next entry whose name starts with the letters typed.
    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey && event.key !== ' ') {
      clearTimeout(typedTimer);
      typed += event.key;
      typedTimer = setTimeout(() => {
        typed = '';
      }, 700);
      const from = typed.length === 1 ? index + 1 : Math.max(0, index);
      for (let k = 0; k < entries.length; k += 1) {
        const entry = entries[(from + k) % entries.length];
        if (entry.name.startsWith(typed)) {
          setActive(entry);
          break;
        }
      }
    }
  }

  // ── open ──
  renderChosen();
  openPath(String(start ?? '').trim() || (typeof root === 'string' && root ? root : ''), { walk: true }).then(() => {
    if (closed) return;
    // Phone, by finger: the rows answer taps without the list being focused, and a focus ring round a
    // list as long as its folder would only be noise. A keyboard attached to the phone reaches the list
    // with Tab, as it reaches everything else.
    if (phone && window.matchMedia?.('(pointer: coarse)').matches) return;
    const focus = document.activeElement;
    // (Phone: the list is as long as its folder, and focusing it must not move the sheet to it.)
    if (!focus || focus === document.body || (dialogHandle.dialog.contains(focus) && focus.closest('.dialog__head'))) list.focus({ preventScroll: phone });
  });

  return dialogHandle.closed.then((value) => {
    closed = true;
    controller?.abort();
    reloadFiltered.cancel();
    clearTimeout(typedTimer);
    if (multi) return Array.isArray(value) && value.length ? value : null;
    return typeof value === 'string' && value ? value : null;
  });
}
