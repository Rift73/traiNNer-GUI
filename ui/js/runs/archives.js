// "Experiments folders" block under the run list: the trainer's own experiments folder, then each folder
// added with POST /api/archives (read-only; its runs join the list). Read when the screen mounts and after
// an add or a forget; it does not poll.

import { h, mount as mountNodes, fmt } from '../dom.js';
import { api } from '../api.js';
import { conn } from '../state/conn.js';
import { runs } from '../state/runs.js';
import { persist } from '../store.js';
import { toast, confirmDialog, notice, loading } from '../ui.js';

/** The backend may list a new folder's runs one call later: read once more after this delay. */
const SECOND_READ_MS = 2000;
/** persist key of the disclosure's open state */
const OPEN_KEY = 'runs-archives-open';

/** `status` of an added folder in words. */
export function archiveStatusText(archive) {
  if (archive?.status === 'checking') return 'Being read';
  if (archive?.status === 'available') return Number.isFinite(archive.runCount) ? fmt.plural(archive.runCount, 'run') : 'Available';
  if (archive?.status === 'unavailable') return archive.error ? `Unavailable: ${archive.error}` : 'Unavailable';
  return archive?.status ? String(archive.status) : 'Not reported';
}

/**
 * @param {HTMLElement} host
 * @returns {{refresh(): Promise<void>, destroy(): void}}
 */
export function mountArchives(host) {
  let destroyed = false;
  let controller = null;
  let timer = 0;
  let busy = false;
  let list = null; // last answer of GET /api/archives
  let error = '';

  const status = h('div', { class: 'runs__archives-status', role: 'status', 'aria-live': 'polite' });
  const items = h('ul', { class: 'runs__archives-list' });
  const addBtn = h('button', { class: 'btn btn--sm', type: 'button', onclick: add }, 'Add an experiments folder');
  const countEl = h('span', { class: 'runs__archives-count muted num' });
  // A disclosure, closed unless the person left it open: one line when the folders are not in use.
  const root = h(
    'details',
    { class: 'runs__archives', open: Boolean(persist.get(OPEN_KEY, false)) },
    h('summary', { class: 'runs__archives-summary' }, h('span', { class: 'runs__archives-title' }, 'Experiments folders'), countEl),
    h('div', { class: 'runs__archives-body' }, items, status, h('div', { class: 'runs__archives-foot' }, addBtn)),
  );
  const onToggle = () => persist.set(OPEN_KEY, root.open);
  root.addEventListener('toggle', onToggle);
  host.appendChild(root);
  const unsubConn = conn.subscribe(() => render());
  render();
  refresh();

  async function refresh({ force = false } = {}) {
    controller?.abort();
    const ctrl = new AbortController();
    controller = ctrl;
    if (list === null) mountNodes(status, loading('Reading the added folders…'));
    try {
      const answer = await api.archives({ signal: ctrl.signal });
      if (destroyed || ctrl.signal.aborted) return;
      list = Array.isArray(answer) ? answer.filter((a) => a && typeof a === 'object') : [];
      error = '';
    } catch (e) {
      if (e?.name === 'AbortError' || destroyed) return;
      error = e.message;
    }
    render(force);
  }

  function secondRead() {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (destroyed) return;
      refresh();
      runs.refresh();
    }, SECOND_READ_MS);
  }

  /** @param {boolean} [force] rebuild even when focus is inside (after the person's own add or forget) */
  function render(force = false) {
    if (destroyed) return;
    const trainerRoot = conn.get().info?.trainerRoot;
    const own = typeof trainerRoot === 'string' && trainerRoot ? `${trainerRoot.replace(/\/+$/, '')}/experiments` : null;
    const rows = [
      h(
        'li',
        { class: 'runs__archive' },
        h('span', { class: 'runs__archive-path mono truncate', title: own ?? '' }, own ?? "The trainer's experiments folder"),
        h('span', { class: 'runs__archive-state muted' }, "The trainer's own folder"),
      ),
    ];
    for (const a of list ?? []) {
      const forget = h('button', { class: 'btn btn--sm btn--quiet', type: 'button', onclick: () => remove(a) }, 'Forget folder');
      forget.setAttribute('aria-label', `Forget folder ${a.path ?? a.id}`);
      rows.push(
        h(
          'li',
          { class: 'runs__archive' },
          h('span', { class: 'runs__archive-path mono truncate', title: a.path ?? '' }, a.path ?? String(a.id ?? '')),
          h('span', { class: a.status === 'unavailable' ? 'runs__archive-state runs__archive-state--err' : 'runs__archive-state muted' }, archiveStatusText(a)),
          forget,
        ),
      );
    }
    // Rebuilt only when nothing in the list holds focus (a background re-read must not drop it).
    const hadFocus = items.contains(document.activeElement);
    if (force || !hadFocus) mountNodes(items, rows);
    // The focused button went with its row: keep focus in the block instead of losing it to the page.
    if (force && hadFocus) addBtn.focus();
    const count = list === null ? '' : list.length ? `${fmt.int(list.length)} added` : 'none added';
    if (countEl.textContent !== count) countEl.textContent = count;
    if (error) mountNodes(status, notice('err', `The added experiments folders could not be read: ${error}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => refresh() }, 'Try again')));
    else if (list !== null) status.replaceChildren();
    addBtn.disabled = busy;
  }

  async function add() {
    let pickPath;
    try {
      ({ pickPath } = await import('../components/path-picker.js'));
    } catch (e) {
      toast(`The folder picker could not be opened: ${e.message}`, { kind: 'err' });
      return;
    }
    const path = await pickPath({ title: 'Choose a folder that holds experiment folders', mode: 'folder', confirmLabel: 'Add this folder' });
    if (!path || destroyed) return;
    busy = true;
    render();
    try {
      const added = await api.addArchive(path);
      const count = added?.status === 'available' && Number.isFinite(added.runCount) ? `: ${fmt.plural(added.runCount, 'run')} found` : '';
      toast(`Added the experiments folder ${added?.path ?? path}${count}.`, { kind: 'ok' });
      runs.refresh();
      await refresh();
      secondRead();
    } catch (e) {
      toast(`The experiments folder was not added: ${e.message}`, { kind: 'err' });
    } finally {
      busy = false;
      render();
    }
  }

  async function remove(archive) {
    const path = archive.path ?? String(archive.id);
    const ok = await confirmDialog({
      title: 'Forget this experiments folder?',
      body: h(
        'div',
        { class: 'stack' },
        h('p', { class: 'mono runs__archive-confirm' }, path),
        h('p', null, 'The folder and its files are not touched. Only its runs leave the run list; adding the folder again brings them back.'),
      ),
      confirmLabel: 'Forget folder',
      danger: true,
    });
    if (!ok || destroyed) return;
    try {
      const answer = await api.removeArchive(archive.id);
      toast(answer?.removed === false ? `The folder ${path} was no longer added.` : `Forgot the experiments folder ${path}.`, { kind: 'ok' });
      runs.refresh();
      await refresh({ force: true });
    } catch (e) {
      toast(`The folder was not forgotten: ${e.message}`, { kind: 'err' });
    }
  }

  return {
    refresh,
    destroy() {
      destroyed = true;
      controller?.abort();
      clearTimeout(timer);
      unsubConn();
      root.removeEventListener('toggle', onToggle);
      root.remove();
    },
  };
}
