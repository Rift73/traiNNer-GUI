// Recipe screen frame: the bar (what is open, its check result, actions) and the two working
// panes. The panes themselves live in ../recipe/: form.js (options by section),
// source-pane.js (YAML, changes, compare, issues) and library.js (choosing a recipe).

import { h, mount as setChildren, clamp, fmt, uid } from '../dom.js';
import { persist } from '../store.js';
import { api } from '../api.js';
import { recipe } from '../state/recipe.js';
import { catalog } from '../state/catalog.js';
import { NO_SCALE_TASKS } from '../state/data.js';
import { badge, confirmDialog, countIssues, notice, toast } from '../ui.js';
import { addToQueue, saveRecipe, startTraining } from '../actions.js';
import { resumeFacts } from '../recipe/resume.js';

const LAYOUTS = [
  { id: 'form', label: 'Options' },
  { id: 'split', label: 'Both' },
  { id: 'source', label: 'YAML' },
];

async function loadPart(importer, exportName) {
  try {
    const module = await importer();
    if (typeof module[exportName] !== 'function') throw new Error(`${exportName} is missing`);
    return { module, error: null };
  } catch (error) {
    console.error(error);
    return { module: null, error };
  }
}

/** Load a picked recipe, asking first when that would throw away unsaved work. */
async function loadPicked(picked) {
  if (!picked || !picked.doc || typeof picked.doc.yaml !== 'string') return;
  if (recipe.get().phase === 'open' && recipe.isDirty() && recipe.hasChanges()) {
    const ok = await confirmDialog({
      title: 'Replace the open recipe?',
      body: 'The open recipe has changes that are not saved. Opening another recipe discards them.',
      confirmLabel: 'Discard changes and open',
      danger: true,
    });
    if (!ok) return;
  }
  recipe.load(picked.doc, { label: picked.label, saved: picked.saved !== false });
}

export function mount(el) {
  const root = h('section', { class: 'recipe', 'aria-label': 'Recipe' });
  el.appendChild(root);

  let disposed = false;
  let phase = null;
  let partCleanups = [];
  let barEl = null;
  let library = null;
  let listsRevision = catalog.get().revision;
  let editorCalls = 0;
  const titleId = uid('recipe-title');

  const disposeParts = () => {
    for (const cleanup of partCleanups) {
      try {
        if (typeof cleanup === 'function') cleanup();
      } catch (error) {
        console.error(error);
      }
    }
    partCleanups = [];
  };

  // ───────── nothing open: choose a recipe ─────────

  async function showLibrary() {
    disposeParts();
    barEl = null;
    setChildren(root);
    const host = h('div', { class: 'recipe__library' });
    root.appendChild(host);
    const part = await loadPart(() => import('../recipe/library.js'), 'mountLibrary');
    if (disposed || phase !== 'empty') return;
    if (part.error) {
      setChildren(host, h('div', { style: { padding: '16px' } }, notice('err', `The recipe chooser failed to load: ${part.error.message}`)));
      return;
    }
    library = part.module;
    partCleanups.push(part.module.mountLibrary(host, { onPick: loadPicked }));
  }

  // ───────── a recipe is open ─────────

  async function showEditor() {
    // A later call (the lists changed while the parts were still loading) replaces this one.
    const mine = ++editorCalls;
    disposeParts();
    setChildren(root);

    // The bar takes focus when a recipe was just chosen (see sync). Its children are rebuilt as the facts
    // change, the bar itself is not, so focus placed on it stays put.
    barEl = h('header', { class: 'recipe__bar', tabindex: '-1', role: 'group', 'aria-labelledby': titleId });
    const formHost = h('div', { class: 'recipe__form' });
    const sourceHost = h('div', { class: 'recipe__source' });
    const divider = h('div', {
      class: 'recipe__divider',
      role: 'separator',
      tabindex: '0',
      'aria-orientation': 'vertical',
      'aria-label': 'Resize the YAML pane',
      title: 'Drag to resize. Arrow keys also work.',
    });
    const body = h('div', { class: 'recipe__body', dataset: { layout: persist.get('recipe-layout', 'split') } }, formHost, divider, sourceHost);
    root.append(barEl, body);

    // divider: drag or arrow keys set the YAML pane width
    const setWidth = (px) => {
      const width = clamp(Math.round(px), 300, Math.max(300, body.clientWidth - 360));
      body.style.setProperty('--source-w', `${width}px`);
      persist.set('recipe-source-w', width);
    };
    const savedWidth = persist.get('recipe-source-w', null);
    if (Number.isFinite(savedWidth)) body.style.setProperty('--source-w', `${savedWidth}px`);
    divider.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      divider.setPointerCapture(event.pointerId);
      const move = (ev) => setWidth(body.getBoundingClientRect().right - ev.clientX);
      const up = () => {
        divider.removeEventListener('pointermove', move);
        divider.removeEventListener('pointerup', up);
        divider.removeEventListener('pointercancel', up);
      };
      divider.addEventListener('pointermove', move);
      divider.addEventListener('pointerup', up);
      divider.addEventListener('pointercancel', up);
    });
    divider.addEventListener('keydown', (event) => {
      const step = event.key === 'ArrowLeft' ? 24 : event.key === 'ArrowRight' ? -24 : 0;
      if (!step) return;
      event.preventDefault();
      setWidth(sourceHost.getBoundingClientRect().width + step);
    });

    renderBar(body);

    const [form, source, lib] = await Promise.all([
      loadPart(() => import('../recipe/form.js'), 'mountForm'),
      loadPart(() => import('../recipe/source-pane.js'), 'mountSourcePane'),
      loadPart(() => import('../recipe/library.js'), 'openLibraryDialog'),
    ]);
    if (disposed || phase !== 'open' || mine !== editorCalls) return;
    library = lib.module;
    if (form.error) setChildren(formHost, h('div', { style: { padding: '16px' } }, notice('err', `The options form failed to load: ${form.error.message}`)));
    else partCleanups.push(form.module.mountForm(formHost));
    if (source.error) setChildren(sourceHost, h('div', { style: { padding: '16px' } }, notice('err', `The YAML pane failed to load: ${source.error.message}`)));
    else partCleanups.push(source.module.mountSourcePane(sourceHost));
  }

  async function detachCheckpoint() {
    const before = recipe.get().yaml;
    try {
      const fresh = await api.freshRecipe(before);
      if (disposed || recipe.get().yaml !== before || recipe.get().pending) {
        toast('The recipe changed. Choose Make fresh recipe again to use the latest edits.');
        return;
      }
      recipe.setYaml(fresh.yaml, { immediate: true });
      toast('Recipe ready for a new run from iteration 0. Your other edits are preserved.', { kind: 'ok' });
    } catch (error) {
      toast(`Could not detach the checkpoint: ${error.message}`, { kind: 'err' });
    }
  }

  function renderBar(body = root.querySelector('.recipe__body')) {
    if (!barEl) return;
    const s = recipe.get();
    const { errors, warnings } = countIssues(s.issues);
    const title = s.summary?.name || s.name || 'Untitled recipe';

    const facts = [];
    if (s.summary?.task) {
      const taskId = String(s.summary.task);
      facts.push(h('span', { class: 'recipe__fact', title: `task: ${taskId}` }, h('span', { class: 'muted' }, 'Task '), catalog.taskLabel(taskId) || taskId));
    }
    if (s.summary?.architecture) facts.push(h('span', { class: 'recipe__fact' }, h('span', { class: 'muted' }, 'Network '), String(s.summary.architecture)));
    // A scale is a fact of a recipe that upscales: classification, segmentation and detection do not show one.
    if (Number.isFinite(s.summary?.scale) && !NO_SCALE_TASKS.includes(String(s.summary?.task ?? ''))) facts.push(h('span', { class: 'recipe__fact' }, h('span', { class: 'muted' }, 'Scale '), `${s.summary.scale}×`));
    // A recipe that continues a saved training state says where it continues, once the check has accepted
    // that state (a refused one is in the issues instead). The other facts of the state are in the title
    // and in the dialog shown before training starts.
    const resume = s.ok && !errors ? s.summary?.resume : null;
    if (resume) {
      const at = Number.isInteger(resume.iteration) ? `iteration ${fmt.int(resume.iteration)}` : 'a saved training state';
      const about = resumeFacts(resume).map(([label, value]) => `${label}: ${value}`).join('\n');
      facts.push(h('span', { class: 'recipe__fact', title: about || null }, h('span', { class: 'muted' }, 'Continues from '), at));
    }

    if (s.configCurrent && (s.config?.path?.resume_state || s.config?.auto_resume)) {
      facts.push(h('button', {
        class: 'btn btn--sm', type: 'button', onclick: detachCheckpoint, disabled: Boolean(s.pending),
        title: 'Remove resume state and restore the original starting weights. Keep edits to this recipe.',
      }, 'Make fresh recipe'));
    }

    // The last verdict stays up while a re-check runs; a small spinner beside it says it is being refreshed.
    const checking = s.check !== 'done' || s.pending > 0;
    let verdict;
    if (s.check === 'failed') verdict = badge('Check could not run', 'err');
    else if (!s.hasResult) verdict = badge('Checking');
    else if (errors || !s.ok) verdict = badge(fmt.plural(errors, 'error'), 'err');
    else verdict = badge('Passed check', 'ok');
    const busy = checking && s.check !== 'failed' ? h('span', { class: 'spinner', role: 'img', 'aria-label': 'Checking', title: 'Checking' }) : null;

    const layout = body?.dataset.layout ?? 'split';
    const canStart = s.check === 'done' && s.ok && errors === 0 && s.pending === 0;

    setChildren(
      barEl,
      h(
        'div',
        { class: 'recipe__ident' },
        h('h1', { class: 'recipe__title truncate', id: titleId, title }, title),
        h(
          'div',
          { class: 'recipe__meta' },
          facts,
          s.baseLabel ? h('span', { class: 'recipe__fact truncate', title: s.path ?? '' }, h('span', { class: 'muted' }, 'From '), s.baseLabel) : null,
        ),
      ),
      h(
        'div',
        { class: 'recipe__verdict', role: 'status' },
        verdict,
        warnings && s.hasResult && s.check !== 'failed' ? badge(fmt.plural(warnings, 'warning'), 'warn') : null,
        busy,
        recipe.isDirty() ? badge(s.savedYaml === null ? 'Not saved' : 'Unsaved changes') : null,
      ),
      h(
        'div',
        { class: 'seg', role: 'group', 'aria-label': 'Show' },
        LAYOUTS.map((item) =>
          h(
            'button',
            {
              class: 'btn btn--sm',
              type: 'button',
              'aria-pressed': String(layout === item.id),
              onclick: () => {
                body.dataset.layout = item.id;
                persist.set('recipe-layout', item.id);
                renderBar(body);
              },
            },
            item.label,
          ),
        ),
      ),
      h(
        'div',
        { class: 'recipe__actions' },
        h(
          'button',
          {
            class: 'btn',
            type: 'button',
            onclick: async () => {
              if (!library?.openLibraryDialog) return;
              loadPicked(await library.openLibraryDialog({ title: 'Choose a recipe' }));
            },
          },
          'Choose another',
        ),
        h('button', { class: 'btn', type: 'button', disabled: s.pending > 0, onclick: async () => {
          const { openVramTest } = await import('../recipe/vram-test.js');
          openVramTest();
        } }, 'VRAM test'),
        h('button', { class: 'btn', type: 'button', disabled: s.saving, onclick: () => saveRecipe() }, s.saving ? 'Saving…' : 'Save recipe'),
        h(
          'button',
          {
            class: 'btn',
            type: 'button',
            title: 'Keep a copy of this recipe in the queue. It starts only after you release it.',
            onclick: () => addToQueue(),
          },
          'Add to queue',
        ),
        h(
          'button',
          {
            class: 'btn btn--primary',
            type: 'button',
            'aria-disabled': String(!canStart),
            title: canStart ? '' : 'Available once the recipe passes its check.',
            onclick: () => startTraining(),
          },
          'Start training',
        ),
      ),
    );
  }

  // The bar is rebuilt only when something it shows changes, so focus inside it survives typing elsewhere.
  let barKey = '';
  const barSignature = () => {
    const s = recipe.get();
    const { errors, warnings } = countIssues(s.issues);
    return JSON.stringify([s.summary, s.name, s.baseLabel, s.path, s.check, s.hasResult, s.ok, errors, warnings, s.pending > 0, s.saving, recipe.isDirty(), s.savedYaml === null, catalog.get().phase]);
  };

  function sync() {
    const s = recipe.get();
    if (s.phase !== phase) {
      const before = phase;
      phase = s.phase;
      barKey = '';
      if (phase === 'empty') showLibrary();
      else showEditor();
      // A recipe was just chosen in the chooser: its list is gone, and with it the focus. Go on from the
      // recipe's title bar instead of the top of the page. (The first draw of the screen moves nothing.)
      if (before === 'empty' && phase === 'open') barEl?.focus({ preventScroll: true });
    }
    // The service read its lists of networks, losses and options again: the form was built on the old ones.
    // Never rebuilt while the person works in it: the rebuild waits until the focus has left the screen
    // (see the focusout listener below).
    const revision = catalog.get().revision;
    if (revision !== listsRevision && !(phase === 'open' && root.contains(document.activeElement))) {
      listsRevision = revision;
      if (phase === 'open') showEditor();
    }
    if (phase === 'open') {
      const key = barSignature();
      if (key !== barKey) {
        barKey = key;
        renderBar();
      }
    }
  }

  const unsubscribe = recipe.subscribe(sync);
  const unsubscribeCatalog = catalog.subscribe(sync);
  // A rebuild owed to changed lists happens once the focus has left the screen.
  root.addEventListener('focusout', () => {
    if (catalog.get().revision === listsRevision) return;
    setTimeout(() => {
      if (!disposed) sync();
    }, 0);
  });
  sync();

  return () => {
    disposed = true;
    unsubscribe();
    unsubscribeCatalog();
    disposeParts();
  };
}
