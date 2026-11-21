// The recipe's YAML pane: raw editor, changes since the starting point, A/B compare by option,
// and the check's issues. YAML text in `recipe` state is the single source of truth; this pane
// reads recipe.get().yaml and writes with recipe.setYaml(). It never parses YAML itself.

import { h, mount, debounce, uid, fmt } from '../dom.js';
import { api } from '../api.js';
import { persist } from '../store.js';
import { recipe, parsePath } from '../state/recipe.js';
import { tabs, badge, emptyState, notice, loading, issueList, toast, confirmDialog, copyText, countIssues } from '../ui.js';
import { highlightHtml, locatePath } from './yaml-highlight.js';
import { diffText, revertHunk, diffConfigs, compactValue } from './diff.js';
import { openLibraryDialog } from './library.js';

const TABS = [
  { id: 'yaml', label: 'YAML' },
  { id: 'changes', label: 'Changes' },
  { id: 'compare', label: 'Compare' },
  { id: 'issues', label: 'Issues' },
];

const MODIFIERS = new Set(['Shift', 'Control', 'Alt', 'Meta', 'CapsLock']);

const smallButton = (label, onclick, extra = {}) => h('button', { class: 'btn btn--sm', type: 'button', onclick, ...extra }, label);

/** Badge with the number visible and the word for screen readers and hover. */
function countBadge(n, kind, one, many) {
  const words = fmt.plural(n, one, many);
  const el = badge(String(n), kind);
  el.title = words;
  el.append(h('span', { class: 'visually-hidden' }, ` ${n === 1 ? one : many}`));
  return el;
}

// ───────── line diff view (Changes tab and Compare's text mode) ─────────

function diffRow(op, lines) {
  const kind = op.type;
  const text = kind === 'add' ? lines.b[op.b] : lines.a[op.a];
  const sign = kind === 'add' ? '+' : kind === 'del' ? '−' : ' ';
  return h(
    'div',
    { class: `diff__row diff__row--${kind}` },
    h('span', { class: 'diff__num', 'aria-hidden': 'true' }, op.a >= 0 ? String(op.a + 1) : ''),
    h('span', { class: 'diff__num', 'aria-hidden': 'true' }, op.b >= 0 ? String(op.b + 1) : ''),
    h('span', { class: 'diff__sign', 'aria-hidden': 'true' }, sign),
    kind === 'eq' ? null : h('span', { class: 'visually-hidden' }, kind === 'add' ? 'Added: ' : 'Removed: '),
    h('span', { class: 'diff__text' }, text),
  );
}

/**
 * Unified diff with collapsed unchanged stretches.
 * @param {ReturnType<typeof diffText>} d
 * @param {{onRevert?: (hunk: object) => void}} [options]
 */
export function renderDiffView(d, { onRevert = null } = {}) {
  const lines = { a: d.aLines, b: d.bLines };
  const root = h('div', { class: 'diff' });
  const gap = (from, to) => {
    if (to <= from) return;
    const count = to - from;
    const button = h(
      'button',
      {
        class: 'diff__gap',
        type: 'button',
        onclick: () => {
          const frag = document.createDocumentFragment();
          for (let i = from; i < to; i += 1) frag.appendChild(diffRow(d.ops[i], lines));
          button.replaceWith(frag);
        },
      },
      `Show ${fmt.plural(count, 'unchanged line')}`,
    );
    root.appendChild(button);
  };
  let prev = 0;
  for (const hunk of d.hunks) {
    gap(prev, hunk.opStart);
    const first = hunk.bLen ? hunk.bStart + 1 : hunk.bStart;
    const last = hunk.bStart + Math.max(hunk.bLen, 1);
    root.appendChild(
      h(
        'div',
        { class: 'diff__hunk' },
        h('span', { class: 'diff__range' }, hunk.bLen ? `Lines ${first}–${last}` : `Removed after line ${hunk.bStart}`),
        onRevert ? h('button', { class: 'btn btn--sm', type: 'button', onclick: () => onRevert(hunk) }, 'Revert this change') : null,
      ),
    );
    for (let i = hunk.opStart; i < hunk.opEnd; i += 1) root.appendChild(diffRow(d.ops[i], lines));
    prev = hunk.opEnd;
  }
  gap(prev, d.ops.length);
  return root;
}

function diffCounts(d) {
  return h(
    'span',
    { class: 'diff__counts' },
    h('span', { class: 'diff__added', title: fmt.plural(d.added, 'line added', 'lines added') }, `+${d.added}`, h('span', { class: 'visually-hidden' }, ' added')),
    h('span', { class: 'diff__removed', title: fmt.plural(d.removed, 'line removed', 'lines removed') }, `−${d.removed}`, h('span', { class: 'visually-hidden' }, ' removed')),
  );
}

// ───────── the pane ─────────

export function mountSourcePane(el) {
  const cleanups = [];
  const timers = new Set();
  const later = (fn, ms) => {
    const t = setTimeout(() => {
      timers.delete(t);
      fn();
    }, ms);
    timers.add(t);
    return t;
  };

  let active = persist.get('recipe-source-tab', 'yaml');
  if (!TABS.some((t) => t.id === active)) active = 'yaml';

  const panels = Object.fromEntries(TABS.map((t) => [t.id, h('div', { class: 'src__panel', role: 'tabpanel', id: uid('src-panel'), hidden: true })]));
  const onTab = (id) => {
    active = id;
    persist.set('recipe-source-tab', id);
    showActive();
  };
  const strip = tabs({ items: TABS, active, label: 'Recipe text', onChange: onTab });
  for (const t of TABS) {
    const tabButton = strip.el.querySelector(`[data-tab="${t.id}"]`);
    tabButton?.setAttribute('aria-controls', panels[t.id].id);
    panels[t.id].setAttribute('aria-label', t.label);
  }
  const root = h('div', { class: 'src' }, strip.el, Object.values(panels));
  el.appendChild(root);

  // ───────── YAML tab: textarea over a highlighted <pre> ─────────

  const hintId = uid('src-hint');
  const HINT = 'Tab inserts two spaces. Press Escape, then Tab, to move out of the editor.';
  const pre = h('pre', { class: 'src__pre', 'aria-hidden': 'true' });
  const input = h('textarea', { class: 'src__input', spellcheck: 'false', wrap: 'off', autocomplete: 'off', autocapitalize: 'off', 'aria-label': 'Recipe YAML', 'aria-describedby': hintId, title: HINT });
  const gutterInner = h('div', { class: 'src__lines' });
  const gutter = h('div', { class: 'src__gutter', 'aria-hidden': 'true' }, gutterInner);
  const code = h('div', { class: 'src__code' }, pre, input);
  const marksList = h('ul', { class: 'visually-hidden', 'aria-label': 'Issues by line' });
  const status = h('span', { class: 'src__status' });
  let wrap = persist.get('recipe-source-wrap', false) === true;
  const wrapButton = smallButton('Wrap lines', () => {
    wrap = !wrap;
    persist.set('recipe-source-wrap', wrap);
    applyWrap();
  }, { 'aria-pressed': String(wrap) });
  const copyButton = smallButton('Copy', async () => {
    const ok = await copyText(input.value);
    toast(ok ? 'Copied the YAML.' : 'The browser did not allow copying. Select the text and copy it instead.', { kind: ok ? 'info' : 'err' });
  });
  const toolbar = h('div', { class: 'src__toolbar' }, copyButton, smallButton('Download', download), wrapButton, status);
  mount(
    panels.yaml,
    toolbar,
    h('div', { class: 'src__editor' }, gutter, code),
    h('p', { class: 'visually-hidden', id: hintId }, HINT),
    marksList,
  );

  let lineCount = -1;
  let tabReleases = false;
  let issueMarks = new Map();

  function download() {
    const s = recipe.get();
    const name = (s.name || s.summary?.name || 'recipe').replace(/[\\/:*?"<>|]+/g, '_');
    const url = URL.createObjectURL(new Blob([input.value], { type: 'text/yaml' }));
    const a = h('a', { href: url, download: `${name}.yml` });
    document.body.appendChild(a);
    a.click();
    a.remove();
    // revoke on the next turn (synchronously can cancel the download); untracked so cleanup cannot skip it
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function applyWrap() {
    wrapButton.setAttribute('aria-pressed', String(wrap));
    input.setAttribute('wrap', wrap ? 'soft' : 'off');
    pre.classList.toggle('src__pre--wrap', wrap);
    input.classList.toggle('src__input--wrap', wrap);
    syncSize();
    syncScroll();
  }

  /** In wrap mode the highlighted text must wrap at exactly the textarea's inner width. */
  function syncSize() {
    pre.style.width = wrap ? `${input.clientWidth}px` : '';
    syncGutterHeights();
  }

  function syncGutterHeights() {
    const rows = gutterInner.children;
    if (!wrap) {
      for (const row of rows) row.style.height = '';
      return;
    }
    const lines = pre.children;
    for (let i = 0; i < rows.length && i < lines.length; i += 1) rows[i].style.height = `${lines[i].offsetHeight}px`;
  }

  function syncScroll() {
    pre.style.transform = `translate(${-input.scrollLeft}px, ${-input.scrollTop}px)`;
    gutterInner.style.transform = `translateY(${-input.scrollTop}px)`;
  }

  function renderHighlight() {
    const text = input.value;
    pre.innerHTML = highlightHtml(text);
    const count = pre.childElementCount;
    if (count !== lineCount) {
      // line numbers are rebuilt only when the number of lines changes
      lineCount = count;
      const frag = document.createDocumentFragment();
      for (let i = 0; i < count; i += 1) frag.appendChild(h('div', { class: 'src__ln' }, String(i + 1)));
      gutterInner.replaceChildren(frag);
      renderMarks();
    }
    if (wrap) syncGutterHeights();
  }

  function computeMarks() {
    const s = recipe.get();
    const marks = new Map();
    for (const issue of s.issues ?? []) {
      let line = locatePath(input.value, parsePath(issue.path), { partial: true });
      // a parse error has no option path, but the parser's message names the line ("line 370, column 9")
      if (line === null && !issue.path) {
        const m = /\bline (\d+)/.exec(String(issue.message ?? ''));
        if (m) line = Math.min(Number(m[1]) - 1, lineCount - 1);
      }
      if (line === null || line < 0) continue;
      const mark = marks.get(line) ?? { level: 'warning', messages: [] };
      if (issue.level === 'error') mark.level = 'error';
      mark.messages.push(`${issue.level === 'error' ? 'Error' : 'Warning'}: ${issue.message}`);
      marks.set(line, mark);
    }
    issueMarks = marks;
  }

  function renderMarks() {
    const rows = gutterInner.children;
    for (const row of rows) {
      if (row.title) {
        row.className = 'src__ln';
        row.removeAttribute('title');
      }
    }
    const items = [];
    for (const [line, mark] of [...issueMarks].sort((a, b) => a[0] - b[0])) {
      const row = rows[line];
      if (row) {
        row.className = `src__ln src__ln--${mark.level === 'error' ? 'err' : 'warn'}`;
        row.title = mark.messages.join('\n');
      }
      items.push(h('li', null, `Line ${line + 1}: ${mark.messages.join(' ')}`));
    }
    mount(marksList, items);
  }

  function caretText() {
    const s = recipe.get();
    // 'failed' is not "checking": the check itself could not run (the Issues tab says why)
    if (s.check === 'failed' && s.pending === 0) return { text: 'Check could not run', err: true };
    if (s.check !== 'done' || s.pending > 0) return { text: 'Checking…', err: false };
    if (!s.configCurrent) return { text: 'Does not parse', err: true };
    const pos = input.selectionStart ?? 0;
    const before = input.value.slice(0, pos);
    const line = (before.match(/\n/g)?.length ?? 0) + 1;
    const col = pos - before.lastIndexOf('\n');
    return { text: `Line ${line}, column ${col}`, err: false };
  }

  function renderStatus() {
    const { text, err } = caretText();
    if (status.textContent !== text) status.textContent = text;
    status.classList.toggle('src__status--err', err);
  }

  /** Insert through execCommand so the browser's undo stack keeps working. */
  function insertText(text) {
    let ok = false;
    try {
      ok = document.execCommand('insertText', false, text);
    } catch {
      ok = false;
    }
    if (!ok) {
      input.setRangeText(text, input.selectionStart, input.selectionEnd, 'end');
      onInput();
    }
  }

  function selectedLines() {
    const v = input.value;
    const s = input.selectionStart;
    let e = input.selectionEnd;
    // a selection that ends at the start of a line does not include that line
    if (e > s && v[e - 1] === '\n') e -= 1;
    const start = v.lastIndexOf('\n', s - 1) + 1;
    let end = v.indexOf('\n', e);
    if (end < 0) end = v.length;
    return { start, end, s, e: input.selectionEnd };
  }

  function indent() {
    const v = input.value;
    if (!v.slice(input.selectionStart, input.selectionEnd).includes('\n')) {
      insertText('  ');
      return;
    }
    const { start, end, s, e } = selectedLines();
    const block = v.slice(start, end).split('\n');
    input.setSelectionRange(start, end);
    insertText(block.map((line) => `  ${line}`).join('\n'));
    input.setSelectionRange(s + 2, e + 2 * block.length);
  }

  function outdent() {
    const v = input.value;
    const { start, end, s, e } = selectedLines();
    const block = v.slice(start, end).split('\n');
    let firstRemoved = 0;
    let total = 0;
    const next = block.map((line, i) => {
      const n = line.startsWith('\t') ? 1 : /^ {0,2}/.exec(line)[0].length;
      if (i === 0) firstRemoved = n;
      total += n;
      return line.slice(n);
    });
    if (total === 0) return;
    input.setSelectionRange(start, end);
    insertText(next.join('\n'));
    input.setSelectionRange(Math.max(start, s - firstRemoved), Math.max(start, e - total));
  }

  function onInput() {
    renderHighlight();
    recipe.setYaml(input.value);
    renderStatus();
  }

  input.addEventListener('input', onInput);
  input.addEventListener('scroll', syncScroll, { passive: true });
  input.addEventListener('keydown', (event) => {
    if (MODIFIERS.has(event.key)) return;
    if (event.key === 'Escape') {
      tabReleases = true;
      return;
    }
    if (event.key === 'Tab') {
      if (tabReleases || event.ctrlKey || event.altKey || event.metaKey) {
        tabReleases = false;
        return; // let focus move on
      }
      event.preventDefault();
      if (event.shiftKey) outdent();
      else indent();
      return;
    }
    tabReleases = false;
    if (event.key === 'Enter' && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey && !event.isComposing) {
      event.preventDefault();
      const v = input.value;
      const lineStart = v.lastIndexOf('\n', input.selectionStart - 1) + 1;
      const indentText = /^[ \t]*/.exec(v.slice(lineStart, input.selectionStart))[0];
      insertText(`\n${indentText}`);
    }
  });
  input.addEventListener('blur', () => {
    tabReleases = false;
  });
  // selectionchange covers most caret moves; keyup/mouseup cover browsers that fire it late
  input.addEventListener('keyup', renderStatus);
  input.addEventListener('mouseup', renderStatus);
  const onSelection = () => {
    if (document.activeElement === input) renderStatus();
  };
  document.addEventListener('selectionchange', onSelection);
  cleanups.push(() => document.removeEventListener('selectionchange', onSelection));

  const resize = new ResizeObserver(() => syncSize());
  resize.observe(input);
  cleanups.push(() => resize.disconnect());

  /** A form edit or a revert changed the text: replace it without losing the place. */
  function takeExternalYaml(yaml) {
    if (yaml === input.value) return;
    const focused = document.activeElement === input;
    const { selectionStart, selectionEnd, scrollTop, scrollLeft } = input;
    input.value = yaml;
    if (focused) input.setSelectionRange(Math.min(selectionStart, yaml.length), Math.min(selectionEnd, yaml.length));
    input.scrollTop = scrollTop;
    input.scrollLeft = scrollLeft;
    renderHighlight();
    computeMarks();
    renderMarks();
    syncScroll();
  }

  function flashLine(path) {
    if (active !== 'yaml') return;
    const line = locatePath(input.value, parsePath(path), { partial: true });
    if (line === null) return;
    const lineEl = pre.children[line];
    if (!lineEl) return;
    input.scrollTop = Math.max(0, lineEl.offsetTop - input.clientHeight / 2 + lineEl.offsetHeight / 2);
    syncScroll();
    const row = gutterInner.children[line];
    lineEl.classList.add('yaml__line--flash');
    row?.classList.add('src__ln--flash');
    later(() => {
      lineEl.classList.remove('yaml__line--flash');
      row?.classList.remove('src__ln--flash');
    }, 1200);
  }
  cleanups.push(recipe.onFocusRequest(flashLine));

  input.value = recipe.get().yaml;
  renderHighlight();
  computeMarks();
  renderMarks();
  applyWrap();
  renderStatus();

  // ───────── Changes tab ─────────

  function renderChanges() {
    const s = recipe.get();
    const d = diffText(s.baseYaml, s.yaml, 3);
    const label = s.baseLabel || 'the starting point';
    if (!d.hunks.length) {
      mount(panels.changes, emptyState({ title: 'No changes yet', text: `Edits you make to this recipe show up here, compared with ${label}.` }));
      return;
    }
    const revertAll = smallButton('Revert all changes', async () => {
      const body = `The recipe goes back to ${label}. The edits made since then are discarded.`;
      if (await confirmDialog({ title: 'Revert all changes?', body, confirmLabel: 'Revert all changes', danger: true })) recipe.revertToBase();
    });
    const keep = smallButton('Keep as new starting point', () => {
      recipe.rebase();
      toast('Changes are now measured from here.');
    });
    mount(
      panels.changes,
      h('div', { class: 'src__head' }, h('span', { class: 'src__headtext truncate', title: label }, `Compared with ${label}`), diffCounts(d), revertAll, keep),
      h('div', { class: 'src__scroll' }, renderDiffView(d, { onRevert: (hunk) => revertOne(d, hunk, s) })),
    );
  }

  /** Revert one hunk, but only against the text it was computed from (a re-render is debounced). */
  function revertOne(d, hunk, rendered) {
    const s = recipe.get();
    if (s.yaml !== rendered.yaml || s.baseYaml !== rendered.baseYaml) {
      renderChanges();
      toast('The recipe changed while this list was open. Check the change again, then revert it.');
      return;
    }
    recipe.setYaml(revertHunk(d, hunk), { immediate: true });
  }

  const updateChangesBadge = debounce(() => {
    const s = recipe.get();
    const n = s.yaml === s.baseYaml ? 0 : (() => {
      const d = diffText(s.baseYaml, s.yaml, 0);
      return d.added + d.removed;
    })();
    strip.setBadge('changes', n > 0 ? countBadge(n, '', 'changed line', 'changed lines') : null);
  }, 250);
  cleanups.push(() => updateChangesBadge.cancel());

  // ───────── Compare tab ─────────

  const cmp = { picked: null, config: null, error: null, loading: false, mode: 'options', filter: 'diff', search: '', token: 0 };
  let cmpResults = null;
  let cmpRowsKey = null;
  let cmpRows = [];

  const bName = () => cmp.picked?.doc?.name || 'B';

  async function chooseCompare() {
    const picked = await openLibraryDialog({ title: 'Choose a recipe to compare', pickLabel: 'Compare with this recipe' });
    if (!picked || !picked.doc || typeof picked.doc.yaml !== 'string') return;
    const token = ++cmp.token;
    Object.assign(cmp, { picked, config: null, error: null, loading: true });
    cmpRowsKey = null;
    if (active === 'compare') renderCompare();
    try {
      const result = await api.preflight(picked.doc.yaml);
      if (token !== cmp.token) return;
      cmp.config = result?.config && typeof result.config === 'object' ? result.config : null;
      if (!cmp.config) cmp.error = `The check returned no parsed options for ${picked.doc.name || 'this recipe'}, so there is nothing to compare by option. The text diff still works.`;
    } catch (error) {
      if (token !== cmp.token) return;
      cmp.error = `${picked.doc.name || 'The recipe'} could not be checked: ${error.message}`;
    }
    cmp.loading = false;
    if (active === 'compare') renderCompare();
  }

  function renderCompare() {
    if (!cmp.picked) {
      cmpResults = null;
      mount(
        panels.compare,
        emptyState({
          title: 'Compare with another recipe',
          text: 'See which options differ from a template or a saved recipe, and copy values across.',
          actions: [h('button', { class: 'btn btn--primary', type: 'button', onclick: chooseCompare }, 'Choose a recipe to compare')],
        }),
      );
      return;
    }
    const textMode = cmp.mode === 'text';
    const textToggle = smallButton('Text diff', () => {
      cmp.mode = textMode ? 'options' : 'text';
      renderCompare();
    }, { 'aria-pressed': String(textMode) });
    const head = h(
      'div',
      { class: 'src__head' },
      h('span', { class: 'src__headtext truncate', title: cmp.picked.label }, h('span', { class: 'muted' }, 'B: '), cmp.picked.label),
      smallButton('Change', chooseCompare),
      textToggle,
    );
    let controls = null;
    if (!textMode) {
      const searchId = uid('src-search');
      // the controls are built once per B and mode; typing in the search only re-renders the results
      const filterSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Show options' });
      for (const [id, text] of [['diff', 'Differences'], ['all', 'All']]) {
        filterSeg.appendChild(smallButton(text, (event) => {
          cmp.filter = id;
          for (const b of filterSeg.children) b.setAttribute('aria-pressed', String(b === event.currentTarget));
          renderCompareResults();
        }, { 'aria-pressed': String(cmp.filter === id) }));
      }
      const onSearch = (event) => {
        cmp.search = event.currentTarget.value;
        renderCompareResults();
      };
      controls = h(
        'div',
        { class: 'src__controls' },
        filterSeg,
        h('label', { class: 'visually-hidden', for: searchId }, 'Find an option'),
        h('input', { class: 'input src__search', id: searchId, type: 'search', placeholder: 'Find an option', value: cmp.search, autocomplete: 'off', spellcheck: 'false', oninput: onSearch }),
      );
    }
    cmpResults = h('div', { class: 'src__scroll', role: 'region', 'aria-label': textMode ? 'Text differences' : 'Option differences', tabindex: '0' });
    mount(panels.compare, head, controls, cmpResults);
    renderCompareResults();
  }

  function renderCompareResults() {
    if (!cmpResults) return;
    const s = recipe.get();
    if (cmp.mode === 'text') {
      const d = diffText(s.yaml, cmp.picked.doc.yaml, 3);
      if (!d.hunks.length) {
        mount(cmpResults, emptyState({ title: 'The texts are the same', text: `This recipe and ${bName()} have identical YAML.` }));
        return;
      }
      mount(cmpResults, h('div', { class: 'src__subhead' }, `This recipe against ${bName()}`, diffCounts(d)), renderDiffView(d));
      return;
    }
    if (cmp.loading) {
      mount(cmpResults, loading(`Checking ${bName()}…`));
      return;
    }
    const parts = [];
    if (cmp.error) {
      mount(cmpResults, notice('err', cmp.error));
      return;
    }
    if (!s.config) {
      mount(cmpResults, notice('warn', 'This recipe has no parsed options yet. They appear once its check has run.'));
      return;
    }
    if (s.check === 'done' && !s.configCurrent) {
      parts.push(notice('warn', 'This recipe does not parse right now. The table uses the last options that did.'));
    }
    if (cmpRowsKey?.a !== s.config || cmpRowsKey?.b !== cmp.config) {
      cmpRows = diffConfigs(s.config, cmp.config);
      cmpRowsKey = { a: s.config, b: cmp.config };
    }
    const query = cmp.search.trim().toLowerCase();
    const shown = cmpRows.filter((row) => (cmp.filter === 'all' || row.state !== 'same') && (!query || row.path.toLowerCase().includes(query)));
    const differing = cmpRows.filter((row) => row.state !== 'same').length;
    parts.push(h('p', { class: 'src__subhead' }, `${fmt.plural(differing, 'option differs', 'options differ')} of ${fmt.int(cmpRows.length)}.`));
    if (!shown.length) {
      parts.push(h('p', { class: 'src__none' }, query ? 'No options match.' : 'No options differ.'));
      mount(cmpResults, parts);
      return;
    }
    const groups = new Map();
    for (const row of shown) {
      if (!groups.has(row.group)) groups.set(row.group, []);
      groups.get(row.group).push(row);
    }
    const table = h(
      'table',
      { class: 'table src__table' },
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', { class: 'src__colhead', scope: 'col' }, 'Option'),
          h('th', { class: 'src__colhead', scope: 'col' }, 'This recipe'),
          h('th', { class: 'src__colhead truncate', scope: 'col', title: bName() }, bName()),
          h('th', { class: 'src__colhead', scope: 'col' }, h('span', { class: 'visually-hidden' }, 'State and action')),
        ),
      ),
      [...groups].map(([group, rows]) =>
        h(
          'tbody',
          null,
          h('tr', null, h('th', { class: 'src__group', colspan: '4', scope: 'colgroup' }, group, ' ', h('span', { class: 'muted' }, String(rows.length)))),
          rows.map(compareRow),
        ),
      ),
    );
    parts.push(table);
    mount(cmpResults, parts);
  }

  function valueCell(present, value) {
    if (!present) return h('td', { class: 'src__val muted' }, 'Not set');
    const text = compactValue(value);
    const short = text.length > 60 ? `${text.slice(0, 57)}…` : text;
    return h('td', { class: 'src__val', title: text.length > 60 ? text : null }, short);
  }

  const STATE_WORDS = { different: 'Different', 'only-a': 'Only here', 'only-b': 'Only in B', same: 'Same' };

  function compareRow(row) {
    // only where B's value maps to one edit in A (a type-matched item missing in A has no index)
    const canUse = (row.state === 'different' || row.state === 'only-b') && Array.isArray(row.edit);
    const use = () =>
      recipe
        .applyEdits([{ path: row.edit, value: row.b }])
        .then(() => toast(`Set ${row.path} from ${bName()}.`, { kind: 'ok' }))
        .catch((error) => toast(error.message, { kind: 'err' }));
    return h(
      'tr',
      { class: `src__row src__row--${row.state}` },
      // <wbr> after separators so a long option name wraps between words, not inside one
      h('td', { class: 'src__path', title: row.path }, row.path.split(/(?<=[_.\]])/).flatMap((piece, i) => (i ? [h('wbr'), piece] : [piece]))),
      valueCell(row.hasA, row.a),
      valueCell(row.hasB, row.b),
      h('td', { class: 'src__act' }, h('span', { class: 'src__state' }, STATE_WORDS[row.state]), canUse ? smallButton("Use B's value", use) : null),
    );
  }

  // ───────── Issues tab ─────────

  function renderIssues() {
    const s = recipe.get();
    const { errors, warnings } = countIssues(s.issues);
    if (s.check === 'failed') {
      const again = smallButton('Check again', () => recipe.checkNow());
      mount(panels.issues, h('div', { class: 'src__pad' }, notice('err', `The recipe could not be checked: ${s.checkError || 'no reason was given'}`, again)));
      return;
    }
    const hasResult = s.ok !== null || s.issues.length > 0;
    if (s.check !== 'done' && !hasResult) {
      mount(panels.issues, loading('Checking the recipe…'));
      return;
    }
    if (!s.issues.length && s.ok) {
      mount(panels.issues, emptyState({ title: 'No errors or warnings', text: 'The trainer service accepted this recipe.' }));
      return;
    }
    let summary;
    if (!s.issues.length) summary = 'The recipe did not pass its check, and the check named no option.';
    else {
      const parts = [];
      if (errors) parts.push(fmt.plural(errors, 'error'));
      if (warnings) parts.push(fmt.plural(warnings, 'warning'));
      summary = `${parts.join(' and ')}. Select one to find it in the recipe.`;
    }
    if (s.check !== 'done') summary = `Checking again… Last result: ${summary}`;
    mount(
      panels.issues,
      h('p', { class: 'src__subhead', role: 'status' }, summary),
      h(
        'div',
        { class: 'src__scroll' },
        s.issues.length ? issueList(s.issues, { onSelect: (issue) => recipe.requestFocus(issue.path) }) : null,
        // How the real check (POST /api/preflight) behaves: it stops at the first structural error.
        errors
          ? h(
              'p',
              { class: 'src__pad muted' },
              'The check reports one structural error at a time (a wrong type, an unknown or missing option) and looks at folders and files only once the structure is valid. Fixing an error can reveal the next one.',
            )
          : null,
      ),
    );
  }

  function updateIssuesBadge() {
    const { errors, warnings } = countIssues(recipe.get().issues);
    const nodes = [];
    if (errors) nodes.push(countBadge(errors, 'err', 'error', 'errors'));
    if (warnings) nodes.push(countBadge(warnings, 'warn', 'warning', 'warnings'));
    strip.setBadge('issues', nodes.length ? nodes : null);
  }

  // ───────── switching and state updates ─────────

  const renderers = { changes: renderChanges, compare: renderCompare, issues: renderIssues };

  function showActive() {
    for (const t of TABS) panels[t.id].hidden = t.id !== active;
    if (active === 'yaml') {
      // the editor is built once and kept; only its size may need a refresh after being hidden
      syncSize();
      syncScroll();
    } else renderers[active]();
  }

  const renderChangesSoon = debounce(() => active === 'changes' && renderChanges(), 150);
  const renderCompareSoon = debounce(() => active === 'compare' && renderCompareResults(), 150);
  cleanups.push(() => renderChangesSoon.cancel(), () => renderCompareSoon.cancel());

  let prev = recipe.get();
  const unsubscribe = recipe.subscribe((s) => {
    const before = prev;
    prev = s;
    if (s.phase !== 'open') return;
    if (s.yaml !== before.yaml) takeExternalYaml(s.yaml);
    if (s.issues !== before.issues) {
      computeMarks();
      renderMarks();
    }
    if (s.check !== before.check || s.pending !== before.pending || s.configCurrent !== before.configCurrent) renderStatus();

    if (s.yaml !== before.yaml || s.baseYaml !== before.baseYaml) updateChangesBadge();
    if (s.issues !== before.issues) updateIssuesBadge();

    if (active === 'changes' && (s.yaml !== before.yaml || s.baseYaml !== before.baseYaml || s.baseLabel !== before.baseLabel)) renderChangesSoon();
    if (active === 'issues' && (s.issues !== before.issues || s.check !== before.check || s.checkError !== before.checkError || s.ok !== before.ok)) renderIssues();
    if (active === 'compare' && cmp.picked) {
      const textChanged = cmp.mode === 'text' && s.yaml !== before.yaml;
      const configChanged = cmp.mode === 'options' && (s.config !== before.config || s.configCurrent !== before.configCurrent || s.check !== before.check);
      if (textChanged || configChanged) renderCompareSoon();
    }
  });
  cleanups.push(unsubscribe);

  updateChangesBadge.flush();
  updateIssuesBadge();
  showActive();

  return () => {
    for (const fn of cleanups) fn();
    for (const t of timers) clearTimeout(t);
    timers.clear();
    cmp.token += 1;
    root.remove();
  };
}
