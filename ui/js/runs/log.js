// Log tab of a run: the lines the service sent (GET /api/run, then GET /api/run/monitor with a cursor).
// The detail owns the polling and hands lines over with replace() / append(); this module only shows
// them. New lines are appended in place, so focus, scroll position and a text selection survive polls.

import { h, fmt, debounce, uid } from '../dom.js';
import { toast, copyText } from '../ui.js';
import { classifyLogLine } from './metrics.js';

/** Lines kept on screen; older ones are dropped (and the view says how many). */
export const MAX_LOG_LINES = 5000;

/**
 * @param {HTMLElement} host
 * @param {{getName: () => string}} options
 * @returns {{replace(lines: string[]): void, append(lines: string[]): void, setShown(shown: boolean): void, focusEnd(): void, destroy(): void}}
 */
export function createLogView(host, { getName }) {
  let lines = [];
  const lineEls = [];
  let dropped = 0;
  let follow = true;
  let shown = false;
  let filterText = '';
  let downloadUrl = null;

  const followInput = h('input', { type: 'checkbox', checked: true, onchange: () => setFollow(followInput.checked) });
  const wrapInput = h('input', {
    type: 'checkbox',
    onchange: () => {
      viewport.classList.toggle('runs__log--wrap', wrapInput.checked);
      if (follow) pin();
    },
  });
  // A phone's line is a third of a log line: there (mobile.html) the lines start wrapped. The switch
  // still turns it off.
  const startWrapped = document.documentElement.dataset.edition === 'mobile';
  wrapInput.checked = startWrapped;
  const filterInput = h('input', {
    class: 'input runs__filter',
    type: 'search',
    placeholder: 'Show lines containing…',
    'aria-label': 'Show lines containing',
    oninput: debounce(() => {
      filterText = filterInput.value.trim().toLowerCase();
      applyFilter();
    }, 120),
  });
  // Not a live region: it changes on every poll. The filter input points at it with aria-describedby.
  const countId = uid('runs-log-count');
  const countEl = h('span', { class: 'runs__count muted num', id: countId });
  filterInput.setAttribute('aria-describedby', countId);
  const jumpBtn = h('button', { class: 'btn btn--sm', type: 'button', hidden: true, onclick: () => setFollow(true) }, 'Jump to newest');
  const copyBtn = h('button', { class: 'btn btn--sm', type: 'button', onclick: copyLog }, 'Copy');
  const downloadBtn = h('button', { class: 'btn btn--sm', type: 'button', onclick: downloadLog }, 'Download');
  const toggle = (input, text) => h('label', { class: 'switch' }, input, h('span', { class: 'switch__track' }), text);
  const toolbar = h(
    'div',
    { class: 'runs__toolbar' },
    toggle(followInput, 'Follow'),
    jumpBtn,
    h('div', { class: 'runs__filter-wrap' }, filterInput),
    countEl,
    toggle(wrapInput, 'Wrap lines'),
    copyBtn,
    downloadBtn,
  );
  const droppedEl = h('p', { class: 'runs__log-note muted', hidden: true });
  const viewport = h('div', { class: startWrapped ? 'runs__log runs__log--wrap mono' : 'runs__log mono', tabindex: '0', role: 'region', 'aria-label': 'Log lines, oldest first', onscroll: onScroll });
  const empty = h('p', { class: 'runs__pad muted', tabindex: '-1' }, 'This run has no log lines yet.');
  const wrap = h('div', { class: 'runs__logtab' }, toolbar, droppedEl, viewport);
  host.append(empty, wrap);
  sync();

  function lineEl(text) {
    const level = classifyLogLine(text);
    const word = h('span', { class: 'runs__lvl' }, level === 'error' ? 'Error' : level === 'warn' ? 'Warning' : '');
    const node = h('div', { class: level ? `runs__line runs__line--${level}` : 'runs__line' }, word, h('span', { class: 'runs__text' }, text));
    node.hidden = !matches(text);
    return node;
  }

  function matches(text) {
    return !filterText || text.toLowerCase().includes(filterText);
  }

  function applyFilter() {
    lines.forEach((text, i) => {
      lineEls[i].hidden = !matches(text);
    });
    sync();
    if (follow) pin();
  }

  function sync() {
    const has = lines.length > 0;
    empty.hidden = has;
    wrap.hidden = !has;
    copyBtn.disabled = !has;
    downloadBtn.disabled = !has;
    const count = filterText ? `${fmt.int(lineEls.filter((n) => !n.hidden).length)} of ${fmt.int(lines.length)} lines` : fmt.plural(lines.length, 'line');
    if (countEl.textContent !== count) countEl.textContent = count;
    droppedEl.hidden = dropped === 0;
    const note = dropped
      ? `${fmt.plural(dropped, 'older line was', 'older lines were')} dropped from this view, which keeps the newest ${fmt.int(MAX_LOG_LINES)}. Copy and Download hold what is shown.`
      : '';
    if (droppedEl.textContent !== note) droppedEl.textContent = note;
  }

  function pin() {
    if (shown) viewport.scrollTop = viewport.scrollHeight;
  }

  function setFollow(on) {
    follow = on;
    followInput.checked = on;
    jumpBtn.hidden = on;
    if (on) pin();
  }

  function onScroll() {
    // Scrolling away from the newest line turns Follow off; our own pin() lands at the bottom.
    const fromBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    if (follow && fromBottom > 8) setFollow(false);
    else if (!follow && fromBottom <= 2) setFollow(true);
  }

  function push(incoming) {
    if (!incoming.length) return;
    const frag = document.createDocumentFragment();
    for (const raw of incoming) {
      const text = String(raw ?? '');
      const node = lineEl(text);
      lines.push(text);
      lineEls.push(node);
      frag.appendChild(node);
    }
    viewport.appendChild(frag);
    if (lines.length > MAX_LOG_LINES) {
      const extra = lines.length - MAX_LOG_LINES;
      lines.splice(0, extra);
      for (const node of lineEls.splice(0, extra)) node.remove();
      dropped += extra;
    }
  }

  async function copyLog() {
    const ok = await copyText(lines.join('\n'));
    toast(ok ? `Copied ${fmt.plural(lines.length, 'log line')}.` : 'The log was not copied: the browser refused access to the clipboard.', { kind: ok ? 'ok' : 'err' });
  }

  function downloadLog() {
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    downloadUrl = URL.createObjectURL(new Blob([`${lines.join('\n')}\n`], { type: 'text/plain' }));
    const name = `${String(getName() || 'run').replace(/[\\/:*?"<>|]+/g, '_')}.log`;
    const a = h('a', { href: downloadUrl, download: name, hidden: true });
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  return {
    /** The service started the log over (first load, or `logReset`): show these lines instead. */
    replace(next) {
      lines = [];
      lineEls.length = 0;
      dropped = 0;
      viewport.replaceChildren();
      push(Array.isArray(next) ? next : []);
      sync();
      if (follow) pin();
    },
    /** New lines after the cursor. */
    append(next) {
      const incoming = Array.isArray(next) ? next : [];
      if (!incoming.length) return;
      push(incoming);
      sync();
      if (follow) pin();
    },
    /** The tab became visible or hidden: a hidden viewport cannot be scrolled, so pin on show. */
    setShown(on) {
      shown = on;
      if (on && follow) pin();
    },
    /** Move focus to the log lines, scrolled to the newest one (call after the tab is shown). */
    focusEnd() {
      setFollow(true);
      (lines.length ? viewport : empty).focus?.();
      pin();
    },
    destroy() {
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      downloadUrl = null;
      host.replaceChildren();
    },
  };
}
