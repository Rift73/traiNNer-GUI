// Shared interface pieces: toasts, dialogs, tabs, issue lists, empty states.
// Markup here matches the classes in css/base.css.

import { h, mount, uid } from './dom.js';
import { createCombobox, toChoices } from './components/combobox.js';

// ───────── toasts ─────────

let toastHost = null;

/**
 * @param {string} message
 * @param {{kind?: 'info'|'ok'|'err', action?: {label: string, run: () => void}, timeout?: number}} [options]
 */
export function toast(message, { kind = 'info', action = null, timeout = kind === 'err' ? 9000 : 4500 } = {}) {
  if (!toastHost) {
    toastHost = h('div', { class: 'toast-host', role: 'region', 'aria-label': 'Notifications' });
    document.body.appendChild(toastHost);
  }
  const el = h(
    'div',
    { class: `toast toast--${kind}`, role: kind === 'err' ? 'alert' : 'status' },
    h('span', { class: 'grow' }, message),
    action
      ? h(
          'button',
          {
            class: 'btn btn--sm',
            type: 'button',
            onclick: () => {
              el.remove();
              action.run();
            },
          },
          action.label,
        )
      : null,
    h('button', { class: 'btn btn--sm btn--quiet', type: 'button', 'aria-label': 'Dismiss', onclick: () => el.remove() }, '✕'),
  );
  toastHost.appendChild(el);
  if (timeout > 0) setTimeout(() => el.remove(), timeout);
  return el;
}

// ───────── dialogs ─────────

/**
 * Open a modal built on <dialog>, so focus trapping and Escape come from the browser.
 * @param {{title: string, body: Node|Node[], foot?: (close: (result?: any) => void) => Node[], wide?: boolean, flush?: boolean}} options
 * @returns {{dialog: HTMLDialogElement, close: (result?: any) => void, closed: Promise<any>}}
 */
export function openDialog({ title, body, foot = null, wide = false, flush = false }) {
  const titleId = uid('dialog-title');
  let result;
  let settle;
  const closed = new Promise((resolve) => {
    settle = resolve;
  });
  const close = (value) => {
    result = value;
    dialog.close();
  };
  const dialog = h(
    'dialog',
    { class: 'dialog', 'aria-labelledby': titleId, style: wide ? { width: 'min(960px, 94vw)' } : null },
    h(
      'div',
      { class: 'dialog__head' },
      h('h2', { class: 'dialog__title', id: titleId }, title),
      h('button', { class: 'btn btn--quiet btn--sm', type: 'button', 'aria-label': 'Close', onclick: () => close(undefined) }, '✕'),
    ),
    h('div', { class: `dialog__body${flush ? ' dialog__body--flush' : ''}` }, body),
  );
  if (foot) dialog.appendChild(h('div', { class: 'dialog__foot' }, foot(close)));
  dialog.addEventListener('close', () => {
    dialog.remove();
    settle(result);
  });
  // Click on the backdrop closes, like Escape.
  dialog.addEventListener('mousedown', (event) => {
    if (event.target === dialog) close(undefined);
  });
  document.body.appendChild(dialog);
  dialog.showModal();
  return { dialog, close, closed };
}

/** @returns {Promise<boolean>} */
export function confirmDialog({ title, body, confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger = false }) {
  const { closed } = openDialog({
    title,
    body: typeof body === 'string' ? h('p', null, body) : body,
    foot: (close) => [
      // A destructive confirmation starts on Cancel, so Enter cannot destroy by reflex.
      h('button', { class: 'btn', type: 'button', autofocus: danger, onclick: () => close(false) }, cancelLabel),
      h(
        'button',
        { class: `btn ${danger ? 'btn--danger' : 'btn--primary'}`, type: 'button', autofocus: !danger, onclick: () => close(true) },
        confirmLabel,
      ),
    ],
  });
  return closed.then((value) => value === true);
}

/**
 * Ask before an action that makes the trainer start training, on the phone edition only (mobile.html
 * marks the document): a button on a phone is easy to touch by accident, and the trainer may be far
 * away. The desktop page acts at once: there this resolves true without asking.
 * @returns {Promise<boolean>}
 */
export function confirmOnPhone(options) {
  return document.documentElement.dataset.edition === 'mobile' ? confirmDialog(options) : Promise.resolve(true);
}

/**
 * Ask for one line of text. With `choices` (known values, as plain values or {value, note}) the field
 * lists them under it; any other text can still be typed.
 * @param {{title: string, label: string, value?: string, help?: string, confirmLabel?: string, validate?: (v: string) => string|null,
 *   choices?: Array|null}} options
 * @returns {Promise<string|null>}
 */
export function promptDialog({ title, label, value = '', help = '', confirmLabel = 'Save', validate = null, choices = null }) {
  const inputId = uid('prompt');
  const error = h('p', { class: 'field__error', role: 'alert' });
  const listed = Array.isArray(choices) && choices.length ? toChoices(choices) : null;
  // Enter in the list takes the highlighted choice; Enter with none highlighted answers the prompt.
  const combo = listed ? createCombobox({ id: inputId, inputClass: 'input--path', getChoices: () => listed, listLabel: label, onEnter: () => submit() }) : null;
  const input = combo ? combo.input : h('input', { class: 'input', id: inputId, type: 'text', value, autocomplete: 'off', spellcheck: 'false' });
  if (combo) input.value = value;
  let closeDialog;
  const submit = () => {
    const text = input.value.trim();
    const problem = validate ? validate(text) : text ? null : 'Enter a value.';
    if (problem) {
      error.textContent = problem;
      input.setAttribute('aria-invalid', 'true');
      input.focus();
      return;
    }
    closeDialog(text);
  };
  if (!combo) {
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submit();
      }
    });
  }
  const { closed, close } = openDialog({
    title,
    body: h(
      'div',
      { class: 'field' },
      h('label', { class: 'field__label', for: inputId }, label),
      combo ? combo.el : input,
      help ? h('p', { class: 'field__help' }, help) : null,
      error,
    ),
    foot: (closeFn) => [
      h('button', { class: 'btn', type: 'button', onclick: () => closeFn(null) }, 'Cancel'),
      h('button', { class: 'btn btn--primary', type: 'button', onclick: submit }, confirmLabel),
    ],
  });
  closeDialog = close;
  input.focus();
  input.select();
  return closed.then((result) => (typeof result === 'string' ? result : null));
}

// ───────── tabs ─────────

/**
 * Accessible tab strip (arrow keys move between tabs). The caller owns the panels.
 * @param {{items: Array<{id: string, label: string}>, active: string, onChange: (id: string) => void, label: string}} options
 */
export function tabs({ items, active, onChange, label }) {
  const buttons = new Map();
  const badges = new Map();
  const select = (id, focus = false) => {
    for (const [key, button] of buttons) {
      const on = key === id;
      button.setAttribute('aria-selected', String(on));
      button.tabIndex = on ? 0 : -1;
      if (on && focus) button.focus();
    }
  };
  const el = h('div', { class: 'tabs', role: 'tablist', 'aria-label': label });
  items.forEach((item, index) => {
    const badgeSlot = h('span');
    const button = h(
      'button',
      {
        class: 'tab',
        type: 'button',
        role: 'tab',
        dataset: { tab: item.id },
        onclick: () => {
          select(item.id);
          onChange(item.id);
        },
        onkeydown: (event) => {
          const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
          if (!step) return;
          event.preventDefault();
          const next = items[(index + step + items.length) % items.length];
          select(next.id, true);
          onChange(next.id);
        },
      },
      item.label,
      badgeSlot,
    );
    buttons.set(item.id, button);
    badges.set(item.id, badgeSlot);
    el.appendChild(button);
  });
  select(active);
  return {
    el,
    setActive: (id) => select(id),
    /** Put a small node (usually a badge) after a tab label; pass null to clear it. */
    setBadge(id, node) {
      const slot = badges.get(id);
      if (slot) mount(slot, node);
    },
  };
}

// ───────── small pieces ─────────

/** @param {'ok'|'warn'|'err'|'run'|''} kind */
export function badge(text, kind = '') {
  return h('span', { class: `badge${kind ? ` badge--${kind}` : ''}` }, text);
}

/** Role chip. role: 'gt' (ground truth), 'lq' (low-quality input), 'out' (model output). */
export function roleChip(role, text) {
  return h('span', { class: `role role--${role}` }, text);
}

export function loading(text = 'Loading…') {
  return h('div', { class: 'loading', role: 'status' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), text);
}

/** @param {{title: string, text?: string|Node, actions?: Node[]}} options */
export function emptyState({ title, text = '', actions = [] }) {
  return h(
    'div',
    { class: 'empty' },
    h('h2', { class: 'empty__title' }, title),
    text ? h('p', { class: 'empty__text' }, text) : null,
    actions.length ? h('div', { class: 'empty__actions' }, actions) : null,
  );
}

/** @param {'err'|'warn'|'ok'|''} kind */
export function notice(kind, body, action = null) {
  return h(
    'div',
    { class: `notice${kind ? ` notice--${kind}` : ''}`, role: kind === 'err' ? 'alert' : 'status' },
    h('div', { class: 'notice__body' }, body),
    action,
  );
}

/**
 * List of check results. Items become buttons when onSelect is given.
 * @param {Array<{level: string, path?: string, message: string}>} issues
 * @param {{onSelect?: (issue: object) => void}} [options]
 */
export function issueList(issues, { onSelect = null } = {}) {
  const ordered = [...issues].sort((a, b) => Number(b.level === 'error') - Number(a.level === 'error'));
  return h(
    'div',
    { class: 'issues', role: 'list' },
    ordered.map((issue) => {
      const isError = issue.level === 'error';
      const content = [
        h('span', { class: 'issue__level' }, isError ? 'Error' : 'Warning'),
        h('span', { class: 'issue__path' }, issue.path || ''),
        h('span', { class: 'issue__message' }, issue.message),
      ];
      const cls = `issue issue--${isError ? 'error' : 'warning'}`;
      return onSelect
        ? h('button', { class: cls, type: 'button', role: 'listitem', onclick: () => onSelect(issue) }, content)
        : h('div', { class: cls, role: 'listitem' }, content);
    }),
  );
}

/** Four crop marks for a positioned frame inside a .stage. */
export function cropmarks() {
  return ['tl', 'tr', 'bl', 'br'].map((corner) => h('i', { class: `cropmark cropmark--${corner}`, 'aria-hidden': 'true' }));
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/** Count issues by level. */
export function countIssues(issues) {
  let errors = 0;
  let warnings = 0;
  for (const issue of issues ?? []) {
    if (issue.level === 'error') errors += 1;
    else warnings += 1;
  }
  return { errors, warnings };
}
