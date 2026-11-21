// A text input with its known values in a list under it: pick with the mouse or the keyboard, or type.
// The one control for "a value with known choices". It only shows and picks: what a picked or typed
// value means (write it, refuse it, ask first) is the caller's, through onPick / onEnter / onEscape.
//
// Keyboard (WAI-ARIA combobox with a listbox popup):
//   ArrowDown / ArrowUp   open the list, then move through it (Alt+ArrowDown opens without moving)
//   Home / End            first / last choice while the list is open
//   Enter                 take the highlighted choice; with none highlighted, the typed text (onEnter,
//                         or the key's usual meaning around the field when the caller gives none)
//   Escape                close the list and keep the text; with the list closed it is the caller's (onEscape)
//   Tab                   leaves the field: the list closes and the caller's blur handler decides
// Typing opens the list and narrows it. Opened without typing, the list shows every choice.
// Mouse: a click in the field or on its arrow opens the list; a click on a choice takes it.

import { h, uid } from '../dom.js';

/**
 * @typedef {{value: any, text: string, note?: string|null, unset?: boolean}} Choice
 *   value: what is written; text: what the input shows and what typing is matched against;
 *   note: a few quiet words after the text ("recommended for this task");
 *   unset: the choice that is no value at all ("Decided from the network"): taking it removes the option.
 */

/** How the empty text is written where it is a choice of its own: a field cannot show nothing as a value. */
export const EMPTY_TEXT = '""';

const textOf = (value) => (typeof value === 'string' ? value : JSON.stringify(value));

/**
 * Choices from a mixed list: plain values, {value, text?, note?} or {unset: true, text, note?}.
 * Entries without a usable text are dropped. The empty text is a choice like any other, shown as "".
 */
export function toChoices(list) {
  const out = [];
  const seen = new Set();
  for (const item of Array.isArray(list) ? list : []) {
    const plain = item !== null && typeof item === 'object' && !Array.isArray(item);
    const unset = plain && item.unset === true;
    const record = plain && ('value' in item || unset);
    const value = unset ? undefined : record ? item.value : item;
    if (!unset && (value === undefined || value === null)) continue;
    const empty = value === '';
    const text = record && typeof item.text === 'string' && item.text ? item.text : empty ? EMPTY_TEXT : unset ? '' : textOf(value);
    if (typeof text !== 'string' || !text || seen.has(text)) continue;
    seen.add(text);
    const note = record && typeof item.note === 'string' && item.note ? item.note : empty ? 'empty text' : null;
    out.push(unset ? { value: undefined, text, note, unset: true } : { value, text, note });
  }
  return out;
}

/** The choice whose text is `text`: exact first, then without regard to case; null when there is none. */
export function matchChoice(choices, text) {
  const t = String(text ?? '').trim();
  if (!t) return null;
  const lower = t.toLowerCase();
  return choices.find((c) => c.text === t) ?? choices.find((c) => c.text.toLowerCase() === lower) ?? null;
}

/**
 * The choices that contain `query`, those that start with it first; every choice for an empty query.
 * The order of the list is kept inside each of the two groups.
 */
export function filterChoices(choices, query) {
  const q = String(query ?? '').trim().toLowerCase();
  if (!q) return choices;
  const starts = [];
  const holds = [];
  for (const choice of choices) {
    const at = choice.text.toLowerCase().indexOf(q);
    if (at === 0) starts.push(choice);
    else if (at > 0) holds.push(choice);
  }
  return [...starts, ...holds];
}

const LIST_MAX_HEIGHT = 288;
const LIST_MIN_WIDTH = 200;
const GAP = 2;

/**
 * @param {{
 *   id?: string, inputClass?: string, placeholder?: string,
 *   getChoices: () => Choice[],            read each time the list opens or narrows, so it follows its source
 *   closed?: boolean,                      true: only a listed value can be taken; false: any text may be typed
 *   onPick?: (choice: Choice) => void,     a choice was taken from the list (the input already shows its text)
 *   onEnter?: (text: string) => void,      Enter with no highlighted choice
 *   onEscape?: () => void,                 Escape while the list is closed
 *   current?: () => string,                the text of the value in use, marked in the list
 *   listLabel?: string,                    accessible name of the list ("Optimizers")
 *   attrs?: object                         more attributes for the input (aria-label, data-fld, …)
 * }} options
 * @returns {{el: HTMLElement, input: HTMLInputElement, button: HTMLButtonElement, isOpen: () => boolean,
 *   open: () => void, close: () => void, destroy: () => void}}
 */
export function createCombobox({ id = uid('combo'), inputClass = '', placeholder = '', getChoices, closed = false, onPick = null, onEnter = null, onEscape = null, current = null, listLabel = 'Choices', attrs = {} }) {
  const listId = `${id}-list`;
  const hintId = `${id}-hint`;
  const input = h('input', {
    class: `input combo__input${inputClass ? ` ${inputClass}` : ''}`,
    id,
    type: 'text',
    role: 'combobox',
    autocomplete: 'off',
    spellcheck: 'false',
    placeholder: placeholder || null,
    'aria-autocomplete': 'list',
    'aria-expanded': 'false',
    'aria-controls': listId,
    'aria-haspopup': 'listbox',
    ...attrs,
  });
  // The hint is read with the field's own description, never instead of it.
  const describedBy = [input.getAttribute('aria-describedby'), hintId].filter(Boolean).join(' ');
  input.setAttribute('aria-describedby', describedBy);
  const hint = h('span', { class: 'visually-hidden', id: hintId }, closed ? 'Choose from the list. Typing narrows it.' : 'Choose from the list, or type another value.');
  // Not a tab stop: the arrow keys do the same from the field. It stays reachable for touch and screen readers.
  const button = h('button', { class: 'combo__button', type: 'button', tabindex: '-1', 'aria-label': 'Show the choices', 'aria-controls': listId, 'aria-expanded': 'false' }, h('span', { class: 'combo__chevron', 'aria-hidden': 'true' }));
  const list = h('div', { class: 'combo__list', id: listId, role: 'listbox', 'aria-label': listLabel, hidden: true });
  const el = h('span', { class: 'combo' }, input, button, list, hint);

  let open = false;
  let typed = false; // the person typed since the list opened or the value was shown: the list narrows
  let shown = []; // the choices drawn, in order
  let active = -1; // index into `shown` of the highlighted choice
  let watching = false;

  const optionId = (index) => `${listId}-${index}`;

  function place() {
    if (!open) return;
    const rect = input.getBoundingClientRect();
    const viewH = window.innerHeight;
    const viewW = window.innerWidth;
    const width = Math.min(Math.max(rect.width, LIST_MIN_WIDTH), viewW - 2 * GAP);
    const below = viewH - rect.bottom - GAP * 2;
    const above = rect.top - GAP * 2;
    const wanted = Math.min(LIST_MAX_HEIGHT, list.scrollHeight + 2);
    // Under the field when it fits there or has more room there; above it otherwise.
    const up = below < wanted && above > below;
    const height = Math.max(60, Math.min(LIST_MAX_HEIGHT, up ? above : below));
    list.style.width = `${width}px`;
    list.style.maxHeight = `${height}px`;
    list.style.left = `${Math.max(GAP, Math.min(rect.left, viewW - width - GAP))}px`;
    if (up) {
      list.style.top = '';
      list.style.bottom = `${viewH - rect.top + GAP}px`;
    } else {
      list.style.bottom = '';
      list.style.top = `${rect.bottom + GAP}px`;
    }
  }

  // The list is fixed to the viewport so no scrolling pane can clip it: it follows the field while open.
  const onMove = (event) => {
    if (event.type === 'scroll' && list.contains(event.target)) return;
    // A pane that scrolled the field out of what it shows takes the list with it: a list pointing at a
    // field nobody can see is closed (the typed text stays in the field).
    const pane = event.type === 'scroll' && event.target instanceof Element && event.target.contains(input) ? event.target.getBoundingClientRect() : null;
    const rect = input.getBoundingClientRect();
    if (rect.bottom < (pane ? pane.top : 0) || rect.top > (pane ? pane.bottom : window.innerHeight)) return hide();
    place();
  };
  function watch(on) {
    if (watching === on) return;
    watching = on;
    const method = on ? 'addEventListener' : 'removeEventListener';
    window[method]('resize', onMove);
    document[method]('scroll', onMove, true);
  }

  function setActive(index, reveal = true) {
    if (active >= 0) list.children[active]?.setAttribute('aria-selected', 'false');
    active = index >= 0 && index < shown.length ? index : -1;
    if (active < 0) {
      input.removeAttribute('aria-activedescendant');
      return;
    }
    const node = list.children[active];
    node.setAttribute('aria-selected', 'true');
    input.setAttribute('aria-activedescendant', node.id);
    if (!reveal) return;
    // Scroll only the list, never the page.
    const top = node.offsetTop;
    const bottom = top + node.offsetHeight;
    if (top < list.scrollTop) list.scrollTop = top;
    else if (bottom > list.scrollTop + list.clientHeight) list.scrollTop = bottom - list.clientHeight;
  }

  function draw() {
    const all = getChoices();
    const query = typed ? input.value : '';
    shown = filterChoices(all, query);
    const inUse = current ? String(current() ?? '').trim().toLowerCase() : '';
    const rows = shown.map((choice, index) =>
      h(
        'div',
        // In use: the choice the field shows; with an empty field, the choice that stands for no value.
        { class: `combo__option${(inUse ? choice.text.toLowerCase() === inUse : choice.unset) ? ' combo__option--current' : ''}`, role: 'option', id: optionId(index), 'aria-selected': 'false', dataset: { index: String(index) } },
        h('span', { class: 'combo__text' }, choice.text),
        choice.note ? h('span', { class: 'combo__note' }, choice.note) : null,
      ),
    );
    // What the list cannot offer is said in it: nothing listed at all, nothing matching, or text that will be kept as typed.
    const text = input.value.trim();
    let foot = null;
    if (!all.length) foot = closed ? 'No choices are listed.' : 'No choices are listed. Type your own.';
    else if (!shown.length) foot = closed ? 'No listed value matches.' : `No listed value matches. Enter keeps "${text}" as typed.`;
    else if (!closed && typed && text && !matchChoice(all, text)) foot = `Enter keeps "${text}" as typed.`;
    list.replaceChildren(...rows, ...(foot ? [h('div', { class: 'combo__foot', role: 'presentation' }, foot)] : []));
    active = -1;
    input.removeAttribute('aria-activedescendant');
  }

  /** @param {'current'|'first'|'last'|'none'} start which choice is highlighted once the list is drawn */
  function show(start = 'none') {
    const was = open;
    open = true;
    list.hidden = false;
    draw();
    input.setAttribute('aria-expanded', 'true');
    button.setAttribute('aria-expanded', 'true');
    el.classList.add('combo--open');
    watch(true);
    place();
    if (start === 'first') setActive(0);
    else if (start === 'last') setActive(shown.length - 1);
    else if (start === 'current') {
      const now = matchChoice(shown, current ? current() : input.value);
      if (now) setActive(shown.indexOf(now));
      else if (!was) list.scrollTop = 0;
    }
  }

  function hide() {
    if (!open) return;
    open = false;
    typed = false;
    list.hidden = true;
    active = -1;
    input.setAttribute('aria-expanded', 'false');
    button.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
    el.classList.remove('combo--open');
    watch(false);
  }

  function take(choice) {
    input.value = choice.text;
    hide();
    if (onPick) onPick(choice);
  }

  input.addEventListener('input', () => {
    typed = true;
    // With nothing listed there is nothing to narrow: typing stays plain typing. The list still opens
    // on request (arrow key, click) and says that it is empty.
    if (!open && !getChoices().length) return;
    show('none');
    // A closed list has nothing but its choices to offer, so the first match is ready for Enter.
    if (closed && shown.length && input.value.trim()) setActive(0);
  });

  input.addEventListener('keydown', (event) => {
    if (event.isComposing) return;
    const { key } = event;
    if (key === 'ArrowDown' || key === 'ArrowUp') {
      event.preventDefault();
      if (!open) {
        typed = false;
        show(event.altKey ? 'none' : 'current');
        if (!event.altKey && active < 0) setActive(key === 'ArrowDown' ? 0 : shown.length - 1);
        return;
      }
      if (!shown.length) return;
      if (active < 0) setActive(key === 'ArrowDown' ? 0 : shown.length - 1);
      else setActive((active + (key === 'ArrowDown' ? 1 : shown.length - 1)) % shown.length);
    } else if (open && (key === 'Home' || key === 'End') && shown.length && active >= 0) {
      // Only once a choice is highlighted: before that Home and End move the caret in the text.
      event.preventDefault();
      setActive(key === 'Home' ? 0 : shown.length - 1);
    } else if (key === 'Enter') {
      if (open && active >= 0) {
        event.preventDefault();
        take(shown[active]);
      } else {
        hide();
        // Without an onEnter the key keeps whatever it means around the field (a form's own Enter).
        if (onEnter) {
          event.preventDefault();
          onEnter(input.value);
        }
      }
    } else if (key === 'Escape') {
      if (open) {
        // The first Escape only closes the list: a dialog around the field stays open.
        event.preventDefault();
        event.stopPropagation();
        hide();
      } else if (onEscape) onEscape();
    } else if (key === 'Tab') hide();
  });

  // A click in the field opens the full list; a second click closes it again. `click` comes after the
  // browser has focused the field and placed the caret, so the list never opens for a field that is
  // not the focused one.
  input.addEventListener('click', () => {
    if (open && !typed) hide();
    else if (!open && document.activeElement === input) {
      typed = false;
      show('current');
    }
  });

  // The arrow and the list never take the focus from the field.
  button.addEventListener('mousedown', (event) => event.preventDefault());
  button.addEventListener('click', () => {
    if (open) hide();
    else {
      input.focus({ preventScroll: true });
      typed = false;
      show('current');
    }
  });
  list.addEventListener('mousedown', (event) => event.preventDefault());
  list.addEventListener('click', (event) => {
    const node = event.target instanceof Element ? event.target.closest('.combo__option') : null;
    if (!node || !list.contains(node)) return;
    const choice = shown[Number(node.dataset.index)];
    if (choice) take(choice);
  });
  list.addEventListener('mousemove', (event) => {
    const node = event.target instanceof Element ? event.target.closest('.combo__option') : null;
    if (!node) return;
    const index = Number(node.dataset.index);
    if (index !== active) setActive(index, false);
  });

  input.addEventListener('blur', hide);

  return {
    el,
    input,
    button,
    isOpen: () => open,
    open: () => show('current'),
    close: hide,
    destroy() {
      hide();
    },
  };
}
