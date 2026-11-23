// Field rows of the options form: one control per field kind (see schema.js fieldKind()).
// A row never writes YAML: it calls ctx.apply(path, value), which goes through recipe.applyEdits().
// Text inputs commit on blur or Enter (Escape restores); switches and chips commit at once.
// A value with known choices is a field with its choices in a list under it (components/combobox.js):
// a choice taken from the list commits at once, typed text like any text input.
// When an edit is refused the row shows the value from config again.

import { h, uid } from '../dom.js';
import { toast } from '../ui.js';
import { createCombobox, matchChoice, toChoices, EMPTY_TEXT } from '../components/combobox.js';
import { kindFromValue, isPlainObject } from './schema.js';

const PATH_KEY = /path|dataroot|pretrain|resume_state|engine_dir|plugin/;
const MORE_AT = 140;

/** Stable string for a config path; numbers and digit strings compare equal. */
export const pathKey = (path) => path.map(String).join('.');

// Folders under path.* that the trainer writes to or reads from (PathOptions in the schema).
const PATH_FOLDERS = new Set(['experiments_root', 'models', 'resume_models', 'training_states', 'log', 'visualization', 'results_root']);
const WEIGHT_EXTENSIONS = ['.safetensors', '.pth'];

/**
 * What the Browse button of an option picks, or null when the option is not a path.
 * Data paths are the ones the recipe check requires to exist: dataroot_*, meta_info,
 * path.pretrain_network_*, path.resume_state; plus the output folders under path.
 */
export function pathRole(path, key) {
  const k = String(key);
  const parent = path.length > 1 ? String(path[path.length - 2]) : '';
  if (/dataroot/.test(k)) return { mode: 'folder', title: `Choose the folder for ${k}`, extensions: undefined };
  // A list of image names (.txt), or the table itself for a dataset of numeric records (.csv).
  if (k === 'meta_info') return { mode: 'file', title: 'Choose the meta_info file', extensions: ['.txt', '.csv'] };
  if (path[0] === 'path' && path.length === 2) {
    if (/^pretrain_network_/.test(k)) return { mode: 'file', title: `Choose the weights for ${k}`, extensions: WEIGHT_EXTENSIONS, weights: true };
    // The optical flow weights RDUAL reads; not a network checkpoint, so no structure to inspect.
    if (k === 'neuflow_weights') return { mode: 'file', title: 'Choose the NeuFlow v2 weights', extensions: WEIGHT_EXTENSIONS };
    if (k === 'resume_state') return { mode: 'file', title: 'Choose the training state to resume', extensions: ['.state'] };
    if (PATH_FOLDERS.has(k)) return { mode: 'folder', title: `Choose the folder for ${k}`, extensions: undefined };
  }
  if (k === 'engine_dir' || (parent !== 'path' && /_dir$/.test(k))) return { mode: 'folder', title: `Choose the folder for ${k}`, extensions: undefined };
  return null;
}

// Parts written by other modules, loaded on first use. A part that is missing or broken resolves to
// null, and the buttons that need it stay out of the row.
const parts = {};
function part(name, exportName) {
  if (!(name in parts)) {
    parts[name] = import(`../components/${name}.js`).then(
      (mod) => (typeof mod?.[exportName] === 'function' ? mod : null),
      () => null,
    );
  }
  return parts[name];
}

function firstSentence(text) {
  const match = /^[\s\S]*?[.!?](\s|$)/.exec(text);
  return match ? match[0].trim() : text;
}

/** A description; long text (or `lead` with the text behind it) opens with "More". */
function describe(text, lead = '') {
  if (lead) {
    if (!text) return h('p', { class: 'fld__desc' }, lead);
    return describeParts(lead, text);
  }
  if (!text) return null;
  if (text.length <= MORE_AT) return h('p', { class: 'fld__desc' }, text);
  const head = firstSentence(text);
  return describeParts(head, text.slice(head.length).trim(), text);
}

function describeParts(head, rest, whole = null) {
  if (!rest) return h('p', { class: 'fld__desc' }, whole ?? head);
  const restEl = h('span', { hidden: true }, ` ${rest}`);
  const toggle = h(
    'button',
    {
      class: 'fld__more',
      type: 'button',
      'aria-expanded': 'false',
      onclick: () => {
        const open = restEl.hidden;
        restEl.hidden = !open;
        toggle.setAttribute('aria-expanded', String(open));
        toggle.textContent = open ? 'Less' : 'More';
      },
    },
    'More',
  );
  return h('p', { class: 'fld__desc' }, head, restEl, ' ', toggle);
}

// ───────── value <-> text ─────────

/**
 * A number as a field shows it: the shorter of the plain and the scientific spelling, so a small
 * learning rate reads 1e-4 and not 0.0001. A whole number is never respelled (500000 stays 500000).
 */
function numText(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '';
  const plain = String(value);
  if (Number.isInteger(value)) return plain;
  const sci = value.toExponential();
  return sci.length < plain.length ? sci : plain;
}

/**
 * Parse one number; `type` 'int' rejects fractions. Returns {ok, value, error}, and `floatLiteral`: the
 * text as typed when it is a fraction in scientific notation (1e-4), which the YAML then keeps as
 * `!!float 1e-4`. A whole number stays a plain number however it was typed (1e3 is written 1000), so an
 * option that needs an integer never becomes a float.
 */
export function parseNumber(text, type) {
  const t = text.trim();
  // Number('') is 0 and Number('0x10') is 16: only accept decimal and exponent notation.
  if (!/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t)) return { ok: false, error: type === 'int' ? 'Enter a whole number.' : 'Enter a number, for example 0.5 or 1e-4.' };
  const n = Number(t);
  if (!Number.isFinite(n)) return { ok: false, error: 'Enter a finite number.' };
  if (type === 'int' && !Number.isInteger(n)) return { ok: false, error: 'Enter a whole number.' };
  if (type !== 'int' && !Number.isInteger(n) && /e/i.test(t)) return { ok: true, value: n, floatLiteral: t };
  return { ok: true, value: n };
}

function listText(value) {
  if (Array.isArray(value)) return value.map((v) => (typeof v === 'string' ? v : JSON.stringify(v))).join(', ');
  if (typeof value === 'string') return value;
  return '';
}

function parseList(text, itemType) {
  const parts = text.split(',').map((p) => p.trim()).filter((p) => p !== '');
  if (itemType === 'string') return { ok: true, value: parts };
  const out = [];
  for (const part of parts) {
    const r = parseNumber(part, itemType);
    if (!r.ok) return { ok: false, error: `"${part}": ${r.error}` };
    out.push(r.value);
  }
  return { ok: true, value: out };
}

const jsonText = (value) => (value === undefined || value === null ? '' : JSON.stringify(value));

// ───────── the row ─────────

/**
 * @param {{path: Array<string|number>, key: string, kind: object, value: any, isSet: boolean,
 *   unknown?: boolean, label?: string, caption?: string, removable?: boolean,
 *   note?: (value: any, isSet: boolean) => string|null, itemLabel?: (i: number) => string}} spec
 *   caption: plain words shown before the option key (the key stays visible);
 *   removable: offer "Remove" (keys the schema does not describe, free-form options);
 *   note: a line under the control, recomputed whenever the row is updated;
 *   lead: description in the task's words, with the schema's description behind "More";
 *   pickTitle: title of the Browse dialog;
 *   choices: {getChoices: () => Choice[], listLabel?: string, unsetText?: string} for a text option whose
 *     usual values are known from elsewhere (the installed optimizers). Any other text can still be typed.
 *     A choice marked `unset` removes the option; `unsetText` is what the empty field then says.
 * @param {{apply: (path: Array, value: any, floatLiteral?: string) => Promise<any>, remove: (path: Array) => Promise<any>,
 *   edit: (edits: Array) => Promise<any>}} ctx
 *   apply's floatLiteral: the number as typed in scientific notation (1e-4), sent with the edit so the
 *   YAML keeps that spelling (`!!float 1e-4`); only passed when the person typed one.
 * @returns {{el: HTMLElement, path: Array, key: string, update: (value: any, isSet: boolean) => void,
 *   setIssues: (issues: Array) => void, focus: () => void, contains: (node: Node) => boolean}}
 */
export function createField(spec, ctx) {
  const { path } = spec;
  const kind = spec.kind ?? kindFromValue(spec.value);
  const id = uid('fld');
  const labelId = `${id}-label`;
  const key = pathKey(path);
  const tag = h('span', { class: 'fld__tag' });
  const localError = h('p', { class: 'field__error', role: 'alert', hidden: true });
  const issueSlot = h('div', { class: 'fld__issues' });
  const noteEl = spec.note ? h('p', { class: 'fld__note', hidden: true }) : null;
  let current = { value: spec.value, isSet: spec.isSet };

  const label = h(
    'div',
    { class: 'fld__label' },
    h(
      'div',
      { class: 'fld__head' },
      spec.caption ? h('span', { class: 'fld__caption' }, spec.caption) : null,
      h('label', { class: 'fld__key', for: id, id: labelId }, spec.label ?? spec.key),
      tag,
    ),
    // `lead`: the task's own words for the option; the schema's text then waits behind "More".
    describe(kind.description, spec.lead ?? ''),
  );

  const showError = (message) => {
    localError.textContent = message ?? '';
    localError.hidden = !message;
    for (const input of control.inputs) {
      if (message) input.setAttribute('aria-invalid', 'true');
      else if (!input.dataset.issue) input.removeAttribute('aria-invalid');
    }
  };

  const refused = (error) => {
    toast(`${spec.key} was not changed: ${error?.message ?? error}`, { kind: 'err' });
    control.show(current.value, current.isSet);
  };
  // floatLiteral: a number as the person typed it in scientific notation, kept as that spelling in the YAML.
  const commit = (value, floatLiteral) => {
    showError(null);
    return Promise.resolve(typeof floatLiteral === 'string' ? ctx.apply(path, value, floatLiteral) : ctx.apply(path, value)).catch(refused);
  };
  // Several edits at once (list items: remove one, set one); the backend applies them in order.
  const edit = (edits) => {
    showError(null);
    return Promise.resolve(ctx.edit(edits)).catch(refused);
  };
  // "Use default" and "Remove" delete the key, so the YAML no longer names it.
  const drop = () => {
    showError(null);
    return Promise.resolve(ctx.remove(path)).catch(refused);
  };

  // picks: what the form says Browse picks here ('file' for a row of archive files), for the control's own hint.
  const control = buildControl(kind, { id, labelId, key: spec.key, path, commit, edit, drop, showError, getCurrent: () => current, itemLabel: spec.itemLabel, listEditor: spec.listEditor, choices: spec.choices ?? null, picks: spec.browse?.mode ?? null });
  // When the default already is none, "Use default" covers it; an explicit none is offered only
  // where the schema allows null and the default is something else.
  const defaultIsNone = kind.hasDefault && kind.default === null;
  const noneButton = kind.nullable && !defaultIsNone
    ? h('button', { class: 'btn btn--quiet btn--sm fld__none', type: 'button', onclick: () => commit(null) }, 'Set to none')
    : null;
  const removable = spec.removable || spec.unknown;
  const dropButton =
    removable || kind.hasDefault
      ? h('button', { class: 'btn btn--quiet btn--sm fld__none', type: 'button', title: removable ? `Remove ${spec.key} from the recipe` : `Remove ${spec.key} from the recipe, so the default applies`, onclick: drop }, removable ? 'Remove' : 'Use default')
      : null;

  // Browse (and Inspect structure for weight files) on options that hold a path.
  // spec.browse: a role the form knows from the task (annotation_file), else the option name decides.
  const role = ['string', 'stringOrList', 'any'].includes(kind.kind) ? spec.browse ?? pathRole(path, spec.key) : null;
  let browseButton = null;
  let inspectButton = null;
  const textValues = () => {
    const v = current.isSet ? current.value : kind.default;
    return (Array.isArray(v) ? v : [v]).filter((x) => typeof x === 'string' && x);
  };
  if (role) {
    browseButton = h(
      'button',
      {
        class: 'btn btn--sm',
        type: 'button',
        hidden: true,
        onclick: async () => {
          const mod = await part('path-picker', 'pickPath');
          if (!mod) return;
          const values = textValues();
          const picked = await mod.pickPath({
            title: spec.pickTitle ?? role.title,
            mode: role.mode,
            start: values[0] || undefined,
            extensions: role.extensions,
            confirmLabel: role.mode === 'folder' ? 'Use this folder' : 'Use this file',
          });
          if (typeof picked !== 'string' || !picked) return;
          // A list of several folders keeps them all: the picked one is added to the list.
          const value = Array.isArray(current.value) && current.isSet && current.value.length > 1 ? [...current.value, picked] : kind.kind === 'stringOrList' && Array.isArray(current.value) ? [picked] : picked;
          commit(value);
        },
      },
      'Browse',
    );
    part('path-picker', 'pickPath').then((mod) => {
      if (mod) browseButton.hidden = false;
    });
    if (role.weights) {
      inspectButton = h(
        'button',
        {
          class: 'btn btn--sm',
          type: 'button',
          hidden: true,
          onclick: async () => {
            const mod = await part('weights', 'openWeightsDialog');
            const file = textValues()[0];
            if (mod && file) mod.openWeightsDialog({ path: file });
          },
        },
        'Inspect structure',
      );
    }
  }
  let weightsReady = false;
  if (inspectButton) {
    part('weights', 'openWeightsDialog').then((mod) => {
      weightsReady = Boolean(mod);
      paint();
    });
  }

  const el = h(
    'div',
    { class: 'fld', dataset: { path: key } },
    label,
    h(
      'div',
      { class: 'fld__control' },
      h('div', { class: 'fld__line' }, control.el, browseButton, inspectButton, noneButton || dropButton ? h('span', { class: 'fld__acts' }, noneButton, dropButton) : null),
      noteEl,
      localError,
      issueSlot,
    ),
  );
  for (const input of control.inputs) input.dataset.fld = key;

  function paint() {
    const effective = current.isSet ? current.value : kind.default;
    el.classList.toggle('fld--default', !current.isSet);
    tag.textContent = spec.unknown ? 'Not in the schema' : current.isSet ? 'Set' : kind.hasDefault ? 'Default' : 'Not set';
    tag.className = `fld__tag${spec.unknown ? ' fld__tag--unknown' : ''}`;
    if (noneButton) noneButton.hidden = effective === null || effective === undefined;
    if (dropButton) dropButton.hidden = !current.isSet;
    if (browseButton) browseButton.textContent = Array.isArray(current.value) && current.isSet && current.value.length > 1 ? (role.mode === 'file' ? 'Add a file' : 'Add a folder') : 'Browse';
    if (inspectButton) inspectButton.hidden = !weightsReady || !textValues().some((v) => /\.safetensors$/i.test(v));
    if (noteEl) {
      // A note is a sentence, or {text, warn: true} for one that says something is missing.
      const said = spec.note(current.value, current.isSet);
      const text = typeof said === 'string' ? said : said?.text ?? '';
      const cls = said && typeof said === 'object' && said.warn ? 'fld__warn' : 'fld__note';
      if (noteEl.className !== cls) noteEl.className = cls;
      if (noteEl.textContent !== text) noteEl.textContent = text;
      noteEl.hidden = !text;
    }
  }

  const focused = () => (control.holdsFocus ? control.holdsFocus() : control.el.contains(document.activeElement));
  // A value that changed while the control held focus is drawn once focus leaves it, unless the
  // person left typed text there (the blur commits that text, and its result redraws the row).
  let stale = false;
  control.el.addEventListener('focusout', (event) => {
    if (!stale || control.el.contains(event.relatedTarget)) return;
    queueMicrotask(() => {
      if (!stale || focused() || (control.isDirty && control.isDirty())) return;
      stale = false;
      control.show(current.value, current.isSet);
    });
  });

  paint();
  control.show(current.value, current.isSet);

  return {
    el,
    path,
    key: spec.key,
    pathKey: key,
    update(value, isSet) {
      const same = current.isSet === isSet && JSON.stringify(current.value) === JSON.stringify(value);
      current = { value, isSet };
      paint();
      if (same) return;
      // Never move the caret of the control being typed in: the new value is shown when focus leaves.
      if (focused()) stale = true;
      else {
        stale = false;
        showError(null);
        control.show(value, isSet);
      }
    },
    setIssues(issues) {
      issueSlot.replaceChildren(...issueNodes(issues));
      const hasError = issues.some((issue) => issue.level === 'error');
      for (const input of control.inputs) {
        if (hasError) {
          input.dataset.issue = '1';
          input.setAttribute('aria-invalid', 'true');
        } else {
          delete input.dataset.issue;
          if (localError.hidden) input.removeAttribute('aria-invalid');
        }
      }
    },
    focus() {
      const target = control.inputs[0];
      if (target) target.focus({ preventScroll: true });
    },
  };
}

// ───────── controls ─────────
// Each returns {el, inputs: [focusable controls], show(value, isSet)}.

function buildControl(kind, env) {
  // Edited item by item: lists the caller asks for (class names), and lists of texts whose usual values
  // the schema names. Their order and repeats matter (a list of weights often runs beside them).
  if (kind.kind === 'list' && (env.listEditor || (kind.itemType === 'string' && kind.itemSuggestions?.length))) return listControl(kind, env);
  switch (kind.kind) {
    case 'objectList':
      return tableControl(kind, env);
    case 'bool':
      return switchControl(kind, env);
    case 'enum':
      // The schema lists every value the option takes: only a listed one is written.
      return comboControl(kind, env, {
        closed: true,
        getChoices: () => toChoices(kind.options.map((o) => ({ value: o, note: kind.optionNote ? kind.optionNote(o) : null }))),
      });
    case 'int':
    case 'number':
      return textControl(kind, env, {
        num: true,
        // Scientific notation needs the letter e, which a phone's number pad does not have.
        sci: kind.kind === 'number',
        format: numText,
        parse: (text) => parseNumber(text, kind.kind),
      });
    case 'literalOrNumber':
      // A number, or one of a few words: the words are offered, a number is typed.
      return comboControl(kind, env, {
        closed: false,
        getChoices: () => toChoices(kind.literals),
        format: (v) => (typeof v === 'string' ? v : numText(v)),
        parse: (text) => {
          const t = text.trim();
          if (kind.literals.includes(t)) return { ok: true, value: t };
          const r = parseNumber(t, kind.numType);
          return r.ok ? r : { ok: false, error: `Enter ${kind.numType === 'int' ? 'a whole number' : 'a number'} or ${kind.literals.map((l) => `"${l}"`).join(' or ')}.` };
        },
        placeholderHint: kind.literals.join(', '),
      });
    case 'string': {
      const format = (v) => (typeof v === 'string' ? v : v === null || v === undefined ? '' : String(v));
      // Usual values known from the schema (a free string that also lists values) or from the caller
      // (what the installed trainer offers): offered in a list, and any other text can be typed.
      const listed = env.choices?.getChoices ?? (kind.suggestions && kind.suggestions.length ? () => toChoices(kind.suggestions) : null);
      if (listed) return comboControl(kind, env, { closed: false, getChoices: listed, format, listLabel: env.choices?.listLabel, unsetText: env.choices?.unsetText });
      return textControl(kind, env, {
        path: PATH_KEY.test(env.key),
        format,
        parse: (text) => ({ ok: true, value: text }),
      });
    }
    case 'stringOrList':
      return textControl(kind, env, {
        path: true,
        format: listText,
        parse: (text) => parseList(text, 'string'),
        hint: `Separate several ${env.picks === 'file' ? 'files' : 'folders'} with commas.`,
      });
    case 'list':
      return textControl(kind, env, {
        format: listText,
        parse: (text) => parseList(text, kind.itemType),
        hint: 'Separate values with commas.',
      });
    case 'pair':
      return pairControl(kind, env);
    case 'enumList':
      // Every value the schema allows, item by item: the same list editor, taking only listed values.
      return listControl({ ...kind, itemType: 'string', itemSuggestions: kind.options, itemsClosed: true }, env);
    case 'any':
      // No type in the schema: follow the value the recipe has.
      return buildControl({ ...kind, ...kindFromValue(env.getCurrent().value ?? kind.default), description: kind.description }, env);
    case 'loose':
      // An option known by name only (a parameter listed without a value to start from): read as
      // "Add option" reads a value, so a number, a list and plain text can all be typed.
      return textControl(kind, env, {
        path: true,
        format: (v) => (typeof v === 'string' ? v : JSON.stringify(v)),
        parse: (text) => {
          try {
            return { ok: true, value: JSON.parse(text) };
          } catch {
            return { ok: true, value: text.trim() };
          }
        },
        hint: 'Read as JSON; plain text is taken as a string.',
      });
    default:
      return textControl(kind, env, {
        path: true,
        format: jsonText,
        parse: (text) => {
          try {
            return { ok: true, value: JSON.parse(text) };
          } catch (error) {
            return { ok: false, error: `This is not valid JSON: ${error.message}` };
          }
        },
        hint: 'Written as JSON, for example [[0, 2], [0, 2]].',
      });
  }
}

/** Value shown by a control: the set value, else the schema default (drawn lighter by the row). */
const shown = (kind, value, isSet) => (isSet ? value : kind.default);

function switchControl(kind, env) {
  const input = h('input', {
    type: 'checkbox',
    id: env.id,
    onchange: () => env.commit(input.checked),
  });
  const state = h('span', { class: 'fld__state' });
  const el = h('label', { class: 'switch' }, input, h('span', { class: 'switch__track', 'aria-hidden': 'true' }), state);
  return {
    el,
    inputs: [input],
    show(value, isSet) {
      const v = shown(kind, value, isSet);
      input.checked = v === true;
      state.textContent = v === true ? 'On' : v === false ? 'Off' : 'Not set';
    },
  };
}

/**
 * A value with known choices: a field with the choices in a list under it.
 * A closed list (opts.closed) takes only a listed value: typing narrows the list, and text that is
 * not listed is refused here, as the service would refuse it. An open list takes any text: a listed
 * value is written as it is listed (with its type: a number stays a number), any other spelling
 * exactly as typed.
 * A choice marked `unset` takes the option out of the recipe; the empty text, where it is a choice, is
 * shown and typed as "".
 * opts: {closed, getChoices: () => Choice[], format?: (value) => string, parse?: (text) => {ok, value, error},
 *   placeholderHint?: string, listLabel?: string, unsetText?: string (what the empty field says)}
 */
function comboControl(kind, env, opts) {
  let shownText = '';
  let inFlight = 0;
  const plainText = opts.format ?? ((v) => (typeof v === 'string' ? v : JSON.stringify(v)));
  // The empty text is shown as "" where it is one of the choices, so it cannot be mistaken for "not set".
  const emptyListed = () => opts.getChoices().some((choice) => choice.value === '');
  const textOf = (v) => (v === '' && emptyListed() ? EMPTY_TEXT : plainText(v));
  const send = (value, text) => {
    shownText = text;
    input.value = text;
    inFlight += 1;
    Promise.resolve(env.commit(value)).finally(() => {
      inFlight -= 1;
    });
  };
  // The choice that is no value: the option is taken out of the recipe (or set to none where it cannot be).
  const unset = () => {
    shownText = '';
    input.value = '';
    if (!env.getCurrent().isSet) return env.showError(null);
    if (env.drop) env.drop();
    else send(null, '');
  };
  const take = (choice) => {
    env.showError(null);
    if (choice.unset) return unset();
    // The list already shows the choice in the field, so only a different one is written.
    if (choice.text === shownText) {
      input.value = shownText;
      return;
    }
    send(choice.value, choice.text);
  };
  const tryCommit = () => {
    if (input.value === shownText) {
      env.showError(null);
      return;
    }
    const text = input.value.trim();
    const choices = opts.getChoices();
    if (text === '') {
      // Emptying an option whose default is none removes the key instead of writing null.
      if (kind.nullable && kind.hasDefault && kind.default === null && env.drop) {
        shownText = input.value;
        env.drop();
        return;
      }
      if (choices.some((choice) => choice.unset)) return unset();
      if (kind.nullable) return send(null, '');
      if (opts.closed || kind.kind !== 'string') return env.showError(opts.closed ? 'Choose one of the listed values.' : 'Enter a value.');
    }
    if (opts.closed) {
      const hit = matchChoice(choices, text);
      if (!hit) return env.showError(`"${text}" is not one of the listed values. Choose one from the list.`);
      return take(hit);
    }
    // The listed spelling keeps the listed value; any other text is the person's own and is kept as typed.
    const exact = choices.find((choice) => choice.text === text);
    if (exact) return take(exact);
    const result = opts.parse ? opts.parse(text) : { ok: true, value: text };
    if (!result.ok) return env.showError(result.error);
    return send(result.value, result.value === '' ? textOf('') : text);
  };
  const combo = createCombobox({
    id: env.id,
    inputClass: 'input--path',
    closed: opts.closed,
    getChoices: opts.getChoices,
    current: () => shownText,
    listLabel: opts.listLabel ?? `Values of ${env.key}`,
    onPick: take,
    onEnter: tryCommit,
    onEscape: () => {
      input.value = shownText;
      env.showError(null);
    },
  });
  const { input } = combo;
  input.addEventListener('blur', tryCommit);
  return {
    el: combo.el,
    inputs: [input],
    // Typed text not committed yet, or a commit still on its way: the row is not redrawn under it.
    isDirty: () => input.value !== shownText || inFlight > 0,
    show(value, isSet) {
      const v = shown(kind, value, isSet);
      shownText = v === undefined || v === null ? '' : textOf(v);
      input.value = shownText;
      combo.close();
      input.placeholder = opts.unsetText ?? (v === null && (isSet || kind.hasDefault) ? 'None' : 'Not set');
      if (opts.placeholderHint && !shownText) input.placeholder = opts.placeholderHint;
    },
  };
}

function textControl(kind, env, opts) {
  const input = h('input', {
    class: `input${opts.num ? ' input--num' : ''}${opts.path ? ' input--path' : ''}`,
    id: env.id,
    type: 'text',
    autocomplete: 'off',
    spellcheck: 'false',
    inputmode: opts.num ? (opts.sci ? 'text' : 'decimal') : null,
    autocapitalize: opts.sci ? 'off' : null,
  });
  let shownText = '';
  let typed = false; // the person typed in the field since it was last drawn or written
  input.addEventListener('input', () => {
    typed = true;
  });
  const tryCommit = () => {
    const text = input.value;
    if (text === shownText) {
      // Leaving the field as it was drawn writes nothing. Scientific notation the person typed is
      // written even when the field shows the same text: 0.0001 is drawn as 1e-4 whatever the YAML
      // spells, and typing 1e-4 over it asks for that spelling.
      const again = typed && opts.sci ? opts.parse(text) : null;
      if (!again?.ok || !again.floatLiteral) {
        env.showError(null);
        return;
      }
    }
    let result;
    // Emptying an option whose default is none removes the key instead of writing null.
    if (text.trim() === '' && kind.nullable && kind.hasDefault && kind.default === null && env.drop) {
      shownText = text;
      env.drop();
      return;
    }
    if (text.trim() === '' && kind.nullable) result = { ok: true, value: null };
    else if (text.trim() === '' && kind.kind !== 'string') result = { ok: false, error: 'Enter a value.' };
    else result = opts.parse(text);
    if (!result.ok) {
      env.showError(result.error);
      return;
    }
    shownText = text;
    typed = false;
    inFlight += 1;
    Promise.resolve(env.commit(result.value, result.floatLiteral)).finally(() => {
      inFlight -= 1;
    });
  };
  let inFlight = 0;
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      tryCommit();
    } else if (event.key === 'Escape') {
      input.value = shownText;
      typed = false;
      env.showError(null);
    }
  });
  input.addEventListener('blur', tryCommit);
  const hint = opts.hint ? h('span', { class: 'visually-hidden', id: `${env.id}-hint` }, opts.hint) : null;
  if (hint) input.setAttribute('aria-describedby', `${env.id}-hint`);
  return {
    el: h('span', { class: 'fld__text' }, input, hint),
    inputs: [input],
    // Typed text not committed yet, or a commit still on its way: the row is not redrawn under it.
    isDirty: () => input.value !== shownText || inFlight > 0,
    show(value, isSet) {
      const v = shown(kind, value, isSet);
      shownText = v === undefined || v === null ? '' : opts.format(v);
      input.value = shownText;
      typed = false;
      input.placeholder = v === null && (isSet || kind.hasDefault) ? 'None' : 'Not set';
      if (opts.placeholderHint && !shownText) input.placeholder = opts.placeholderHint;
      if (opts.hint) input.title = opts.hint;
    },
  };
}

function pairControl(kind, env) {
  const make = (suffix) =>
    h('input', {
      class: 'input input--num',
      id: suffix ? `${env.id}-${suffix}` : env.id,
      type: 'text',
      inputmode: 'decimal',
      autocomplete: 'off',
      spellcheck: 'false',
    });
  const from = make('');
  const to = make('to');
  let shownTexts = ['', ''];
  const tryCommit = () => {
    if (from.value === shownTexts[0] && to.value === shownTexts[1]) return;
    if (from.value.trim() === '' && to.value.trim() === '' && kind.nullable) {
      env.commit(null);
      return;
    }
    const a = parseNumber(from.value, kind.itemType);
    const b = parseNumber(to.value, kind.itemType);
    if (!a.ok || !b.ok) {
      env.showError(`${!a.ok ? `From: ${a.error}` : ''}${!a.ok && !b.ok ? ' ' : ''}${!b.ok ? `To: ${b.error}` : ''}`);
      return;
    }
    shownTexts = [from.value, to.value];
    env.commit([a.value, b.value]);
  };
  const container = h(
    'span',
    {
      class: 'fld__pair',
      // Commit when focus leaves the pair, not when it moves from "from" to "to".
      onfocusout: (event) => {
        if (!container.contains(event.relatedTarget)) tryCommit();
      },
    },
    h('span', { class: 'visually-hidden', id: `${env.id}-fromlabel` }, 'from'),
    from,
    h('span', { class: 'fld__dash', 'aria-hidden': 'true' }, 'to'),
    h('label', { class: 'visually-hidden', for: `${env.id}-to` }, `${env.key} to`),
    to,
  );
  from.setAttribute('aria-describedby', `${env.id}-fromlabel`);
  for (const input of [from, to]) {
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        tryCommit();
      } else if (event.key === 'Escape') {
        from.value = shownTexts[0];
        to.value = shownTexts[1];
        env.showError(null);
      }
    });
  }
  return {
    el: container,
    inputs: [from, to],
    show(value, isSet) {
      const v = shown(kind, value, isSet);
      const pair = Array.isArray(v) ? v : [];
      shownTexts = [numText(pair[0]), numText(pair[1])];
      from.value = shownTexts[0];
      to.value = shownTexts[1];
      from.placeholder = v === null ? 'None' : 'Not set';
      to.placeholder = from.placeholder;
    },
  };
}

/**
 * A list of texts edited item by item (task_options.class_names, compress_algorithms): change, add,
 * move, remove. Changing or removing one item touches only that item in the YAML; adding or moving sets
 * the whole list, which the backend writes on one line. The order and repeats are the recipe's: nothing
 * is sorted or merged, because a list of weights often runs beside this one, position by position.
 * With `kind.itemSuggestions` every item (and the add box) offers those values in a list; `itemsClosed`
 * makes them the only values taken (a schema enum). Items that are not text (a number written without
 * quotes) are shown as they are and stay untouched until they are edited.
 */
function listControl(kind, env) {
  const suggestions = Array.isArray(kind.itemSuggestions) ? kind.itemSuggestions : null;
  const closed = Boolean(suggestions && kind.itemsClosed);
  const getChoices = () => toChoices(suggestions ?? []);
  const what = suggestions ? 'value' : 'name';
  const rows = h('ol', { class: 'fld__list' });
  const hint = h('p', { class: 'fld__note', hidden: true }, 'Adding or moving rewrites this list on one line; comments inside it are lost.');
  const nameOf = (i) => (env.itemLabel ? env.itemLabel(i) : `Item ${i}`);
  let refocus = null; // {act, index} of the button that was pressed, so focus survives the redraw
  let combos = []; // the item fields' lists, closed before their rows are replaced
  // What the rows show: the recipe's list, or the default's items while the option is not set.
  const values = () => {
    const c = env.getCurrent();
    if (c.isSet) return Array.isArray(c.value) ? c.value : [];
    return Array.isArray(kind.default) ? kind.default : [];
  };
  // An item of a list the recipe does not have yet cannot be addressed on its own: the whole list is written.
  const whole = () => !env.getCurrent().isSet || !Array.isArray(env.getCurrent().value);
  /** The text as it is written: a listed spelling for a closed list, else exactly as typed; null when refused. */
  const accept = (text) => {
    if (!closed) return text;
    const hit = matchChoice(getChoices(), text);
    if (hit) return hit.value;
    env.showError(`"${text}" is not one of the listed values. Choose one from the list.`);
    return null;
  };
  const add = () => {
    const text = addInput.value.trim();
    if (!text) return env.showError(`Enter a ${what} to add.`);
    const value = accept(text);
    if (value === null) return;
    addInput.value = '';
    env.commit([...values(), value]);
  };
  const addCombo = suggestions
    ? createCombobox({ id: env.id, inputClass: 'input--path', placeholder: 'Value to add', closed, getChoices, listLabel: `Values of ${env.key}`, onEnter: add })
    : null;
  const addInput = addCombo ? addCombo.input : h('input', { class: 'input input--path', id: env.id, type: 'text', autocomplete: 'off', spellcheck: 'false', placeholder: 'Name to add' });
  const addButton = h('button', { class: 'btn btn--sm', type: 'button' }, 'Add');
  addButton.addEventListener('click', add);
  if (!addCombo) {
    addInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        add();
      }
    });
  }
  const itemRow = (item, i, all) => {
    const shownText = typeof item === 'string' ? item : JSON.stringify(item);
    const attrs = { 'aria-label': `${env.key}, ${nameOf(i)}`, 'data-fld': `${pathKey(env.path)}.${i}`, 'data-index': String(i), 'data-shown': shownText };
    const save = () => {
      const text = input.value.trim();
      if (text === shownText) return;
      if (!text) {
        input.value = shownText;
        return;
      }
      const value = accept(text);
      if (value === null) return;
      env.showError(null);
      if (whole()) env.commit(all.map((x, j) => (j === i ? value : x)));
      else env.edit([{ path: [...env.path, i], value }]);
    };
    const combo = suggestions
      ? createCombobox({
          inputClass: 'input--path',
          closed,
          getChoices,
          current: () => shownText,
          listLabel: `Values of ${env.key}`,
          attrs,
          onPick: save,
          onEnter: save,
          onEscape: () => {
            input.value = shownText;
            env.showError(null);
          },
        })
      : null;
    if (combo) combos.push(combo);
    const input = combo ? combo.input : h('input', { class: 'input input--path', type: 'text', autocomplete: 'off', spellcheck: 'false', ...attrs });
    input.value = shownText;
    input.addEventListener('blur', save);
    if (!combo) {
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          save();
        } else if (event.key === 'Escape') input.value = shownText;
      });
    }
    const move = (to, act) => {
      const next = [...all];
      const [moved] = next.splice(i, 1);
      next.splice(to, 0, moved);
      refocus = { act, index: to };
      env.commit(next);
    };
    const button = (act, text, disabled, onclick) =>
      h('button', { class: 'btn btn--sm btn--quiet', type: 'button', disabled, dataset: { act, index: String(i) }, 'aria-label': `${text} ${nameOf(i)}`, onclick }, text);
    return h(
      'li',
      { class: 'fld__item' },
      h('span', { class: 'fld__itemno mono', 'aria-hidden': 'true' }, nameOf(i)),
      combo ? combo.el : input,
      // One group, so the three stay together: beside the field where there is room, under it where not.
      h(
        'span',
        { class: 'fld__itemacts' },
        button('up', 'Move up', i === 0, () => move(i - 1, 'up')),
        button('down', 'Move down', i === all.length - 1, () => move(i + 1, 'down')),
        button('remove', 'Remove', false, () => {
          refocus = { act: 'remove', index: Math.min(i, all.length - 2) };
          if (whole()) env.commit(all.filter((_, j) => j !== i));
          else env.edit([{ path: [...env.path, i], remove: true }]);
        }),
      ),
    );
  };
  const el = h('div', { class: 'fld__listbox' }, rows, h('div', { class: 'fld__line' }, addCombo ? addCombo.el : addInput, addButton), hint);
  let drawn = null; // JSON of the list the rows show
  return {
    el,
    inputs: [addInput],
    // The rows always follow the recipe, also while focus is inside the list: only the rows are
    // redrawn (the add box stays), and focus is put back where it was or where the edit leads.
    holdsFocus: () => false,
    show() {
      const all = values();
      const json = JSON.stringify(all);
      const active = document.activeElement;
      let keep = null;
      if (json !== drawn) {
        if (active && rows.contains(active)) {
          keep = active.tagName === 'INPUT'
            ? { input: true, index: Number(active.dataset.index), shown: active.dataset.shown, value: active.value, start: active.selectionStart, end: active.selectionEnd }
            : { act: active.dataset.act, index: Number(active.dataset.index) };
        }
        drawn = json;
        for (const combo of combos) combo.close();
        combos = [];
        rows.replaceChildren(...all.map((item, i) => itemRow(item, i, all)));
        hint.hidden = all.length === 0;
      }
      const goal = refocus ?? (keep && !keep.input ? keep : null);
      refocus = null;
      if (goal) {
        const { act, index } = goal;
        const row = rows.children[Math.min(index, rows.children.length - 1)];
        // The same button on the row now at that place; a disabled one (first or last row) gives way
        // to the row's other move button, then its name.
        const target = !row ? addInput : row.querySelector(`[data-act="${act}"]:not(:disabled)`) ?? row.querySelector('[data-act]:not(:disabled)') ?? row.querySelector('input');
        target.focus({ preventScroll: true });
      } else if (keep?.input) {
        // A name being typed: the row at the same place keeps the caret, and the typed text when
        // that row still holds the name the typing started from.
        const input = rows.children[keep.index]?.querySelector('input');
        if (input) {
          if (input.dataset.shown === keep.shown) input.value = keep.value;
          input.focus({ preventScroll: true });
          try {
            input.setSelectionRange(keep.start, keep.end);
          } catch {
            // no selection on this control
          }
        } else addInput.focus({ preventScroll: true });
      }
    },
  };
}

/**
 * A list of small records (task_options.categories: [{id, name}]) as a table: change a cell, remove a
 * row, add a row. A cell or a removal touches only that entry in the YAML; adding sets the whole list.
 * Like listControl, the rows always follow the recipe and focus is put back after a redraw.
 */
function tableControl(kind, env) {
  const cols = kind.columns;
  const parseCell = (col, text) => {
    const t = text.trim();
    if (col.type === 'string') return t ? { ok: true, value: t } : { ok: false, error: `Enter the ${col.key}.` };
    return parseNumber(t, col.type);
  };
  const rowStyle = { '--cols': String(cols.length) };
  const cellText = (v) => (v === undefined || v === null ? '' : typeof v === 'string' ? v : String(v));
  const head = h('div', { class: 'fld__tablerow fld__tablehead', 'aria-hidden': 'true', style: rowStyle }, cols.map((c) => h('span', { class: 'mono' }, c.key)), h('span'));
  const rows = h('div', { class: 'fld__tablebody' });
  const addInputs = cols.map((c, i) => h('input', { class: `input${c.type === 'string' ? '' : ' input--num'}`, id: i === 0 ? env.id : `${env.id}-${c.key}`, type: 'text', autocomplete: 'off', spellcheck: 'false', 'aria-label': `New ${c.key}`, inputmode: c.type === 'string' ? null : 'numeric' }));
  const addButton = h('button', { class: 'btn btn--sm', type: 'button' }, 'Add');
  const hint = h('p', { class: 'fld__note', hidden: true }, 'Adding rewrites this list on one line; comments inside it are lost.');
  let drawn = null;
  let refocus = null;
  const values = () => {
    const c = env.getCurrent();
    return c.isSet && Array.isArray(c.value) ? c.value : [];
  };
  const add = () => {
    const item = {};
    for (const [i, col] of cols.entries()) {
      const r = parseCell(col, addInputs[i].value);
      if (!r.ok) {
        env.showError(`${col.key}: ${r.error}`);
        addInputs[i].focus();
        return;
      }
      item[col.key] = r.value;
    }
    for (const input of addInputs) input.value = '';
    addInputs[0].focus();
    env.commit([...values(), item]);
  };
  addButton.addEventListener('click', add);
  for (const input of addInputs) {
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        add();
      }
    });
  }
  const rowEl = (item, i) => {
    const record = item && typeof item === 'object' && !Array.isArray(item) ? item : {};
    const cells = cols.map((col) => {
      const shownText = cellText(record[col.key]);
      const input = h('input', {
        class: `input${col.type === 'string' ? '' : ' input--num'}`,
        type: 'text',
        autocomplete: 'off',
        spellcheck: 'false',
        inputmode: col.type === 'string' ? null : 'numeric',
        'aria-label': `${col.key}, row ${i + 1}`,
        dataset: { fld: `${pathKey(env.path)}.${i}.${col.key}`, index: String(i), col: col.key, shown: shownText },
      });
      input.value = shownText;
      const save = () => {
        if (input.value === shownText) return;
        const r = parseCell(col, input.value);
        if (!r.ok) return env.showError(`${col.key} in row ${i + 1}: ${r.error}`);
        env.edit([{ path: [...env.path, i, col.key], value: r.value }]);
      };
      input.addEventListener('blur', save);
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          save();
        } else if (event.key === 'Escape') input.value = shownText;
      });
      return input;
    });
    const remove = h(
      'button',
      {
        class: 'btn btn--sm btn--quiet',
        type: 'button',
        dataset: { act: 'remove', index: String(i) },
        'aria-label': `Remove row ${i + 1}`,
        onclick: () => {
          refocus = { index: Math.min(i, values().length - 2) };
          env.edit([{ path: [...env.path, i], remove: true }]);
        },
      },
      'Remove',
    );
    return h('div', { class: 'fld__tablerow', style: rowStyle }, cells, remove);
  };
  const el = h('div', { class: 'fld__listbox' }, head, rows, h('div', { class: 'fld__tablerow', style: rowStyle }, addInputs, addButton), hint);
  return {
    el,
    inputs: [addInputs[0]],
    holdsFocus: () => false,
    show() {
      const all = values();
      const json = JSON.stringify(all);
      const active = document.activeElement;
      let keep = null;
      if (json !== drawn) {
        if (active && rows.contains(active) && active.tagName === 'INPUT') keep = { index: Number(active.dataset.index), col: active.dataset.col, shown: active.dataset.shown, value: active.value };
        drawn = json;
        rows.replaceChildren(...all.map(rowEl));
        head.hidden = all.length === 0;
        hint.hidden = all.length === 0;
      }
      if (refocus) {
        const row = rows.children[refocus.index];
        refocus = null;
        (row?.querySelector('[data-act]') ?? addInputs[0]).focus({ preventScroll: true });
      } else if (keep) {
        const row = rows.children[keep.index];
        const input = row ? [...row.children].find((x) => x.dataset?.col === keep.col) : null;
        if (input) {
          if (input.dataset.shown === keep.shown) input.value = keep.value;
          input.focus({ preventScroll: true });
        } else addInputs[0].focus({ preventScroll: true });
      }
    },
  };
}

const issueNodes = (issues) =>
  issues.map((i) => h('p', { class: i.level === 'error' ? 'field__error' : 'fld__warn' }, i.level === 'error' ? 'Error: ' : 'Warning: ', i.message));

// ───────── type picker (network_g) ─────────

/**
 * A type id picked from the list or typed. A typed id that is listed in `accept` (without regard to
 * case) is written as listed; any other id is written as typed, as the YAML would take it. `onPick(id)`
 * does the change (it may ask first) and resolves true when it was applied; otherwise the field shows
 * the recipe's value again. Returns a field-like object the form registers like any other row.
 * The list shows `ids` first, each with `optionLabel(id)`, then the rest of `accept` with `restNote`.
 */
export function typeField({ id, path, ids, accept = ids, getValue, description, onPick, optionLabel = null, restNote = null, note = null, listLabel = 'Types' }) {
  let shownText = '';
  const getChoices = () => {
    const first = new Set(ids.map((x) => x.toLowerCase()));
    return toChoices([
      ...ids.map((x) => ({ value: x, note: optionLabel ? optionLabel(x) : null })),
      ...accept.filter((x) => !first.has(x.toLowerCase())).map((x) => ({ value: x, note: restNote })),
    ]);
  };
  const combo = createCombobox({
    id,
    inputClass: 'input--path',
    closed: false,
    getChoices,
    current: () => shownText,
    listLabel,
    attrs: { 'data-fld': pathKey(path) },
    onPick: () => commit(),
    onEnter: () => commit(),
    onEscape: () => {
      input.value = shownText;
      showError(null);
    },
  });
  const { input } = combo;
  const error = h('p', { class: 'field__error', role: 'alert', hidden: true });
  const issueSlot = h('div', { class: 'fld__issues' });
  const tag = h('span', { class: 'fld__tag' });
  const noteEl = h('p', { class: 'fld__note', hidden: true });
  const paintNote = () => {
    const text = note ? note(getValue()) : null;
    noteEl.textContent = text ?? '';
    noteEl.hidden = !text;
  };
  const showError = (msg) => {
    error.textContent = msg ?? '';
    error.hidden = !msg;
    if (msg) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  };
  let busy = false;
  const show = () => {
    const t = getValue();
    shownText = typeof t === 'string' ? t : '';
    input.value = shownText;
    input.placeholder = 'Not set';
    tag.textContent = shownText ? 'Set' : 'Not set';
    paintNote();
  };
  // Enter, then the blur caused by a confirm dialog, both fire: handle one change at a time.
  const commit = async () => {
    if (busy) return;
    busy = true;
    try {
      const next = input.value.trim();
      showError(null);
      if (next === shownText) return;
      // A block cannot be without its type: an emptied field shows the recipe's type again.
      if (!next) return show();
      // A listed type is written as it is listed; any other id is the person's own (a network added to
      // the trainer, a type this installation lacks) and is written as typed. The check says what runs.
      const match = accept.find((x) => x.toLowerCase() === next.toLowerCase()) ?? next;
      if (await onPick(match)) {
        shownText = match;
        input.value = match;
      } else show();
    } finally {
      busy = false;
    }
  };
  // Typed text is taken when focus leaves the field; Enter and a pick from the list go through the list.
  input.addEventListener('change', commit);
  // A value that changed while the field held focus is drawn when focus leaves, unless a pick is
  // still being applied or text was typed (the change event handles that).
  let stale = false;
  input.addEventListener('blur', () => {
    queueMicrotask(() => {
      if (!stale || busy || document.activeElement === input || input.value.trim() !== shownText) return;
      stale = false;
      show();
    });
  });
  show();
  const row = h(
    'div',
    { class: 'fld', dataset: { path: pathKey(path) } },
    h(
      'div',
      { class: 'fld__label' },
      h('div', { class: 'fld__head' }, h('label', { class: 'fld__key', for: id }, String(path[path.length - 1])), tag),
      description ? h('p', { class: 'fld__desc' }, description) : null,
    ),
    h(
      'div',
      { class: 'fld__control' },
      combo.el,
      noteEl,
      error,
      issueSlot,
    ),
  );
  return {
    el: row,
    path,
    key: String(path[path.length - 1]),
    pathKey: pathKey(path),
    update: () => {
      if (document.activeElement !== input) show();
      else {
        stale = true;
        paintNote();
      }
    },
    setIssues: (issues) => {
      issueSlot.replaceChildren(...issueNodes(issues));
      if (issues.some((i) => i.level === 'error')) input.setAttribute('aria-invalid', 'true');
      else if (error.hidden) input.removeAttribute('aria-invalid');
    },
    focus: () => input.focus({ preventScroll: true }),
  };
}

// ───────── generic object editor ─────────

/**
 * Editor for free-form objects (optim_g, scheduler, network_d, val.metrics): one row per existing key,
 * control from the current value's type, nested objects one level deep, plus an "Add option" row.
 * The editor's rows are registered through `register` so the form can update them in place.
 * `rowExtras(path, key, value)` may return more of a row's spec (`choices`, `note`) for a key whose
 * usual values the caller knows; `knownOptions(path)` may return {options: [{name, hasDefault, default,
 * choices?}]} for the object at `path` (this one or a nested one): the option names "Add option" offers,
 * each with the value it starts from and the values it usually takes. null: nothing is known.
 * `special` lets the caller draw a key it knows the structure of: `special.row(path, key, value)` may
 * return the element for that key (null: the usual row), and `special.more(path, object)` elements for
 * what the object can hold and does not yet (placed before "Add option"). `special.first(path, object)`
 * may return elements that come before the object's own rows (the type a block lacks), and
 * `special.adder(path, object)` what stands in place of "Add option" where a new key is more than a name
 * and a value (null: the usual row). All four reach nested objects too.
 * @param {{path: Array, value: any, ctx: object, register: (field: object) => void, depth?: number, firstKeys?: string[],
 *   rowExtras?: Function|null, knownOptions?: Function|null,
 *   special?: {row?: Function, more?: Function, first?: Function, adder?: Function}|null}} options
 */
export function genericEditor({ path, value, ctx, register, depth = 0, firstKeys = [], rowExtras = null, knownOptions = null, special = null }) {
  const obj = isPlainObject(value) ? value : {};
  const keys = Object.keys(obj);
  const ordered = [...firstKeys.filter((k) => keys.includes(k)), ...keys.filter((k) => !firstKeys.includes(k))];
  const rows = special?.first ? [...(special.first(path, obj) ?? [])] : [];
  for (const key of ordered) {
    const child = obj[key];
    const own = special?.row ? special.row([...path, key], key, child) : null;
    if (own) {
      rows.push(own);
      continue;
    }
    if (isPlainObject(child) && depth === 0) {
      rows.push(
        h(
          'div',
          { class: 'form__nested' },
          h(
            'div',
            { class: 'form__nestedhead' },
            h('p', { class: 'form__nestedtitle mono' }, key),
            h(
              'button',
              {
                class: 'btn btn--sm btn--quiet',
                type: 'button',
                'aria-label': `Remove ${key}`,
                onclick: () => Promise.resolve(ctx.remove([...path, key])).catch((err) => toast(`${key} was not removed: ${err?.message ?? err}`, { kind: 'err' })),
              },
              'Remove',
            ),
          ),
          genericEditor({ path: [...path, key], value: child, ctx, register, depth: 1, rowExtras, knownOptions, special }),
        ),
      );
      continue;
    }
    const extra = rowExtras ? rowExtras([...path, key], key, child) : null;
    const field = createField({ path: [...path, key], key, kind: kindFromValue(child), value: child, isSet: true, removable: true, ...(extra ?? {}) }, ctx);
    register(field);
    rows.push(field.el);
  }
  // Nested objects (a metric, a per-optimizer schedule) take new options too. The name list is offered
  // only where the caller can know names for this object (`knownOptions.applies`, when it has one).
  const knows = knownOptions && (typeof knownOptions.applies !== 'function' || knownOptions.applies(path));
  if (special?.more) rows.push(...(special.more(path, obj) ?? []));
  const adder = special?.adder ? special.adder(path, obj) : null;
  if (adder) rows.push(adder);
  else if (depth <= 1) rows.push(addOptionRow(path, ctx, knows ? () => knownOptions(path) : null, () => keys));
  return h('div', { class: 'form__generic' }, rows);
}

/**
 * The "Add option" row of an object that takes keys the form has no row for: a name, a value, a button.
 * `known()` may return {options: [{name, hasDefault, default, choices?}]}: the names the object is known
 * to take (those not set yet are offered) and, for a name with `choices`, the values it usually takes.
 * Any other name and any other value can be typed. The value is read as JSON, else taken as text.
 */
export function addOptionRow(path, ctx, known = null, setKeys = () => []) {
  const keyId = uid('addkey');
  const valueId = uid('addval');
  // With a list of the options this object takes, the name is chosen from it (those the recipe does not
  // set yet) or typed; a chosen name brings the value the option starts from, ready to change.
  let filled = '';
  const listed = () => {
    const answer = known ? known() : null;
    return (Array.isArray(answer?.options) ? answer.options : []).filter((option) => option && typeof option.name === 'string');
  };
  const offered = () => {
    const have = new Set(setKeys());
    return listed().filter((option) => !have.has(option.name));
  };
  /** The values the option named in the name field usually takes; none for most options. */
  const valueChoices = () => {
    const name = keyInput.value.trim();
    const option = name ? listed().find((item) => item.name === name) : null;
    return toChoices(Array.isArray(option?.choices) ? option.choices : []);
  };
  const valueCombo = known
    ? createCombobox({ id: valueId, inputClass: 'input--path', placeholder: 'value, for example 1e-4', closed: false, listLabel: 'Values of the option', getChoices: valueChoices, current: () => valueInput.value, onEnter: () => submit() })
    : null;
  const valueInput = valueCombo ? valueCombo.input : h('input', { class: 'input input--path', id: valueId, type: 'text', autocomplete: 'off', spellcheck: 'false', placeholder: 'value, for example 1e-4' });
  const nameCombo = known
    ? createCombobox({
        id: keyId,
        inputClass: 'input--path',
        placeholder: 'option name',
        closed: false,
        listLabel: 'Option names',
        getChoices: () => toChoices(offered().map((option) => ({ value: option.name, note: option.hasDefault ? `starts at ${JSON.stringify(option.default)}` : null }))),
        onPick: (choice) => {
          const option = offered().find((item) => item.name === choice.value);
          if (option?.hasDefault && (valueInput.value.trim() === '' || valueInput.value === filled)) {
            // A text default is shown as the text itself where the option lists its values, so it matches the list.
            filled = typeof option.default === 'string' && Array.isArray(option.choices) ? option.default : JSON.stringify(option.default);
            valueInput.value = filled;
          }
          valueInput.focus();
          valueInput.select();
          // An option with listed values shows them at once.
          if (valueCombo && valueChoices().length) valueCombo.open();
        },
        onEnter: () => submit(),
      })
    : null;
  const keyInput = nameCombo ? nameCombo.input : h('input', { class: 'input input--path', id: keyId, type: 'text', autocomplete: 'off', spellcheck: 'false', placeholder: 'option name' });
  const error = h('p', { class: 'field__error', role: 'alert', hidden: true });
  const submit = () => {
    const key = keyInput.value.trim();
    const text = valueInput.value.trim();
    let problem = !key ? 'Enter the option name.' : /[.\s[\]/]/.test(key) ? 'An option name cannot contain dots, slashes, brackets or spaces.' : !text ? 'Enter a value.' : null;
    let value;
    let floatLiteral;
    if (!problem) {
      // A listed value is written as it is listed (the text "601" stays text); anything else is read as
      // JSON, and plain text that is not JSON is taken as a string, so `adamw` works without quotes.
      const picked = valueChoices().find((choice) => choice.text === text);
      if (picked) value = picked.value;
      else {
        try {
          value = JSON.parse(text);
        } catch {
          value = text;
        }
        // A number typed in scientific notation (1e-4) keeps that spelling in the YAML. What counts as
        // a number is still decided by JSON alone, so text such as 007 stays text.
        if (typeof value === 'number') {
          if (Number.isFinite(value)) floatLiteral = parseNumber(text, 'number').floatLiteral;
          else problem = 'Enter a finite number.';
        }
      }
    }
    error.textContent = problem ?? '';
    error.hidden = !problem;
    if (problem) return;
    // `ctx.add`, where the form has one, also sends focus to the new row once it is drawn.
    const write = ctx.add ?? ctx.apply;
    Promise.resolve(floatLiteral ? write([...path, key], value, floatLiteral) : write([...path, key], value)).then(
      () => {
        keyInput.value = '';
        valueInput.value = '';
        filled = '';
      },
      (err) => toast(`${key} was not added: ${err?.message ?? err}`, { kind: 'err' }),
    );
  };
  const onKey = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      submit();
    }
  };
  // A field with a list handles Enter itself (it may be taking a choice); a plain field submits here.
  if (!nameCombo) keyInput.addEventListener('keydown', onKey);
  if (!valueCombo) valueInput.addEventListener('keydown', onKey);
  return h(
    'div',
    { class: 'fld fld--add' },
    h('div', { class: 'fld__label' }, h('span', { class: 'fld__key' }, 'Add option'), h('p', { class: 'fld__desc' }, 'The value is read as JSON; plain text is taken as a string.')),
    h(
      'div',
      { class: 'fld__control' },
      h(
        'div',
        { class: 'fld__line' },
        h('label', { class: 'visually-hidden', for: keyId }, 'Option name'),
        nameCombo ? nameCombo.el : keyInput,
        h('label', { class: 'visually-hidden', for: valueId }, 'Value'),
        valueCombo ? valueCombo.el : valueInput,
        h('button', { class: 'btn btn--sm', type: 'button', onclick: submit }, 'Add option'),
      ),
      error,
    ),
  );
}
