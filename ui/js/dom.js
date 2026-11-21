// DOM and formatting helpers shared by every module. No framework: views build nodes with h().

/**
 * Create an element.
 *   h('button', { class: 'btn', onclick: fn, 'aria-pressed': 'true', dataset: { id: 3 } }, 'Label')
 * Props: `class`, `dataset`, `style` (object), `on<event>` handlers, `ref` (callback),
 * boolean attributes (true adds, false/null/undefined omits), everything else is an attribute.
 * Children: strings, numbers, nodes, arrays; null/undefined/false are skipped.
 */
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') el.className = value;
      else if (key === 'dataset') Object.assign(el.dataset, value);
      else if (key === 'style' && typeof value === 'object') setStyle(el, value);
      else if (key === 'ref') value(el);
      else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2).toLowerCase(), value);
      else if (key === 'value' || key === 'checked' || key === 'disabled' || key === 'selected' || key === 'indeterminate') el[key] = value;
      else if (value === true) el.setAttribute(key, '');
      else el.setAttribute(key, String(value));
    }
  }
  append(el, children);
  return el;
}

/** Custom properties (`--name`) only take effect through setProperty; plain assignment drops them. */
function setStyle(el, styles) {
  for (const [name, value] of Object.entries(styles)) {
    if (value === null || value === undefined) continue;
    if (name.startsWith('--')) el.style.setProperty(name, String(value));
    else el.style[name] = value;
  }
}

function append(el, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) append(el, child);
    else if (child instanceof Node) el.appendChild(child);
    else el.appendChild(document.createTextNode(String(child)));
  }
}

/** Replace all children of `el`. */
export function mount(el, ...children) {
  const frag = document.createDocumentFragment();
  append(frag, children);
  el.replaceChildren(frag);
  return el;
}

export function debounce(fn, ms) {
  let timer = 0;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  wrapped.cancel = () => clearTimeout(timer);
  wrapped.flush = (...args) => {
    clearTimeout(timer);
    fn(...args);
  };
  return wrapped;
}

export const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));

let uidCounter = 0;
export const uid = (prefix = 'id') => `${prefix}-${++uidCounter}`;

const intFormat = new Intl.NumberFormat('en-US');
const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
});

/** Formatters return '' for absent values so callers can decide what "unknown" looks like. */
export const fmt = {
  int(n) {
    return Number.isFinite(n) ? intFormat.format(n) : '';
  },
  /** Compact but exact enough for losses and learning rates. */
  num(n, digits = 4) {
    if (!Number.isFinite(n)) return '';
    if (n === 0) return '0';
    const abs = Math.abs(n);
    if (abs >= 1e5 || abs < 1e-3) return n.toExponential(Math.max(0, digits - 1));
    return String(Number(n.toPrecision(digits)));
  },
  percent(part, whole) {
    if (!Number.isFinite(part) || !Number.isFinite(whole) || whole <= 0) return '';
    return `${((part / whole) * 100).toFixed(1)}%`;
  },
  dateTime(iso) {
    const d = iso ? new Date(iso) : null;
    return d && !Number.isNaN(d.getTime()) ? dateTimeFormat.format(d) : '';
  },
  /** "3 min ago". Relative to the browser clock; only for timestamps the API supplied. */
  ago(iso) {
    const d = iso ? new Date(iso) : null;
    if (!d || Number.isNaN(d.getTime())) return '';
    const s = Math.round((Date.now() - d.getTime()) / 1000);
    if (s < 0) return fmt.dateTime(iso);
    if (s < 10) return 'just now';
    if (s < 60) return `${s} s ago`;
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return `${Math.floor(s / 86400)} d ago`;
  },
  dims(w, h) {
    return Number.isFinite(w) && Number.isFinite(h) ? `${w} × ${h}` : '';
  },
  plural(n, one, many = `${one}s`) {
    return `${fmt.int(n)} ${n === 1 ? one : many}`;
  },
};
