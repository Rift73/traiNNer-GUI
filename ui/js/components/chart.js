// Dependency-free SVG line chart for training telemetry.
//   const chart = createChart(host, { title, height, scaleToggle, smoothing });
//   chart.update({ series, xLabel });  chart.destroy();
// Options: `scaleToggle: false` drops the Linear / Log switch, `smoothing: false` drops the smoothing
// slider. Axis ranges are available on every chart. X ticks: see xTicks().
// series: [{name, points: [{x, y}]}] with points sorted by x. A y of null (or any non-finite value)
// is a gap: the line breaks there, it is never drawn as zero (see runs/metrics.js: historyPoints).
// update() is cheap and keeps hidden series, y scale, smoothing and the cursor.
// Pure helpers (niceTicks, logTicks, formatTick, decimate, smooth, segments) are exported for checks.

import { h, fmt, uid, clamp } from '../dom.js';

const SVG = 'http://www.w3.org/2000/svg';
const SLOTS = 8;
const HEIGHT = 220;
// The bottom pad holds two lines: tick labels, then the x-axis title on its own line below them.
const PAD = { top: 10, right: 14, bottom: 40, left: 54 };

// ───────── pure helpers ─────────

function niceStep(raw) {
  const exp = Math.floor(Math.log10(raw));
  const base = 10 ** exp;
  const f = raw / base;
  const m = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return m * base;
}

/**
 * Linear ticks covering [min, max] with 1-2-5 steps. A flat range (min == max) is widened around
 * the value so a constant learning rate still gets an axis.
 * @returns {{min: number, max: number, step: number, ticks: number[]}}
 */
export function niceTicks(min, max, count = 5) {
  let lo = min;
  let hi = max;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { min: 0, max: 1, step: 0.2, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] };
  if (lo > hi) [lo, hi] = [hi, lo];
  if (lo === hi) {
    const pad = lo === 0 ? 1 : Math.abs(lo) * 0.1;
    lo -= pad;
    hi += pad;
  }
  const step = niceStep((hi - lo) / Math.max(1, count));
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const ticks = [];
  // Rounding to the step's precision avoids 0.30000000000000004 labels.
  const digits = Math.max(0, -Math.floor(Math.log10(step)) + 1);
  for (let v = start, i = 0; v <= end + step / 2 && i < 50; v += step, i += 1) ticks.push(Number(v.toFixed(Math.min(20, digits))));
  return { min: start, max: end, step, ticks };
}

/** Least distance between two x labels, in px. */
const X_LABEL_GAP = 56;

/** The next "nice" step (1, 2, 5 times a power of ten) above or below `step`. */
function niceNeighbour(step, up) {
  const exp = Math.floor(Math.log10(step) + 1e-9);
  const base = 10 ** exp;
  const m = Math.round(step / base);
  if (up) return m === 1 ? 2 * base : m === 2 ? 5 * base : 10 * base;
  return m === 1 ? 0.5 * base : m === 2 ? base : 2 * base;
}

/**
 * X ticks across the data domain [x0, x1] of a plot `plotW` px wide: at least three and at most about
 * eight labels, never closer than X_LABEL_GAP. An integer domain spanning under 10 gets every integer
 * (or every second one); otherwise steps are 1, 2 or 5 times a power of ten (never below 1 for
 * integers). The first and last data positions are labelled too when they fit beside the others.
 * @returns {{min: number, max: number, step: number, ticks: number[], labels: string[]}}
 */
export function xTicks(x0, x1, plotW, { integer = false } = {}) {
  if (!Number.isFinite(x0) || !Number.isFinite(x1)) return { min: 0, max: 1, step: 1, ticks: [], labels: [] };
  const lo = Math.min(x0, x1);
  const hi = Math.max(x0, x1);
  if (lo === hi) {
    // One position: widen the domain around it and label it alone.
    const pad = integer ? 1 : lo === 0 ? 1 : Math.abs(lo) * 0.1;
    return { min: lo - pad, max: hi + pad, step: pad, ticks: [lo], labels: [formatTick(lo, integer ? 1 : pad)] };
  }
  const span = hi - lo;
  const width = Math.max(1, plotW);
  const maxLabels = clamp(Math.floor(width / X_LABEL_GAP) + 1, 3, 8);
  const digits = (s) => Math.min(20, Math.max(0, -Math.floor(Math.log10(s)) + 1));
  const aligned = (s) => {
    const out = [];
    const start = Math.ceil(lo / s - 1e-9) * s;
    for (let v = start, i = 0; v <= hi + s * 1e-9 && i < 200; v += s, i += 1) out.push(Number(v.toFixed(digits(s))));
    return out;
  };
  let step = 0;
  let ticks = null;
  if (integer && span < 10) {
    for (const s of [1, 2]) {
      if (Math.floor(span / s) + 1 <= maxLabels) {
        step = s;
        ticks = [];
        for (let v = lo; v <= hi; v += s) ticks.push(v);
        break;
      }
    }
  }
  if (!ticks) {
    step = niceStep(span / Math.max(1, maxLabels - 1));
    if (integer) step = Math.max(1, step);
    ticks = aligned(step);
    for (let guard = 0; ticks.length > maxLabels && guard < 30; guard += 1) {
      step = niceNeighbour(step, true);
      ticks = aligned(step);
    }
    for (let guard = 0; ticks.length < 3 && guard < 30; guard += 1) {
      const smaller = niceNeighbour(step, false);
      if (integer && smaller < 1) break;
      step = smaller;
      ticks = aligned(step);
    }
  }
  const labels = ticks.map((v) => formatTick(v, step));
  // The first and last data positions, when they are not ticks already and leave room for their label.
  if (integer) {
    const px = width / span;
    for (const end of [lo, hi]) {
      if (ticks.includes(end)) continue;
      const nearest = ticks.reduce((d, v) => Math.min(d, Math.abs(v - end)), Infinity);
      if (nearest * px >= X_LABEL_GAP * 0.8) {
        const at = end === lo ? 0 : ticks.length;
        ticks.splice(at, 0, end);
        labels.splice(at, 0, formatTick(end, 1));
      }
    }
  }
  return { min: lo, max: hi, step, ticks, labels };
}

/**
 * Log ticks for a positive range: powers of ten, plus 2 and 5 when the range spans few decades.
 * @returns {{min: number, max: number, ticks: number[]}}
 */
export function logTicks(min, max) {
  if (!(min > 0) || !Number.isFinite(max)) return { min: 1, max: 10, ticks: [1, 10] };
  const lo = 10 ** Math.floor(Math.log10(min)) || min;
  let hi = Math.min(Number.MAX_VALUE, 10 ** Math.ceil(Math.log10(max)));
  if (hi === lo) hi = Math.min(Number.MAX_VALUE, lo * 10);
  const decades = Math.round(Math.log10(hi / lo));
  const mults = decades <= 2 ? [1, 2, 5] : [1];
  const ticks = [];
  for (let exp = Math.floor(Math.log10(min)); exp <= Math.ceil(Math.log10(max)) + (min === max ? 1 : 0); exp++) {
    const d = 10 ** exp;
    for (const m of mults) {
      const v = Number((d * m).toPrecision(6));
      if (v > 0 && Number.isFinite(v) && v <= hi * 1.0001) ticks.push(v);
    }
  }
  return { min: lo, max: hi, ticks };
}

const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumSignificantDigits: 3 });

/** Compact tick label. `step` sets the precision so neighbouring labels differ. */
export function formatTick(v, step) {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  // The step decides the notation so one axis never mixes "5000" with "10K" or "0.001" with "2e-4".
  if (abs >= 1e4 || (step && step >= 1000)) {
    const label = compact.format(v);
    if (!step || (compact.format(v - step) !== label && compact.format(v + step) !== label)) return label;
  }
  if (abs < 1e-3 || (step && step < 1e-3 && abs < 1)) {
    const digits = step ? clamp(Math.floor(Math.log10(abs)) - Math.floor(Math.log10(step)), 0, 6) : 2;
    return v.toExponential(digits).replace(/\.?0+e/, 'e').replace('e+', 'e');
  }
  const decimals = step ? clamp(-Math.floor(Math.log10(step) + 1e-9), 0, 8) : 3;
  return String(Number(v.toFixed(decimals)));
}

/**
 * Min/max decimation: per pixel column keep the lowest and highest point (in their original
 * order), so spikes survive while the path has at most ~2 points per column.
 * @param {Array<{x: number, y: number}>} points sorted by x
 */
export function decimate(points, x0, x1, columns) {
  if (points.length <= columns * 2 || x1 <= x0) return points;
  const out = [];
  const width = (x1 - x0) / columns;
  let col = -1;
  let lo = -1; // indices into points, so the original order of min and max is known
  let hi = -1;
  const flush = () => {
    if (lo < 0) return;
    if (lo === hi) out.push(points[lo]);
    else if (lo < hi) out.push(points[lo], points[hi]);
    else out.push(points[hi], points[lo]);
  };
  for (let i = 0; i < points.length; i += 1) {
    const p = points[i];
    const c = Math.min(columns - 1, Math.floor((p.x - x0) / width));
    if (c !== col) {
      flush();
      col = c;
      lo = i;
      hi = i;
    } else {
      if (p.y < points[lo].y) lo = i;
      if (p.y > points[hi].y) hi = i;
    }
  }
  flush();
  return out;
}

const finite = (p) => typeof p.y === 'number' && Number.isFinite(p.y);

/**
 * Runs of consecutive points with a real value. A null value ends a run, so the line has a gap.
 * @param {Array<{x: number, y: number|null}>} points sorted by x
 * @returns {Array<Array<{x: number, y: number}>>}
 */
export function segments(points) {
  const out = [];
  let run = [];
  for (const p of points) {
    if (finite(p)) run.push(p);
    else if (run.length) {
      out.push(run);
      run = [];
    }
  }
  if (run.length) out.push(run);
  return out;
}

/** Exponential moving average, a display aid only. weight in [0, 1). */
export function smooth(points, weight) {
  if (!(weight > 0)) return points;
  let s = points.length ? points[0].y : 0;
  return points.map((p) => {
    s = s * weight + p.y * (1 - weight);
    return { x: p.x, y: s };
  });
}

function nearestIndex(sorted, x) {
  let lo = 0;
  let hi = sorted.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(sorted[lo - 1] - x) <= Math.abs(sorted[lo] - x)) return lo - 1;
  return lo;
}

function pointAt(points, x) {
  let lo = 0;
  let hi = points.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].x === x) return points[mid];
    if (points[mid].x < x) lo = mid + 1;
    else hi = mid - 1;
  }
  return null;
}

function svg(tag, attrs) {
  const el = document.createElementNS(SVG, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  return el;
}

// When one end is automatic and all measurements fall outside the fixed end,
// keep the requested bound and leave a small, ordered interval on its other side.
function axisWindow(low, high, fixedMin, fixedMax, log = false) {
  let min = fixedMin ?? low;
  let max = fixedMax ?? high;
  if (Number.isFinite(min) && Number.isFinite(max) && min <= max &&
      (min < max || (fixedMin === null && fixedMax === null))) return [min, max];
  if (Number.isFinite(min) && fixedMax === null) {
    max = log ? min * 1.1 : min + Math.max(Math.abs(min) * 0.1, 1);
  } else if (Number.isFinite(max) && fixedMin === null) {
    min = log ? max / 1.1 : max - Math.max(Math.abs(max) * 0.1, 1);
  } else {
    return log ? [1, 10] : [0, 1];
  }
  return [min, max];
}

// ───────── component ─────────

/**
 * @param {HTMLElement} host
 * @param {{title?: string, height?: number, scaleToggle?: boolean, smoothing?: boolean, onXRangeChange?: Function}} [options]
 *   scaleToggle: false leaves out the Linear / Log switch (the scale stays linear);
 *   smoothing: false leaves out the smoothing slider (lines are never smoothed). Both default to true.
 */
export function createChart(host, { title = 'Chart', height = HEIGHT, scaleToggle = true, smoothing: smoothingControl = true, onXRangeChange } = {}) {
  const id = uid('chart');
  const state = {
    series: [],
    markers: [],
    xLabel: 'Iteration',
    hidden: new Set(),
    slots: new Map(), // name -> colour slot, assigned once so a series keeps its colour
    log: false,
    smoothing: 0,
    bounds: { xMin: null, xMax: null, yMin: null, yMax: null },
    cursorX: null, // data x of the cursor, kept across updates
    pointerInside: false,
    width: 0,
    frame: null, // geometry of the last draw
  };

  // controls
  const linearBtn = h('button', { class: 'btn btn--sm', type: 'button', 'aria-pressed': 'true', onclick: () => setLog(false) }, 'Linear');
  const logBtn = h('button', { class: 'btn btn--sm', type: 'button', 'aria-pressed': 'false', onclick: () => setLog(true) }, 'Log');
  const smoothId = `${id}-smooth`;
  const smoothOut = h('output', { class: 'chart__smooth-value num', for: smoothId }, '0');
  const smoothInput = h('input', {
    class: 'chart__range',
    id: smoothId,
    type: 'range',
    min: '0',
    max: '0.95',
    step: '0.05',
    value: '0',
    oninput: () => {
      state.smoothing = Number(smoothInput.value) || 0;
      smoothOut.textContent = state.smoothing.toFixed(2);
      draw();
    },
  });
  const rangeFields = {};
  const fields = Object.keys(state.bounds).map((key) => {
    const axis = key[0].toUpperCase();
    const end = key.endsWith('Min') ? 'minimum' : 'maximum';
    const input = h('input', { class: 'input num', id: id + key, type: 'number',
      step: axis === 'X' ? '1' : 'any', min: axis === 'X' ? '0' : null,
      inputmode: axis === 'X' ? 'numeric' : 'decimal', placeholder: 'Auto' });
    rangeFields[key] = input;
    return h('label', { for: input.id }, `${axis} ${end}`, input);
  });
  const rangeError = h('p', { class: 'chart__range-error', role: 'alert', hidden: true });
  const rangeButton = h('button', { type: 'button', class: 'btn btn--sm', 'aria-controls': id + '-bounds',
    'aria-expanded': 'false', onclick: () => {
      rangeForm.hidden = !rangeForm.hidden;
      rangeButton.setAttribute('aria-expanded', String(!rangeForm.hidden));
    } }, 'Axis ranges');
  const rangeForm = h('form', { class: 'chart__bounds', id: id + '-bounds', hidden: true,
    'aria-label': title + ' axis ranges', novalidate: true,
    onsubmit: (event) => { event.preventDefault(); applyRanges(); } },
    h('div', { class: 'chart__bound-fields' }, fields),
    h('div', { class: 'chart__bar' },
      h('button', { type: 'submit', class: 'btn btn--sm' }, 'Apply'),
      h('button', { type: 'button', class: 'btn btn--sm', onclick: () => {
        for (const input of Object.values(rangeFields)) input.value = '';
        applyRanges();
      } }, 'Reset to auto')),
    rangeError);
  const bar = h('div', { class: 'chart__bar' },
    scaleToggle ? h('div', { class: 'seg', role: 'group', 'aria-label': 'Y scale' }, linearBtn, logBtn) : null,
    smoothingControl ? [h('label', { class: 'chart__smooth', for: smoothId }, 'Smoothing'), smoothInput, smoothOut] : null,
    rangeButton);

  function updateRangeButton() {
    const custom = Object.values(state.bounds).some((v) => v !== null);
    rangeButton.textContent = custom ? 'Axis ranges · Custom' : 'Axis ranges';
  }

  function applyRanges() {
    const next = {};
    function invalid(key, message) {
      const input = rangeFields[key];
      input.setAttribute('aria-invalid', 'true');
      input.focus();
      rangeError.textContent = message;
      rangeError.hidden = false;
    }
    for (const [key, input] of Object.entries(rangeFields)) {
      input.removeAttribute('aria-invalid');
      const value = input.value === '' ? null : input.valueAsNumber;
      if (input.validity.badInput || (value !== null && !Number.isFinite(value))) {
        return invalid(key, 'Enter a number or leave the field blank for Auto.');
      }
      if (key[0] === 'x' && value !== null && (!Number.isSafeInteger(value) || value < 0)) {
        return invalid(key, 'Iteration bounds must be non-negative whole numbers.');
      }
      if (key[0] === 'y' && value !== null && state.log && value <= 0) {
        return invalid(key, 'Log scale needs positive Y bounds.');
      }
      next[key] = value;
    }
    for (const axis of ['x', 'y']) {
      const [min, max] = [next[axis + 'Min'], next[axis + 'Max']];
      if (min !== null && max !== null && (min >= max || !Number.isFinite(max - min))) {
        return invalid(axis + 'Max', `Choose a ${axis.toUpperCase()} maximum greater than its minimum within a finite range.`);
      }
    }
    rangeError.hidden = true;
    const xChanged = next.xMin !== state.bounds.xMin || next.xMax !== state.bounds.xMax;
    state.bounds = next;
    updateRangeButton();
    hideCursor();
    renderLegend();
    draw();
    if (xChanged) onXRangeChange?.();
  }

  const plotSvg = svg('svg', { class: 'chart__svg', 'aria-hidden': 'true', height });
  const tooltip = h('div', { class: 'chart__tip', hidden: true });
  const hint = h('span', { class: 'visually-hidden', id: `${id}-hint` }, 'Use the Left and Right arrow keys to read the reported values point by point.');
  const live = h('div', { class: 'visually-hidden', 'aria-live': 'polite' });
  const plot = h(
    'div',
    {
      class: 'chart__plot',
      tabindex: '0',
      role: 'group',
      'aria-label': `${title} chart`,
      'aria-describedby': `${id}-hint`,
      style: { height: `${height}px` },
      onpointermove: onPointer,
      onpointerdown: onPointer,
      onpointerleave: () => {
        state.pointerInside = false;
        if (document.activeElement !== plot) hideCursor();
      },
      onkeydown: onKey,
      onblur: () => {
        if (!state.pointerInside) hideCursor();
      },
    },
    plotSvg,
    tooltip,
  );
  const legend = h('ul', { class: 'chart__legend', 'aria-label': `${title} series` });
  const note = h('p', { class: 'chart__note', hidden: true }, 'Solid lines are smoothed for display; the faint lines are the reported values.');
  const root = h('div', { class: 'chart' }, bar, rangeForm, plot, legend, note, hint, live);
  host.appendChild(root);

  const legendItems = new Map(); // name -> {li, input, value}

  const observer = new ResizeObserver(() => {
    const w = Math.floor(plot.clientWidth);
    if (w && w !== state.width) draw();
  });
  observer.observe(plot);

  function slotOf(name) {
    if (!state.slots.has(name)) state.slots.set(name, state.slots.size);
    return state.slots.get(name);
  }

  function colourStyle(name) {
    const slot = slotOf(name);
    // Beyond eight series a colour repeats, so the repeat is dashed: identity never rests on hue alone.
    return { colour: `var(--series-${(slot % SLOTS) + 1})`, dashed: slot >= SLOTS };
  }

  function visibleSeries() {
    return state.series.filter((s) => !state.hidden.has(s.name) && s.real.length);
  }

  function logAllowed() {
    const vis = visibleSeries();
    return [state.bounds.yMin, state.bounds.yMax].every((v) => v === null || v > 0) && vis.length > 0 && vis.every((s) => s.real.every((p) => p.y > 0));
  }

  function setLog(on) {
    if (on && !logAllowed()) return;
    state.log = on;
    draw();
  }

  function syncControls() {
    const allowed = logAllowed();
    if (state.log && !allowed) state.log = false;
    logBtn.disabled = !allowed;
    logBtn.title = allowed ? '' : 'Log scale needs positive values and positive Y bounds.';
    linearBtn.setAttribute('aria-pressed', String(!state.log));
    logBtn.setAttribute('aria-pressed', String(state.log));
    note.hidden = !(state.smoothing > 0);
  }

  function renderLegend() {
    const names = new Set(state.series.map((s) => s.name));
    for (const [name, item] of legendItems) {
      if (!names.has(name)) {
        item.li.remove();
        legendItems.delete(name);
      }
    }
    state.series.forEach((s, index) => {
      let item = legendItems.get(s.name);
      if (!item) {
        const inputId = `${id}-s${legendItems.size}-${index}`;
        const { colour, dashed } = colourStyle(s.name);
        const input = h('input', {
          type: 'checkbox',
          id: inputId,
          checked: !state.hidden.has(s.name),
          onchange: () => {
            if (input.checked) state.hidden.delete(s.name);
            else state.hidden.add(s.name);
            draw();
          },
        });
        const value = h('span', { class: 'chart__legend-value num' });
        const li = h(
          'li',
          { class: 'chart__legend-item' },
          input,
          h(
            'label',
            { class: 'chart__legend-label', for: inputId },
            h('span', { class: dashed ? 'chart__swatch chart__swatch--dashed' : 'chart__swatch', style: { '--swatch': colour }, 'aria-hidden': 'true' }),
            h('span', { class: 'chart__legend-name mono' }, s.name),
            value,
          ),
        );
        item = { li, input, value };
        legendItems.set(s.name, item);
      }
      // Move only when out of place: moving a focused checkbox would drop its focus.
      if (legend.children[index] !== item.li) legend.insertBefore(item.li, legend.children[index] ?? null);
      const { xMin, xMax } = state.bounds;
      const ranged = xMin !== null || xMax !== null;
      const last = ranged ? s.real.findLast((p) => (xMin === null || p.x >= xMin) && (xMax === null || p.x <= xMax)) : s.real.at(-1);
      const text = last ? `${ranged ? 'last in range' : 'latest'} ${fmt.num(last.y)}` : ranged ? 'no values in range' : 'no values';
      if (item.value.textContent !== text) item.value.textContent = text;
    });
  }

  function draw() {
    syncControls();
    const width = Math.floor(plot.clientWidth);
    state.width = width;
    plotSvg.replaceChildren();
    plotSvg.setAttribute('width', width);
    plotSvg.setAttribute('viewBox', `0 0 ${width || 1} ${height}`);
    state.frame = null;
    if (!width) return;

    const vis = visibleSeries();
    const all = state.series.filter((s) => s.points.length);
    // x domain from every series so hiding one does not shift the axis
    let x0 = Infinity;
    let x1 = -Infinity;
    for (const s of all) {
      x0 = Math.min(x0, s.points[0].x);
      x1 = Math.max(x1, s.points[s.points.length - 1].x);
    }
    const plotW = Math.max(10, width - PAD.left - PAD.right);
    const plotH = height - PAD.top - PAD.bottom;
    const g = svg('g');
    plotSvg.appendChild(g);
    if (!Number.isFinite(x0) && state.bounds.xMin === null && state.bounds.xMax === null) {
      tooltip.hidden = true;
      const t = svg('text', { x: PAD.left + plotW / 2, y: PAD.top + plotH / 2, 'text-anchor': 'middle', class: 'chart__empty' });
      t.textContent = all.length ? 'All series are hidden.' : 'No values to draw.';
      g.appendChild(t);
      return;
    }
    for (const step of state.markers) {
      x0 = Math.min(x0, step);
      x1 = Math.max(x1, step);
    }
    const bounds = state.bounds;
    [x0, x1] = axisWindow(x0, x1, bounds.xMin, bounds.xMax);
    const xt = xTicks(x0, x1, plotW, { integer: state.xInteger });
    const xMin = xt.min;
    const xMax = xt.max;
    const sx = (x) => PAD.left + ((x - xMin) / (xMax - xMin)) * plotW;

    const inWindow = (point) => point.x >= xMin && point.x <= xMax;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (const s of vis) for (const point of s.real) {
      if (!inWindow(point)) continue;
      y0 = Math.min(y0, point.y);
      y1 = Math.max(y1, point.y);
    }
    const hasValues = Number.isFinite(y0);
    [y0, y1] = axisWindow(y0, y1, bounds.yMin, bounds.yMax, state.log);
    const count = Math.max(2, Math.floor(plotH / 40));
    const autoY = state.log ? logTicks(y0, y1) : niceTicks(y0, y1, count);
    const yMin = bounds.yMin ?? autoY.min;
    const yMax = bounds.yMax ?? autoY.max;
    const transform = state.log ? Math.log10 : (y) => y;
    const bottom = transform(yMin);
    const span = transform(yMax) - bottom;
    const sy = (y) => PAD.top + plotH - ((transform(y) - bottom) / span) * plotH;
    let yTicks = autoY.ticks;
    let yStep = autoY.step;
    const customY = bounds.yMin !== null || bounds.yMax !== null;
    if (customY) {
      if (state.log && yMax / yMin < 2) {
        const narrow = niceTicks(yMin, yMax, count);
        yTicks = narrow.ticks;
        yStep = narrow.step;
      }
      yTicks = [yMin, ...yTicks.filter((v) => v > yMin && v < yMax &&
        sy(yMin) - sy(v) >= 18 && sy(v) - sy(yMax) >= 18), yMax];
    }

    // grid and axes (recessive: --rule hairlines, muted labels)
    for (const v of yTicks) {
      const y = Math.round(sy(v)) + 0.5;
      g.appendChild(svg('line', { class: 'chart__grid', x1: PAD.left, x2: PAD.left + plotW, y1: y, y2: y }));
      const label = svg('text', { class: 'chart__tick', x: PAD.left - 6, y, dy: '0.32em', 'text-anchor': 'end' });
      label.textContent = customY && (v === bounds.yMin || v === bounds.yMax)
        ? String(v) : formatTick(v, yStep ?? v / 10);
      g.appendChild(label);
    }
    xt.ticks.forEach((v, i) => {
      if (v < xMin - 1e-9 || v > xMax + 1e-9) return;
      const x = Math.round(sx(v)) + 0.5;
      g.appendChild(svg('line', { class: 'chart__grid', x1: x, x2: x, y1: PAD.top, y2: PAD.top + plotH }));
      const label = svg('text', { class: 'chart__tick', x, y: PAD.top + plotH + 16, 'text-anchor': 'middle' });
      label.textContent = xt.labels[i];
      g.appendChild(label);
    });
    g.appendChild(svg('line', { class: 'chart__axis', x1: PAD.left, x2: PAD.left + plotW, y1: PAD.top + plotH + 0.5, y2: PAD.top + plotH + 0.5 }));
    // Its own line, centred under the plot, so it can never meet the last tick label.
    const xl = svg('text', { class: 'chart__axis-label', x: PAD.left + plotW / 2, y: height - 4, 'text-anchor': 'middle' });
    xl.textContent = state.xLabel;
    g.appendChild(xl);

    for (const step of state.markers) {
      if (step < xMin || step > xMax) continue;
      const x = sx(step);
      const line = svg('line', { class: 'chart__resume', x1: x, x2: x, y1: PAD.top, y2: PAD.top + plotH,
        stroke: 'currentColor', 'stroke-dasharray': '5 4' });
      const title = svg('title');
      title.textContent = `Resumed at iteration ${fmt.int(step)}`;
      line.appendChild(title);
      g.appendChild(line);
    }
    const toPath = (points) => {
      let d = '';
      for (const p of points) d += `${d ? 'L' : 'M'}${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`;
      return d;
    };
    const clip = svg('clipPath', { id: id + '-clip' });
    clip.appendChild(svg('rect', { x: PAD.left, y: PAD.top, width: plotW, height: plotH }));
    g.appendChild(clip);
    const lines = svg('g', { 'clip-path': `url(#${id}-clip)` });
    for (const s of vis) {
      const { colour, dashed } = colourStyle(s.name);
      const style = `stroke: ${colour}${dashed ? '; stroke-dasharray: 5 3' : ''}`;
      // One path per run of real values: a null value is a gap in the line, never a drop to zero.
      for (const full of s.segs) {
        const seg = bounds.xMin !== null || bounds.xMax !== null ? full.filter(inWindow) : full;
        if (!seg.length) continue;
        if (seg.length === 1) {
          lines.appendChild(svg('circle', { class: 'chart__dot', cx: sx(seg[0].x), cy: sy(seg[0].y), r: 3, style: `fill: ${colour}` }));
          continue;
        }
        const raw = decimate(seg, xMin, xMax, Math.round(plotW));
        if (state.smoothing > 0) {
          lines.appendChild(svg('path', { class: 'chart__line chart__line--raw', d: toPath(raw), style }));
          const sm = decimate(smooth(seg, state.smoothing), xMin, xMax, Math.round(plotW));
          lines.appendChild(svg('path', { class: 'chart__line', d: toPath(sm), style }));
        } else {
          lines.appendChild(svg('path', { class: 'chart__line', d: toPath(raw), style }));
        }
      }
    }
    g.appendChild(lines);
    if (!hasValues) {
      const empty = svg('text', { class: 'chart__empty', x: PAD.left + plotW / 2, y: PAD.top + plotH / 2, 'text-anchor': 'middle' });
      empty.textContent = all.length && !vis.length ? 'No visible series.' : 'No measurements in this iteration range.';
      g.appendChild(empty);
    }
    const cursorLayer = svg('g', { class: 'chart__cursor-layer', 'clip-path': `url(#${id}-clip)` });
    g.appendChild(cursorLayer);

    // Union of the x values reported by the visible series: the cursor moves between real points only.
    const xsSet = new Set();
    for (const s of vis) for (const p of s.real) if (inWindow(p)) xsSet.add(p.x);
    const xs = [...xsSet].sort((a, b) => a - b);
    state.frame = { sx, sy, xMin, xMax, plotW, plotH, xs, vis, cursorLayer };
    drawCursor(false);
  }

  function readout(x) {
    const vis = state.frame?.vis ?? [];
    const rows = vis.map((s) => {
      const p = pointAt(s.points, x);
      return { name: s.name, text: p && finite(p) ? fmt.num(p.y, 6) : p ? 'no value logged' : 'not reported at this point' };
    });
    return { head: `${state.xLabel} ${fmt.int(x) || fmt.num(x)}`, rows };
  }

  function drawCursor(announce) {
    const f = state.frame;
    if (!f) return;
    f.cursorLayer.replaceChildren();
    if (state.cursorX === null || !f.xs.length) {
      tooltip.hidden = true;
      return;
    }
    const x = f.xs[nearestIndex(f.xs, state.cursorX)];
    state.cursorX = x;
    const px = f.sx(x);
    f.cursorLayer.appendChild(svg('line', { class: 'chart__cursor', x1: px, x2: px, y1: PAD.top, y2: PAD.top + f.plotH }));
    for (const s of f.vis) {
      const p = pointAt(s.points, x);
      if (p && finite(p)) f.cursorLayer.appendChild(svg('circle', { class: 'chart__mark', cx: px, cy: f.sy(p.y), r: 3.5, style: `stroke: ${colourStyle(s.name).colour}` }));
    }
    const { head, rows } = readout(x);
    tooltip.replaceChildren(
      h('div', { class: 'chart__tip-head' }, head),
      ...rows.map((r) =>
        h(
          'div',
          { class: 'chart__tip-row' },
          h('span', { class: 'chart__swatch', style: { '--swatch': colourStyle(r.name).colour }, 'aria-hidden': 'true' }),
          h('span', { class: 'chart__tip-name mono' }, r.name),
          h('span', { class: 'chart__tip-value num' }, r.text),
        ),
      ),
    );
    tooltip.hidden = false;
    // Keep the tooltip inside the plot: flip to the left of the cursor near the right edge.
    const tipW = tooltip.offsetWidth || 180;
    const left = px + 12 + tipW > state.width ? Math.max(0, px - 12 - tipW) : px + 12;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${PAD.top}px`;
    if (announce) live.textContent = `${head}. ${rows.map((r) => `${r.name} ${r.text}`).join('. ')}.`;
  }

  function hideCursor() {
    state.cursorX = null;
    drawCursor(false);
  }

  function onPointer(event) {
    const f = state.frame;
    if (!f || !f.xs.length) return;
    state.pointerInside = true;
    const rect = plot.getBoundingClientRect();
    const px = event.clientX - rect.left;
    state.cursorX = f.xMin + ((px - PAD.left) / f.plotW) * (f.xMax - f.xMin);
    drawCursor(false);
  }

  function onKey(event) {
    const f = state.frame;
    if (!f || !f.xs.length) return;
    let index = state.cursorX === null ? -1 : nearestIndex(f.xs, state.cursorX);
    if (event.key === 'ArrowRight') index = index < 0 ? 0 : Math.min(f.xs.length - 1, index + 1);
    else if (event.key === 'ArrowLeft') index = index < 0 ? f.xs.length - 1 : Math.max(0, index - 1);
    else if (event.key === 'Home') index = 0;
    else if (event.key === 'End') index = f.xs.length - 1;
    else if (event.key === 'Escape') {
      hideCursor();
      return;
    } else return;
    event.preventDefault();
    state.cursorX = f.xs[index];
    drawCursor(true);
  }

  return {
    /** @param {{series: Array<{name: string, points: Array<{x: number, y: number}>}>, xLabel?: string}} data */
    iterationRange: () => ({ from: state.bounds.xMin, to: state.bounds.xMax }),
    update({ series, xLabel, markers = [], plotHeight = height, yKey }) {
      if (state.yKey !== yKey) {
        state.yKey = yKey;
        state.bounds.yMin = state.bounds.yMax = null;
        rangeFields.yMin.value = rangeFields.yMax.value = '';
        rangeFields.yMin.removeAttribute('aria-invalid');
        rangeFields.yMax.removeAttribute('aria-invalid');
        rangeError.hidden = true;
        updateRangeButton();
      }
      if (plotHeight !== height) {
        height = plotHeight;
        plot.style.height = `${height}px`;
        plotSvg.setAttribute('height', height);
      }
      state.markers = markers;
      state.series = [...(series ?? [])]
        .map((s) => {
          const points = Array.isArray(s.points) ? s.points : [];
          return { ...s, points, real: points.filter(finite), segs: segments(points) };
        })
        .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
      // Iterations are whole numbers: then the x axis never gets fractional ticks.
      state.xInteger = state.series.every((s) => s.points.every((p) => Number.isInteger(p.x)));
      // Assign colour slots in name order the first time each name appears.
      for (const s of state.series) slotOf(s.name);
      if (xLabel) state.xLabel = xLabel;
      renderLegend();
      draw();
    },
    destroy() {
      observer.disconnect();
      root.remove();
    },
  };
}
