// Image comparison stage: one or two real images over the same displayed rectangle, with wipe,
// side-by-side, blink and difference modes, zoom and pan, and a readout measured from the pixels.
//
// Geometry lives in the pixel space of the primary image (A, or B when A failed). The frame is
// that image's natural size times `scale` (CSS px per image pixel) and its top-left corner sits at
// `pan` (viewport px). B is stretched to the same frame, so for a super-resolution pair each
// low-quality pixel covers several CSS px, which `image-rendering: pixelated` keeps visible as a
// square. The frame is sized with width/height (not a CSS scale transform) so the crop marks keep
// their size and the browser resamples the image once.

import { h, mount, clamp, fmt } from '../dom.js';
import { cropmarks, roleChip } from '../ui.js';

const MODE_LABEL = { wipe: 'Wipe', side: 'Side by side', blink: 'Blink', diff: 'Difference' };
const ALL_MODES = ['wipe', 'side', 'blink', 'diff'];
const PRESETS = [1, 2, 4, 8];
const LADDER = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];
const AMPS = [1, 4, 16];
/** The phone edition (mobile.html marks the document). Only there do two fingers pinch the image. */
const PHONE = typeof document !== 'undefined' && document.documentElement?.dataset.edition === 'mobile';
/** room kept around the frame for the crop marks (14px mark + 4px gap) and a little air */
export const MARGIN = 24;
/** largest frame edge requested from the browser; beyond this some engines drop the image */
const MAX_EDGE_PX = 32768;
const ZOOM_TITLE =
  'Zoom is a percent of the image’s real pixels, in CSS pixels: 100% puts one image pixel on one CSS pixel (several screen pixels on a high-density display).';

// ───────── pure helpers (exported so they can be checked without a browser) ─────────

/** "4× exactly", "about 3.95× (sizes are not a whole multiple)", … ; '' when a size is unknown. */
export function sizeRelation(wa, ha, wb, hb) {
  if (![wa, ha, wb, hb].every((n) => Number.isInteger(n) && n > 0)) return '';
  // Integer cross-multiplication: no floating-point doubt about "exactly".
  if (wa * hb === ha * wb) {
    if (wa % wb === 0) return wa === wb ? '1× exactly (same size)' : `${wa / wb}× exactly`;
    if (wb % wa === 0) return `1/${wb / wa}× exactly`;
  } else if (wa % wb === 0 && ha % hb === 0) {
    return `${wa / wb}× wide, ${ha / hb}× high (not the same in both directions)`;
  }
  const ratio = Math.sqrt((wa * ha) / (wb * hb));
  return `about ${ratio.toFixed(2)}× (sizes are not a whole multiple)`;
}

/** Largest scale that shows the whole image with room for the crop marks. */
export function fitScale(vw, vh, iw, ih) {
  return Math.max(1e-3, Math.min((vw - 2 * MARGIN) / iw, (vh - 2 * MARGIN) / ih));
}

/** Zoom range: never smaller than fit or 100%, never a frame edge beyond MAX_EDGE_PX (but fit always allowed). */
export function scaleLimits(vw, vh, iw, ih) {
  const fit = fitScale(vw, vh, iw, ih);
  return { fit, min: Math.min(fit, 1), max: Math.max(fit, Math.min(16, MAX_EDGE_PX / Math.max(iw, ih))) };
}

/**
 * Clamp one axis of the pan. A frame that fits is centred; a larger one may not leave more
 * than MARGIN of empty stage at either edge, so the image can never be lost off-screen.
 */
export function clampAxis(pos, size, view) {
  if (size + 2 * MARGIN <= view) return (view - size) / 2;
  return clamp(pos, view - size - MARGIN, MARGIN);
}

/** New pan so the image point under `p` stays under `p` when the scale goes from s0 to s1. */
export function zoomAt(p, pan, s0, s1) {
  return p - (p - pan) * (s1 / s0);
}

/** Next step on the zoom ladder (dir +1 / -1), from any current scale. */
export function stepScale(scale, dir) {
  if (dir > 0) return LADDER.find((v) => v > scale * 1.001) ?? LADDER[LADDER.length - 1];
  return [...LADDER].reverse().find((v) => v < scale / 1.001) ?? LADDER[0];
}

/** Image pixel index under viewport coordinate p, or -1 outside the image. */
export function pixelAt(p, pan, scale, n) {
  const i = Math.floor((p - pan) / scale);
  return i >= 0 && i < n ? i : -1;
}

/** Nearest-neighbour index in an axis of size nb for pixel x of an axis of size na (pixel-centre sampling). */
export function mapIndex(x, na, nb) {
  return Math.min(nb - 1, Math.floor(((x + 0.5) * nb) / na));
}

export function zoomLabel(isFit, scale) {
  const pct = scale * 100;
  const text = `${pct < 10 ? pct.toFixed(1) : Math.round(pct)}%`;
  return isFit ? `Fit, ${text}` : text;
}

/**
 * |A − B| per channel, B resampled to A's size by nearest neighbour, times amp (the clamped
 * output array saturates at 255). Alpha is ignored: the result is opaque.
 */
export function computeDiff(a, wa, ha, b, wb, hb, amp, out) {
  const xmap = new Int32Array(wa);
  for (let x = 0; x < wa; x += 1) xmap[x] = mapIndex(x, wa, wb) * 4;
  for (let y = 0; y < ha; y += 1) {
    const rowB = mapIndex(y, ha, hb) * wb * 4;
    let i = y * wa * 4;
    for (let x = 0; x < wa; x += 1, i += 4) {
      const j = rowB + xmap[x];
      out[i] = Math.abs(a[i] - b[j]) * amp;
      out[i + 1] = Math.abs(a[i + 1] - b[j + 1]) * amp;
      out[i + 2] = Math.abs(a[i + 2] - b[j + 2]) * amp;
      out[i + 3] = 255;
    }
  }
  return out;
}

/** The sentence of a rejected load(), never empty. */
export function reasonOf(error) {
  const text = typeof error === 'string' ? error : error && typeof error.message === 'string' ? error.message : '';
  return text.trim() || 'No reason was given.';
}

/**
 * Why a plain GET address did not give an image: the service answers errors as JSON {error}, which an
 * <img> cannot show, so ask once more with fetch. Object and data URLs have no server to ask (null).
 * Resolves to null when the address answers normally (the bytes really are not an image).
 */
async function explainFailure(url) {
  if (typeof url !== 'string' || !url || /^(blob|data):/i.test(url)) return null;
  let response;
  try {
    response = await fetch(url, { cache: 'no-store' });
  } catch {
    return 'The trainer service did not answer.';
  }
  if (response.ok) {
    response.body?.cancel().catch(() => {});
    return null;
  }
  try {
    const payload = await response.json();
    if (payload && typeof payload.error === 'string' && payload.error.trim()) return payload.error.trim();
  } catch {
    // Not JSON: fall back to the status line.
  }
  return `The service answered ${`${response.status} ${response.statusText}`.trim()}.`;
}

// ───────── component ─────────

/**
 * @typedef {{label: string, role: 'gt'|'lq'|'out'|'', load: () => Promise<string>, note?: string}} CompareSource
 *   load() resolves to an address an <img> can show: an object URL or a same-origin GET address. The
 *   component never revokes it (the caller made it). It may reject; its message is shown on that side.
 *   note: a short provenance line shown under the label ("Original output, 1024 × 1024").
 * @param {HTMLElement} host   element to fill (the component sizes itself to 100% of it)
 * @param {{modes?: Array<'wipe'|'side'|'blink'|'diff'>, emptyText?: string}} [options]
 * @returns {{setSources(sources: {a: CompareSource|null, b: CompareSource|null}): void, destroy(): void}}
 */
export function createCompare(host, options = {}) {
  const modes = Array.isArray(options.modes) && options.modes.length ? ALL_MODES.filter((m) => options.modes.includes(m)) : [...ALL_MODES];
  if (!modes.length) modes.push('wipe');
  const emptyText = options.emptyText || 'Select an image to look at it here.';
  const listeners = new AbortController();
  const on = (el, type, fn, opts = {}) => el.addEventListener(type, fn, { ...opts, signal: listeners.signal });

  const st = {
    token: 0,
    sources: { a: null, b: null },
    /** {el, url, w, h} once decoded */
    img: { a: null, b: null },
    err: { a: '', b: '' },
    loading: { a: false, b: false },
    /** ImageData, null when the browser refuses to read it, undefined when not read yet */
    pix: { a: undefined, b: undefined },
    mode: modes.includes('wipe') ? 'wipe' : modes[0],
    wipe: 50,
    blink: 'a',
    amp: 1,
    interp: 'pixels',
    fit: true,
    scale: 1,
    pan: { x: 0, y: 0 },
    /** size of the last shown primary image: zoom and pan survive a switch to an image of the same size */
    prev: null,
    diffKey: '',
    diffError: '',
    pointer: null,
  };
  let drag = null;
  let wipeDrag = null;
  let msgKey = '';

  // ── toolbar ──
  const btn = (label, onclick, props = {}) => h('button', { class: 'btn btn--sm compare__btn', type: 'button', onclick, ...props }, label);
  const seg = (label, buttons) => h('div', { class: 'seg', role: 'group', 'aria-label': label }, buttons);

  const modeButtons = new Map(modes.map((m) => [m, btn(MODE_LABEL[m], () => setMode(m))]));
  const modeSeg = seg('Comparison', [...modeButtons.values()]);
  const blinkState = h('span', { class: 'compare__note' });
  const blinkButton = btn('Switch image', () => flip(), { 'aria-keyshortcuts': 'Space', title: 'Switch image (Space on the image)' });
  const blinkGroup = h('div', { class: 'compare__group' }, blinkButton, blinkState);
  const ampButtons = new Map(AMPS.map((k) => [k, btn(`×${k}`, () => setAmp(k), { 'aria-label': `Amplify ${k} times` })]));
  const ampGroup = h('div', { class: 'compare__group' }, h('span', { class: 'compare__note' }, 'Amplify'), seg('Difference amplification', [...ampButtons.values()]));
  const interpButtons = {
    pixels: btn('Pixels', () => setInterp('pixels'), { title: 'Show enlarged image pixels as squares (nearest neighbour). An image shown smaller than its pixels is always smoothed.' }),
    smooth: btn('Smooth', () => setInterp('smooth'), { title: 'Let the browser smooth enlarged images' }),
  };
  const interpGroup = h('div', { class: 'compare__group' }, h('span', { class: 'compare__note' }, 'Interpolation'), seg('Interpolation', [interpButtons.pixels, interpButtons.smooth]));
  const zoomOutButton = h('button', { class: 'btn btn--sm btn--icon compare__btn', type: 'button', 'aria-label': 'Zoom out', onclick: () => zoomStep(-1) }, '−');
  const zoomInButton = h('button', { class: 'btn btn--sm btn--icon compare__btn', type: 'button', 'aria-label': 'Zoom in', onclick: () => zoomStep(1) }, '+');
  const fitButton = btn('Fit', () => setFit());
  const presetButtons = new Map(PRESETS.map((k) => [k, btn(`${k * 100}%`, () => zoomTo(k))]));
  const zoomText = h('span', { class: 'compare__zoomtext' });
  const zoomGroup = h('div', { class: 'compare__group compare__zoom', title: ZOOM_TITLE }, zoomOutButton, seg('Zoom', [fitButton, ...presetButtons.values()]), zoomInButton, zoomText);
  const bar = h('div', { class: 'compare__bar' }, modeSeg, blinkGroup, ampGroup, interpGroup, zoomGroup);

  // ── viewports ──
  const image = () => h('img', { class: 'compare__img', alt: '', draggable: 'false', decoding: 'async' });
  const imgA = image();
  const imgB = image();
  const imgB2 = image();
  const diffCanvas = h('canvas', { class: 'compare__img' });
  const layerB = h('div', { class: 'compare__layer' }, imgB);
  const frame1 = h('div', { class: 'compare__frame' }, imgA, layerB, diffCanvas, cropmarks());
  const frame2 = h('div', { class: 'compare__frame' }, imgB2, cropmarks());
  const labelA = h('div', { class: 'compare__label' });
  const labelB = h('div', { class: 'compare__label' });
  const label2 = h('div', { class: 'compare__label compare__label--wide' });
  const divider = h(
    'div',
    { class: 'compare__divider', role: 'slider', tabindex: '0', 'aria-label': 'Wipe position', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-orientation': 'horizontal' },
    h('span', { class: 'compare__handle', 'aria-hidden': 'true' }),
  );
  const view1 = h('div', { class: 'compare__view', tabindex: '0', role: 'group' }, frame1, labelA, labelB, divider);
  const view2 = h('div', { class: 'compare__view', tabindex: '0', role: 'group' }, frame2, label2);
  const msg = h('div', { class: 'compare__msg', role: 'status' });
  const alertEl = h('div', { class: 'compare__alert', role: 'status' });
  const body = h('div', { class: 'compare__body' }, view1, view2, msg, alertEl);
  const sizesEl = h('div', { class: 'compare__sizes' });
  // Deliberately not a live region: it changes on every pointer move.
  const pointerEl = h('div', { class: 'compare__pointer' });
  const readout = h('div', { class: 'compare__readout' }, sizesEl, pointerEl);
  const root = h('div', { class: 'stage compare' }, bar, body, readout);
  host.appendChild(root);

  // ── loading ──
  function setSources({ a = null, b = null } = {}) {
    st.token += 1;
    const token = st.token;
    st.sources = { a, b };
    st.img = { a: null, b: null };
    st.err = { a: '', b: '' };
    st.loading = { a: Boolean(a), b: Boolean(b) };
    st.pix = { a: undefined, b: undefined };
    st.diffKey = '';
    st.diffError = '';
    st.pointer = null;
    st.blink = 'a';
    for (const el of [imgA, imgB, imgB2]) el.removeAttribute('src');
    render();
    if (a) loadSide('a', a, token);
    if (b) loadSide('b', b, token);
  }

  async function loadSide(side, source, token) {
    try {
      const url = await source.load();
      if (token !== st.token) return;
      if (typeof url !== 'string' || !url) throw new Error('No image address was given.');
      const el = new Image();
      el.src = url;
      try {
        await el.decode();
      } catch {
        if (token !== st.token) return;
        // A GET address that answered an error (404, 409 "This source changed after validation.")
        // reaches here too: show the service's sentence rather than a decoding failure.
        const why = await explainFailure(url);
        throw new Error(why || 'The browser could not decode this image.');
      }
      if (token !== st.token) return;
      if (!el.naturalWidth || !el.naturalHeight) throw new Error('The image has no size.');
      st.img[side] = { el, url, w: el.naturalWidth, h: el.naturalHeight };
    } catch (error) {
      if (token !== st.token) return;
      st.err[side] = reasonOf(error);
    } finally {
      if (token === st.token) {
        st.loading[side] = false;
        settled();
      }
    }
  }

  function settled() {
    if (st.loading.a || st.loading.b) return;
    const A = st.img.a;
    const B = st.img.b;
    if (A) imgA.src = A.url;
    if (B) {
      imgB.src = B.url;
      imgB2.src = B.url;
    }
    const G = geo();
    if (G && !(st.prev && st.prev.w === G.w && st.prev.h === G.h)) st.fit = true;
    if (G) st.prev = { w: G.w, h: G.h };
    if (A && B && st.mode === 'diff') ensureDiff();
    render();
  }

  /** The image whose pixels define the frame. */
  function geo() {
    return st.img.a ?? st.img.b ?? null;
  }

  function shown() {
    if (st.loading.a || st.loading.b) return '';
    if (st.img.a && st.img.b) return 'both';
    return st.img.a ? 'a' : st.img.b ? 'b' : '';
  }

  const labelOf = (side) => st.sources[side]?.label || (side === 'a' ? 'Image A' : 'Image B');
  const chip = (side) => {
    const role = st.sources[side]?.role;
    return role ? roleChip(role, labelOf(side)) : h('span', { class: 'role' }, labelOf(side));
  };
  const noteOf = (side) => {
    const note = st.sources[side]?.note;
    return typeof note === 'string' ? note.trim() : '';
  };
  /** Label plus its provenance note (when the caller gave one). */
  const caption = (side) => {
    const note = noteOf(side);
    return note ? [chip(side), h('span', { class: 'compare__labelnote' }, note)] : chip(side);
  };
  /** Label of a side whose load failed, in that side's corner; the full sentence is in the alert too. */
  const failCaption = (side) => [chip(side), h('span', { class: 'compare__labelnote compare__labelnote--err' }, `Not loaded: ${st.err[side]}`)];
  const failures = () => ['a', 'b'].filter((s) => st.err[s]).map((s) => `${labelOf(s)} could not be loaded: ${st.err[s]}`);

  // ── pixels ──
  function readPixels(side) {
    if (st.pix[side] !== undefined) return st.pix[side];
    const im = st.img[side];
    let pixels = null;
    try {
      const canvas = document.createElement('canvas');
      canvas.width = im.w;
      canvas.height = im.h;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(im.el, 0, 0);
      pixels = ctx.getImageData(0, 0, im.w, im.h);
    } catch {
      // Tainted canvas or out of memory: the values are simply not shown.
      pixels = null;
    }
    st.pix[side] = pixels;
    return pixels;
  }

  function ensureDiff() {
    const A = st.img.a;
    const B = st.img.b;
    if (!A || !B) return;
    const key = `${st.token}:${st.amp}`;
    if (st.diffKey === key) return;
    st.diffKey = key;
    const pa = readPixels('a');
    const pb = readPixels('b');
    if (!pa || !pb) {
      st.diffError = 'This browser did not allow reading the pixels, so the difference cannot be computed.';
      return;
    }
    st.diffError = '';
    try {
      diffCanvas.width = A.w;
      diffCanvas.height = A.h;
      const out = new ImageData(A.w, A.h);
      computeDiff(pa.data, A.w, A.h, pb.data, B.w, B.h, st.amp, out.data);
      diffCanvas.getContext('2d').putImageData(out, 0, 0);
    } catch (error) {
      st.diffError = `The difference could not be computed: ${error.message}`;
    }
  }

  // ── rendering ──
  function setMessage(key, node) {
    if (key === msgKey) return; // do not re-announce an unchanged status
    msgKey = key;
    msg.hidden = !node;
    mount(msg, node);
  }

  function render() {
    const show = shown();
    const both = show === 'both';
    const mode = both ? st.mode : show ? 'single' : 'none';
    body.dataset.mode = mode;

    const loadingAny = st.loading.a || st.loading.b;
    if (!st.sources.a && !st.sources.b) setMessage(`empty`, h('p', { class: 'compare__empty' }, emptyText));
    else if (loadingAny) setMessage(`loading:${st.token}`, h('p', { class: 'compare__wait' }, h('span', { class: 'spinner compare__spin', 'aria-hidden': 'true' }), 'Loading image…'));
    else if (!show) setMessage(`fail:${st.token}`, h('div', { class: 'compare__fail' }, failures().map((text) => h('p', null, text))));
    else setMessage('', null);

    const problems = show ? failures() : [];
    if (mode === 'diff' && st.diffError) problems.push(st.diffError);
    const alertKey = problems.join('\n');
    if (alertKey !== alertEl.dataset.key) {
      // Only touch the live region when its text really changes.
      alertEl.dataset.key = alertKey;
      alertEl.hidden = !problems.length;
      mount(alertEl, problems.map((text) => h('p', null, text)));
    }

    const showA = show === 'a' || (both && (st.mode === 'wipe' || st.mode === 'side' || (st.mode === 'blink' && st.blink === 'a')));
    const showB = show === 'b' || (both && (st.mode === 'wipe' || (st.mode === 'blink' && st.blink === 'b')));
    frame1.hidden = !show;
    view2.hidden = mode !== 'side';
    view1.tabIndex = show ? 0 : -1; // nothing to zoom or pan: keep it out of the Tab order
    imgA.hidden = !showA;
    layerB.hidden = !showB;
    diffCanvas.hidden = mode !== 'diff' || Boolean(st.diffError);
    divider.hidden = mode !== 'wipe';
    if (mode !== 'wipe') layerB.style.clipPath = '';

    // Labels: A always in the left corner, B in the right one, so position itself says which is which.
    // A side that failed keeps its corner and says so; a side that was never given has no label.
    let left = null;
    let right = null;
    if (mode === 'diff') left = 'Difference, computed in this browser';
    else if (both) {
      left = showA ? caption('a') : null;
      right = showB ? caption('b') : null;
    } else if (show === 'a') {
      left = caption('a');
      right = st.err.b ? failCaption('b') : null;
    } else if (show === 'b') {
      if (st.err.a) {
        left = failCaption('a');
        right = caption('b');
      } else left = caption('b');
    }
    mount(labelA, left);
    mount(labelB, right);
    mount(label2, mode === 'side' ? caption('b') : null);
    // Alone in its view, a label may use the width (side by side, difference, one image).
    labelA.classList.toggle('compare__label--wide', !right);
    labelA.hidden = !labelA.firstChild;
    labelB.hidden = !labelB.firstChild;
    label2.hidden = !label2.firstChild;
    divider.dataset.a = st.sources.a?.role || '';
    divider.dataset.b = st.sources.b?.role || '';

    modeSeg.hidden = !both || modes.length < 2;
    for (const [m, el] of modeButtons) el.setAttribute('aria-pressed', String(m === st.mode));
    blinkGroup.hidden = mode !== 'blink';
    blinkState.textContent = `Shown: ${labelOf(st.blink)}`;
    ampGroup.hidden = mode !== 'diff';
    for (const [k, el] of ampButtons) el.setAttribute('aria-pressed', String(k === st.amp));
    interpGroup.hidden = !show;
    zoomGroup.hidden = !show;
    for (const [k, el] of Object.entries(interpButtons)) el.setAttribute('aria-pressed', String(k === st.interp));

    const keys = 'Plus and minus zoom, 0 fits, 1 shows 100%, arrow keys pan';
    const what =
      mode === 'single' ? labelOf(show) : both ? `${labelOf('a')} and ${labelOf('b')}, ${MODE_LABEL[st.mode].toLowerCase()}` : 'no image';
    view1.setAttribute('aria-label', `Image view: ${mode === 'side' ? labelOf('a') : what}. ${keys}${mode === 'blink' ? ', Space switches image' : ''}.`);
    view2.setAttribute('aria-label', `Image view: ${labelOf('b')}. ${keys}.`);

    renderSizes();
    renderPointer();
    layout();
  }

  function renderSizes() {
    const parts = [];
    for (const side of ['a', 'b']) {
      if (!st.sources[side]) continue;
      const im = st.img[side];
      const text = im ? fmt.dims(im.w, im.h) : st.err[side] ? 'Not loaded' : '';
      parts.push(h('span', { class: 'compare__fact', title: noteOf(side) || null }, chip(side), h('span', { class: 'num' }, text)));
    }
    if (st.img.a && st.img.b) {
      const relation = sizeRelation(st.img.a.w, st.img.a.h, st.img.b.w, st.img.b.h);
      parts.push(
        h(
          'span',
          { class: 'compare__fact', title: `Width and height of ${labelOf('a')} divided by those of ${labelOf('b')}` },
          h('span', { class: 'muted' }, 'Size ratio'),
          h('span', { class: 'num' }, relation),
        ),
      );
    }
    mount(sizesEl, parts);
  }

  function renderPointer() {
    const G = geo();
    const p = st.pointer;
    if (!p || !G || !shown()) {
      pointerEl.replaceChildren();
      return;
    }
    const primary = st.img.a ? 'a' : 'b';
    const parts = [
      h('span', { class: 'compare__fact num', title: `Column and row in ${labelOf(primary)}, counted from 0 at the top left` }, `Pixel ${p.x}, ${p.y}`),
    ];
    for (const side of ['a', 'b']) {
      const im = st.img[side];
      if (!im) continue;
      const pixels = readPixels(side);
      if (!pixels) continue;
      const x = side === primary ? p.x : mapIndex(p.x, G.w, im.w);
      const y = side === primary ? p.y : mapIndex(p.y, G.h, im.h);
      const i = (y * im.w + x) * 4;
      const d = pixels.data;
      parts.push(h('span', { class: 'compare__fact' }, chip(side), h('span', { class: 'num' }, `R ${d[i]} G ${d[i + 1]} B ${d[i + 2]}`)));
    }
    mount(pointerEl, parts);
  }

  function layout() {
    const G = geo();
    if (!G || frame1.hidden) return;
    const vw = view1.clientWidth;
    const vh = view1.clientHeight;
    if (!vw || !vh) return; // not laid out yet; the ResizeObserver calls again
    const lim = scaleLimits(vw, vh, G.w, G.h);
    st.scale = st.fit ? lim.fit : clamp(st.scale, lim.min, lim.max);
    const fw = G.w * st.scale;
    const fh = G.h * st.scale;
    // Whole CSS pixels for the corner so pixelated squares do not shimmer while panning.
    st.pan = { x: Math.round(clampAxis(st.pan.x, fw, vw)), y: Math.round(clampAxis(st.pan.y, fh, vh)) };
    for (const frame of [frame1, frame2]) {
      frame.style.width = `${fw}px`;
      frame.style.height = `${fh}px`;
      frame.style.transform = `translate(${st.pan.x}px, ${st.pan.y}px)`;
    }
    if (!divider.hidden) {
      // The wipe position is a percent of the viewport, so the divider stays reachable at any zoom.
      const cut = clamp((vw * st.wipe) / 100 - st.pan.x, 0, fw);
      layerB.style.clipPath = `inset(0 0 0 ${cut}px)`;
      divider.style.left = `${st.wipe}%`;
      const now = Math.round(st.wipe);
      divider.setAttribute('aria-valuenow', String(now));
      divider.setAttribute('aria-valuetext', `${now}% from the left`);
    }
    // Labels sit in the top corners of the visible part of the frame.
    const top = `${Math.max(st.pan.y, 0) + 6}px`;
    const left = `${Math.max(st.pan.x, 0) + 6}px`;
    Object.assign(labelA.style, { top, left });
    Object.assign(label2.style, { top, left });
    Object.assign(labelB.style, { top, right: `${Math.max(vw - (st.pan.x + fw), 0) + 6}px` });

    // "Pixels" means nearest neighbour where an image is enlarged on the screen, so its pixels show
    // as squares. Where it is shown smaller than its pixels (a 4096 px image at Fit), nearest
    // neighbour would drop pixels and alias, which misleads a person judging quality: the
    // browser's smooth downscale is used there instead.
    const dpr = window.devicePixelRatio || 1;
    const renderingFor = (im) => (st.interp === 'pixels' && im && st.scale * dpr * (G.w / im.w) > 1 + 1e-9 ? 'pixelated' : 'auto');
    imgA.style.imageRendering = renderingFor(st.img.a);
    diffCanvas.style.imageRendering = renderingFor(st.img.a);
    imgB.style.imageRendering = renderingFor(st.img.b);
    imgB2.style.imageRendering = renderingFor(st.img.b);

    zoomText.textContent = zoomLabel(st.fit, st.scale);
    fitButton.setAttribute('aria-pressed', String(st.fit));
    for (const [k, el] of presetButtons) {
      el.setAttribute('aria-pressed', String(!st.fit && Math.abs(st.scale - k) < 1e-6));
      el.disabled = k > lim.max + 1e-9;
    }
    zoomInButton.disabled = st.scale >= lim.max - 1e-9;
    zoomOutButton.disabled = st.scale <= lim.min + 1e-9;
  }

  // ── actions ──
  function setMode(mode) {
    st.mode = mode;
    if (mode === 'diff') ensureDiff();
    render();
  }

  function setAmp(k) {
    st.amp = k;
    ensureDiff();
    render();
  }

  function setInterp(kind) {
    st.interp = kind;
    render();
  }

  function flip() {
    st.blink = st.blink === 'a' ? 'b' : 'a';
    render();
  }

  function setFit() {
    st.fit = true;
    layout();
  }

  function center() {
    return { x: view1.clientWidth / 2, y: view1.clientHeight / 2 };
  }

  function zoomTo(target, point = center()) {
    const G = geo();
    if (!G || !view1.clientWidth) return;
    const lim = scaleLimits(view1.clientWidth, view1.clientHeight, G.w, G.h);
    const next = clamp(target, lim.min, lim.max);
    st.pan = { x: zoomAt(point.x, st.pan.x, st.scale, next), y: zoomAt(point.y, st.pan.y, st.scale, next) };
    st.scale = next;
    st.fit = false;
    layout();
    renderPointer();
  }

  function zoomStep(dir) {
    zoomTo(stepScale(st.scale, dir));
  }

  function panBy(dx, dy) {
    st.pan = { x: st.pan.x + dx, y: st.pan.y + dy };
    layout();
  }

  // ── pointer, wheel, keyboard ──
  const local = (event, view) => {
    const r = view.getBoundingClientRect();
    return { x: event.clientX - r.left, y: event.clientY - r.top };
  };

  function trackPointer(event, view) {
    const G = geo();
    if (!G || !shown()) return;
    const p = local(event, view);
    const x = pixelAt(p.x, st.pan.x, st.scale, G.w);
    const y = pixelAt(p.y, st.pan.y, st.scale, G.h);
    const next = x < 0 || y < 0 ? null : { x, y };
    if (next?.x === st.pointer?.x && next?.y === st.pointer?.y) return;
    st.pointer = next;
    renderPointer();
  }

  for (const view of [view1, view2]) {
    // Fingers on this view, on the phone edition only. One drags the image; two pinch it: the distance
    // between them sets the zoom about the point between them, and moving both together moves the image.
    // On the desktop page no finger is ever noted here, so the map stays empty, every branch below that
    // asks it is false, and mouse and touch behave as they did before the pinch existed (each new pointer
    // takes over the drag).
    const fingers = new Map();
    let pinch = null;
    const span = () => {
      const [a, b] = [...fingers.values()];
      const r = view.getBoundingClientRect();
      return { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, x: (a.x + b.x) / 2 - r.left, y: (a.y + b.y) / 2 - r.top };
    };
    on(view, 'pointerdown', (event) => {
      if ((event.button !== 0 && event.button !== 1) || !geo() || !shown()) return;
      event.preventDefault(); // no native image drag, no text selection, no middle-button autoscroll
      view.focus({ preventScroll: true });
      view.setPointerCapture(event.pointerId);
      if (PHONE && event.pointerType === 'touch') {
        fingers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        if (fingers.size === 2) {
          drag = null;
          pinch = span();
          view.dataset.dragging = '';
          return;
        }
        if (fingers.size > 2) return;
      }
      drag = { id: event.pointerId, x: event.clientX, y: event.clientY, px: st.pan.x, py: st.pan.y };
      view.dataset.dragging = '';
    });
    on(view, 'pointermove', (event) => {
      if (fingers.has(event.pointerId)) fingers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (pinch && fingers.size === 2) {
        const G = geo();
        if (!G || !view1.clientWidth) return;
        const now = span();
        const lim = scaleLimits(view1.clientWidth, view1.clientHeight, G.w, G.h);
        const next = clamp(st.scale * (now.d / pinch.d), lim.min, lim.max);
        st.pan = { x: zoomAt(pinch.x, st.pan.x, st.scale, next) + now.x - pinch.x, y: zoomAt(pinch.y, st.pan.y, st.scale, next) + now.y - pinch.y };
        st.scale = next;
        st.fit = false;
        pinch = now;
        layout();
        return;
      }
      if (drag && drag.id === event.pointerId) {
        st.pan = { x: drag.px + event.clientX - drag.x, y: drag.py + event.clientY - drag.y };
        layout();
      }
      trackPointer(event, view);
    });
    const end = (event) => {
      if (fingers.delete(event.pointerId) && pinch) {
        pinch = null;
        // The finger that stays goes on dragging from where the image is now.
        const [rest] = [...fingers.entries()];
        drag = rest ? { id: rest[0], x: rest[1].x, y: rest[1].y, px: st.pan.x, py: st.pan.y } : null;
        if (!drag) delete view.dataset.dragging;
        return;
      }
      if (!drag || drag.id !== event.pointerId) return;
      drag = null;
      delete view.dataset.dragging;
    };
    on(view, 'pointerup', end);
    on(view, 'pointercancel', end);
    on(view, 'pointerleave', () => {
      if (drag) return;
      st.pointer = null;
      renderPointer();
    });
    on(
      view,
      'wheel',
      (event) => {
        if (!geo() || !shown()) return;
        event.preventDefault();
        const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? view.clientHeight : 1;
        const dy = clamp(event.deltaY * unit, -300, 300);
        zoomTo(st.scale * Math.exp(-dy * 0.002), local(event, view));
      },
      { passive: false },
    );
    on(view, 'dblclick', (event) => {
      if (!geo() || !shown()) return;
      if (st.fit) zoomTo(1, local(event, view));
      else setFit();
    });
    on(view, 'keydown', (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey || !geo() || !shown()) return;
      const step = (event.shiftKey ? 0.5 : 0.1) * Math.min(view.clientWidth, view.clientHeight);
      switch (event.key) {
        case '+':
        case '=':
          zoomStep(1);
          break;
        case '-':
        case '_':
          zoomStep(-1);
          break;
        case '0':
          setFit();
          break;
        case '1':
          zoomTo(1);
          break;
        // An arrow moves the view that way over the image, so the image moves the other way.
        case 'ArrowLeft':
          panBy(step, 0);
          break;
        case 'ArrowRight':
          panBy(-step, 0);
          break;
        case 'ArrowUp':
          panBy(0, step);
          break;
        case 'ArrowDown':
          panBy(0, -step);
          break;
        case ' ':
          if (shown() !== 'both' || st.mode !== 'blink') return;
          flip();
          break;
        default:
          return;
      }
      event.preventDefault();
    });
  }

  // ── wipe divider ──
  function moveWipe(event) {
    const r = view1.getBoundingClientRect();
    if (!r.width) return;
    st.wipe = clamp(((event.clientX - r.left) / r.width) * 100, 0, 100);
    layout();
  }
  on(divider, 'pointerdown', (event) => {
    if (event.button !== 0) return;
    event.stopPropagation(); // not a pan
    event.preventDefault();
    divider.focus({ preventScroll: true });
    divider.setPointerCapture(event.pointerId);
    wipeDrag = event.pointerId;
    moveWipe(event);
  });
  on(divider, 'pointermove', (event) => {
    if (wipeDrag === event.pointerId) moveWipe(event);
  });
  const endWipe = (event) => {
    if (wipeDrag === event.pointerId) wipeDrag = null;
  };
  on(divider, 'pointerup', endWipe);
  on(divider, 'pointercancel', endWipe);
  on(divider, 'dblclick', (event) => event.stopPropagation());
  on(divider, 'keydown', (event) => {
    const moves = { ArrowLeft: -2, ArrowDown: -2, ArrowRight: 2, ArrowUp: 2, PageDown: -10, PageUp: 10 };
    if (event.key in moves) st.wipe = clamp(st.wipe + moves[event.key], 0, 100);
    else if (event.key === 'Home') st.wipe = 0;
    else if (event.key === 'End') st.wipe = 100;
    else return; // zoom keys bubble to the viewport
    event.preventDefault();
    event.stopPropagation();
    layout();
  });

  // Re-fit (in Fit) or re-clamp (zoomed) whenever the stage changes size.
  const resize = new ResizeObserver(() => layout());
  resize.observe(body);

  render();

  return {
    setSources,
    destroy() {
      st.token += 1; // loads still in flight are ignored
      resize.disconnect();
      listeners.abort();
      st.pix = { a: null, b: null };
      st.img = { a: null, b: null };
      // Addresses came from the caller's load() (object URLs or GET addresses); this component
      // created none, so it revokes none.
      root.remove();
    },
  };
}
