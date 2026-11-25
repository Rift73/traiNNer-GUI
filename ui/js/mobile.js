// Phone edition (mobile.html). The application is the same one the desktop page runs: this module only
// adds what a phone needs around it, then loads js/main.js unchanged.
//   - where the person is, as attributes on <html> that css/mobile*.css lays the screens out by
//     (one pane at a time instead of two side by side), and a link back to the level above;
//   - the chosen tab kept in view in a strip wider than the screen, the next step that would only repeat
//     the open page, and the run's name in the top bar once the page's own has scrolled away;
//   - room for the on-screen keyboard;
//   - the colour of the browser's own bars.
// Nothing here talks to the service, and nothing here is loaded by index.html.

import { persist } from './store.js';

const root = document.documentElement;
const app = document.querySelector('.app');

// ───────── first open ─────────
// This edition keeps its own preferences (store.js). A phone shows one pane of the recipe at a time, and
// the first thing wanted on it is usually how training is going.

if (persist.get('recipe-layout', null) !== 'source') persist.set('recipe-layout', 'form');
if (persist.get('recipe-source-wrap', null) === null) persist.set('recipe-source-wrap', true);
if (!/^#\/\w/.test(location.hash) && persist.get('view', null) === null) history.replaceState(null, '', '#/runs');

// ───────── where the person is ─────────
// #/runs                         home: the queue block and the run list
// #/runs/<run>                   item: one run                      back to Runs
// #/runs/compare/<a>[/<b>]       item: two runs side by side        back to the first run
// #/runs/queue|analysis|…        part: that part's list             back to Runs
// #/runs/queue|analysis|…/<id>   item: one entry, job, study        back to the part's list
// #/recipe, #/data               home

const PARTS = { queue: 'Queue', analysis: 'Analysis jobs', studies: 'Studies', curricula: 'Curricula' };

function placeOf(hash) {
  const [, screen = '', ...rest] = String(hash).replace(/^#/, '').split('/');
  const segments = rest.filter(Boolean);
  const place = { screen, level: 'home', part: '', back: null };
  if (screen !== 'runs' || !segments.length) return place;
  const [first, second] = segments;
  if (PARTS[first]) {
    place.part = first;
    place.level = second ? 'item' : 'part';
    place.back = second ? { href: `#/runs/${first}`, label: PARTS[first] } : { href: '#/runs', label: 'Runs' };
  } else if (first === 'compare') {
    place.part = 'compare';
    place.level = 'item';
    // Only a managed run has a run page to go back to.
    place.back = second && !/^(job|tb)(:|%3a)/i.test(second) ? { href: `#/runs/${second}`, label: 'Run' } : { href: '#/runs', label: 'Runs' };
  } else {
    place.part = 'run';
    place.level = 'item';
    place.back = { href: '#/runs', label: 'Runs' };
  }
  return place;
}

const backText = document.createElement('span');
const back = document.createElement('a');
back.className = 'mback';
back.hidden = true;
back.append(backText);
app.prepend(back);

// The list a person came from keeps its place: a pane that is not shown does not keep its scroll offset
// in every browser, so the offset is noted while the list is in view and put back when it returns.
const HOME_LIST = '.runs__list';
let homeScroll = 0;
document.addEventListener(
  'scroll',
  (event) => {
    if (root.dataset.level === 'home' && event.target instanceof Element && event.target.matches(HOME_LIST)) homeScroll = event.target.scrollTop;
  },
  true,
);

let lastKey = '';
function place() {
  const p = placeOf(location.hash);
  const key = `${p.screen}/${p.level}/${p.part}`;
  root.dataset.screen = p.screen;
  root.dataset.level = p.level;
  root.dataset.part = p.part;
  back.hidden = !p.back;
  if (p.back) {
    back.href = p.back.href;
    backText.textContent = p.back.label;
    back.setAttribute('aria-label', `Back to ${p.back.label}`);
  }
  if (key === lastKey) return;
  lastKey = key;
  if (p.screen === 'runs' && p.level === 'home') {
    requestAnimationFrame(() => {
      const list = document.querySelector(HOME_LIST);
      if (list) list.scrollTop = homeScroll;
    });
  }
}
place();
window.addEventListener('hashchange', place);

// ───────── one run: a tab that was tapped gets the screen ─────────
// A run's page scrolls as one: its facts, the tabs, the open tab. Tapping a tab brings the tabs to the
// top edge, so the tab's content (a log, a chart, a sheet of images) has the whole screen under them.

const calm = window.matchMedia?.('(prefers-reduced-motion: reduce)');

document.addEventListener('click', (event) => {
  const tab = event.target instanceof Element ? event.target.closest('.runs__part > .tabs [role="tab"]') : null;
  if (!tab) return;
  const strip = tab.closest('.tabs');
  const scroller = strip.parentElement;
  // Not strip.offsetTop: a strip that is already held at the top reports where it is held.
  const natural = () => {
    let top = 0;
    for (let el = strip.previousElementSibling; el; el = el.previousElementSibling) top += el.offsetHeight;
    return top;
  };
  let touched = false;
  const touch = () => {
    touched = true;
  };
  scroller.addEventListener('touchstart', touch, { once: true, passive: true });
  requestAnimationFrame(() => {
    const top = natural();
    scroller.scrollTo({ top, behavior: calm?.matches || scroller.scrollTop > top ? 'auto' : 'smooth' });
  });
  // The page redrawing under a glide can stop it short; the place is then set outright.
  setTimeout(() => {
    scroller.removeEventListener('touchstart', touch);
    if (touched || !strip.isConnected) return;
    const top = natural();
    if (Math.abs(scroller.scrollTop - top) > 2) scroller.scrollTop = top;
  }, 520);
});

// ───────── tab strips: the chosen tab in view, and a sign that there are more ─────────
// A strip wider than the screen is moved sideways (never the page: so not scrollIntoView) until the
// chosen tab is in its middle, at once when the strip appears and gliding when another tab is chosen.
// data-more says on which side more tabs wait; css/mobile.css fades that edge.

const viewEl = document.getElementById('view');

function markEdges(strip) {
  const max = strip.scrollWidth - strip.clientWidth;
  const start = strip.scrollLeft > 1;
  const end = max - strip.scrollLeft > 1;
  const more = start && end ? 'both' : start ? 'start' : end ? 'end' : '';
  if (more && strip.dataset.more !== more) strip.dataset.more = more;
  else if (!more && strip.dataset.more) delete strip.dataset.more;
}

function centreTab(tab, glide) {
  const strip = tab.closest('.tabs');
  if (!strip || !strip.clientWidth) return;
  // From the boxes on screen, so it holds whichever element the tab's offsets are measured from.
  const offset = tab.getBoundingClientRect().left - strip.getBoundingClientRect().left - strip.clientLeft;
  const max = strip.scrollWidth - strip.clientWidth;
  const left = Math.max(0, Math.min(max, Math.round(strip.scrollLeft + offset - (strip.clientWidth - tab.offsetWidth) / 2)));
  if (Math.abs(strip.scrollLeft - left) >= 1) strip.scrollTo({ left, behavior: glide && !calm?.matches ? 'smooth' : 'auto' });
  markEdges(strip);
}

const chosenTab = (strip) => strip.querySelector('[role="tab"][aria-selected="true"]');

// A strip's size decides both: first shown (it may be drawn while hidden), turned, or a tab renamed.
const stripSizes = new WeakMap();
const stripWatch =
  typeof ResizeObserver === 'function'
    ? new ResizeObserver((entries) => {
        for (const { target } of entries) {
          const first = !stripSizes.get(target);
          stripSizes.set(target, target.clientWidth);
          const tab = first && chosenTab(target);
          if (tab) centreTab(tab, false);
          else markEdges(target);
        }
      })
    : null;
const stripsSeen = new WeakSet();

document.addEventListener(
  'scroll',
  (event) => {
    if (event.target instanceof Element && event.target.matches('.tabs')) markEdges(event.target);
  },
  true,
);

// ───────── the run's name in the top bar ─────────
// Deep in a tab, the name at the head of the run's page has scrolled away. The bar then shows it (a copy
// for the eye only: the page's heading stays the heading), while that heading is out of view under the
// tabs that are held at the top.

const barTitle = document.createElement('span');
barTitle.className = 'mtitle';
barTitle.setAttribute('aria-hidden', 'true');
app.append(barTitle);

let titleFor = null; // the heading being watched
let titleSeen = null;

// A run's name is one long word ("2x_DUAL_LIGHT_C128G4D16_PU21_GAN"). Wider than the screen, it would
// break before its last letter; with a place to break after each separator it breaks there. (<wbr> adds
// no text, so the page still finds the name it wrote; a new name is written plain and is marked again.)
function breakName(heading) {
  const text = heading.textContent;
  if (heading.childElementCount || !/[_.-]./.test(text)) return;
  const parts = text.split(/([_.-]+)/); // a run of separators at every odd place
  const nodes = [];
  for (let i = 0; i < parts.length; i += 2) {
    if (i) nodes.push(document.createElement('wbr'));
    nodes.push(parts[i] + (parts[i + 1] ?? ''));
  }
  heading.replaceChildren(...nodes);
}

function watchTitle() {
  const heading = root.dataset.part === 'run' && viewEl ? viewEl.querySelector('.runs__name') : null;
  const name = heading?.textContent ?? '';
  if (heading) breakName(heading);
  if (barTitle.textContent !== name) barTitle.textContent = name;
  if (heading === titleFor) return;
  titleSeen?.disconnect();
  titleSeen = null;
  titleFor = heading;
  delete root.dataset.titled;
  if (!heading || typeof IntersectionObserver !== 'function') return;
  // The heading leads the page, so it has left the top edge before the held tabs could cover it.
  titleSeen = new IntersectionObserver(
    (entries) => {
      const shown = entries[entries.length - 1].isIntersecting;
      if (shown || root.dataset.part !== 'run') delete root.dataset.titled;
      else root.dataset.titled = '';
    },
    { root: heading.closest('.runs__part') },
  );
  titleSeen.observe(heading);
}

// One watcher for the screen area: strips that appear, a tab that is chosen, the run's page drawn again.
// Redraws come in bursts (a log, a chart), so the look over the screen is made once per frame.
let viewLook = 0;
function lookOverView() {
  viewLook = 0;
  if (!viewEl) return;
  for (const strip of viewEl.querySelectorAll('.tabs')) {
    if (stripsSeen.has(strip)) continue;
    stripsSeen.add(strip);
    stripWatch?.observe(strip);
    const tab = chosenTab(strip);
    if (tab) centreTab(tab, false);
    else markEdges(strip);
  }
  watchTitle();
}

if (viewEl) {
  new MutationObserver((records) => {
    for (const r of records) {
      if (r.type === 'attributes' && r.target.getAttribute('aria-selected') === 'true' && r.target.matches('[role="tab"]')) centreTab(r.target, true);
    }
    if (!viewLook && records.some((r) => r.type === 'childList')) viewLook = requestAnimationFrame(lookOverView);
  }).observe(viewEl, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-selected'] });
}

// ───────── the next step, on the page it leads to ─────────
// A link to the page that is open is marked (data-here) and css/mobile.css leaves it out. "Running:
// <run>" with Open run, on that run's own page, then says nothing the page's head does not: <html> gets
// data-next-here and the whole line steps aside upright. Only there, and only when the line has no other
// action: "The queue is paused" keeps its sentence and Continue the queue on the entry it points to.
// Any other message brings the line back, because it is looked at each time it is drawn.

const nextEl = document.getElementById('next');
const decoded = (text) => {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
};

function nextHere() {
  const actions = nextEl ? [...nextEl.querySelectorAll('.btn')] : [];
  const at = decoded(location.hash);
  let elsewhere = 0;
  for (const action of actions) {
    const here = action.matches('a[href]') && decoded(action.getAttribute('href')) === at;
    action.toggleAttribute('data-here', here);
    if (!here) elsewhere += 1;
  }
  if (actions.length && !elsewhere && root.dataset.part === 'run') root.dataset.nextHere = '';
  else delete root.dataset.nextHere;
}

if (nextEl) new MutationObserver(nextHere).observe(nextEl, { childList: true });
window.addEventListener('hashchange', () => {
  nextHere();
  watchTitle();
});

// Ticking a series can add a chart above the list of series. Browsers that do not hold the page still
// for that (Safari) would move the row from under the finger, so the row is put back where it was.
// (Listening on the way down, so the place is noted before the page's own handler redraws.)
document.addEventListener(
  'change',
  (event) => {
    const input = event.target instanceof Element && event.target.closest('.runs__picker-item') ? event.target : null;
    const scroller = input?.closest('.runs__part');
    if (!scroller) return;
    const before = input.getBoundingClientRect().top;
    // A chart's notes arrive a moment after the chart, so the row is held for half a second, or until
    // the person moves the page themselves.
    const until = performance.now() + 500;
    let held = true;
    const release = () => {
      held = false;
    };
    scroller.addEventListener('touchstart', release, { once: true, passive: true });
    scroller.addEventListener('wheel', release, { once: true, passive: true });
    const settle = () => {
      // The list may have been drawn again: the same series is then another element with the same id.
      const now = input.isConnected ? input : input.id ? document.getElementById(input.id) : null;
      if (!held || !now) return;
      const moved = now.getBoundingClientRect().top - before;
      if (Math.abs(moved) > 1) scroller.scrollTop += moved;
      if (performance.now() < until) requestAnimationFrame(settle);
      else {
        scroller.removeEventListener('touchstart', release);
        scroller.removeEventListener('wheel', release);
      }
    };
    requestAnimationFrame(settle);
  },
  true,
);

// ───────── the options form: a new section starts at its top ─────────
// On this edition the whole recipe screen scrolls as one (css/mobile-screens.css), so the form's own
// pane, which the form resets when another section is chosen, is no longer what moves.

function formToTop(event) {
  const from = event.target instanceof Element ? event.target.closest('.form__navitem, .form__picker, .form__search') : null;
  const scroller = from?.closest('.recipe');
  const body = scroller?.querySelector('.recipe__body');
  if (!body) return;
  requestAnimationFrame(() => {
    const top = scroller.scrollTop + body.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
    if (scroller.scrollTop > top) scroller.scrollTop = top;
  });
}
document.addEventListener('click', (event) => {
  if (event.target instanceof Element && event.target.closest('.form__navitem')) formToTop(event);
});
document.addEventListener('change', (event) => {
  if (event.target instanceof Element && event.target.matches('.form__picker')) formToTop(event);
});
document.addEventListener('input', (event) => {
  if (event.target instanceof Element && event.target.matches('.form__search')) formToTop(event);
});

// ───────── the on-screen keyboard ─────────
// The keyboard covers the lower part of the page without resizing it. While a field is being typed in,
// the shell takes the height that is still visible and the two bars at the bottom step aside, so the
// field and what is around it stay in view.

const viewport = window.visualViewport;
const typing = (el) => el instanceof HTMLElement && (el.isContentEditable || el.matches('input:not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="button"]):not([type="file"]), textarea, select'));

let arrangedFor = null; // the field the page was last moved for

function fitKeyboard() {
  if (!viewport) return;
  const field = document.activeElement;
  const open = window.innerHeight - viewport.height > 140 && typing(field);
  if (!open) {
    arrangedFor = null;
    if (root.dataset.keyboard) {
      delete root.dataset.keyboard;
      root.style.removeProperty('--vvh');
      root.style.removeProperty('--kb');
    }
    return;
  }
  // --vvh: the height still visible. --kb: what is covered below it (a sheet rests on that).
  root.dataset.keyboard = 'open';
  root.style.setProperty('--vvh', `${Math.round(viewport.height)}px`);
  root.style.setProperty('--kb', `${Math.max(0, Math.round(window.innerHeight - viewport.height - viewport.offsetTop))}px`);
  // The sizes follow the keyboard; the page itself is moved once per field. The browser may have
  // pushed the page up to show the field: the shell now fits, so that is undone and the field is
  // brought into view inside its own pane instead.
  if (arrangedFor === field) return;
  arrangedFor = field;
  if (window.scrollY) window.scrollTo(0, 0);
  requestAnimationFrame(() => field.scrollIntoView?.({ block: 'nearest' }));
}
// Never while a finger is at work. A tap is a press, a change of focus, then the click, and the browser
// looks again at what lies under the finger before the click. Fitting the shell in between (focus has
// just left the field, so the sheet would drop back by the height of the keyboard) puts another control
// there, and the tap presses that one. So from the moment a finger goes down until its click has been
// delivered, nothing is fitted; the shell is fitted once, right after, to whatever is true by then.
let pressing = false;
let settle = 0;
let watchdog = 0;

function refit() {
  if (!pressing) fitKeyboard();
}

function pressed() {
  pressing = true;
  clearTimeout(settle);
  // A lifted finger that this page never hears of (its target was redrawn away under it) must not leave
  // the shell unfitted for good.
  clearTimeout(watchdog);
  watchdog = setTimeout(() => released(0), 5000);
}

/** The finger is up: fit after `wait` ms, which is the time its click may still take to arrive. */
function released(wait) {
  if (!pressing) return;
  clearTimeout(settle);
  settle = setTimeout(() => {
    clearTimeout(watchdog);
    pressing = false;
    // What is true now decides: the field still typed in, the keyboard gone, a sheet closed by the tap.
    fitKeyboard();
  }, wait);
}

if (viewport) {
  viewport.addEventListener('resize', refit);
  viewport.addEventListener('scroll', refit);
  document.addEventListener('focusin', refit);
  document.addEventListener('focusout', () => setTimeout(refit, 60));

  // Both families of events are heard, so one of them always reports the end of a touch.
  const early = { capture: true, passive: true };
  document.addEventListener('touchstart', pressed, early);
  document.addEventListener('pointerdown', pressed, early);
  document.addEventListener('touchend', () => released(350), early);
  document.addEventListener('pointerup', () => released(350), early);
  document.addEventListener('touchcancel', () => released(0), early);
  // A finger that starts to scroll gives up its pointer but is still down: its touchend reports the end.
  document.addEventListener('pointercancel', (event) => event.pointerType !== 'touch' && released(0), early);
  // The click has arrived: when its own handlers are done, nothing is left to protect.
  document.addEventListener('click', () => released(0), true);
  window.addEventListener('blur', () => released(0));
}

// ───────── the height of the screen area ─────────
// A part that is one tall thing with its own scrolling (a log, the YAML text) is given exactly the room
// between the bars, whatever the next-step line takes at the moment.

if (viewEl && typeof ResizeObserver === 'function') {
  new ResizeObserver(() => root.style.setProperty('--view-h', `${viewEl.clientHeight}px`)).observe(viewEl);
}

// ───────── the browser's bars follow the theme ─────────

const barColour = document.querySelector('meta[name="theme-color"]');
if (barColour) {
  const sync = () => {
    const colour = getComputedStyle(root).getPropertyValue('--raised').trim();
    if (colour) barColour.setAttribute('content', colour);
  };
  new MutationObserver(sync).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  sync();
}

// ───────── the application ─────────

import('./main.js').then(place, (error) => {
  console.error(error);
  const view = document.getElementById('view');
  if (view) view.textContent = `The application did not load: ${error.message}`;
});
