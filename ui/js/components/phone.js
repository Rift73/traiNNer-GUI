// "Open on phone" dialog on GET/POST /api/phone-access: the private link that opens this trainer on a
// phone signed in to the same Tailscale account, as a QR code the service drew and as text.
//
// Rules of this part:
//   - Nothing is asked of the service before the dialog opens, after it closes or while the page is
//     hidden. While it is open it follows the service by itself: a start that is under way, a PC whose
//     Tailscale is not connected, a link that stopped working, a service that stopped answering.
//   - Three actions, one for each kind of asking. What the dialog does by itself sends 'retry': the
//     service spaces those out as it sees fit. The person's Check now or Try again sends 'check': the
//     service tries at once. Neither turns on what was turned off, so a dialog left open in one window
//     cannot undo a choice made elsewhere since (the command line): the answer is then 'off', and the
//     dialog shows that. 'on' is sent by Turn on phone access alone, the one press that means it.
//   - The link stays inside the dialog: it is not put in a title, a toast, the address, the console or a
//     stored preference. The QR code is the service's own image, held only while the dialog is open.
//
//   openPhoneDialog() -> Promise<void> when closed

import { h, mount as fill, uid } from '../dom.js';
import { api } from '../api.js';
import { openDialog, loading, cropmarks, copyText } from '../ui.js';

// How long after an answer the service is asked again, in ms. A ready link is looked at again because
// asking is also what makes the service test that the link still works.
const EVERY = { starting: 1500, retry: 5000, ready: 8000, offline: 5000 };
// Shown edge of the code, in CSS px (css/phone.css draws it at this size).
const CODE_SIZE = 220;
// The states in which phone access is on but there is no link.
const DOWN = ['disconnected', 'unavailable', 'stopped'];

// ───────── pure helpers (exported for checks) ─────────

function isHttps(url) {
  try {
    return new URL(url).protocol === 'https:';
  } catch {
    return false;
  }
}

/**
 * What the dialog shows for one answer of the service. A state it does not know reads as 'unavailable',
 * and so does 'ready' without a link: no link is ever made up here.
 * `auto`: the service says trying again by itself is worth it (`retryable: true`).
 * `manual`: the person may ask for another try (everything but `retryable: false`).
 * @returns {{kind: 'ready'|'starting'|'off'|'disconnected'|'stopped'|'unavailable', url: string, detail: string, auto: boolean, manual: boolean}}
 */
export function phoneView(answer) {
  const state = typeof answer?.state === 'string' ? answer.state : '';
  const detail = typeof answer?.detail === 'string' ? answer.detail.trim().slice(0, 400) : '';
  if (state === 'ready') {
    const url = typeof answer.url === 'string' ? answer.url.trim() : '';
    if (isHttps(url)) return { kind: 'ready', url, detail: '', auto: false, manual: false };
    return { kind: 'unavailable', url: '', detail: 'The service said phone access is ready but gave no link.', auto: false, manual: true };
  }
  const kind = ['starting', 'off', ...DOWN].includes(state) ? state : 'unavailable';
  const down = DOWN.includes(kind);
  return { kind, url: '', detail, auto: down && answer?.retryable === true, manual: down && answer?.retryable !== false };
}

// The one button of each state: what it says, what it says while it works, and what it sends.
// 'on' turns phone access on and belongs to the off state alone. 'check' starts it again at once where
// it is already on, and leaves it off where it was turned off meanwhile. '' only asks the service for
// its state again: after a request that failed, the state decides which button comes next.
const ACTIONS = {
  off: { label: 'Turn on phone access', working: 'Turning on…', send: 'on' },
  disconnected: { label: 'Check now', working: 'Checking…', send: 'check' },
  unavailable: { label: 'Try again', working: 'Trying again…', send: 'check' },
  stopped: { label: 'Try again', working: 'Trying again…', send: 'check' },
  failure: { label: 'Try again', working: 'Trying again…', send: '' },
};

/** The button a shown state has; null when there is nothing for the person to press. */
export function phoneAction(shown) {
  if (!shown) return null;
  if (DOWN.includes(shown.kind) && !shown.manual) return null;
  return ACTIONS[shown.kind] ?? null;
}

// What a screen reader hears when the state changes, and what is said when a press changed nothing.
const SAID = {
  ready: 'Phone access is ready. Scan the code or copy the link.',
  starting: 'Starting phone access.',
  off: 'Phone access is turned off.',
  stopped: 'Phone access is not running.',
  disconnected: 'Tailscale is not connected on this PC.',
  unavailable: 'Phone access could not start.',
  failure: 'Phone access could not be checked.',
};
const UNCHANGED = {
  disconnected: 'Checked just now: Tailscale is still not connected on this PC.',
  unavailable: 'Tried again just now: phone access still could not start.',
  stopped: 'Tried again just now: phone access is still not running.',
  failure: 'Tried again just now: still no answer.',
};

// ───────── dialog ─────────

/** @returns {Promise<void>} */
export function openPhoneDialog() {
  const controller = new AbortController();
  const { signal } = controller;
  let closedNow = false;
  let timer = 0;
  let sayTimer = 0;
  let busy = false; // one request at a time
  let queued = null; // what the person pressed while a request was out
  let view = null; // the last answer, as phoneView()
  let failure = null; // or the request that failed,
  let failedSend = ''; // and what it had sent
  let shown = null; // what the panel was last built for
  let shownKey = '';
  let pressed = null; // the action the person pressed, until the service settles on an answer
  let firstAnswer = true;
  let answeredAt = 0;
  // The code as last drawn by the service: the link it is for and its object URL.
  let code = { url: '', src: '' };
  let drawCode = null; // draws the code into the ready panel now shown

  const panel = h('div', { class: 'phone__panel' }, loading('Checking phone access…'));
  // Announced, not shown: the panel says the same to the eye.
  const said = h('p', { class: 'visually-hidden', role: 'status' });
  // Shown at the foot: what a press led to when the panel itself did not change.
  const note = h('p', { class: 'phone__note muted', 'aria-hidden': 'true' });
  // Stable nodes, changed in place: a button keeps its focus while it works.
  const actionButton = h('button', { class: 'btn btn--primary', type: 'button', hidden: true, onclick: () => press() });
  const closeButton = h('button', { class: 'btn btn--primary', type: 'button' }, 'Close');

  function say(text, show = false) {
    note.textContent = show ? text : '';
    // Emptied first, so that the same sentence said twice is announced twice.
    said.textContent = '';
    clearTimeout(sayTimer);
    sayTimer = setTimeout(() => {
      said.textContent = text;
    }, 60);
  }

  // ── the QR code ──

  function dropCode() {
    if (code.src) URL.revokeObjectURL(code.src);
    code = { url: '', src: '' };
  }

  function codeStage(url) {
    const image = h('img', {
      class: 'phone__code',
      alt: 'QR code of the link to this trainer',
      width: CODE_SIZE,
      height: CODE_SIZE,
      hidden: true,
      // Sharp squares when the service's image is smaller than it is shown; smooth when it is larger.
      onload: () => image.classList.toggle('phone__code--crisp', image.naturalWidth < CODE_SIZE * (window.devicePixelRatio || 1)),
      onerror: () => trouble('the image could not be read.'),
    });
    const plate = h('div', { class: 'phone__plate' }, image, cropmarks());
    const missing = h('p', { class: 'phone__nocode', hidden: true });
    let drawing = false;

    function trouble(reason) {
      plate.hidden = true;
      missing.hidden = false;
      missing.textContent = `The code could not be drawn: ${reason} The link beside it still works.`;
    }

    function put(src) {
      missing.hidden = true;
      plate.hidden = false;
      image.src = src;
      image.hidden = false;
    }

    // Called again with every answer that says 'ready': a code that could not be drawn is tried again.
    drawCode = async () => {
      if (drawing || !image.isConnected) return;
      if (code.url === url && code.src) {
        if (image.getAttribute('src') !== code.src) put(code.src);
        return;
      }
      drawing = true;
      try {
        const blob = await api.phoneAccessQr({ signal });
        if (closedNow) return;
        dropCode();
        code = { url, src: URL.createObjectURL(blob) };
        if (image.isConnected) put(code.src);
      } catch (error) {
        if (error?.name !== 'AbortError' && image.isConnected) trouble(error.message);
      } finally {
        drawing = false;
      }
    };
    return h('div', { class: 'stage phone__stage' }, plate, missing);
  }

  // ── what each state shows ──

  const lead = (text) => h('h3', { class: 'phone__lead' }, text);
  const waiting = (text) => h('p', { class: 'phone__wait' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), text);
  const reason = (text) => h('div', { class: 'notice notice--err' }, h('div', { class: 'notice__body' }, text || 'The service gave no reason.'));

  const CONTENT = {
    ready(now) {
      const captionId = uid('phone-link');
      // Text, not a field: a field would cut a long link short, and the whole link is what is checked.
      const link = h('span', { class: 'mono phone__link', 'aria-labelledby': captionId }, now.url);
      const copy = h(
        'button',
        {
          class: 'btn',
          type: 'button',
          onclick: async () => {
            if (await copyText(now.url)) {
              say('Copied the link.', true);
              return;
            }
            const range = document.createRange();
            range.selectNodeContents(link);
            const selection = window.getSelection();
            selection?.removeAllRanges();
            selection?.addRange(range);
            say('The link could not be copied. It is selected: press Ctrl+C.', true);
          },
        },
        'Copy link',
      );
      return h(
        'div',
        { class: 'phone__ready' },
        codeStage(now.url),
        h(
          'div',
          { class: 'phone__how' },
          lead('Scan this code with your phone'),
          h(
            'ol',
            { class: 'phone__steps' },
            h('li', null, 'On the phone, open the Tailscale app (it is in the App Store) and connect with the same account as this PC.'),
            h('li', null, "Point the phone's camera at the code and tap the link that appears."),
            h('li', null, 'On an iPhone, keep it one tap away: tap Share in Safari, then Add to Home Screen.'),
          ),
          h('div', { class: 'field' }, h('p', { class: 'field__label', id: captionId }, 'Or send the link to the phone'), h('div', { class: 'phone__linkrow' }, link, copy)),
          h(
            'p',
            { class: 'field__help' },
            'Only devices signed in to Tailscale with the same account as this PC can open the link, and only while traiNNer GUI is running here. It is not on the public internet.',
          ),
        ),
      );
    },

    starting() {
      return h(
        'div',
        { class: 'phone__state' },
        h('div', { class: 'phone__wait' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), lead('Starting phone access…')),
        h('p', null, 'The PC is opening a private Tailscale link to this trainer. This can take up to half a minute.'),
      );
    },

    off() {
      return h(
        'div',
        { class: 'phone__state' },
        lead('Phone access is turned off'),
        h('p', null, 'It is off for this trainer, and it stays off until you turn it on.'),
        h(
          'p',
          null,
          'Turning it on gives devices signed in to Tailscale with the same account as this PC a private link to this trainer. It stays on for later launches.',
        ),
      );
    },

    // On, but not running: between two tries, or while an earlier link is still being closed.
    stopped(now) {
      return h(
        'div',
        { class: 'phone__state' },
        lead('Phone access is not running'),
        now.auto
          ? waiting('It is being started again. Leave this open: the code shows as soon as the link is ready.')
          : h('p', null, now.manual ? 'Try again starts it.' : 'Close traiNNer GUI and open it again to start it.'),
      );
    },

    // The service's own reason is left out here: whichever it is, what to do is the two steps below.
    disconnected(now) {
      return h(
        'div',
        { class: 'phone__state' },
        lead('Connect Tailscale on this PC'),
        h('p', null, 'The phone reaches this trainer through Tailscale, and Tailscale is not connected on this PC.'),
        h(
          'ol',
          { class: 'phone__steps' },
          h('li', null, 'Open Tailscale on this PC. On Windows it is the Tailscale icon near the clock, in the taskbar.'),
          h('li', null, 'Sign in, or switch it to Connected.'),
        ),
        now.auto ? waiting('Leave this open: it keeps checking, and shows the code once Tailscale is connected. Check now looks at once.') : null,
      );
    },

    // The service says what is wrong and what to do about it.
    unavailable(now) {
      return h('div', { class: 'phone__state' }, lead('Phone access could not start'), reason(now.detail));
    },

    failure(now) {
      if (now.offline) {
        return h(
          'div',
          { class: 'phone__state' },
          lead('The trainer service is not answering'),
          waiting('Phone access cannot be checked until it does. This looks again every few seconds.'),
        );
      }
      return h(
        'div',
        { class: 'phone__state' },
        lead(now.started ? 'Phone access could not be started' : 'Phone access could not be checked'),
        reason(now.detail),
        // The page is newer than the service that is running: it does not know the endpoint.
        now.status === 404 ? h('p', null, 'The trainer service that is running does not offer this yet. Close traiNNer GUI and open it again.') : null,
      );
    },
  };

  // ── drawing ──

  /** @returns {boolean} true when the panel was built again */
  function render() {
    const now = failure ? { kind: 'failure', detail: failure.message, status: failure.status, offline: failure.offline, started: Boolean(failedSend) } : view;
    if (!now) return false;
    let rebuilt = false;
    // Each new try begins as 'starting'. It does not take down the instructions or the reason the
    // person is reading: they stay until the try ends in something else.
    const hold = now.kind === 'starting' && DOWN.includes(shown?.kind);
    if (!hold) {
      const key = JSON.stringify([now.kind, now.url, now.detail, now.auto, now.manual, now.status, now.offline, now.started]);
      if (key !== shownKey) {
        // Focus inside a panel that goes away would fall out of the dialog's order.
        const focusInside = panel.contains(document.activeElement);
        shownKey = key;
        shown = now;
        drawCode = null;
        fill(panel, CONTENT[now.kind](now));
        if (focusInside) closeButton.focus();
        say(SAID[now.kind]);
        rebuilt = true;
      }
    }
    if (shown?.kind === 'ready') drawCode?.();
    renderAction();
    return rebuilt;
  }

  function renderAction() {
    const action = pressed ?? phoneAction(shown);
    if (!action) {
      // A hidden button cannot hold the focus: hand it on first.
      if (document.activeElement === actionButton) closeButton.focus();
      actionButton.hidden = true;
      closeButton.classList.add('btn--primary');
      return;
    }
    actionButton.hidden = false;
    actionButton.textContent = pressed ? action.working : action.label;
    // Not disabled while it works: a disabled button would drop the focus it holds.
    if (pressed) actionButton.setAttribute('aria-disabled', 'true');
    else actionButton.removeAttribute('aria-disabled');
    closeButton.classList.remove('btn--primary');
  }

  // ── talking to the service ──

  /** @param {''|'on'|'check'|'retry'} [send] absent: only look, and give way to a request that is out */
  async function ask(send) {
    if (closedNow) return;
    if (busy) {
      if (send !== undefined) queued = send;
      return;
    }
    busy = true;
    clearTimeout(timer);
    let answer = null;
    let error = null;
    try {
      answer = send ? await api.setPhoneAccess(send, { signal }) : await api.phoneAccess({ signal });
    } catch (caught) {
      error = caught;
    }
    busy = false;
    if (closedNow || error?.name === 'AbortError') return;
    if (queued !== null) {
      // The person pressed meanwhile: what they asked for decides, not this older answer.
      const next = queued;
      queued = null;
      ask(next);
      return;
    }
    take(answer, error, send || '');
  }

  function take(answer, error, sent) {
    answeredAt = Date.now();
    const wasPressed = pressed;
    if (error) {
      failure = error;
      failedSend = sent;
      pressed = null;
    } else {
      failure = null;
      const next = phoneView(answer);
      const first = firstAnswer;
      firstAnswer = false;
      // Opening this is asking for the link. What stopped phone access earlier (Tailscale not yet
      // connected when the trainer was opened) may have been put right since, so the service is asked
      // to try once before anything is said.
      if (first && next.manual) {
        ask('retry');
        return;
      }
      view = next;
      if (next.kind !== 'starting') pressed = null;
    }
    const rebuilt = render();
    if (wasPressed && !pressed && !rebuilt && UNCHANGED[shown?.kind]) say(UNCHANGED[shown.kind], true);
    plan();
  }

  /** What happens next without the person; nothing while the page is hidden. */
  function plan() {
    clearTimeout(timer);
    if (closedNow || document.hidden) return;
    if (failure) {
      if (failure.offline) timer = setTimeout(() => ask(), EVERY.offline);
      return;
    }
    if (view.kind === 'starting') timer = setTimeout(() => ask(), EVERY.starting);
    else if (view.kind === 'ready') timer = setTimeout(() => ask(), EVERY.ready);
    else if (view.auto) timer = setTimeout(() => ask('retry'), EVERY.retry);
  }

  function press() {
    if (pressed) return;
    const action = phoneAction(shown);
    if (!action) return;
    pressed = action;
    if (shown.kind === 'off') {
      // Said at once: the request to start it is on its way.
      failure = null;
      view = { kind: 'starting', url: '', detail: '', auto: false, manual: false };
    }
    render();
    ask(action.send);
  }

  // The person comes back to this window, often from Tailscale itself: look at once instead of waiting.
  function wake() {
    if (closedNow || document.hidden || busy || Date.now() - answeredAt < 1200) return;
    ask(!failure && view?.manual ? 'retry' : undefined);
  }

  function visibility() {
    if (document.hidden) clearTimeout(timer);
    else wake();
  }

  const { closed, close } = openDialog({
    title: 'Open on phone',
    body: h('div', { class: 'phone' }, panel, said),
    foot: () => [note, actionButton, closeButton],
  });
  closeButton.addEventListener('click', () => close());
  document.addEventListener('visibilitychange', visibility);
  window.addEventListener('focus', wake);
  ask();

  return closed.then(() => {
    closedNow = true;
    clearTimeout(timer);
    clearTimeout(sayTimer);
    controller.abort();
    document.removeEventListener('visibilitychange', visibility);
    window.removeEventListener('focus', wake);
    dropCode();
  });
}
