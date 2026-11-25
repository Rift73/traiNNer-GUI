// The training queue on the Runs screen: the summary block above the run list, the queue view with one
// entry's detail and lineage, the replay dialog and the "Extend training" dialog.
// An entry is a saved copy of a recipe. Only "Release" (a person pressing it) authorises training;
// nothing here releases an entry by itself.

import { h, mount, fmt, uid } from '../dom.js';
import { api, newRequestId } from '../api.js';
import { queue, entryStatus, ENTRY_ACTIONS, isContinuation, ordinal, placeWords, statusPlaceWords } from '../state/queue.js';
import { runs, runStatus } from '../state/runs.js';
import { confirmForceStop, openAsRecipe, addToQueue, go, runHref, queueHref, studiesHref, curriculaHref } from '../actions.js';
import { recipe } from '../state/recipe.js';
import { toast, openDialog, confirmDialog, confirmOnPhone, issueList, badge, notice, loading, copyText } from '../ui.js';
import { highlightedPre } from '../recipe/yaml-highlight.js';

// ───────── words ─────────

const PAUSED_LINE = 'The queue is paused. Its active run saves a checkpoint and keeps its place.';

const STATE_TEXT = {
  held: 'Saved. It will not start until you release it.',
  pending: 'It starts when the trainer is free, after the released entries ahead of it in line.',
  running: 'Its training run is starting or running.',
  stopping: 'A stop was requested. The training run is shutting down.',
  completed: 'Its training run finished.',
  stopped: 'Its training run was stopped before it finished.',
  failed: 'It did not finish.',
  cancelled: 'It was withdrawn before it started and will not run.',
};

/** Sentence for the toast after an action, and for a refusal. */
const ACTION_WORDS = {
  release: { done: (name) => `Released ${name}.`, failed: (name) => `${name} was not released` },
  hold: { done: (name) => `Put ${name} on hold.`, failed: (name) => `${name} was not put on hold` },
  cancel: { done: (name) => `Cancelled ${name}.`, failed: (name) => `${name} was not cancelled` },
  forceStop: { done: (name) => `Force stop requested for ${name}.`, failed: (name) => `Force stop was refused for ${name}` },
  stop: { done: (name) => `Stop training requested for ${name}.`, failed: (name) => `Training of ${name} was not stopped` },
  continue: { failed: (name) => `${name} was not added to the queue` },
};

/** A phase entry's `name` is its recipe's (the same for every phase): name it by its phase instead. */
const entryName = (entry) => (isPhase(entry) ? phaseWords(entry) : entry?.name || 'this entry');
const shortDigest = (digest) => (typeof digest === 'string' ? digest.slice(0, 12) : '');
const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const allowed = (action, entry) => ENTRY_ACTIONS[action].from.includes(entry?.status);
/** How many iterations a continuation that extends a completed run trains beyond resumeStep, or null. */
function moreIterations(entry) {
  const n = Number(entry?.additionalIterations);
  return entry?.additionalIterations !== undefined && entry?.additionalIterations !== null && Number.isFinite(n) && n > 0 ? n : null;
}
/** "50,000 more iterations", "1 more iteration". */
const moreCount = (n) => `${fmt.int(n)} more iteration${n === 1 ? '' : 's'}`;
/** ", 50,000 more iterations" for an extension; '' otherwise. */
const moreWords = (entry) => (moreIterations(entry) === null ? '' : `, ${moreCount(moreIterations(entry))}`);

/**
 * A replay carries its source run's name: tag it so it does not read as an ordinary entry. A continuation
 * (it carries resumeStep) is tagged as one, not as a replay.
 */
function replayTag(entry) {
  if (entry?.pauseAttempt && entry.runStatus === 'stopped' && Number.isInteger(entry.completedIteration)) {
    return h('span', { class: 'queue__tag' }, `Paused at ${fmt.int(entry.completedIteration)}`);
  }
  if (!entry?.sourceRunId || isPhase(entry)) return null;
  if (isContinuation(entry)) {
    const from = Number.isFinite(Number(entry.resumeStep)) ? ` from iteration ${fmt.int(Number(entry.resumeStep))}` : '';
    return h('span', { class: 'queue__tag', title: `Continues run ${entry.sourceRunId}${from}${moreWords(entry)}` }, 'Continuation');
  }
  return h('span', { class: 'queue__tag', title: `Replay of run ${entry.sourceRunId}` }, 'Replay');
}

/** A controlled-study cell (studyId, studyName, armId, armName, seed): "Lower LR, seed 41". */
function studyWords(entry) {
  if (!entry?.studyId) return '';
  const arm = entry.armName || entry.armId || '';
  const seed = Number.isFinite(entry.seed) ? `seed ${entry.seed}` : '';
  return [arm, seed].filter(Boolean).join(', ');
}

function studyTag(entry) {
  return entry?.studyId ? h('span', { class: 'queue__tag', title: `Study ${entry.studyName || entry.studyId}` }, 'Study') : null;
}

/** Study fields that change what a row shows, for row signatures. */
const studyKey = (entry) => (entry?.studyId ? [entry.studyId, entry.studyName ?? '', entry.armId ?? '', entry.armName ?? '', entry.seed ?? ''] : '');

// ───────── curriculum phases ─────────
// A phase entry: curriculumId (= groupId), phaseId 'phase-N', phaseName, mode, startStep, targetStep,
// updates, weights?, revision, previousEntryId?, replacedBy?, predecessorPhaseId?, edits, handoff?.
// Its `name` is the recipe's name (the same for every phase), so rows lead with "Phase N: <phaseName>".
// After release the service also sets sourceRunId to the handoff's run: that is not a replay.

const MODE_WORDS = { start: 'Start', continue: 'Continue', finetune: 'Fine-tune' };
const isPhase = (entry) => Boolean(entry?.curriculumId);
const phaseNumber = (phaseId) => {
  const match = /^phase-(\d+)$/.exec(String(phaseId ?? ''));
  return match ? Number(match[1]) : null;
};
/** "Phase 2: Fine-tune on faces". */
function phaseWords(entry) {
  if (!isPhase(entry)) return '';
  const n = phaseNumber(entry.phaseId);
  const name = entry.phaseName || '';
  if (n === null) return name || 'Curriculum phase';
  return name ? `Phase ${n}: ${name}` : `Phase ${n}`;
}
const revisionWords = (entry) => (isPhase(entry) && Number.isInteger(entry.revision) && entry.revision > 1 ? `Revision ${entry.revision}` : '');

// Curriculum names are not on the entries: read GET /api/curricula once when a curriculum entry is first
// seen, and again when an unknown curriculum appears. Unknown or unreadable: the tag alone, never the id.
const curriculumNames = new Map();
const curriculumAsked = new Set();
const curriculumListeners = new Set();
let curriculumFetch = null;
const curriculumName = (id) => curriculumNames.get(id) ?? '';
function wantCurriculumNames(entries) {
  const ids = [...new Set(entries.filter(isPhase).map((entry) => entry.curriculumId))];
  if (curriculumFetch || !ids.some((id) => !curriculumNames.has(id) && !curriculumAsked.has(id))) return;
  curriculumFetch = api
    .curricula()
    .then((list) => {
      for (const item of Array.isArray(list) ? list : []) if (item?.id && typeof item.name === 'string' && item.name) curriculumNames.set(item.id, item.name);
    })
    .catch(() => null)
    .finally(() => {
      ids.forEach((id) => curriculumAsked.add(id));
      curriculumFetch = null;
      for (const fn of curriculumListeners) fn();
    });
}

/** "Curriculum" tag; the curriculum's own name is on the link line under the row (curriculumLink). */
function curriculumTag(entry) {
  if (!isPhase(entry)) return null;
  const name = curriculumName(entry.curriculumId);
  return h('span', { class: 'queue__tag', title: name ? `Curriculum ${name}` : null }, 'Curriculum');
}

/** "Revision 2" on a phase row's status line (nothing for the first revision). */
const revisionMeta = (entry) => (revisionWords(entry) ? h('span', { class: 'queue__place' }, revisionWords(entry)) : null);

/** Link to the curriculum, outside a row's own link (links do not nest). */
function curriculumLink(entry) {
  if (!isPhase(entry)) return null;
  const name = curriculumName(entry.curriculumId);
  return h(
    'a',
    { class: 'queue__curriculum-link', href: curriculaHref(entry.curriculumId), dataset: { focus: 'curriculum' }, 'aria-label': name ? `Curriculum ${name}` : `Curriculum of ${phaseWords(entry)}` },
    name ? `Curriculum ${name}` : 'Open the curriculum',
  );
}

/** The current entry of this phase's predecessor among `pool` (queue list and/or queue group), or null. */
function predecessorOf(entry, pool) {
  if (!isPhase(entry) || !entry.predecessorPhaseId) return null;
  const candidates = pool.filter((item) => item.curriculumId === entry.curriculumId && item.phaseId === entry.predecessorPhaseId);
  const current = candidates.filter((item) => !item.replacedBy);
  const list = current.length ? current : candidates;
  return list.sort((a, b) => (b.revision ?? 0) - (a.revision ?? 0))[0] ?? null;
}

/** A held later phase whose predecessor is known not to have completed: the entry it waits for. */
function waitsFor(entry, pool) {
  if (!isPhase(entry) || entry.status !== 'held' || entry.replacedBy || entry.mode === 'start') return null;
  const previous = predecessorOf(entry, pool);
  if (!previous) return null;
  return previous.status === 'completed' && previous.runId ? null : previous;
}

/** "Waits for phase 1 to complete". */
const waitsWords = (previous, entry) => {
  const n = phaseNumber(previous?.phaseId ?? entry.predecessorPhaseId);
  return n === null ? 'Waits for the previous phase to complete' : `Waits for phase ${n} to complete`;
};

/** The revision that replaced this one: its number from the list when known, else the next one (the service numbers them so). */
function replacementWords(entry, pool) {
  const newer = pool.find((item) => item.id === entry.replacedBy);
  const n = Number.isInteger(newer?.revision) ? newer.revision : Number.isInteger(entry.revision) ? entry.revision + 1 : null;
  return n === null ? 'a newer revision' : `revision ${n}`;
}

/** Phase fields that change what a row shows, for row signatures. */
const phaseKey = (entry, pool) =>
  isPhase(entry)
    ? [entry.curriculumId, curriculumName(entry.curriculumId), entry.phaseId ?? '', entry.phaseName ?? '', entry.revision ?? '', entry.replacedBy ?? '', waitsFor(entry, pool)?.id ?? '', entry.replacedBy ? replacementWords(entry, pool) : '']
    : '';

/** Whether to offer an action: never Release on a replaced revision or a phase that waits for its predecessor. */
function offered(action, entry, pool) {
  if (!allowed(action, entry)) return false;
  if (action === 'forceStop') return Boolean(entry.runId && entry.attemptId);
  if (action === 'release' && isPhase(entry) && (entry.replacedBy || waitsFor(entry, pool))) return false;
  return true;
}

/** Line under a phase row: replaced, or waiting for the phase before. */
function phaseRowNote(entry, pool) {
  if (!isPhase(entry)) return null;
  if (entry.replacedBy) {
    return h('p', { class: 'queue__blocked' }, 'Replaced by ', h('a', { href: queueHref(entry.replacedBy), dataset: { focus: 'replaced' } }, replacementWords(entry, pool)));
  }
  const previous = waitsFor(entry, pool);
  return previous ? h('p', { class: 'queue__blocked' }, waitsWords(previous, entry)) : null;
}

/** Entry name with arm and seed for a study cell, and the replay and study tags, for list rows. */
function nameLine(entry) {
  if (isPhase(entry)) {
    // Line 1: the phase (wraps between words, never cut) and the tag. Revision and curriculum name are
    // on the status line and the link line.
    return h(
      'span',
      { class: 'queue__name-line', title: entry.name ? `Recipe ${entry.name}` : null },
      h('span', { class: 'queue__phase-name' }, phaseWords(entry)),
      curriculumTag(entry),
    );
  }
  const cell = studyWords(entry);
  return h(
    'span',
    { class: 'queue__name-line' },
    h('span', { class: 'queue__name' }, entryName(entry)),
    cell ? h('span', { class: 'queue__cell' }, cell) : null,
    replayTag(entry),
    studyTag(entry),
  );
}

function statusBadge(status) {
  const st = entryStatus(status);
  return badge(st.label, st.kind);
}

function timeEl(iso) {
  const text = fmt.dateTime(iso);
  return text ? h('time', { datetime: iso, title: iso }, text) : null;
}

/** A run reference: name when the run list knows it, else the id; status word when known. */
function runLink(runId, rawStatus) {
  const known = runs.byId?.(runId) ?? null;
  const status = rawStatus ?? known?.status;
  const st = status ? runStatus(status) : null;
  return h(
    'span',
    { class: 'queue__run' },
    h('a', { href: runHref(runId), class: 'queue__run-link' }, known?.name || runId),
    st ? badge(st.label, st.kind) : null,
  );
}

// ───────── shared pieces ─────────

/** A control that gives way to another after it worked: "Add to queue" becomes the link to the continuation. */
const FOCUS_NEXT = { 'act-continue': 'continuation', 'act-extend': 'continuation' };

/**
 * Re-render list items in place: an item is rebuilt only when its signature changes, and focus inside a
 * rebuilt item moves to the element with the same data-focus key.
 */
function reconcile(listEl, cache, items, signature, render) {
  const seen = new Set();
  const active = document.activeElement;
  let focusKey = null;
  let focusId = null;
  items.forEach((item, index) => {
    const id = String(item.id);
    seen.add(id);
    const sig = signature(item);
    let slot = cache.get(id);
    if (!slot || slot.sig !== sig) {
      const el = render(item);
      el.dataset.entry = id;
      if (slot) {
        if (slot.el.contains(active)) {
          focusKey = active.dataset?.focus ?? null;
          focusId = id;
        }
        slot.el.replaceWith(el);
      }
      slot = { el, sig };
      cache.set(id, slot);
    }
    if (listEl.children[index] !== slot.el) listEl.insertBefore(slot.el, listEl.children[index] ?? null);
  });
  let lostId = null;
  for (const [id, slot] of cache) {
    if (!seen.has(id)) {
      if (slot.el.contains(active)) lostId = id;
      slot.el.remove();
      cache.delete(id);
    }
  }
  if (focusId && !listEl.contains(document.activeElement)) {
    const el = cache.get(focusId)?.el;
    const target = (focusKey && (el?.querySelector(`[data-focus="${focusKey}"]`) || el?.querySelector(`[data-focus="${FOCUS_NEXT[focusKey]}"]`))) || el?.querySelector('a');
    target?.focus();
  }
  // id of an item that held focus and left this list (it may now be in another list)
  return lostId;
}

/** Put focus on the row link of entry `id` inside `root`, when focus was lost with a removed row. */
function refocusEntry(root, id) {
  if (!id || root.contains(document.activeElement)) return;
  for (const el of root.querySelectorAll('[data-entry]')) {
    if (el.dataset.entry === id) {
      el.querySelector('a')?.focus();
      return;
    }
  }
}

/** Run one action on an entry, with a danger confirmation for cancel and stop. */
async function runAction(action, entry) {
  const name = entryName(entry);
  if (action === 'cancel') {
    const ok = await confirmDialog({
      title: `Cancel ${name}?`,
      body: 'Cancelling cannot be undone. The entry will never start; it stays in the finished list, where its recipe text can still be opened.',
      confirmLabel: 'Cancel entry',
      cancelLabel: 'Keep entry',
      danger: true,
    });
    if (!ok) return null; // declined: nothing was asked of the service
  } else if (action === 'forceStop') {
    if (!(await confirmForceStop(name))) return null;
  } else if (action === 'stop') {
    const ok = await confirmDialog({
      title: `Stop training ${name}?`,
      body: 'The trainer is asked to stop this training run now. The entry does not start again by itself.',
      confirmLabel: 'Stop training',
      cancelLabel: 'Keep training',
      danger: true,
    });
    if (!ok) return null; // declined: nothing was asked of the service
  } else if (action === 'release') {
    // A released entry trains as soon as the trainer is free: the phone edition asks first.
    const ok = await confirmOnPhone({
      title: `Release ${name}?`,
      body: queue.get().paused
        ? 'The queue is paused, so the entry waits until you continue the queue. Then it starts when the trainer is free, after the released entries ahead of it in line.'
        : 'It joins the end of the line and starts when the trainer is free, after the released entries ahead of it.',
      confirmLabel: 'Release',
    });
    if (!ok) return null;
  }
  try {
    const result = action === 'forceStop' ? await runs.forceStop(entry.runId, entry.attemptId) : await queue[action](entry.id);
    if (action === 'forceStop') queue.refresh();
    actionErrors.delete(entry.id);
    let message = ACTION_WORDS[action].done(entryName(result ?? entry));
    if (action === 'release' && queue.get().paused) message += ' The queue is paused, so it waits until you continue the queue.';
    toast(message, { kind: 'ok' });
    return true;
  } catch (error) {
    // The refusal stays beside the button that was pressed (and in any other copy of that entry's
    // controls) until the entry's state changes or an action on it succeeds; the toast repeats it.
    actionErrors.set(entry.id, { action, message: error.message, status: entry.status });
    toast(`${ACTION_WORDS[action].failed(name)}: ${error.message}`, { kind: 'err' });
    queue.refresh();
    return false;
  }
}

/** Last refused action per entry id: {action, message, status at the time}. */
const actionErrors = new Map();
function actionError(entry) {
  const record = actionErrors.get(entry?.id);
  if (!record) return null;
  if (record.status !== entry.status) {
    actionErrors.delete(entry.id);
    return null;
  }
  return record;
}
const actionErrorKey = (entry) => actionError(entry)?.message ?? '';
/** The refusal, written where the action was pressed. */
function actionErrorEl(entry) {
  const record = actionError(entry);
  if (!record) return null;
  return h('p', { class: 'queue__action-error', role: 'alert', dataset: { errorFor: entry.id } }, `${ACTION_WORDS[record.action].failed(entryName(entry))}: ${record.message}`);
}

// Busy buttons use aria-disabled instead of `disabled`: a disabled button drops keyboard focus to the
// page, and the row or detail that holds it is then re-rendered without anywhere to put focus back.
const busyActions = new Set();

/** Button for one action; ignores presses while a request for this entry and action is open. */
function actionButton(action, entry, { small = true, quiet = false } = {}) {
  const danger = ['cancel', 'stop', 'forceStop'].includes(action);
  const cls = ['btn', 'queue__btn', small ? 'btn--sm' : '', danger && !quiet ? 'btn--danger' : '', quiet ? 'btn--quiet' : ''].filter(Boolean).join(' ');
  const key = `${action}:${entry.id}`;
  const button = h(
    'button',
    {
      class: cls,
      type: 'button',
      dataset: { focus: `act-${action}`, busyKey: key },
      'aria-label': `${ENTRY_ACTIONS[action].label}: ${entryName(entry)}`,
      'aria-disabled': busyActions.has(key) ? 'true' : null,
      onclick: async () => {
        if (busyActions.has(key)) return;
        busyActions.add(key);
        button.setAttribute('aria-disabled', 'true');
        try {
          const ok = await runAction(action, entry);
          // When the row or detail was not rebuilt by the refresh, write the refusal beside this button.
          const holder = button.parentElement;
          if (ok === false && holder?.isConnected) {
            const sibling = holder.nextElementSibling;
            if (sibling?.dataset?.errorFor === String(entry.id)) sibling.remove();
            const el = actionErrorEl({ ...entry, status: actionErrors.get(entry.id)?.status ?? entry.status });
            if (el) holder.after(el);
          }
        } finally {
          busyActions.delete(key);
          // The row may have been rebuilt meanwhile (its copy was drawn busy): clear every copy.
          for (const el of document.querySelectorAll('.queue__btn[aria-disabled]')) {
            if (el.dataset.busyKey === key) el.removeAttribute('aria-disabled');
          }
          button.removeAttribute('aria-disabled');
        }
      },
    },
    ENTRY_ACTIONS[action].label,
  );
  return button;
}

// ───────── continuation of a stopped run ─────────

const PAUSED_WAIT = ' The queue is paused, so it waits until you continue the queue.';

/** Where a released continuation stands in the line, as the end of the toast sentence. */
function lineWords(entry) {
  const s = queue.get();
  if (entry.status === 'held') return ' It is on hold and will not start until you release it.';
  if (s.paused) return PAUSED_WAIT;
  const open = queue.open();
  const active = (s.activeId ? queue.byId(s.activeId) : null) ?? open.find((item) => entryStatus(item.status).active) ?? null;
  const released = open.filter((item) => item.status === 'pending');
  const index = released.findIndex((item) => item.id === entry.id);
  const ahead = index >= 0 ? index : released.length;
  if (active && active.id !== entry.id) {
    const doing = active.status === 'stopping' ? 'stopping' : 'training';
    const others = ahead > 0 ? ` and ${fmt.plural(ahead, 'released entry', 'released entries')} ahead of it` : '';
    return ` It starts after ${entryName(active)}, which is ${doing} now${others}.`;
  }
  if (ahead > 0) return ` It is ${ordinal(ahead + 1)} in line, after the released entries ahead of it.`;
  if (runs.active().length) return ' It starts when the run that is training now ends.';
  return ' It is next in line and starts when the trainer is free.';
}

/**
 * Queue a continuation of stopped run `runId`: the same run continues from its newest saved checkpoint
 * when the trainer is free. With `additionalIterations` it extends completed run `runId` instead: a new
 * run continues it from its final training state for that many more iterations. Released on arrival, so
 * the phone edition asks first (`confirm: false` when the caller's own dialog is the question). Toasts the
 * outcome; a refusal only when `toastRefusal`.
 * @param {{runId: string, name?: string, additionalIterations?: number|null, confirm?: boolean, toastRefusal?: boolean}} run
 * @returns {Promise<{ok: true, entry: object, existed: boolean}|{ok: false, message: string, offline: boolean}|null>}
 *   null = the person declined; nothing was asked of the service.
 */
export async function addContinuation({ runId, name, additionalIterations = null, confirm = true, toastRefusal = true }) {
  const label = name || runId || 'this run';
  const extend = additionalIterations !== null && additionalIterations !== undefined;
  if (confirm) {
    const ok = await confirmOnPhone({
      title: `Add ${label} to the queue?`,
      body:
        (extend
          ? `This run resumes from its final saved training state for ${moreCount(Number(additionalIterations))} when the trainer is free, after the released entries ahead of it in line.`
          : 'This run resumes from its newest saved checkpoint when the trainer is free, after the released entries ahead of it in line.') +
        (queue.get().paused ? ' The queue is paused, so it waits until you continue the queue.' : ''),
      confirmLabel: 'Add to queue',
    });
    if (!ok) return null;
  }
  try {
    await queue.refresh();
  } catch {
    // the store keeps what it had; the answer below still says what was queued
  }
  const before = queue.continuationOf(runId)?.id ?? null;
  let entry;
  try {
    entry = extend ? await queue.resume(runId, { additionalIterations }) : await queue.resume(runId);
  } catch (error) {
    if (toastRefusal) toast(`${label} was not added to the queue: ${error.message}`, { kind: 'err' });
    queue.refresh();
    return { ok: false, message: error.message, offline: Boolean(error.offline) };
  }
  const live = queue.byId(entry?.id) ?? entry ?? {};
  // A new continuation arrives released and without a run; one seen open before the press, or one that has
  // moved on (training, stopping, a run of its own), was already there. A held one not seen before is said
  // as added and on hold (the service saves new ones released, so the store was likely stale).
  const existed = (before !== null && live.id === before) || !['pending', 'held'].includes(live.status) || Boolean(live.runId);
  const action = { label: 'Open queue', run: () => go(queueHref(live.id)) };
  // Three sentences: they stay up longer than a one-line toast.
  const timeout = 9000;
  if (existed) {
    const words = statusPlaceWords(live);
    const paused = live.status === 'pending' && queue.get().paused ? PAUSED_WAIT : '';
    // An open extension of the same run keeps the amount it was queued with.
    const theirs = moreIterations(live);
    const other = extend && theirs !== null && theirs !== Number(additionalIterations) ? ` It was queued with ${moreCount(theirs)}; cancel that entry to queue another amount.` : '';
    toast(`${label} is already in the queue to continue (${words}). Nothing was added.${paused}${other}`, { kind: 'info', action, timeout });
  } else if (extend) {
    // Said from the entry the service returned, not from what was typed.
    const step = Number.isFinite(Number(live.resumeStep)) && live.resumeStep !== null && live.resumeStep !== '' ? Number(live.resumeStep) : null;
    const more = moreIterations(live);
    const from = step !== null ? ` from iteration ${fmt.int(step)}` : '';
    const amount = more !== null ? ` for ${moreCount(more)}${step !== null ? ` (new total ${fmt.int(step + more)})` : ''}` : '';
    toast(`Added ${label} to the queue. It continues${from}${amount}.${lineWords(live)}`, { kind: 'ok', action, timeout });
  } else {
    const from = Number.isFinite(Number(live.resumeStep)) ? ` It continues from iteration ${fmt.int(Number(live.resumeStep))}.` : ' It resumes when the trainer is free.';
    toast(`Added ${label} to the queue.${from}${lineWords(live)}`, { kind: 'ok', action, timeout });
  }
  return { ok: true, entry: live, existed };
}

/**
 * What the run detail says about extending this completed run: {available, resumeStep?, reason?}, or null
 * when it says nothing (then the service decides when asked).
 * @param {object|null} run the run detail (GET /api/run)
 * @returns {{available: boolean, resumeStep?: number, reason?: string}|null}
 */
export function extendAvailability(run) {
  const value = run?.extendTraining;
  if (isPlainObject(value) && typeof value.available === 'boolean') return value;
  // Without that word, one case is already plain in the detail: the service refuses a run that ended by
  // early stopping (its patience is used up, so a continuation would stop again at once).
  if (run?.completionReason === 'earlyStopping') {
    return { available: false, reason: 'This run ended by early stopping. Its patience is used up, so a continuation would stop again at once.' };
  }
  return null;
}

function superseded(entry) {
  const owner = queue.forRun(entry.runId);
  return owner && owner.id !== entry.id;
}

/** A finished entry whose stopped run can be continued from its checkpoint (study cells too; not phases). */
function continuable(entry) {
  if (entry?.status !== 'stopped' || !entry.runId || isPhase(entry) || superseded(entry)) return false;
  return entry.runStatus === undefined || entry.runStatus === null || String(entry.runStatus).toLowerCase() === 'stopped';
}

/** A finished entry whose completed run can be extended by more iterations (study cells too; not phases). */
function extendable(entry) {
  if (entry?.status !== 'completed' || !entry.runId || isPhase(entry) || superseded(entry)) return false;
  return entry.runStatus === undefined || entry.runStatus === null || String(entry.runStatus).toLowerCase() === 'completed';
}

/** A finished entry whose run can be followed by a continuation: continued (stopped) or extended (completed). */
const followable = (entry) => continuable(entry) || extendable(entry);

/**
 * The newest finished continuation of run `runId` that really trained (it has a run of its own). One that
 * was cancelled before it started continued nothing.
 */
const continuedBefore = (runId) => queue.continuationsOf(runId).find((item) => !entryStatus(item.status).open && item.runId && item.runId !== runId) ?? null;

/** Continuation fields that change what a row or the facts show, for signatures. */
function continuationKey(entry) {
  if (!followable(entry)) return '';
  const open = queue.continuationOf(entry.runId);
  const done = continuedBefore(entry.runId);
  return [open?.id ?? '', open ? statusPlaceWords(open) : '', open?.additionalIterations ?? '', done?.id ?? '', done?.status ?? ''];
}

/** "Continuation in the queue: released, next" with a link to that entry (", 50,000 more iterations" for an extension). */
function continuationLine(continuation) {
  return h(
    'span',
    null,
    'Continuation in the queue: ',
    h('a', { href: queueHref(continuation.id), dataset: { focus: 'continuation', continuesRun: continuation.sourceRunId ?? '' } }, statusPlaceWords(continuation)),
    moreWords(continuation),
  );
}

/** "Continued before in <entry>": the newest finished continuation, so an older stop is not read as the latest state. */
function continuedBeforeLine(runId) {
  if (queue.continuationOf(runId)) return null;
  const done = continuedBefore(runId);
  if (!done) return null;
  return h(
    'p',
    { class: 'queue__blocked muted' },
    'Continued before in ',
    h('a', { href: queueHref(done.id), dataset: { focus: 'continued' } }, `${entryName(done)} (${entryStatus(done.status).label.toLowerCase()})`),
  );
}

/** "Add to queue" for a stopped entry's run; ignores presses while its request is open. */
function continueButton(entry, { small = true } = {}) {
  const key = `continue:${entry.id}`;
  const name = entryName(entry);
  const words = `Add to queue: continue ${name} from its newest saved checkpoint`;
  const button = h(
    'button',
    {
      class: ['btn', 'queue__btn', small ? 'btn--sm' : ''].filter(Boolean).join(' '),
      type: 'button',
      title: words,
      dataset: { focus: 'act-continue', busyKey: key },
      'aria-label': words,
      'aria-disabled': busyActions.has(key) ? 'true' : null,
      onclick: async () => {
        if (busyActions.has(key)) return;
        busyActions.add(key);
        button.setAttribute('aria-disabled', 'true');
        try {
          const result = await addContinuation({ runId: entry.runId, name });
          if (result?.ok) actionErrors.delete(entry.id);
          else if (result && !result.ok) {
            actionErrors.set(entry.id, { action: 'continue', message: result.message, status: entry.status });
            // When the row or detail was not rebuilt by the refresh, write the refusal beside this button.
            const holder = button.parentElement;
            if (holder?.isConnected) {
              const sibling = holder.nextElementSibling;
              if (sibling?.dataset?.errorFor === String(entry.id)) sibling.remove();
              const el = actionErrorEl(entry);
              if (el) holder.after(el);
            }
          }
        } finally {
          busyActions.delete(key);
          for (const el of document.querySelectorAll('.queue__btn[aria-disabled]')) {
            if (el.dataset.busyKey === key) el.removeAttribute('aria-disabled');
          }
          button.removeAttribute('aria-disabled');
        }
      },
    },
    'Add to queue',
  );
  return button;
}

/** "Extend training…" for a completed entry's run: opens the dialog that asks how many more iterations. */
function extendButton(entry, { small = true } = {}) {
  const name = entryName(entry);
  const words = `Extend training: continue ${name} for more iterations`;
  return h(
    'button',
    {
      class: ['btn', 'queue__btn', small ? 'btn--sm' : ''].filter(Boolean).join(' '),
      type: 'button',
      title: words,
      dataset: { focus: 'act-extend' },
      'aria-label': words,
      onclick: () => openExtendDialog({ runId: entry.runId, name }),
    },
    'Extend training…',
  );
}

/** "Pause the queue" / "Continue the queue", kept in place across updates. */
function pauseControl({ small = true } = {}) {
  let paused = false;
  let busy = false;
  const button = h('button', {
    class: `btn queue__btn${small ? ' btn--sm' : ''}`,
    type: 'button',
    onclick: async () => {
      if (busy) return;
      const next = !paused;
      // Continuing lets released entries start training: the phone edition asks first.
      if (!next && !(await confirmOnPhone({ title: 'Resume the queue?', body: 'The paused run resumes from its checkpoint when the trainer is free.', confirmLabel: 'Resume the queue' }))) return;
      if (busy) return;
      busy = true;
      button.setAttribute('aria-disabled', 'true');
      try {
        await queue.setPaused(next);
        toast(next ? 'Paused the queue. Any active queue run will save and stop.' : 'Resumed the queue. Training continues when the trainer is free.', { kind: 'ok' });
      } catch (error) {
        toast(`The queue was not ${next ? 'paused' : 'continued'}: ${error.message}`, { kind: 'err' });
        queue.refresh();
      } finally {
        busy = false;
        button.removeAttribute('aria-disabled');
      }
    },
  });
  const line = h('p', { class: 'queue__paused', hidden: true }, PAUSED_LINE);
  const update = (value) => {
    paused = Boolean(value);
    const label = paused ? 'Resume the queue' : 'Pause the queue';
    if (button.textContent !== label) button.textContent = label;
    line.hidden = !paused;
  };
  update(false);
  return { button, line, update };
}

/** Loading or error line for the queue store; null when ready. */
function storeProblem(s) {
  if (s.phase === 'error') return notice('err', `The queue could not be read: ${s.error} It is tried again shortly.`);
  if (s.phase !== 'ready') return loading('Reading the queue…');
  return null;
}

// ───────── summary block ─────────

/** Open entries listed in the summary block; the rest are counted on one line. */
const SUMMARY_MAX = 4;

/** "2 released, 1 on hold" for the entries not listed (training ones first, when any). */
function restWords(rest) {
  const training = rest.filter((entry) => entryStatus(entry.status).active).length;
  const released = rest.filter((entry) => entry.status === 'pending').length;
  const held = rest.filter((entry) => entry.status === 'held').length;
  return [training ? `${fmt.int(training)} training` : '', released ? `${fmt.int(released)} released` : '', held ? `${fmt.int(held)} on hold` : '']
    .filter(Boolean)
    .join(', ');
}

/**
 * Block above the run list in the Runs screen's left column.
 * @param {HTMLElement} host
 * @param {{selectedId?: string|null}} options
 * @returns {{setSelected(id: string|null): void, destroy(): void}}
 */
export function mountQueueSummary(host, { selectedId = null } = {}) {
  let selected = selectedId ?? null;
  const titleId = `queue-summary-${Math.random().toString(36).slice(2, 8)}`;
  const count = h('span', { class: 'queue__count' });
  const pause = pauseControl();
  const problem = h('div', { class: 'queue__problem' });
  const list = h('ol', { class: 'queue__list', 'aria-labelledby': titleId });
  const more = h('a', { class: 'queue__more', href: queueHref(), hidden: true });
  const empty = h(
    'p',
    { class: 'queue__empty', hidden: true },
    'Nothing is queued. Use Add to queue on the Recipe screen. ',
    h('a', { href: '#/recipe' }, 'Open the Recipe screen'),
  );
  const root = h(
    'section',
    { class: 'queue__summary', 'aria-labelledby': titleId },
    h('div', { class: 'queue__summary-head' }, h('h2', { class: 'queue__summary-title', id: titleId }, 'Queue ', count), pause.button),
    pause.line,
    problem,
    list,
    more,
    empty,
    h('a', { class: 'queue__all', href: queueHref() }, 'All queue entries'),
  );
  host.appendChild(root);

  const cache = new Map();
  let lastProblem = '';

  const render = () => {
    const s = queue.get();
    const open = queue.open();
    const ready = s.phase === 'ready' || s.entries.length > 0;
    count.textContent = ready ? `${fmt.int(open.length)} open` : '';
    pause.update(s.paused);
    pause.button.hidden = !ready;

    const problemKey = s.phase === 'error' ? `e:${s.error}` : ready ? '' : 'loading';
    if (problemKey !== lastProblem) {
      lastProblem = problemKey;
      mount(problem, ready && s.phase !== 'error' ? null : storeProblem(s));
    }

    empty.hidden = !(ready && open.length === 0);
    // At most SUMMARY_MAX rows (open() order: training first, then released by their place in line, then on
    // hold), so the run list below stays in view; the rest is one line linking to the queue view.
    const shown = open.slice(0, SUMMARY_MAX);
    const pool = s.entries;
    wantCurriculumNames(shown);
    reconcile(
      list,
      cache,
      shown,
      (entry) =>
        JSON.stringify([
          entry.name, entry.status, entry.attemptId, entry.sourceRunId ?? '', studyKey(entry), phaseKey(entry, pool), placeWords(entry, open), entry.blockedReason ?? '',
          actionErrorKey(entry), entry.id === selected,
        ]),
      (entry) => summaryRow(entry, open, pool),
    );
    const rest = open.slice(SUMMARY_MAX);
    const moreText = rest.length ? `and ${fmt.int(rest.length)} more: ${restWords(rest)}` : '';
    more.hidden = !moreText;
    if (more.textContent !== moreText) more.textContent = moreText;
  };

  const summaryRow = (entry, open, pool) => {
    const place = placeWords(entry, open);
    const current = entry.id === selected;
    const action = ['forceStop', 'release', 'hold'].find((key) => offered(key, entry, pool));
    return h(
      'li',
      { class: `queue__row${current ? ' queue__row--current' : ''}` },
      h(
        'a',
        { class: 'queue__link', href: queueHref(entry.id), 'aria-current': current ? 'page' : null, dataset: { focus: 'link' } },
        nameLine(entry),
        h('span', { class: 'queue__meta' }, statusBadge(entry.status), revisionMeta(entry), place ? h('span', { class: 'queue__place' }, place) : null),
      ),
      curriculumLink(entry),
      entry.blockedReason ? h('p', { class: 'queue__blocked' }, entry.blockedReason) : null,
      phaseRowNote(entry, pool),
      action ? h('div', { class: 'queue__row-actions' }, actionButton(action, entry)) : null,
      actionErrorEl(entry),
    );
  };

  const unsubscribe = queue.subscribe(render);
  curriculumListeners.add(render);
  const unwatch = queue.watch();
  render();

  return {
    setSelected(id) {
      const next = id ?? null;
      if (next === selected) return;
      selected = next;
      render();
    },
    destroy() {
      unsubscribe();
      curriculumListeners.delete(render);
      unwatch();
      root.remove();
    },
  };
}

// ───────── queue view ─────────

/**
 * Main area for #/runs/queue and #/runs/queue/<entryId>.
 * @param {HTMLElement} host
 * @param {{entryId?: string|null}} options
 * @returns {{setEntry(id: string|null): void, destroy(): void}}
 */
export function mountQueueView(host, { entryId = null } = {}) {
  let selected = entryId ?? null;
  const pause = pauseControl({ small: false });
  // "Add the open recipe": only while a recipe is open; addToQueue() asks, saves and toasts by itself.
  const addRecipe = h('button', { class: 'btn', type: 'button', hidden: true, onclick: () => addToQueue() }, 'Add the open recipe');
  const syncAddRecipe = () => {
    const r = recipe.get();
    const isOpen = r.phase === 'open';
    addRecipe.hidden = !isOpen;
    const name = r.summary?.name || r.name || '';
    const title = isOpen ? (name ? `Add a copy of ${name} to the queue` : 'Add a copy of the open recipe to the queue') : '';
    if (!title) addRecipe.removeAttribute('title');
    else if (addRecipe.getAttribute('title') !== title) addRecipe.setAttribute('title', title);
  };
  const condition = h('p', { class: 'queue__condition', role: 'status', 'aria-live': 'polite' });
  const problem = h('div', { class: 'queue__problem' });
  const openTitle = h('h2', { class: 'queue__section-title' }, 'Open entries');
  const openList = h('ol', { class: 'queue__list queue__list--wide' });
  // What a move did, for a screen reader (the rows say it to the eye); and the refusal of the last move.
  const moveStatus = h('p', { class: 'visually-hidden', role: 'status', 'aria-live': 'polite' });
  const moveProblem = h('div', { class: 'queue__move-problem' });
  const openEmpty = h(
    'p',
    { class: 'queue__empty', hidden: true },
    'Nothing is queued. Use Add to queue on the Recipe screen. ',
    h('a', { href: '#/recipe' }, 'Open the Recipe screen'),
  );
  const finishedSummary = h('summary', { class: 'queue__finished-summary' }, 'Finished entries');
  const finishedList = h('ol', { class: 'queue__list queue__list--wide' });
  const finished = h('details', { class: 'queue__finished', hidden: true }, finishedSummary, finishedList);
  const detailHost = h('div', { class: 'queue__detail' });

  const root = h(
    'section',
    { class: 'queue', 'aria-label': 'Training queue' },
    h(
      'header',
      { class: 'queue__head' },
      // No separate paused notice here: the condition sentence says it, beside its pause control.
      h('div', { class: 'queue__head-text' }, h('h1', { class: 'queue__title' }, 'Queue'), condition),
      h('div', { class: 'queue__head-actions' }, addRecipe, pause.button),
    ),
    problem,
    h('div', { class: 'queue__body' }, h('div', { class: 'queue__lists' }, openTitle, moveProblem, openList, moveStatus, openEmpty, finished), detailHost),
  );
  host.appendChild(root);

  const openCache = new Map();
  const finishedCache = new Map();
  let lastProblem = '';
  let detail = null;
  let autoOpened = false;

  // ── the line: moving a released entry that waits ──
  // While a row is being dragged, and while the request of a move is out, the lists are left as they are:
  // a poll must not pull a row from under the finger or put a dropped row back for a moment. What the
  // store learned meanwhile is drawn when the hold ends.
  let hold = false;
  let moving = false;
  let drag = null;

  /** Where a waiting released entry stands in the line, when the line has more than one entry; else null. */
  const linePlace = (entry, open) => {
    if (entry.status !== 'pending') return null;
    const line = open.filter((item) => item.status === 'pending');
    const at = line.findIndex((item) => item.id === entry.id);
    return at >= 0 && line.length > 1 ? { first: at === 0, last: at === line.length - 1 } : null;
  };

  /** Ask the service for the new order; the answer (or, after a refusal, a fresh reading) is drawn. */
  const applyOrder = async (expected, order, entry) => {
    const name = entryName(entry);
    moving = true;
    hold = true;
    root.classList.add('queue--moving');
    openList.setAttribute('aria-busy', 'true');
    mount(moveProblem, null);
    moveStatus.textContent = '';
    try {
      await queue.reorder(expected, order);
      const place = placeWords(queue.byId(entry.id) ?? entry);
      moveStatus.textContent = place ? `${name} is now ${place === 'Next' ? 'next in line' : place}.` : `${name} was moved.`;
    } catch (error) {
      // The store was read again before the refusal arrived here: the list below is the line as it is.
      const said = `${name} was not moved: ${error.message}${error.status === 409 ? ' The list was read again and shows the line as it is now.' : ''}`;
      // The toast is seen wherever the list is scrolled to (and is the alert); the line above the list stays.
      const line = notice('err', said);
      line.removeAttribute('role');
      mount(moveProblem, line);
      toast(said, { kind: 'err' });
    } finally {
      moving = false;
      hold = false;
      root.classList.remove('queue--moving');
      openList.removeAttribute('aria-busy');
      render();
    }
  };

  /** "Move up" / "Move down": one place, without dragging. */
  const moveBy = (entry, delta) => {
    if (hold || moving) return;
    const ids = queue.line().map((item) => item.id);
    const from = ids.indexOf(entry.id);
    const to = from + delta;
    if (from < 0 || to < 0 || to >= ids.length) return;
    const order = [...ids];
    order.splice(to, 0, order.splice(from, 1)[0]);
    applyOrder(ids, order, entry);
  };

  const moveButton = (entry, delta, disabled) =>
    h(
      'button',
      {
        class: 'btn btn--sm queue__btn queue__move',
        type: 'button',
        title: delta < 0 ? 'Move up in the line' : 'Move down in the line',
        'aria-label': `${delta < 0 ? 'Move up' : 'Move down'}: ${entryName(entry)}`,
        'aria-disabled': disabled ? 'true' : null,
        dataset: { focus: delta < 0 ? 'act-up' : 'act-down' },
        onclick: () => {
          if (!disabled) moveBy(entry, delta);
        },
      },
      h('span', { 'aria-hidden': 'true' }, delta < 0 ? '↑' : '↓'),
    );

  /** The nearest ancestor that scrolls up and down (the list's pane on a desktop, the page on a phone). */
  const scrollerOf = (el) => {
    for (let node = el.parentElement; node && node !== document.body && node !== document.documentElement; node = node.parentElement) {
      const flow = getComputedStyle(node).overflowY;
      if ((flow === 'auto' || flow === 'scroll') && node.scrollHeight > node.clientHeight) return node;
    }
    return document.scrollingElement ?? document.documentElement;
  };

  /** Place the dragged row under the pointer and make room for it among the other rows of the line. */
  const placeDrag = () => {
    const d = drag;
    if (!d) return;
    const size = d.heights[d.from];
    const lastIndex = d.rows.length - 1;
    const least = d.tops[0] - d.tops[d.from];
    const most = d.tops[lastIndex] + d.heights[lastIndex] - size - d.tops[d.from];
    const dy = Math.min(most, Math.max(least, d.y - d.startY + (d.scroller.scrollTop - d.startScroll)));
    const centre = d.tops[d.from] + dy + size / 2;
    let to = d.from;
    d.rows.forEach((el, i) => {
      if (i === d.from) return;
      const middle = d.tops[i] + d.heights[i] / 2;
      const shift = i < d.from ? (centre < middle ? size : 0) : centre > middle ? -size : 0;
      if (shift > 0) to -= 1;
      else if (shift < 0) to += 1;
      el.style.transform = shift ? `translateY(${shift}px)` : '';
    });
    d.to = to;
    d.row.style.transform = `translateY(${dy}px)`;
  };

  /** One frame of a drag: scroll when the pointer is held near an edge of the scroller, then place the rows. */
  const dragFrame = () => {
    const d = drag;
    if (!d) return;
    const whole = d.scroller === document.scrollingElement || d.scroller === document.documentElement;
    const box = whole ? { top: 0, bottom: window.innerHeight } : d.scroller.getBoundingClientRect();
    const edge = 56;
    const step = d.y < box.top + edge ? -Math.ceil((box.top + edge - d.y) / 6) : d.y > box.bottom - edge ? Math.ceil((d.y - (box.bottom - edge)) / 6) : 0;
    if (step) d.scroller.scrollTop += Math.max(-14, Math.min(14, step));
    placeDrag();
    d.frame = requestAnimationFrame(dragFrame);
  };

  const endDrag = (commit) => {
    const d = drag;
    if (!d) return;
    drag = null;
    cancelAnimationFrame(d.frame);
    d.handle.removeEventListener('pointermove', d.onMove);
    d.handle.removeEventListener('pointerup', d.onUp);
    d.handle.removeEventListener('pointercancel', d.onCancel);
    d.handle.removeEventListener('lostpointercapture', d.onCancel);
    window.removeEventListener('keydown', d.onKey, true);
    try {
      d.handle.releasePointerCapture(d.pointerId);
    } catch {
      // the capture was already gone
    }
    root.classList.remove('queue--dragging');
    d.row.classList.remove('queue__row--dragging');
    for (const el of d.rows) el.style.transform = '';
    if (!commit || d.to === d.from) {
      hold = false;
      render();
      return;
    }
    const order = [...d.ids];
    order.splice(d.to, 0, order.splice(d.from, 1)[0]);
    // The rows stay where they were dropped until the service has answered.
    const after = d.rows[d.rows.length - 1].nextElementSibling;
    const byId = new Map(d.ids.map((entryId, i) => [entryId, d.rows[i]]));
    for (const entryId of order) openList.insertBefore(byId.get(entryId), after);
    applyOrder(d.ids, order, d.entry);
  };

  /** A press on a row's handle (mouse, pen or finger) picks the row up. */
  const beginDrag = (event, entry) => {
    if (drag || hold || moving || !event.isPrimary || (event.pointerType === 'mouse' && event.button !== 0)) return;
    const line = queue.line();
    const rows = line.map((item) => openCache.get(String(item.id))?.el);
    const from = line.findIndex((item) => item.id === entry.id);
    if (from < 0 || line.length < 2 || rows.some((el) => !el?.isConnected)) return;
    event.preventDefault(); // no text selection, and the focus stays where it was
    const handle = event.currentTarget;
    const boxes = rows.map((el) => el.getBoundingClientRect());
    const scroller = scrollerOf(openList);
    const d = {
      entry,
      handle,
      pointerId: event.pointerId,
      row: rows[from],
      rows,
      ids: line.map((item) => item.id),
      tops: boxes.map((box) => box.top),
      heights: boxes.map((box) => box.height),
      from,
      to: from,
      startY: event.clientY,
      y: event.clientY,
      scroller,
      startScroll: scroller.scrollTop,
      frame: 0,
      onMove: (e) => {
        if (e.pointerId === d.pointerId) d.y = e.clientY;
      },
      onUp: (e) => {
        if (e.pointerId !== d.pointerId) return;
        d.y = e.clientY;
        placeDrag();
        endDrag(true);
      },
      onCancel: (e) => {
        if (e.pointerId === d.pointerId) endDrag(false);
      },
      onKey: (e) => {
        if (e.key !== 'Escape') return;
        e.preventDefault();
        e.stopPropagation();
        endDrag(false);
      },
    };
    drag = d;
    hold = true;
    mount(moveProblem, null);
    try {
      handle.setPointerCapture(event.pointerId);
    } catch {
      // without capture the handle still gets the moves made over it; a release elsewhere cancels below
    }
    handle.addEventListener('pointermove', d.onMove);
    handle.addEventListener('pointerup', d.onUp);
    handle.addEventListener('pointercancel', d.onCancel);
    handle.addEventListener('lostpointercapture', d.onCancel);
    window.addEventListener('keydown', d.onKey, true);
    root.classList.add('queue--dragging');
    d.row.classList.add('queue__row--dragging');
    d.frame = requestAnimationFrame(dragFrame);
  };

  /** The grip of a row in the line. Pointer only: the Move up and Move down buttons are the other way. */
  const dragHandle = (entry) =>
    h('span', {
      class: 'queue__handle',
      'aria-hidden': 'true',
      title: 'Drag to move in the line',
      onpointerdown: (event) => beginDrag(event, entry),
      oncontextmenu: (event) => event.preventDefault(),
    });

  const conditionText = (s, open) => {
    const active = s.activeId ? queue.byId(s.activeId) : open.find((entry) => entryStatus(entry.status).active);
    const released = open.filter((entry) => entry.status === 'pending').length;
    const held = open.filter((entry) => entry.status === 'held').length;
    const first = s.paused ? 'The queue is paused, so released entries do not start' : 'Released entries start when the trainer is free';
    const activeWord = !active ? '' : active.status === 'stopping' ? 'stopping' : entryStatus(active.status).active ? 'training' : 'starting';
    const activePart = active ? `${entryName(active)} is ${activeWord}` : 'nothing from the queue is training';
    const counts =
      released === 0 && held === 0
        ? 'no entries are waiting'
        : `${fmt.plural(released, 'entry', 'entries')} released and ${fmt.int(held)} on hold`;
    return `${first}; ${activePart}; ${counts}.`;
  };

  /**
   * Ids of entries that begin a consecutive run of one study's entries in `list` (when that study has
   * more than one entry there). The order is left as it is: the study name is shown once above the run.
   */
  const studyRunStarts = (list) => {
    const counts = new Map();
    for (const entry of list) if (entry.studyId) counts.set(entry.studyId, (counts.get(entry.studyId) ?? 0) + 1);
    const starts = new Set();
    list.forEach((entry, i) => {
      if (entry.studyId && counts.get(entry.studyId) > 1 && list[i - 1]?.studyId !== entry.studyId) starts.add(entry.id);
    });
    return starts;
  };

  const rowSig = (open, starts) => (entry) => {
    const pool = queue.get().entries;
    return JSON.stringify([
      entry.name, entry.status, entry.sourceRunId ?? '', entry.resumeStep ?? '', entry.additionalIterations ?? '', studyKey(entry), phaseKey(entry, pool), starts.has(entry.id), entry.updatedAt,
      placeWords(entry, open), entry.blockedReason ?? '', entry.runId ?? '', entry.runStatus ?? '', continuable(entry), extendable(entry), continuationKey(entry),
      actionErrorKey(entry), entry.id === selected, linePlace(entry, open) ?? '',
    ]);
  };

  const viewRow = (entry, open, starts) => {
    const place = placeWords(entry, open);
    const current = entry.id === selected;
    const st = entryStatus(entry.status);
    const pool = queue.get().entries;
    const actions = st.open ? Object.keys(ENTRY_ACTIONS).filter((action) => offered(action, entry, pool)) : [];
    const canContinue = continuable(entry);
    const canExtend = extendable(entry);
    const follows = canContinue || canExtend;
    const continuation = follows ? queue.continuationOf(entry.runId) : null;
    // A released entry that waits, with others: it can be moved in the line (grip, or the two buttons).
    const inLine = linePlace(entry, open);
    return h(
      'li',
      { class: `queue__row queue__row--wide${current ? ' queue__row--current' : ''}${inLine ? ' queue__row--movable' : ''}` },
      starts.has(entry.id) ? h('p', { class: 'queue__study-head' }, `Study ${entry.studyName || entry.studyId}`) : null,
      h(
        'a',
        { class: 'queue__link', href: queueHref(entry.id), 'aria-current': current ? 'page' : null, dataset: { focus: 'link' } },
        nameLine(entry),
        h(
          'span',
          { class: 'queue__meta' },
          statusBadge(entry.status),
          revisionMeta(entry),
          place ? h('span', { class: 'queue__place' }, place) : null,
          !st.open && entry.updatedAt ? h('span', { class: 'queue__time' }, timeEl(entry.updatedAt)) : null,
        ),
      ),
      curriculumLink(entry),
      entry.blockedReason ? h('p', { class: 'queue__blocked' }, entry.blockedReason) : null,
      phaseRowNote(entry, pool),
      actions.length || inLine
        ? h(
            'div',
            { class: 'queue__row-actions' },
            actions.map((action) => actionButton(action, entry, { quiet: action === 'cancel' })),
            inLine ? h('span', { class: 'queue__moves' }, moveButton(entry, -1, inLine.first), moveButton(entry, 1, inLine.last)) : null,
          )
        : null,
      follows ? continuedBeforeLine(entry.runId) : null,
      continuation ? h('p', { class: 'queue__blocked' }, continuationLine(continuation)) : null,
      canContinue && !continuation ? h('div', { class: 'queue__row-actions' }, continueButton(entry)) : null,
      canExtend && !continuation ? h('div', { class: 'queue__row-actions' }, extendButton(entry)) : null,
      actionErrorEl(entry),
      inLine ? dragHandle(entry) : null,
    );
  };

  const render = () => {
    // A drag or its request is open: the lists wait (see hold above); the entry's own pane does not.
    if (hold) {
      detail?.update();
      return;
    }
    const s = queue.get();
    const ready = s.phase === 'ready' || s.entries.length > 0;
    const open = queue.open();
    const done = queue.finished();
    wantCurriculumNames(s.entries);

    pause.update(s.paused);
    pause.button.hidden = !ready;
    const text = ready ? conditionText(s, open) : '';
    if (condition.textContent !== text) condition.textContent = text;

    const problemKey = s.phase === 'error' ? `e:${s.error}` : ready ? '' : 'loading';
    if (problemKey !== lastProblem) {
      lastProblem = problemKey;
      mount(problem, ready && s.phase !== 'error' ? null : storeProblem(s));
    }

    const openLabel = ready ? `Open entries (${fmt.int(open.length)})` : 'Open entries';
    if (openTitle.textContent !== openLabel) openTitle.textContent = openLabel;
    openEmpty.hidden = !(ready && open.length === 0);
    const openStarts = studyRunStarts(open);
    const lostOpen = reconcile(openList, openCache, open, rowSig(open, openStarts), (entry) => viewRow(entry, open, openStarts));

    finished.hidden = done.length === 0;
    const finishedLabel = `Finished entries (${fmt.int(done.length)})`;
    if (finishedSummary.textContent !== finishedLabel) finishedSummary.textContent = finishedLabel;
    const noStarts = new Set(); // finished entries are newest first, not in line order: no study runs
    const lostDone = reconcile(finishedList, finishedCache, done, rowSig(open, noStarts), (entry) => viewRow(entry, open, noStarts));
    // An entry that finished while a control in its row had focus: follow it into the finished list
    // (open the list so the link can take focus).
    if (lostOpen && finishedCache.has(lostOpen) && !root.contains(document.activeElement)) finished.open = true;
    refocusEntry(root, lostOpen || lostDone);

    // A selected finished entry is shown in the list: open the finished list once for that selection.
    if (selected && !autoOpened) {
      const entry = queue.byId(selected);
      if (entry) {
        autoOpened = true;
        if (!entryStatus(entry.status).open) finished.open = true;
      }
    }

    detail?.update();
  };

  const showDetail = () => {
    detail?.destroy();
    detail = null;
    if (!selected) {
      mount(
        detailHost,
        h(
          'div',
          { class: 'queue__intro' },
          h('h2', { class: 'queue__section-title' }, 'What the queue is'),
          h('p', null, 'Each queue entry is a saved copy of a recipe. An entry on hold never starts by itself.'),
          h('p', null, 'Released entries wait in a line and start one at a time, from the front of the line, when the trainer is free. A newly released entry joins the end.'),
          h('p', null, 'To change the order, drag a released entry by its handle, or use its Move up and Move down buttons. Entries that are training, on hold or finished have no place in the line.'),
          h('p', null, 'Select an entry to see its recipe text, its runs and what you can do with it.'),
        ),
      );
      return;
    }
    detail = mountEntryDetail(detailHost, selected);
  };

  const unsubscribe = queue.subscribe(render);
  curriculumListeners.add(render);
  const unsubscribeRecipe = recipe.subscribe(syncAddRecipe);
  const unwatch = queue.watch();
  syncAddRecipe();
  render();
  showDetail();

  return {
    setEntry(id) {
      const next = id ?? null;
      if (next === selected) return;
      selected = next;
      autoOpened = false;
      render();
      showDetail();
    },
    destroy() {
      endDrag(false);
      unsubscribe();
      curriculumListeners.delete(render);
      unsubscribeRecipe();
      unwatch();
      detail?.destroy();
      detail = null;
      root.remove();
    },
  };
}

// ───────── one entry ─────────

/** Detail of one entry. update() is called on every store change and re-renders only what changed. */
function mountEntryDetail(host, id) {
  const controller = new AbortController();
  const { signal } = controller;
  const titleId = `queue-entry-${Math.random().toString(36).slice(2, 8)}`;
  const facts = h('div', { class: 'queue__facts' });
  const lineage = h('div', { class: 'queue__lineage', 'aria-live': 'polite' }, loading('Reading the replays and runs of this entry…'));
  const recipeBody = h('div', { class: 'queue__recipe-body', 'aria-live': 'polite' }, loading('Reading the saved recipe text…'));
  const openRecipe = h('button', { class: 'btn btn--sm', type: 'button', disabled: true }, 'Open as recipe');
  const copy = h('button', { class: 'btn btn--sm', type: 'button', disabled: true }, 'Copy');
  const lineageTitle = h('h3', { class: 'queue__part-title' }, 'Replays and runs of this entry');
  const lineagePart = h('section', { class: 'queue__part' }, lineageTitle, lineage);
  const recipePart = h(
    'section',
    { class: 'queue__part' },
    h('div', { class: 'queue__part-head' }, h('h3', { class: 'queue__part-title' }, 'Saved recipe text'), h('div', { class: 'row' }, openRecipe, copy)),
    recipeBody,
  );
  const root = h('article', { class: 'queue__entry', 'aria-labelledby': titleId }, facts, lineagePart, recipePart);
  mount(host, root);

  /** The entry as last read: the store's copy wins when present (it is newer), the detail call fills gaps. */
  let fetched = null;
  let yaml = null;
  let missing = null;
  let factsSig = '';
  let groupSig = '';
  let groupId = null;
  let groupController = null;

  const current = () => {
    const live = queue.byId(id);
    return live ? { ...(fetched ?? {}), ...live } : fetched;
  };

  openRecipe.addEventListener('click', () => {
    const entry = current();
    if (yaml) openAsRecipe({ yaml, name: entry?.name ?? '', label: `queue entry ${entryName(entry)}` });
  });
  copy.addEventListener('click', async () => {
    if (!yaml) return;
    const ok = await copyText(yaml);
    toast(ok ? 'Copied the recipe text.' : 'The recipe text could not be copied. Select it and copy it by hand.', { kind: ok ? 'ok' : 'err' });
  });

  /** Entries of this curriculum from the last queue group answer: the predecessor may not be in the queue list. */
  let groupEntries = [];
  /** The curriculum's entries were read (or the read failed): what is not found now is not coming. */
  let groupKnown = false;
  /**
   * A later phase whose phase before it is not on hand yet, while the curriculum's entries are still being
   * read. Release is held back for that moment, so it cannot show up and then go away again.
   */
  const predecessorPending = (entry, all) =>
    isPhase(entry) && entry.status === 'held' && !entry.replacedBy && entry.mode !== 'start' && Boolean(entry.predecessorPhaseId) && Boolean(entry.groupId) && !groupKnown && !predecessorOf(entry, all);
  let pinnedOpen = false;
  /** The queue list's copies (newer) over the group's. */
  const pool = () => {
    const byId = new Map(groupEntries.map((item) => [item.id, item]));
    for (const item of queue.get().entries) byId.set(item.id, item);
    return [...byId.values()];
  };

  /** Rows of a curriculum phase: curriculum, mode, iterations, weights, revision, and the pinned handoff. */
  const phaseRows = (entry) => {
    if (!isPhase(entry)) return [];
    const name = curriculumName(entry.curriculumId);
    const handoff = isPlainObject(entry.handoff) ? entry.handoff : null;
    const files = Array.isArray(handoff?.files) ? handoff.files : [];
    const span = Number.isFinite(entry.startStep) && Number.isFinite(entry.targetStep) ? `${fmt.int(entry.startStep)} to ${fmt.int(entry.targetStep)}${entry.mode === 'finetune' ? ' (new count)' : ''}` : null;
    // A fine-tune can also start from the weights the phase before kept as best on a validation dataset.
    const bestOn = entry.weights && typeof entry.weights === 'object' && typeof entry.weights.bestValidation === 'string' ? entry.weights.bestValidation : null;
    const weights =
      entry.mode === 'finetune' ? (entry.weights === 'ema' ? 'EMA weights' : entry.weights === 'raw' ? 'Raw weights' : bestOn !== null ? `The best-scoring weights on ${bestOn}` : null) : null;
    const pinned = files.length
      ? h(
          'details',
          {
            class: 'queue__pinned',
            open: pinnedOpen,
            ontoggle: (event) => {
              pinnedOpen = event.currentTarget.open;
            },
          },
          h('summary', { class: 'queue__pinned-summary', dataset: { focus: 'pinned' } }, fmt.plural(files.length, 'pinned file', 'pinned files')),
          h(
            'ul',
            { class: 'queue__pinned-list' },
            files.map((file) =>
              h(
                'li',
                { class: 'queue__pinned-item' },
                file?.option ? h('span', { class: 'mono' }, String(file.option)) : null,
                h('span', { class: 'mono queue__wrap', title: file?.source ? String(file.source) : null }, String(file?.path ?? file?.source ?? '')),
              ),
            ),
          ),
        )
      : null;
    return [
      ['Curriculum', h('a', { href: curriculaHref(entry.curriculumId), dataset: { focus: 'curriculum' } }, name ? name : 'Open the curriculum')],
      ['Mode', entry.mode ? MODE_WORDS[entry.mode] ?? String(entry.mode) : null],
      ['Iterations', span],
      ['Weights', weights],
      ['Revision', revisionWords(entry) ? String(entry.revision) : null],
      ['Recipe name', entry.name ? h('span', { class: 'mono' }, entry.name) : null],
      ['Source run', handoff?.runId ? runLink(handoff.runId) : !handoff && entry.sourceRunId ? runLink(entry.sourceRunId) : null],
      ['Source entry', handoff?.entryId ? h('a', { href: queueHref(handoff.entryId) }, entryName(pool().find((item) => item.id === handoff.entryId)) || 'the preceding phase') : null],
      ['Source iteration', Number.isFinite(handoff?.step) ? fmt.int(handoff.step) : null],
      ['Seed', handoff && Number.isFinite(handoff.seed) ? h('span', { class: 'mono' }, String(handoff.seed)) : null],
      ['Pinned files', pinned],
    ];
  };

  const renderFacts = (entry) => {
    const st = entryStatus(entry.status);
    const open = queue.open();
    const place = placeWords(entry, open);
    const paused = queue.get().paused;
    const all = pool();
    const pending = predecessorPending(entry, all);
    const actions = Object.keys(ENTRY_ACTIONS).filter((action) => offered(action, entry, all) && !(action === 'release' && pending));
    const previous = waitsFor(entry, all);
    const canContinue = continuable(entry);
    const canExtend = extendable(entry);
    const follows = canContinue || canExtend;
    const continuation = follows ? queue.continuationOf(entry.runId) : null;
    let sentence = STATE_TEXT[entry.status] ?? '';
    if (entry.status === 'pending' && paused) sentence += ' The queue is paused, so it will not start until you continue the queue.';
    if (entry.status === 'failed' && entry.error) sentence += ' The service reported why below.';
    const rows = [
      ...phaseRows(entry),
      ['Study', entry.studyId ? h('a', { href: studiesHref(entry.studyId) }, entry.studyName || entry.studyId) : null],
      ['Arm', entry.studyId && (entry.armName || entry.armId) ? h('span', { title: entry.armId || null }, entry.armName || entry.armId) : null],
      ['Seed', entry.studyId && Number.isFinite(entry.seed) ? h('span', { class: 'mono' }, String(entry.seed)) : null],
      ['Created', timeEl(entry.createdAt)],
      ['Updated', timeEl(entry.updatedAt)],
      ['Place in line', place || null],
      ['Recipe file', entry.recipePath ? h('span', { class: 'mono' }, entry.recipePath) : null],
      ['Recipe fingerprint', entry.recipeDigest ? h('span', { class: 'mono', title: entry.recipeDigest }, shortDigest(entry.recipeDigest)) : null],
      ['Run', entry.runId ? runLink(entry.runId, entry.runStatus) : null],
      ['Repeats run', entry.sourceRunId && !isPhase(entry) && !isContinuation(entry) ? runLink(entry.sourceRunId) : null],
      ['Continues run', entry.sourceRunId && !isPhase(entry) && isContinuation(entry) ? runLink(entry.sourceRunId) : null],
      ['From iteration', !isPhase(entry) && isContinuation(entry) && Number.isFinite(Number(entry.resumeStep)) ? fmt.int(Number(entry.resumeStep)) : null],
      ['Additional iterations', !isPhase(entry) && isContinuation(entry) && moreIterations(entry) !== null ? fmt.int(moreIterations(entry)) : null],
      ['Archived recipe', entry.sourceRecipe ? h('span', { class: 'mono' }, String(entry.sourceRecipe)) : null],
      ['Archive folder', entry.sourceRoot ? h('span', { class: 'mono' }, String(entry.sourceRoot)) : null],
    ].filter(([, value]) => value !== null && value !== '');

    const active = document.activeElement;
    const focusKey = facts.contains(active) ? active.dataset?.focus ?? 'title' : null;
    mount(
      facts,
      h(
        'header',
        { class: 'queue__entry-head' },
        h(
          'h2',
          { class: 'queue__entry-title', id: titleId, tabindex: '-1', dataset: { focus: 'title' } },
          entryName(entry),
          studyWords(entry) ? h('span', { class: 'queue__cell' }, ` ${studyWords(entry)}`) : null,
          revisionWords(entry) ? h('span', { class: 'queue__cell' }, ` ${revisionWords(entry)}`) : null,
        ),
        replayTag(entry),
        studyTag(entry),
        curriculumTag(entry),
        badge(st.label, st.kind),
      ),
      sentence ? h('p', { class: 'queue__state-text' }, sentence) : null,
      entry.replacedBy
        ? notice(
            '',
            h(
              'span',
              null,
              'Replaced by ',
              h('a', { href: queueHref(entry.replacedBy), dataset: { focus: 'replaced' } }, replacementWords(entry, all)),
              '. Use that entry instead.',
            ),
          )
        : null,
      previous
        ? notice(
            '',
            h(
              'span',
              null,
              `${waitsWords(previous, entry)}. `,
              h('a', { href: queueHref(previous.id), dataset: { focus: 'previous' } }, entryName(previous)),
              ` is ${entryStatus(previous.status).label.toLowerCase()}. Release becomes possible once it has completed.`,
            ),
          )
        : null,
      entry.blockedReason ? notice('warn', h('span', null, h('strong', null, 'Waiting: '), entry.blockedReason)) : null,
      entry.status === 'failed' && entry.error ? notice('err', entry.error) : null,
      Array.isArray(entry.issues) && entry.issues.length ? issueList(entry.issues) : null,
      Array.isArray(entry.replayIssues) && entry.replayIssues.length
        ? [
            notice('warn', 'This replay of an archived run has inputs that must be resolved to explicit local paths. It cannot be released; edit the recipe and add it to the queue as a new entry.'),
            issueList(entry.replayIssues),
          ]
        : null,
      actions.length
        ? h('div', { class: 'queue__actions' }, actions.map((action) => actionButton(action, entry, { small: false, quiet: false })))
        : null,
      continuation ? notice('', continuationLine(continuation)) : null,
      canContinue && !continuation
        ? [
            h('div', { class: 'queue__actions' }, continueButton(entry, { small: false })),
            h('p', { class: 'queue__state-text' }, 'This run resumes from its newest saved checkpoint when the trainer is free.'),
          ]
        : null,
      canExtend && !continuation
        ? [
            h('div', { class: 'queue__actions' }, extendButton(entry, { small: false })),
            h('p', { class: 'queue__state-text' }, 'This run resumes from its final checkpoint for the number of iterations you enter.'),
          ]
        : null,
      follows ? continuedBeforeLine(entry.runId) : null,
      actionErrorEl(entry),
      rows.length ? h('dl', { class: 'kv queue__kv' }, rows.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])) : null,
    );
    if (focusKey && !facts.contains(document.activeElement)) {
      (facts.querySelector(`[data-focus="${focusKey}"]`) || facts.querySelector(`[data-focus="${FOCUS_NEXT[focusKey]}"]`) || facts.querySelector('[data-focus="title"]'))?.focus();
    }
  };

  const renderMissing = () => {
    mount(
      facts,
      h('header', { class: 'queue__entry-head' }, h('h2', { class: 'queue__entry-title', id: titleId }, 'Queue entry')),
      notice('err', missing, h('a', { class: 'btn btn--sm', href: queueHref() }, 'All queue entries')),
    );
    groupController?.abort();
    lineagePart.hidden = true;
    recipePart.hidden = true;
  };

  const loadGroup = async (gid) => {
    groupController?.abort();
    groupController = new AbortController();
    const local = groupController;
    try {
      const group = await api.queueGroup(gid, { signal: local.signal });
      if (local.signal.aborted) return;
      renderGroup(group);
    } catch (error) {
      if (local.signal.aborted || error.name === 'AbortError') return;
      mount(lineage, h('p', { class: 'muted' }, error.message));
      // The phase before could not be read: the actions are offered and the service decides.
      if (!groupKnown) {
        groupKnown = true;
        update();
      }
    }
  };

  const renderGroup = (group) => {
    let entries = Array.isArray(group?.entries) ? group.entries : [];
    const runIds = Array.isArray(group?.runIds) ? group.runIds : [];
    const others = entries.filter((item) => item.id !== id);
    // A study's entries share groupId = studyId: the group is the study's other cells. A curriculum's
    // share groupId = curriculumId: every phase and revision of it.
    const study = Boolean(current()?.studyId);
    const curriculum = isPhase(current());
    if (curriculum) {
      groupEntries = entries;
      groupKnown = true;
      entries = [...entries].sort(
        (a, b) => (phaseNumber(a.phaseId) ?? Infinity) - (phaseNumber(b.phaseId) ?? Infinity) || (a.revision ?? 0) - (b.revision ?? 0),
      );
      update(); // the predecessor's state may now be known (it is part of the facts' signature)
    }
    const titleText = curriculum ? 'Other phases and revisions of this curriculum' : study ? 'Other entries of this study' : 'Replays and runs of this entry';
    if (lineageTitle.textContent !== titleText) lineageTitle.textContent = titleText;
    if (others.length === 0 && runIds.length === 0) {
      mount(
        lineage,
        h(
          'p',
          { class: 'muted' },
          curriculum ? 'This curriculum has no other entries in the queue.' : study ? 'This study has no other entries in the queue.' : 'This entry has no replays and has not started a run yet.',
        ),
      );
      return;
    }
    mount(
      lineage,
      entries.length
        ? h(
            'div',
            { class: 'queue__lineage-part' },
            h('h4', { class: 'queue__lineage-title' }, 'Queue entries'),
            h(
              'ul',
              { class: 'queue__lineage-list' },
              entries.map((item) =>
                h(
                  'li',
                  { class: 'queue__lineage-item' },
                  item.id === id
                    ? h('span', { class: 'queue__lineage-self' }, `${entryName(item)} (this entry)`)
                    : h('a', { href: queueHref(item.id) }, entryName(item)),
                  studyWords(item) ? h('span', { class: 'queue__cell' }, studyWords(item)) : null,
                  revisionWords(item) ? h('span', { class: 'queue__cell' }, revisionWords(item)) : null,
                  statusBadge(item.status),
                  isPhase(item) && item.replacedBy ? h('span', { class: 'muted' }, 'replaced') : null,
                  item.sourceRunId && !isPhase(item) && isContinuation(item)
                    ? h(
                        'span',
                        { class: 'muted' },
                        'continues ',
                        h('a', { href: runHref(item.sourceRunId) }, runs.byId?.(item.sourceRunId)?.name || item.sourceRunId),
                        Number.isFinite(Number(item.resumeStep)) ? ` from iteration ${fmt.int(Number(item.resumeStep))}` : '',
                        moreWords(item),
                      )
                    : item.sourceRunId && !isPhase(item)
                      ? h('span', { class: 'muted' }, 'replay of ', h('a', { href: runHref(item.sourceRunId) }, runs.byId?.(item.sourceRunId)?.name || item.sourceRunId))
                      : null,
                  item.createdAt ? h('span', { class: 'queue__time' }, timeEl(item.createdAt)) : null,
                ),
              ),
            ),
          )
        : null,
      runIds.length
        ? h(
            'div',
            { class: 'queue__lineage-part' },
            h('h4', { class: 'queue__lineage-title' }, 'Runs'),
            h('ul', { class: 'queue__lineage-list' }, runIds.map((runId) => h('li', { class: 'queue__lineage-item' }, runLink(runId)))),
          )
        : null,
    );
  };

  const update = () => {
    if (missing) return;
    const entry = current();
    if (!entry) return;
    if (isPhase(entry)) wantCurriculumNames([entry]);
    const sig = JSON.stringify([
      entry.name, entry.status, entry.updatedAt, entry.createdAt, entry.blockedReason, entry.runId, entry.runStatus, entry.error,
      entry.issues, entry.replayIssues, entry.recipePath, entry.recipeDigest, entry.sourceRunId, studyKey(entry), placeWords(entry, queue.open()), queue.get().paused,
      phaseKey(entry, pool()), entry.mode, entry.startStep, entry.targetStep, entry.weights, entry.handoff ?? null, actionErrorKey(entry),
      isPhase(entry) ? (() => { const previous = waitsFor(entry, pool()); return previous ? [previous.id, previous.status] : ''; })() : '',
      predecessorPending(entry, pool()),
      entry.resumeStep ?? null, entry.additionalIterations ?? null, continuable(entry), extendable(entry), continuationKey(entry),
    ]);
    if (sig !== factsSig) {
      factsSig = sig;
      renderFacts(entry);
    }
    // Lineage: refetch when the store's view of this group changes (a new replay, a run started).
    if (entry.groupId) {
      const members = queue.get().entries.filter((item) => item.groupId === entry.groupId);
      const gsig = JSON.stringify([entry.groupId, members.map((item) => [item.id, item.status, item.runId ?? '', item.sourceRunId ?? '', item.resumeStep ?? '', item.additionalIterations ?? '', studyKey(item), item.replacedBy ?? '', item.revision ?? ''])]);
      if (gsig !== groupSig) {
        groupSig = gsig;
        groupId = entry.groupId;
        loadGroup(groupId);
      }
    } else if (groupSig !== 'none') {
      groupSig = 'none';
      mount(lineage, h('p', { class: 'muted' }, 'The service did not report a recipe line for this entry.'));
    }
  };

  const loadEntry = async () => {
    try {
      const doc = await api.queueEntry(id, { signal });
      if (signal.aborted) return;
      const { yaml: text, ...rest } = doc ?? {};
      fetched = rest;
      yaml = typeof text === 'string' ? text : null;
      if (yaml) {
        mount(recipeBody, h('div', { class: 'queue__yaml', tabindex: '0', role: 'region', 'aria-label': 'Saved recipe text' }, highlightedPre(yaml)));
        openRecipe.disabled = false;
        copy.disabled = false;
      } else {
        mount(recipeBody, h('p', { class: 'muted' }, 'The service returned no recipe text for this entry.'));
      }
      update();
    } catch (error) {
      if (signal.aborted || error.name === 'AbortError') return;
      if (error.status === 404) {
        // "Queue entry was not found." or "The saved queue recipe is unavailable.": the fresh queue says which.
        await queue.refresh();
        if (signal.aborted) return;
        if (!queue.byId(id)) {
          missing = error.message;
          renderMissing();
          return;
        }
      }
      // 400 "The saved queue recipe changed. Submit the desired recipe as a new entry.", the recipe file
      // being unavailable, or the service not answering: the backend's sentence, then a fresh queue.
      mount(recipeBody, notice('err', error.message));
      queue.refresh();
    }
  };

  const initial = queue.byId(id);
  if (initial) update();
  else mount(facts, h('header', { class: 'queue__entry-head' }, h('h2', { class: 'queue__entry-title', id: titleId }, 'Queue entry')), loading('Reading the queue entry…'));
  loadEntry();

  return {
    update,
    destroy() {
      controller.abort();
      groupController?.abort();
      root.remove();
    },
  };
}

// ───────── replay dialog ─────────

/** The recorded launch command ({argv, cwd, …} from invocation.json) as text, its folder and seed. */
function invocationParts(value) {
  const argv = Array.isArray(value?.argv) ? value.argv.map(String) : Array.isArray(value) ? value.map(String) : null;
  const command = argv ? argv.join(' ') : typeof value === 'string' ? value : '';
  const seedAt = argv ? argv.indexOf('--manual_seed') : -1;
  return {
    command,
    cwd: typeof value?.cwd === 'string' ? value.cwd : '',
    seed: seedAt >= 0 && argv[seedAt + 1] !== undefined ? argv[seedAt + 1] : '',
  };
}

/**
 * What a replay of this run would repeat, then "Add replay to queue" or "Open as recipe".
 * @param {{id: string, name?: string}} run
 * @returns {Promise<void>} resolves when the dialog closes
 */
export async function openReplayDialog(run) {
  const controller = new AbortController();
  const requestId = newRequestId(); // one id per dialog: a retried add returns the same entry
  const runName = run?.name || run?.id || 'this run';
  let info = null;
  let busy = false;

  const status = h('div', { class: 'queue__replay-status', role: 'status', 'aria-live': 'polite' }, loading('Reading the recipe this run recorded…'));
  const content = h('div', { class: 'queue__replay-content' });
  const body = h(
    'div',
    { class: 'stack queue__replay' },
    h(
      'p',
      null,
      'A replay is a new queue entry with the same recipe text this run was trained with and the seed the run recorded, when it recorded one. It is saved on hold, so nothing starts until you release it from the queue.',
    ),
    status,
    content,
  );

  let addButton;
  let openButton;
  let closeDialog;
  const { closed } = openDialog({
    title: `Replay ${runName}`,
    body,
    wide: true,
    foot: (close) => {
      closeDialog = close;
      addButton = h('button', { class: 'btn btn--primary queue__btn', type: 'button', disabled: true, onclick: () => add() }, 'Add replay to queue');
      openButton = h(
        'button',
        {
          class: 'btn',
          type: 'button',
          disabled: true,
          onclick: () => {
            if (!info?.yaml) return;
            const yaml = info.yaml;
            close('open');
            openAsRecipe({ yaml, name: run?.name ?? '', label: `replay of run ${runName}` });
          },
        },
        'Open as recipe',
      );
      return [h('button', { class: 'btn', type: 'button', autofocus: true, onclick: () => close(null) }, 'Cancel'), openButton, addButton];
    },
  });

  const add = async () => {
    if (busy || !info) return;
    busy = true;
    // aria-disabled, not disabled: a disabled button would drop focus out of the dialog's controls.
    addButton.setAttribute('aria-disabled', 'true');
    mount(status, loading('Adding the replay to the queue…'));
    try {
      const entry = await queue.replay(run.id, { requestId });
      closeDialog('added');
      toast(`Added a replay of ${runName} to the queue on hold.`, {
        kind: 'ok',
        action: { label: 'Open queue', run: () => go(queueHref(entry?.id)) },
      });
    } catch (error) {
      mount(status, notice('err', `The replay was not added to the queue: ${error.message}`));
    } finally {
      busy = false;
      addButton.removeAttribute('aria-disabled');
    }
  };

  (async () => {
    try {
      info = await api.replayInfo(run.id, { signal: controller.signal });
      if (controller.signal.aborted) return;
      const yaml = typeof info?.yaml === 'string' ? info.yaml : '';
      const invocation = invocationParts(info?.sourceInvocation);
      const rows = [
        ['Repeats run', info?.sourceRunId ? h('span', { class: 'mono' }, info.sourceRunId) : null],
        ['Recipe fingerprint', info?.recipeDigest ? h('span', { class: 'mono', title: info.recipeDigest }, shortDigest(info.recipeDigest)) : null],
        ['Recorded seed', invocation.seed ? h('span', { class: 'mono' }, invocation.seed) : null],
        ['Recorded command', invocation.command ? h('span', { class: 'mono queue__wrap' }, invocation.command) : null],
        ['Working folder', invocation.cwd ? h('span', { class: 'mono' }, invocation.cwd) : null],
        ['Archived recipe', info?.sourceRecipe ? h('span', { class: 'mono' }, String(info.sourceRecipe)) : null],
        ['Archive folder', info?.sourceRoot ? h('span', { class: 'mono' }, String(info.sourceRoot)) : null],
      ].filter(([, value]) => value);
      const replayIssues = Array.isArray(info?.replayIssues) ? info.replayIssues : [];
      mount(status, null);
      mount(
        content,
        rows.length ? h('dl', { class: 'kv queue__kv' }, rows.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])) : null,
        replayIssues.length
          ? [
              notice(
                'warn',
                'This archived recipe has inputs that are not explicit local paths. A replay added from it stays on hold and cannot be released. Open it as a recipe, change those inputs to local paths, and add the edited recipe to the queue as a new entry.',
              ),
              issueList(replayIssues),
            ]
          : null,
        yaml
          ? h('div', { class: 'queue__yaml queue__yaml--dialog', tabindex: '0', role: 'region', 'aria-label': 'Recipe text of the replay' }, highlightedPre(yaml))
          : h('p', { class: 'muted' }, 'The service returned no recipe text for this run.'),
      );
      if (yaml) {
        addButton.disabled = false;
        openButton.disabled = false;
      }
    } catch (error) {
      if (controller.signal.aborted || error.name === 'AbortError') return;
      mount(status, notice('err', error.message));
    }
  })();

  await closed;
  controller.abort();
}

// ───────── extend training dialog ─────────

/** Digit-group separators a person may type ("50,000", "50 000"). */
const GROUPING = /[\s,]/g;

/** "50,000", "50 000" or "50000" → 50000; null unless a whole number from 1 to Number.MAX_SAFE_INTEGER. */
function readIterations(text) {
  const digits = String(text ?? '').replace(GROUPING, '');
  if (!/^\d+$/.test(digits)) return null;
  const n = Number(digits);
  return Number.isSafeInteger(n) && n >= 1 ? n : null;
}

/** The field's message for text that readIterations refuses. */
function iterationsProblem(text) {
  const digits = String(text ?? '').replace(GROUPING, '');
  if (!digits) return 'Enter how many more iterations to train, 1 or more.';
  if (/^\d+$/.test(digits) && Number(digits) >= 1) return 'That number is too large.';
  return 'Enter a whole number, 1 or more, such as 50000.';
}

/** A number the service reported (a number or numeric text), or null. */
function reported(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) return Number(value);
  return null;
}

/** What still works for a completed run that cannot be extended. Nothing offered here starts a run by itself. */
function extendAlternatives() {
  return h(
    'div',
    { class: 'queue__options' },
    h('p', null, 'Nothing was started. What still works for this run:'),
    h(
      'ul',
      { class: 'queue__options-list' },
      h('li', null, 'Fine-tune from its saved weights: on the run\'s Files tab choose a checkpoint, then "Fine-tune from this…". That starts a new run from the weights; the optimizer state and the iteration count start again.'),
      h('li', null, 'Replay: repeats the run from the start. In the Replay dialog, "Open as recipe" lets you set a larger total first.'),
    ),
  );
}

/**
 * "Extend training" of completed run `runId`: asks how many more iterations, then adds a released
 * continuation that trains from the run's final training state to the new total. The dialog is the
 * confirmation on both editions. A refusal stays in the dialog with what was typed.
 * @param {{runId: string, name?: string, run?: object|null}} options  run: its loaded detail, when at hand
 * @returns {Promise<void>} resolves when the dialog closes
 */
export async function openExtendDialog({ runId, name, run = null }) {
  const label = name || run?.name || runId || 'this run';
  const phone = document.documentElement.dataset.edition === 'mobile';
  const controller = new AbortController();
  const opener = document.activeElement;
  let open = true;
  let sending = false;
  let input = null; // the field, once it is offered
  let flag = () => {};
  let sync = () => {};
  let closeDialog;
  let cancelButton;
  let addButton;

  const status = h('div', { class: 'queue__replay-status', role: 'status', 'aria-live': 'polite' }, loading('Reading this run and the queue…'));
  const content = h('div', { class: 'queue__replay-content' });
  const body = h('div', { class: 'stack queue__replay' }, status, content);

  const { closed, dialog } = openDialog({
    title: `Extend training of ${label}`,
    body,
    foot: (close) => {
      closeDialog = close;
      cancelButton = h('button', { class: 'btn', type: 'button', autofocus: true, onclick: () => close(null) }, 'Cancel');
      // aria-disabled, not disabled: a disabled button would drop focus out of the dialog's controls.
      addButton = h('button', { class: 'btn btn--primary', type: 'button', 'aria-disabled': 'true', onclick: () => submit() }, 'Add to queue');
      return [cancelButton, addButton];
    },
  });
  closed.then(() => {
    open = false;
    controller.abort();
  });
  // One short field and a few lines: a column that reads, not the full width a recipe text gets.
  // (On the phone it is a sheet across the whole width, whatever is set here.)
  dialog.style.width = 'min(560px, 94vw)';

  /** Nothing can be added: only Close is left. */
  const onlyClose = () => {
    addButton.hidden = true;
    cancelButton.textContent = 'Close';
  };

  async function submit() {
    if (sending || !input || !open) return;
    const n = readIterations(input.value);
    if (n === null) {
      flag(iterationsProblem(input.value));
      input.focus();
      return;
    }
    sending = true;
    addButton.setAttribute('aria-disabled', 'true');
    mount(status, loading('Adding the continuation to the queue…'));
    let result;
    try {
      result = await addContinuation({ runId, name: label, additionalIterations: n, confirm: false, toastRefusal: false });
    } catch (error) {
      result = { ok: false, message: error.message, offline: Boolean(error.offline) };
    }
    sending = false;
    // Added, or one was already there: the toast from addContinuation says which.
    if (result?.ok) {
      if (open) closeDialog('added');
      return;
    }
    const message = result?.message || 'The service did not answer.';
    if (!open) {
      toast(`${label} was not added to the queue: ${message}`, { kind: 'err' });
      return;
    }
    // A continuation of this run is in the queue after all (queued elsewhere meanwhile, or with another
    // amount): that entry is what there is to act on, not the other ways to train.
    if (!result?.offline) {
      try {
        await queue.refresh();
      } catch {
        // the store keeps what it had
      }
      if (!open) return;
      const existing = queue.continuationOf(runId);
      if (existing) {
        showExisting(existing, `Nothing was added to the queue: ${message}`);
        return;
      }
    }
    sync();
    mount(status, notice('err', `Nothing was added to the queue: ${message}`), result?.offline ? null : extendAlternatives());
    status.scrollIntoView({ block: 'nearest' }); // a short sheet scrolls: the answer is not left below its edge
  }

  /** One open continuation per run: its entry is offered instead of a field. `refusal`: what the service just said. */
  function showExisting(existing, refusal = null) {
    const more = moreIterations(existing);
    input = null;
    mount(status, refusal ? notice('err', refusal) : null);
    mount(
      content,
      notice(
        '',
        h(
          'span',
          null,
          `A continuation of this run is already in the queue (${statusPlaceWords(existing)}${more !== null ? `, ${moreCount(more)}` : ''}). `,
          refusal ? 'To queue another amount, cancel that entry first. ' : null,
          h(
            'a',
            {
              href: queueHref(existing.id),
              onclick: (event) => {
                if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
                event.preventDefault();
                closeDialog(null);
                go(queueHref(existing.id));
              },
            },
            'Open its queue entry',
          ),
        ),
      ),
    );
    onlyClose();
  }

  (async () => {
    let detail = run;
    let readError = null;
    const reads = [Promise.resolve().then(() => queue.refresh()).catch(() => null)];
    if (!detail) {
      reads.push(
        api.run(runId, { signal: controller.signal }).then(
          (value) => {
            detail = value;
          },
          (error) => {
            readError = error;
          },
        ),
      );
    }
    await Promise.all(reads);
    if (!open || controller.signal.aborted) return;
    if (readError) {
      mount(status, notice('err', `This run could not be read: ${readError.message}`));
      onlyClose();
      return;
    }

    // One open continuation per run: offer its entry instead of a field.
    const existing = queue.continuationOf(runId);
    if (existing) {
      showExisting(existing);
      return;
    }

    const availability = extendAvailability(detail);
    if (availability?.available === false) {
      const reason = typeof availability.reason === 'string' && availability.reason.trim() ? availability.reason : "This run's final training state is not available.";
      mount(status, null);
      mount(content, notice('err', reason), extendAlternatives());
      onlyClose();
      return;
    }

    // The service continues from the iteration the run completed at and counts the new total from there:
    // its own word (availability), else the completion it recorded, else the recipe's total (older runs).
    const total = reported(detail?.total);
    const step = reported(availability?.resumeStep) ?? reported(detail?.completedIteration) ?? total;
    const trained = step === null ? 'Not reported' : `${fmt.plural(step, 'iteration')}${total !== null && total !== step ? ` (the recipe's total was ${fmt.int(total)})` : ''}`;

    const inputId = uid('extend');
    const error = h('p', { class: 'field__error', id: `${inputId}-error`, role: 'alert' });
    flag = (problem) => {
      error.textContent = problem ?? '';
      if (problem) input.setAttribute('aria-invalid', 'true');
      else input.removeAttribute('aria-invalid');
    };
    input = h('input', {
      class: 'input',
      id: inputId,
      type: 'text',
      inputmode: 'numeric',
      autocomplete: 'off',
      spellcheck: 'false',
      value: '',
      'aria-describedby': `${inputId}-help ${inputId}-error`,
      oninput: () => sync(),
      onchange: () => {
        if (input.value.trim() && readIterations(input.value) === null) flag(iterationsProblem(input.value));
      },
      onkeydown: (event) => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        submit();
      },
    });
    const newTotal = h('span', { 'aria-live': 'polite' }, '–');
    sync = () => {
      const n = readIterations(input.value);
      // A corrected field loses its message at once; a mistake is named when the field is left, or on Add.
      if (n !== null) flag(null);
      const text = n === null ? '–' : step !== null ? fmt.plural(step + n, 'iteration') : `the original total plus ${fmt.int(n)}`;
      if (newTotal.textContent !== text) newTotal.textContent = text;
      if (n === null || sending) addButton.setAttribute('aria-disabled', 'true');
      else addButton.removeAttribute('aria-disabled');
    };
    const rows = [
      ['Trained so far', trained],
      ['New total', newTotal],
      ['Continues from', `its final training state${step !== null ? `, iteration ${fmt.int(step)}` : ''}`],
    ];
    mount(status, null);
    mount(
      content,
      h(
        'div',
        { class: 'field runs__resume-field' },
        h('label', { class: 'field__label', for: inputId }, 'Additional iterations'),
        input,
        h('p', { class: 'field__help', id: `${inputId}-help` }, 'A whole number, such as 50000.'),
        error,
      ),
      h('dl', { class: 'kv queue__kv' }, rows.map(([key, value]) => [h('dt', null, key), h('dd', null, value)])),
      h(
        'p',
        null,
        'This run keeps its checkpoints and training history. It resumes from the final checkpoint and trains to the new total when it reaches the front of the queue.' +
          (queue.get().paused ? PAUSED_WAIT : ''),
      ),
    );
    sync();
    // On the phone edition the keyboard would cover the sheet: the field waits to be touched.
    const at = document.activeElement;
    if (!phone && (at === cancelButton || at === dialog || at === document.body || !dialog.contains(at))) input.focus();
  })();

  await closed;
  controller.abort();
  // The browser returns focus to the opener. When the opener gave way meanwhile (a continuation of this run
  // is now in the queue in its place), focus goes to the link to that continuation instead.
  const at = document.activeElement;
  if ((!at || at === document.body) && opener instanceof HTMLElement && (!opener.isConnected || opener.closest('[hidden]'))) {
    document.querySelector(`[data-continues-run="${CSS.escape(String(runId))}"]`)?.focus({ preventScroll: true });
  }
}
