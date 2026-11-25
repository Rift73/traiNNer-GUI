// The trainer's own iteration count of the runs that are on screen (GET /api/runs/<id>/progress). It is read
// by itself, apart from the run list and the run's monitor, so a counter can follow every step without
// anything else being read again. Nothing is estimated: a value is a count the trainer reported, or null.
// One answer can confirm several iterations at once. The counters then step through them one by one, always
// behind the newest confirmed count and never past it; where stepping would not show what happens now, they
// go straight to that count (see `confirm`).
// A run is asked about while the run list says it trains, and once more when it ends (the trainer's final
// count); a run never seen training is not asked at all.

import { api } from '../api.js';
import { runs, runStatus } from './runs.js';
import { conn } from './conn.js';

/** While the count moves (it changed within MOVING_FOR_MS), a run is asked about this long after each answer. */
const MOVING_MS = 500;
const MOVING_FOR_MS = 2000;
/** A count that changes more rarely is asked about once a second. */
const BASE_MS = 1000;
/** The longest wait between two answers, however slow the link or idle the trainer. */
const MAX_MS = 4000;
/** A trainer that publishes no count may still start to (its first iteration): it is asked again this much later. */
const NULL_MS = 5000;
/** A question that has had no answer by then is given up and counts as a failure. */
const TIMEOUT_MS = 10000;
/** Waits after the first, second, third and every later failure in a row; the final count gives up after the last. */
const RETRY_MS = [2000, 4000, 8000, 15000];
/**
 * The counts one answer confirmed are stepped through in the time the answer took to come after the one
 * before it (so they end about when the next is due), within these bounds.
 */
const PLAY_MIN_MS = 250;
const PLAY_MAX_MS = 1000;
/** A step is one count and takes one painted frame at least: more counts than fit are not stepped through. */
const FRAME_MS = 1000 / 60;
/**
 * After a longer silence between two counts nothing is stepped through. Longer than any wait that is planned
 * (MAX_MS, NULL_MS, a slow answer on top), so only a real stall is taken for one.
 */
const STALL_MS = 8000;
const SPEED_WINDOW_MS = 5000;

// String(id) -> {id, listeners, value, shown, attemptId, changedAt, confirmedAt, stepMs, nextStepAt, resync,
//   dirty, timer, ctrl, frame, failures, rtt, off, active, owed, stale}.
// `value`: the newest count the trainer confirmed. `shown`: the count the counters show, which steps up to
//   `value` (`stepMs` apart, the next at `nextStepAt`) and is never past it.
// `resync`: the next count is shown at once (the page was hidden, the link was lost, the run started).
// `active`: the run list says the run trains. `owed`: it ended while followed and its final count is still to be read.
// `stale`: the last question brought no count (none published, or no answer), so `value` is an earlier reading.
const entries = new Map();
// The service has no such route (an older service): nothing is asked until it is back online.
let unsupported = false;
let connState = conn.get().state;

/** Stop asking about one run: the pending question is dropped and the one under way is abandoned. */
function halt(entry) {
  clearTimeout(entry.timer);
  entry.timer = 0;
  const ctrl = entry.ctrl;
  entry.ctrl = null;
  ctrl?.abort();
}

const wanted = (entry) => !unsupported && !entry.off && entry.listeners.size > 0 && (entry.active || entry.owed);

function schedule(entry, ms) {
  clearTimeout(entry.timer);
  entry.timer = 0;
  if (!wanted(entry) || document.hidden) return;
  entry.timer = setTimeout(() => ask(entry), ms);
}

// One frame does both: it takes the next step towards the confirmed count when that is due, and tells the
// counters. So a counter is painted at most once per frame, and every step is one count that was painted: a
// late frame skips none (the steps after it come later, and the next answer plans them anew).
function paint(entry) {
  if (entry.frame) return;
  entry.frame = requestAnimationFrame(() => {
    entry.frame = 0;
    let moved = entry.dirty;
    entry.dirty = false;
    if (entry.shown !== entry.value) {
      const stepping = entry.stepMs > 0 && entry.shown !== null && entry.value !== null && entry.shown < entry.value;
      const now = performance.now();
      if (!stepping) {
        entry.shown = entry.value;
        moved = true;
      } else if (now >= entry.nextStepAt) {
        entry.shown += 1;
        entry.nextStepAt = Math.max(entry.nextStepAt + entry.stepMs, now);
        moved = true;
      }
    }
    if (moved) {
      for (const token of [...entry.listeners]) {
        try {
          token.fn(entry.shown);
        } catch (error) {
          console.error(error);
        }
      }
    }
    if (entry.listeners.size && entry.shown !== entry.value) paint(entry);
  });
}

/** The counters go straight to the confirmed count: nothing is stepped through. */
function settle(entry) {
  entry.stepMs = 0;
  if (entry.shown === entry.value) return;
  entry.shown = entry.value;
  entry.dirty = true;
  paint(entry);
}

// Sample the confirmed optimizer counter, independently of animation and trainer logging.
// Use the elapsed time between actual readings so network/poll jitter does not change the denominator.
function resetSpeed(entry) {
  clearInterval(entry.speedTimer);
  entry.speedTimer = 0;
  if (entry.speed == null) return;
  entry.speed = null;
  entry.dirty = true;
  paint(entry);
}

function startSpeedWindow(entry) {
  let count = entry.value;
  let time = entry.confirmedAt;
  entry.speedTimer = setInterval(() => {
    const elapsed = entry.confirmedAt - time;
    const fresh = elapsed > 0 && elapsed <= SPEED_WINDOW_MS + STALL_MS &&
      performance.now() - entry.confirmedAt < SPEED_WINDOW_MS;
    entry.speed = fresh ? 1000 * (entry.value - count) / elapsed : null;
    count = entry.value;
    time = entry.confirmedAt;
    entry.dirty = true;
    paint(entry);
  }, SPEED_WINDOW_MS);
}

/** No reading: the counters go back to the run's summary. */
function clear(entry, attemptId = null) {
  resetSpeed(entry);
  entry.value = null;
  entry.attemptId = attemptId;
  settle(entry);
}

/**
 * A count the trainer confirmed. As the steady progress of one attempt it is stepped up to: every count
 * since the one shown gets its turn, the first at once and the last about when the next answer is due.
 * The counters go straight to it instead where stepping would show the past as if it were happening now:
 * the first count, another attempt, a count that went back, a run that has ended, the first count after the
 * page was hidden or the link was lost, a long silence, or more counts than there are frames to paint them.
 */
function confirm(entry, reading, attemptId, now) {
  const gap = now - entry.confirmedAt;
  const from = entry.shown;
  const direct = entry.resync || attemptId !== entry.attemptId || !entry.active || from === null || reading < from || gap > STALL_MS;
  // Counts that were still being stepped to keep their turn; otherwise the first new one shows at once.
  const pending = !direct && entry.stepMs > 0 && entry.value !== null && from < entry.value;
  if (entry.resync || attemptId !== entry.attemptId || reading < entry.value || gap > STALL_MS) resetSpeed(entry);
  entry.resync = false;
  entry.attemptId = attemptId;
  entry.confirmedAt = now;
  if (reading !== entry.value) {
    entry.value = reading;
    entry.changedAt = now;
  }
  if (entry.active && !entry.speedTimer) startSpeedWindow(entry);
  const backlog = direct ? 0 : reading - from;
  const within = Math.min(PLAY_MAX_MS, Math.max(PLAY_MIN_MS, gap));
  if (backlog < 1 || backlog * FRAME_MS > within) {
    settle(entry);
    return;
  }
  entry.stepMs = within / backlog;
  entry.nextStepAt = pending ? Math.min(entry.nextStepAt, now + entry.stepMs) : now;
  paint(entry);
}

// The next question is asked only after the previous answer, so a run never has two under way.
async function ask(entry) {
  clearTimeout(entry.timer);
  entry.timer = 0;
  if (!wanted(entry) || document.hidden || entry.ctrl) return;
  const ctrl = new AbortController();
  entry.ctrl = ctrl;
  let timedOut = false;
  const limit = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, TIMEOUT_MS);
  const started = performance.now();
  let res = null;
  let error = null;
  try {
    res = await api.runProgress(entry.id, { signal: ctrl.signal });
  } catch (caught) {
    error = caught;
  } finally {
    clearTimeout(limit);
  }
  // Released, hidden, or the run started or ended meanwhile (which asks anew): this question ends without a word.
  if (entry.ctrl !== ctrl) return;
  entry.ctrl = null;

  if (error) {
    if (!timedOut && error.status === 404 && /unknown api endpoint/i.test(error.message ?? '')) {
      stopAll();
      return;
    }
    if (!timedOut && error.status >= 400 && error.status < 500) {
      // The service does not know this run (or it is imported): its counter goes back to the run's summary.
      entry.off = true;
      entry.owed = false;
      clear(entry);
      return;
    }
    // Offline, timed out or a service fault: the last value stays, since it is still the last real reading.
    // What is confirmed after the link is back is shown at once, not stepped through.
    resetSpeed(entry);
    entry.failures += 1;
    entry.stale = true;
    entry.resync = true;
    if (!entry.active && entry.failures > RETRY_MS.length) {
      entry.owed = false;
      return;
    }
    schedule(entry, RETRY_MS[Math.min(entry.failures - 1, RETRY_MS.length - 1)]);
    return;
  }

  const now = performance.now();
  const rtt = now - started;
  entry.failures = 0;
  const reading = Number.isInteger(res?.iteration) && res.iteration >= 0 ? res.iteration : null;
  const attemptId = typeof res?.attemptId === 'string' ? res.attemptId : null;
  // An answer without a count leaves the last count of the same attempt in place: it is still what that
  // trainer was seen to reach (its signal is taken away when it exits, a moment before the run is said to
  // have ended). Only the answer of another attempt clears it.
  entry.stale = reading === null;
  if (reading !== null) confirm(entry, reading, attemptId, now);
  else if (attemptId !== entry.attemptId) clear(entry, attemptId);
  else resetSpeed(entry);
  if (!entry.active) {
    // The final count of a run that ended: nothing more to ask.
    entry.owed = false;
    return;
  }
  if (reading === null) {
    schedule(entry, NULL_MS);
    return;
  }
  // A count that moves is asked about twice a second, so that few counts come at once and the counters are
  // little behind. A slow link (a phone over Tailscale) is never kept busy more than a third of the time: the
  // quicker of the last two answers measures it, so one late answer does not slow the next question. A
  // trainer that is not stepping (validation, saving, stopping) is asked less and less often; its first new
  // count brings the pace back.
  const link = Math.min(rtt, entry.rtt ?? rtt);
  entry.rtt = rtt;
  const idle = now - entry.changedAt;
  const pace = idle < MOVING_FOR_MS ? MOVING_MS : idle < 15000 ? BASE_MS : idle < 120000 ? 2 * BASE_MS : MAX_MS;
  schedule(entry, Math.min(MAX_MS, Math.max(pace, 2 * link)));
}

/** Follow the run list: ask while the run trains, and once more when it ends while followed. */
function sync(entry, key) {
  const run = runs.get().list.find((item) => String(item.id) === key);
  const active = Boolean(run) && runStatus(run.status).active;
  if (active === entry.active) return;
  entry.active = active;
  if (active) {
    // A run that starts, or starts again, is another attempt: a count read before says nothing about it, so
    // it is dropped here, before anything could show it for the new attempt.
    entry.stale = false;
    entry.resync = true;
    clear(entry);
  } else {
    resetSpeed(entry);
    // Ended: no step is taken after it; the next answer is the trainer's final count (from its exit receipt).
    settle(entry);
  }
  // Started: the loop begins. Ended: a question asked while it still trained is dropped and asked anew.
  entry.owed = !active;
  entry.failures = 0;
  halt(entry);
  ask(entry);
}

/** An older service: every counter falls back to the run's summary and nothing more is asked. */
function stopAll() {
  unsupported = true;
  for (const entry of entries.values()) {
    halt(entry);
    entry.owed = false;
    clear(entry);
  }
}

function get(id) {
  if (id === undefined || id === null) return null;
  return entries.get(String(id))?.value ?? null;
}

runs.subscribe(() => {
  for (const [key, entry] of entries) sync(entry, key);
});

// A service that comes back (it may have been updated and restarted) is asked again without a page reload.
conn.subscribe((s) => {
  const was = connState;
  connState = s.state;
  if (s.state !== 'online' || was === 'online') return;
  unsupported = false;
  for (const entry of entries.values()) {
    entry.off = false;
    entry.resync = true;
    // At once, also where a retry was still waiting its turn.
    if (!entry.ctrl) ask(entry);
  }
});

// Nothing is asked while the page is hidden, and no step is left over for the return: every run that trains,
// or still owes its final count, is then asked at once and its counters go straight to the answer.
document.addEventListener('visibilitychange', () => {
  for (const entry of entries.values()) {
    if (document.hidden) {
      resetSpeed(entry);
      halt(entry);
      entry.resync = true;
      settle(entry);
    } else if (!entry.timer && !entry.ctrl) ask(entry);
  }
});

export const progress = {
  /**
   * Follow the trainer's count of one run from the time it is seen training until the caller goes away; a
   * run that ends is not released (its counter then ends on the trainer's final count). `listener(iteration
   * |null)` is called with each count as it is shown. The run head, its list row and the rail share one loop
   * of questions, and one sequence of steps, per run. Returns release().
   */
  follow(id, listener) {
    if (id === undefined || id === null || typeof listener !== 'function') return () => {};
    const key = String(id);
    let entry = entries.get(key);
    const created = !entry;
    if (created) {
      entry = {
        id, listeners: new Set(), value: null, shown: null, attemptId: null, changedAt: 0, confirmedAt: 0, stepMs: 0, nextStepAt: 0,
        resync: true, dirty: false, timer: 0, ctrl: null, frame: 0, failures: 0, rtt: null, off: false, active: false, owed: false, stale: false,
      };
      entries.set(key, entry);
    }
    // One token per call: the same function following twice needs two releases.
    const token = { fn: listener };
    entry.listeners.add(token);
    if (created) sync(entry, key);
    else if (!entry.timer && !entry.ctrl) ask(entry);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      entry.listeners.delete(token);
      if (entry.listeners.size || entries.get(key) !== entry) return;
      halt(entry);
      resetSpeed(entry);
      if (entry.frame) cancelAnimationFrame(entry.frame);
      entry.frame = 0;
      entries.delete(key);
    };
  },
  /** The newest count the trainer confirmed for a followed run, or null. The counters may still be stepping up to it. */
  get,
  /** Confirmed updates per second from the latest five-second window of this attempt. */
  speedOf(run) {
    const entry = entries.get(String(run?.id));
    if (!entry?.active || entry.stale || (run.attemptId && entry.attemptId !== run.attemptId)) return null;
    return entry.speed ?? null;
  },
  /** The displayed live counter, falling back to the run summary when unavailable. */
  iterationOf(run) {
    const entry = entries.get(String(run?.id));
    const live = entry && Number.isInteger(entry.shown) ? entry.shown : null;
    if (live === null) return run?.iteration;
    // A reading belongs to the attempt that reported it. Beside a summary of another attempt (a summary lags a
    // resume by a moment) it says nothing, so the two are never mixed.
    if (entry.attemptId && run.attemptId && entry.attemptId !== run.attemptId) return run.iteration;
    // While it trains and its count is being read, the trainer's count wins outright: after a resume from an
    // earlier checkpoint it rightly goes back below the last logged iteration.
    if (runStatus(run.status).active && !entry.stale) return live;
    // Ended while followed: the trainer's final count; the summary of a stopped run is its last log line,
    // which can be older. Or the count could not be read lately: the log may have moved past it. Either way
    // the further of the two.
    return Number.isFinite(run.iteration) ? Math.max(run.iteration, live) : live;
  },
};
