// Trainer environment dialog on GET /api/environment: the trainer machine as the service reports it.
// Nothing is estimated; an absent field shows as "Not reported".
// Below it: where the choices of the recipe form come from (GET /api/catalog), and reading them again from
// this installation (POST /api/catalog/refresh) after a network, loss or other part was added to the trainer.
//
//   openEnvironmentDialog() -> Promise<void> when closed

import { h, mount as fill, fmt, uid } from '../dom.js';
import { api } from '../api.js';
import { catalog } from '../state/catalog.js';
import { recipe } from '../state/recipe.js';
import { openDialog, notice, loading, copyText, toast } from '../ui.js';

const GPU_LINE = 'The graphics card is not read here: the service does not start CUDA for this. A run records its device when it starts.';
const NO_TUNING = 'No tuning variables are set';

// ───────── pure helpers (exported for checks) ─────────

/** Whole gibibytes (2^30 bytes), labelled GiB; '' when absent. The exact byte counts go in `title`. */
export function gigabytes(bytes) {
  return Number.isFinite(bytes) ? `${fmt.int(Math.round(bytes / 2 ** 30))} GiB` : '';
}

/** "273 GiB free of 1,007 GiB"; '' when either number is absent. */
export function diskText(free, total) {
  if (!Number.isFinite(free) || !Number.isFinite(total)) return '';
  return `${gigabytes(free)} free of ${gigabytes(total)}`;
}

const FACTS = [
  ['python', 'Python'],
  ['interpreter', 'Interpreter'],
  ['platform', 'Platform'],
  ['kernel', 'Kernel'],
  ['machine', 'Machine'],
  ['logicalCpus', 'Logical CPUs'],
  ['checkout', 'Trainer folder'],
  ['state', 'Service state folder'],
  ['llvm', 'LLVM (clang++)'],
];

const entriesOf = (value) => (value && typeof value === 'object' && !Array.isArray(value) ? Object.entries(value) : []);

function factValue(env, key) {
  const value = env?.[key];
  if (key === 'logicalCpus') return Number.isFinite(value) ? fmt.int(value) : '';
  if (key === 'llvm' && value === null) return 'Not found';
  return typeof value === 'string' ? value : '';
}

/** Plain-text summary for "Copy as text". */
export function environmentText(env) {
  const lines = ['Trainer environment'];
  for (const [key, label] of FACTS) lines.push(`${label}: ${factValue(env, key) || 'Not reported'}`);
  const disk = diskText(env?.diskFreeBytes, env?.diskTotalBytes);
  lines.push(`Disk: ${disk ? `${disk} (${env.diskFreeBytes} of ${env.diskTotalBytes} bytes)` : 'Not reported'}`);
  const packages = entriesOf(env?.packages);
  lines.push('Packages:');
  if (packages.length) for (const [name, version] of packages) lines.push(`  ${name} ${version}`);
  else lines.push('  Not reported');
  const tuning = entriesOf(env?.environment);
  lines.push('Tuning variables:');
  if (tuning.length) for (const [name, value] of tuning) lines.push(`  ${name}=${value}`);
  else lines.push(`  ${NO_TUNING}`);
  lines.push(GPU_LINE);
  return lines.join('\n');
}

/** The lists a catalog answer holds, with the words that count them. */
const LISTS = [
  ['architectures', 'network'],
  ['losses', 'loss', 'losses'],
  ['datasets', 'dataset type'],
  ['optimizers', 'optimizer'],
  ['schedulers', 'scheduler'],
  ['metrics', 'metric'],
];

/** "192 networks, 38 losses, 10 dataset types"; '' when the answer holds no list. */
export function catalogCounts(cat) {
  return LISTS.filter(([key]) => Array.isArray(cat?.[key]))
    .map(([key, one, many]) => fmt.plural(cat[key].length, one, many))
    .join(', ');
}

/** Where the lists come from, as the service says it (`source`, `generatedAt`); '' when it does not say. */
export function catalogSourceText(cat) {
  if (cat?.source === 'packaged') return 'These are the lists that came with the trainer. They have not been read from this installation.';
  if (cat?.source === 'local') {
    const when = typeof cat.generatedAt === 'string' && cat.generatedAt ? fmt.dateTime(cat.generatedAt) : '';
    return when ? `These lists were read from this installation on ${when}.` : 'These lists were read from this installation.';
  }
  return '';
}

const REFRESH_HELP =
  "Read them again after adding a network, a loss or another part to the trainer, so the recipe form offers it and the recipe check knows it. This loads the trainer's code on the CPU, apart from training, and can take up to two minutes.";

/** Everything built on the lists takes a newer answer: the form's choices and the open recipe's check. */
function applyLists(cat) {
  catalog.replace(cat);
  if (recipe.get().phase === 'open') recipe.checkNow();
}

// One reading at a time for the page, whichever dialog started it: a dialog opened while it runs follows
// the same request, and its result is applied even when every dialog has been closed meanwhile.
let pendingRefresh = null;
let dialogsOpen = 0;
function readListsAgain() {
  if (!pendingRefresh) {
    pendingRefresh = api
      .catalogRefresh()
      .then((cat) => {
        applyLists(cat);
        return cat;
      })
      .finally(() => {
        pendingRefresh = null;
      });
    // With no dialog open to say it, the outcome is said in a toast.
    pendingRefresh.then(
      () => {
        if (!dialogsOpen) toast('The lists of networks, losses and options were read again.', { kind: 'ok' });
      },
      (error) => {
        if (!dialogsOpen) toast(`The lists were not read again, and the earlier ones stay in use: ${String(error?.message ?? 'the service gave no reason.').slice(0, 240)}`, { kind: 'err' });
      },
    );
  }
  return pendingRefresh;
}

// ───────── dialog ─────────

/** @returns {Promise<void>} */
export function openEnvironmentDialog() {
  const controller = new AbortController();
  let current = null;
  const content = h('div', { class: 'environment', 'aria-live': 'polite' }, loading('Reading the trainer environment…'));
  const copyButton = h(
    'button',
    {
      class: 'btn',
      type: 'button',
      disabled: true,
      onclick: async () => {
        if (!current) return;
        const ok = await copyText(environmentText(current));
        toast(ok ? 'Copied the trainer environment as text.' : 'The text could not be copied.', { kind: ok ? 'ok' : 'err' });
      },
    },
    'Copy as text',
  );
  const retryButton = h('button', { class: 'btn', type: 'button', hidden: true, onclick: () => load() }, 'Try again');

  const absent = () => h('span', { class: 'muted' }, 'Not reported');

  // ── the choices of the recipe form ──
  // Stable nodes, changed in place: the button keeps its focus while the lists are read.
  const listsId = uid('environment-lists');
  const listsLead = h('p', { class: 'environment__lists-lead' }, 'Reading the lists…');
  const listsIssues = h('div', { class: 'environment__lists-issues' });
  const listsStatus = h('p', { class: 'environment__lists-status muted', role: 'status' });
  const listsError = h('div', { class: 'environment__lists-error' });
  const refreshButton = h('button', { class: 'btn btn--sm', type: 'button', onclick: () => refreshLists() }, 'Read the lists again');
  const listsSection = h(
    'section',
    { class: 'environment__lists', 'aria-labelledby': listsId },
    h('h3', { class: 'environment__heading', id: listsId }, 'Choices the recipe form offers'),
    listsLead,
    listsIssues,
    h('p', { class: 'field__help' }, REFRESH_HELP),
    h('div', { class: 'environment__lists-action' }, refreshButton, listsStatus),
    listsError,
  );
  let closedNow = false;

  function renderLists(cat) {
    const counts = catalogCounts(cat);
    listsLead.textContent = [catalogSourceText(cat), counts ? `They hold ${counts}.` : ''].filter(Boolean).join(' ') || 'The service did not say what its lists hold.';
    const issues = Array.isArray(cat?.signatureIssues) ? cat.signatureIssues.filter((item) => item && typeof item === 'object') : [];
    fill(
      listsIssues,
      issues.length
        ? notice('warn', [
            `${fmt.plural(issues.length, 'entry', 'entries')} could not be read completely, so the form may not show all of ${issues.length === 1 ? 'its' : 'their'} options:`,
            h('ul', { class: 'environment__lists-list' }, issues.map((item) => h('li', null, h('span', { class: 'mono' }, String(item.id ?? 'unnamed')), item.message ? `: ${item.message}` : ''))),
          ])
        : null,
    );
  }

  async function loadLists() {
    try {
      // Asked afresh: the lists may have been read again since this page loaded them (from here, from
      // another window or from the command line).
      api.catalog.reset();
      const cat = await api.catalog();
      if (closedNow) return;
      renderLists(cat);
      // The page was built on an older reading: what this dialog describes is then also what it uses.
      // The options schema belongs to the same reading (it names the fields of what the lists hold), so
      // the cached one is dropped first: the form that rebuilds on the new lists reads the new schema.
      const used = catalog.get();
      if (used.phase === 'ready' && used.stamp !== `${cat?.source ?? ''}|${cat?.generatedAt ?? ''}`) {
        api.schema.reset();
        applyLists(cat);
      }
    } catch (error) {
      if (!closedNow) listsLead.textContent = `The lists could not be read: ${error.message}`;
    }
  }

  /** Show a reading that is under way (started here or by an earlier dialog) and its outcome. */
  function follow(reading) {
    // Not disabled while it works: a disabled button would drop the focus it holds.
    refreshButton.setAttribute('aria-disabled', 'true');
    refreshButton.textContent = 'Reading the lists…';
    listsStatus.textContent = 'This can take up to two minutes. The dialog can stay open or be closed meanwhile.';
    fill(listsError, null);
    reading
      .then(
        (cat) => {
          if (closedNow) return;
          renderLists(cat);
          listsStatus.textContent = 'The lists were read again.';
        },
        (error) => {
          if (closedNow) return;
          listsStatus.textContent = '';
          fill(listsError, notice('err', h('span', { class: 'environment__lists-refusal', tabindex: '0' }, `The lists were not read again, and the earlier ones stay in use: ${error?.message ?? 'the service gave no reason.'}`)));
        },
      )
      .finally(() => {
        refreshButton.removeAttribute('aria-disabled');
        refreshButton.textContent = 'Read the lists again';
      });
  }

  function refreshLists() {
    if (pendingRefresh) return; // already reading: a second press changes nothing
    follow(readListsAgain());
  }

  function render(env) {
    const disk = diskText(env?.diskFreeBytes, env?.diskTotalBytes);
    const packages = entriesOf(env?.packages);
    const tuning = entriesOf(env?.environment);
    fill(
      content,
      h(
        'dl',
        { class: 'kv environment__facts' },
        FACTS.map(([key, label]) => {
          const value = factValue(env, key);
          const literal = key !== 'logicalCpus' && key !== 'platform' && key !== 'machine' && key !== 'python';
          return [h('dt', null, label), h('dd', { class: value && literal ? 'mono environment__value' : 'environment__value' }, value || absent())];
        }),
        h('dt', null, 'Disk'),
        h('dd', { title: disk ? `${fmt.int(env.diskFreeBytes)} bytes free of ${fmt.int(env.diskTotalBytes)} bytes` : null }, disk || absent()),
      ),
      h('h3', { class: 'environment__heading' }, 'Packages'),
      packages.length
        ? h(
            'table',
            { class: 'table environment__table' },
            h('caption', { class: 'visually-hidden' }, 'Installed packages'),
            h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Package'), h('th', { scope: 'col' }, 'Version'))),
            h(
              'tbody',
              null,
              packages.map(([name, version]) => h('tr', null, h('td', { class: 'mono' }, name), h('td', { class: 'mono' }, String(version)))),
            ),
          )
        : h('p', { class: 'muted' }, 'Not reported'),
      h('p', { class: 'field__help' }, 'Only installed packages are listed.'),
      h('h3', { class: 'environment__heading' }, 'Tuning variables'),
      tuning.length
        ? h('dl', { class: 'kv environment__facts' }, tuning.map(([name, value]) => [h('dt', { class: 'mono' }, name), h('dd', { class: 'mono environment__value' }, String(value))]))
        : h('p', null, `${NO_TUNING}.`),
      h('p', { class: 'environment__gpu' }, GPU_LINE),
    );
  }

  async function load() {
    retryButton.hidden = true;
    copyButton.disabled = true;
    fill(content, loading('Reading the trainer environment…'));
    try {
      const env = await api.environment({ signal: controller.signal });
      current = env;
      render(env);
      copyButton.disabled = false;
    } catch (error) {
      if (error?.name === 'AbortError') return;
      fill(content, notice('err', `The trainer environment could not be read: ${error.message}`));
      retryButton.hidden = false;
    }
  }

  const { closed } = openDialog({
    title: 'Trainer environment',
    wide: true,
    body: h('div', { class: 'environment__body' }, content, listsSection),
    foot: (close) => [retryButton, copyButton, h('button', { class: 'btn btn--primary', type: 'button', onclick: () => close() }, 'Close')],
  });
  dialogsOpen += 1;
  load();
  loadLists();
  if (pendingRefresh) follow(pendingRefresh);
  return closed.then(() => {
    closedNow = true;
    dialogsOpen -= 1;
    controller.abort();
  });
}
