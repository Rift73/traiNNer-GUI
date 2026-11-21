// Result of a completed loss report job (analysis kind `loss-report`): the gradient of each generator
// loss term, measured at the network's output or at one named parameter, on a few training batches,
// without applying an update. The producer is studio/loss_report.py (`_measure`, `loss_report`).
// `job` may be null.
//
//   renderLossReport(host, {job, result, artifacts}) -> cleanup
//   lossReportProgressText(progress, job) -> '' (the service reports no progress for this kind)
//
// Shapes:
//   result = {task, architecture, device, compiled, precision: 'float32' | <amp dtype>, update, seed,
//     target: 'generator_output' | <parameter name>, generatorWeights, discriminatorWeights,
//     optimizerUpdates: 0, batches: [batch, …]}
//   batch = {totalLoss, totalGradientNorm, independentTermResidualNorm,
//     adversarialBalance: null | {ganNorm, otherNorm, ganToOtherNorm, cosineWithOther}, terms: [term, …]}
//   term = {term: 'l_g_l1', weightedLoss, connected, gradientNorm?, cosineWithTotal?, squaredNormShare}
//   Any number can be null: the service sends a value that is not finite as null.
//   `generatorWeights` is the path the network was loaded from, whether the job or the recipe named it;
//   null means neither did.
//   `independentTermResidualNorm` is the size of (total gradient − sum of the terms' gradients). A custom
//   backward step can gate the sum, so the terms measured one by one need not add up to the total.
//
// With several batches every value in the terms table and in the balance is the mean of the batches in
// which it is finite; the batches themselves are listed in "Each batch".

import { h, fmt } from '../dom.js';
import { notice } from '../ui.js';
import { isObject, isText, baseName, remainder } from './split.js';

const P = 'lossreport';
/** Below this part of the total's size, the difference between the sum of the terms and the total is rounding. */
export const ADD_UP_LIMIT = 0.001;

const NOT_CONNECTED = 'This term does not reach what was measured, so it has no gradient there.';
const TERMS_HELP =
  "Share is each term's part of the summed squared gradient sizes. Direction compares a term's gradient with the total: 1 is the same way, 0 unrelated, -1 the opposite way.";
const BALANCE_HELP = 'The other terms are measured together in one pass, so anything that acts on their sum is included.';
const NO_WEIGHTS = 'None: neither the job nor the recipe named a file';

const BATCH_KEYS = new Set(['totalLoss', 'totalGradientNorm', 'independentTermResidualNorm', 'adversarialBalance', 'terms']);
const TERM_KEYS = new Set(['term', 'weightedLoss', 'connected', 'gradientNorm', 'cosineWithTotal', 'squaredNormShare']);
const BALANCE_KEYS = new Set(['ganNorm', 'otherNorm', 'ganToOtherNorm', 'cosineWithOther']);

// ───────── pure helpers (exported for checks) ─────────

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/** The service reports no progress for this kind, so this is always ''. */
export function lossReportProgressText() {
  return '';
}

/** A number with up to four significant digits: 12.47, 0.0003417, 1.2e-7. '' for a value that is not a number. */
export function numText(n) {
  if (!isNum(n)) return '';
  if (n === 0) return '0';
  const rounded = Number(n.toPrecision(4));
  const abs = Math.abs(rounded);
  return abs >= 1e6 || abs < 1e-4 ? rounded.toExponential() : String(rounded);
}

/** A direction (cosine) with three decimals; never "-0.000". '' for a value that is not a number. */
export function directionText(n) {
  if (!isNum(n)) return '';
  const text = n.toFixed(3);
  return Number(text) === 0 ? '0.000' : text;
}

/** A share (0 to 1) as a percentage with one decimal; a share above zero never prints as 0.0%. */
export function shareText(share) {
  if (!isNum(share)) return '';
  const percent = share * 100;
  return percent > 0 && percent < 0.05 ? 'Under 0.1%' : `${percent.toFixed(1)}%`;
}

/** The mean of the finite numbers in `values`; null when there is none. */
export function mean(values) {
  const finite = (Array.isArray(values) ? values : []).filter(isNum);
  return finite.length ? finite.reduce((sum, v) => sum + v, 0) / finite.length : null;
}

/** The first sentence. `terms` and `batches` are counts. */
export function leadText({ terms = 0, batches = 0, update = null, optimizerUpdates = null } = {}) {
  const on = `on ${fmt.plural(batches, 'batch', 'batches')}${isNum(update) ? ` at update ${fmt.int(update)}` : ''}.`;
  // Said only when the job states it: 0 updates means no weight moved.
  const kept = isNum(optimizerUpdates) && optimizerUpdates > 0 ? ` The job reported ${fmt.plural(optimizerUpdates, 'optimizer update')}.` : optimizerUpdates === 0 ? ' No weights were changed.' : '';
  if (!terms) return `No loss term was measured ${on}${kept}`;
  const what = terms === 1 ? 'The gradient of 1 loss term was measured' : `The gradients of ${fmt.int(terms)} loss terms were measured`;
  return `${what} ${on}${kept}`;
}

/**
 * One row per loss term, in the order the service names them, each value the mean of the batches in
 * which it is finite. `connected` is false only for a term that is connected in no batch.
 */
export function termRows(batches) {
  const byName = new Map();
  for (const batch of Array.isArray(batches) ? batches : []) {
    for (const item of isObject(batch) && Array.isArray(batch.terms) ? batch.terms : []) {
      if (!isObject(item) || !isText(item.term)) continue;
      let row = byName.get(item.term);
      if (!row) {
        row = { term: item.term, connected: false, loss: [], norm: [], share: [], direction: [] };
        byName.set(item.term, row);
      }
      if (item.connected === true) row.connected = true;
      row.loss.push(item.weightedLoss);
      row.norm.push(item.gradientNorm);
      row.share.push(item.squaredNormShare);
      row.direction.push(item.cosineWithTotal);
    }
  }
  return [...byName.values()].map((row) => ({
    term: row.term,
    connected: row.connected,
    weightedLoss: mean(row.loss),
    gradientNorm: mean(row.norm),
    share: mean(row.share),
    direction: mean(row.direction),
  }));
}

/**
 * Whether the terms, measured one by one, add up to the total gradient.
 * @param {number|null} residual size of (total gradient − sum of the terms' gradients)
 * @param {number|null} total size of the total gradient
 * @returns {{kind: 'ok'|'warn'|'unknown', text: string}}
 */
const ADD_UP_WHY =
  'Something between the losses and the measured point changes the sum, for example a clamp or a custom backward step. The shares and directions above describe each term on its own, not its part in the real update.';

export function addUpVerdict(residual, total) {
  if (!isNum(residual)) return { kind: 'unknown', text: 'The service could not compare the sum of the terms with the total.' };
  const size = numText(residual);
  const within = `Measured one by one, the terms add up to the total gradient (difference ${size}, within rounding).`;
  if (residual === 0) return { kind: 'ok', text: within };
  if (!isNum(total)) {
    return { kind: 'unknown', text: `Measured one by one, the terms differ from the total gradient by ${size}. The total's size was not measured, so the difference cannot be judged.` };
  }
  const ratio = total > 0 ? residual / total : null;
  if (ratio !== null && ratio < ADD_UP_LIMIT) return { kind: 'ok', text: within };
  const against = ratio === null ? `the difference is ${size}, while the total's size is 0.` : `the difference is ${size}, ${numText(ratio * 100)}% of the total's size.`;
  return {
    kind: 'warn',
    text: `Measured one by one, the terms do not add up to the total gradient: ${against} ${ADD_UP_WHY}`,
  };
}

/**
 * The same question over several batches: each batch is judged on its own numbers, so a batch the
 * service could not compare is never covered by the others.
 * @returns {{kind: 'ok'|'warn'|'unknown', text: string}}
 */
export function addUpAcross(batches) {
  const list = (Array.isArray(batches) ? batches : []).filter(isObject);
  if (list.length <= 1) return addUpVerdict(list[0]?.independentTermResidualNorm ?? null, list[0]?.totalGradientNorm ?? null);
  const n = list.length;
  const each = list.map((batch, index) => ({
    index,
    residual: batch.independentTermResidualNorm,
    total: batch.totalGradientNorm,
    kind: addUpVerdict(batch.independentTermResidualNorm, batch.totalGradientNorm).kind,
  }));
  const unknown = each.filter((item) => item.kind === 'unknown').length;
  const unknownText = unknown ? ` In ${fmt.int(unknown)} of ${fmt.int(n)} batches the comparison could not be made.` : '';
  const off = each.filter((item) => item.kind === 'warn');
  if (off.length) {
    const ratio = (item) => (isNum(item.total) && item.total > 0 ? item.residual / item.total : Infinity);
    const worst = off.reduce((a, b) => (ratio(b) > ratio(a) ? b : a));
    const size = numText(worst.residual);
    const against = Number.isFinite(ratio(worst)) ? `${size}, ${numText(ratio(worst) * 100)}% of the total's size` : `${size}, while the total's size is 0`;
    return {
      kind: 'warn',
      text: `Measured one by one, the terms do not add up to the total gradient in ${fmt.int(off.length)} of ${fmt.int(n)} batches: the largest difference is ${against} (batch ${fmt.int(worst.index + 1)}). ${ADD_UP_WHY}${unknownText}`,
    };
  }
  const ok = each.filter((item) => item.kind === 'ok');
  if (!ok.length) return { kind: 'unknown', text: 'The service could not compare the sum of the terms with the total.' };
  const where = ok.length === n ? `in all ${fmt.int(n)} batches` : `in ${fmt.int(ok.length)} of ${fmt.int(n)} batches`;
  const largest = Math.max(...ok.map((item) => item.residual));
  return { kind: 'ok', text: `Measured one by one, the terms add up to the total gradient ${where} (largest difference ${numText(largest)}, within rounding).${unknownText}` };
}

// ───────── renderer ─────────

/**
 * Draw a loss report result into `host`.
 * @param {HTMLElement} host
 * @param {{job?: object|null, result?: object|null, artifacts?: Array<object>}} options
 * @returns {() => void} cleanup
 */
export function renderLossReport(host, { result = null } = {}) {
  const r = isObject(result) ? result : {};
  const batches = Array.isArray(r.batches) ? r.batches.filter(isObject) : [];
  const rawBlock = (rest) =>
    rest
      ? h(
          'details',
          { class: `${P}__raw` },
          h('summary', { class: `${P}__raw-summary` }, 'Everything else the job reported'),
          h('pre', { class: `${P}__json`, tabindex: '0' }, JSON.stringify(rest, null, 2)),
        )
      : null;
  const finish = (root) => {
    host.appendChild(root);
    return () => root.remove();
  };

  if (!batches.length) return finish(h('div', { class: P }, notice('warn', 'The job finished without a measured batch.'), rawBlock(remainder(r, []))));

  const count = batches.length;
  const many = count > 1;
  /** A value's text, or "Not measured" for a value the service sent as null. */
  const value = (text) => text || h('span', { class: `${P}__muted` }, 'Not measured');
  const wrapped = (label, table) => h('div', { class: `${P}__table-wrap`, tabindex: '0', role: 'region', 'aria-label': label }, table);
  /** Keys of `result` that are drawn below; everything else goes into the closing block. */
  const shown = [];

  // 1. The outcome.
  const rows = termRows(batches);
  if (isNum(r.update)) shown.push(['update']);

  // 2. Facts; a value the job did not report has no row.
  const facts = [];
  if (isText(r.architecture)) {
    shown.push(['architecture']);
    facts.push(['Network', h('span', { class: 'mono' }, r.architecture)]);
  }
  if (isText(r.target)) {
    shown.push(['target']);
    facts.push(['Measured at', r.target === 'generator_output' ? "The network's output" : ['Parameter ', h('span', { class: 'mono' }, r.target)]]);
  }
  if (isText(r.precision)) {
    shown.push(['precision']);
    facts.push(['Number type', r.precision === 'float32' ? '32-bit floats' : h('span', { class: 'mono' }, r.precision)]);
  }
  if (isNum(r.seed) || isText(r.seed)) {
    shown.push(['seed']);
    facts.push(['Seed', String(r.seed)]);
  }
  const fileValue = (path) => h('span', { class: `${P}__file`, title: path }, baseName(path) || path);
  if (isText(r.generatorWeights) || r.generatorWeights === null) {
    shown.push(['generatorWeights']);
    facts.push(['Generator weights', isText(r.generatorWeights) ? fileValue(r.generatorWeights) : NO_WEIGHTS]);
  }
  if (isText(r.discriminatorWeights) || r.discriminatorWeights === null) {
    shown.push(['discriminatorWeights']);
    if (isText(r.discriminatorWeights)) facts.push(['Discriminator weights', fileValue(r.discriminatorWeights)]);
  }

  // 3. The loss terms.
  const totalLoss = mean(batches.map((batch) => batch.totalLoss));
  const totalNorm = mean(batches.map((batch) => batch.totalGradientNorm));
  const termRow = (row) => {
    const head = [h('td', { class: `${P}__term` }, row.term), h('td', { class: 'num' }, value(numText(row.weightedLoss)))];
    if (!row.connected) return h('tr', null, head, h('td', { class: `${P}__off`, colspan: '3', title: NOT_CONNECTED }, 'Not connected'));
    const share = isNum(row.share) ? Math.min(1, Math.max(0, row.share)) : null;
    return h(
      'tr',
      null,
      head,
      h('td', { class: 'num' }, value(numText(row.gradientNorm))),
      h(
        'td',
        { class: `num ${P}__share` },
        value(shareText(row.share)),
        share === null ? null : h('span', { class: `${P}__track`, 'aria-hidden': 'true' }, h('span', { class: `${P}__bar`, style: { width: `${(share * 100).toFixed(1)}%` } })),
      ),
      h('td', { class: 'num' }, value(directionText(row.direction))),
    );
  };
  const quantity = (n) => numText(n) || 'not measured';
  const termsPart = [
    h('h4', { class: `${P}__title` }, 'Loss terms'),
    many && rows.length ? h('p', { class: `${P}__note` }, `Each value is the mean over the batches in which it was measured, out of ${fmt.int(count)}.`) : null,
    rows.length
      ? [
          wrapped(
            'Loss terms',
            h(
              'table',
              { class: `table ${P}__table` },
              h(
                'thead',
                null,
                h(
                  'tr',
                  null,
                  h('th', { scope: 'col' }, 'Term'),
                  h('th', { scope: 'col', class: 'num' }, 'Weighted loss'),
                  h('th', { scope: 'col', class: 'num' }, 'Gradient size'),
                  h('th', { scope: 'col', class: 'num' }, 'Share'),
                  h('th', { scope: 'col', class: 'num' }, 'Direction'),
                ),
              ),
              h('tbody', null, rows.map(termRow)),
            ),
          ),
          h('p', { class: `${P}__note` }, TERMS_HELP),
        ]
      : h('p', { class: `${P}__note` }, 'The job listed no loss term.'),
    h('p', { class: `${P}__note` }, many ? `Mean total loss ${quantity(totalLoss)}; mean total gradient size ${quantity(totalNorm)}.` : `Total loss ${quantity(totalLoss)}; total gradient size ${quantity(totalNorm)}.`),
  ];

  // 4. Whether the terms add up to the total.
  const verdict = addUpAcross(batches);
  const addUpPart = [h('h4', { class: `${P}__title` }, 'Do the terms add up?'), verdict.kind === 'warn' ? notice('warn', verdict.text) : h('p', { class: `${P}__verdict` }, verdict.text)];

  // 5. The GAN term against all the others, in the batches that have it.
  const balances = batches.map((batch) => (isObject(batch.adversarialBalance) ? batch.adversarialBalance : null));
  const withBalance = balances.filter(Boolean);
  let balancePart = null;
  if (withBalance.length) {
    const of = (key) => mean(withBalance.map((balance) => balance[key]));
    const ratio = numText(of('ganToOtherNorm'));
    let scope = null;
    if (withBalance.length < count) scope = `Measured in ${fmt.int(withBalance.length)} of ${fmt.int(count)} batches.${withBalance.length > 1 ? ' Each value is the mean of those.' : ''}`;
    else if (many) scope = `Each value is the mean over the batches in which it was measured, out of ${fmt.int(count)}.`;
    const balanceFacts = [
      ['GAN gradient size', value(numText(of('ganNorm')))],
      ['All other terms together', value(numText(of('otherNorm')))],
      ['GAN compared with the others', ratio ? `${ratio} times as large` : value('')],
      ['Direction against the others', value(directionText(of('cosineWithOther')))],
    ];
    balancePart = [
      h('h4', { class: `${P}__title` }, 'Adversarial balance'),
      scope ? h('p', { class: `${P}__note` }, scope) : null,
      h('dl', { class: `kv ${P}__facts` }, balanceFacts.map(([label, shownValue]) => [h('dt', null, label), h('dd', null, shownValue)])),
      h('p', { class: `${P}__note` }, BALANCE_HELP),
    ];
  }

  // 6. The batches one by one.
  const batchesPart = many
    ? h(
        'details',
        { class: `${P}__batches` },
        h('summary', { class: `${P}__batches-summary` }, 'Each batch'),
        wrapped(
          'Each batch',
          h(
            'table',
            { class: `table ${P}__table` },
            h(
              'thead',
              null,
              h(
                'tr',
                null,
                h('th', { scope: 'col', class: 'num' }, 'Batch'),
                h('th', { scope: 'col', class: 'num' }, 'Total loss'),
                h('th', { scope: 'col', class: 'num' }, 'Total gradient size'),
                h('th', { scope: 'col', class: 'num' }, 'Difference from the sum of terms'),
                withBalance.length ? h('th', { scope: 'col', class: 'num' }, 'GAN compared with the others') : null,
              ),
            ),
            h(
              'tbody',
              null,
              batches.map((batch, index) =>
                h(
                  'tr',
                  null,
                  h('td', { class: 'num' }, fmt.int(index + 1)),
                  h('td', { class: 'num' }, value(numText(batch.totalLoss))),
                  h('td', { class: 'num' }, value(numText(batch.totalGradientNorm))),
                  h('td', { class: 'num' }, value(numText(batch.independentTermResidualNorm))),
                  withBalance.length ? h('td', { class: 'num' }, value(numText(balances[index]?.ganToOtherNorm))) : null,
                ),
              ),
            ),
          ),
        ),
      )
    : null;

  // 7. What is left of the answer. The batches stay in it while they hold more than is drawn above.
  const known = (object, keys) => Object.keys(object).every((key) => keys.has(key));
  const plain =
    batches.length === r.batches.length &&
    batches.every(
      (batch) =>
        known(batch, BATCH_KEYS) &&
        (batch.adversarialBalance === null || batch.adversarialBalance === undefined || (isObject(batch.adversarialBalance) && known(batch.adversarialBalance, BALANCE_KEYS))) &&
        Array.isArray(batch.terms) &&
        batch.terms.every((item) => isObject(item) && isText(item.term) && known(item, TERM_KEYS)),
    );
  if (plain) shown.push(['batches']);

  return finish(
    h(
      'div',
      { class: P },
      h('p', { class: `${P}__lead` }, leadText({ terms: rows.length, batches: count, update: r.update, optimizerUpdates: r.optimizerUpdates })),
      facts.length ? h('dl', { class: `kv ${P}__facts` }, facts.map(([label, shownValue]) => [h('dt', null, label), h('dd', null, shownValue)])) : null,
      termsPart,
      addUpPart,
      balancePart,
      batchesPart,
      rawBlock(remainder(r, shown)),
    ),
  );
}
