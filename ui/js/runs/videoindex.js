// The analysis kind index-video-clips (the indexes of a clip-training recipe's videos): its progress words
// and its result, for the analysis jobs view. The job itself is started from the training videos panel
// (recipe/video-prep.js).
//
//   videoIndexProgressText(progress) -> string
//   renderVideoIndex(host, {job, result, artifacts}) -> cleanup

import { h, mount, fmt } from '../dom.js';

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/** "Indexing ep01.mkv, 3 of 12 done", from {stage, total, current, action, episode}. */
export function videoIndexProgressText(progress) {
  if (!isObject(progress)) return '';
  const counted = Number.isFinite(progress.current) && Number.isFinite(progress.total) && progress.total > 0;
  const count = counted ? `${fmt.int(Math.min(progress.current, progress.total))} of ${fmt.int(progress.total)} done` : '';
  if (progress.stage === 'done') return count ? `Finishing, ${count}` : 'Finishing';
  const name = typeof progress.episode === 'string' && progress.episode ? progress.episode : '';
  // The last event the service published: the video being read, or the one just finished.
  const episode = !name ? '' : progress.action === 'indexing' ? `Indexing ${name}` : progress.action === 'indexed' ? `Indexed ${name}` : progress.action === 'skipped' ? `Reused the index of ${name}` : '';
  if (episode && count) return `${episode}, ${count}`;
  return episode || count || '';
}

/** The counts the job reports, as it reports them: an absent number stays absent. */
export function renderVideoIndex(host, { result } = {}) {
  const r = isObject(result) ? result : {};
  const rows = [
    ['Videos', r.total],
    ['Indexes written', r.written],
    ['Reused', r.skipped],
    ['Ready', r.ready],
  ].filter(([, value]) => Number.isFinite(value));
  mount(
    host,
    rows.length
      ? h('dl', { class: 'kv' }, rows.map(([label, value]) => [h('dt', null, label), h('dd', { class: 'num' }, fmt.int(value))]))
      : h('p', { class: 'muted' }, 'The job reported no counts.'),
  );
  return () => mount(host);
}
