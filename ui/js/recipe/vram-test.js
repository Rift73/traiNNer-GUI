import { api } from '../api.js';
import { h, mount } from '../dom.js';
import { recipe } from '../state/recipe.js';
import { openDialog } from '../ui.js';

const gib = (bytes) => (bytes / (1024 ** 3)).toFixed(2) + ' GiB';

export function openVramTest() {
  const draft = recipe.get();
  const info = h('div', { role: 'status', 'aria-live': 'polite' }, 'Checking trainer…');
  const error = h('p', { role: 'alert' });
  const start = h('button', { type: 'button', class: 'btn btn--primary', disabled: true, onclick: () => act(() => api.startVramTest(recipe.get().yaml)) }, 'Test 20 iterations');
  const cancel = h('button', { type: 'button', class: 'btn', hidden: true, onclick: () => act(() => api.cancelVramTest(id)) }, 'Cancel test');
  const modal = openDialog({
    title: 'VRAM test',
    body: h('div', { class: 'stack' },
      h('p', null, `Test ${draft.summary?.name || draft.name || 'this recipe'} for 20 training updates, including gradient accumulation. Compiled recipes can take time to warm up. The test skips validation and saves no checkpoints or run history.`),
      info, error),
    foot: (close) => [h('button', { type: 'button', class: 'btn', onclick: () => close() }, 'Close'), cancel, start],
  });
  let id = null;
  let timer;
  modal.closed.then(() => clearTimeout(timer));

  function display(result) {
    id = result.id;
    const busy = ['running', 'cancelling'].includes(result.status);
    start.disabled = busy;
    cancel.hidden = !busy;
    cancel.disabled = result.status === 'cancelling';
    if (result.status === 'idle') {
      mount(info, h('p', null, 'Ready to test.'));
      return;
    }
    const label = { running: 'Testing', completed: 'Completed', cancelling: 'Cancelling', cancelled: 'Cancelled',
      failed: result.outOfMemory ? 'Out of GPU memory' : 'Test failed' }[result.status];
    const rows = (result.devices || []).map((d) => h('section', null,
      h('strong', null, `GPU ${d.device} · ${d.name}`),
      h('div', null, `Peak allocated by PyTorch: ${gib(d.allocated)}`),
      h('div', null, `Peak reserved by PyTorch: ${gib(d.reserved)}`),
      h('div', null, `Highest device use sampled at update boundaries: ${gib(d.sampledUsed)} of ${gib(d.total)}`)));
    mount(info, h('strong', null, `${label} · ${result.name} · ${result.iterations}/${result.target}`), ...rows,
      rows.length ? h('p', { class: 'muted' }, 'Device usage includes other applications. Later recipe phases or larger inputs can use more memory.') : null);
    error.textContent = result.error || '';
    if (busy && modal.dialog.open) timer = setTimeout(() => act(api.vramTest), 800);
  }

  async function act(request) {
    clearTimeout(timer);
    start.disabled = cancel.disabled = true;
    error.textContent = '';
    try {
      const result = await request();
      if (modal.dialog.open) display(result);
    } catch (e) {
      if (!modal.dialog.open) return;
      const issues = (e.issues || []).filter(issue => issue.level === 'error');
      error.textContent = issues.length ? issues.map(issue => issue.message).join(' ') : e.message;
      start.disabled = false;
      cancel.disabled = false;
    }
  }
  act(api.vramTest);
}
