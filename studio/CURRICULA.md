# Local training curricula

A fine-tune phase can set `weights` to `{"bestValidation":"Val Dataset"}` to start its primary network from the preceding phase's best score on that dataset. Set `val.save_best` in the preceding phase. Release uses the file produced by that exact attempt, including its evaluated raw or EMA role. The handoff records the metric, value, and actual validation step. The new phase starts with fresh optimizer state and discriminator weights. Use `continue` for full-state continuation or `raw` / `ema` for the last numbered checkpoint.

Use `CurriculumManager(queue)` with the same `QueueManager` instance as the queue routes. The service can expose these methods directly.

| Method | Suggested route | Input |
| --- | --- | --- |
| `create(name, base_yaml, phases, request_id=None)` | `POST /api/curricula` | `{"name": string, "baseYaml": string, "phases": Phase[], "requestId"?: string}` |
| `list_curricula()` | `GET /api/curricula` | None |
| `detail(curriculum_id)` | `GET /api/curriculum?id=…` | Curriculum ID |
| `retry_phase(curriculum_id, phase_id, edits=None, request_id=None)` | `POST /api/curriculum/retry` | `{"id": string, "phaseId": string, "edits"?: Edit[] or null, "requestId"?: string}` |

A submitted phase has `{"name": string, "mode": "start" or "continue" or "finetune", "updates": positive integer, "edits": Edit[], "weights"?: "raw" or "ema"}`. Use `start` for the first phase. Later phases reference the immediately preceding phase. `weights` applies to `finetune` and defaults to `raw`. Each edit uses the existing render payload with `path` and either `value` or `remove: true`. Edits apply independently to the original YAML. Studio stores that YAML, the original edits and the rendered phase YAML with comments retained.

List returns summaries with `id`, `name`, `createdAt`, `baseDigest`, `phaseCount`, and `phaseEntries` (phase ID to current queue entry ID). Create and detail return the summary plus `baseYaml` and `phases`. Each detail phase has its original phase fields, `id`, optional `predecessorPhaseId`, `startStep`, `targetStep`, `entry` (current queue entry) and `revisions` (all its queue entries in publication order). Use queue detail for a revision's saved YAML and the linked run detail for metrics, checkpoints and its effective launch. Release through the existing `queue.release(entry_id)` route. The queue owns Stop, Hold and Cancel.

All entries begin held. Release only succeeds for the current revision and, after the first phase, after verified completion of its predecessor. Release pins the source entry, run, attempt, numbered step and file versions in `entry.handoff`. Dispatch rechecks the pin and lease before launching. Creation and revision never release an entry. The same request ID and payload replay the original creation or revision. A different payload using that ID raises `ValueError`. Revision request IDs are scoped to a phase. Preflight failures can supply an `issues` array with `phaseId`.

- Start begins at step zero and may use pretrained inputs from the base recipe. Full-state imports belong to Continue.
- Continue restores the immediately preceding phase's exact `targetStep.state` and matching numbered model files, seed, optimizers and schedulers. `targetStep = startStep + updates`. Continue requires compatible architecture, optimizer and scheduler settings. The trainer validates the saved loader cursor and geometry. The cursor records dataset indices. The checkpoint pin records file metadata. Dataset contents and augmentation RNG state are outside that record.
- Fine-tune uses the previous phase's verified final numbered raw or EMA weights. It copies the selected model privately, plus the numbered discriminator when both phases use the same active GAN discriminator. Optimizers, schedulers, sampler and step are initialized for a fresh run. `startStep = 0` and `targetStep = updates`. Fine-tune can use the numbered weights after the predecessor's training-state file has been removed.

`retry_phase` returns a new held queue entry. Omit `edits` to reuse the current edits, pass `[]` to use the base settings, or supply a replacement edit list. The phase's mode and update count stay fixed. A held, cancelled, stopped or failed phase can be revised. Completed phases remain the source for successors. Revisions retain prior runs and forward/backward entry links. A stopped or failed phase retries from its original phase boundary in a new run after explicit release.
