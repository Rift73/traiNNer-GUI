# traiNNer GUI frontend: design notes

Static frontend, no build step, no dependencies. Native ES modules and plain CSS. Serve `ui/` as it is on
the same origin as `/api`. `package.json` only marks the folder as ES modules for tooling.

## Identity: the proofing bench

The app exists to judge images, so the chrome is **neutral gray (R = G = B)**: a tinted surround shifts
perceived colour. Colour is reserved for meaning and is used the same way on every screen:

| Token | Meaning | Where |
| --- | --- | --- |
| `--gt` cyan | ground truth / reference | chips, wipe handle, chart series |
| `--lq` yellow | low-quality input | same |
| `--out` magenta | model output, a run that is training | same, run status |
| `--ok` `--warn` `--err` | check results | badges, issues, notices |

YAML text has its own three `--syn-*` colours; in code, colour means token type and nothing else. Chart
series use `--series-1..8` (in `runs.css`), never the role colours.

Images are shown on the **stage** (`.stage`, the same neutral booth in both themes) with crop marks.
Type is Bahnschrift (DIN, the engineering-drawing face; `font-stretch: var(--narrow)` for dense data) with
Cascadia Mono only for literal text: YAML, paths, option keys, log lines. Flat surfaces separated by
1px rules; no cards, no shadows except dialogs; 3px radius on controls only.

The header is a **pipeline rail**: Recipe, Data, Runs, in working order, each showing its live state.
The recipe comes first because it decides what data is needed (paired folders, class folders, images and
an annotation file, images and masks). Under the rail one **next-step line** says what stands between the
person and a training run: choose a recipe, choose and check its data, look it over, start it. That line
is the first-run workflow; there is no separate wizard.

## Rules every screen follows

- **Nothing invented.** Show only what the API returned. Absent or null fields render as absent
  ("Not reported"), never as zero, an estimate or a placeholder number. No fake progress bars: a bar
  needs a real `iteration` and `total`. Derived values (a difference image, a smoothed curve) are
  labelled as derived.
- **Copy.** Sentence case. Plain verbs. A button says what it does ("Check folders", "Start training")
  and keeps that name in the toast that follows. Errors say what happened and what to do; they do not
  apologise. Empty screens say what to do next. No all-caps labels, no "→" on buttons, no middle-dot
  separators. Say "ground truth" and "low quality" in full at least once per screen; GT/LQ after that.
- **Accessibility.** Real `<button>`, `<a>`, `<label for>`; visible focus (global `:focus-visible`);
  every control reachable by keyboard; `aria-live`/`role="status"` for results that appear later; status
  never by colour alone (pair colour with a word); respect `prefers-reduced-motion`.
- **Motion** only in answer to an action (open, expand, drag). No entrance animations.
- **Responsive** down to about 720px wide: panes stack.
- **CSS.** Use the tokens in `css/tokens.css` (never hard-code a colour outside tokens, except pixel data
  drawn to a canvas) and the shared classes in `css/base.css`. Prefix your own classes with your module
  name (`.runs__…`, `.compare__…`) and keep selectors one class deep to avoid specificity fights.

## File map

```
index.html              page; links every stylesheet, loads js/main.js
mobile.html             the phone edition's page: the same stylesheets and modules, then the three below
css/mobile.css          phone edition: sizes, shell, shared components    js/mobile-boot.js (theme before paint)
css/mobile-screens.css  phone edition: every screen as one column         js/mobile.js (where the person is, keyboard)
css/tokens.css          colour, type, space tokens (light + dark)
css/base.css            shell + shared components
css/recipe.css          recipe screen frame            js/views/recipe.js
css/recipe-form.css     options form                   js/recipe/form.js, schema.js, sections.js, fields.js
css/recipe-source.css   YAML pane + recipe chooser     js/recipe/source-pane.js, library.js, yaml-highlight.js, diff.js
css/data.css            Data screen                    js/views/data.js
css/compare.css         image comparison stage         js/components/compare.js
css/runs.css            Runs screen                    js/views/runs.js, js/components/chart.js, js/runs/detail.js,
                                                       telemetry.js, log.js, metrics.js, provenance.js, archives.js
css/queue.css           queue, entries, replay         js/runs/queue.js
css/validation.css      Validation tab (images, scores) js/runs/validation.js, scores.js, classes.js (per-class report)
css/evaluation.css      runs and evaluations side by side js/runs/evaluation.js
css/analysis.css        analysis jobs                  js/runs/analysis.js
css/studies.css         controlled studies             js/runs/studies.js
css/curricula.css       curricula (training in phases) js/runs/curricula.js
css/drift.css           weight drift result            js/runs/drift.js
css/taskdata.css        data audit result              js/runs/taskdata.js
css/holdout.css         evaluation, split leakage      js/runs/holdout.js
css/registration.css    pair alignment result          js/runs/registration.js
css/split.css           validation split result        js/runs/split.js
css/difference.css      image difference result        js/runs/difference.js
css/patches.css         patch extraction result        js/runs/patches.js
css/convert.css         weights conversion result      js/runs/convert.js
css/storage.css         disk space for checkpoints     js/runs/storage.js
css/frontier.css        weighing two scores            js/runs/frontier.js
css/average.css         a checkpoint average           js/runs/average.js
css/parameters.css      a network's parameter list     js/runs/parameters.js
css/lossreport.css      a loss report                  js/runs/lossreport.js
css/timing.css          how fast a run trained         js/runs/timing.js
css/runfile.css         run file dialogs               js/runs/runfile.js
css/path-picker.css     folder and file picker         js/components/path-picker.js
css/weights.css         checkpoint structure dialog    js/components/weights.js
css/environment.css     trainer environment dialog     js/components/environment.js
css/phone.css           Open on phone: rail button, dialog  js/components/phone.js (index.html only)
css/combobox.css        a value with known choices     js/components/combobox.js
js/main.js              rail, next-step line, router (#/recipe, #/data, #/runs[/…])
js/api.js               one function per endpoint; ApiError; imageUrl(), sizedUrl(), artifactUrl()
js/store.js             createStore, persist (localStorage), createChannel
js/dom.js               h(), mount(), fmt, debounce, clamp, uid
js/ui.js                toast, dialogs, tabs, issueList, badge, roleChip, emptyState, notice, loading, cropmarks
js/actions.js           startTraining(), addToQueue(), saveRecipe(), openAsRecipe(), go(), …Href()
js/state/*.js           conn, recipe, data, runs, queue, catalog: shared state
```

The service's own description of its API is `studio/API.md` (and `studio/CURRICULA.md`). `js/api.js` has
one function per endpoint, and its doc comments give each request and answer as the frontend uses them.

## Foundation reference

Read the source for exact signatures; this is the map.

- **`dom.js`**: `h(tag, props, ...children)` (props: `class`, `dataset`, `style` object, `on<event>`,
  `ref`, attributes; `value/checked/disabled` set as properties); `mount(el, ...children)` replaces
  children; `fmt.int/num/percent/dateTime/ago/dims/plural` return `''` for absent input; `debounce`
  (with `.cancel/.flush`), `clamp`, `uid`.
- **`api.js`**: `api.status/templates/recipes/recipe/saveRecipe/preflight/render/catalog/schema/scan/
  previewBlob/runs/run/startRun/stopRun/resumeRun`. All reject with `ApiError {message, status, issues,
  offline}`; pass `{signal}` to cancel. `catalog()` and `schema()` are cached. `imageUrl(path)` resolves
  to an object URL for a dataset image (cached, 4 at a time).
- **`state/recipe.js`**: `recipe.get()` gives `{phase, path, name, yaml, baseYaml, baseLabel, savedYaml,
  check, hasResult, ok, issues, summary, config, configCurrent, checkError, pending, saving}`.
  `recipe.subscribe(fn)` returns an unsubscribe. `setYaml(text)` for raw edits (check is debounced);
  `applyEdits([{path, value}])` for form edits (queued through `POST /api/render`, returns a promise that
  rejects when refused); `load(doc, {label, saved})`, `save(name)`, `revertToBase()`, `rebase()`,
  `isDirty()`, `hasChanges()`, `requestFocus(path)` / `onFocusRequest(fn)`, `datasetFolders()`.
  Helpers `parsePath(str)`, `getAt(obj, path)`.
- **`state/data.js`**: `data.get()` gives `{gt, lq, scale, phase, result, error, scanned}`;
  `setInputs(patch)`, `scan()`, `isCurrent()`.
- **`state/runs.js`**: `runs.get()` gives `{list, phase, error, selectedId}`; `watch()` (fast polling,
  returns release), `refresh()`, `select(id)`, `upsert(run)`, `start(yaml)`, `stop(id)`, `resume(id)`,
  `active()`. `runStatus(status)` gives `{kind: 'run'|'ok'|'err'|'', active, label}`.
- **`ui.js`**: `toast(msg, {kind, action})`, `openDialog({title, body, foot, wide, flush})`,
  `confirmDialog`, `promptDialog`, `tabs({items, active, onChange, label})` returning
  `{el, setActive, setBadge}`, `issueList(issues, {onSelect})`, `badge(text, kind)`,
  `roleChip('gt'|'lq'|'out', text)`, `emptyState`, `notice(kind, body, action)`, `loading(text)`,
  `cropmarks()`, `copyText`, `countIssues`.
- **Classes** (`base.css`): `.btn` (`--primary --danger --quiet --sm --icon`, `[aria-pressed]`), `.seg`,
  `.input` (`--num --path`), `.select`, `.textarea`, `.field` (`__label __help __error`), `.switch`,
  `.tabs/.tab`, `.pane` (`__head __title __body __body--pad __foot`), `.badge`, `.role`, `.issues/.issue`,
  `.empty`, `.loading`, `.spinner`, `.progress` (`--run --ok --err`), `.kv`, `.table`, `.stage`,
  `.cropmark`, `.dialog`, `.notice`, `.row`, `.grow`, `.stack`, `.mono`, `.num`, `.muted`, `.truncate`,
  `.visually-hidden`.

## Module contract

A view or part exports a mount function that appends into the element it is given and returns a cleanup
function. Cleanup must remove every subscription, timer, observer, global listener and object URL the
module created. Re-render only what changed: never rebuild a subtree that holds keyboard focus or a
text selection in answer to a background update.

## Second round: queue, history, validation, tasks (2026-11-20)

`studio/API.md` is the contract (queue, history, validation gallery, run monitor, checkpoint structure);
`js/api.js` has one documented function for each route.

### Words

- **Queue entry**: a saved copy of a recipe. States, always shown with these words: `held` "On hold"
  (never starts by itself), `pending` "Released" (starts when the trainer is free, in release order),
  `running` "Training", `stopping` "Stopping", `completed`, `stopped`, `failed`, `cancelled`.
  Verbs: "Release", "Put on hold", "Cancel entry", "Stop training", "Pause the queue" /
  "Continue the queue" (pausing stops further entries from starting; one that is training goes on).
- **Start training** starts now. **Add to queue** keeps a copy for later. Never blur the two.
- **Replay**: a new held entry that repeats a run's recipe with its recorded seed.
- **Imported run**: an experiment found in the trainer's `experiments/` folder that this service did not
  launch (`source: "cli"`, status `imported`). It can be read and compared, not stopped or resumed.
- **Reduced preview**: dataset previews are at most 1536 px on the long edge (`PREVIEW_MAX`). When an
  image is larger, say so beside it ("Shown reduced from 4000 × 3000"); never present a reduced
  image as the original. Validation outputs (`image.url` without `maxSize`) are original files.
- Task wording comes from `state/catalog.js`: `catalog.taskLabel(id)` and `folderWords(taskId)`
  (what `dataroot_gt` / `dataroot_lq` hold for that task: masks and images for segmentation).

### Shared state and actions added

- **`state/queue.js`**: `queue.get()` gives `{paused, activeId, entries, phase, error}`; `watch()`,
  `refresh()`, `open()`, `finished()`, `held()`, `pending()`, `byId(id)`, `forRun(runId)`,
  `submit(yaml, {recipePath})`, `replay(runId)`, `release/hold/cancel/stop(id)`, `setPaused(bool)`.
  `entryStatus(status)` gives `{kind, label, open, waiting, active}`; `ENTRY_ACTIONS` lists each action's
  label and the states it is allowed from.
- **`state/catalog.js`**: `catalog.load()`, `catalog.get()` (`{phase, tasks, architectures, losses}`),
  `catalog.task(id)`, `catalog.taskLabel(id)`, `catalog.networksFor(taskId)`, `folderWords(taskId)`.
- **`state/runs.js`**: `runStatus()` also returns `imported` and `resumable`; `isImported(run)`;
  `runs.byId(id)`; `runs.start(yaml, requestId)`.
- **`state/recipe.js`**: `applyEdits` accepts `{path, remove: true}`; a list cannot be appended to by
  index (set the whole list); `save(name, {overwrite})` sends the file revision and rejects with status 409
  when the saved file changed; state has `revision`.
- **`actions.js`**: `startTraining()`, `addToQueue()`, `saveRecipe()`, `openAsRecipe({yaml, name, label})`,
  `go(hash)`, `runHref(id)`, `queueHref(id?)`, `compareHref(left, right?)`.
- **`api.js`**: `newRequestId()`, `PREVIEW_MAX`, `THUMB_SIZE`, `imageUrl(path, {maxSize})` (dataset files,
  object URLs), `sizedUrl(url, maxSize)` (gallery addresses, plain GET, usable as `<img src>`),
  `artifactUrl(runId, artifact)`.

### Runs screen: routes and parts

The hash after `#/runs/` selects what the main area shows:

| Hash | Main area | Module |
| --- | --- | --- |
| `#/runs` | the selected or newest run, or the empty state | `views/runs.js` |
| `#/runs/<runId>` | one run: Telemetry, Validation, Log, Files, Recipe | `views/runs.js` |
| `#/runs/queue`, `#/runs/queue/<entryId>` | the queue and one entry | `runs/queue.js` |
| `#/runs/compare/<left>[/<right>]` | two runs or evaluation jobs side by side; a side is a run id or `job:<jobId>` | `runs/evaluation.js` |

Part contracts (each returns a handle; `destroy()` is the cleanup of the module contract):

- `runs/queue.js`: `mountQueueSummary(host, {selectedId})` gives `{setSelected(id), destroy()}`: the block
  above the run list (open entries, pause control, link to the queue).
  `mountQueueView(host, {entryId})` gives `{setEntry(id), destroy()}`: the main area for the queue routes.
  `openReplayDialog(run)`: what a replay of this run would repeat, then "Add replay to queue" or
  "Open as recipe".
- `runs/validation.js`: `mountValidation(host, {run})` gives `{setRun(run), destroy()}`: the Validation tab
  (saved validation images, per iteration and per sample, on the comparison stage against the recorded
  ground truth and input, and against another iteration).
- `runs/evaluation.js`: `mountRunCompare(host, {leftId, rightId})` gives `{setRuns(leftId, rightId), destroy()}`:
  recorded evaluations of two sides (`POST /api/compare`), each a run or a completed evaluation job
  (`job:<jobId>`), per-image wins and losses or the whole-dataset score, the two outputs on the comparison
  stage, and one logged series of both runs on one chart.
- `components/path-picker.js`: `pickPath({title, mode: 'folder'|'file', start, extensions, confirmLabel})`
  resolves to an absolute path on the trainer machine or `null`.
- `components/weights.js`: `openWeightsDialog({path, compareWith})`: structure of a `.safetensors` file,
  optionally against a second file. Structure only: it never says two checkpoints hold the same values.
- `components/compare.js`: `createCompare(host, {modes, emptyText})` gives `{setSources({a, b}), destroy()}`;
  a source is `{label, role, load: () => Promise<string>, note?: string}` where `load` resolves to an image
  address (object URL or GET address) and `note` is a short provenance line shown under the label
  (for example "Shown reduced from 4000 × 3000").

Each part is loaded with a dynamic `import()` by its host, which shows a notice when the part is missing or
fails, so one broken part never takes the screen down.

## Third round: analysis jobs, measured scores, provenance, more tasks (2026-11-20)

### Words

- **Analysis job**: work the person starts on purpose that is not training: "Prediction" (run a checkpoint
  on an image or a folder), "ONNX export", "Loss report" (gradient size of each configured loss term on a
  few batches). It runs on the trainer machine's CPU, one at a time, in a private job folder. Never call it
  a run, and never suggest it touches training.
- **Scores**: the numbers a validation pass measured (`GET /api/run/validation`), as opposed to "Images"
  (the saved outputs). **Class results**: the per-class table of one evaluation (segmentation,
  classification).
- **Provenance**: what each attempt of a run actually ran with (command, device, versions, seed), as the
  service recorded it. Say "not recorded" for anything absent; never fill in from the current machine.
- **Experiments folder**: a folder of finished experiments added for reading (`/api/archives`). Runs found
  there can be read, compared and replayed as a recipe, not stopped or resumed.
- **Class folders**: classification reads one folder per split with one subfolder per class.
  `folderWords(taskId).paired === false` marks a task with a single folder; a part must then not ask for or
  describe a second one.

### Routes and parts added

| Hash | Main area | Module |
| --- | --- | --- |
| `#/runs/analysis`, `#/runs/analysis/<jobId>` | analysis jobs and one job | `runs/analysis.js` |

- `runs/analysis.js`: `mountAnalysisView(host, {jobId})` gives `{setJob(id|null), destroy()}`.
  `openAnalysisDialog({kind, yaml, weights, input, sourceLabel})` resolves to the created job or `null`
  and opens the job when one was created. Entry points: a checkpoint in a run's Files tab ("Try on an
  image", "Export ONNX"), the Losses section of the Recipe form ("Measure these losses"), and
  "New analysis job" in the view.
- `runs/validation.js`: the Validation tab has two views, "Images" and "Scores" (measured rows per
  dataset and metric, a chart per metric, and class results for rows that carry class counts).
- `runs/provenance.js`: `mountProvenance(host, {run})` gives `{destroy()}`: the Provenance tab.
- `runs/archives.js`: `mountArchives(host)` gives `{destroy()}`: the "Experiments folders" block under
  the run list (add a folder with the picker, forget one).
- `components/environment.js`: `openEnvironmentDialog()`: what the trainer machine reports about itself
  (opened from the connection chip in the rail).
- `actions.js`: `analysisHref(id?)`.
- `api.js`: `validation`, `classes`, `provenance`, `archives`, `addArchive`, `removeArchive`, `alignment`,
  `environment`, `analysisJobs`, `analysisJob`, `analysisLog`, `analysisResult`, `analysisStart`,
  `analysisCancel`.

### Studies, import, packing, detection (same day, later contracts)

- **Study**: one base recipe, a few named **arms** (lists of option edits), and the same **seeds** for every
  arm. Creating a study saves every arm/seed cell as an ordinary queue entry **on hold**; "Release" stays the
  only thing that authorises training. A comparison of two arms gives one row per seed; rows are never
  pooled into one verdict and no significance is claimed.
- **Dataset packing**: a fourth analysis kind that copies paired folders into two LMDB databases and
  answers with the recipe lines that point at them. It needs no recipe and writes a full copy to disk:
  say so before it starts.
- **NeoSR import**: a neosr TOML translated into a draft by the service. Entries it could not map stay in
  a `neosr_review` section that must be resolved and removed before training; the form says so.
- **Whole-image tasks** (classification, detection): the catalog marks them with
  `datasetGeometry: 'whole_images'`; `folderWords()` then has `paired: false`, and for detection a `file`
  entry for the annotation file. The Data screen lists what is there (class folders, images); the check
  for these tasks is the data audit described in the fourth round below, never an imitation of the paired
  folder check.

| Hash | Main area | Module |
| --- | --- | --- |
| `#/runs/studies`, `#/runs/studies/<studyId>` | studies and one study (arms × seeds grid, compare) | `runs/studies.js` |

- `runs/studies.js`: `mountStudiesView(host, {studyId})` gives `{setStudy(id|null), destroy()}`;
  `openStudyDialog({yaml, sourceLabel, name})` resolves to the created study or `null`.
- `runs/analysis.js`: `openAnalysisDialog` also takes `kind: 'pack-dataset'` with
  `dataset: {gtRoots, lqRoots, scale, filenameTemplate}`.
- `actions.js`: `createStudy()` (also reachable from the "Add to queue" dialog), `studiesHref(id?)`.
- `api.js`: `studies`, `study`, `createStudy`, `studyCompare`, `importNeosr`.
- `state/data.js`: `options` (the extra scan request fields: several roots, name template, phase, crop
  size, source, manifest, task), so a folder check asks what the trainer will ask.
  `recipe.datasetFolders()` carries the matching facts of each dataset.

## Fourth round: the first training as one path, phases, audits, measured checkpoints (2026-11-20/21)

### The path of a first training

Recipe, then Data, then Runs. Each step hands over to the next in words, on the next-step line:

1. **Choose a recipe.** A template names example folders. When the only errors of a recipe are data paths
   that are not there, the next step is "choose your own data", with the error list as the second button.
2. **Choose and check its data** on the Data screen, which leads with the open recipe's datasets:
   "Change…" writes a chosen folder or file straight into the recipe, and the check that fits the task
   runs there (the paired folder check, or the data audit for classification, segmentation and detection).
3. **Look it over.** "Start training" and "Add to queue" open a dialog that shows what is about to train,
   read from the checked recipe (task, network, each dataset's places on disk, iterations, batch size) and
   what the last data check really covered. A folder check counts there only when the folders checked are
   this recipe's own.
4. **Start it**, or add it to the queue on hold.

### Words

- **Curriculum**: one recipe trained in ordered **phases**; said once where it is introduced as "a
  curriculum trains one recipe in ordered phases". Phase modes are "Start" (begins at iteration 0),
  "Continue" (restores the previous phase's weights, optimizer, scheduler and iteration count, so those
  settings cannot change; data and loss settings can) and "Fine-tune" (a new count at iteration 0 from the
  previous phase's final "Raw weights" or "EMA weights"). `updates` are called iterations and a phase's
  clock is written "iterations 4 to 8". Every phase is a queue entry saved on hold; a later phase can be
  released only when the one before it has completed, and the screen says what a phase waits for before
  any click. A changed phase is a new **revision**; earlier revisions stay listed.
- **Data audit** (`task-data`): the service lists every dataset of a classification, segmentation or
  detection recipe and opens files to check them. What was **counted in full** and what was **checked by
  opening files** (a sample, unless every file was asked for) are always kept apart, in words and in layout.
- **Weight drift**: how far the numbers of a candidate checkpoint moved from a baseline. It measures
  movement, never quality, and every view of it ends by saying so.
- **Evaluation**: one measurement of a checkpoint on the recipe's validation data, on the CPU. It is not
  part of a run's validation history.
- **Split leakage**: identical encoded files in both the training and the validation data. A partial check
  never reads as a clean one.
- **Best checkpoint**: two different answers that are never merged, "best score observed" and "best
  checkpoint still available"; the weights of the best score can be missing, damaged or not matchable.
- **Supported or not** is said at the place of use: an action that the service refuses for a task is left
  out with one plain sentence (ONNX export for object detection), and any other refusal is shown where the
  person clicked, in the service's own words.
- **Training samples** ("What the model received"): the first sample fed to the model for an update the
  recipe asked to capture (`logger.training_sample_steps`), saved after the batch was fed and before the
  optimizer step. Each tensor is shown with its shape, and an enlarged picture says by how much.
- **Pair alignment check** (`pair-registration`): on a sample of a recipe's pairs, how far the low quality
  image is shifted against its ground truth. A finding is a "possible shift", never a verdict, and the size
  of the sample is always said ("Sampled 10 of 240 pairs").
- **Across seeds**: under a study comparison, a description of the matched seeds (mean, median, lowest,
  highest, spread). It says itself that it is not a significance test, and lists what was left out and why.
- **Continues from**: a recipe that resumes a saved training state says where it continues, and only once
  the service has accepted that state; a refused state is in the issues, with no claim beside it.

### Routes, parts and shared state added

| Hash | Main area | Module |
| --- | --- | --- |
| `#/runs/curricula`, `#/runs/curricula/<curriculumId>` | curricula and one curriculum (phases, revisions, handoff) | `runs/curricula.js` |

- `runs/curricula.js` (`css/curricula.css`): `mountCurriculaView(host, {curriculumId})` gives
  `{setCurriculum(id|null), destroy()}`; `openCurriculumDialog({yaml, sourceLabel, name})` resolves to the
  created curriculum or `null`. The planner and the revise dialog check each phase's recipe with the
  service before saving, because creating a curriculum does not run the recipe check.
- Result renderers, each `render…(host, {job, result, artifacts})` returning a cleanup, loaded on demand by
  the analysis jobs view and reused where the job is started inline:
  `runs/drift.js` (`css/drift.css`) `renderWeightDrift`; `runs/taskdata.js` (`css/taskdata.css`)
  `renderTaskData`; `runs/holdout.js` (`css/holdout.css`) `renderEvaluation`, `renderSplitLeakage`.
- `runs/analysis.js`: kinds `weight-drift`, `task-data`, `evaluate`, `split-leakage` beside the earlier
  ones; detection prediction takes a lowest score and a box limit; a kind the module does not know still
  shows legibly (parameters, files, result in a collapsed block).
- `runs/detail.js`: the Files tab is "choose a checkpoint, then act" (one action bar for the selected
  file) with "Which checkpoint measured best?" above it; a run that is a curriculum phase says so.
- `views/data.js`: the pane leads with "Data for <recipe>": one entry per dataset in the task's words, each
  place with "Change…" (writes into the recipe at once and says so) and the check that fits it. Jobs started
  here (the data audit, the split leakage check) keep their options and a one-line outcome in the pane and
  draw their result in the main area, behind a switch beside the folder check. A result whose recipe data
  changed afterwards says so twice, in the pane and above the result itself, each with "Check again". The
  two-folder tool for folders outside the recipe sits under "Check other folders". One primary button per
  state: the training data's check for paired recipes, "Check the data" for audited tasks.
- `views/runs.js`: on a narrow window the parts stack their list above the open item. Opening an item then
  scrolls the part (and only the part: never `scrollIntoView`, which also moves clipped ancestors that
  nobody can scroll back) so the item is in view, and follows it briefly while list and item fill.
- `state/data.js`: `audit` (the last data audit: `{yaml, jobId, status, errors, warnings, error}`),
  `data.setAudit()`, `data.auditIsCurrent()` (an audit stays valid until the recipe's task, task options or
  datasets change), `AUDIT_TASKS`. The state follows a running audit by itself, so the rail stays true
  when the Data screen is closed.
- `actions.js`: `createCurriculum()` and `createStudy()` are offered inside the "Add to queue" dialog;
  `auditTaskData()`; `curriculaHref(id?)`.
- `api.js`: `curricula`, `curriculum`, `createCurriculum`, `reviseCurriculumPhase`, `analysisStartExact`
  (keeps a `null` in the body), `analysisEvaluations`, `analysisImages`, `checkpointSelection`.
- `recipe/schema.js`: `SERVICE_OPTIONS`, a one-line-per-option table for options the service checks but
  its schema does not list (today `checkpoint_rng` and `logger.training_sample_steps`; a dotted key is
  placed inside its section by `withNestedServiceOptions`). The schema's own definition wins once it
  exists. `props()` also reads an optional section (`anyOf [null, section]`), so such a section shows its
  options instead of "not in the schema".
- `runs/samples.js` (`css/samples.css`): `mountTrainingSamples(host, {runId, active})` gives
  `{setActive, setShown, destroy}`: the "Training samples" tab of a run, newest update first, older ones
  behind "Show more"; it reads again only while the run is active and the tab is shown.
- `runs/registration.js` (`css/registration.css`): `renderPairRegistration` and
  `pairRegistrationProgressText`, for the analysis kind `pair-registration`. The same job starts inline on
  the Data screen ("Measure alignment"), the third job there beside the audit and the split leakage check.
- `recipe/resume.js`: `resumeFacts`, `resumeSentence`, `renderResume` and `resumeRefused(issues, resume?)`
  for the check's `summary.resume`. The Recipe bar says "Continues from iteration N"; the dialog before
  training has a "Starts from" row with what the saved state holds.
- `runs/studies.js`: `renderComparison(answer, arms)` (the per-seed table; whole-dataset scores are named
  as such, never mixed silently with means of per-image scores) and `renderSeedSummary(answer, armName?)`.
- `runs/analysis.js`: prediction for frame sequences takes "Frames per window"; an autoencoder recipe reads
  "Autoencoder weights" and "reconstructed"; saved output images are offered only where the service makes
  them. A request keeps its id only while its body is unchanged, so a retry cannot be refused as a clash.
- `runs/curricula.js`: a planner or revise dialog cannot be closed while it saves, and a refusal lands in
  the open dialog under the phase it belongs to.
- `main.js`: the next-step line leaves out a button that only leads to the screen already open; a step may
  carry `hereText`, the same sentence naming the control on that screen instead.
- `actions.js`: the dialog before training shows "Data check" in one of four states (being checked, did not
  finish, not checked, checked; errors as a badge) and reads the run the service returns, so a run that
  failed at launch is reported as "Training did not start: …" and never as started.
- `api.js`: `trainingSamples`.
- `css/base.css`: `.view` clips with `overflow: clip`, so no focus move or scroll request can shift a
  screen that has no scrollbar to bring it back.

## Fifth round: evidence across runs, and a run that travels (2026-11-21)

### What a person can now do

- **Compare measurements, whoever made them.** The compare screen takes a training run or a CPU evaluation
  job on either side (`#/runs/compare/<left>[/<right>]`, a side being a run id or `job:<id>`). An evaluation
  job has one measurement; a run has one per measured iteration, and an iteration measured by more than one
  attempt lists each. Scores that the evaluator computes over a whole dataset (classification, segmentation,
  detection) are shown as such and never beside an image-by-image list they do not have.
- **Look at the classes.** The per-class report is one part (`runs/classes.js`) for a run's validation and
  for an evaluation job, so the two read alike.
- **Pick the checkpoint that measured best and is still there, and act on it.** The Files tab's ranking says
  how a file was tied to its score (by the save ID the measurement recorded, or through the log, the weaker
  evidence). A job started on such a file carries the file's version, so a file replaced since is refused
  by the service instead of being measured under another file's score.
- **Take a run elsewhere, or continue from it.** A **run file** describes a run (recipe, seed, how it
  started, where its inputs were) and holds no weights and no data. "Fine-tune from this…" on a checkpoint
  prepares a recipe that starts from those weights; "Save a run file…" on the run's Recipe tab writes the
  file; "Open a run file" in the recipe chooser maps its inputs to this machine, shows what differs and
  opens the result as a recipe. None of them saves, queues or starts anything.

### Rules added

- **What is offered comes from the service.** The recipe check answers with the actions a recipe's task and
  network support (`summary.actions`: predict, holdout, export, with the kinds of input each takes). The
  analysis dialog and the Files tab offer exactly those; a missing action is said in one sentence without a
  reason of our own. The fixed rules (no ONNX export for detection or for sequence networks) remain only
  for a service that does not report actions.
- **The everyday action comes first.** With a checkpoint chosen: try it, evaluate it, fine-tune from it,
  export it; the tools that look inside the file (structure, drift, path) follow, drawn quieter.
- **A document whose exact writing matters stays text.** A run file is checked by a digest of its content.
  Parsed and written again by a browser it can change (1000000.0 becomes 1000000), so it is read, saved and
  sent back as the text the service wrote; only a parsed copy is used for display (`api.js`,
  `reproductionExportText`, `reproductionPrepareText`).
- **"Not verified" never reads as fine.** In the run file's comparison with the original run, data and device
  are named as not verified with the service's own reason.
- **Unknown is not the same as free.** A later curriculum phase whose phase before it is still being read
  does not offer Release for that moment.
- **An id appears only where it tells two things apart.** The attempt is a column of the scores table, a
  line of the chart and a word in the ranking only when the rows shown come from more than one attempt; a
  measurement is numbered ("measurement 1 of 2") only where an iteration was measured more than once.
- **A long answer announces one sentence.** A check result is tables and lists; only its outcome sits in the
  live region ("6 problems must be solved first. The details are below."). When the answer lands below what
  a dialog shows, the dialog scrolls it to the top, so the outcome is never only a new button in the foot.
- **A refusal says the rule it applied.** Two sides with no dataset and metric in common are told that
  scores are matched by the name each recipe gives its dataset, with the names each side recorded.

### Parts

- `runs/runfile.js` (`css/runfile.css`): `openFineTuneDialog({run, artifact})` resolves `{yaml, name,
  label}` or null; `openRunFileExport({run, artifacts, preselect})`; `openRunFileImport()` resolves what the
  recipe chooser's other import returns (`{doc, label, saved: false}`) or null; `renderPrepared(answer)`
  draws the answer of a check. Enter in a one-line field sends the check; a field edited afterwards marks
  the answer as old and withdraws "Open as recipe".
- `runs/classes.js`: `mountClassReport(host, {evaluation, request, onClose, headingText, sort, retry,
  headingLevel})` gives `{destroy()}`; used by `runs/scores.js` (a run's validation) and `runs/holdout.js`
  (an evaluation job's "Results by class").
- `runs/holdout.js`: an evaluation job's report says what kind of score its table holds
  (`aggregationSentences`), lists "Results by class" where counts were kept, and leads to the compare
  screen ("Compare with its run" when the checkpoint's path names one run of the list, `runOfWeights`).
- `runs/evaluation.js`: sides parsed from the route token; "Training runs" and "Evaluation jobs" in both
  pickers; a job side has one measurement and no logged history; its recipe is the job's `evaluation.yml`.
- `runs/scores.js`: the Attempt column only when the rows shown come from more than one attempt.
- `runs/detail.js`: `fileActions({…, actions, finetune})`, `isPrimaryWeights(path)`; `mountBestCheckpoint`
  reports the files it tied to scores (`onArtifacts`), words the evidence and names an attempt only where an
  iteration was measured by more than one.
- `runs/analysis.js`: `openAnalysisDialog({…, weightsVersion})`; gates read `summary.actions`.
- `api.js`: `reproductionExport`, `reproductionPrepare`, their text-keeping forms, `analysisValidation`,
  `analysisClasses`; `compare` takes `{job: id}` sides and evaluation ids.

## Sixth round: preparing data, and reading what a job left behind (2026-11-21)

### What a person can now do

- **Make a validation split that can be made again.** "Make a validation split…" on the Data screen (and
  the kind "Validation split" under "New analysis job") starts the job `split-dataset` for a restoration
  recipe whose training data are paired folders: a share of the pairs, what stays on the same side (each
  pair, a folder, a top-level folder) and a seed. Nothing is copied or moved. The result (`runs/split.js`)
  says how many pairs went where and how the achieved share compares with the one asked for, gives the two
  lists (download, path on the trainer machine, checksum, the names themselves) and the record of the
  split, shows the recipe lines that use the lists, and offers "Open the recipe with this split", which
  opens an unsaved recipe and saves or starts nothing.
- **Cut the data into training patches.** "Extract patches…" on the Data screen (kind "Patch extraction")
  starts the job `extract-patches` for a paired dataset kept as folders: which data, the patch size and the
  step, both in low quality pixels, with the ground truth size shown as the person types. The result
  (`runs/patches.js`) says how many patch pairs came from how many source pairs, where the folders, the
  list and the record of every patch's origin are, shows the recipe lines, and offers "Open the recipe with
  these patches", which changes only the dataset's own options. Where the recipe crops more than a patch
  holds (`lq_size`), it says so before and after.
- **Convert a .pth checkpoint.** The kind "Weights conversion" (`convert-weights`) saves the weights of a
  PyTorch .pth file as a safetensors file. The result (`runs/convert.js`) names the dictionary that was
  read, the number types and what was kept with the file, and gives the file to download.
- **Read a pair alignment check.** The result opens with its outcome in one sentence ("3 of the 10 measured
  pairs may be shifted: 2 in brightness and 1 in colour only."), names each shift in words ("2.00 px left,
  2.00 px up"; the raw x and y stay in the title), explains under "How to read the table" what a shift, the
  brightness plane, the two colour planes, the match and the patches are, and says where one flagged pair
  can be looked at and measured on its own.
- **Measure the difference between two saved images.** "Measure the difference…" under the comparison stage
  of a run's Validation tab hands the two images on the stage to the job `image-difference` (and the ground
  truth as the reference when two outputs are compared); the kind is also under "New analysis job" for any
  two PNG files. The result (`runs/difference.js`) says how many pixels differ and by how much per channel,
  shows the saved difference image enlarged, and with a reference says which image is closer at how many
  pixels.
- **Keep what an evaluation decided.** The evaluation dialog offers, under "Also keep", the outputs the
  recipe check names for the recipe (`summary.actions.holdout.optionalExports`): the output images or
  masks, the detected boxes, each image's predicted class. A classifier's kept decisions are a file to
  download and a table in the result ("Decisions per image": true class, predicted class, probability, the
  other candidates; "Only wrong decisions").
- **See which scores a run has.** A run without a recorded validation summary no longer reads as a run
  without scores: the Scores view lists the averages the trainer printed in its log, says whether scores
  per image were kept (and leads to the compare screen for them), and says what only a recorded summary
  has.

### Rules added

- **A score is named by where it was read.** The same number can come from the training log (one average
  per validation pass), from the per-image records (their mean) or from the recorded summary. Each table
  says which, and the three are never merged into one unnamed column.
- **The outcome first, the measurement after.** A result opens with the sentence a person would say to a
  colleague; counts, tables and the job's own words follow; everything else the job reported sits in a
  closed block at the end.
- **A direction is a word.** Left, right, up, down, with the unit; a signed pair of numbers stays in the
  title for the person who needs it.
- **A picture must not contradict its numbers.** A difference image whose small differences would be
  invisible is brightened, or shown as where the images differ at all, with the factor or the rule stated;
  the file as saved is one press away.
- **An action that needs a pinned file is offered only where the service can pin it.** A saved image goes
  to a job with the version its link carries; where a gallery carries none, the button is not shown, and
  the two files can still be chosen by path in the dialog.
- **Optional outputs come from the service's list.** A checkbox per name the recipe check gives; an unknown
  name is shown as it is; no list at all falls back to the one fixed rule (output images for restoration).
- **A fragment is offered as an action only for the recipe it belongs to.** The split's recipe lines can
  always be copied; "Open the recipe with this split" appears when the open recipe trains on the folders
  the split was made from, asks before replacing a validation dataset, and gives way to "already uses this
  split" once it does.
- **A field's error leaves when the field is edited.** The next "Start job" checks it again.
- **A question is asked only when its answer matters.** `openAsRecipe` asks before discarding unsaved
  work. A caller that built the new text from the open recipe's current text passes `derived: true` (after
  checking that the text did not change meanwhile): nothing is discarded, so nothing is asked.
- **A long choice is grouped, and described once.** The kind picker lists the kinds by what they work on
  (a checkpoint, a recipe's data, saved images), names only; the chosen kind's description stands under
  the list, and each radio still carries its own for a screen reader. The form's fields stay in view.
- **The order on the Data screen is the order of the work.** Identical files, the validation split, the
  patches, the alignment of a sample; the patches line says to split first.

### Parts

- `runs/split.js` (`css/split.css`): `renderSplitDataset(host, {job, result, artifacts})` and
  `splitDatasetProgressText(progress)`.
- `runs/difference.js` (`css/difference.css`): `renderImageDifference(host, {job, result, artifacts})` and
  `imageDifferenceProgressText(progress)`.
- `runs/patches.js` (`css/patches.css`): `renderExtractPatches(host, {job, result, artifacts})` and
  `extractPatchesProgressText(progress)`; it reuses the list, path and YAML helpers that `runs/split.js`
  exports.
- `runs/convert.js` (`css/convert.css`): `renderConvertWeights(host, {job, result, artifacts})` and
  `convertWeightsProgressText(progress)`.
- `runs/analysis.js`: kinds `split-dataset`, `extract-patches`, `image-difference` and `convert-weights`
  (every kind the service accepts now has a form and a result part); `KIND_GROUPS`;
  `openAnalysisDialog({…, left, right, reference})`, each a path or a saved image `{run | job | analysis:
  id, path, version, label}` (the label is shown, never sent); "Also keep" from `optionalExports`; a
  missing action is said of the recipe ("with the options the recipe sets"), since the same network can be
  offered with other options.
- `actions.js`: `openAsRecipe({…, derived})`.
- `runs/holdout.js`: "Decisions per image" from `classificationResults` (`parseDecisions`,
  `decisionSummary`).
- `runs/registration.js`: `conclusion`, `shiftWords`, the legend.
- `runs/scores.js`: the view for a run without a recorded summary (`loadOther`, `renderOther`).
- `runs/validation.js`: "Measure the difference…" (`measurePlan`).
- `views/data.js`: "A validation split" (`renderSplit`) and "Patches for training" (`renderPatches`).

## Seventh round: from a finished run to a file that ships, by keyboard too (2026-11-21)

### What a person can now do

- **Split labelled data.** Class folders, images with masks and COCO annotations get a validation split
  like paired folders do, worded in images instead of pairs. For these the service writes the whole recipe:
  the result page offers "Open the generated recipe" directly under the counts, then the lists.
- **See whether the checkpoints will fit.** The Files tab of a run ends with "Disk space for checkpoints":
  one sentence (what is on disk, what is still to come, what is free), the service's working in a closed
  block. While a run trains, a shortage is also said above the tabs with a button that leads to the detail.
- **Weigh two scores.** Under "Which checkpoint measured best?", a closed block lists the checkpoints that
  no other checkpoint beats on both of two chosen scores, with "Choose" for a file that is still there.
  A score the service cannot orient gets "Higher is better / Lower is better", as the ranking does.
- **Export what the network allows.** The export form shows "Accept other image sizes" only where the
  recipe check says the graph can take them, and "Frames per window" only for a network that reads
  sequences. An action the service does not offer is explained in the service's own sentence.
- **Go on from a converted file.** A weights conversion gives the saved file's path, "Try on an image…" and
  "Export to ONNX…"; every file of a job that has a path on the trainer machine has "Copy path".
- **Read the trainer's lists again.** The trainer environment dialog says where the choices of the recipe
  form come from (the lists that came with the trainer, or this installation and when), names entries that
  could not be read completely, and reads them again after a network or loss was added to the trainer.
- **Work without a mouse.** Choosing a recipe and opening a run hand the focus to what was opened; the skip
  link keeps the address; a button that leads to another screen leaves the focus on that screen.

### Rules added

- **What the service can do is asked, never remembered.** No rule of the kind "this task cannot export"
  lives in the frontend: an action is offered when the recipe check lists it (`summary.actions`), a
  missing one carries the service's reason (`summary.unavailableActions`), and before the answer arrives
  nothing is held back; the start request stays the authority.
- **Focus goes where the person went.** A list that disappears on choosing hands its focus to the thing
  chosen: the recipe's title bar, the run's name. A screen replaced from a control inside it (or by a
  button that was disabled while it worked) puts the focus on the new screen's main region. Focus in the
  rail stays in the rail, and the first screen moves nothing.
- **A link inside the page must not be a navigation.** The address after `#` is the route; anything that
  only moves the focus (the skip link) does so in script and leaves the address alone.
- **The next step stands before its evidence.** On a result page the one primary button sits directly under
  the outcome sentence; lists, records and recipe lines follow.
- **A warning appears only when there is something to act on.** The disk forecast is one calm sentence
  when the saves fit and a warning only when they do not; above the tabs nothing is shown unless the run
  is training and short of room. The forecast is read when a training run is opened and then at most
  every five minutes, never while the page is hidden.
- **A control that works for a while keeps its focus.** It is marked `aria-disabled` and ignores a second
  press instead of being disabled, and says in a status line beside it how long the work can take.
- **Lists that change bump a revision.** `catalog.replace(answer)` raises `revision`; a form built on the
  lists rebuilds itself when the number changes, and the open recipe is checked again. A rebuild that is
  owed never happens under the person's cursor: it waits until the focus has left the screen.
- **Work started in a dialog outlives the dialog.** A request that can take minutes belongs to the page,
  not to the dialog that started it: there is one at a time, a dialog opened meanwhile follows the same
  request, and an outcome that arrives with no dialog open is said in a toast.

### Parts

- `runs/storage.js` (`css/storage.css`): `mountCheckpointStorage(host, {runId, getRun, request})` →
  `{update(run), destroy()}`; `storageVerdict(report)` → `{kind: 'fits'|'tight'|'full'|'done'|'unknown',
  sentence, help}`, also used by the run header's warning (`watchDisk` in `runs/detail.js`).
- `runs/frontier.js` (`css/frontier.css`): `mountCheckpointFrontier(host, {runId, hasFile, onPick,
  selection, request})` → `{refresh(), destroy()}`; nothing is read until its block is opened.
- `runs/detail.js`: the Files tab mounts both (`loadFrontier`, `loadStorage`, each left out if it fails to
  load); `fileActions` holds no fixed rule; the note above the list gives the service's reason per action.
- `runs/analysis.js`: `dynamicSpatial` and `clipSize` on export; `split-dataset` for the four tasks the
  service splits (`SPLIT_TYPES`), grouping by `sample | parent | scene` for labelled data; "Copy path" from
  `absolutePath`; a part's progress words receive the job as a second argument.
- `runs/split.js`: words by task, lists matched by file name (text, JSON lines, COCO), "Images per class",
  the generated recipe (`recipePath`) opened through `openAsRecipe` without `derived`, since it replaces
  the open recipe.
- `runs/convert.js`: the saved file's path and the two buttons that start a job on it.
- `components/environment.js`: `catalogCounts`, `catalogSourceText`, "Read the lists again"
  (`api.catalogRefresh()`); `state/catalog.js`: `replace`, `revision`.
- `api.js`: `checkpointStorage`, `checkpointFrontier`, `catalogRefresh`.
- `views/recipe.js`, `views/runs.js`, `main.js`: the focus handovers and the skip link.
- `views/data.js`: "A validation split" for the four tasks (`SPLIT_TYPES`).

## Eighth round: tools that answer a question before training does (2026-11-21)

### What a person can now do

- **Average checkpoints.** On a run's Files tab, a numbered checkpoint that has at least one sibling of
  the same network gets "Average with others…". The dialog lists that network's checkpoints of the run
  with the chosen one ticked; at least two must be ticked. The result page names the averaged file and its
  path, and offers "Try on an image…", "Evaluate…" and "Export to ONNX…", each opening the job dialog
  with the run's own recipe and the averaged file.
- **List a network's parameters.** The Network section of the recipe form has "List this network's
  parameters": every named tensor with shape, size and number type, what the optimizer filter selects, a
  filter by name, and "Copy name" for `train.optim_include_params_g` and `optim_exclude_params_g`. For a
  recipe with a weight transfer, one more box shows what the transfer copies, skips and leaves fresh, and
  warns when fresh values are left out of training.
- **Read a loss report.** A loss report has a page of its own: each term's weighted loss, gradient size,
  share and direction, whether the terms measured one by one add up to the total, and the adversarial
  term against all the others.
- **Measure flicker.** "Also keep" in the evaluation form offers "Frame-to-frame errors" where the recipe
  check lists it. The result has "Change between frames" per dataset: how many changes were scored, what
  was left out and why, the largest error, and the scored changes on demand. Detection evaluations offer
  "Operating points": the best F1 per category with the confidence threshold that gives it.
- **Split video by scene.** A paired-video recipe gets the validation split too: whole scenes go to one
  side, a file used by more than one scene is named, and the result is two scene lists and the recipe
  that uses them.

### Rules added

- **A tool lives where its question comes up.** Averaging starts from the checkpoint (Files tab), the
  parameter list from the network (recipe form), the loss report from the losses. The bare "New analysis
  job" picker offers only what can be started without that context: a kind that needs a run's files is
  never listed there, and opened from its place it is a dialog of its own, without the picker.
- **A started job takes the focus.** When a dialog starts a job, the heading of the job's page receives
  the focus once. The button that opened the dialog is often on a page the job page replaced.
- **A preview that costs nothing is not held back by an unfinished recipe.** The parameter list needs
  only the network section, so it starts without a passing recipe check; the service stays the authority.
- **Optional parts are offered when listed and drawn when present.** An extra export shows only where the
  recipe check lists it, and its result section only when the result carries it. A dataset without
  scores says so in one quiet line; the "no scores" warning is kept for a result with nothing in it.
- **A verdict only at the clear ends.** Where a page reads a number for the person (do the terms add up,
  are fresh values left out of training) it gives a sentence only when the number is unambiguous, and
  otherwise states the figures.
- **Long lists come on demand and stay bounded.** The first rows come with the result; "Show all" reads
  the list file once, the filter works on what was read, focus stays where it was while rows arrive, and
  above a stated bound the page points to the download instead.

### Parts

- `runs/average.js` (`css/average.css`): `renderAverageCheckpoints`, `averageCheckpointsProgressText`;
  one `api.run` for the run's name and recipe; the three buttons call `openAnalysisDialog({kind, yaml,
  weights, weightsVersion, sourceLabel})`.
- `runs/parameters.js` (`css/parameters.css`): `renderNetworkParameters`; the list file and the transfer
  receipt are read only when asked for.
- `runs/lossreport.js` (`css/lossreport.css`): `renderLossReport`; no requests.
- `runs/holdout.js`: "Change between frames" (`temporalSummary`, `temporalResults`), "Operating points"
  (`detectionOperatingSummary`, `detectionOperatingPoints`), "Everything else the job reported".
- `runs/split.js`: the scene split (`counts.windows`, `physicalIdentity`, scene lists as JSON,
  `video_scene_manifest`).
- `runs/analysis.js`: kinds `average-checkpoints` (opened with `average: {run: {id, name}, role,
  checkpoints: [{step, path, version}], selected}`) and `network-parameters` (`inspectTransfer`); the
  group "With a recipe's network"; `saveTemporalErrors` and `saveDetectionOperatingPoints` among the
  optional exports; `pairedvideodataset` in `SPLIT_TYPES` (no `groupBy` is sent); `startedJobId` for the
  focus handover.
- `runs/detail.js`: `averageCandidates(path, paths)` and the `average` file action.
- `recipe/form.js`: `listParameters()` in the Network section; `openRecipeJob(kind, what)`, shared with
  "Measure these losses".
- `views/data.js`: "A validation split" worded in scenes for paired video.

## Ninth round: the weights that scored best, and where the time went (2026-11-21)

### What a person can now do

- **Keep the weights that scored best.** The Validation section of the recipe form has "Keep the best
  weights by": a choice among the recipe's own validation metrics, or "Do not keep them". Classification,
  segmentation and detection recipes get the same row as a text field, because their metric names are
  not listed anywhere the form can read. In a run's checkpoint ranking, a point whose weights were kept
  names the file and says it holds exactly what was scored, also between two numbered saves. Picked in
  the Files tab, such a file offers the same actions as a numbered checkpoint, "Fine-tune from this…"
  included.
- **Start a phase from the best weights.** A fine-tune phase of a curriculum has a third choice next to
  raw and EMA weights: "The best-scoring weights on a validation dataset", with the dataset's name. The
  queue shows the choice on the entry.
- **Average a network that uses batch normalisation.** The averaging dialog has "Take batch-norm
  statistics from the latest chosen checkpoint", off by default. The result states how many tensors were
  taken from which iteration instead of averaged. A refused average whose reason is a differing buffer
  gets one line pointing to the option. The checkpoints listed are the ones the service names for that
  network; when it has more than the dialog lists, the dialog says so.
- **See how fast a run trained.** Under the charts of the Telemetry tab, "How fast this run trained":
  the logged rate, the rate counting all the time since the first iteration, the share of time spent
  waiting for data, and peak memory, each with its range and the log lines it comes from. Another run
  can be chosen to compare the rate at the iterations both logged.

### Rules added

- **The service's list wins; the page's own list is the fallback.** Where the service can name the
  candidates (checkpoints to average), the dialog uses that answer. A service without the route gets the
  list the page already had, and nothing on the page says which of the two was used.
- **Counts say what they count.** A transfer is worded from the network's side ("N of the network's M
  tensors are copied; K keep their starting values"). Names left out are called "names from the weights
  file or the network", because the service counts both. A shortened list states the total it was cut
  from.
- **A reading comes with its basis, a verdict only at the clear ends.** Timing sentences state median,
  range, the number of log lines and the iterations covered. Advice appears only when more than half of
  the time went to waiting for data; a small share gets one quiet line; everything between gets the
  figures alone.
- **A comparison says what was compared.** Two runs are compared only at the iterations both logged, and
  the sentence states how many those are.
- **An option the form cannot check is explained, not guessed.** The curriculum planner cannot see
  whether the earlier phase keeps its best weights or which validation datasets it has, so it asks for
  the name, says what the earlier phase's recipe needs, and shows the service's refusal as written.

### Parts

- `runs/timing.js` (`css/timing.css`): `mountRunTiming(host, {runId, runs})` gives `{refresh, destroy}`;
  nothing is read before the block is opened; `api.runTiming(id, {against})`.
- `runs/detail.js`: the timing block in the Telemetry part; `average()` asks `api.averageCandidates`
  first; `EVIDENCE_WORDS.retainedValidationWeights`; `PRIMARY_WEIGHTS_RE` also matches `best-…` files.
- `runs/analysis.js`: `batchNorm: 'fromLatest'` on `average-checkpoints`, `average.more`, the hint under
  a refused average.
- `runs/average.js`: the fact "Batch-norm statistics" (`copiedBatchNormTensors`, `batchNormSourceStep`).
- `runs/parameters.js`: totals from `tensorCount`; the transfer sentence and "Names left out of the
  copy".
- `runs/split.js`: `generatedKeysText(recipeDatasetKeys)`: which datasets of the recipe were written and
  that the others are kept.
- `recipe/form.js`: the `val.save_best` row (`bestChoices()`, `bestNote()`; see "A value with known
  choices").
- `runs/curricula.js`: `weights: {bestValidation: name}` on a fine-tune phase; `runs/queue.js` shows it.
- `api.js`: `averageCandidates(id, role, params)`, `runTiming(id, params)`.

## Tenth round: a last pass as a newcomer with a labelled task (2026-11-21)

### What a person can now do

- **Measure on the GPU.** The evaluation dialog has "Where to measure": "On the CPU" (chosen every time
  the dialog opens) or "On the GPU". The dialog's two sentences about where the job runs follow the
  choice. A refusal because the GPU is in use keeps the service's sentence and adds what to do. The job
  page says "Measured on", and the result names the card when the service does.
- **See how an evaluation was tiled.** The result's lead says the tile size, overlap, tile mode and where
  the tiles were put together, when the result states them.
- **Choose the best-weights metric for any task.** "Keep the best weights by" lists the metrics the
  recipe check names for this recipe, each with its direction, for classification, segmentation and
  detection too. Under it: which validation datasets best weights are kept for, or a warning that none
  is scored.
- **Know before saving whether a phase can start from best weights.** The curriculum planner asks the
  service what the earlier phase keeps and says it under the dataset name: that it keeps none, that no
  dataset is scored, or by which metric for which datasets (offered as suggestions for the name).
- **Resume with another number of data loader workers.** Next to "Resume", "Resume with other workers…"
  appears only for a run the service marks as able to; it asks for the number and shows the run's own.

### Rules added

- **A summary shows only facts of the task.** A scale is a fact of a recipe that upscales: template
  rows, the template's facts and the recipe header do not show one for classification, segmentation and
  detection, like the next-step line already did.
- **A sentence about where something runs follows the choice.** No text in a dialog may contradict the
  option chosen in it.
- **The costlier resource is never the default.** The GPU is chosen by the person each time; nothing
  remembers it.
- **A button that disables itself gives the focus back.** A button that starts work and is disabled while
  it runs (the data check, Stop, Resume) gets the focus again when the answer is in, unless the person
  moved on or a dialog is open.
- **An option appears only where the service says it applies.** The worker choice needs
  `resumeWorkerOverride: true`; a missing preview route or field leaves the page as it was, silently.

### Parts

- `runs/analysis.js`: `device: 'cuda'` on `evaluate` only when chosen; `deviceWords`, `gpuBusyHint`;
  `syncDevice()` (lead, note, training warning).
- `runs/holdout.js`: `measuredHowText(device, precision, {deviceName, deviceIndex})`, `tilesText(tiles)`.
- `recipe/form.js`: `snapshotMetrics()` from `summary.snapshotMetrics`; the kept line from
  `summary.bestValidationDatasets`.
- `runs/curricula.js`: `readPhasePreview`, `phasePreviewWords`, `syncPreviews` on
  `api.curriculumPhasePreview({baseYaml, edits})` of the earlier phase.
- `runs/detail.js`: `resumeWithWorkers()`; `act(kind, request, done, failed, opener)` restores focus.
- `views/data.js`: `setPhase` restores focus to the button that started the check.
- `recipe/library.js`, `views/recipe.js`: `AUDIT_TASKS` hides the scale; `recipe/sections.js`: Basics lede.
- `api.js`: `curriculumPhasePreview(body)`, `resumeRun(id, {trainWorkers})`; `state/runs.js`:
  `resume(id, params)`.

### A task without images (same day, last change)

A table task (`tabular_regression`: CSV files of numbers, columns named by the task options `features`
and `targets`) is worded without images wherever the page knows the task:

- `state/catalog.js`: `FOLDER_WORDS.tabular_regression` has `records: true` and `file: {key: 'meta_info',
  name: 'CSV file'}`. A part that reads `folderWords(task).records` must not ask for, list or show a
  folder or an image.
- Data screen (`views/data.js`): each dataset shows its CSV file with "Change…" and one line saying where
  the columns are named; "Check other folders" and the image list are not offered.
- `state/data.js`: `NO_SCALE_TASKS` (the labelled tasks and the table task) hides the scale in the
  template list, the recipe header and the next-step line.
- `recipe/form.js`: the Datasets lede; `recipe/fields.js`: the `meta_info` picker also lists `.csv`.
- `runs/analysis.js`: where the recipe check says an action takes `records`, a prediction asks for an
  "Input records file" (no folder), and an ONNX export asks for no height or width and sends none.
- `runs/holdout.js`: the lead says when the model's output is grayscale or colour planes only
  (`outputPixelFormat` `y` or `uv`); ordinary colour output is not mentioned.

## A value with known choices (2026-11-21)

Wherever an editable value has choices the page knows, it is one control: a text field with its choices
in a list under it (`components/combobox.js`, `css/combobox.css`). No `<datalist>` anywhere. A `<select>`
stays only for choosing what to look at or act on (a section, an arm, a dataset, a checkpoint, a
folder), where typing another name could never work.

### Rules

- **Two kinds of list, said by the control.** *Open* (`closed: false`): the list helps, any typed spelling
  is kept exactly as typed, and the list says so (`Enter keeps "x" as typed.`). A listed spelling writes
  the listed value. *Closed* (`closed: true`): only a listed value is taken; other text is refused in
  place, before the service is asked, in the same words everywhere.
- **Which is which follows the service, never the page.** A schema `enum` is closed, because the trainer
  refuses anything else. Everything else is open: the `examples` a schema gives for a text (an
  activation, a storage backend), a catalog list (tasks; network, loss and dataset types; optimizers,
  schedulers, metrics) and the `choices` of an optimizer's or metric's parameter. The installed trainer
  lists what it knows, a class of the person's own may still exist, and the service's check has the last
  word. A value that matches no listed name gets one quiet line under the field (`Not one of the 10
  optimizers the installed trainer lists.`); it is not an error of the page.
- **A type the page cannot describe loses nothing.** Changing a network or a loss to a type the schema
  does not describe writes the `type` only: every option stays, is shown as it is in the recipe, and "Add
  option" adds more. The prompt that names the options to be removed appears only for a known type that
  does not have them.
- **Names are matched without regard to case, and written as chosen or typed.** A listed name is written
  in the list's spelling; for an optimizer that is its display name (`AdamW`, `SOAP`), which is also what
  the list shows. A recipe's `adamw` is marked as the value in use and nothing is rewritten until the
  person picks. A spelling the list does not have is kept exactly.
- **Leaving an option out can be a choice.** Where an absent option means something (the task: "Decided
  from the network"; best weights: "Do not keep them"), that sentence is the first row of the list and the
  field's placeholder. Picking it, or emptying the field, takes the key out of the recipe; the sentence is
  never written as a value. A prompt that guarded the old control still guards this one (leaving the task
  to the network when the recipe has task options).
- **An empty text is a value.** Where the trainer lists the empty text, the list shows it as `""` with the
  note "empty text"; picking it, or typing `""`, writes an empty string, and the field shows `""`.
- **A list of values is edited item by item.** One row per item, in order, each a field with the same
  choices; a box under them adds one, the arrows move one, "Remove" takes one out. Changing one item
  writes that item only, so the other lines, their comments and a list paired with this one by position
  (weights, probabilities) are untouched. Repeats are kept. Text stays text: a listed `601` is written as
  a quoted string, never as a number. Items of an `enum` are closed; items with `examples` are open.
  Where the row is too narrow for an item's three buttons, they go under its field together
  (`.fld__itemacts`); nothing is cut off and the pane never scrolls sideways.
- **Typing keeps its type.** A number-or-word field (`auto` or a count) lists the words and still reads a
  typed number as a number. Booleans stay switches.
- **Keys.** Arrow keys open the list and move in it; typing narrows it; Enter takes the highlighted
  choice, else the typed text; the first Escape closes the list only (a dialog around it stays open), the
  second puts the field's value back; Tab leaves and commits like any text field. A click in the field or
  on its arrow opens the list. The list never takes the focus.
- **An empty list never opens by itself.** Typing in a field whose choices have not arrived is plain
  typing; opened on request, the list says that nothing is listed.
- **A new row gets the focus.** After "Add option", "Add a loss" or "Add discriminator" the section is
  redrawn; focus goes to the row that was added, not to the page. For a block that was added, that is its
  `type` row.
- **A free-form block can be taken out whole.** A block the recipe has and the schema does not lay out
  (per-optimizer schedules, a discriminator) ends with "Remove <name>", which asks first.
- **Lists and fields change together.** When the environment dialog finds that the lists were read again
  elsewhere, the cached schema is dropped before the new lists are applied, so a new name never meets the
  old fields.
- **A block's options follow its type.** An optimizer, a schedule (the shared one and each optimizer's
  own) and a metric get the same two things from the catalog entry of their `type`: "Add option" lists the
  parameters, and an option whose values are listed gets them on its row. Changing the type draws the
  section again, so no row keeps the values of the type it had.
- **A schedule made of schedules is edited schedule by schedule.** Where the catalog says a type holds
  other schedules (some entry names it in `allowedParents`), its `schedulers` list is one block per
  schedule, in order (`schedulers[0]`, …): the type with the types the catalog allows there, the options,
  "Add option", and a list of its own where that type holds schedules too, to any depth. A type of one's
  own can be typed; a listed type the catalog does not allow there gets one quiet line. A change inside a
  schedule writes that key only, so the other schedules, the milestones and keys nobody listed stay as
  they are. "Add schedule" and the move buttons write the list as a whole; removing asks first and takes
  out that entry only. Focus follows the schedule that moved. An entry that is not a mapping, and a list
  under a type the catalog does not know as holding schedules, stay one row with their text.
- **A service that says less is filled in from the schema.** A service that was started before it listed
  something still serves the current schema. Where its answer has no entry for a list, or names a type
  without its parameters, the form reads the schema's own copy (`x-trainner-catalog`); dataset types come
  from the `examples` of the dataset's `type`. What the service does say always wins.

### Parts

- `components/combobox.js`: `createCombobox({id, getChoices, closed, onPick, onEnter, onEscape, current,
  listLabel, attrs})` gives `{el, input, button, open(), close()}`; `toChoices`, `matchChoice`,
  `filterChoices`, `EMPTY_TEXT`. A choice is `{value, text, note?}` or `{unset: true, text, note?}`.
  Choices are read each time the list opens, so a list that arrives late needs no redraw. The list is
  fixed to the viewport and placed by script (under the field, or above it near the bottom edge), so
  neither a scrolling pane nor a dialog clips it.
- `recipe/schema.js`: `examples` on a text (also inside a nullable one) become the kind's `suggestions`;
  on the items of an array, `itemSuggestions`.
- `recipe/fields.js`: `comboControl` (schema enums, texts with suggestions, number-or-word fields, and
  any row given `choices: {getChoices, listLabel?, unsetText?}`), `listControl` (a list of values, item
  by item), `typeField` (network type), `addOptionRow(path, ctx, known, setKeys)` (the option names a
  block takes, each with the value it starts from, and the values that option lists).
- `recipe/form.js`: `knownList(path)` maps a `type` row to its catalog list (`train.optim_*`,
  `train.scheduler`, `train.schedulers.*`, `val.metrics.*`, `network_*`); `knownRow` adds a parameter's
  `choices` to its row; `blockListName(path)` says which list describes a block (`train.optim_*`,
  `val.metrics.*`, and every schedule: `isSchedulePath(path)` is true for `train.scheduler`,
  `train.schedulers.*` and any `….schedulers[i]` under them) and `knownOptionsAt(path)` gives "Add option"
  that entry's parameters; `scheduleSpecial` and `scheduleParts(listPath, list, exists)` draw the schedules
  inside a schedule, through the `special` hook of `genericEditor` (`recipe/fields.js`); `partsOf(entry)`
  reads `allowedParents`; `focusAfter.marks` names the button (`data-mark`) that gets the focus back; `blockTypes(c)` is part of the section signature; `listedIn(name)` reads a list
  from the service, else `packagedList(name)` from the schema; `taskRow()`; `bestChoices()`; the dataset
  type row in `fieldExtras`; the loss picker; `ctx.add` (focus after an addition).
- `components/environment.js`: `loadLists()` calls `api.schema.reset()` before `applyLists()`.
- `state/catalog.js` keeps `optimizers` (with `displayName` and `parameters[].choices`), `schedulers`
  (with `parameters[].choices`), `datasets`, `metrics` beside the earlier lists.
- `ui.js`: `promptDialog({choices})`. `runs/curricula.js`: option paths of the base recipe, validation
  dataset names of the previous phase. `runs/studies.js`: dataset and metric names the runs recorded.

## The phone edition (2026-11-21)

`mobile.html` is the same application for a phone held in one hand, reached from outside the house. It is
not a second app: it loads every stylesheet and module `index.html` loads, so each recipe, data and run
feature and each live action is the one the desktop page has. What differs is the page around them.

### Rules

- **One application, two pages.** Nothing is copied. `mobile.html` adds `css/mobile.css`,
  `css/mobile-screens.css`, `js/mobile-boot.js` and `js/mobile.js`, and marks the document
  `<html data-edition="mobile">`. `index.html` loads none of the four. A module that must behave
  differently reads that mark, says why in a comment, and leaves the desktop path as it was; a new screen
  or part needs a look at 430 px wide before it ships, and usually a few lines in `mobile-screens.css`.
- **Its own preferences.** `store.js` keeps this edition's preferences and its recipe draft under their own
  prefix, so looking at the phone page on the desktop machine changes nothing on the desktop page.
- **The shell.** A slim bar at the top (the name, or a link back; the connection; the theme), the screen,
  the next step, then the three stations as a bar at the bottom with their live state. `js/main.js` fills
  the same three elements as on the desktop; `mobile.css` only places them. Held sideways, the stations
  become a rail at the left and the next step one line in the top bar.
- **One column, one thing that scrolls.** No screen shows two panes side by side. Where the desktop has a
  list beside an open item, the phone has the list as a screen and the item as the next one:
  Runs → a run; Runs → Queue, Studies, Curricula, Analysis jobs → one entry; the recipe chooser's list →
  one recipe. `js/mobile.js` says where the person is with `data-screen`, `data-level` (`home`, `part`,
  `item`) and `data-part` on `<html>`, from the address alone, and shows the link back to the level above.
  The chooser keeps its own `data-pane` (`list`, `inspect`), because it also lives in a sheet.
- **Deeper, the next step steps back.** On a screen's first level the next step has its whole sentence and
  both actions across the width. One level down it is one line with its first action.
- **A run is one page.** Its facts, its tabs, the open tab. The tabs stay at the top edge; tapping one
  brings them there. Telemetry draws the chart first and lists the series to tick under it. A log starts
  with its lines wrapped.
- **Sizes for a finger.** 16 px text, controls 44 px high (38 px for the small ones), and a row that does
  one thing is hit anywhere on it. Nothing depends on hover.
- **Clear of the hardware.** The layout reads the notch and home-bar margins only through `--safe-t`,
  `--safe-r`, `--safe-b`, `--safe-l`. While a field is typed in, the two bars at the bottom step aside and
  the shell (and a sheet) take the height the keyboard leaves (`--vvh`, `--kb`).
- **Dialogs are sheets.** Every dialog rises from the bottom edge at full width, its buttons across the
  foot, clear of the home bar.
- **The comparison stage answers fingers.** One finger moves the image, two pinch it, the wipe line drags.
  (Pinch is in `components/compare.js`, on the phone edition only.)
- **Starting training asks first.** On the phone, Resume, Release and Continue the queue ask before they
  act (`confirmOnPhone` in `ui.js`), as Start training already does everywhere. The desktop page acts at
  once, as before.
- **Type.** Titles and station names use the display face; text uses `--font-ui`, which names a regular
  system face where the display face has only a bold cut.

### Parts

- `mobile.html`: the page. `js/mobile-boot.js`: the theme before first paint, from this edition's own
  preference, else the phone's setting.
- `js/mobile.js`: `placeOf(hash)` → `{screen, level, part, back}`; the back link (`.mback`); the run's tabs
  to the top edge; holding a ticked series row in place when a chart appears above it; the options form
  back to its top when the section changes; the keyboard (`data-keyboard`, `--vvh`, `--kb`); `--view-h`
  (the height of the screen area, for a part that fills it); the colour of the browser's bars. It ends by
  importing `js/main.js`, unchanged.
- `css/mobile.css`: sizes and faces, the shell, upright and sideways, controls, tabs, sheets, toasts.
  `css/mobile-screens.css`: Runs (overview, one run, the parts), the comparison stage, Recipe (the chooser,
  the open recipe), the folder picker.
- Modules that read the mark: `store.js` (prefix), `runs/log.js` (lines start wrapped), `recipe/library.js`
  (`data-pane`, the back button), `ui.js` (`confirmOnPhone`, used by `runs/detail.js`, `runs/queue.js`,
  `runs/curricula.js`, `main.js`).

## Open on phone (2026-11-22)

The rail's **Open on phone** button, on the desktop page only, opens a dialog on `/api/phone-access`: the
private Tailscale link to this trainer as a QR code the service drew (`/api/phone-access/qr.png`) and as
text, with the steps for the phone beside it. It replaces reading the link from a terminal.

- **One press to the code.** Opening the dialog is asking for the link. When phone access is on but not
  up (Tailscale was not connected when the trainer opened), the dialog asks the service to try once before
  it says anything, so what it shows is the state now and not the state at launch.
- **It recovers by itself.** While open it follows a start under way, keeps asking while Tailscale is not
  connected or the link is not running, looks again when the person returns to the window, and tests a
  shown link every few seconds. Instructions the person is reading are not rebuilt by a try that ends as
  it began. Nothing is asked before the dialog opens, after it closes or while the page is hidden.
- **Off is the person's choice.** In the `off` state nothing is started until **Turn on phone access** is
  pressed. That button sends `on`. **Check now** and **Try again** send `check` for an immediate attempt
  if access is still enabled. Automatic attempts use `retry` and follow the service's backoff.
- **The link stays in the dialog.** Not in a title, a toast, the address, the console or a preference; the
  code's image is dropped on close. The dialog says who can open the link.
- **The code is on the stage.** White paper (`--code-paper`) with crop marks on the neutral booth, the
  same in both themes, because a camera reads it.
- **Said twice.** Each state change is announced through a hidden `role="status"` line. What a press led to
  when the panel did not change is shown at the foot. The state's one button is a stable node in the foot
  that keeps its focus while it works.

Parts: `js/components/phone.js` (`openPhoneDialog()`; `phoneView(answer)` and `phoneAction(shown)` are pure
and exported for checks), `css/phone.css`, three functions in `js/api.js`, the button in `js/main.js`
(which reads the phone edition's mark and leaves that page's bar as it was). `mobile.html` does not link
`css/phone.css`: it is the page the dialog leads to.

## How the UI is checked

Every interaction is exercised in a real browser against a running service: reading (real runs, real
validation images, real analysis jobs) against the service in use, and anything that changes state (queue
actions, replays, studies, curricula, analysis jobs) against a separate private copy of the same service
code that holds only clearly labelled test data and cannot start training. No test data, test path or
development note belongs in the app or in this folder.
