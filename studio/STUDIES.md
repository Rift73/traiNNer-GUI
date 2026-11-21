# Controlled ablation studies

Construct `StudyManager(queue: QueueManager)` with the service's existing queue.
Study creation uses the queue's launch preflight. Queue controls own release,
hold, cancellation, and stopping.

## Create

Call `create(name, base_yaml, arms, seeds, request_id=None)`. A JSON handler
can map the fields directly from this request.

```json
{
  "name": "Learning rate ablation",
  "baseYaml": "<complete training YAML>",
  "arms": [
    {"name": "Baseline", "edits": []},
    {"name": "Lower LR", "edits": [{"path": ["train", "optim_g", "lr"], "value": 0.00005}]}
  ],
  "seeds": [0, 41],
  "requestId": "lr-study-1"
}
```

Arms have unique names, ordered `render` edits, and the same ordered list of
distinct integer seeds. Edits use `{path, value}` or `{path, remove: true}`.
The study assigns `manual_seed`. The limit is 128 arm/seed entries per study.
Study recipes start fresh training runs and may load pretrained weights.

Creation preserves the original YAML and edit instructions. Each cell stores
the rendered and preflight-normalized YAML using the existing comment-preserving
renderer. One queue index update publishes the study and every entry as
`held`. An identical request ID and payload returns the existing study.
Reusing the request ID with a different payload raises `ValueError`.
`StudyRejected` carries `issues: [{level, path, message, armId, armName, seed?}]`
when rendering or launch preflight fails.

## Read and release

`list_studies()` returns `[{id, name, createdAt, baseDigest, armCount,
seedCount, entryCount}]` with newest studies first.

`detail(study_id)` and `create(...)` return that same header with
`{baseYaml, seeds: number[], arms: [{id, name, edits, entries}]}` added.
Arm IDs are local to their study. Each entry is an ordinary queue entry with
`{studyId, studyName, armId, armName, seed}` added and `groupId = studyId`.
Its `recipeDigest` identifies the held YAML, available through
`QueueManager.detail(entry_id)`. Linked runs supply `runId` and `runStatus`.

The owner calls `QueueManager.release(entry_id)` to authorize each entry.
Dispatch repeats launch preflight and requires the same saved YAML.

## Compare

Call `compare(study_id, left_arm_id, right_arm_id, dataset, metric, step=None)`
with a shared optional validation iteration. The JSON request fields are
`studyId`, `leftArmId`, `rightArmId`, `dataset`, `metric`, and `step`.

The response repeats those selections and adds
`pairs: [{seed, leftEntryId, rightEntryId, leftStatus, rightStatus,
leftRunId?, rightRunId?, status, comparison?, error?}]`.
Pair status is `unpaired` until both run IDs exist, `unavailable` when a
recorded result cannot be read, `incompatible` when the comparison finds
incompatible or unmatched recordings, or `compared` when matched scores exist.
`comparison` is the existing recorded-validation comparison response. With
`step` omitted, each pair uses its latest common recorded iteration and returns
the actual iteration. Results are descriptive per seed.
For each cohort, the seed summary reports distributions of the left arm, right arm and paired difference (right minus left). A cohort shares the same two iterations, scoring protocol and complete scored population. Missing or partially matched populations appear under exclusions. Sample standard deviation requires at least two training seeds. The result describes observed runs and makes no significance claim.
