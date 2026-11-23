# Run recipe reproduction
## Launching a saved preparation

Save the output of `./trainner-gui reproduce prepare bundle.json` as `prepared.json`. Use `./trainner-gui reproduce queue prepared.json` for a held entry or `./trainner-gui reproduce start prepared.json` to launch immediately. Both carry the checkpoint observations into staging. Supply `--request-id` when a client may retry a request.

The routines in `studio.reproduction` return JSON data and a candidate YAML. The service decides whether to save or enqueue a candidate after an explicit user action.

| Python callable | Suggested client input |
| --- | --- |
| `export_run_recipe(runs, run_id, *, attempt_id=None, mode="replay", weight_path=None, weights_version=None, hash_inputs=False)` | Run `id`, optional `attemptId`, `mode`, `weightPath`, `weightsVersion` and `hashInputs` |
| `prepare_reproduction(library, bundle, *, mappings=None, source_root=None, name=None, seed=None, edits=None)` | `{bundle, mappings?, sourceRoot?, name?, seed?, edits?}` |

`runs` is the existing `RunManager`. `library` is the destination `RecipeLibrary`. Export handles an exited managed attempt or a read-only run with saved YAML. A managed export defaults to the recorded current attempt. Imported runs have no managed attempt selector.

## Bundle fields

From the CLI, export with `./trainner-gui reproduce export RUN_ID > run.json`. Prepare it locally with `./trainner-gui reproduce prepare run.json --map /old/data /new/data`. The output contains the proposed YAML and any issues. Weights-only export takes `--mode weights-only --weight-path ARTIFACT` and an optional `--weights-version`.

The export returns `format: "trainner-run-recipe"`, `version: 1`, `mode`, `digest`, `origin`, `recipe`, `seed`, `initialization`, `provenance`, `inputs` and `limitations`. Preserve it as one JSON object. The digest covers the other fields in canonical JSON. It detects edits to the bundle and provides a stable reference. It supplies no signature or authorization. `origin` contains the run `id`, `name`, `source`, `status`, optional `attemptId` and the original `checkout` when known.

- `recipe` contains `submitted`, `effective` and `archived` plus the selected `basis`. The strings retain original comments. A missing provenance field uses null. `provenance` carries the recorded `launch`, `runtime` and `invocation` objects when available. `seed` contains `value` and `evidence`.
- `inputs` preserves each declared input's recipe `field`, `kind`, `value` and known `sourcePath`. File inputs can carry launch stat evidence and an identity observed during export. Dataset declarations record paths. They carry no content fingerprint.
- Replay preserves the selected attempt's actual initialization. Kinds are `fresh`, `pretrained`, `full-state` and `unverified`. For a full-state replay, relocate the numbered state and companion weights together. The native preflight determines which files and starting step the trainer will use.
- `weights-only` requires a primary generator or autoencoder artifact. Pass the optional `weightsVersion` from selection to reject a changed file at export. The bundle's `initialization.weight` holds its artifact key, source path, role and observed identity. Preparation uses that file as pretrained input. It clears resume state and inherited training-model inputs and starts fresh optimizer, scheduler and step state. A separately declared teacher input is preserved. Edits use the existing render payload and are accepted for weights-only. Replay permits a destination name and installation of the recorded seed.

## Preparing on the destination

`mappings` contains `{"from": "/old/location", "to": "/local/location"}` objects. Matching uses path components and the longest prefix. Linux, Windows drive paths, WSL drive-mount aliases, and WSL or network UNC source prefixes are supported. A destination must resolve on this WSL host. A mapping can point to a directory or an individual file. Traversal above a source root and a mapped child's symlink escape are rejected. Map the literal `~` prefix explicitly for a recorded home-relative input.

A managed attempt's original cwd anchors relative inputs. Supply `sourceRoot` for an archive whose original cwd is missing. The archive's current folder does not establish that cwd. A single archived recipe and a single complete log within 128 KiB can supply an explicitly reported startup seed and step. Other archive startup facts remain unverified. Each path in a YAML list is resolved individually, retaining list order. Absolute old paths are preserved as absolute destinations unless a mapping applies. Managed output paths are rebound for a new run. An unrecorded seed can be supplied explicitly. This does not verify the original seed.

Prepare returns `{ok, yaml, initialization, compatibility, inputs, issues, bundleDigest, inputPrecondition}`.

- `ok` means the candidate passed input and recipe checks. `yaml` is null when a path could not be resolved. Other preflight failures can return the candidate for inspection. `initialization` reports `kind`, `seed`, `seedEvidence`, `optimizerState` and the native `resume` report when applicable.
- `compatibility` has `environment`, `source`, `config` and `data`. It compares recorded package, platform and environment facts with destination metadata and uses the already prepared native fingerprint helper to compare training sources. When the original source manifest was retained, the report includes counts and up to 50 paths per added, removed or changed category. A missing helper leaves source verification unavailable. Device availability, data contents, worker state, and deterministic numerical replay require separate evidence.
- `inputs` reports original and local locations and identity status. An embedded save ID survives ordinary copies. A version token identifies a local file instance and may change on relocation. A file with only matching launch stat metadata stays weaker evidence. A conflicting save ID, file size, or full-state companion is an error. The candidate can be valid with warnings for environment drift or historical data the archive lacks.
- The bundle carries no weight, training-state, dataset, or source payloads. Mappings, compatibility checks, and the candidate remain in the response unless the owner explicitly sends them to a save or queue action.


## Transferring an initial generator to a changed head

A weights-only recipe can set `path.transfer_g: {}` beside `path.pretrain_network_g`. At training initialization this loads matching parameter and persistent-buffer names from a native PTH or safetensors checkpoint. Shapes must agree. Floating-point weights can use the destination precision. The destination keeps its class names and output order. A same-sized head will also match by name and shape. To keep specific tensors fresh, list their exact destination keys in `path.transfer_g.exclude`. Unknown exclusions and zero matching parameter keys reject the import.

Studio preflight reports the requested exclusions and `status: atModelInitialization` in `summary.transferG`. A full-state resume reports `inactiveOnResume` and uses its existing loader. In managed attempt provenance, `runtime.transferG` gives the source path, selected wrapper, loaded names, skipped names with reasons, and counts of source keys, destination keys, loaded parameters and buffers. Shape and dtype mismatches include both values. The skipped count includes keys present on just one side. CLI training logs the counts and each skipped key.

## Verifying portable checkpoint copies

With `hash_inputs=True`, export streams SHA-256 over the declared checkpoint files consumed by the replay. Weights-only export hashes the selected weight and any declared teacher. An unavailable requested input prevents a hashed export. Dataset contents and implicit companions absent from the archived recipe are outside this check. Ordinary exports inspect file metadata. Launch capture records a local file version alongside size and modification time.

Prepare verifies any `observed.sha256` in the bundle. A destination row reports `sha256Match` for identical exported bytes or `sha256Mismatch` and an error for changed bytes of the same size. A size difference also fails. Without a digest, a copy preserving size and modification time can give `launchMetadataMatch`; a matching embedded save ID or local version has its own status.

For inputs with a launch receipt, `inputs[].launchEvidence` compares that receipt with the file observed at export. Values are `fileVersionMatch`, `launchMetadataMatch`, `different`, and `unverified`. Older receipts may contain only size and time. Even a successful SHA-256 comparison in that case establishes byte identity from export to prepare. The original launch retains its historical evidence level. Full-state optimizer, scheduler, sampler and random-stream metadata remain in the native preflight response at `initialization.resume`.

## Starting or queuing a prepared recipe

Pass the exact returned `yaml` and `inputPrecondition` together. The value is null on a failed prepare or when there are no checkpoint inputs. Otherwise its shape is `{version: 1, bundleDigest, recipeSha256, files: {<path option>: {path, fileVersion, sha256?}}, digest}`. Its checksum binds the observations supplied by a local caller. Each role pins the destination file observed during preparation, including implicit native full-state companions. An optional SHA-256 covers bytes observed at export. A replay with a recorded official NeuFlow digest can pin the cache path by SHA-256 alone while that cache file is missing. Preparation and queue submission stay offline. The supervisor obtains and verifies the official file before staging it. Source launch evidence stays in the preparation report.

- Direct start takes `runs.start(yaml, request_id, input_precondition=condition)`. Before running mutable preflight checks on a prepared HTTP retry, call `runs.prepared_start_retry(yaml, request_id, input_precondition=condition)`. Return an existing match immediately. On `None`, run ordinary typed preflight and start with the same text and condition.
- Held submission takes `queue.submit(yaml, request_id, recipe_path=path, input_precondition=condition)`. The condition is durable in the entry. Owner release, serial dispatch and resource admission follow the normal queue flow. A changed source at submission or dispatch requires a fresh preparation. After acceptance, the same request ID, YAML and condition retrieve the existing run or entry. A changed or omitted condition conflicts with that request ID.
- During launch the supervisor privately stages each pinned file before spawning the trainer. It compares the version of the open source descriptor before and after copying. When the pin includes a digest it verifies the bytes sent to the copy. The staged set is published together. A staging mismatch leaves a failed attempt and no trainer. A later managed resume selects its numbered full state while retaining private paths for auxiliary inputs such as a teacher.
- Dataset contents and subsequent changes inside the owner's private run directory are outside this precondition. Preparing and submitting a held entry leave training stopped. An ordinary start with no condition keeps its existing input policy.

