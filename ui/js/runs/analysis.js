// Analysis jobs: prediction, ONNX export and loss report, started on purpose by the person. They run on
// the trainer machine's CPU in their own process, never touch training, and keep their output in a
// private job folder. List and detail for #/runs/analysis[/<jobId>], and the start dialog.

import { h, mount, fmt, uid } from '../dom.js';
import { api, newRequestId, imageUrl, sizedUrl, THUMB_SIZE, PREVIEW_MAX } from '../api.js';
import { recipe } from '../state/recipe.js';
import { catalog } from '../state/catalog.js';
import { runs } from '../state/runs.js';
import { go, runHref } from '../actions.js';
import { toast, openDialog, confirmDialog, badge, notice, loading, emptyState, copyText } from '../ui.js';
import { highlightedPre } from '../recipe/yaml-highlight.js';

// ───────── words ─────────

const KINDS = {
  predict: {
    label: 'Prediction',
    noun: 'a prediction',
    text: 'Runs a checkpoint on one image or a folder of images and saves what it produces.',
  },
  evaluate: {
    label: 'Evaluation',
    noun: 'an evaluation',
    title: 'Evaluate a checkpoint',
    text: "Measures a checkpoint on the recipe's validation data with the trainer's own evaluator.",
  },
  export: {
    label: 'ONNX export',
    noun: 'an ONNX export',
    text: 'Converts a checkpoint into a portable ONNX graph that other programs can run.',
  },
  'pack-dataset': {
    label: 'Dataset packing',
    noun: 'a dataset packing job',
    title: 'Pack a dataset',
    noRecipe: true,
    text: 'Copies paired image folders into two LMDB databases that can be moved to another machine, and gives the recipe lines that point at them.',
  },
  'loss-report': {
    label: 'Loss report',
    noun: 'a loss report',
    text: "Measures the gradients of the recipe's configured loss terms on up to 32 training batches, without updating any weights.",
  },
  'weight-drift': {
    label: 'Weight drift',
    noun: 'a weight drift measurement',
    title: 'Measure weight drift',
    noRecipe: true,
    text: 'Measures how far the numbers in a checkpoint moved from a baseline, tensor by tensor.',
  },
  'convert-weights': {
    label: 'Weights conversion',
    noun: 'a weights conversion',
    title: 'Convert a .pth checkpoint to safetensors',
    noRecipe: true,
    text: 'Saves the network weights of a PyTorch .pth file as a safetensors file with the same tensor names and number types. The .pth file is not changed.',
  },
  'task-data': {
    label: 'Data audit',
    noun: 'a data audit',
    title: 'Start a data audit',
    noWeights: true,
    text: 'Lists and checks every dataset of a classification, segmentation or detection recipe.',
  },
  'split-leakage': {
    label: 'Split leakage check',
    noun: 'a split leakage check',
    title: 'Check for split leakage',
    noWeights: true,
    text: 'Looks for identical files in both the training and the validation data of a recipe.',
  },
  'pair-registration': {
    label: 'Pair alignment check',
    noun: 'a pair alignment check',
    title: 'Check pair alignment',
    noWeights: true,
    text: 'Measures how far the low quality images of a sample of pairs are shifted against their ground truth.',
  },
  'split-dataset': {
    label: 'Validation split',
    noun: 'a validation split',
    title: 'Make a validation split',
    noWeights: true,
    text: "Divides a recipe's training data into a training list and a validation list that stay the same every time, and gives the recipe that uses them.",
  },
  'extract-patches': {
    label: 'Patch extraction',
    noun: 'a patch extraction',
    title: 'Extract training patches',
    noWeights: true,
    text: "Cuts a recipe's paired images into small paired patches, saved as PNG files with their list and the recipe lines that use them.",
  },
  'image-difference': {
    label: 'Image difference',
    noun: 'an image difference',
    title: 'Measure the difference between two images',
    noRecipe: true,
    text: 'Compares two saved PNG images pixel by pixel, optionally against a reference, and saves the difference as an image.',
  },
  'average-checkpoints': {
    label: 'Checkpoint average',
    noun: 'a checkpoint average',
    title: 'Average checkpoints of a run',
    noRecipe: true,
    text: 'Averages the weights of several checkpoints of one run into one new weights file.',
  },
  'network-parameters': {
    label: 'Network parameters',
    noun: 'a parameter list',
    title: "List a network's parameters",
    noWeights: true,
    text: "Lists every named parameter and buffer of the recipe's network, with shape and size, before any data is chosen.",
  },
  'index-video-clips': {
    label: 'Video indexes',
    noun: 'a video index preparation',
    title: 'Prepare video indexes',
    noWeights: true,
    // Started from the training videos of a clip-training recipe (recipe/video-prep.js), not from the dialog.
    noDialog: true,
    text: 'Reads each training video of a clip-training recipe once and writes its index into a folder named index next to it. Indexed videos are reused.',
  },
};

/** The job the dialog started last; its page takes the focus once when it is first drawn. */
let startedJobId = null;

/** Kinds whose result and progress are drawn by their own module (loaded on demand). */
const PARTS = {
  'weight-drift': { load: () => import('./drift.js'), file: 'drift.js', render: 'renderWeightDrift', progress: 'driftProgressText' },
  'task-data': { load: () => import('./taskdata.js'), file: 'taskdata.js', render: 'renderTaskData', progress: 'taskDataProgressText' },
  evaluate: { load: () => import('./holdout.js'), file: 'holdout.js', render: 'renderEvaluation', progress: 'evaluationProgressText' },
  'split-leakage': { load: () => import('./holdout.js'), file: 'holdout.js', render: 'renderSplitLeakage', progress: 'splitLeakageProgressText' },
  'pair-registration': { load: () => import('./registration.js'), file: 'registration.js', render: 'renderPairRegistration', progress: 'pairRegistrationProgressText' },
  'split-dataset': { load: () => import('./split.js'), file: 'split.js', render: 'renderSplitDataset', progress: 'splitDatasetProgressText' },
  'image-difference': { load: () => import('./difference.js'), file: 'difference.js', render: 'renderImageDifference', progress: 'imageDifferenceProgressText' },
  'extract-patches': { load: () => import('./patches.js'), file: 'patches.js', render: 'renderExtractPatches', progress: 'extractPatchesProgressText' },
  'convert-weights': { load: () => import('./convert.js'), file: 'convert.js', render: 'renderConvertWeights', progress: 'convertWeightsProgressText' },
  'average-checkpoints': { load: () => import('./average.js'), file: 'average.js', render: 'renderAverageCheckpoints', progress: 'averageCheckpointsProgressText' },
  'network-parameters': { load: () => import('./parameters.js'), file: 'parameters.js', render: 'renderNetworkParameters', progress: 'networkParametersProgressText' },
  'loss-report': { load: () => import('./lossreport.js'), file: 'lossreport.js', render: 'renderLossReport', progress: 'lossReportProgressText' },
  'index-video-clips': { load: () => import('./videoindex.js'), file: 'videoindex.js', render: 'renderVideoIndex', progress: 'videoIndexProgressText' },
};
/**
 * Checkpoint average (studio/analysis_jobs.py `_average_inputs`): numbered .safetensors files of one role of
 * one managed run, at least two, each step once. The role as the service names it, in words.
 */
const ROLE_WORDS = { net_g: 'the network', net_g_ema: 'the EMA network', net_ae: 'the autoencoder', net_ae_ema: 'the EMA autoencoder', net_d: 'the discriminator' };
const AVERAGE_ONE = 'This run has only one numbered checkpoint of this kind, so there is nothing to average.';
const AVERAGE_NONE = 'This run has no numbered checkpoint of this kind, so there is nothing to average.';
const AVERAGE_FEW = 'Choose at least two checkpoints.';
const AVERAGE_BN_LABEL = 'Take batch-norm statistics from the latest chosen checkpoint';
const AVERAGE_BN_HELP =
  'For a network with batch normalisation. Its running mean, variance and counter are copied from the latest chosen checkpoint instead of being averaged. Without this, checkpoints whose counters differ are refused.';
const AVERAGE_MORE = 'This run has more checkpoints of this kind: only the newest 250 are listed.';
const AVERAGE_NOTE =
  "Each checkpoint counts the same. The result holds weights only: it can be tried, evaluated, exported or used as starting weights, but it cannot continue this run's training state.";
/** Network parameters (studio/analysis_jobs.py): the recipe text is enough; `inspectTransfer` asks for the transfer preview. */
const TRANSFER_HELP =
  "Runs the recipe's weight transfer on a temporary network on the CPU and reports which tensors are copied from the starting weights and which start fresh.";
const PARAMETERS_NOTE = 'The counts describe the bare network as it is built. Wrappers and optimizer state added during training can change its footprint.';
/** "1,000 to 3,000" for three or more steps, "1 and 2" for two. */
const stepRange = (steps) => (steps.length === 2 ? `${fmt.int(steps[0])} and ${fmt.int(steps[1])}` : `${fmt.int(steps[0])} to ${fmt.int(steps[steps.length - 1])}`);
/**
 * Patch extraction (studio/analysis_jobs.py, studio/dataset_patches.py): a restoration recipe, a
 * PairedImageDataset stored as folders on disk, a whole-number scale, and a step of at most the patch size.
 */
const PATCH_TASK = 'Patch extraction covers restoration recipes whose data are paired ground truth and low quality folders (dataset type PairedImageDataset).';
const PATCH_NONE = `${PATCH_TASK} This recipe has none.`;
const PATCH_SCALE = 'Patch extraction needs the recipe to set its scale as a whole number of at least 1.';
const PATCH_STEP = 'The step can be at most the patch size, so every part of an image lands in a patch.';
const PATCH_NOTE =
  "The patches are written as PNG files into the job's folder. The source images are not changed. Make the validation split first, so patches of one image do not land on both sides.";
/** Validation split: refused unless datasets.train is of a type the service splits for the recipe's task (studio/analysis_jobs.py). */
const SPLIT_GATE =
  'A validation split needs a recipe whose training data are paired folders, paired frame folders, class folders, images with masks or a COCO annotation file.';
/** The training dataset types the service splits for each task, lowercase as it compares them (studio/analysis_jobs.py, studio/split_audit.py). */
const SPLIT_TYPES = {
  image_restoration: ['pairedimagedataset', 'pairedvideodataset'],
  image_classification: ['classfolderdataset'],
  semantic_segmentation: ['imagemaskdataset'],
  object_detection: ['cocodetectiondataset'],
};
/** Paired frame folders are split by whole scene only (studio/analysis_jobs.py: `groupBy` left out, or "scene"). */
const SPLIT_VIDEO = 'pairedvideodataset';
const SPLIT_SCENES = 'Whole scenes stay together: every window of a scene goes to one side.';
const SPLIT_SHARE_HELP_VIDEO = 'Of the training windows. The rest stay for training.';
const SPLIT_NOTE_VIDEO = 'Nothing is copied or moved. The result is two scene lists, and the recipe with its training and validation data set to them.';
const SPLIT_NOTE = 'Nothing is copied or moved. The result is two lists of file names and the recipe lines that use them.';
const SPLIT_NOTE_LABELLED = 'Nothing is copied or moved. The result is two lists, and the recipe with its training and validation data set to them.';
const SPLIT_SHARE_HELP = 'Of the training pairs. The rest stay for training.';
const SPLIT_SHARE_HELP_LABELLED = 'Of the training images. The rest stay for training.';
const SPLIT_GROUP_HELP = 'Frames of one scene or crops of one photo belong on the same side. Otherwise validation is measured on near-copies of training images.';
/**
 * Labelled data (studio/supervised_split.py `_groups`): identical files are always joined; a folder choice
 * groups by the image's folder (or its first folder) below the data folder, by name, so images in no
 * subfolder share one group; for class folders the class folder itself is left out of that name.
 */
const SPLIT_GROUP_HELP_LABELLED = `${SPLIT_GROUP_HELP} Identical files always stay together. With a folder choice, images that are in no subfolder form one group.`;
const SPLIT_GROUP_HELP_CLASSES = `${SPLIT_GROUP_HELP_LABELLED} The folders meant are those inside each class folder, and folders of the same name count as one.`;
/** `groupBy` of a validation split: what stays on the same side. Paired data, then labelled data. */
const GROUP_WORDS = { pair: 'Each pair on its own', parent: 'Pairs in the same folder', scene: 'Everything under the same top-level folder' };
const GROUP_WORDS_LABELLED = { sample: 'Each image on its own', parent: 'Images in the same folder', scene: 'Everything under the same top-level folder' };
const groupWords = (task) => (typeof task === 'string' && task && task !== 'image_restoration' ? GROUP_WORDS_LABELLED : GROUP_WORDS);
/** "Frames per window" that is not a whole number from 1 (prediction and sequence export). */
const CLIP_ERROR = 'Enter the frames per window as a whole number of at least 1, or leave it empty.';
/** A fraction as a percentage without float noise: 0.07 -> "7%". */
const shareText = (fraction) => `${Number((fraction * 100).toPrecision(12))}%`;
const DIFF_NOTE = 'The images must be PNG files of the same size. The service checks that each file is still the one chosen.';
/** The sides of an image difference, in the words of its form. */
const DIFF_SIDES = { left: 'First image', right: 'Second image', reference: 'Reference' };
/**
 * A saved image handed to the start dialog, `{run|job|analysis: id, path, version, label}`, as what the
 * service takes (exactly one owner key, path and version) and the words shown for it. null for anything else.
 */
function savedImage(value) {
  if (!isPlainObject(value)) return null;
  const owners = ['run', 'job', 'analysis'].filter((key) => typeof value[key] === 'string' && value[key]);
  if (owners.length !== 1 || typeof value.path !== 'string' || !value.path || typeof value.version !== 'string' || !value.version) return null;
  // Without a label the file's own name is shown, as the literal text it is.
  const named = typeof value.label === 'string' && value.label.trim() !== '';
  return {
    selection: { [owners[0]]: value[owners[0]], path: value.path, version: value.version },
    label: named ? value.label.trim() : baseName(value.path),
    literal: !named,
  };
}
/** Pair alignment: refused unless the split is a PairedImageDataset and the recipe scale a positive whole number (studio/analysis_jobs.py). */
const PAIR_NONE =
  'The pair alignment check covers restoration recipes whose data are paired ground truth and low quality folders (dataset type PairedImageDataset). This recipe has none.';
const PAIR_SCALE = 'The pair alignment check needs the recipe to set its scale as a whole number of at least 1.';
/** Dataset key in the app's set words: "Training data", "Validation data (val_2)". */
const SET_WORDS = { train: 'Training data', val: 'Validation data', test: 'Test data' };
const setWords = (key) => {
  const phase = String(key ?? '').split('_')[0];
  const word = SET_WORDS[phase];
  return word ? (key === phase ? word : `${word} (${key})`) : String(key ?? '');
};
/** Networks the service predicts as frame sequences (traiNNer/archs/arch_info.py TEMPORAL_ARCHS, trainner_config/tasks.py recurrent_architectures). */
const SEQUENCE_ARCHS = ['tscunet', 'temporalspan', 'temporalspanv2', 'tfdat', 'rdual', 'rsr_compact'];
const isSequence = (task, arch) => task === 'recurrent_restoration' || SEQUENCE_ARCHS.includes(String(arch ?? '').toLowerCase());
const INPUT_HELP = 'An image file, or a folder whose images are all predicted. Paths are on the trainer machine.';
const INPUT_HELP_RECORDS =
  "A .csv file with the recipe's feature columns, named in its first row. Target columns may be present and are ignored. The result is a file of predictions, one row per record. Paths are on the trainer machine.";
const INPUT_HELP_SEQUENCE = 'A folder of PNG or JPEG frames, in numbered order within each scene. Paths are on the trainer machine.';
const INPUT_HELP_AUTOENCODER =
  "An image file, or a folder whose images are all reconstructed. Each image's width and height must divide evenly by the autoencoder's scale. Paths are on the trainer machine.";
const EVAL_NOTE = 'Runs on the CPU with 32-bit floats, so it is slower than validation during training.';
const EVAL_NOTE_GPU = 'Runs on the GPU with 32-bit floats.';
const NO_VAL =
  'This recipe has no validation dataset (val, or a name that starts with val_), so there is nothing to compare its training data with. Add one on the Recipe screen first.';
const LEAK_TASKS = 'The split leakage check covers restoration recipes with paired image folders.';
const LEAK_NOTE = "Compares the encoded bytes of the recipe's training and validation files. Large datasets take a while.";
const NO_HOLDOUT =
  'This recipe has no validation or test dataset (a dataset whose name starts with val or test), so there is nothing to evaluate. Add one on the Recipe screen first.';
/** Validation and test dataset keys of a recipe config: the splits an evaluation can measure. */
/** studio/evaluate.py: `name.partition("_")[0] in {"val", "test"}`, so "validation" or "val2" is not one. */
const holdoutKeys = (config) => (isPlainObject(config?.datasets) ? Object.keys(config.datasets).filter((key) => ['val', 'test'].includes(key.split('_')[0])) : null);
/** Saved evaluation images: refused for tasks with their own scores (studio/evaluate.py, provider_scores). */
const SAVE_ELSEWHERE = 'Saved output images are for restoration and reconstruction recipes. To see outputs for this task, use a prediction job.';
const SAVE_HELP = "Keeps the model's output for each image so it can be looked at. Restoration and reconstruction tasks.";
/**
 * Optional outputs of an evaluation, by the request field the service names in
 * `summary.actions.holdout.optionalExports` (studio/analysis_jobs.py reads each as a boolean).
 */
const EXPORT_WORDS = {
  saveImages: { label: 'The output images', help: SAVE_HELP },
  saveDetections: { label: 'The detected boxes', help: 'One COCO results file per dataset, to download or load into other tools.' },
  saveClassifications: {
    label: "Each image's predicted class",
    help: 'For every image: the true class, the predicted class and the most likely classes with their probabilities, as a file and as a table in the result.',
  },
  saveTemporalErrors: {
    label: 'Frame-to-frame errors',
    help: 'For each pair of consecutive frames: how far the change in the output is from the change in the reference. Works without saving images.',
  },
  saveDetectionOperatingPoints: { label: 'Operating points', help: 'Precision and recall of the detector at each score threshold, per dataset.' },
};
const EXPORT_MASKS = 'The predicted masks, as PNG images.';
const EXPORT_UNKNOWN = 'An output this screen has no description for.';
const EXPORT_NONE = "This task's evaluation keeps its scores only.";
/** Request fields of an evaluation that an optional output can never stand for. */
const EVALUATE_FIELDS = ['kind', 'yaml', 'weights', 'weightsVersion', 'split', 'device', 'requestId'];
/** Body without its request id, keys sorted at every level: two starts with the same text are the same request. */
function requestKey(params) {
  const sort = (value) =>
    Array.isArray(value) ? value.map(sort) : isPlainObject(value) ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, sort(value[key])])) : value;
  const { requestId: _omit, ...rest } = params;
  return JSON.stringify(sort(rest));
}
const partModules = new Map();
/** The module of a part kind, imported once per page. */
function loadPart(kind) {
  if (!partModules.has(kind)) partModules.set(kind, PARTS[kind].load());
  return partModules.get(kind);
}

/** Tasks the data audit covers (studio/analysis_jobs.py). */
const AUDIT_TASKS = ['image_classification', 'semantic_segmentation', 'object_detection'];
// Tasks whose data the service checks (the job kind task-data): the labelled image tasks, and tables of
// numeric records. Only the data check reads this list; AUDIT_TASKS keeps deciding everything else.
const DATA_CHECK_TASKS = [...AUDIT_TASKS, 'tabular_regression'];
const AUDIT_ELSEWHERE = 'The data audit covers classification, segmentation and detection recipes. Paired image folders are checked on the Data screen.';
const DRIFT_NOTE = 'Both files must be .safetensors. The result measures movement of the weights, not quality.';
/** ONNX export: the size fields of a single-image graph and of one that takes frame sequences; the box that lets the graph take other sizes. */
const EXPORT_HEIGHT_HELP = '1 to 32768 px. The graph is exported for this size.';
const EXPORT_HEIGHT_HELP_FRAMES = '1 to 32768 px, of one frame. The graph is exported for this size.';
const EXPORT_WIDTH_HELP = '1 to 32768 px.';
const EXPORT_WIDTH_HELP_FRAMES = '1 to 32768 px, of one frame.';
const EXPORT_ANY_SIZE_HELP = 'The exported graph then takes any height and width, one image at a time. The size above is still needed: it is the example the graph is built with.';
/** At most this many images of a detection result get a table; the file has all of them. */
const MAX_DETECTION_IMAGES = 100;
/** Up to this many images, each is drawn with its boxes straight away; above it, on request. */
const AUTO_OVERLAYS = 12;

const STATUS = {
  running: { label: 'Running', kind: 'run', open: true },
  cancelling: { label: 'Cancelling', kind: 'run', open: true },
  completed: { label: 'Completed', kind: 'ok', open: false },
  failed: { label: 'Failed', kind: 'err', open: false },
  cancelled: { label: 'Cancelled', kind: '', open: false },
};

const VIEW_LEAD =
  "An analysis job runs on the trainer machine's CPU, separately from training; an evaluation can use the GPU instead. Its results stay in a private job folder. One job runs at a time.";
const DIALOG_LEAD_GPU =
  "This job runs on the trainer machine's GPU and needs it to itself, so it does not run beside a training. Its results stay in a private job folder. One job runs at a time.";
const DIALOG_LEAD =
  "This job runs on the trainer machine's CPU, separately from training. Its results stay in a private job folder. One job runs at a time.";

/** Outputs per result page: the backend's default page size (GET /api/analysis/result, limit 1..250). */
const RESULT_PAGE = 50;
/** At most this many classification predictions are drawn as table rows. */
const MAX_PREDICTION_ROWS = 500;
/** Above this many classes, the raw logits of a row are written in one cell instead of a column each. */
const MAX_LOGIT_COLUMNS = 16;

/** Stage words of a dataset packing job (studio/dataset_pack.py progress "stage"). */
const PACK_STAGES = {
  validating: 'Checking the source folders',
  packing: 'Copying images into the databases',
  verifying: 'Verifying the copy before publishing it',
};

const PACK_COST = 'It writes a full copy of the images to disk on the trainer machine.';
const PACK_STOP = 'Stop packing. Nothing is published if it has not finished.';

/** Folder lists of a pack request: gtRoots/lqRoots, or the single gt/lq the backend also accepts. */
function packRoots(dataset) {
  const list = (many, one) => (Array.isArray(many) ? many.map(String) : typeof one === 'string' && one ? [one] : []);
  return { gt: list(dataset?.gtRoots, dataset?.gt), lq: list(dataset?.lqRoots, dataset?.lq) };
}

/** "1.23 GB" in decimal units, exact bytes for `title`. */
function bytesText(n) {
  if (!Number.isFinite(n)) return '';
  const units = [['TB', 1e12], ['GB', 1e9], ['MB', 1e6], ['kB', 1e3]];
  const [unit, size] = units.find(([, s]) => n >= s) ?? ['bytes', 1];
  return unit === 'bytes' ? `${fmt.int(n)} bytes` : `${(n / size).toFixed(2)} ${unit}`;
}

const bytesEl = (n) => (Number.isFinite(n) ? h('span', { title: `${fmt.int(n)} bytes` }, bytesText(n)) : null);

const PARAM_LABELS = {
  destination: 'Destination',
  weights: 'Weights',
  input: 'Input',
  discriminator: 'Discriminator weights',
  height: 'Input height',
  width: 'Input width',
  opset: 'ONNX opset',
  step: 'Update number',
  batches: 'Batches',
  parameter: 'Generator parameter',
  left: 'Baseline checkpoint',
  right: 'Candidate checkpoint',
  auditLimit: 'Files to open in each dataset',
  sampleLimit: 'Examples to keep',
  scoreThreshold: 'Lowest score to keep',
  topLimit: 'Most boxes per image',
  split: 'Datasets measured',
  saveImages: 'Save output images',
  saveDetections: 'Keep the detected boxes',
  saveClassifications: "Keep each image's predicted class",
  saveTemporalErrors: 'Keep the frame-to-frame errors',
  saveDetectionOperatingPoints: 'Keep the operating points',
  groupLimit: 'Examples to list',
  clipSize: 'Frames per window',
};
/** Labels that differ for one kind. */
const KIND_PARAM_LABELS = {
  'pair-registration': { split: 'Which data', sampleLimit: 'Pairs to measure', radius: 'Search radius', seed: 'Seed' },
  'split-dataset': { groupBy: 'Keep together', validationFraction: 'Share for validation', seed: 'Seed' },
  'extract-patches': { split: 'Which data', cropSize: 'Patch size', stride: 'Step', scale: 'Scale' },
  'convert-weights': { weights: 'Checkpoint', payload: 'Weights dictionary' },
  'image-difference': { left: DIFF_SIDES.left, right: DIFF_SIDES.right, reference: DIFF_SIDES.reference, region: 'Part of the image' },
  'average-checkpoints': { run: 'Run', role: 'Network', inputs: 'Checkpoints', batchNorm: 'Batch-norm statistics' },
  'network-parameters': { weights: 'Starting weights' },
  'index-video-clips': { roots: 'Videos', settings: 'Descale settings' },
  evaluate: { device: 'Measured on' },
};
/** "CPU" or "GPU" for where a job ran; any other word as the service wrote it. */
export const deviceWords = (device) => {
  const key = String(device).toLowerCase();
  return key === 'cpu' ? 'CPU' : key === 'cuda' ? 'GPU' : String(device);
};
/** Where an evaluation is measured: the help under the choice, the line shown while a training runs, and the hint after a refusal. */
const DEVICE_HELP = 'The GPU is faster. A GPU evaluation needs the GPU to itself: it starts only while no training is running, and it holds the GPU until it finishes.';
const GPU_TRAINING_NOTE = 'A training is running now: the service will refuse a GPU evaluation until it finishes.';
const GPU_BUSY_HINT = 'The GPU is in use. Wait for it to be free, or choose "On the CPU" and start again.';
const GPU_BUSY_RE = /lease is occupied|Waiting for an active managed trainer|Waiting for an ordinary/i;
/** The hint for a refused evaluation that asked for the GPU while the GPU was taken; '' for anything else. */
export const gpuBusyHint = (body, message) => (body?.kind === 'evaluate' && body.device === 'cuda' && GPU_BUSY_RE.test(String(message ?? '')) ? GPU_BUSY_HINT : '');

const IMAGE_RE = /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i;

export const jobHref = (id) => (id ? `#/runs/analysis/${encodeURIComponent(id)}` : '#/runs/analysis');
/** A kind this module does not know reads "Analysis job"; its id is shown beside it as a detail (kindDetail). */
const kindLabel = (kind) => KINDS[kind]?.label ?? 'Analysis job';
/** The kind picker's groups, by what a job works on. A kind named in no group is listed under "Other". */
const KIND_GROUPS = [
  // 'average-checkpoints' is listed only when the dialog was opened with a run's checkpoints (it needs a run).
  ['With a checkpoint', ['predict', 'evaluate', 'export', 'loss-report', 'weight-drift', 'convert-weights', 'average-checkpoints']],
  ["With a recipe's network", ['network-parameters']],
  ["With a recipe's data",['task-data', 'split-leakage', 'pair-registration', 'split-dataset', 'extract-patches', 'pack-dataset']],
  ['With saved images', ['image-difference']],
];
const kindDetail = (kind) => (!KINDS[kind] && kind ? h('span', { class: 'analysis__kind-id' }, String(kind)) : null);
/** "a prediction", "an ONNX export", "a loss report" (keeps ONNX in capitals). */
const kindNoun = (kind) => KINDS[kind]?.noun ?? 'an analysis job';
const statusOf = (status) => STATUS[status] ?? { label: status ? String(status).charAt(0).toUpperCase() + String(status).slice(1) : 'Unknown', kind: '', open: false };
const baseName = (path) => {
  const text = String(path ?? '').replace(/[\\/]+$/, '');
  const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
  return cut < 0 ? text : text.slice(cut + 1);
};
const folderOf = (path) => {
  const text = String(path ?? '').trim();
  const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
  return cut > 0 ? text.slice(0, cut) : '';
};
const shortDigest = (digest) => (typeof digest === 'string' ? digest.slice(0, 12) : '');

function timeEl(iso) {
  const text = fmt.dateTime(iso);
  return text ? h('time', { datetime: iso, title: iso }, text) : null;
}

/** "net_g_2.safetensors on 0005.png", with full paths in `title`. */
function whatLine(job) {
  const p = job?.params ?? {};
  const parts = [];
  const titles = [];
  if (job?.kind === 'network-parameters') {
    // Weights are named only when a transfer preview was asked: the recipe's own starting weights.
    if (p.weights) {
      parts.push('with a transfer preview');
      titles.push(`Starting weights: ${p.weights}`);
    }
    return { text: parts.join(' '), title: titles.join('\n') };
  }
  if (job?.kind === 'average-checkpoints') {
    // "3 checkpoints, iterations 1,000 to 3,000"
    const inputs = Array.isArray(p.inputs) ? p.inputs.filter(isPlainObject) : [];
    const steps = inputs.map((entry) => entry.step).filter(Number.isFinite).sort((a, b) => a - b);
    if (inputs.length) parts.push(`${fmt.plural(inputs.length, 'checkpoint')}${steps.length >= 2 && steps.length === inputs.length ? `, iterations ${stepRange(steps)}` : ''}`);
    for (const entry of inputs) if (typeof entry.path === 'string' && entry.path) titles.push(entry.path);
    return { text: parts.join(' '), title: titles.join('\n') };
  }
  if (p.weights) {
    parts.push(baseName(p.weights));
    titles.push(p.weights);
  }
  if (job?.kind === 'predict' && p.input) {
    parts.push(`on ${baseName(p.input)}`);
    titles.push(p.input);
  }
  // "at 64 × 64, any size": the size the graph was built with, and that it takes others too.
  if (job?.kind === 'export' && Number.isFinite(p.height) && Number.isFinite(p.width)) parts.push(`at ${fmt.dims(p.width, p.height)}${p.dynamicSpatial === true ? ', any size' : ''}`);
  if (job?.kind === 'loss-report') {
    if (Number.isFinite(p.batches)) parts.push(fmt.plural(p.batches, 'batch', 'batches'));
    if (p.discriminator) titles.push(p.discriminator);
  }
  if (job?.kind === 'pack-dataset') {
    const roots = packRoots(p.dataset);
    if (roots.gt.length) {
      parts.push(`${baseName(roots.gt[0])}, ${fmt.plural(roots.gt.length, 'folder pair', 'folder pairs')}`);
      roots.gt.forEach((gt, i) => titles.push(`${gt} with ${roots.lq[i] ?? '(no partner)'}`));
    }
  }
  if (job?.kind === 'weight-drift' && (p.left || p.right)) {
    // "net_g_2 to net_g_8": baseline first.
    parts.push(`${stemName(p.left) || '(no baseline)'} to ${stemName(p.right) || '(no candidate)'}`);
    if (p.left) titles.push(`Baseline: ${p.left}`);
    if (p.right) titles.push(`Candidate: ${p.right}`);
  }
  if (job?.kind === 'convert-weights' && typeof p.payload === 'string' && p.payload) {
    // "classifier.pth, dictionary params_ema"
    if (parts.length) parts[parts.length - 1] += ',';
    parts.push(`dictionary ${p.payload}`);
  }
  if (job?.kind === 'extract-patches') {
    // "24 px patches of the training data, step 20": sizes are in low quality pixels.
    const size = Number.isFinite(p.cropSize) ? `${fmt.int(p.cropSize)} px patches` : 'patches';
    parts.push(`${size} of the ${setWords(p.split ?? 'train').toLowerCase()}${Number.isFinite(p.stride) ? `, step ${fmt.int(p.stride)}` : ''}`);
  }
  if (job?.kind === 'split-dataset') {
    const share =Number.isFinite(p.validationFraction) ? `${shareText(p.validationFraction)} for validation` : '';
    const seed = Number.isInteger(p.seed) ? `seed ${p.seed}` : '';
    if (share || seed) parts.push([share, seed].filter(Boolean).join(', '));
  }
  if (job?.kind === 'image-difference' && (p.left || p.right)) {
    // "a.png against b.png": first image first.
    parts.push(`${baseName(p.left) || '(no first image)'} against ${baseName(p.right) || '(no second image)'}`);
    for (const side of Object.keys(DIFF_SIDES)) if (typeof p[side] === 'string' && p[side]) titles.push(`${DIFF_SIDES[side]}: ${p[side]}`);
  }
  if (job?.kind === 'pair-registration') parts.push(`${Number.isFinite(p.sampleLimit) ? fmt.plural(p.sampleLimit, 'pair') : 'pairs'} of the ${setWords(p.split ?? 'train').toLowerCase()}`);
  if (job?.kind === 'evaluate') parts.push(typeof p.split === 'string' && p.split ? `on ${p.split}` : 'on every validation and test dataset');
  if (job?.kind === 'task-data') {
    if (p.auditLimit === null) parts.push('every file opened');
    else if (Number.isFinite(p.auditLimit)) parts.push(`up to ${fmt.plural(p.auditLimit, 'file', 'files')} opened per dataset`);
  }
  return { text: parts.join(' '), title: titles.join('\n') };
}

/** File name without its extension: "net_g_8.safetensors" -> "net_g_8". */
function stemName(path) {
  const name = baseName(path);
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(0, dot) : name;
}

/** Same file, written two ways: separators unified, doubled and trailing ones dropped. */
const samePath = (a, b) => {
  const norm = (s) => String(s).trim().replace(/\\/g, '/').replace(/\/{2,}/g, '/').replace(/\/+$/, '');
  return norm(a) === norm(b);
};

/** Key to words: "output_kind" -> "Output kind", "outputCount" -> "Output count". */
function keyWords(key) {
  const words = String(key)
    .replace(/_/g, ' ')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : String(key);
}

function scalarText(value) {
  if (value === null || value === undefined) return 'Not reported';
  if (typeof value === 'number') return Number.isInteger(value) ? fmt.int(value) : fmt.num(value);
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  return String(value);
}

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/** A task id in words from the catalog (once it has loaded), the id itself otherwise; the id stays in `title`. */
function taskText(id) {
  const known = catalog.get().phase === 'ready' && catalog.get().tasks?.some((task) => task.id === id);
  return h('span', { title: id }, known ? catalog.taskLabel(id) : id);
}

/**
 * The classification predictions file (traiNNer/tasks/predict.py): {task, architecture, output_kind,
 * class_names, image_size, predictions: [{input, class_id, class_name, logits}]}. null for another shape.
 */
function readPredictions(doc) {
  if (!isPlainObject(doc) || !Array.isArray(doc.predictions)) return null;
  const rows = doc.predictions;
  const ok = rows.every((row) => isPlainObject(row) && typeof row.input === 'string' && Number.isInteger(row.class_id) && Array.isArray(row.logits));
  if (!ok) return null;
  return { rows, classNames: Array.isArray(doc.class_names) ? doc.class_names.map(String) : null };
}

/** Table of classification predictions: input file, predicted class, raw logits exactly as given. */
function predictionsTable({ rows, classNames }) {
  const shown = rows.slice(0, MAX_PREDICTION_ROWS);
  const classCount = classNames?.length ?? Math.max(0, ...shown.map((row) => row.logits.length));
  const perColumn = classCount > 0 && classCount <= MAX_LOGIT_COLUMNS;
  const className = (i) => classNames?.[i] ?? `class ${i}`;
  const predicted = (row) => `${row.class_name ?? className(row.class_id)} (ID ${row.class_id})`;
  return h(
    'div',
    { class: 'stack' },
    rows.length > shown.length
      ? h('p', { class: 'analysis__note' }, `Showing the first ${fmt.int(shown.length)} of ${fmt.int(rows.length)} predictions. Download the file for all of them.`)
      : null,
    h(
      'div',
      { class: 'analysis__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Predicted classes' },
      h(
        'table',
        { class: 'table analysis__table' },
        h(
          'thead',
          null,
          h(
            'tr',
            null,
            h('th', { scope: 'col', rowspan: perColumn ? '2' : null }, 'Input file'),
            h('th', { scope: 'col', rowspan: perColumn ? '2' : null }, 'Predicted class'),
            h('th', { scope: 'colgroup', colspan: perColumn ? String(classCount) : null }, 'Raw logits'),
          ),
          perColumn ? h('tr', null, Array.from({ length: classCount }, (_, i) => h('th', { scope: 'col', class: 'num' }, className(i)))) : null,
        ),
        h(
          'tbody',
          null,
          shown.map((row) =>
            h(
              'tr',
              null,
              h('td', { class: 'mono', title: row.input }, baseName(row.input)),
              h('td', null, predicted(row)),
              perColumn
                ? Array.from({ length: classCount }, (_, i) => h('td', { class: 'num mono' }, i < row.logits.length ? String(row.logits[i]) : ''))
                : h('td', { class: 'mono analysis__logits' }, row.logits.map((value, i) => `${className(i)}: ${String(value)}`).join(', ')),
            ),
          ),
        ),
      ),
    ),
  );
}

/** A frame-sequence prediction: frames read and produced, the window, scenes, and why boundary frames have no output. */
function framesBlock(result) {
  const n = (key) => (Number.isFinite(result[key]) ? fmt.int(result[key]) : null);
  const rows = [
    ['Input frames', n('inputFrames')],
    ['Frames produced', n('centerFrames')],
    ['Boundary frames without output', n('boundaryFramesOmitted')],
    ['Frames per window', n('clipSize')],
    ['Scenes', n('scenes')],
  ].filter(([, value]) => value !== null);
  const omitted = Number.isFinite(result.boundaryFramesOmitted) ? result.boundaryFramesOmitted : null;
  const span = Number.isFinite(result.clipSize) ? ` of ${fmt.plural(result.clipSize, 'frame')}` : '';
  return h(
    'div',
    { class: 'analysis__frames' },
    h('h4', { class: 'analysis__sub-title' }, 'Frames'),
    h('dl', { class: 'kv analysis__kv' }, rows.map(([label, value]) => [h('dt', null, label), h('dd', { class: 'num' }, value)])),
    h(
      'p',
      { class: 'analysis__note' },
      `Only frames with a full window${span} around them are produced, so the first and last frames of each scene have no output${omitted !== null ? ` (${fmt.plural(omitted, 'frame')} here)` : ''}. Different scenes and folders are kept apart.`,
    ),
  );
}

/**
 * The detection predictions file (traiNNer/tasks/object_detection/inference.py and tasks/predict.py):
 * {task: 'object_detection', output_kind: 'detections', box_format: 'xyxy', coordinate_space:
 * 'original_image_pixels', score_threshold, top_limit, class_names, category_ids, …, predictions: [{input,
 * file_name, width, height, detections: [{box_xyxy: [x1, y1, x2, y2], category_id, class_id, class_name,
 * score}]}]}. null for another shape.
 */
function readDetections(doc) {
  if (!isPlainObject(doc) || !Array.isArray(doc.predictions)) return null;
  if (doc.task !== 'object_detection' && doc.output_kind !== 'detections') return null;
  if (doc.box_format !== undefined && doc.box_format !== 'xyxy') return null;
  const box = (d) => isPlainObject(d) && Array.isArray(d.box_xyxy) && d.box_xyxy.length === 4 && d.box_xyxy.every(Number.isFinite);
  const ok = doc.predictions.every((row) => isPlainObject(row) && typeof row.input === 'string' && Array.isArray(row.detections) && row.detections.every(box));
  if (!ok) return null;
  return {
    rows: doc.predictions,
    scoreThreshold: Number.isFinite(doc.score_threshold) ? doc.score_threshold : null,
    topLimit: Number.isFinite(doc.top_limit) ? doc.top_limit : null,
  };
}

const detectionName = (d) => (typeof d.class_name === 'string' && d.class_name ? d.class_name : Number.isFinite(d.category_id) ? `category ${d.category_id}` : 'unnamed');
/** Scores to 3 decimals for reading; the exact value stays in `title`. */
const scoreText = (score) => (Number.isFinite(score) ? score.toFixed(3) : 'Not reported');
const pixelCell = (value) => h('td', { class: 'num mono', title: String(value) }, Number.isInteger(value) ? String(value) : value.toFixed(1));

/** One image's boxes: category ID and name, score, corners in original pixels. */
function boxesTable(row) {
  return h(
    'div',
    { class: 'analysis__table-wrap', tabindex: '0', role: 'region', 'aria-label': `Boxes in ${row.file_name || baseName(row.input)}` },
    h(
      'table',
      { class: 'table analysis__table' },
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', { scope: 'col', class: 'num' }, 'Box'),
          h('th', { scope: 'col', class: 'num' }, 'Category ID'),
          h('th', { scope: 'col' }, 'Category'),
          h('th', { scope: 'col', class: 'num' }, 'Score'),
          h('th', { scope: 'col', class: 'num' }, 'Left x'),
          h('th', { scope: 'col', class: 'num' }, 'Top y'),
          h('th', { scope: 'col', class: 'num' }, 'Right x'),
          h('th', { scope: 'col', class: 'num' }, 'Bottom y'),
        ),
      ),
      h(
        'tbody',
        null,
        row.detections.map((d, i) =>
          h(
            'tr',
            null,
            h('td', { class: 'num' }, String(i + 1)),
            h('td', { class: 'num mono' }, Number.isFinite(d.category_id) ? String(d.category_id) : ''),
            h('td', null, detectionName(d)),
            h('td', { class: 'num mono', title: Number.isFinite(d.score) ? String(d.score) : null }, scoreText(d.score)),
            d.box_xyxy.map(pixelCell),
          ),
        ),
      ),
    ),
  );
}

const SVG_NS = 'http://www.w3.org/2000/svg';
/** An SVG element (h() makes HTML elements only). */
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) if (value !== null && value !== undefined) el.setAttribute(key, String(value));
  return el;
}

/** Any result value as labelled values: objects as key/value lists, lists of rows as tables. */
function renderValue(value, depth = 0) {
  if (Array.isArray(value)) {
    if (value.length === 0) return h('span', { class: 'muted' }, 'None');
    if (value.every(isPlainObject)) {
      const nested = value.some((row) => Object.values(row).some((v) => Array.isArray(v) || isPlainObject(v)));
      if (nested || depth > 5) {
        return h(
          'ol',
          { class: 'analysis__items' },
          value.map((row, index) => h('li', { class: 'analysis__item' }, h('h5', { class: 'analysis__item-title' }, `${index + 1}`), renderValue(row, depth + 1))),
        );
      }
      const columns = [...new Set(value.flatMap((row) => Object.keys(row)))];
      return h(
        'div',
        { class: 'analysis__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Result table' },
        h(
          'table',
          { class: 'table analysis__table' },
          h('thead', null, h('tr', null, columns.map((key) => h('th', { scope: 'col', class: value.some((row) => typeof row[key] === 'number') ? 'num' : null }, keyWords(key))))),
          h(
            'tbody',
            null,
            value.map((row) =>
              h(
                'tr',
                null,
                columns.map((key) => h('td', { class: typeof row[key] === 'number' ? 'num' : null }, key in row ? scalarText(row[key]) : '')),
              ),
            ),
          ),
        ),
      );
    }
    return h('span', null, value.map(scalarText).join(', '));
  }
  if (isPlainObject(value)) {
    const keys = Object.keys(value);
    if (!keys.length) return h('span', { class: 'muted' }, 'None');
    return h(
      'dl',
      { class: 'kv analysis__kv' },
      keys.map((key) => [h('dt', null, keyWords(key)), h('dd', null, key === 'task' && typeof value[key] === 'string' ? taskText(value[key]) : renderValue(value[key], depth + 1))]),
    );
  }
  const text = scalarText(value);
  return value === null || value === undefined ? h('span', { class: 'muted' }, text) : typeof value === 'string' && /[\\/]/.test(value) ? h('span', { class: 'mono' }, text) : text;
}

// ───────── jobs view ─────────

/**
 * Main area for #/runs/analysis and #/runs/analysis/<jobId>.
 * @param {HTMLElement} host
 * @param {{jobId?: string|null}} options
 * @returns {{setJob(id: string|null): void, destroy(): void}}
 */
export function mountAnalysisView(host, { jobId = null } = {}) {
  let selected = jobId ?? null;
  let jobs = [];
  let phase = 'loading';
  let listError = '';
  let timer = 0;
  let destroyed = false;
  let controller = null;
  let detail = null;

  const newButton = h('button', { class: 'btn btn--primary', type: 'button', onclick: () => openAnalysisDialog({}) }, 'New analysis job');
  const listStatus = h('div', { class: 'analysis__problem', role: 'status' }, loading('Reading analysis jobs…'));
  const list = h('ol', { class: 'analysis__list', 'aria-label': 'Analysis jobs' });
  const empty = h(
    'div',
    { class: 'analysis__empty', hidden: true },
    emptyState({
      title: 'No analysis jobs yet',
      text: h(
        'div',
        { class: 'stack' },
        Object.keys(KINDS).map((key) => h('p', null, `${KINDS[key].label}: ${KINDS[key].text}`)),
        h('p', null, "Start one from a run's Files tab (with that run's recipe and checkpoint), from the Recipe screen, or with New analysis job."),
      ),
    }),
  );
  const detailHost = h('div', { class: 'analysis__detail' });
  const root = h(
    'section',
    { class: 'analysis', 'aria-label': 'Analysis jobs' },
    h(
      'header',
      { class: 'analysis__head' },
      h('div', { class: 'analysis__head-text' }, h('h1', { class: 'analysis__title' }, 'Analysis jobs'), h('p', { class: 'analysis__lead' }, VIEW_LEAD)),
      newButton,
    ),
    h('div', { class: 'analysis__body' }, h('div', { class: 'analysis__lists' }, listStatus, list, empty), detailHost),
  );
  host.appendChild(root);

  const rows = new Map();
  const rowSig = (job) => JSON.stringify([job.kind, job.status, job.createdAt, job.params, job.id === selected]);
  const renderRow = (job) => {
    const st = statusOf(job.status);
    const what = whatLine(job);
    const current = job.id === selected;
    return h(
      'li',
      { class: `analysis__row${current ? ' analysis__row--current' : ''}`, dataset: { job: job.id } },
      h(
        'a',
        { class: 'analysis__link', href: jobHref(job.id), 'aria-current': current ? 'page' : null },
        h('span', { class: 'analysis__row-top' }, h('span', { class: 'analysis__kind' }, kindLabel(job.kind), kindDetail(job.kind) ? ' ' : null, kindDetail(job.kind)), badge(st.label, st.kind)),
        what.text ? h('span', { class: 'analysis__what', title: what.title || null }, what.text) : null,
        job.createdAt ? h('span', { class: 'analysis__time' }, timeEl(job.createdAt)) : null,
      ),
    );
  };

  const renderList = () => {
    mount(listStatus, phase === 'loading' ? loading('Reading analysis jobs…') : listError ? notice('err', `The analysis jobs could not be read: ${listError} It is tried again shortly.`) : null);
    empty.hidden = !(phase === 'ready' && jobs.length === 0);
    // Rows are rebuilt only when what they show changed; a focused link moves to its new copy.
    const active = document.activeElement;
    let refocus = null;
    const seen = new Set();
    jobs.forEach((job, index) => {
      seen.add(job.id);
      const sig = rowSig(job);
      let slot = rows.get(job.id);
      if (!slot || slot.sig !== sig) {
        const el = renderRow(job);
        if (slot) {
          if (slot.el.contains(active)) refocus = el;
          slot.el.replaceWith(el);
        }
        slot = { el, sig };
        rows.set(job.id, slot);
      }
      if (list.children[index] !== slot.el) list.insertBefore(slot.el, list.children[index] ?? null);
    });
    for (const [id, slot] of rows) {
      if (!seen.has(id)) {
        slot.el.remove();
        rows.delete(id);
      }
    }
    if (refocus && !root.contains(document.activeElement)) refocus.querySelector('a')?.focus();
  };

  const schedule = () => {
    clearTimeout(timer);
    if (destroyed || document.hidden) return;
    const moving = jobs.some((job) => statusOf(job.status).open);
    timer = setTimeout(poll, listError ? 8000 : moving ? 2500 : 20000);
  };

  const poll = async () => {
    clearTimeout(timer);
    controller?.abort();
    controller = new AbortController();
    const local = controller;
    try {
      const answer = await api.analysisJobs({ signal: local.signal });
      if (local.signal.aborted) return;
      jobs = (Array.isArray(answer) ? answer : []).slice().sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
      phase = 'ready';
      listError = '';
    } catch (error) {
      if (local.signal.aborted || error.name === 'AbortError') return;
      listError = error.message;
      if (phase === 'loading') phase = 'error';
    }
    renderList();
    detail?.listChanged(jobs.find((job) => job.id === selected) ?? null);
    schedule();
  };

  const onVisibility = () => {
    if (document.hidden) clearTimeout(timer);
    else poll();
  };
  document.addEventListener('visibilitychange', onVisibility);

  const showDetail = () => {
    detail?.destroy();
    detail = null;
    if (!selected) {
      mount(
        detailHost,
        h(
          'div',
          { class: 'analysis__intro' },
          h('h2', { class: 'analysis__section-title' }, 'What analysis jobs are'),
          h(
            'p',
            null,
            'Jobs you start on purpose to look at a recipe, a checkpoint, a dataset or saved images: a prediction, an evaluation, an ONNX export, a loss report, a dataset packing job, a weight drift measurement, a weights conversion, a data audit, a split leakage check, a pair alignment check, a validation split, a patch extraction or an image difference.',
          ),
          h('p', null, VIEW_LEAD),
          h('p', null, 'Select a job to see its parameters, its log and its result.'),
        ),
      );
      return;
    }
    detail = mountJobDetail(detailHost, selected, { onChange: () => poll() });
  };

  poll();
  showDetail();

  return {
    setJob(id) {
      const next = id ?? null;
      if (next === selected) return;
      selected = next;
      renderList();
      showDetail();
    },
    destroy() {
      destroyed = true;
      clearTimeout(timer);
      controller?.abort();
      document.removeEventListener('visibilitychange', onVisibility);
      detail?.destroy();
      detail = null;
      root.remove();
    },
  };
}

// ───────── one job ─────────

/** Rows for a pack request's `dataset`: the folder pairs, scale, name template and anything else sent. */
function packParamRows(dataset) {
  const roots = packRoots(dataset);
  const rows = [
    ['Folder pairs', roots.gt.length ? folderPairsTable(roots) : null],
    ['Scale', Number.isFinite(dataset.scale) ? `${fmt.int(dataset.scale)}×` : null],
    ['Name template', typeof dataset.filenameTemplate === 'string' ? h('span', { class: 'mono' }, dataset.filenameTemplate) : h('span', { class: 'muted' }, 'Not set: {} (the same name on both sides)')],
  ];
  for (const key of Object.keys(dataset)) {
    if (['gtRoots', 'lqRoots', 'gt', 'lq', 'scale', 'filenameTemplate'].includes(key)) continue;
    rows.push([keyWords(key), renderValue(dataset[key])]);
  }
  return rows.filter(([, value]) => value !== null);
}

/** Ground truth and low-quality folders, paired by position. */
function folderPairsTable(roots) {
  const count = Math.max(roots.gt.length, roots.lq.length);
  return h(
    'div',
    { class: 'analysis__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Folder pairs' },
    h(
      'table',
      { class: 'table analysis__table' },
      h('thead', null, h('tr', null, h('th', { scope: 'col', class: 'num' }, 'Pair'), h('th', { scope: 'col' }, 'Ground truth folder'), h('th', { scope: 'col' }, 'Low-quality folder'))),
      h(
        'tbody',
        null,
        Array.from({ length: count }, (_, i) =>
          h('tr', null, h('td', { class: 'num' }, String(i + 1)), h('td', { class: 'mono' }, roots.gt[i] ?? ''), h('td', { class: 'mono' }, roots.lq[i] ?? '')),
        ),
      ),
    ),
  );
}

/** A bar for a part kind that counts {current, total} (video indexes); null for any other kind or without both numbers. */
function countBar(job) {
  const p = job?.progress;
  if (job?.kind !== 'index-video-clips' || !isPlainObject(p) || !Number.isFinite(p.current) || !Number.isFinite(p.total) || p.total <= 0) return null;
  const done = Math.min(p.current, p.total);
  return h(
    'div',
    { class: 'progress progress--run', role: 'progressbar', 'aria-label': 'Videos indexed', 'aria-valuemin': '0', 'aria-valuemax': String(p.total), 'aria-valuenow': String(done) },
    h('div', { class: 'progress__bar', style: { width: `${Math.min(100, (done / p.total) * 100)}%` } }),
  );
}

/** Progress a running job carries: {stage, completedPairs, totalPairs, encodedBytes} for packing. */
function progressBlock(progress) {
  const done = progress.completedPairs;
  const total = progress.totalPairs;
  const hasBar = Number.isFinite(done) && Number.isFinite(total) && total > 0;
  const stage = typeof progress.stage === 'string' ? PACK_STAGES[progress.stage] ?? keyWords(progress.stage) : '';
  const known = new Set(['stage', 'completedPairs', 'totalPairs', 'encodedBytes']);
  const rows = [
    ['Stage', stage || null],
    ['Pairs', Number.isFinite(done) ? (Number.isFinite(total) ? `${fmt.int(done)} of ${fmt.int(total)}` : fmt.int(done)) : null],
    ['Copied', bytesEl(progress.encodedBytes)],
    ...Object.keys(progress)
      .filter((key) => !known.has(key))
      .map((key) => [keyWords(key), renderValue(progress[key])]),
  ].filter(([, value]) => value !== null && value !== '');
  return h(
    'div',
    { class: 'analysis__progress' },
    hasBar
      ? h(
          'div',
          {
            class: 'progress progress--run',
            role: 'progressbar',
            'aria-label': 'Pairs copied',
            'aria-valuemin': '0',
            'aria-valuemax': String(total),
            'aria-valuenow': String(Math.min(done, total)),
          },
          h('div', { class: 'progress__bar', style: { width: `${Math.min(100, (done / total) * 100)}%` } }),
        )
      : null,
    rows.length ? h('dl', { class: 'kv analysis__kv' }, rows.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])) : null,
  );
}

/** The `dataset` fragment as YAML lines. Strings that are not plain-safe are written JSON-quoted (valid YAML). */
function fragmentYaml(fragment) {
  const plain = (s) => /^[A-Za-z0-9_./~][A-Za-z0-9_./~-]*$/.test(s) && !/^(?:true|false|yes|no|on|off|null|~|[-+]?\d[\d_.eE+-]*)$/i.test(s);
  const scalar = (v) => (typeof v === 'string' ? (plain(v) ? v : JSON.stringify(v)) : v === null || v === undefined ? 'null' : String(v));
  const lines = [];
  const walk = (value, indent) => {
    const pad = '  '.repeat(indent);
    for (const [key, item] of Object.entries(value)) {
      if (Array.isArray(item)) {
        lines.push(`${pad}${key}:`);
        for (const entry of item) lines.push(`${pad}  - ${isPlainObject(entry) ? JSON.stringify(entry) : scalar(entry)}`);
      } else if (isPlainObject(item)) {
        lines.push(`${pad}${key}:`);
        walk(item, indent + 1);
      } else {
        lines.push(`${pad}${key}: ${scalar(item)}`);
      }
    }
  };
  walk(fragment, 0);
  return `${lines.join('\n')}\n`;
}

function mountJobDetail(host, id, { onChange }) {
  const controller = new AbortController();
  const { signal } = controller;
  const titleId = uid('analysis-job');
  const facts = h('div', { class: 'analysis__facts' }, loading('Reading the job…'));
  const progressSlot = h('div', { class: 'analysis__progress-slot' });
  let progressSig = '';
  const renderProgress = () => {
    const open = statusOf(job.status).open && isPlainObject(job.progress);
    mount(progressSlot, open ? (PARTS[job.kind] ? [countBar(job), partProgress(job.progress)] : progressBlock(job.progress)) : null);
  };
  const logPre = h('pre', { class: 'analysis__log', tabindex: '0', 'aria-label': 'Job log' });
  const logNote = h('p', { class: 'analysis__note', hidden: true }, 'Only the last 16 KB of the log are shown.');
  const logPart = h('section', { class: 'analysis__part', hidden: true }, h('h3', { class: 'analysis__part-title' }, 'Log'), logNote, logPre);
  const resultBody = h('div', { class: 'analysis__result', 'aria-live': 'polite' });
  const resultPart = h('section', { class: 'analysis__part', hidden: true }, h('h3', { class: 'analysis__part-title' }, 'Result'), resultBody);
  const root = h('article', { class: 'analysis__job', 'aria-labelledby': titleId }, facts, resultPart, logPart);
  mount(host, root);

  let job = null;
  let factsSig = '';
  let timer = 0;
  let resultLoaded = false;
  let lastLog = null;
  let compare = null;
  let stageToken = 0;
  let cancelBusy = false;
  /** Module of a part kind once imported (null before, or when it failed), and the cleanup of its result. */
  let partModule = null;
  let partWanted = false;
  let partCleanup = null;

  /** Progress words of a part kind from its own module; nothing while it loads or when it has no such export. */
  const partProgress = (progress) => {
    const info = PARTS[job.kind];
    if (!partModule) {
      if (!partWanted) {
        partWanted = true;
        loadPart(job.kind)
          .then((mod) => {
            if (signal.aborted) return;
            partModule = mod;
            if (job && statusOf(job.status).open) renderProgress();
          })
          .catch(() => {
            // The result view says so when it is needed; progress words are optional.
          });
      }
      return null;
    }
    const words = partModule[info.progress];
    if (typeof words !== 'function') return null;
    let text = '';
    try {
      // The job comes along for parts whose words depend on what was asked (pairs or images, for a split).
      text = words(progress, job);
    } catch {
      return null;
    }
    return typeof text === 'string' && text ? h('p', { class: 'analysis__progress-text' }, text) : null;
  };

  const renderFacts = () => {
    const st = statusOf(job.status);
    const p = job.params ?? {};
    const paramRows = Object.keys(p)
      .filter((key) => key !== 'kind')
      .flatMap((key) => {
        const value = p[key];
        if (key === 'dataset' && job.kind === 'pack-dataset' && isPlainObject(value)) return packParamRows(value);
        // Weight drift: content fingerprints of the two files as the service read them.
        if (key === 'fileVersions' && isPlainObject(value)) {
          const sideWords = (side) =>
            job.kind === 'image-difference' && DIFF_SIDES[side]
              ? `${DIFF_SIDES[side]} file fingerprint`
              : job.kind === 'convert-weights' && side === 'weights'
                ? 'Checkpoint file fingerprint'
                : job.kind === 'network-parameters' && side === 'weights'
                  ? 'Starting weights file fingerprint'
                : side === 'left'
                ? 'Baseline file fingerprint'
                : side === 'right'
                  ? 'Candidate file fingerprint'
                  : `${keyWords(side)} file fingerprint`;
          return Object.keys(value).map((side) => [
            sideWords(side),
            typeof value[side] === 'string' ? h('span', { class: 'mono', title: value[side] }, shortDigest(value[side])) : renderValue(value[side]),
          ]);
        }
        // Image difference: the versions the request pinned are said beside each side, not listed as hashes.
        if (key === 'expectedVersions' && job.kind === 'image-difference' && isPlainObject(value)) return [];
        if (job.kind === 'image-difference' && DIFF_SIDES[key] && typeof value === 'string') {
          const pinned = isPlainObject(p.expectedVersions) && typeof p.expectedVersions[key] === 'string';
          return [[KIND_PARAM_LABELS[job.kind][key], [h('span', { class: 'mono', title: value }, value), pinned ? h('span', { class: 'analysis__pinned' }, ' Chosen as a saved image.') : null]]];
        }
        if (key === 'region' && job.kind === 'image-difference' && isPlainObject(value) && ['x', 'y', 'width', 'height'].every((name) => Number.isFinite(value[name]))) {
          return [[KIND_PARAM_LABELS[job.kind].region, `${fmt.dims(value.width, value.height)} pixels from ${value.x}, ${value.y} (left, top)`]];
        }
        if (job.kind === 'extract-patches' && key === 'split' && typeof value === 'string') return [[KIND_PARAM_LABELS[job.kind].split, h('span', { title: value }, setWords(value))]];
        if (job.kind === 'extract-patches' && (key === 'cropSize' || key === 'stride') && Number.isFinite(value)) return [[KIND_PARAM_LABELS[job.kind][key], fmt.plural(value, 'low quality pixel')]];
        if (job.kind === 'extract-patches' && key === 'scale' && Number.isFinite(value)) return [[KIND_PARAM_LABELS[job.kind].scale, `${fmt.int(value)}×`]];
        if (job.kind === 'convert-weights' && key === 'payload') {
          const named = typeof value === 'string' && value !== '';
          return [[KIND_PARAM_LABELS[job.kind].payload, named ? h('span', { class: 'mono' }, value) : h('span', { class: 'muted' }, "Not set: the file's only one")]];
        }
        // ONNX export: said only when the graph was asked to take other sizes.
        if (job.kind === 'export' && key === 'dynamicSpatial' && typeof value === 'boolean') return value ? [['Other image sizes', 'Accepted']] : [];
        // A link, so that a job that failed (and so has no result page) still leads back to the run's checkpoints.
        if (job.kind === 'average-checkpoints' && key === 'run' && typeof value === 'string' && value) return [[KIND_PARAM_LABELS[job.kind].run, h('a', { class: 'mono', href: runHref(value) }, value)]];
        if (job.kind === 'average-checkpoints' && key === 'task' && typeof value === 'string' && value) return [['Task', taskText(value)]];
        if (job.kind === 'average-checkpoints' && key === 'batchNorm' && typeof value === 'string') {
          return [[KIND_PARAM_LABELS[job.kind].batchNorm, value === 'fromLatest' ? h('span', { title: value }, 'Taken from the latest checkpoint') : h('span', { class: 'mono' }, value)]];
        }
        if (job.kind === 'average-checkpoints' && key === 'role' && typeof value === 'string' && ROLE_WORDS[value]) {
          const words = ROLE_WORDS[value];
          return [[KIND_PARAM_LABELS[job.kind].role, h('span', { title: value }, words.charAt(0).toUpperCase() + words.slice(1))]];
        }
        // The steps, ascending as the service lists them; each file's path stands in the title.
        if (job.kind === 'average-checkpoints' && key === 'inputs' && Array.isArray(value) && value.length && value.every((entry) => isPlainObject(entry) && Number.isFinite(entry.step))) {
          return [
            [
              KIND_PARAM_LABELS[job.kind].inputs,
              h('span', { title: value.map((entry) => (typeof entry.path === 'string' ? entry.path : '')).filter(Boolean).join('\n') || null }, `Iterations ${value.map((entry) => fmt.int(entry.step)).join(', ')}`),
            ],
          ];
        }
        // Paired frame folders are always split by whole scene.
        if (job.kind === 'split-dataset' && key === 'groupBy' && value === 'scene' && String(p.datasetType ?? '').toLowerCase() === SPLIT_VIDEO) {
          return [[KIND_PARAM_LABELS[job.kind].groupBy, h('span', { title: value }, 'Whole scenes')]];
        }
        if (job.kind === 'split-dataset' && key === 'groupBy' && typeof value === 'string') {
          const words =groupWords(p.task)[value] ?? GROUP_WORDS[value] ?? GROUP_WORDS_LABELLED[value];
          if (words) return [[KIND_PARAM_LABELS[job.kind].groupBy, h('span', { title: value }, words)]];
        }
        if (job.kind === 'split-dataset' && key === 'validationFraction' && Number.isFinite(value)) {
          return [[KIND_PARAM_LABELS[job.kind].validationFraction, h('span', { title: String(value) }, shareText(value))]];
        }
        if (job.kind === 'split-dataset' && key === 'seed' && Number.isInteger(value)) return [[KIND_PARAM_LABELS[job.kind].seed, String(value)]];
        // Video indexes: the descale settings one per row; a null linear-light choice means the stored values are used.
        if (job.kind === 'index-video-clips' && key === 'settings' && isPlainObject(value)) {
          const rows = [];
          if (Number.isFinite(value.width) && Number.isFinite(value.height)) rows.push(['Descale size', `${fmt.int(value.width)} × ${fmt.int(value.height)}`]);
          if ('kernel' in value) rows.push(['Descale kernel', renderValue(value.kernel)]);
          if ('border' in value) rows.push(['Descale border', renderValue(value.border)]);
          if ('descale_linear_light' in value) rows.push(['Linear light', value.descale_linear_light === null ? 'Stored values' : value.descale_linear_light === true ? 'On' : value.descale_linear_light === false ? 'Off' : renderValue(value.descale_linear_light)]);
          for (const k of Object.keys(value).filter((k) => !['width', 'height', 'kernel', 'border', 'descale_linear_light'].includes(k))) rows.push([keyWords(k), renderValue(value[k])]);
          return rows;
        }
        // Where an evaluation was measured is one row: the job's own word when it has one, else the request's.
        if (job.kind === 'evaluate' && key === 'device' && typeof value === 'string' && value) return job.device ? [] : [[KIND_PARAM_LABELS[job.kind].device, deviceWords(value)]];
        let shown;
        if (key === 'parameter' && (value === null || value === undefined)) shown = h('span', { class: 'muted' }, 'Not set: the generator output is observed');
        else if (key === 'auditLimit' && value === null) shown = 'Every file';
        else if (key === 'split' && job.kind === 'evaluate' && (value === null || value === undefined)) shown = 'All validation and test datasets';
        else if (typeof value === 'string' && /[\\/]/.test(value)) shown = h('span', { class: 'mono', title: value }, value);
        else shown = renderValue(value);
        if (job.kind === 'pair-registration' && key === 'split' && typeof value === 'string') shown = h('span', { title: value }, setWords(value));
        if (job.kind === 'pair-registration' && key === 'radius' && Number.isFinite(value)) shown = fmt.plural(value, 'low quality pixel');
        return [[KIND_PARAM_LABELS[job.kind]?.[key] ?? PARAM_LABELS[key] ?? keyWords(key), shown]];
      });
    if (job.kind === 'pack-dataset' && !('destination' in p)) paramRows.push(['Destination', h('span', { class: 'muted' }, "Not set: stored in this job's own folder")]);
    const rows = [
      ['Created', timeEl(job.createdAt)],
      ['Finished', timeEl(job.finishedAt)],
      [job.kind === 'evaluate' ? 'Measured on' : 'Device', job.device ? deviceWords(job.device) : null],
      ...paramRows,
      ['Request fingerprint', job.requestDigest ? h('span', { class: 'mono', title: job.requestDigest }, shortDigest(job.requestDigest)) : null],
      ['Job id', h('span', { class: 'mono' }, job.id)],
    ].filter(([, value]) => value !== null && value !== undefined && value !== '');

    progressSig = JSON.stringify([statusOf(job.status).open, job.progress ?? null]);
    renderProgress();
    const hadFocus = facts.contains(document.activeElement);
    const cancel =
      job.status === 'running'
        ? h('button', { class: 'btn btn--danger analysis__btn', type: 'button', 'aria-disabled': cancelBusy ? 'true' : null, dataset: { focus: 'cancel' }, onclick: () => cancelJob() }, 'Cancel job')
        : null;
    mount(
      facts,
      h(
        'header',
        { class: 'analysis__job-head' },
        h('h2', { class: 'analysis__job-title', id: titleId, tabindex: '-1' }, kindLabel(job.kind)),
        kindDetail(job.kind),
        badge(st.label, st.kind),
        cancel,
      ),
      job.status === 'cancelling' ? h('p', { class: 'analysis__note' }, 'Cancel was requested. The worker is stopping.') : null,
      progressSlot,
      job.status === 'failed' && job.error ? notice('err', job.error) : null,
      // The way on from the one refusal a person can lift: statistics that differ between the checkpoints.
      job.status === 'failed' && job.kind === 'average-checkpoints' && job.params?.batchNorm !== 'fromLatest' && /buffer\b.*\bdiffers/i.test(String(job.error ?? ''))
        ? h('p', { class: 'analysis__note' }, `If this network uses batch normalisation, average again from the run's Files tab with "${AVERAGE_BN_LABEL}" ticked.`)
        : null,
      h('dl', { class: 'kv analysis__kv' }, rows.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])),
    );
    if (hadFocus && !facts.contains(document.activeElement)) (facts.querySelector('[data-focus="cancel"]') || facts.querySelector(`#${titleId}`))?.focus();
    // A job just started from the dialog: its page takes the focus once. The button that opened the dialog is
    // often on a page this one replaced, and focus would otherwise fall back to the top of the document.
    if (startedJobId && String(job.id ?? '') === startedJobId) {
      startedJobId = null;
      const active = document.activeElement;
      const typing = active && /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName);
      if (!document.querySelector('dialog[open]') && !typing) facts.querySelector(`#${titleId}`)?.focus({ preventScroll: true });
    }
  };

  const cancelJob = async () => {
    if (cancelBusy || !job) return;
    const ok = await confirmDialog({
      title: `Cancel ${kindNoun(job.kind)}?`,
      body:
        job.kind === 'pack-dataset'
          ? `${PACK_STOP} The job ends as cancelled; start a new job to pack again.`
          : 'The worker is stopped and the job ends as cancelled. It cannot be continued; start a new job to run it again.',
      confirmLabel: 'Cancel job',
      cancelLabel: 'Keep running',
      danger: true,
    });
    if (!ok) return;
    cancelBusy = true;
    try {
      job = await api.analysisCancel(id);
      toast(`Cancel requested for ${kindNoun(job.kind)}.`, { kind: 'ok' });
    } catch (error) {
      toast(`The job was not cancelled: ${error.message}`, { kind: 'err' });
    } finally {
      cancelBusy = false;
      factsSig = '';
      refresh();
      onChange?.();
    }
  };

  const loadLog = async () => {
    try {
      const log = await api.analysisLog(id, { signal });
      if (signal.aborted) return;
      logPart.hidden = false;
      const text = typeof log?.text === 'string' ? log.text : '';
      logNote.hidden = !log?.truncated;
      if (text !== lastLog) {
        const atEnd = logPre.scrollHeight - logPre.scrollTop - logPre.clientHeight < 24;
        lastLog = text;
        logPre.textContent = text || 'The log is empty.';
        if (atEnd) logPre.scrollTop = logPre.scrollHeight;
      }
    } catch (error) {
      if (signal.aborted || error.name === 'AbortError') return;
      logPart.hidden = false;
      if (lastLog === null) logPre.textContent = error.message;
    }
  };

  const showOnStage = async (stageHost, sources) => {
    const token = ++stageToken;
    try {
      const mod = await import('../components/compare.js');
      if (signal.aborted || token !== stageToken) return;
      if (!compare) compare = mod.createCompare(stageHost, { modes: ['wipe', 'side', 'blink'] });
      compare.setSources(sources);
    } catch (error) {
      if (signal.aborted) return;
      mount(stageHost, notice('err', `The comparison stage could not be loaded: ${error.message}`));
    }
  };

  const artifactLink = (artifact) => h('a', { href: artifact.url, download: baseName(artifact.path) || 'output', class: 'mono' }, artifact.path);
  /** "Copy path" for a file the service located on the trainer machine (`absolutePath`); nothing otherwise. */
  const copyPathButton = (artifact) => {
    const path = artifact?.absolutePath;
    if (artifact?.available === false || typeof path !== 'string' || !path) return null;
    return h(
      'button',
      {
        class: 'btn btn--sm btn--quiet',
        type: 'button',
        title: path,
        onclick: async () => {
          const ok = await copyText(path);
          // The path is not on the page, so a failed copy says it.
          toast(ok ? 'Copied the path.' : `Copying failed. The path is ${path}`, { kind: ok ? 'ok' : 'err' });
        },
      },
      'Copy path',
      h('span', { class: 'visually-hidden' }, ` of ${baseName(artifact.path) || 'the file'}`),
    );
  };
  /** One entry of a list of files: the download link and, where known, "Copy path". */
  const fileItem = (artifact) => h('li', { class: 'analysis__file' }, artifactLink(artifact), copyPathButton(artifact));

  const copyButton = (label, value, done) =>
    h(
      'button',
      {
        class: 'btn btn--sm',
        type: 'button',
        onclick: async () => {
          const ok = await copyText(value);
          toast(ok ? done : 'Copying failed. Select the text and copy it by hand.', { kind: ok ? 'ok' : 'err' });
        },
      },
      label,
    );
  const pathRow = (value) => (typeof value === 'string' && value ? h('span', { class: 'analysis__path-row' }, h('span', { class: 'mono' }, value), copyButton('Copy path', value, 'Copied the path.')) : null);

  /** Dataset packing: totals, the two databases, the source folders and the recipe lines. */
  const renderPackResult = (answer) => {
    const r = isPlainObject(answer?.result) ? answer.result : {};
    const artifacts = Array.isArray(answer?.artifacts) ? answer.artifacts : [];
    const shown = new Set(['destination', 'gtPath', 'lqPath', 'inventoryPath', 'pairs', 'encodedBytes', 'scale', 'phase', 'dataset', 'directorySyncWarning', 'outputCount']);
    const rows = [
      ['Pairs', Number.isFinite(r.pairs) ? fmt.int(r.pairs) : null],
      ['Size of the copy', bytesEl(r.encodedBytes)],
      ['Scale', Number.isFinite(r.scale) ? `${fmt.int(r.scale)}×` : null],
      ['Phase', typeof r.phase === 'string' ? r.phase : null],
      ['Dataset folder', pathRow(r.destination)],
      ['Ground truth database', pathRow(r.gtPath)],
      ['Low-quality database', pathRow(r.lqPath)],
      ['Inventory file', pathRow(r.inventoryPath)],
      ...Object.keys(r)
        .filter((key) => !shown.has(key))
        .map((key) => [keyWords(key), renderValue(r[key])]),
    ].filter(([, value]) => value !== null && value !== '');
    const fragment = isPlainObject(r.dataset) ? fragmentYaml(r.dataset) : '';
    const roots = packRoots(job.params?.dataset);
    mount(
      resultBody,
      r.directorySyncWarning ? notice('warn', `The dataset was published, but the folder holding it could not be synced to disk: ${r.directorySyncWarning}`) : null,
      h('dl', { class: 'kv analysis__kv' }, rows.map(([label, value]) => [h('dt', null, label), h('dd', null, value)])),
      roots.gt.length
        ? [
            h('h4', { class: 'analysis__sub-title' }, 'Source folders'),
            folderPairsTable(roots),
            h('p', { class: 'analysis__note' }, 'Per-folder counts and a record of every packed file (size, checksum, dimensions) are in the inventory file.'),
          ]
        : null,
      fragment
        ? [
            h('div', { class: 'analysis__part-head' }, h('h4', { class: 'analysis__sub-title' }, 'Recipe lines'), copyButton('Copy', fragment, 'Copied the recipe lines.')),
            h('p', null, 'Paste these lines into the dataset section of a recipe, in place of dataroot_gt, dataroot_lq and io_backend.'),
            h('div', { class: 'analysis__yaml', tabindex: '0', role: 'region', 'aria-label': 'Recipe lines' }, highlightedPre(fragment)),
            h('p', { class: 'analysis__note' }, "Written as YAML from the job's dataset fragment."),
          ]
        : null,
      artifacts.length ? [h('h4', { class: 'analysis__sub-title' }, 'Outputs'), h('ul', { class: 'analysis__files' }, artifacts.map(fileItem))] : null,
    );
  };

  /** Weight drift and data audit: drawn by their own module; when it is missing or throws, the files stay downloadable. */
  const renderPartResult = async (answer) => {
    const info = PARTS[job.kind];
    const result = answer?.result;
    const artifacts = Array.isArray(answer?.artifacts) ? answer.artifacts : [];
    const partHost = h('div', { class: 'analysis__part-result' });
    const fail = (error) => {
      partCleanup = null;
      mount(
        resultBody,
        notice('err', `The result could not be shown here: ${error?.message || String(error)}`),
        artifacts.length
          ? [h('h4', { class: 'analysis__sub-title' }, 'Outputs'), h('ul', { class: 'analysis__files' }, artifacts.map(fileItem))]
          : h('p', { class: 'muted' }, 'This job saved no output files.'),
      );
    };
    mount(resultBody, partHost);
    let mod;
    try {
      mod = await loadPart(job.kind);
    } catch (error) {
      if (!signal.aborted) fail(new Error(`${info.file} could not be loaded (${error?.message || error}).`));
      return;
    }
    if (signal.aborted) return;
    try {
      if (typeof mod[info.render] !== 'function') throw new Error(`${info.file} has no ${info.render}.`);
      const cleanup = mod[info.render](partHost, { job, result, artifacts });
      partCleanup = typeof cleanup === 'function' ? cleanup : null;
    } catch (error) {
      fail(error);
    }
  };

  /** A kind this module does not know: its files as download links, the result as JSON in a closed <details>. */
  const renderUnknownResult = (answer) => {
    const result = answer?.result;
    let next = answer?.next ?? null;
    const files = h('ul', { class: 'analysis__files' });
    const add = (items) => {
      for (const a of items) files.appendChild(fileItem(a));
    };
    const more = h('button', { class: 'btn btn--sm analysis__btn', type: 'button', hidden: next === null }, 'Show more outputs');
    more.addEventListener('click', async () => {
      if (next === null) return;
      more.setAttribute('aria-disabled', 'true');
      try {
        const page = await api.analysisResult(id, { offset: next, limit: RESULT_PAGE }, { signal });
        if (signal.aborted) return;
        add(Array.isArray(page?.artifacts) ? page.artifacts : []);
        next = page?.next ?? null;
        more.hidden = next === null;
      } catch (error) {
        if (signal.aborted || error.name === 'AbortError') return;
        toast(`More outputs could not be read: ${error.message}`, { kind: 'err' });
      } finally {
        more.removeAttribute('aria-disabled');
      }
    });
    const artifacts = Array.isArray(answer?.artifacts) ? answer.artifacts : [];
    add(artifacts);
    mount(
      resultBody,
      h('p', { class: 'analysis__note' }, 'This screen does not know this kind of job yet, so its result is shown as the service wrote it.'),
      h('h4', { class: 'analysis__sub-title' }, 'Outputs'),
      artifacts.length ? files : h('p', { class: 'muted' }, 'This job saved no output files.'),
      more,
      result !== undefined
        ? h('details', { class: 'analysis__raw' }, h('summary', null, 'Result as JSON'), h('pre', { class: 'analysis__log', tabindex: '0' }, JSON.stringify(result, null, 2)))
        : null,
    );
  };

  const renderResult = (answer) => {
    if (job.kind === 'pack-dataset') {
      renderPackResult(answer);
      return;
    }
    if (PARTS[job.kind]) {
      renderPartResult(answer);
      return;
    }
    if (!KINDS[job.kind]) {
      renderUnknownResult(answer);
      return;
    }
    const result = answer?.result;
    const artifacts = Array.isArray(answer?.artifacts) ? [...answer.artifacts] : [];
    let next = answer?.next ?? null;
    const p = job.params ?? {};
    // Frame sequences (traiNNer/tasks/restoration_predict.py predict_sequence) and autoencoder reconstructions.
    const frames = job.kind === 'predict' && isPlainObject(result) && ['inputFrames', 'centerFrames', 'boundaryFramesOmitted', 'clipSize', 'scenes'].some((key) => key in result);
    const reconstruction = job.kind === 'predict' && isPlainObject(result) && result.task === 'autoencoder';
    const outWord = reconstruction ? 'Reconstruction' : 'Prediction';
    const shownResult = frames ? Object.fromEntries(Object.entries(result).filter(([key]) => !['inputFrames', 'centerFrames', 'boundaryFramesOmitted', 'clipSize', 'scenes'].includes(key))) : result;
    const singlePrediction = job.kind === 'predict' && artifacts.length === 1 && next === null && IMAGE_RE.test(artifacts[0].path) && !/[\\/]/.test(artifacts[0].path);
    const images = artifacts.filter((a) => IMAGE_RE.test(a.path));
    const stageHost = h('div', { class: 'analysis__stage' });
    const outputs = h('div', { class: 'analysis__outputs' });
    const more = h('button', { class: 'btn btn--sm analysis__btn', type: 'button', hidden: next === null }, 'Show more outputs');

    const addArtifacts = (items) => {
      const grid = [];
      const files = [];
      for (const artifact of items) {
        if (!singlePrediction && IMAGE_RE.test(artifact.path)) {
          grid.push(
            h(
              'li',
              { class: 'analysis__thumb' },
              h(
                'button',
                {
                  class: 'analysis__thumb-btn',
                  type: 'button',
                  title: artifact.path,
                  onclick: () => {
                    stageHost.hidden = false;
                    showOnStage(stageHost, { a: { label: artifact.path, role: 'out', load: async () => artifact.url }, b: null });
                  },
                },
                // Fixed box (width/height plus CSS aspect-ratio) so lazy images do not shift the grid.
                h('img', { src: sizedUrl(artifact.url, THUMB_SIZE), alt: artifact.path, loading: 'lazy', decoding: 'async', width: String(THUMB_SIZE), height: String(THUMB_SIZE), class: 'analysis__thumb-img' }),
                h('span', { class: 'analysis__thumb-name' }, baseName(artifact.path)),
              ),
              artifactLink({ ...artifact, path: 'Download' }),
            ),
          );
        } else {
          files.push(fileItem(artifact));
        }
      }
      if (grid.length) outputs.appendChild(h('ul', { class: 'analysis__grid' }, grid));
      if (files.length) outputs.appendChild(h('ul', { class: 'analysis__files' }, files));
    };

    more.addEventListener('click', async () => {
      if (next === null) return;
      more.setAttribute('aria-disabled', 'true');
      try {
        const page = await api.analysisResult(id, { offset: next, limit: RESULT_PAGE }, { signal });
        if (signal.aborted) return;
        addArtifacts(Array.isArray(page?.artifacts) ? page.artifacts : []);
        next = page?.next ?? null;
        more.hidden = next === null;
      } catch (error) {
        if (signal.aborted || error.name === 'AbortError') return;
        toast(`More outputs could not be read: ${error.message}`, { kind: 'err' });
      } finally {
        more.removeAttribute('aria-disabled');
      }
    });

    // Classification: one JSON file with the predicted class and raw logits per input.
    // Detection writes the same file name with boxes per image (traiNNer/tasks/object_detection/inference.py).
    const predictionsFile = job.kind === 'predict' ? artifacts.find((a) => baseName(a.path).toLowerCase() === 'predictions.json') ?? null : null;
    const detection = isPlainObject(result) && (result.task === 'object_detection' || result.output_kind === 'detections');
    const classHost = predictionsFile
      ? h('div', { class: 'analysis__classes', 'aria-live': 'polite' }, loading(detection ? 'Reading the detected boxes…' : 'Reading the predicted classes…'))
      : null;

    mount(
      resultBody,
      result !== undefined ? renderValue(shownResult) : null,
      frames ? framesBlock(result) : null,
      classHost ? h('h4', { class: 'analysis__sub-title' }, detection ? 'Detected boxes' : 'Predicted classes') : null,
      classHost,
      h('h4', { class: 'analysis__sub-title' }, 'Outputs'),
      artifacts.length ? null : h('p', { class: 'muted' }, 'This job saved no output files.'),
      singlePrediction ? stageHost : null,
      singlePrediction ? h('p', { class: 'analysis__note' }, `Input: a preview of ${baseName(p.input)}, at most ${fmt.int(PREVIEW_MAX)} px on the long edge. ${reconstruction ? 'Reconstruction: the saved output file, the same size as the input.' : 'Prediction: the saved output file.'}`) : null,
      outputs,
      more,
      !singlePrediction && images.length ? stageHost : null,
    );
    addArtifacts(artifacts);
    if (predictionsFile) loadPredictions(classHost, predictionsFile, p);
    if (singlePrediction) {
      const out = artifacts[0];
      showOnStage(stageHost, {
        a: p.input ? { label: 'Input', role: 'lq', load: () => imageUrl(p.input), note: 'Preview' } : null,
        b: { label: outWord, role: 'out', load: async () => out.url },
      });
    } else {
      stageHost.hidden = true;
    }
  };

  /** One image drawn with its boxes: an SVG in the original image's pixel space, so the preview scales with it. */
  const overlayFigure = (row) => {
    const figure = h('figure', { class: 'analysis__boxes-figure' });
    const caption = h(
      'figcaption',
      { class: 'analysis__note', title: row.input },
      `${baseName(row.file_name || row.input)}: a preview, at most ${fmt.int(PREVIEW_MAX)} px on the long edge, with the boxes drawn at the scale of the original ${fmt.dims(row.width, row.height)} image. The table lists the same boxes.`,
    );
    const W = row.width;
    const H = row.height;
    const svg = svgEl('svg', {
      class: 'analysis__boxes-svg',
      viewBox: `0 0 ${W} ${H}`,
      role: 'img',
      'aria-label': `${baseName(row.file_name || row.input)} with ${fmt.plural(row.detections.length, 'box', 'boxes')} drawn over it`,
    });
    const picture = svgEl('image', { width: W, height: H, preserveAspectRatio: 'none' });
    svg.appendChild(picture);
    const unit = Math.max(W, H) / 60; // label size in image pixels, readable at any display size
    row.detections.forEach((d, i) => {
      const [x1, y1, x2, y2] = d.box_xyxy;
      svg.appendChild(svgEl('rect', { class: 'analysis__box', x: x1, y: y1, width: Math.max(0, x2 - x1), height: Math.max(0, y2 - y1), 'vector-effect': 'non-scaling-stroke' }));
      const label = svgEl('text', { class: 'analysis__box-label', x: x1 + unit * 0.2, y: Math.max(unit, y1 - unit * 0.25), 'font-size': unit, 'stroke-width': unit / 5 });
      label.textContent = `${i + 1} ${detectionName(d)} ${scoreText(d.score)}`;
      svg.appendChild(label);
    });
    figure.append(svg, caption);
    imageUrl(row.input)
      .then((src) => {
        if (!signal.aborted) picture.setAttribute('href', src);
      })
      .catch((error) => {
        if (!signal.aborted) figure.replaceWith(notice('err', `The image ${baseName(row.input)} could not be shown: ${error.message} The table lists its boxes.`));
      });
    return figure;
  };

  /** Detection result: the cutoff and limit the job ran with, then per image its boxes as a table (and drawn). */
  const detectionView = ({ rows, scoreThreshold, topLimit }, p, artifact) => {
    const shown = rows.slice(0, MAX_DETECTION_IMAGES);
    const kept = rows.reduce((sum, row) => sum + row.detections.length, 0);
    // job.params says whether the person set a value; the file says what the predictor used.
    const cutoffSet = Number.isFinite(p.scoreThreshold);
    const limitSet = Number.isFinite(p.topLimit);
    const cutoff = cutoffSet ? p.scoreThreshold : scoreThreshold;
    const limit = limitSet ? p.topLimit : topLimit;
    const cutoffText = Number.isFinite(cutoff) ? `boxes scoring above ${String(cutoff)}${cutoffSet ? '' : " (the model's default)"}` : 'a cutoff that was not reported';
    const limitText = Number.isFinite(limit) ? `at most ${fmt.plural(limit, 'box', 'boxes')} per image${limitSet ? '' : ' (the default)'}` : 'a box limit that was not reported';
    const autoDraw = shown.length <= AUTO_OVERLAYS;
    return h(
      'div',
      { class: 'stack' },
      h(
        'p',
        { class: 'analysis__detect-line' },
        `Ran with ${cutoffText} and ${limitText}. Kept ${fmt.plural(kept, 'box', 'boxes')} in ${fmt.plural(rows.length, 'image', 'images')}.`,
      ),
      h('p', { class: 'analysis__note' }, 'Box corners are in pixels of the original image, measured from its top left corner.'),
      rows.length > shown.length
        ? h('p', { class: 'analysis__note' }, `Showing the first ${fmt.int(shown.length)} of ${fmt.int(rows.length)} images. `, artifactLink({ ...artifact, path: 'Download predictions.json' }), ' for all of them.')
        : null,
      h(
        'ol',
        { class: 'analysis__detect-list' },
        shown.map((row) => {
          const name = row.file_name || baseName(row.input);
          const drawHost = h('div', { class: 'analysis__boxes-host' });
          if (row.detections.length && Number.isFinite(row.width) && Number.isFinite(row.height) && row.width > 0 && row.height > 0) {
            if (autoDraw) drawHost.appendChild(overlayFigure(row));
            else {
              const show = h(
                'button',
                {
                  class: 'btn btn--sm',
                  type: 'button',
                  onclick: () => {
                    const fig = overlayFigure(row);
                    mount(drawHost, fig);
                    fig.querySelector('svg')?.setAttribute('tabindex', '-1');
                    fig.querySelector('svg')?.focus();
                  },
                },
                `Show ${name} with its boxes`,
              );
              drawHost.appendChild(show);
            }
          }
          return h(
            'li',
            { class: 'analysis__detect-image' },
            h(
              'h5',
              { class: 'analysis__item-title', title: row.input },
              name,
              Number.isFinite(row.width) && Number.isFinite(row.height) ? h('span', { class: 'analysis__detect-dims' }, ` ${fmt.dims(row.width, row.height)} px, ${fmt.plural(row.detections.length, 'box', 'boxes')} kept`) : null,
            ),
            row.detections.length ? [drawHost, boxesTable(row)] : h('p', { class: 'muted' }, 'No boxes above the cutoff.'),
          );
        }),
      ),
    );
  };

  /** Fetch the predictions file (a plain GET of its artifact address) and show it. */
  const loadPredictions = async (classHost, artifact, p) => {
    const fallback = (sentence) => mount(classHost, h('p', null, sentence, ' ', artifactLink({ ...artifact, path: 'Download predictions.json' })));
    let raw;
    try {
      const response = await fetch(artifact.url, { signal });
      if (!response.ok) {
        fallback(`The predictions file could not be read here: the service answered ${response.status}.`);
        return;
      }
      raw = await response.text();
    } catch (error) {
      if (signal.aborted || error.name === 'AbortError') return;
      fallback('The predictions file could not be read here: the trainer service did not answer.');
      return;
    }
    if (signal.aborted) return;
    let doc;
    try {
      doc = JSON.parse(raw);
    } catch {
      fallback('The predictions file is not valid JSON, so it is not shown here.');
      return;
    }
    const boxes = readDetections(doc);
    if (boxes) {
      mount(classHost, detectionView(boxes, p, artifact));
      return;
    }
    const parsed = readPredictions(doc);
    if (!parsed) {
      fallback('The predictions file does not have the expected shape, so it is not shown here.');
      return;
    }
    const { rows, classNames } = parsed;
    if (rows.length === 1 && p.input && rows[0].input === p.input) {
      // One input file: the image beside its predicted class.
      const row = rows[0];
      const img = h('img', { class: 'analysis__input-img', alt: `Input ${baseName(row.input)}` });
      imageUrl(row.input)
        .then((src) => {
          if (!signal.aborted) img.src = src;
        })
        .catch((error) => {
          if (!signal.aborted) img.replaceWith(notice('err', `The input image could not be shown: ${error.message}`));
        });
      const name = (i) => classNames?.[i] ?? `class ${i}`;
      mount(
        classHost,
        h(
          'div',
          { class: 'analysis__single-class' },
          h(
            'figure',
            { class: 'analysis__input' },
            img,
            h('figcaption', { class: 'analysis__note', title: row.input }, `${baseName(row.input)}: a preview, at most ${fmt.int(PREVIEW_MAX)} px on the long edge.`),
          ),
          h(
            'div',
            { class: 'stack' },
            h('dl', { class: 'kv analysis__kv' }, h('dt', null, 'Predicted class'), h('dd', null, h('strong', null, `${row.class_name ?? name(row.class_id)} (ID ${row.class_id})`))),
            h(
              'div',
              { class: 'analysis__table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Raw logits' },
              h(
                'table',
                { class: 'table analysis__table' },
                h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Class'), h('th', { scope: 'col', class: 'num' }, 'Raw logit'))),
                h(
                  'tbody',
                  null,
                  row.logits.map((value, i) => h('tr', null, h('td', null, `${name(i)} (ID ${i})`), h('td', { class: 'num mono' }, String(value)))),
                ),
              ),
            ),
          ),
        ),
      );
      return;
    }
    mount(classHost, predictionsTable(parsed));
  };

  const loadResult = async () => {
    resultPart.hidden = false;
    mount(resultBody, loading('Reading the result…'));
    try {
      const answer = await api.analysisResult(id, { offset: 0, limit: RESULT_PAGE }, { signal });
      if (signal.aborted) return;
      // Task ids are shown in words from the catalog when it loads; the id otherwise.
      await catalog.load().catch(() => null);
      if (signal.aborted) return;
      renderResult(answer);
    } catch (error) {
      if (signal.aborted || error.name === 'AbortError') return;
      mount(resultBody, notice('err', `The result could not be read: ${error.message}`, h('button', { class: 'btn btn--sm', type: 'button', onclick: () => loadResult() }, 'Try again')));
    }
  };

  const apply = (next) => {
    job = next;
    const sig = JSON.stringify([job.status, job.finishedAt, job.error, job.params, cancelBusy]);
    const psig = JSON.stringify([statusOf(job.status).open, job.progress ?? null]);
    if (sig === factsSig && psig !== progressSig) {
      progressSig = psig;
      renderProgress();
    }
    if (sig !== factsSig) {
      factsSig = sig;
      renderFacts();
    }
    if (job.status === 'completed' && !resultLoaded) {
      resultLoaded = true;
      loadResult();
    }
  };

  const schedule = () => {
    clearTimeout(timer);
    if (signal.aborted || document.hidden || !job || !statusOf(job.status).open) return;
    timer = setTimeout(refresh, 2500);
  };

  const refresh = async () => {
    clearTimeout(timer);
    try {
      const next = await api.analysisJob(id, { signal });
      if (signal.aborted) return;
      const wasOpen = job ? statusOf(job.status).open : true;
      apply(next);
      // Log: while it runs, and once more when it ends so the last lines are shown.
      if (wasOpen || lastLog === null) await loadLog();
      if (wasOpen && !statusOf(job.status).open) onChange?.();
    } catch (error) {
      if (signal.aborted || error.name === 'AbortError') return;
      if (error.status === 404 && !job) {
        mount(
          facts,
          h('header', { class: 'analysis__job-head' }, h('h2', { class: 'analysis__job-title', id: titleId }, 'Analysis job')),
          notice('err', error.message, h('a', { class: 'btn btn--sm', href: jobHref(null) }, 'All analysis jobs')),
        );
        return;
      }
      if (!job) mount(facts, notice('err', `The job could not be read: ${error.message}`));
      else toast(`The job could not be refreshed: ${error.message}`, { kind: 'err' });
    }
    schedule();
  };

  const onVisibility = () => {
    if (document.hidden) clearTimeout(timer);
    else if (job && statusOf(job.status).open) refresh();
  };
  document.addEventListener('visibilitychange', onVisibility);
  refresh();

  return {
    /** The list poll saw this job: catch a status change between this view's own polls. */
    listChanged(listed) {
      if (!listed || !job || listed.status === job.status) return;
      refresh();
    },
    destroy() {
      controller.abort();
      clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
      stageToken += 1;
      compare?.destroy();
      compare = null;
      try {
        partCleanup?.();
      } catch {
        // a part's own cleanup failing must not keep this view alive
      }
      partCleanup = null;
      root.remove();
    },
  };
}

// ───────── start dialog ─────────

/** `error`: an element from errorSlot(), shown under the help and read with the control. */
function field({ label, id, help = '', optional = false, control, extra = [], error = null }) {
  const helpId = help ? `${id}-help` : null;
  const described = [helpId, error?.id].filter(Boolean).join(' ');
  if (described) control.setAttribute('aria-describedby', described);
  return h(
    'div',
    { class: 'field analysis__field' },
    h('label', { class: 'field__label', for: id }, label, optional ? h('span', { class: 'analysis__optional' }, ' (optional)') : null),
    h('div', { class: 'analysis__field-row' }, control, extra),
    help ? h('p', { class: 'field__help', id: helpId }, help) : null,
    error,
  );
}

/** The place under a field where its error is written, in words. Empty and hidden until there is one. */
const errorSlot = (id) => h('p', { class: 'field__error', id, hidden: true });
/** Write `message` into a field's error slot and mark the controls it is about; '' clears both. */
function fieldError(slot, controls, message) {
  slot.textContent = message;
  slot.hidden = !message;
  for (const control of controls) {
    if (message) control.setAttribute('aria-invalid', 'true');
    else control.removeAttribute('aria-invalid');
  }
}

/**
 * Start an analysis job. Any field may be preset by the caller.
 * @param {{kind?: 'predict'|'evaluate'|'export'|'loss-report'|'pack-dataset'|'weight-drift'|'task-data'|'split-leakage'|'pair-registration'|'split-dataset'|'extract-patches'|'image-difference'|'convert-weights'|'average-checkpoints'|'network-parameters'|null,
 *   yaml?: string, weights?: string, input?: string, sourceLabel?: string,
 *   left?: string|object, right?: string|object, reference?: string|object,
 *   dataset?: {gtRoots?: string[], lqRoots?: string[], scale?: number|null, filenameTemplate?: string|null}|null}} [options]
 *   `dataset` presets a dataset packing job (and selects that kind when `kind` is not given); `left` and
 *   `right` as strings preset the baseline and candidate of a weight drift measurement, or the first and
 *   second image of an image difference. For an image difference each of `left`, `right` and `reference`
 *   may instead be a saved image, `{run|job|analysis: id, path, version, label}`: it is sent as the service
 *   takes it (the owner key, `path` and `version`); `label` is only shown. With `kind: 'convert-weights'`,
 *   `weights` presets the .pth file to convert.
 * @returns {Promise<object|null>} the created job, or null when cancelled
 */
/**
 * `weightsVersion`: the version of the `weights` file as the checkpoint ranking reported it. It travels with
 * the start request only while the weights field still names that file, so the service can refuse a file
 * that was replaced since it was chosen.
 */
/**
 * `average`: the checkpoints a checkpoint average can take, `{run: {id, name}, role, checkpoints: [{step, path,
 * version|null}], selected: [step]}`. With it the kind picker is shown with "Checkpoint average" chosen; without
 * it that kind is not offered, since it needs a run.
 */
export async function openAnalysisDialog({ kind = null, yaml = '', weights = '', weightsVersion = null, input = '', sourceLabel = '', dataset = null, left = '', right = '', reference = '', average = null } = {}) {
  // One request id per body: pressing Start again with the same body returns the same job (no second
  // start); any change to the body takes a new id, so it is not refused as "another analysis".
  let requestId = newRequestId();
  let lastSent = null; // {key, body, exact}
  // A checkpoint average needs a run and its checkpoints: asked for without them, the dialog opens as a bare one.
  const averageGiven = isPlainObject(average) && typeof average.run?.id === 'string' && average.run.id !== '' && typeof average.role === 'string' && Array.isArray(average.checkpoints);
  if (averageGiven && !kind) kind = 'average-checkpoints';
  if (kind === 'average-checkpoints' && !averageGiven) kind = null;
  const packPreset = isPlainObject(dataset) && !kind;
  let chosen = KINDS[kind] ? kind : packPreset ? 'pack-dataset' : 'predict';
  // A checkpoint average is always opened for one run's checkpoints, so it is a preset like the other file actions.
  const presetKind = Boolean(KINDS[kind]) || packPreset;
  const needsRecipe = () => !KINDS[chosen].noRecipe;
  const needsWeights = () => needsRecipe() && !KINDS[chosen].noWeights;

  // The recipe text: the caller's, else the open recipe.
  const r = recipe.get();
  const text = yaml || (r.phase === 'open' ? r.yaml : '');
  const label = yaml ? sourceLabel || 'the recipe given' : r.phase === 'open' ? `the open recipe${r.summary?.name || r.name ? ` (${r.summary?.name || r.name})` : ''}` : '';

  const ids = { weights: uid('an-weights'), input: uid('an-input'), height: uid('an-h'), width: uid('an-w'), opset: uid('an-opset'), batches: uid('an-batches'), step: uid('an-step'), parameter: uid('an-param'), disc: uid('an-disc') };
  const input_ = (id, value = '', extra = {}) => h('input', { class: 'input', id, type: 'text', value, autocomplete: 'off', spellcheck: 'false', ...extra });
  const weightsInput = input_(ids.weights, weights, { class: 'input input--path analysis__grow' });
  const inputInput = input_(ids.input, input, { class: 'input input--path analysis__grow' });
  const heightInput = input_(ids.height, '', { class: 'input input--num', inputmode: 'numeric' });
  const widthInput = input_(ids.width, '', { class: 'input input--num', inputmode: 'numeric' });
  const opsetInput = input_(ids.opset, '', { class: 'input input--num', inputmode: 'numeric', placeholder: '17' });
  // Export, when the service offers them for the recipe: a graph that also takes other image sizes
  // (actions.export.dynamicSpatial), and the window of a graph that takes frame sequences.
  const exportIds = { anySize: uid('an-any-size'), clip: uid('an-export-clip') };
  const anySizeBox = h('input', { type: 'checkbox', id: exportIds.anySize, 'aria-describedby': `${exportIds.anySize}-help` });
  const anySizeField = h(
    'div',
    { class: 'field analysis__field', hidden: true },
    h('div', { class: 'analysis__check' }, anySizeBox, h('label', { for: exportIds.anySize }, 'Accept other image sizes')),
    h('p', { class: 'field__help', id: `${exportIds.anySize}-help` }, EXPORT_ANY_SIZE_HELP),
  );
  const exportClipInput = input_(exportIds.clip, '', { class: 'input input--num', inputmode: 'numeric' });
  const exportClipError = errorSlot(`${exportIds.clip}-error`);
  const exportClipField = h(
    'div',
    { hidden: true },
    field({
      label: 'Frames per window',
      id: exportIds.clip,
      optional: true,
      help: 'Leave empty when the checkpoint fixes the window. The service says so when it needs the number.',
      control: exportClipInput,
      error: exportClipError,
    }),
  );
  const batchesInput = input_(ids.batches, '', { class: 'input input--num', inputmode: 'numeric', placeholder: '1' });
  const stepInput = input_(ids.step, '', { class: 'input input--num', inputmode: 'numeric', placeholder: '1' });
  const parameterInput = input_(ids.parameter, '', { class: 'input analysis__grow' });
  const dialogController = new AbortController();
  const sizeNote = h('p', { class: 'field__help', role: 'status', hidden: true }, 'Filled from task_options.image_size in the recipe, which this task requires.');
  const discInput = input_(ids.disc, '', { class: 'input input--path analysis__grow' });

  // Dataset packing fields (no recipe, no weights).
  const packIds = { gt: uid('an-gt'), lq: uid('an-lq'), scale: uid('an-scale'), template: uid('an-template'), dest: uid('an-dest') };
  const presetRoots = packRoots(isPlainObject(dataset) ? dataset : null);
  const area = (id, lines) => h('textarea', { class: 'textarea analysis__grow analysis__paths', id, rows: '3', spellcheck: 'false', autocomplete: 'off', value: lines.join('\n') });
  const gtArea = area(packIds.gt, presetRoots.gt);
  const lqArea = area(packIds.lq, presetRoots.lq);
  const scaleInput = input_(packIds.scale, Number.isFinite(dataset?.scale) ? String(dataset.scale) : '', { class: 'input input--num', inputmode: 'numeric' });
  const templateInput = input_(packIds.template, typeof dataset?.filenameTemplate === 'string' && dataset.filenameTemplate !== '{}' ? dataset.filenameTemplate : '', { class: 'input input--path', placeholder: '{}' });
  const destInput = input_(packIds.dest, '', { class: 'input input--path analysis__grow' });
  const lines = (el) => el.value.split('\n').map((line) => line.trim()).filter(Boolean);
  /** "Browse": append a folder as a new line. */
  const appendFolder = (target, title) =>
    h(
      'button',
      {
        class: 'btn btn--sm',
        type: 'button',
        onclick: async () => {
          try {
            const { pickPath } = await import('../components/path-picker.js');
            const existing = lines(target);
            const picked = await pickPath({ title, mode: 'folder', start: existing.length ? folderOf(existing[existing.length - 1]) : '' });
            if (picked) target.value = [...existing, picked].join('\n');
          } catch (error) {
            toast(`The file browser could not be opened: ${error.message}`, { kind: 'err' });
          }
          target.focus();
        },
      },
      'Browse',
    );
  /** Destination "Browse": choose the parent folder; the person types the new folder's name after it. */
  const browseParent = h(
    'button',
    {
      class: 'btn btn--sm',
      type: 'button',
      onclick: async () => {
        try {
          const { pickPath } = await import('../components/path-picker.js');
          const picked = await pickPath({ title: 'Choose the folder that will hold the new dataset folder', mode: 'folder', start: folderOf(destInput.value), confirmLabel: 'Use as parent folder' });
          if (picked) destInput.value = `${picked.replace(/[\\/]+$/, '')}/`;
        } catch (error) {
          toast(`The file browser could not be opened: ${error.message}`, { kind: 'err' });
        }
        destInput.focus();
      },
    },
    'Browse',
  );

  const browse = (target, { mode, title, extensions = null }) =>
    h(
      'button',
      {
        class: 'btn btn--sm',
        type: 'button',
        onclick: async () => {
          try {
            const { pickPath } = await import('../components/path-picker.js');
            const picked = await pickPath({ title, mode, start: folderOf(target.value) || (mode === 'folder' ? target.value : ''), extensions });
            if (picked) target.value = picked;
          } catch (error) {
            toast(`The file browser could not be opened: ${error.message}`, { kind: 'err' });
          }
          target.focus();
        },
      },
      mode === 'folder' ? 'Browse folder' : mode === 'file' && extensions === null ? 'Browse file' : 'Browse',
    );
  const ckpt = ['.safetensors', '.pth'];

  // Weight drift (no recipe): baseline and candidate .safetensors files.
  const driftIds = { left: uid('an-left'), right: uid('an-right') };
  const pathText = (value) => (typeof value === 'string' ? value : '');
  const leftInput = input_(driftIds.left, pathText(left), { class: 'input input--path analysis__grow' });
  const rightInput = input_(driftIds.right, pathText(right), { class: 'input input--path analysis__grow' });
  const drift = ['.safetensors'];

  // Validation split: share of the training pairs (in percent), what stays on the same side, and the seed
  // (studio/analysis_jobs.py: validationFraction strictly between 0 and 1, groupBy pair|parent|scene, an integer seed).
  const splitIds = { share: uid('an-share'), group: uid('an-group'), seed: uid('an-split-seed') };
  const shareInput = input_(splitIds.share, '10', { class: 'input input--num', inputmode: 'decimal' });
  const shareError = errorSlot(`${splitIds.share}-error`);
  const groupSelect = h('select', { class: 'select', id: splitIds.group }, Object.entries(GROUP_WORDS).map(([value, words]) => h('option', { value }, words)));
  const splitNote = h('p', { class: 'analysis__note' }, SPLIT_NOTE);
  // The task whose data would be split, once the recipe has been read (a recipe without one is a restoration
  // recipe); null while that is unknown. Labelled data (classification, segmentation, detection) is grouped
  // by image, not by pair, so the form's words follow the task.
  const splitTask = () => (typeof task === 'string' && task ? task : isPlainObject(recipeConfig) ? recipeConfig.task || 'image_restoration' : null);
  const splitLabelled = () => splitTask() !== null && splitTask() !== 'image_restoration';
  // Paired frame folders (a restoration recipe whose training data are a PairedVideoDataset) are split by whole
  // scene only: there is no choice to make, so a sentence stands where the select would be and no groupBy is sent.
  const splitVideo = () =>
    splitTask() === 'image_restoration' && isPlainObject(recipeConfig?.datasets?.train) && String(recipeConfig.datasets.train.type ?? '').toLowerCase() === SPLIT_VIDEO;
  const groupField = field({ label: 'Keep together', id: splitIds.group, help: SPLIT_GROUP_HELP, control: groupSelect });
  const sceneLine = h('p', { class: 'analysis__note', hidden: true }, SPLIT_SCENES);
  let splitWorded = false;
  /** Written once, while the form is still hidden behind the recipe check: the choices and words for labelled data and for frame folders. */
  const wordSplit = () => {
    if (splitWorded || splitTask() === null) return;
    splitWorded = true;
    const say = (id, words) => {
      const el = groups['split-dataset'].querySelector(`[id="${id}-help"]`);
      if (el) el.textContent = words;
    };
    if (splitVideo()) {
      groupField.hidden = true;
      sceneLine.hidden = false;
      say(splitIds.share, SPLIT_SHARE_HELP_VIDEO);
      splitNote.textContent = SPLIT_NOTE_VIDEO;
      return;
    }
    if (!splitLabelled()) return;
    mount(groupSelect, Object.entries(GROUP_WORDS_LABELLED).map(([value, words]) => h('option', { value, selected: value === 'sample' }, words)));
    groupSelect.value = 'sample';
    say(splitIds.share, SPLIT_SHARE_HELP_LABELLED);
    say(splitIds.group, splitTask() === 'image_classification' ? SPLIT_GROUP_HELP_CLASSES : SPLIT_GROUP_HELP_LABELLED);
    splitNote.textContent = SPLIT_NOTE_LABELLED;
  };
  const splitSeedInput = input_(splitIds.seed, '0', { class: 'input input--num', inputmode: 'numeric' });
  const splitSeedError = errorSlot(`${splitIds.seed}-error`);

  // Image difference (no recipe): two PNG files, an optional reference and an optional part of the image.
  // A side handed over as a saved image is shown by its label until the person chooses a file instead.
  const png = ['.png'];
  const diffSide = (key, preset, { help = '', optional = false, title }) => {
    const id = uid(`an-diff-${key}`);
    const inputEl = input_(id, pathText(preset), { class: 'input input--path analysis__grow' });
    const side = { key, input: inputEl, error: errorSlot(`${id}-error`), saved: savedImage(preset), el: h('div') };
    const words = DIFF_SIDES[key];
    const render = () => {
      if (!side.saved) {
        mount(side.el, field({ label: words, id, optional, help, control: inputEl, extra: [browse(inputEl, { mode: 'file', title, extensions: png })], error: side.error }));
        return;
      }
      const labelId = `${id}-label`;
      const helpId = help ? `${id}-saved-help` : null;
      mount(
        side.el,
        h(
          'div',
          { class: 'field analysis__field', role: 'group', 'aria-labelledby': labelId, 'aria-describedby': helpId },
          h('span', { class: 'field__label', id: labelId }, words, optional ? h('span', { class: 'analysis__optional' }, ' (optional)') : null),
          h(
            'div',
            { class: 'analysis__field-row' },
            h('span', { class: 'analysis__saved' }, 'Saved image: ', side.saved.literal ? h('span', { class: 'mono' }, side.saved.label) : side.saved.label),
            h(
              'button',
              {
                class: 'btn btn--sm',
                type: 'button',
                onclick: () => {
                  side.saved = null;
                  render();
                  inputEl.focus();
                },
              },
              'Choose a file instead',
            ),
          ),
          help ? h('p', { class: 'field__help', id: helpId }, help) : null,
        ),
      );
    };
    render();
    return side;
  };
  const diffSides = [
    diffSide('left', left, { title: 'Choose the first image' }),
    diffSide('right', right, { title: 'Choose the second image' }),
    diffSide('reference', reference, {
      optional: true,
      help: 'Usually the ground truth. With it, the result also says which image is closer at how many pixels.',
      title: 'Choose the reference image',
    }),
  ];
  // Part of the image: all four or none (studio/image_difference.py validate_region: whole numbers up to
  // 2**31-1, width and height at least 1).
  const regionId = uid('an-region');
  const regionError = errorSlot(`${regionId}-error`);
  const regionFields = [
    { key: 'x', label: 'Left (x)', low: 0 },
    { key: 'y', label: 'Top (y)', low: 0 },
    { key: 'width', label: 'Width', low: 1 },
    { key: 'height', label: 'Height', low: 1 },
  ].map((part) => {
    const id = `${regionId}-${part.key}`;
    const control = input_(id, '', { class: 'input input--num', inputmode: 'numeric', 'aria-describedby': `${regionId}-help ${regionError.id}` });
    return { ...part, id, control };
  });
  /**
   * "Which data" of a kind that reads one paired dataset of the recipe (pair alignment, patch extraction):
   * filled once from the recipe's PairedImageDataset keys, the training data chosen when it is one of them.
   */
  const pairedSelect = (id, onchange = null) => {
    const select = h('select', { class: 'select', id, onchange }, h('option', { value: 'train' }, setWords('train')));
    let filled = false;
    /** true the one time the list was written. */
    const fill = () => {
      const keys = pairedKeys();
      if (filled || !keys?.length) return false;
      filled = true;
      const wanted = keys.includes('train') ? 'train' : keys[0];
      mount(select, keys.map((key) => h('option', { value: key, selected: key === wanted }, setWords(key))));
      select.value = wanted;
      return true;
    };
    return { select, fill };
  };

  // Patch extraction: which paired dataset, the patch size and the step, both in low quality pixels
  // (studio/dataset_patches.py validate_patch_options: whole numbers from 1, step at most the patch size).
  const patchIds = { split: uid('an-patch-split'), size: uid('an-patch-size'), step: uid('an-patch-step') };
  const cropInput = input_(patchIds.size, '', { class: 'input input--num', inputmode: 'numeric' });
  const cropError = errorSlot(`${patchIds.size}-error`);
  const strideInput = input_(patchIds.step, '', { class: 'input input--num', inputmode: 'numeric' });
  const strideError = errorSlot(`${patchIds.step}-error`);
  const cropField = field({ label: 'Patch size', id: patchIds.size, help: 'In low quality pixels.', control: cropInput, error: cropError });
  const cropHelp = cropField.querySelector('.field__help');
  const wholeFromOne = (raw) => /^\d+$/.test(raw) && Number(raw) >= 1 && Number(raw) <= 2 ** 31 - 1;
  let cropTouched = false; // the person has typed in "Patch size"
  let cropFromRecipe = false; // the field holds the chosen dataset's lq_size, put there by this dialog
  /** Until the person edits the field, it follows the chosen dataset's lq_size (when that is a whole number). */
  const presetCrop = () => {
    if (cropTouched) return;
    const size = recipeConfig?.datasets?.[patchData.select.value]?.lq_size;
    const usable = Number.isInteger(size) && size >= 1;
    if (usable) cropInput.value = String(size);
    else if (cropFromRecipe) cropInput.value = '';
    cropFromRecipe = usable;
  };
  /** The help under "Patch size": where a filled value came from, and the ground truth size it means. */
  const renderCropHelp = () => {
    const raw = cropInput.value.trim();
    const scale = recipeConfig?.scale;
    const words = [
      'In low quality pixels.',
      cropFromRecipe && !cropTouched ? 'Filled from lq_size, the size the recipe crops for training.' : '',
      wholeFromOne(raw) && Number.isInteger(scale) && scale >= 1 ? `The ground truth patch is ${fmt.dims(Number(raw) * scale, Number(raw) * scale)} pixels at ${scale}×.` : '',
    ]
      .filter(Boolean)
      .join(' ');
    if (cropHelp.textContent !== words) cropHelp.textContent = words;
  };
  cropInput.addEventListener('input', () => {
    cropTouched = true;
    renderCropHelp();
  });
  const patchData = pairedSelect(patchIds.split, () => {
    presetCrop();
    renderCropHelp();
    sync();
  });

  // Weights conversion (no recipe): a .pth file of its own (the weights field above also takes
  // .safetensors) and, when the file holds several weights dictionaries, the name of one.
  const convertIds = { file: uid('an-pth'), payload: uid('an-payload') };
  const pth = ['.pth'];
  const pthInput = input_(convertIds.file, kind === 'convert-weights' || /\.pth$/i.test(pathText(weights).trim()) ? pathText(weights) : '', { class: 'input input--path analysis__grow' });
  const pthError = errorSlot(`${convertIds.file}-error`);
  const payloadInput = input_(convertIds.payload, '', { class: 'input mono analysis__grow' });

  // Checkpoint average: one box per numbered checkpoint the caller found for the run and role, ascending by
  // step; the caller's `selected` steps are ticked. Built once, so the ticks stay when another kind is looked at.
  const averageIds = { count: uid('an-avg-count'), error: uid('an-avg-error') };
  const averageError = errorSlot(averageIds.error);
  const averageCount = h('p', { class: 'field__help', id: averageIds.count, 'aria-live': 'polite' });
  const averageSteps = new Set();
  const averageList = (averageGiven ? average.checkpoints : [])
    .filter((entry) => isPlainObject(entry) && Number.isSafeInteger(entry.step) && entry.step >= 1 && !averageSteps.has(entry.step) && averageSteps.add(entry.step))
    .sort((a, b) => a.step - b.step)
    .map((entry) => ({
      step: entry.step,
      path: typeof entry.path === 'string' ? entry.path : '',
      file: typeof entry.path === 'string' ? baseName(entry.path) : '',
      version: typeof entry.version === 'string' && entry.version ? entry.version : null,
      box: h('input', {
        type: 'checkbox',
        id: uid('an-avg'),
        checked: Array.isArray(average.selected) && average.selected.includes(entry.step),
        'aria-describedby': averageIds.error,
        onchange: () => {
          fieldError(averageError, [], '');
          renderAverageCount();
        },
      }),
    }));
  const averageTicked = () => averageList.filter((entry) => entry.box.checked);
  const renderAverageCount = () => {
    averageCount.textContent = `${fmt.int(averageTicked().length)} chosen.`;
  };
  renderAverageCount();
  const averageRun = averageGiven ? (typeof average.run.name === 'string' && average.run.name ? average.run.name : average.run.id) : '';
  const averageLead = averageGiven ? `Checkpoints of ${averageRun}${ROLE_WORDS[average.role] ? `, ${ROLE_WORDS[average.role]}` : ''}.` : '';
  const averageBlocked = () => chosen === 'average-checkpoints' && averageList.length < 2;
  const averageBnId = uid('an-average-bn');
  const averageBnBox = h('input', { type: 'checkbox', id: averageBnId, 'aria-describedby': `${averageBnId}-help` });

  // Network parameters: the transfer preview is offered only for a recipe that sets both path.transfer_g and
  // path.pretrain_network_g (studio/analysis_jobs.py refuses `inspectTransfer` without them).
  const transferId = uid('an-transfer');
  const transferBox = h('input', { type: 'checkbox', id: transferId, 'aria-describedby': `${transferId}-help` });
  const transferField = h(
    'div',
    { class: 'field analysis__field', hidden: true },
    h('div', { class: 'analysis__check' }, transferBox, h('label', { for: transferId }, 'Also show what the transfer would copy')),
    h('p', { class: 'field__help', id: `${transferId}-help` }, TRANSFER_HELP),
  );
  const transferOffered = () => {
    const path = recipeConfig?.path;
    if (!isPlainObject(path)) return false;
    const set = (value) => value !== undefined && value !== null && value !== false && value !== '';
    // A transfer happens only when training starts fresh: a recipe that resumes has none to preview.
    if (recipeConfig.auto_resume === true || set(path.resume_state)) return false;
    return set(path.transfer_g) && typeof path.pretrain_network_g === 'string' && path.pretrain_network_g.trim() !== '';
  };

  // A field's error goes as soon as the person edits that field; the next "Start job" checks it again.
  for (const [slot, controls] of [
    [shareError, [shareInput]],
    [splitSeedError, [splitSeedInput]],
    [cropError, [cropInput]],
    [strideError, [strideInput]],
    [pthError, [pthInput]],
    [exportClipError, [exportClipInput]],
    [regionError, regionFields.map((part) => part.control)],
    ...diffSides.map((side) => [side.error, [side.input]]),
  ]) {
    for (const control of controls) control.addEventListener('input', () => fieldError(slot, controls, ''));
  }

  // Data audit: files opened per dataset (or every file) and examples kept.
  const auditIds = { limit: uid('an-audit'), all: uid('an-audit-all'), samples: uid('an-samples') };
  const auditInput = input_(auditIds.limit, '100', { class: 'input input--num', inputmode: 'numeric' });
  const auditAll = h('input', {
    type: 'checkbox',
    id: auditIds.all,
    onchange: () => {
      auditInput.disabled = auditAll.checked;
    },
  });
  const sampleInput = input_(auditIds.samples, '25', { class: 'input input--num', inputmode: 'numeric' });

  // Detection prediction: optional cutoff and box limit, shown only for an object detection recipe.
  const detectIds = { score: uid('an-score'), top: uid('an-top') };
  const scoreInput = input_(detectIds.score, '', { class: 'input input--num', inputmode: 'decimal' });
  const topInput = input_(detectIds.top, '', { class: 'input input--num', inputmode: 'numeric' });
  const detectionFields = h(
    'div',
    { class: 'analysis__inline', hidden: true },
    field({ label: 'Lowest score to keep', id: detectIds.score, optional: true, help: "A number from 0 to 1. Boxes must score above this. Leave empty for the model's default.", control: scoreInput }),
    field({ label: 'Most boxes per image', id: detectIds.top, optional: true, help: "A whole number, at least 1. Leave empty for the model's default.", control: topInput }),
  );

  // Evaluation: all validation and test datasets, or one of them; and whether to keep output images.
  const evalIds = { split: uid('an-split'), save: uid('an-save') };
  const splitSelect = h('select', { class: 'select', id: evalIds.split }, h('option', { value: '' }, 'All validation and test datasets'));
  const saveImagesBox = h('input', { type: 'checkbox', id: evalIds.save, 'aria-describedby': `${evalIds.save}-help` });
  const saveNote = h('p', { class: 'analysis__note', hidden: true }, SAVE_ELSEWHERE);
  // Where to measure: the CPU every time the dialog opens; the GPU only when the person picks it.
  const deviceIds = { cpu: uid('an-device'), gpu: uid('an-device'), help: uid('an-device-help'), group: uid('an-device-group') };
  const cpuRadio = h('input', { type: 'radio', name: deviceIds.group, id: deviceIds.cpu, value: 'cpu', checked: true });
  const gpuRadio = h('input', { type: 'radio', name: deviceIds.group, id: deviceIds.gpu, value: 'cuda' });
  const gpuTrainingNote = h('p', { class: 'analysis__note', hidden: true }, GPU_TRAINING_NOTE);
  // The service decides; this line only says what it will answer while a training is running.
  // The two sentences that say where the job runs follow the choice, so the dialog never says both.
  const evalNote = h('p', { class: 'analysis__note' }, EVAL_NOTE);
  const dialogLead = h('p', null, DIALOG_LEAD);
  const syncDevice = () => {
    const onGpu = gpuRadio.checked;
    gpuTrainingNote.hidden = !(onGpu && runs.active().length > 0);
    const note = onGpu ? EVAL_NOTE_GPU : EVAL_NOTE;
    if (evalNote.textContent !== note) evalNote.textContent = note;
    const lead = onGpu && chosen === 'evaluate' ? DIALOG_LEAD_GPU : DIALOG_LEAD;
    if (dialogLead.textContent !== lead) dialogLead.textContent = lead;
  };
  cpuRadio.addEventListener('change', syncDevice);
  gpuRadio.addEventListener('change', syncDevice);
  dialogController.signal.addEventListener('abort', runs.subscribe(syncDevice), { once: true });
  const deviceField = h(
    'fieldset',
    { class: 'analysis__exports', 'aria-describedby': deviceIds.help },
    h('legend', { class: 'field__label' }, 'Where to measure'),
    h('div', { class: 'analysis__check' }, cpuRadio, h('label', { for: deviceIds.cpu }, 'On the CPU')),
    h('div', { class: 'analysis__check' }, gpuRadio, h('label', { for: deviceIds.gpu }, 'On the GPU')),
    h('p', { class: 'field__help', id: deviceIds.help }, DEVICE_HELP),
    gpuTrainingNote,
  );
  let saveField = null;
  const ownScores = () => AUDIT_TASKS.includes(task);
  // What the service says this recipe's evaluation can also keep (summary.actions.holdout.optionalExports):
  // one checkbox per named request field and nothing else. null when the service does not say: the fixed
  // rule above (output images, except for tasks with their own scores) then decides.
  const optionalExports = () =>
    Array.isArray(actions?.holdout?.optionalExports)
      ? [...new Set(actions.holdout.optionalExports.filter((name) => typeof name === 'string' && name && !EVALUATE_FIELDS.includes(name)))]
      : null;
  const exportBoxes = new Map();
  const exportsHost = h('div', { hidden: true });
  let exportsFilled = false;
  /** Built once, when the recipe check has answered: the list belongs to the recipe, which this dialog does not change. */
  const fillExports = () => {
    const names = optionalExports();
    if (exportsFilled || !names) return;
    exportsFilled = true;
    if (!names.length) {
      mount(exportsHost, h('p', { class: 'analysis__note' }, EXPORT_NONE));
      return;
    }
    mount(
      exportsHost,
      h(
        'fieldset',
        { class: 'analysis__exports' },
        h('legend', { class: 'field__label' }, 'Also keep'),
        names.map((name) => {
          const id = uid('an-keep');
          const words = EXPORT_WORDS[name];
          const box = h('input', { type: 'checkbox', id, 'aria-describedby': `${id}-help` });
          exportBoxes.set(name, box);
          return h(
            'div',
            { class: 'field analysis__field' },
            h('div', { class: 'analysis__check' }, box, h('label', { for: id }, words ? words.label : h('span', { class: 'mono' }, name))),
            h('p', { class: 'field__help', id: `${id}-help` }, !words ? EXPORT_UNKNOWN : name === 'saveImages' && task === 'semantic_segmentation' ? EXPORT_MASKS : words.help),
          );
        }),
      ),
    );
  };
  let splitKeys = undefined; // undefined: not read yet; null: unknown; else the val*/test* dataset keys
  let splitsFilled = false;
  const fillSplits = () => {
    if (splitsFilled || !Array.isArray(splitKeys)) return;
    splitsFilled = true;
    for (const key of splitKeys) splitSelect.appendChild(h('option', { value: key }, key));
  };

  // Pair alignment: which paired dataset, how many pairs, search radius, seed (studio/analysis_jobs.py:
  // split default train, sampleLimit 1..64 default 12, radius 2..12 default 6, seed 0..2**32-1 default 0).
  const pairIds = { split: uid('an-pair-split'), pairs: uid('an-pairs'), radius: uid('an-radius'), seed: uid('an-seed') };
  const pairData = pairedSelect(pairIds.split);
  const pairSplitSelect = pairData.select;
  const pairsInput = input_(pairIds.pairs, '12', { class: 'input input--num', inputmode: 'numeric' });
  const radiusInput = input_(pairIds.radius, '6', { class: 'input input--num', inputmode: 'numeric' });
  const seedInput = input_(pairIds.seed, '0', { class: 'input input--num', inputmode: 'numeric' });
  /** Both "Which data" lists, once the recipe has been read; the patch size then takes its first value. */
  const fillPairSplits = () => {
    pairData.fill();
    if (patchData.fill()) presetCrop();
  };

  // Frame sequences: frames per window (optional, positive).
  const clipId = uid('an-clip');
  const clipInput = input_(clipId, '', { class: 'input input--num', inputmode: 'numeric' });
  const clipField = h(
    'div',
    { hidden: true },
    field({
      label: 'Frames per window',
      id: clipId,
      optional: true,
      help: 'A whole number, at least 1. Networks with a fixed window bring their own size. Otherwise, when empty, the validation window from the recipe is used, then the training clip size.',
      control: clipInput,
    }),
  );

  // Split leakage: how many matching groups to list (0 to 200; 0 is sent).
  const leakId = uid('an-groups');
  const groupInput = input_(leakId, '100', { class: 'input input--num', inputmode: 'numeric' });

  // The recipe's task, read once through the (non-mutating) preflight. null when it could not be read:
  // the start request is then the authority.
  let taskPromise = null;
  let task = undefined; // undefined: not read yet; null: unknown; else the task id
  const recipeTask = () => {
    if (!text) return Promise.resolve(null);
    if (!taskPromise) {
      taskPromise = api
        .preflight(text, { signal: dialogController.signal })
        .then((check) => ({
          task: check?.summary?.task ?? check?.config?.task ?? null,
          config: check?.config ?? null,
          actions: isPlainObject(check?.summary?.actions) ? check.summary.actions : null,
          unavailable: isPlainObject(check?.summary?.unavailableActions) ? check.summary.unavailableActions : null,
        }))
        .catch(() => ({ task: null, config: null, actions: null, unavailable: null }))
        .then((info) => {
          task = info.task;
          actions = info.actions;
          unavailable = info.unavailable;
          splitKeys = holdoutKeys(info.config);
          recipeConfig = info.config;
          valKeys = isPlainObject(info.config?.datasets) ? Object.keys(info.config.datasets).filter((key) => /^val(?:_|$)/.test(key)) : null;
          return info;
        });
    }
    return taskPromise;
  };
  let recipeConfig = null;
  // What the service says this recipe (its task and its network) can do: summary.actions of the check, each
  // action with the kinds of input it takes. null when the service did not report it (the check is still
  // running, or failed): nothing is then blocked by a rule of this screen, and the start request stays
  // the authority.
  let actions = null;
  // The service's own sentence for each action it does not offer (summary.unavailableActions), or null.
  let unavailable = null;
  const lacks = (name) => isPlainObject(actions) && !isPlainObject(actions[name]);
  const takes = (name, kind) => isPlainObject(actions?.[name]) && Array.isArray(actions[name].inputs) && actions[name].inputs.includes(kind);
  // Only for the words about the input (a folder of frames): what is offered is the service's to say.
  const sequence = () => (isPlainObject(actions?.predict) ? takes('predict', 'sequences') : isSequence(task, recipeConfig?.network_g?.type));
  const networkName = () => String(recipeConfig?.network_g?.type ?? recipeConfig?.network_ae?.type ?? '').trim();
  // Said of the recipe, not of the network type: the same network can be offered with other options
  // (a sequence network that reads its frames another way), and the service gives no reason of its own.
  const notOffered = (what) => `${what} is not available for this recipe: the service does not offer it for this task and this network${networkName() ? ` (${networkName()})` : ''} with the options the recipe sets.`;
  /** A missing action in words: the service's reason, verbatim, when it gave one; this screen's sentence otherwise. */
  const notAvailable = (what, name) => {
    const reason = typeof unavailable?.[name] === 'string' ? unavailable[name].trim() : '';
    return reason ? `${what} is not available for this recipe. ${reason}` : notOffered(what);
  };
  // Export options the service offers for this recipe: a graph that takes other sizes, a graph for frame sequences.
  const anySizeOffered = () => actions?.export?.dynamicSpatial === true;
  const exportFrames = () => takes('export', 'sequences');
  const predictBlocked = () => chosen === 'predict' && lacks('predict');
  const holdoutBlocked = () => chosen === 'evaluate' && lacks('holdout');
  // Pair alignment: the PairedImageDataset keys and the scale, as the service checks them.
  const pairedKeys = () =>
    isPlainObject(recipeConfig?.datasets)
      ? Object.keys(recipeConfig.datasets).filter((key) => String(recipeConfig.datasets[key]?.type ?? '').toLowerCase() === 'pairedimagedataset')
      : null;
  // Patch extraction reads a paired dataset and the scale by the same two rules, said in its own words.
  const pairedGates = { 'pair-registration': { none: PAIR_NONE, scale: PAIR_SCALE }, 'extract-patches': { none: PATCH_NONE, scale: PATCH_SCALE } };
  const pairBlockedWhy = () => {
    const words = pairedGates[chosen];
    if (!words || !isPlainObject(recipeConfig)) return '';
    if (!pairedKeys()?.length) return words.none;
    if (!Number.isInteger(recipeConfig.scale) || recipeConfig.scale < 1) return words.scale;
    return '';
  };
  // Patch extraction, further: a restoration recipe (a recipe without a task counts as one).
  const patchTaskBlocked = () => {
    if (chosen !== 'extract-patches') return false;
    if (typeof task === 'string' && task !== 'image_restoration') return true;
    return isPlainObject(recipeConfig) && (recipeConfig.task || 'image_restoration') !== 'image_restoration';
  };
  // And the chosen dataset must be folders on disk (a dataset that names no io_backend is). Said beside the
  // form, which stays, so that another dataset can be chosen; only "Start job" is withheld.
  const patchDiskWhy = () => {
    if (chosen !== 'extract-patches' || !isPlainObject(recipeConfig) || patchTaskBlocked() || pairBlockedWhy()) return '';
    const key = patchData.select.value;
    const io = recipeConfig.datasets?.[key]?.io_backend;
    if (!isPlainObject(io) || typeof io.type !== 'string' || io.type === 'disk') return '';
    return `Patch extraction reads paired image folders on disk. ${setWords(key)} is stored another way (io_backend type ${io.type}).`;
  };
  // Validation split: the training dataset must be of the type the service splits for the recipe's task
  // (SPLIT_TYPES), as the service checks it. Anything else it refuses is said in its own words.
  const splitBlocked = () => {
    if (chosen !== 'split-dataset' || splitTask() === null) return false;
    const expected = SPLIT_TYPES[splitTask()];
    if (!expected) return true;
    if (!isPlainObject(recipeConfig)) return false;
    const train = isPlainObject(recipeConfig.datasets) ? recipeConfig.datasets.train : null;
    return !isPlainObject(train) || !expected.includes(String(train.type ?? '').toLowerCase());
  };
  // Export is withheld only when the service's list of actions is known and has no export.
  const exportBlocked = () => chosen === 'export' && lacks('export');
  const auditBlocked = () => chosen === 'task-data' && typeof task === 'string' && !DATA_CHECK_TASKS.includes(task);
  const evalBlocked = () => chosen === 'evaluate' && Array.isArray(splitKeys) && splitKeys.length === 0;
  // The leakage check needs `val` or a `val_*` dataset (the job fails with "Provide at least one val or val_* split.").
  let valKeys = undefined;
  // It covers restoration recipes only: for these tasks the job is accepted and then fails with
  // "Choose a restoration recipe to audit split leakage."
  const leakTaskBlocked = () => chosen === 'split-leakage' && AUDIT_TASKS.includes(task);
  const leakBlocked = () => leakTaskBlocked() || (chosen === 'split-leakage' && Array.isArray(valKeys) && valKeys.length === 0);

  const weightsField = h('div');
  // A recipe whose actions take records (a table of numbers) has no image, folder or picture size to ask for.
  const recordsFor = (name) => takes(name, 'records');
  const inputFileBrowse = browse(inputInput, { mode: 'file', title: 'Choose the input file' });
  const inputFolderBrowse = browse(inputInput, { mode: 'folder', title: 'Choose a folder of input images' });
  const groups = {
    predict: h(
      'div',
      { class: 'stack' },
      field({
        label: 'Input image or folder',
        id: ids.input,
        help: INPUT_HELP,
        control: inputInput,
        extra: [inputFileBrowse, inputFolderBrowse],
      }),
      detectionFields,
      clipField,
    ),
    export: h(
      'div',
      { class: 'stack' },
      h(
        'div',
        { class: 'analysis__inline' },
        field({ label: 'Input height', id: ids.height, help: EXPORT_HEIGHT_HELP, control: heightInput }),
        field({ label: 'Input width', id: ids.width, help: EXPORT_WIDTH_HELP, control: widthInput }),
        field({ label: 'ONNX opset', id: ids.opset, optional: true, help: '13 to 20. 17 when empty.', control: opsetInput }),
      ),
      sizeNote,
      anySizeField,
      exportClipField,
    ),
    'loss-report': h(
      'div',
      { class: 'stack' },
      h(
        'div',
        { class: 'analysis__inline' },
        field({ label: 'Batches', id: ids.batches, optional: true, help: 'Training batches to measure, 1 to 32. 1 when empty.', control: batchesInput }),
        field({ label: 'Update number', id: ids.step, optional: true, help: 'The training iteration whose objective is evaluated, from 1. 1 when empty.', control: stepInput }),
      ),
      field({
        label: 'Generator parameter',
        id: ids.parameter,
        optional: true,
        help: 'Name of one generator parameter to measure the gradients at. When empty, the generator output is observed.',
        control: parameterInput,
      }),
      field({
        label: 'Discriminator weights',
        id: ids.disc,
        optional: true,
        help: 'Discriminator checkpoint for recipes with an adversarial loss.',
        control: discInput,
        extra: [browse(discInput, { mode: 'file', title: 'Choose discriminator weights', extensions: ckpt })],
      }),
    ),
    'pack-dataset': h(
      'div',
      { class: 'stack' },
      h('p', { class: 'analysis__note' }, PACK_COST),
      field({
        label: 'Ground truth folders',
        id: packIds.gt,
        help: 'One folder per line, on the trainer machine. Line 1 pairs with line 1 of the low-quality folders, and so on.',
        control: gtArea,
        extra: [appendFolder(gtArea, 'Add a ground truth folder')],
      }),
      field({
        label: 'Low-quality folders',
        id: packIds.lq,
        help: 'One folder per line, in the same order as the ground truth folders.',
        control: lqArea,
        extra: [appendFolder(lqArea, 'Add a low-quality folder')],
      }),
      h(
        'div',
        { class: 'analysis__inline' },
        field({ label: 'Scale', id: packIds.scale, help: 'A whole number: how many times larger the ground truth images are.', control: scaleInput }),
        field({
          label: 'Low-quality name template',
          id: packIds.template,
          optional: true,
          help: 'Applied to the ground truth file name without its extension to give the low-quality name; {} means the same name on both sides. The extension may differ.',
          control: templateInput,
        }),
      ),
      field({
        label: 'Destination',
        id: packIds.dest,
        optional: true,
        help: "It must be a new folder that does not exist yet, inside an existing folder: Browse chooses that parent folder, then type the new folder's name after it. When empty, the dataset is stored in the job's own folder.",
        control: destInput,
        extra: [browseParent],
      }),
    ),
    'weight-drift': h(
      'div',
      { class: 'stack' },
      h('p', { class: 'analysis__note' }, DRIFT_NOTE),
      field({
        label: 'Baseline checkpoint',
        id: driftIds.left,
        help: 'The checkpoint the movement is measured from. A path on the trainer machine.',
        control: leftInput,
        extra: [browse(leftInput, { mode: 'file', title: 'Choose the baseline checkpoint', extensions: drift })],
      }),
      field({
        label: 'Candidate checkpoint',
        id: driftIds.right,
        help: 'The checkpoint compared with the baseline.',
        control: rightInput,
        extra: [browse(rightInput, { mode: 'file', title: 'Choose the candidate checkpoint', extensions: drift })],
      }),
    ),
    evaluate: h(
      'div',
      { class: 'stack' },
      evalNote,
      field({ label: 'Datasets to measure', id: evalIds.split, help: "The recipe's validation and test datasets.", control: splitSelect }),
      deviceField,
      saveNote,
      h(
        'div',
        { class: 'field analysis__field', ref: (el) => (saveField = el) },
        h('div', { class: 'analysis__check' }, saveImagesBox, h('label', { for: evalIds.save }, 'Save output images')),
        h('p', { class: 'field__help', id: `${evalIds.save}-help` }, SAVE_HELP),
      ),
      exportsHost,
    ),
    'split-dataset': h(
      'div',
      { class: 'stack' },
      field({
        label: ['Share for validation', h('span', { class: 'visually-hidden' }, ' in percent')],
        id: splitIds.share,
        help: SPLIT_SHARE_HELP,
        control: shareInput,
        extra: [h('span', { class: 'analysis__unit', 'aria-hidden': 'true' }, '%')],
        error: shareError,
      }),
      groupField,
      sceneLine,
      field({ label: 'Seed', id: splitIds.seed, help: 'The same seed gives the same split again.', control: splitSeedInput, error: splitSeedError }),
      splitNote,
    ),
    'average-checkpoints': h(
      'div',
      { class: 'stack' },
      h('p', { class: 'analysis__source' }, averageLead),
      h(
        'fieldset',
        { class: 'analysis__exports', 'aria-describedby': `${averageIds.count} ${averageIds.error}` },
        h('legend', { class: 'field__label' }, 'Checkpoints to average'),
        h(
          'div',
          { class: 'analysis__steps' },
          averageList.map((entry) =>
            h(
              'div',
              { class: 'analysis__check' },
              entry.box,
              h('label', { for: entry.box.id }, `Iteration ${fmt.int(entry.step)}`),
              entry.file ? h('span', { class: 'mono analysis__step-file', title: entry.path }, entry.file) : null,
            ),
          ),
        ),
        averageCount,
        averageError,
      ),
      averageGiven && average.more === true ? h('p', { class: 'analysis__note' }, AVERAGE_MORE) : null,
      h(
        'div',
        { class: 'field analysis__field' },
        h('div', { class: 'analysis__check' }, averageBnBox, h('label', { for: averageBnId }, AVERAGE_BN_LABEL)),
        h('p', { class: 'field__help', id: `${averageBnId}-help` }, AVERAGE_BN_HELP),
      ),
      h('p', { class: 'analysis__note' }, AVERAGE_NOTE),
    ),
    'network-parameters': h('div', { class: 'stack' }, transferField, h('p', { class: 'analysis__note' }, PARAMETERS_NOTE)),
    'image-difference': h(
      'div',
      { class: 'stack' },
      diffSides.map((side) => side.el),
      h(
        'fieldset',
        { class: 'analysis__region' },
        h('legend', { class: 'field__label' }, 'Part of the image', h('span', { class: 'analysis__optional' }, ' (optional)')),
        h('div', { class: 'analysis__inline' }, regionFields.map((part) => field({ label: part.label, id: part.id, control: part.control }))),
        h('p', { class: 'field__help', id: `${regionId}-help` }, 'In pixels of the original PNG. Leave all four empty to compare the whole image.'),
        regionError,
      ),
      h('p', { class: 'analysis__note' }, DIFF_NOTE),
    ),
    'extract-patches': h(
      'div',
      { class: 'stack' },
      field({ label: 'Which data', id: patchIds.split, help: "The recipe's datasets of paired ground truth and low quality images.", control: patchData.select }),
      cropField,
      field({
        label: 'Step',
        id: patchIds.step,
        optional: true,
        help: 'How far the window moves between patches, in low quality pixels. Smaller than the patch size makes patches overlap. Empty means the patch size: no overlap.',
        control: strideInput,
        error: strideError,
      }),
      h('p', { class: 'analysis__note' }, PATCH_NOTE),
    ),
    'convert-weights': h(
      'div',
      { class: 'stack' },
      field({
        label: 'Checkpoint (.pth)',
        id: convertIds.file,
        help: 'A PyTorch .pth file on the trainer machine.',
        control: pthInput,
        extra: [browse(pthInput, { mode: 'file', title: 'Choose the .pth checkpoint', extensions: pth })],
        error: pthError,
      }),
      field({
        label: 'Weights dictionary',
        id: convertIds.payload,
        optional: true,
        help: 'Only when the file holds several, for example params and params_ema. Left empty, the service takes the file\'s only one, and refuses with the names if there are several.',
        control: payloadInput,
      }),
    ),
    'pair-registration': h(
      'div',
      { class: 'stack' },
      field({ label: 'Which data', id: pairIds.split, help: "The recipe's datasets of paired ground truth and low quality images.", control: pairSplitSelect }),
      h(
        'div',
        { class: 'analysis__inline' },
        field({ label: 'Pairs to measure', id: pairIds.pairs, help: 'A whole number from 1 to 64. 12 when empty.', control: pairsInput }),
        field({ label: 'Search radius', id: pairIds.radius, help: 'In low quality pixels, 2 to 12. 6 when empty.', control: radiusInput }),
        field({ label: 'Seed', id: pairIds.seed, help: 'Chooses which pairs are measured: the same seed measures the same pairs again. A whole number, 0 or more; 0 when empty.', control: seedInput }),
      ),
    ),
    'split-leakage': h(
      'div',
      { class: 'stack' },
      h('p', { class: 'analysis__note' }, LEAK_NOTE),
      field({ label: 'Examples to list', id: leakId, help: 'How many matching groups to list. Every file is still compared. From 0 to 200.', control: groupInput }),
    ),
    'task-data': h(
      'div',
      { class: 'stack' },
      h(
        'div',
        { class: 'analysis__inline' },
        h(
          'div',
          { class: 'field analysis__field' },
          h('label', { class: 'field__label', for: auditIds.limit }, 'Files to open in each dataset'),
          h('div', { class: 'analysis__field-row' }, auditInput),
          h('div', { class: 'analysis__check' }, auditAll, h('label', { for: auditIds.all }, 'Open every file (slower on large datasets)')),
          h('p', { class: 'field__help', id: `${auditIds.limit}-help` }, 'A whole number, 0 or more. Every file is counted; at most this many are opened and checked in each dataset.'),
        ),
        field({ label: 'Examples to keep', id: auditIds.samples, help: 'Example files kept in the result for each dataset, 0 to 100.', control: sampleInput }),
      ),
    ),
  };
  auditInput.setAttribute('aria-describedby', `${auditIds.limit}-help`);

  // Shown in place of a form the service would refuse (an export of a detection recipe, an audit of
  // a restoration recipe), or while the recipe's task is being read.
  const kindGate = h('div', { class: 'analysis__gate' });

  // Rebuilt only when its wording changes, so a sync after a background answer keeps focus in the field.
  let weightsWording = '';
  const renderWeights = () => {
    // An autoencoder recipe has no generator: its weights are named for what they are.
    const network = task === 'autoencoder' ? 'autoencoder' : 'generator';
    const wording = `${chosen === 'loss-report' ? 'optional' : chosen === 'evaluate' ? 'checkpoint' : 'required'} ${network}`;
    if (wording === weightsWording) return;
    weightsWording = wording;
    // The same input is moved into the rebuilt field; moving it drops the focus, so it is given back.
    const typing = document.activeElement === weightsInput;
    mount(
      weightsField,
      field({
        label: chosen === 'evaluate' ? 'Checkpoint' : network === 'autoencoder' ? 'Autoencoder weights' : 'Generator weights',
        id: ids.weights,
        optional: chosen === 'loss-report',
        help: chosen === 'loss-report' ? 'A checkpoint to measure. When empty, the recipe\'s own pretrained weights setting is used.' : 'A .safetensors or .pth checkpoint on the trainer machine.',
        control: weightsInput,
        extra: [browse(weightsInput, { mode: 'file', title: `Choose ${network} weights`, extensions: ckpt })],
      }),
    );
    if (typing) weightsInput.focus();
  };

  // Export: a classification graph must use task_options.image_size. Read through the same preflight;
  // when it fails, nothing is added: the start request is the authority.
  let sizeChecked = false;
  const fillExportSize = async () => {
    if (sizeChecked || !text) return;
    sizeChecked = true;
    const info = await recipeTask();
    if (dialogController.signal.aborted) return;
    const size = info?.config?.task_options?.image_size;
    if (info?.task !== 'image_classification' || typeof size !== 'number' || !Number.isFinite(size)) return;
    if (!heightInput.value.trim()) heightInput.value = String(size);
    if (!widthInput.value.trim()) widthInput.value = String(size);
    sizeNote.hidden = false;
  };

  /** Kinds whose form depends on the recipe's task: shown once the task is known. */
  const taskGated = () =>
    Boolean(text) &&
    (chosen === 'export' || chosen === 'task-data' || chosen === 'evaluate' || chosen === 'split-leakage' || chosen === 'pair-registration' || chosen === 'split-dataset' || chosen === 'extract-patches');
  const waitingForTask = () => taskGated() && task === undefined;
  const blocked = () =>
    exportBlocked() || predictBlocked() || holdoutBlocked() || auditBlocked() || evalBlocked() || leakBlocked() || patchTaskBlocked() || Boolean(pairBlockedWhy()) || splitBlocked() || averageBlocked();
  const startDisabled = () => (needsRecipe() && !text) || blocked() || waitingForTask() || Boolean(patchDiskWhy());

  const renderGate = () => {
    const key = averageBlocked()
      ? `average:${averageList.length}`
      : exportBlocked()
      ? 'export'
      : predictBlocked()
        ? 'predict'
        : holdoutBlocked()
          ? 'holdout'
          : patchTaskBlocked()
            ? 'patch-task'
            : pairBlockedWhy()
              ? `pair:${pairBlockedWhy()}`
              : auditBlocked()
                ? 'audit'
                : evalBlocked()
                  ? 'eval'
                  : leakTaskBlocked()
                    ? 'leak-task'
                    : leakBlocked()
                      ? 'leak'
                      : splitBlocked()
                        ? 'split'
                        : patchDiskWhy()
                          ? `patch-disk:${patchDiskWhy()}`
                          : waitingForTask()
                            ? 'wait'
                            : '';
    if (kindGate.dataset.key === key) return;
    kindGate.dataset.key = key;
    // "Try on an image instead" is offered only when the service offers prediction for this recipe.
    const tryInstead = (label) => (lacks('predict') ? null : h('button', { class: 'btn btn--sm', type: 'button', dataset: { focus: 'try' }, onclick: () => switchKind('predict') }, label));
    if (key.startsWith('average:')) {
      mount(kindGate, notice('', averageList.length ? AVERAGE_ONE : AVERAGE_NONE));
    } else if (key === 'export') {
      mount(kindGate, notice('', notAvailable('ONNX export', 'export'), tryInstead(sequence() ? 'Try on a folder of frames instead' : 'Try on an image instead')));
    } else if (key === 'predict') {
      mount(kindGate, notice('', notAvailable('Prediction', 'predict')));
    } else if (key === 'holdout') {
      mount(kindGate, notice('', notAvailable('Evaluation on validation or test data', 'holdout')));
    } else if (key === 'audit') {
      mount(kindGate, notice('', h('span', null, AUDIT_ELSEWHERE, ' ', h('a', { href: '#/data', onclick: () => closeDialog?.(null) }, 'Open the Data screen'))));
    } else if (key.startsWith('pair:')) {
      mount(kindGate, notice('', pairBlockedWhy()));
    } else if (key === 'leak-task') {
      mount(kindGate, notice('', LEAK_TASKS));
    } else if (key === 'leak') {
      mount(kindGate, notice('', NO_VAL));
    } else if (key === 'split') {
      mount(kindGate, notice('', SPLIT_GATE));
    } else if (key === 'patch-task') {
      mount(kindGate, notice('', PATCH_TASK));
    } else if (key.startsWith('patch-disk:')) {
      mount(kindGate, notice('', patchDiskWhy()));
    } else if (key === 'eval') {
      mount(kindGate, notice('', NO_HOLDOUT));
    } else if (key === 'wait') {
      mount(kindGate, loading(chosen === 'evaluate' ? "Reading the recipe's validation and test datasets…" : "Reading the recipe's task…"));
    } else {
      mount(kindGate, null);
    }
  };

  let startButton = null;
  const sync = () => {
    const hide = blocked() || waitingForTask();
    for (const [key, el] of Object.entries(groups)) el.hidden = key !== chosen || hide;
    // Kinds that read a recipe show where it comes from; dataset packing and weight drift take none.
    recipeLine.hidden = !needsRecipe();
    weightsField.hidden = !needsWeights() || blocked();
    if (needsWeights()) renderWeights();
    detectionFields.hidden = !(chosen === 'predict' && task === 'object_detection');
    clipField.hidden = !(chosen === 'predict' && typeof task === 'string' && sequence());
    const help = groups.predict.querySelector(`[id="${ids.input}-help"]`);
    const records = recordsFor('predict');
    const helpText = records ? INPUT_HELP_RECORDS : typeof task === 'string' && sequence() ? INPUT_HELP_SEQUENCE : task === 'autoencoder' ? INPUT_HELP_AUTOENCODER : INPUT_HELP;
    if (help && help.textContent !== helpText) help.textContent = helpText;
    const inputLabel = groups.predict.querySelector(`label[for="${ids.input}"]`);
    const labelText = records ? 'Input records file' : 'Input image or folder';
    if (inputLabel && inputLabel.textContent !== labelText) inputLabel.textContent = labelText;
    if (inputFolderBrowse && 'hidden' in inputFolderBrowse) inputFolderBrowse.hidden = records;
    // Export of a records model: the graph has no picture size, so none is asked for.
    for (const el of [heightInput, widthInput]) {
      const wrap = el.closest('.field') ?? el.parentElement;
      if (wrap && wrap.hidden !== recordsFor('export')) wrap.hidden = recordsFor('export');
    }
    // Export: the options the service offers for this recipe, and the size words of a graph for frames.
    anySizeField.hidden = !anySizeOffered();
    exportClipField.hidden = !exportFrames();
    for (const [id, words] of [
      [ids.height, exportFrames() ? EXPORT_HEIGHT_HELP_FRAMES : EXPORT_HEIGHT_HELP],
      [ids.width, exportFrames() ? EXPORT_WIDTH_HELP_FRAMES : EXPORT_WIDTH_HELP],
    ]) {
      const el = groups.export.querySelector(`[id="${id}-help"]`);
      if (el && el.textContent !== words) el.textContent = words;
    }
    wordSplit();
    // Network parameters: the transfer preview, once the recipe has been read and sets both of its paths.
    transferField.hidden = !transferOffered();
    fillPairSplits();
    renderCropHelp();
    fillSplits();
    // What an evaluation can also keep: the service's own list when it gives one, the fixed rule otherwise.
    const declared = optionalExports() !== null;
    fillExports();
    exportsHost.hidden = !declared;
    saveNote.hidden = declared || !ownScores();
    if (saveField) saveField.hidden = declared || ownScores();
    renderGate();
    if (startButton) startButton.disabled = startDisabled();
    if (chosen === 'export') fillExportSize();
    // Prediction (detection fields), export (what the service offers), the audit (its tasks only), the
    // evaluation (its datasets), the validation split (its training dataset) and patch extraction (its
    // paired datasets) read the recipe once. So does the parameter list, only to learn whether the recipe sets
    // a weight transfer: its form does not wait for the answer, and a recipe that fails its check still starts.
    if (text && task === undefined && ['predict', 'export', 'task-data', 'evaluate', 'split-leakage', 'pair-registration', 'split-dataset', 'extract-patches', 'network-parameters'].includes(chosen)) {
      recipeTask().then(() => {
        if (!dialogController.signal.aborted) sync();
      });
    }
  };

  const dialogTitle = () => (presetKind ? KINDS[chosen].title ?? `Start ${kindNoun(chosen)}` : 'New analysis job');
  const kindSummary = () => h('p', null, h('strong', null, `${KINDS[chosen].label}. `), KINDS[chosen].text);
  let dialogEl = null;
  /** "Try on an image instead": the same dialog, weights and recipe, as a prediction. */
  const switchKind = (key) => {
    chosen = key;
    syncDevice();
    if (presetKind) {
      mount(kindPicker, kindSummary());
      const title = dialogEl?.querySelector('.dialog__title');
      if (title) title.textContent = dialogTitle();
    } else {
      for (const radio of kindPicker.querySelectorAll('input[type="radio"]')) radio.checked = radio.value === key;
      mount(kindNote, kindSummary());
    }
    mount(errorBox, null);
    sync();
    inputInput.focus();
  };

  const recipeLine = h(
    'div',
    null,
    text
      ? h('p', { class: 'analysis__source' }, 'Recipe: ', h('strong', null, label))
      : notice('warn', h('span', null, "A recipe is needed: the job reads the model and data settings from it. Open one on the Recipe screen, or start the job from a run's Files tab to use that run's recipe. ", h('a', { href: '#/recipe' }, 'Open the Recipe screen'))),
  );

  // The picker lists the kinds by what they work on, names only; the chosen kind's description stands
  // under the list once, so the fields of the form stay in view. Each radio still carries its own
  // description for a screen reader (aria-describedby).
  const kindNote = h('div', { class: 'analysis__kind-note' }, presetKind ? null : kindSummary());
  const kindOption = (key) => {
    const info = KINDS[key];
    const radioId = uid('an-kind');
    return h(
      'div',
      { class: 'analysis__kind-option' },
      h('input', {
        type: 'radio',
        name: `analysis-kind-${requestId}`,
        id: radioId,
        value: key,
        checked: key === chosen,
        'aria-describedby': `${radioId}-text`,
        onchange: () => {
          chosen = key;
          syncDevice();
          mount(kindNote, kindSummary());
          mount(errorBox, null);
          sync();
        },
      }),
      h('label', { for: radioId }, info.label),
      h('span', { class: 'visually-hidden', id: `${radioId}-text` }, info.text),
    );
  };
  const grouped = new Set(KIND_GROUPS.flatMap(([, keys]) => keys));
  const ungrouped = Object.keys(KINDS).filter((key) => !grouped.has(key) && !KINDS[key].noDialog);
  const kindPicker = presetKind
    ? h('div', null, kindSummary())
    : h(
        'fieldset',
        { class: 'analysis__kinds' },
        h('legend', { class: 'field__label' }, 'What to run'),
        [...KIND_GROUPS, ...(ungrouped.length ? [['Other', ungrouped]] : [])].map(([title, keys]) => {
          const known = keys.filter((key) => KINDS[key] && (key !== 'average-checkpoints' || averageGiven));
          if (!known.length) return null;
          const titleId = uid('an-kind-group');
          return h(
            'div',
            { class: 'analysis__kind-group', role: 'group', 'aria-labelledby': titleId },
            h('span', { class: 'analysis__kind-grouptitle', id: titleId }, title),
            h('div', { class: 'analysis__kind-options' }, known.map(kindOption)),
          );
        }),
        kindNote,
      );

  const errorBox = h('div', { class: 'analysis__error', role: 'alert' });
  const body = h(
    'div',
    { class: 'stack analysis__dialog' },
    dialogLead,
    kindPicker,
    recipeLine,
    weightsField,
    groups.predict,
    groups.export,
    groups['loss-report'],
    groups['pack-dataset'],
    groups.evaluate,
    groups['weight-drift'],
    groups['task-data'],
    groups['split-leakage'],
    groups['pair-registration'],
    groups['split-dataset'],
    groups['extract-patches'],
    groups['image-difference'],
    groups['convert-weights'],
    groups['average-checkpoints'],
    groups['network-parameters'],
    kindGate,
    errorBox,
  );
  sync();

  let busy = false;
  let closeDialog;
  let created = null;

  const intOf = (inputEl, name, required) => {
    const raw = inputEl.value.trim();
    if (!raw) return required ? { error: `Enter the ${name}.` } : { value: undefined };
    if (!/^\d+$/.test(raw)) return { error: `Enter the ${name} as a whole number.` };
    return { value: Number(raw) };
  };

  /** Dataset packing request: {kind, dataset: {gtRoots, lqRoots, scale, filenameTemplate?}, destination?}. */
  const packParams = () => {
    const problems = [];
    const gtRoots = lines(gtArea);
    const lqRoots = lines(lqArea);
    if (!gtRoots.length) problems.push('Enter at least one ground truth folder.');
    if (!lqRoots.length) problems.push('Enter at least one low-quality folder.');
    if (gtRoots.length && lqRoots.length && gtRoots.length !== lqRoots.length)
      problems.push(`Enter as many low-quality folders as ground truth folders (now ${gtRoots.length} and ${lqRoots.length}); they are paired by line.`);
    const scale = intOf(scaleInput, 'scale', true);
    if (scale.error) problems.push(scale.error);
    const template = templateInput.value.trim();
    const destination = destInput.value.trim();
    const params = { kind: 'pack-dataset', requestId, dataset: { gtRoots, lqRoots, scale: scale.value, ...(template ? { filenameTemplate: template } : {}) } };
    if (destination) params.destination = destination;
    return { params, problems };
  };

  const start = async () => {
    if (busy || (needsRecipe() && !text)) return;
    mount(errorBox, null);
    if (chosen === 'pack-dataset') {
      const pack = packParams();
      if (pack.problems.length) {
        mount(errorBox, notice('err', pack.problems.join(' ')));
        return;
      }
      await send(pack.params);
      return;
    }
    if (chosen === 'weight-drift') {
      const problems = [];
      const l = leftInput.value.trim();
      const r = rightInput.value.trim();
      if (!l) problems.push('Enter the baseline checkpoint.');
      else if (!/\.safetensors$/i.test(l)) problems.push('The baseline checkpoint must be a .safetensors file.');
      if (!r) problems.push('Enter the candidate checkpoint.');
      else if (!/\.safetensors$/i.test(r)) problems.push('The candidate checkpoint must be a .safetensors file.');
      if (l && r && samePath(l, r)) problems.push('The baseline and the candidate are the same file. Choose two different checkpoints.');
      if (problems.length) {
        mount(errorBox, notice('err', problems.join(' ')));
        return;
      }
      await send({ kind: 'weight-drift', left: l, right: r, requestId });
      return;
    }
    if (chosen === 'image-difference') {
      // Errors are written at the field they are about; focus goes to the first of them.
      const bad = [];
      const params = { kind: 'image-difference' };
      for (const side of diffSides) {
        fieldError(side.error, [side.input], '');
        if (side.saved) {
          params[side.key] = side.saved.selection;
          continue;
        }
        const path = side.input.value.trim();
        const words = DIFF_SIDES[side.key].toLowerCase();
        if (!path) {
          if (side.key !== 'reference') bad.push([side.error, [side.input], `Enter the ${words}: a PNG file on the trainer machine.`]);
        } else if (!/\.png$/i.test(path)) bad.push([side.error, [side.input], `The ${words} must be a PNG file.`]);
        else params[side.key] = path;
      }
      fieldError(regionError, regionFields.map((part) => part.control), '');
      const filled = regionFields.filter((part) => part.control.value.trim());
      if (filled.length) {
        const empty = regionFields.filter((part) => !part.control.value.trim());
        const wrong = filled.filter((part) => {
          const raw = part.control.value.trim();
          return !/^\d+$/.test(raw) || Number(raw) < part.low || Number(raw) > 2 ** 31 - 1;
        });
        const list = (parts) => parts.map((part) => part.label).join(', ');
        const sentences = [
          empty.length ? `Fill in all four, or leave all four empty. Still empty: ${list(empty)}.` : '',
          ...wrong.map((part) => `Enter ${part.label} as a whole number${part.low ? ' of at least 1' : ', 0 or more'}.`),
        ].filter(Boolean);
        if (sentences.length) bad.push([regionError, [...empty, ...wrong].map((part) => part.control), sentences.join(' ')]);
        else params.region = Object.fromEntries(regionFields.map((part) => [part.key, Number(part.control.value.trim())]));
      }
      if (bad.length) {
        for (const [slot, controls, message] of bad) fieldError(slot, controls, message);
        // The first field in reading order, not in the order the problems were found.
        (regionFields.map((part) => part.control).find((control) => bad[0][1].includes(control)) ?? bad[0][1][0]).focus();
        return;
      }
      await send({ ...params, requestId });
      return;
    }
    if (chosen === 'average-checkpoints') {
      if (averageBlocked()) return;
      const ticked = averageTicked();
      if (ticked.length < 2) {
        fieldError(averageError, [], AVERAGE_FEW);
        (averageList.find((entry) => !entry.box.checked) ?? averageList[0]).box.focus();
        return;
      }
      // Ascending by step; the version only where the caller knew it, so the service can refuse a replaced file.
      const checkpoints = ticked.map((entry) => (entry.version ? { step: entry.step, version: entry.version } : { step: entry.step }));
      // `batchNorm` is sent only when asked for: without it the service requires identical statistics.
      await send({ kind: 'average-checkpoints', run: average.run.id, role: average.role, checkpoints, ...(averageBnBox.checked ? { batchNorm: 'fromLatest' } : {}), requestId });
      return;
    }
    if (chosen === 'network-parameters') {
      // Only the recipe text is needed: the service lists the network of a recipe that is not complete yet.
      const params = { kind: 'network-parameters', yaml: text };
      if (transferOffered() && transferBox.checked) params.inspectTransfer = true;
      await send({ ...params, requestId });
      return;
    }
    if (chosen === 'convert-weights') {
      fieldError(pthError, [pthInput], '');
      const file = pthInput.value.trim();
      const problem = !file ? 'Enter the checkpoint: a .pth file on the trainer machine.' : !/\.pth$/i.test(file) ? 'The checkpoint must be a .pth file.' : '';
      if (problem) {
        fieldError(pthError, [pthInput], problem);
        pthInput.focus();
        return;
      }
      const params = { kind: 'convert-weights', weights: file };
      // Only when named: without it the service takes the file's only dictionary.
      const payload = payloadInput.value.trim();
      if (payload) params.payload = payload;
      await send({ ...params, requestId });
      return;
    }
    if (chosen === 'extract-patches') {
      await recipeTask();
      if (dialogController.signal.aborted) return;
      sync();
      if (startDisabled()) return;
      const bad = [];
      fieldError(cropError, [cropInput], '');
      fieldError(strideError, [strideInput], '');
      const rawCrop = cropInput.value.trim();
      const rawStride = strideInput.value.trim();
      if (!wholeFromOne(rawCrop)) bad.push([cropError, [cropInput], 'Enter the patch size as a whole number of at least 1.']);
      if (rawStride && !wholeFromOne(rawStride)) bad.push([strideError, [strideInput], 'Enter the step as a whole number of at least 1, or leave it empty.']);
      else if (rawStride && wholeFromOne(rawCrop) && Number(rawStride) > Number(rawCrop)) bad.push([strideError, [strideInput], PATCH_STEP]);
      if (bad.length) {
        for (const [slot, controls, message] of bad) fieldError(slot, controls, message);
        bad[0][1][0].focus();
        return;
      }
      const params = { kind: 'extract-patches', yaml: text, split: patchData.select.value || 'train', cropSize: Number(rawCrop) };
      // Without a step the service uses the patch size: no overlap.
      if (rawStride) params.stride = Number(rawStride);
      await send({ ...params, requestId });
      return;
    }
    if (chosen === 'split-dataset') {
      await recipeTask();
      if (dialogController.signal.aborted) return;
      sync();
      if (blocked()) return;
      const bad = [];
      fieldError(shareError, [shareInput], '');
      fieldError(splitSeedError, [splitSeedInput], '');
      const rawShare = shareInput.value.trim();
      const share = /^(?:\d+\.?\d*|\.\d+)$/.test(rawShare) ? Number(rawShare) / 100 : NaN;
      if (!(share > 0 && share < 1)) bad.push([shareError, [shareInput], 'Enter the share for validation as a number more than 0 and less than 100.']);
      const rawSeed = splitSeedInput.value.trim();
      if (!/^\d+$/.test(rawSeed) || !Number.isSafeInteger(Number(rawSeed))) bad.push([splitSeedError, [splitSeedInput], 'Enter the seed as a whole number, 0 or more.']);
      if (bad.length) {
        for (const [slot, controls, message] of bad) fieldError(slot, controls, message);
        bad[0][1][0].focus();
        return;
      }
      // Paired frame folders: no groupBy, the service keeps whole scenes together.
      const grouping = splitVideo() ? {} : { groupBy: groupSelect.value || (splitLabelled() ? 'sample' : 'pair') };
      await send({ kind: 'split-dataset', yaml: text, ...grouping, validationFraction: share, seed: Number(rawSeed), requestId });
      return;
    }
    if (chosen === 'pair-registration') {
      await recipeTask();
      if (dialogController.signal.aborted) return;
      sync();
      if (blocked()) return;
      const params = { kind: 'pair-registration', yaml: text, split: pairSplitSelect.value || 'train', requestId };
      const problems = [];
      const whole = (el, key, low, high, words) => {
        const raw = el.value.trim();
        if (!raw) return;
        if (!/^\d+$/.test(raw) || Number(raw) < low || Number(raw) > high) problems.push(words);
        else params[key] = Number(raw);
      };
      whole(pairsInput, 'sampleLimit', 1, 64, 'Enter the pairs to measure as a whole number from 1 to 64.');
      whole(radiusInput, 'radius', 2, 12, 'Enter the search radius as a whole number from 2 to 12.');
      whole(seedInput, 'seed', 0, 2 ** 32 - 1, 'Enter the seed as a whole number, 0 or more.');
      if (problems.length) {
        mount(errorBox, notice('err', problems.join(' ')));
        return;
      }
      await send(params);
      return;
    }
    if (chosen === 'split-leakage') {
      const params = { kind: 'split-leakage', yaml: text, requestId };
      const raw = groupInput.value.trim();
      if (raw && (!/^\d+$/.test(raw) || Number(raw) > 200)) {
        mount(errorBox, notice('err', 'Enter the examples to list as a whole number from 0 to 200.'));
        return;
      }
      if (raw) params.groupLimit = Number(raw); // 0 is valid and sent
      await send(params);
      return;
    }
    // A press before the task was read: read it first, so a form the service refuses is not sent.
    if (taskGated() || chosen === 'predict') {
      await recipeTask();
      if (dialogController.signal.aborted) return;
      sync();
      if (blocked()) return;
    }
    if (chosen === 'task-data') {
      const problems = [];
      let auditLimit = 100;
      if (auditAll.checked) auditLimit = null;
      else {
        const raw = auditInput.value.trim();
        if (raw && (!/^\d+$/.test(raw) || !Number.isSafeInteger(Number(raw)))) problems.push('Enter the files to open in each dataset as a whole number, 0 or more, or choose Open every file.');
        else if (raw) auditLimit = Number(raw);
      }
      let sampleLimit = 25;
      const rawSamples = sampleInput.value.trim();
      if (rawSamples && (!/^\d+$/.test(rawSamples) || Number(rawSamples) > 100)) problems.push('Enter the examples to keep as a whole number from 0 to 100.');
      else if (rawSamples) sampleLimit = Number(rawSamples);
      if (problems.length) {
        mount(errorBox, notice('err', problems.join(' ')));
        return;
      }
      // null ("open every file") must reach the service: send that body exactly as written.
      await send({ kind: 'task-data', yaml: text, auditLimit, sampleLimit, requestId }, { exact: auditLimit === null });
      return;
    }
    const params = { kind: chosen, yaml: text, requestId };
    const problems = [];
    const w = weightsInput.value.trim();
    if (w) params.weights = w;
    else if (chosen !== 'loss-report') problems.push(chosen === 'evaluate' ? 'Enter the checkpoint.' : 'Enter the generator weights.');
    // The version belongs to the file the dialog was opened with; another path is sent without it.
    if (w && weightsVersion && w === String(weights).trim()) params.weightsVersion = weightsVersion;
    if (chosen === 'evaluate') {
      // No split: every validation and test dataset of the recipe.
      if (splitSelect.value) params.split = splitSelect.value;
      // The CPU is the service's default and is not sent.
      if (gpuRadio.checked) params.device = 'cuda';
      // Only what the service offers for this recipe is sent, each as ticked. Without its list, the fixed
      // rule: never for classification, segmentation or detection, where the service refuses it.
      const offered = optionalExports();
      if (offered) {
        for (const name of offered) if (exportBoxes.has(name)) params[name] = exportBoxes.get(name).checked;
      } else if (!ownScores()) params.saveImages = saveImagesBox.checked;
    }
    if (chosen === 'predict') {
      const i = inputInput.value.trim();
      if (i) params.input = i;
      else problems.push(recordsFor('predict') ? 'Enter the file of input records.' : 'Enter an input image or folder.');
      // Frame sequences only: the service refuses clipSize for single images.
      if (typeof task === 'string' && sequence()) {
        const rawClip = clipInput.value.trim();
        if (rawClip && (!/^\d+$/.test(rawClip) || Number(rawClip) < 1 || Number(rawClip) > 2 ** 31 - 1)) problems.push(CLIP_ERROR);
        else if (rawClip) params.clipSize = Number(rawClip);
      }
      // Detection only: the service refuses both keys for any other task. 0 is a valid cutoff.
      if (task === 'object_detection') {
        const rawScore = scoreInput.value.trim();
        if (rawScore) {
          const score = Number(rawScore);
          if (!/^(?:\d+\.?\d*|\.\d+)$/.test(rawScore) || !(score >= 0 && score <= 1)) problems.push('Enter the lowest score to keep as a number from 0 to 1, or leave it empty.');
          else params.scoreThreshold = score;
        }
        const rawTop = topInput.value.trim();
        if (rawTop) {
          if (!/^\d+$/.test(rawTop) || Number(rawTop) < 1 || Number(rawTop) > 2 ** 31 - 1) problems.push('Enter the most boxes per image as a whole number of at least 1, or leave it empty.');
          else params.topLimit = Number(rawTop);
        }
      }
    }
    const ints =
      chosen === 'export'
        ? [...(recordsFor('export') ? [] : [['height', heightInput, 'input height', true], ['width', widthInput, 'input width', true]]), ['opset', opsetInput, 'ONNX opset', false]]
        : chosen === 'loss-report'
          ? [['batches', batchesInput, 'number of batches', false], ['step', stepInput, 'update number', false]]
          : [];
    for (const [key, el, name, required] of ints) {
      const out = intOf(el, name, required);
      if (out.error) problems.push(out.error);
      else if (out.value !== undefined) params[key] = out.value;
    }
    if (chosen === 'loss-report') {
      const parameter = parameterInput.value.trim();
      if (parameter) params.parameter = parameter;
      const d = discInput.value.trim();
      if (d) params.discriminator = d;
    }
    let clipProblem = false;
    if (chosen === 'export') {
      // Sent only where the service offers them for this recipe: it refuses both anywhere else.
      if (anySizeOffered() && anySizeBox.checked) params.dynamicSpatial = true;
      fieldError(exportClipError, [exportClipInput], '');
      const rawClip = exportFrames() ? exportClipInput.value.trim() : '';
      if (rawClip && !wholeFromOne(rawClip)) {
        clipProblem = true;
        fieldError(exportClipError, [exportClipInput], CLIP_ERROR);
      } else if (rawClip) params.clipSize = Number(rawClip);
    }
    if (problems.length) mount(errorBox, notice('err', problems.join(' ')));
    if (clipProblem) exportClipInput.focus();
    if (problems.length || clipProblem) return;
    await send(params);
  };

  const send = async (draft, { exact = false } = {}) => {
    const key = requestKey(draft);
    if (lastSent && lastSent.key !== key) requestId = newRequestId();
    const params = { ...draft, requestId };
    lastSent = { key, body: params, exact };
    busy = true;
    startButton.setAttribute('aria-disabled', 'true');
    mount(errorBox, loading('Starting the job…'));
    try {
      created = await (exact ? api.analysisStartExact(params) : api.analysisStart(params));
      closeDialog('started');
      toast(`Started ${kindNoun(created?.kind ?? chosen)}.`, { kind: 'ok' });
      if (created?.id) {
        startedJobId = String(created.id);
        go(jobHref(created.id));
      }
    } catch (error) {
      // 400 invalid inputs, 409 "An analysis job is already running in this workspace." or a reused request id.
      // The dialog stays open with its inputs; the person presses Start job again (no retry loop here). The
      // 409 also comes for a moment after a job shows completed, while its worker still holds the slot.
      const busyHint =
        error.status === 409 && /already running/i.test(error.message)
          ? ' Wait until it has finished, then press Start job again.'
          : error.status === 409 && /checkpoint has changed/i.test(error.message)
            ? " The file was replaced after it was chosen. Close this, refresh the run and choose the checkpoint again from its Files tab."
            : '';
      if (error.offline) {
        // The request may have reached the service: "Try again" sends the same body with the same id,
        // which returns the job if it was created.
        mount(
          errorBox,
          notice(
            'err',
            h(
              'span',
              null,
              'The service did not answer, so the job may have started. ',
              h('a', { href: jobHref(null), onclick: () => closeDialog?.(null) }, 'See the analysis jobs'),
              ' before starting it again.',
            ),
            h('button', { class: 'btn btn--sm', type: 'button', onclick: () => lastSent && send(lastSent.body, { exact: lastSent.exact }) }, 'Try again'),
          ),
        );
      } else {
        // A GPU evaluation refused because the GPU is taken: the service's sentence, then what to do about it.
        const gpuHint = gpuBusyHint(params, error.message);
        mount(errorBox, notice('err', [`The job did not start: ${error.message}${busyHint}`, gpuHint ? h('p', { class: 'analysis__note' }, gpuHint) : null]));
      }
    } finally {
      busy = false;
      startButton.removeAttribute('aria-disabled');
    }
  };

  const { closed, dialog } = openDialog({
    title: dialogTitle(),
    body,
    wide: true,
    foot: (close) => {
      closeDialog = close;
      startButton = h('button', { class: 'btn btn--primary analysis__btn', type: 'button', disabled: startDisabled(), onclick: () => start() }, 'Start job');
      return [h('button', { class: 'btn', type: 'button', autofocus: true, onclick: () => close(null) }, 'Cancel'), startButton];
    },
  });
  dialogEl = dialog;
  await closed;
  dialogController.abort();
  return created;
}
