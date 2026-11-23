// Schema reading for the options form. Pure functions, no DOM.
// The backend's JSON Schema is the only source of descriptions, types, defaults and enums: the form
// shows what it says and nothing else. fieldKind() boils a schema node down to a small "kind" record
// that fields.js knows how to draw.

import { api } from '../api.js';

/** Load the schema (cached by api.js) and wrap it in a resolver. */
export async function loadSchema() {
  return makeSchema(withNestedServiceOptions(await api.schema()));
}

/**
 * @param {object} root the JSON Schema document
 * @returns {{root: object, deref: (n: object) => object, def: (name: string) => object|null,
 *   rootNode: object, kind: (prop: object) => object, props: (node: object) => object,
 *   variantFor: (kind: object, tag: any) => object|null}}
 */
export function makeSchema(root) {
  const lookup = (ref) => {
    if (typeof ref !== 'string' || !ref.startsWith('#')) return null;
    let node = root;
    for (const raw of ref.slice(1).split('/')) {
      if (raw === '') continue;
      const part = raw.replace(/~1/g, '/').replace(/~0/g, '~');
      node = node && typeof node === 'object' ? node[part] : undefined;
    }
    return node && typeof node === 'object' ? node : null;
  };
  // Follow $ref chains (and single-entry allOf wrappers some generators emit) to the real node.
  const deref = (node) => {
    let current = node;
    for (let guard = 0; current && guard < 32; guard += 1) {
      if (typeof current.$ref === 'string') current = lookup(current.$ref);
      else if (Array.isArray(current.allOf) && current.allOf.length === 1 && !current.type && !current.properties) current = current.allOf[0];
      else break;
    }
    return current ?? {};
  };
  const defs = root.$defs ?? root.definitions ?? {};
  const api_ = {
    root,
    deref,
    lookup,
    def: (name) => (defs[name] && typeof defs[name] === 'object' ? defs[name] : null),
    rootNode: deref(root),
    kind: (prop) => fieldKind(prop, deref),
    /** properties of an object node ({} when it has none) */
    props: (node) => {
      let n = deref(node);
      // A section that may be left out is written as "nothing, or this object" (anyOf / oneOf with null and
      // one object, as `logger` is): its options are those of that one object.
      if (n && !n.properties) {
        const members = (Array.isArray(n.anyOf) ? n.anyOf : Array.isArray(n.oneOf) ? n.oneOf : []).filter((m) => m && m.type !== 'null');
        if (members.length === 1) n = deref(members[0]);
      }
      return n && n.properties && typeof n.properties === 'object' ? n.properties : {};
    },
    /** The object def a discriminated union uses for a given tag value (e.g. network_g type). */
    variantFor(kind, tag) {
      if (!kind || kind.kind !== 'union' || typeof tag !== 'string') return null;
      const lower = tag.toLowerCase();
      if (kind.mapping) {
        const hit = kind.mapping[tag] ?? kind.mapping[lower];
        if (hit) return lookup(hit);
      }
      for (const ref of kind.refs) {
        const node = lookup(ref);
        if (variantTags(node, kind.propertyName).some((t) => String(t).toLowerCase() === lower)) return node;
      }
      return null;
    },
  };
  return api_;
}

/** Tag values a union member accepts on its discriminator property. */
export function variantTags(node, propertyName = 'type') {
  const tagNode = node?.properties?.[propertyName];
  if (!tagNode) return [];
  if (Array.isArray(tagNode.enum)) return tagNode.enum;
  if ('const' in tagNode) return [tagNode.const];
  // A tag that is matched without regard to case is a pattern; `examples` then names its spelling.
  if (typeof tagNode.pattern === 'string' && Array.isArray(tagNode.examples)) return tagNode.examples.filter((x) => typeof x === 'string');
  return [];
}

const isNullNode = (n) => n && (n.type === 'null' || (Array.isArray(n.enum) && n.enum.length === 1 && n.enum[0] === null));
const NUMERIC = new Set(['int', 'number']);
const unique = (list) => [...new Set(list)];
/**
 * The values a text option usually takes, as the schema names them (`examples`). They are offered, never
 * required: any other text stays valid. The empty text counts when it is listed (it is a value of its own).
 */
const examplesOf = (node) => (Array.isArray(node?.examples) ? unique(node.examples.filter((x) => typeof x === 'string')) : []);

/**
 * Normalize a property schema into a field kind.
 * Result: {kind, nullable, hasDefault, default, description, ...kind-specific}
 *   bool | enum {options} | int | number | string {suggestions?} | pair {itemType}
 *   | enumList {options} | list {itemType, itemSuggestions?} | literalOrNumber {literals, numType}
 *   | stringOrList | object {node} | freeObject | map {item} | union {propertyName, mapping, refs}
 *   | any (no type information: control follows the current value) | json (complex: JSON text)
 */
export function fieldKind(prop, deref) {
  const outer = prop && typeof prop === 'object' ? prop : {};
  const inner = deref(outer);
  // Pydantic puts description/default on the property and the shape behind $ref, so read the
  // outer node first and fall back to the target.
  const description = typeof outer.description === 'string' ? outer.description : typeof inner.description === 'string' ? inner.description : '';
  const hasDefault = 'default' in outer || 'default' in inner;
  const info = { nullable: false };
  const shape = classify(inner, deref, info, 0);
  return {
    ...shape,
    nullable: info.nullable,
    hasDefault,
    default: 'default' in outer ? outer.default : inner.default,
    description,
  };
}

function classify(node, deref, info, depth) {
  if (!node || typeof node !== 'object' || depth > 8) return { kind: 'json' };
  if (Array.isArray(node.allOf) && node.allOf.length === 1) return classify(deref(node.allOf[0]), deref, info, depth + 1);

  let variants = null;
  if (Array.isArray(node.anyOf)) variants = node.anyOf;
  else if (Array.isArray(node.oneOf)) variants = node.oneOf;
  else if (Array.isArray(node.type)) variants = node.type.map((t) => ({ ...node, type: t }));
  if (variants) {
    const nonNull = variants.filter((v) => !isNullNode(deref(v)));
    if (nonNull.length < variants.length) info.nullable = true;
    if (nonNull.length === 0) return { kind: 'json' };
    // Discriminated union (network_g, loss items): every member is an object def with a tag property.
    const propertyName = node.discriminator?.propertyName ?? 'type';
    if (nonNull.length > 1 && nonNull.every((v) => typeof v.$ref === 'string' && variantTags(deref(v), propertyName).length)) {
      return { kind: 'union', propertyName, mapping: node.discriminator?.mapping ?? null, refs: nonNull.map((v) => v.$ref) };
    }
    if (nonNull.length === 1) return classify(deref(nonNull[0]), deref, info, depth + 1);
    const parts = nonNull.map((v) => classify(deref(v), deref, { nullable: false }, depth + 1));
    if (parts.every((p) => p.kind === 'enum')) return { kind: 'enum', options: unique(parts.flatMap((p) => p.options)) };
    const literals = parts.filter((p) => p.kind === 'enum' && p.options.every((o) => typeof o === 'string'));
    const nums = parts.filter((p) => NUMERIC.has(p.kind));
    if (nums.length && literals.length + nums.length === parts.length) {
      return { kind: 'literalOrNumber', literals: unique(literals.flatMap((p) => p.options)), numType: nums.some((p) => p.kind === 'number') ? 'number' : 'int' };
    }
    if (parts.every((p) => NUMERIC.has(p.kind))) return { kind: parts.some((p) => p.kind === 'number') ? 'number' : 'int' };
    const hasString = parts.some((p) => p.kind === 'string');
    if (parts.length === 2 && hasString && parts.some((p) => p.kind === 'list' && p.itemType === 'string')) return { kind: 'stringOrList' };
    // A free string that also lists known values: keep the free string, offer the values.
    if (hasString && parts.every((p) => p.kind === 'string' || p.kind === 'enum')) {
      return { kind: 'string', suggestions: unique(parts.flatMap((p) => p.options ?? p.suggestions ?? []).filter((o) => typeof o === 'string')) };
    }
    if (parts.some((p) => p.kind === 'freeObject' || p.kind === 'object') && parts.every((p) => ['freeObject', 'object', 'map'].includes(p.kind))) return { kind: 'freeObject' };
    return { kind: 'json' };
  }

  if (Array.isArray(node.enum)) {
    if (node.enum.length && node.enum.every((o) => typeof o === 'boolean')) return { kind: 'bool' };
    return { kind: 'enum', options: node.enum.filter((o) => o !== null) };
  }
  if ('const' in node) return { kind: 'enum', options: [node.const] };

  switch (node.type) {
    case 'boolean':
      return { kind: 'bool' };
    case 'integer':
      return { kind: 'int' };
    case 'number':
      return { kind: 'number' };
    case 'string': {
      const examples = examplesOf(node);
      return examples.length ? { kind: 'string', suggestions: examples } : { kind: 'string' };
    }
    case 'array': {
      if (Array.isArray(node.prefixItems)) {
        const items = node.prefixItems.map((p) => classify(deref(p), deref, { nullable: false }, depth + 1));
        if (items.length === 2 && items.every((p) => NUMERIC.has(p.kind))) {
          return { kind: 'pair', itemType: items.every((p) => p.kind === 'int') ? 'int' : 'number' };
        }
        return { kind: 'json' };
      }
      if (!node.items || typeof node.items !== 'object') return { kind: 'json' };
      const item = classify(deref(node.items), deref, { nullable: false }, depth + 1);
      if (item.kind === 'enum' && item.options.length && item.options.every((o) => typeof o === 'string')) return { kind: 'enumList', options: item.options };
      // A list of texts whose usual values the schema names: each item is chosen or typed on its own, so
      // the order, repeats and any other spelling stay as the recipe has them.
      if (item.kind === 'string' && item.suggestions?.length) return { kind: 'list', itemType: 'string', itemSuggestions: item.suggestions };
      if (item.kind === 'int' || item.kind === 'number' || item.kind === 'string') return { kind: 'list', itemType: item.kind };
      // A list of small records with plain fields (task_options.categories: {id, name}): a table.
      if (item.kind === 'object') {
        const itemNode = deref(node.items);
        const required = Array.isArray(itemNode.required) ? itemNode.required : [];
        const columns = Object.entries(itemNode.properties ?? {}).map(([key, p]) => {
          const k = classify(deref(p), deref, { nullable: false }, depth + 1);
          return { key, type: k.kind, description: typeof p.description === 'string' ? p.description : '', required: required.includes(key) };
        });
        if (columns.length && columns.length <= 6 && columns.every((col) => ['int', 'number', 'string'].includes(col.type))) return { kind: 'objectList', columns };
      }
      return { kind: 'json', item: item.kind };
    }
    case 'object':
      if (node.properties && Object.keys(node.properties).length) return { kind: 'object', node };
      if (node.additionalProperties && typeof node.additionalProperties === 'object') return { kind: 'map', item: node.additionalProperties };
      return { kind: 'freeObject' };
    default:
      break;
  }
  if (node.properties && typeof node.properties === 'object') return { kind: 'object', node };
  // No type at all ({} or description only): anything goes, so draw from the current value.
  return { kind: 'any' };
}

/**
 * Options the service accepts and checks although GET /api/schema does not list them, as schema
 * nodes in the service's own words (studio/API.md); one line per option. The schema's own definition
 * always wins: an entry is used only while the schema lacks its key, so it drops out by itself once
 * the schema lists the option.
 */
export const SERVICE_OPTIONS = {
  checkpoint_rng: { type: 'boolean', default: false, description: 'Saves the random state at each numbered checkpoint so a resumed run continues the same random sequence. Needs a single training process, at most one GPU, zero data workers, no prefetch mode and prepared data; the check lists what is missing.' },
  // A dotted key is an option inside a section (here `logger`); loadSchema adds it to that section's definition.
  'logger.training_sample_steps': {
    anyOf: [{ type: 'array', items: { type: 'integer', minimum: 1 } }, { type: 'null' }],
    default: null,
    description: "Saves the first sample fed to the model at each of these updates, for example [1, 500]: its tensors and previews, shown in the run's Training samples tab. Off when empty. Each capture pauses the trainer briefly and writes the sample to disk.",
  },
  'path.neuflow_weights': {
    anyOf: [{ type: 'string' }, { type: 'null' }],
    default: null,
    description: 'NeuFlow v2 weights for RDUAL generators without embedded flow. Accepts .safetensors or .pth, and the file must exist. Omit to download the official weights once on first use.',
  },
};

const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

/** The schema's root properties plus the root SERVICE_OPTIONS it does not list (a schema entry wins). */
export function withServiceOptions(props) {
  const out = { ...props };
  for (const [key, prop] of Object.entries(SERVICE_OPTIONS)) if (!key.includes('.') && !hasOwn(out, key)) out[key] = prop;
  return out;
}

/**
 * The schema document with the dotted SERVICE_OPTIONS (one level, `section.key`) added to the definition
 * their section refers to, when that definition lacks the key. Copies only what changes: the cached
 * document is not modified, and a schema that lists the option itself wins.
 */
export function withNestedServiceOptions(doc) {
  const nested = Object.entries(SERVICE_OPTIONS).filter(([key]) => key.includes('.'));
  if (!nested.length || !doc || typeof doc !== 'object') return doc;
  const defsKey = doc.$defs ? '$defs' : doc.definitions ? 'definitions' : null;
  if (!defsKey) return doc;
  const prefix = `#/${defsKey}/`;
  const schema = makeSchema(doc);
  const rootProps = schema.props(schema.rootNode);
  let defs = null;
  for (const [path, prop] of nested) {
    const [section, key] = path.split('.');
    const node = rootProps[section];
    if (!node || typeof node !== 'object') continue;
    // The section's object definition: a $ref, or a $ref inside anyOf/oneOf next to null.
    const ref = [node, ...(Array.isArray(node.anyOf) ? node.anyOf : []), ...(Array.isArray(node.oneOf) ? node.oneOf : [])]
      .map((n) => n?.$ref)
      .find((r) => typeof r === 'string' && r.startsWith(prefix));
    if (!ref) continue;
    const name = ref.slice(prefix.length);
    const current = (defs ?? doc[defsKey])[name];
    if (!current || typeof current.properties !== 'object' || hasOwn(current.properties, key)) continue;
    defs ??= { ...doc[defsKey] };
    defs[name] = { ...current, properties: { ...current.properties, [key]: prop } };
  }
  return defs ? { ...doc, [defsKey]: defs } : doc;
}

/** Kind for a value that has no schema (free-form objects, keys the schema does not know). */
export function kindFromValue(value) {
  if (typeof value === 'boolean') return { kind: 'bool', nullable: false, hasDefault: false, description: '' };
  if (typeof value === 'number') return { kind: 'number', nullable: false, hasDefault: false, description: '' };
  if (typeof value === 'string') return { kind: 'string', nullable: false, hasDefault: false, description: '' };
  return { kind: 'json', nullable: value === null, hasDefault: false, description: '' };
}

/** True when `path` exists in `root` (a key that is present, even with a null value). */
export function hasAt(root, path) {
  let node = root;
  for (const key of path) {
    if (node === null || typeof node !== 'object') return false;
    if (Array.isArray(node)) {
      if (typeof key !== 'number' && !/^\d+$/.test(String(key))) return false;
      if (Number(key) >= node.length) return false;
    } else if (!Object.prototype.hasOwnProperty.call(node, key)) return false;
    node = node[key];
  }
  return true;
}

export const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
