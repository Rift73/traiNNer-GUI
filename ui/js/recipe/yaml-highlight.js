// Text-level YAML highlighting and path locating. Nothing here parses YAML into data (the backend
// does that); it only reads the shape of the text: indentation, sequence dashes, keys, quotes, flow
// brackets and block scalars. Both functions share one line tokenizer so they agree on what is a key.

const NUM_RE =
  /^(?:[-+]?(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][-+]?\d+)?|[-+]?0x[0-9a-fA-F_]+|0o[0-7_]+|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$/;
const BOOL_RE = /^(?:true|True|TRUE|false|False|FALSE)$/;
const NULL_RE = /^(?:~|null|Null|NULL)$/;
const WS = /[ \t]/;

function classifyPlain(word) {
  if (NULL_RE.test(word)) return 'null';
  if (BOOL_RE.test(word)) return 'bool';
  if (NUM_RE.test(word)) return 'num';
  return 'str';
}

/** Index of the closing quote, or -1 when the string continues on the next line. */
function quoteEnd(line, from, quote) {
  for (let j = from; j < line.length; j += 1) {
    const c = line[j];
    if (quote === '"' && c === '\\') {
      j += 1;
      continue;
    }
    if (c === quote) {
      // '' inside a single-quoted string is an escaped quote, not the end
      if (quote === "'" && line[j + 1] === "'") {
        j += 1;
        continue;
      }
      return j;
    }
  }
  return -1;
}

/** A block-context key starting at i: {key, space, end} or null. */
function matchKey(line, i) {
  const c = line[i];
  if (c === undefined) return null;
  let j;
  if (c === '"' || c === "'") {
    const end = quoteEnd(line, i + 1, c);
    if (end < 0) return null;
    j = end + 1;
    let k = j;
    while (k < line.length && WS.test(line[k])) k += 1;
    if (line[k] !== ':' || (k + 1 < line.length && !WS.test(line[k + 1]))) return null;
    return { key: line.slice(i, j), space: line.slice(j, k), end: k + 1 };
  }
  // indicator characters cannot start a plain key
  if ('[]{},#&*!|>%@`-?'.includes(c) && !(c === '-' && line[i + 1] && !WS.test(line[i + 1]))) return null;
  for (j = i; j < line.length; j += 1) {
    if (line[j] === '#' && j > i && WS.test(line[j - 1])) return null; // a comment came first
    if (line[j] === ':' && (j + 1 === line.length || WS.test(line[j + 1]))) break;
  }
  if (j >= line.length) return null;
  let k = j;
  while (k > i && WS.test(line[k - 1])) k -= 1;
  if (k === i) return null;
  return { key: line.slice(i, k), space: line.slice(k, j), end: j + 1 };
}

/**
 * Tokens of one line, as [class, text] pairs whose texts concatenate to the line exactly.
 * `st` carries what continues across lines: open flow brackets, an open quote, a block scalar.
 * Sets st.cont = true when the line is the continuation of something opened earlier (so it is
 * not a structural line for locatePath).
 */
function lineTokens(line, st) {
  const out = [];
  const push = (cls, text) => {
    if (text) out.push([cls, text]);
  };
  const n = line.length;
  const indentLen = /^[ \t]*/.exec(line)[0].length;
  st.cont = false;

  if (st.block) {
    if (line.trim() === '' || indentLen > st.block.indent) {
      st.cont = true;
      push('str', line);
      return out;
    }
    st.block = null;
  }

  let i = 0;
  if (st.quote) {
    st.cont = true;
    const end = quoteEnd(line, 0, st.quote);
    if (end < 0) {
      push('str', line);
      return out;
    }
    push('str', line.slice(0, end + 1));
    st.quote = null;
    i = end + 1;
  } else {
    if (st.flow > 0) st.cont = true;
    push('', line.slice(0, indentLen));
    i = indentLen;
    if (st.flow === 0) {
      if ((line.startsWith('---', i) || line.startsWith('...', i)) && (i + 3 === n || WS.test(line[i + 3]))) {
        push('punct', line.slice(i, i + 3));
        i += 3;
      } else {
        // sequence dashes, possibly several ("- - item")
        while (line[i] === '-' && (i + 1 === n || WS.test(line[i + 1]))) {
          push('punct', '-');
          i += 1;
          const start = i;
          while (i < n && WS.test(line[i])) i += 1;
          push('', line.slice(start, i));
        }
        const key = matchKey(line, i);
        if (key) {
          push('key', key.key);
          push('', key.space);
          push('punct', ':');
          i = key.end;
        }
      }
    }
  }

  // the value part
  while (i < n) {
    const c = line[i];
    if (WS.test(c)) {
      const start = i;
      while (i < n && WS.test(line[i])) i += 1;
      push('', line.slice(start, i));
      continue;
    }
    // '#' starts a comment only at line start or after whitespace; "a#b" and "'#'" are data
    if (c === '#' && (i === 0 || WS.test(line[i - 1]))) {
      push('comment', line.slice(i));
      return out;
    }
    if (c === '"' || c === "'") {
      const end = quoteEnd(line, i + 1, c);
      if (end < 0) {
        push('str', line.slice(i));
        st.quote = c;
        return out;
      }
      let k = end + 1;
      while (k < n && WS.test(line[k])) k += 1;
      const isFlowKey = st.flow > 0 && line[k] === ':';
      push(isFlowKey ? 'key' : 'str', line.slice(i, end + 1));
      i = end + 1;
      continue;
    }
    if (c === '[' || c === '{') {
      st.flow += 1;
      push('punct', c);
      i += 1;
      continue;
    }
    if (c === ']' || c === '}') {
      st.flow = Math.max(0, st.flow - 1);
      push('punct', c);
      i += 1;
      continue;
    }
    if (st.flow > 0 && (c === ',' || (c === ':' && (i + 1 === n || /[\s,\]}]/.test(line[i + 1]))))) {
      push('punct', c);
      i += 1;
      continue;
    }
    if (c === '!' || c === '&' || c === '*') {
      let j = i + 1;
      while (j < n && !WS.test(line[j]) && !(st.flow > 0 && ',[]{}'.includes(line[j]))) j += 1;
      push(c === '!' ? 'tag' : 'anchor', line.slice(i, j));
      i = j;
      continue;
    }
    if ((c === '|' || c === '>') && st.flow === 0) {
      const header = /^[|>][-+0-9]*/.exec(line.slice(i))[0];
      push('punct', header);
      i += header.length;
      st.block = { indent: indentLen };
      continue;
    }
    // plain scalar: runs to a comment, the line end, or (in flow) an indicator
    let j = i;
    while (j < n) {
      const d = line[j];
      if (d === '#' && j > i && WS.test(line[j - 1])) break;
      if (st.flow > 0 && ',[]{}'.includes(d)) break;
      if (st.flow > 0 && d === ':' && (j + 1 === n || /[\s,\]}]/.test(line[j + 1]))) break;
      j += 1;
    }
    if (j === i) j = i + 1;
    let k = j;
    while (k > i && WS.test(line[k - 1])) k -= 1;
    const word = line.slice(i, k);
    const isFlowKey = st.flow > 0 && line[j] === ':';
    push(isFlowKey ? 'key' : classifyPlain(word), word);
    push('', line.slice(k, j));
    i = j;
  }
  return out;
}

/** Tokenize a whole document: one array of [class, text] per line ('\n' split). */
export function tokenizeYaml(text) {
  const st = { flow: 0, quote: null, block: null, cont: false };
  return String(text ?? '').split('\n').map((line) => lineTokens(line, st));
}

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;' };
const escapeHtml = (s) => s.replace(/[&<>]/g, (c) => ESC[c]);

/**
 * Highlighted markup: one block `<span class="yaml__line">` per line, tokens as
 * `<span class="yaml__<kind>">`. Kinds: comment key str num bool null tag anchor punct.
 * HTML strings (not nodes) because the editor re-highlights on every keystroke and one
 * innerHTML assignment is far cheaper than building thousands of nodes.
 */
export function highlightHtml(text) {
  let html = '';
  for (const tokens of tokenizeYaml(text)) {
    html += '<span class="yaml__line">';
    for (const [cls, part] of tokens) html += cls ? `<span class="yaml__${cls}">${escapeHtml(part)}</span>` : escapeHtml(part);
    html += '</span>';
  }
  return html;
}

/** Highlighted `<pre>` for read-only display. */
export function highlightedPre(text, className = 'yaml__pre') {
  const pre = document.createElement('pre');
  pre.className = className;
  pre.innerHTML = highlightHtml(text);
  return pre;
}

function unquoteKey(key) {
  if (key.startsWith('"')) {
    try {
      return JSON.parse(key);
    } catch {
      return key.slice(1, -1);
    }
  }
  if (key.startsWith("'")) return key.slice(1, -1).replace(/''/g, "'");
  return key;
}

/**
 * Structural marks of the document in order: {kind: 'dash'|'key', col, line, key?, inline}.
 * "- type: x" yields a dash at the dash column and a key at the key column, so a mapping that
 * starts on a dash line lines up with its later keys.
 */
function structure(text) {
  const marks = [];
  const st = { flow: 0, quote: null, block: null, cont: false };
  String(text ?? '')
    .split('\n')
    .forEach((line, lineNo) => {
      const tokens = lineTokens(line, st);
      if (st.cont) return;
      let col = 0;
      for (let t = 0; t < tokens.length; t += 1) {
        const [cls, part] = tokens[t];
        if (cls === '') {
          col += part.length;
          continue;
        }
        if (cls === 'punct' && part === '-') {
          const rest = tokens.slice(t + 1).some(([c, p]) => c && c !== 'comment' && !(c === 'punct' && p === '-') && c !== 'key');
          marks.push({ kind: 'dash', col, line: lineNo, inline: rest });
          col += 1;
          continue;
        }
        if (cls === 'key') {
          const after = tokens.slice(t + 2); // skip the ':' (and any spaces) after the key
          const inline = after.some(([c, p]) => c && c !== 'comment' && !(c === 'punct' && p === ':'));
          marks.push({ kind: 'key', col, line: lineNo, key: unquoteKey(part), inline });
        }
        break;
      }
    });
  return marks;
}

function blockEnd(marks, index) {
  const { col, kind, line } = marks[index];
  let j = index + 1;
  // "key:" followed by a sequence at the SAME indent is legal YAML and common
  if (kind === 'key' && j < marks.length && marks[j].kind === 'dash' && marks[j].col === col && marks[j].line !== line) {
    while (j < marks.length && (marks[j].col > col || (marks[j].col === col && marks[j].kind === 'dash'))) j += 1;
    return j;
  }
  while (j < marks.length && marks[j].col > col) j += 1;
  return j;
}

/**
 * Zero-based line where an option is written, or null.
 * @param {string} yamlText
 * @param {Array<string|number>} segments e.g. ['train', 'losses', 1, 'type'] (see parsePath)
 * @param {{partial?: boolean}} [options] partial: return the deepest line found instead of null
 *   (used for issue marks: an issue about a missing key lands on its parent)
 */
export function locatePath(yamlText, segments, { partial = false } = {}) {
  if (!Array.isArray(segments) || segments.length === 0) return null;
  const marks = structure(yamlText);
  let lo = 0;
  let hi = marks.length;
  let found = null;
  let inlineValue = false;
  for (const seg of segments) {
    if (lo >= hi) {
      // a flow value written on one line ("dataroot_gt: [a, b]") holds its items on that line
      if (inlineValue && found !== null) return found;
      return partial ? found : null;
    }
    const childCol = marks[lo].col;
    let hit = -1;
    if (typeof seg === 'number') {
      let count = -1;
      for (let t = lo; t < hi; t += 1) {
        if (marks[t].col === childCol && marks[t].kind === 'dash') {
          count += 1;
          if (count === seg) {
            hit = t;
            break;
          }
        }
      }
    }
    if (hit < 0) {
      const name = String(seg);
      for (let t = lo; t < hi; t += 1) {
        if (marks[t].col === childCol && marks[t].kind === 'key' && marks[t].key === name) {
          hit = t;
          break;
        }
      }
    }
    if (hit < 0) return partial ? found : null;
    found = marks[hit].line;
    inlineValue = marks[hit].inline;
    lo = hit + 1;
    hi = blockEnd(marks, hit);
    // a dash's mapping may start on the dash line itself: its first key is the next mark
  }
  return found;
}
