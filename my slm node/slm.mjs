#!/usr/bin/env node
/**
 * slm.mjs - the Node.js twin of slm.py.
 * Train a small GPT-style model whenever you want, keep it on Hugging Face, use it anywhere.
 * Models are interchangeable with slm.py: train in Python, resume in Node (or the other way round).
 *
 *   node slm.mjs train --repo YOU/my-slm --data my.txt --steps 2000   # train (or resume), then push
 *   node slm.mjs pull  --repo YOU/my-slm                              # download into ./model
 *   node slm.mjs gen   --repo YOU/my-slm "Once upon a time"           # generate (pulls if needed)
 *   node slm.mjs serve --dir model                                    # local web page on :7860
 *   node slm.mjs space --repo YOU/my-slm --space YOU/my-slm-demo      # free online demo (Docker Space)
 *
 * Auth: set HF_TOKEN (a *write* token from huggingface.co/settings/tokens) or run `huggingface-cli login`.
 * `gen` and `serve` need no npm packages. `train` needs: npm install   (uses WebAssembly; for native
 * speed also run: npm install @tensorflow/tfjs-node). push / pull / space need @huggingface/hub.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const CFG = 'config.json', W = 'model.safetensors', TOK = 'tokenizer.json';
const WIKI = 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/';
const SELF = fileURLToPath(import.meta.url);

// ------------------------------------------------------------------ small helpers
function mulberry32(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randnFrom(rng) {
  return () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
}
const fmtMB = (n) => (n / 1e6).toFixed(1);

// ------------------------------------------------------------------ safetensors (float32)
export function readSafetensors(file) {
  const buf = fs.readFileSync(file);
  const n = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.toString('utf8', 8, 8 + n));
  const base = 8 + n, out = {};
  for (const [k, v] of Object.entries(header)) {
    if (k === '__metadata__') continue;
    if (v.dtype !== 'F32') throw new Error(`unsupported dtype ${v.dtype} for ${k}`);
    const [s, e] = v.data_offsets;
    const ab = buf.buffer.slice(buf.byteOffset + base + s, buf.byteOffset + base + e);
    out[k] = { shape: v.shape, data: new Float32Array(ab) };
  }
  return out;
}

export function writeSafetensors(file, tensors) {
  const names = Object.keys(tensors).sort();
  const header = {};
  let off = 0;
  for (const n of names) {
    const nb = tensors[n].data.byteLength;
    header[n] = { dtype: 'F32', shape: tensors[n].shape, data_offsets: [off, off + nb] };
    off += nb;
  }
  header.__metadata__ = { format: 'np' };
  let hs = JSON.stringify(header);
  hs += ' '.repeat((8 - (hs.length % 8)) % 8);
  const hb = Buffer.from(hs, 'utf8');
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(hb.length));
  const parts = [len, hb, ...names.map((n) => Buffer.from(tensors[n].data.buffer, tensors[n].data.byteOffset, tensors[n].data.byteLength))];
  fs.writeFileSync(file, Buffer.concat(parts));
}

// ------------------------------------------------------------------ byte-level BPE tokenizer (HF tokenizer.json compatible)
const B2U = (() => {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const map = new Array(256);
  bs.forEach((b, i) => { map[b] = String.fromCharCode(cs[i]); });
  return map;
})();
const U2B = new Map(B2U.map((c, b) => [c, b]));
const PRE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const SPECIAL = '<|endoftext|>';
const toByteStr = (piece) => { let s = ''; for (const b of Buffer.from(piece, 'utf8')) s += B2U[b]; return s; };

export class Tokenizer {
  constructor(json) {
    const m = json.model;
    this.vocab = new Map(Object.entries(m.vocab));
    this.itos = [];
    for (const [t, i] of this.vocab) this.itos[i] = t;
    this.rank = new Map();
    m.merges.forEach((p, i) => this.rank.set(Array.isArray(p) ? `${p[0]} ${p[1]}` : p, i));
    this.special = new Map((json.added_tokens || []).map((t) => [t.content, t.id]));
    this.specialIds = new Set(this.special.values());
    this.cache = new Map();
    this.size = this.itos.length;
  }
  static fromFile(p) { return new Tokenizer(JSON.parse(fs.readFileSync(p, 'utf8'))); }
  bpe(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = Infinity, bi = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.rank.get(`${parts[i]} ${parts[i + 1]}`);
        if (r !== undefined && r < best) { best = r; bi = i; }
      }
      if (bi < 0) break;
      const a = parts[bi], b = parts[bi + 1], next = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { next.push(a + b); i++; } else next.push(parts[i]);
      }
      parts = next;
    }
    if (this.cache.size < 200000) this.cache.set(word, parts);
    return parts;
  }
  encode(text) {
    const ids = [];
    const chunks = text.split(SPECIAL);
    chunks.forEach((chunk, ci) => {
      if (ci > 0 && this.special.has(SPECIAL)) ids.push(this.special.get(SPECIAL));
      for (const w of chunk.match(PRE) || []) {
        for (const t of this.bpe(toByteStr(w))) { const id = this.vocab.get(t); if (id !== undefined) ids.push(id); }
      }
    });
    return ids;
  }
  decode(ids) {
    const bytes = [];
    for (const id of ids) {
      if (this.specialIds.has(id)) continue;
      for (const ch of this.itos[id] || '') { const b = U2B.get(ch); if (b !== undefined) bytes.push(b); }
    }
    return Buffer.from(bytes).toString('utf8');
  }
}

/** Train a byte-level BPE like the HF trainer does: 256 byte symbols, then the most frequent pair, until `vocabSize`. */
export function trainBPE(text, vocabSize) {
  const counts = new Map();
  for (const line of text.split('\n')) {
    for (const w of line.match(PRE) || []) { const k = toByteStr(w); counts.set(k, (counts.get(k) || 0) + 1); }
  }
  const alphabet = [...B2U].sort((a, b) => a.charCodeAt(0) - b.charCodeAt(0));
  const vocab = [SPECIAL, ...alphabet];
  const t2i = new Map(vocab.map((t, i) => [t, i]));
  const words = [...counts].map(([w, f]) => ({ s: Array.from(w, (c) => t2i.get(c)), f }));
  const KEY = 1048576;
  const pairCount = new Map(), pairWords = new Map();
  const bump = (a, b, f, wi) => {
    const k = a * KEY + b;
    pairCount.set(k, (pairCount.get(k) || 0) + f);
    if (wi >= 0) { let s = pairWords.get(k); if (!s) pairWords.set(k, (s = new Set())); s.add(wi); }
  };
  words.forEach((w, wi) => { for (let i = 0; i < w.s.length - 1; i++) bump(w.s[i], w.s[i + 1], w.f, wi); });
  const merges = [];
  while (vocab.length < vocabSize) {
    let bk = -1, bc = 0;
    for (const [k, c] of pairCount) if (c > bc || (c === bc && k < bk)) { bc = c; bk = k; }
    if (bk < 0 || bc < 1) break;
    const a = Math.floor(bk / KEY), b = bk % KEY;
    const tok = vocab[a] + vocab[b];
    let nid = t2i.get(tok);
    if (nid === undefined) { nid = vocab.length; vocab.push(tok); t2i.set(tok, nid); }
    merges.push([vocab[a], vocab[b]]);
    for (const wi of [...(pairWords.get(bk) || [])]) {
      const w = words[wi];
      let has = false;
      for (let i = 0; i < w.s.length - 1; i++) if (w.s[i] === a && w.s[i + 1] === b) { has = true; break; }
      if (!has) continue;
      for (let i = 0; i < w.s.length - 1; i++) {
        const k = w.s[i] * KEY + w.s[i + 1], c = pairCount.get(k) - w.f;
        if (c <= 0) pairCount.delete(k); else pairCount.set(k, c);
      }
      const ns = [];
      for (let i = 0; i < w.s.length; i++) {
        if (i < w.s.length - 1 && w.s[i] === a && w.s[i + 1] === b) { ns.push(nid); i++; } else ns.push(w.s[i]);
      }
      w.s = ns;
      for (let i = 0; i < ns.length - 1; i++) bump(ns[i], ns[i + 1], w.f, wi);
    }
    pairCount.delete(bk); pairWords.delete(bk);
  }
  return {
    version: '1.0', truncation: null, padding: null,
    added_tokens: [{ id: 0, content: SPECIAL, single_word: false, lstrip: false, rstrip: false, normalized: false, special: true }],
    normalizer: null,
    pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: true },
    post_processor: null,
    decoder: { type: 'ByteLevel', add_prefix_space: true, trim_offsets: true, use_regex: true },
    model: {
      type: 'BPE', dropout: null, unk_token: null, continuing_subword_prefix: null, end_of_word_suffix: null,
      fuse_unk: false, byte_fallback: false, ignore_merges: false,
      vocab: Object.fromEntries(vocab.map((t, i) => [t, i])), merges,
    },
  };
}

// ------------------------------------------------------------------ inference (plain JS, no dependencies)
function matvec(x, Wt, inD, outD, out) {
  out.fill(0);
  for (let i = 0; i < inD; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const o = i * outD;
    for (let j = 0; j < outD; j++) out[j] += xi * Wt[o + j];
  }
}
function layerNorm(x, g, b, out, D) {
  let m = 0; for (let i = 0; i < D; i++) m += x[i]; m /= D;
  let v = 0; for (let i = 0; i < D; i++) { const d = x[i] - m; v += d * d; } v /= D;
  const r = 1 / Math.sqrt(v + 1e-5);
  for (let i = 0; i < D; i++) out[i] = (x[i] - m) * r * g[i] + b[i];
}
const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

export class Model {
  constructor(dir) {
    const t = readSafetensors(path.join(dir, W));
    const c = JSON.parse(fs.readFileSync(path.join(dir, CFG), 'utf8'));
    Object.assign(this, { V: c.V, D: c.D, L: c.L, H: c.H, T: c.T });
    this.dh = this.D / this.H;
    this.w = Object.fromEntries(Object.entries(t).map(([k, v]) => [k, v.data]));
    this.layers = Array.from({ length: this.L }, (_, i) => {
      const l = {};
      for (const n of ['ln1_g', 'ln1_b', 'wqkv', 'wo', 'ln2_g', 'ln2_b', 'w1', 'b1', 'w2', 'b2']) l[n] = this.w[`l${i}_${n}`];
      return l;
    });
  }
  newCache() {
    return Array.from({ length: this.L }, () => ({ K: new Float32Array(this.H * this.T * this.dh), Vc: new Float32Array(this.H * this.T * this.dh) }));
  }
  /** Feed one token at position `pos`; returns the logits when `wantLogits`. */
  step(tok, pos, cache, wantLogits) {
    const { D, H, dh, T, V, w } = this;
    const h = new Float32Array(D), a = new Float32Array(D), qkv = new Float32Array(3 * D);
    const o = new Float32Array(D), proj = new Float32Array(D), u = new Float32Array(4 * D), u2 = new Float32Array(D);
    for (let i = 0; i < D; i++) h[i] = w.wte[tok * D + i] + w.wpe[pos * D + i];
    const sc = new Float32Array(pos + 1), inv = 1 / Math.sqrt(dh);
    this.layers.forEach((l, li) => {
      const { K, Vc } = cache[li];
      layerNorm(h, l.ln1_g, l.ln1_b, a, D);
      matvec(a, l.wqkv, D, 3 * D, qkv);
      for (let hd = 0; hd < H; hd++) {
        const base = (hd * T + pos) * dh;
        for (let d = 0; d < dh; d++) { K[base + d] = qkv[D + hd * dh + d]; Vc[base + d] = qkv[2 * D + hd * dh + d]; }
        let mx = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0; const kb = (hd * T + t) * dh;
          for (let d = 0; d < dh; d++) s += qkv[hd * dh + d] * K[kb + d];
          s *= inv; sc[t] = s; if (s > mx) mx = s;
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { sc[t] = Math.exp(sc[t] - mx); sum += sc[t]; }
        for (let d = 0; d < dh; d++) o[hd * dh + d] = 0;
        for (let t = 0; t <= pos; t++) {
          const p = sc[t] / sum, vb = (hd * T + t) * dh;
          for (let d = 0; d < dh; d++) o[hd * dh + d] += p * Vc[vb + d];
        }
      }
      matvec(o, l.wo, D, D, proj);
      for (let i = 0; i < D; i++) h[i] += proj[i];
      layerNorm(h, l.ln2_g, l.ln2_b, a, D);
      matvec(a, l.w1, D, 4 * D, u);
      for (let i = 0; i < 4 * D; i++) u[i] = gelu(u[i] + l.b1[i]);
      matvec(u, l.w2, 4 * D, D, u2);
      for (let i = 0; i < D; i++) h[i] += u2[i] + l.b2[i];
    });
    if (!wantLogits) return null;
    layerNorm(h, w.lnf_g, w.lnf_b, a, D);
    const logits = new Float32Array(V);
    for (let v = 0; v < V; v++) { let s = 0; const b = v * D; for (let i = 0; i < D; i++) s += a[i] * w.wte[b + i]; logits[v] = s; }
    return logits;
  }
  prefill(ids, cache) {
    let lg = null;
    for (let i = 0; i < ids.length; i++) lg = this.step(ids[i], i, cache, i === ids.length - 1);
    return lg;
  }
  *generate(ids, nNew, { temp = 0.8, topK = 40, topP = 0.95, repPen = 1.1, seed } = {}) {
    const rng = mulberry32(seed ?? Math.floor(Math.random() * 2 ** 31));
    ids = ids.slice(-(this.T - 1));
    let cache = this.newCache();
    let logits = this.prefill(ids, cache), pos = ids.length;
    for (let n = 0; n < nNew; n++) {
      const lg = Float32Array.from(logits);
      for (const t of new Set(ids.slice(-64))) lg[t] = lg[t] > 0 ? lg[t] / repPen : lg[t] * repPen;
      let nxt;
      if (temp <= 0) {
        nxt = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[nxt]) nxt = i;
      } else {
        for (let i = 0; i < lg.length; i++) lg[i] /= temp;
        if (topK && topK < lg.length) {
          const kth = Float32Array.from(lg).sort()[lg.length - topK];
          for (let i = 0; i < lg.length; i++) if (lg[i] < kth) lg[i] = -1e9;
        }
        let mx = -Infinity; for (const x of lg) if (x > mx) mx = x;
        const p = new Float64Array(lg.length); let s = 0;
        for (let i = 0; i < lg.length; i++) { p[i] = Math.exp(lg[i] - mx); s += p[i]; }
        for (let i = 0; i < p.length; i++) p[i] /= s;
        if (topP < 1) {
          const order = Array.from(p.keys()).sort((i, j) => p[j] - p[i]);
          let cs = 0, s2 = 0;
          for (const i of order) { const pi = p[i]; if (cs > topP) p[i] = 0; cs += pi; }
          for (const x of p) s2 += x; for (let i = 0; i < p.length; i++) p[i] /= s2;
        }
        let r = rng(), acc = 0; nxt = p.length - 1;
        for (let i = 0; i < p.length; i++) { acc += p[i]; if (r < acc) { nxt = i; break; } }
      }
      ids.push(nxt);
      yield nxt;
      if (pos >= this.T) {
        const keep = ids.slice(-Math.floor(this.T / 2));
        cache = this.newCache(); logits = this.prefill(keep, cache); pos = keep.length;
      } else { logits = this.step(nxt, pos, cache, true); pos++; }
    }
  }
}

export function loadModel(dir) { return { model: new Model(dir), tok: Tokenizer.fromFile(path.join(dir, TOK)) }; }

// ------------------------------------------------------------------ Hugging Face
export function hfToken() {
  if (process.env.HF_TOKEN) return process.env.HF_TOKEN;
  const home = process.env.HF_HOME || path.join(os.homedir(), '.cache', 'huggingface');
  try { return fs.readFileSync(path.join(home, 'token'), 'utf8').trim(); } catch { return undefined; }
}
async function hub() {
  try { return await import('@huggingface/hub'); }
  catch { throw new Error('this command needs the Hugging Face package: run `npm install` in this folder first'); }
}
export async function pull(repo, out, token = hfToken()) {
  const H = await hub();
  const r = { type: 'model', name: repo };
  fs.mkdirSync(out, { recursive: true });
  let n = 0;
  for await (const f of H.listFiles({ repo: r, accessToken: token, recursive: true })) {
    if (f.type !== 'file' || path.basename(f.path).startsWith('.')) continue;
    const blob = await H.downloadFile({ repo: r, path: f.path, accessToken: token });
    if (!blob) continue;
    const dest = path.join(out, f.path);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, Buffer.from(await blob.arrayBuffer()));
    n++;
  }
  if (!fs.existsSync(path.join(out, W))) throw new Error(`${repo} has no ${W}`);
  console.log(`pulled ${repo} -> ${out} (${n} files)`);
}
const exists409 = (e) => /409|already exist|already created/i.test(String(e && (e.message || e)));
export async function push(repo, folder, token = hfToken(), priv = false, msg = 'update') {
  const H = await hub();
  const r = { type: 'model', name: repo };
  try { await H.createRepo({ repo: r, accessToken: token, ...(priv ? { visibility: 'private' } : {}) }); }
  catch (e) { if (!exists409(e)) throw e; }
  const files = fs.readdirSync(folder).filter((f) => fs.statSync(path.join(folder, f)).isFile())
    .map((f) => ({ path: f, content: new Blob([fs.readFileSync(path.join(folder, f))]) }));
  await H.uploadFiles({ repo: r, accessToken: token, files, commitTitle: msg });
  console.log(`pushed -> https://huggingface.co/${repo}`);
}

// ------------------------------------------------------------------ data
export function cleanWikitext(s) {
  s = s.replaceAll('<unk>', '').replaceAll(' @-@ ', '-').replaceAll(' @,@ ', ',').replaceAll(' @.@ ', '.');
  s = s.replace(/ ([,.;:!?%)\]])/g, '$1').replace(/([(\[$]) /g, '$1');
  return s.replaceAll(" 's", "'s").replaceAll(" n't", "n't").replace(/ {2,}/g, ' ');
}
async function loadText(a) {
  if (a.data && a.data.length) return a.data.map((f) => fs.readFileSync(f, 'utf8')).join('\n');
  console.log('no --data given: downloading WikiText-2 (demo corpus, ~12 MB)');
  const parts = [];
  for (const f of ['train.txt', 'valid.txt']) parts.push(await (await fetch(WIKI + f)).text());
  return cleanWikitext(parts.join('\n'));
}

// ------------------------------------------------------------------ training (TensorFlow.js)
const LAYER_PARAMS = ['ln1_g', 'ln1_b', 'wqkv', 'wo', 'ln2_g', 'ln2_b', 'w1', 'b1', 'w2', 'b2'];
export function paramShapes(c) {
  const { V, D, L, T } = c;
  const s = { wte: [V, D], wpe: [T, D], lnf_g: [D], lnf_b: [D] };
  for (let i = 0; i < L; i++) Object.assign(s, {
    [`l${i}_ln1_g`]: [D], [`l${i}_ln1_b`]: [D], [`l${i}_wqkv`]: [D, 3 * D], [`l${i}_wo`]: [D, D],
    [`l${i}_ln2_g`]: [D], [`l${i}_ln2_b`]: [D], [`l${i}_w1`]: [D, 4 * D], [`l${i}_b1`]: [4 * D],
    [`l${i}_w2`]: [4 * D, D], [`l${i}_b2`]: [D],
  });
  return s;
}
export function initParams(c, seed) {
  const rnd = randnFrom(mulberry32(seed + 12345)), out = {};
  const res = 0.02 / Math.sqrt(2 * c.L);
  for (const [name, shape] of Object.entries(paramShapes(c))) {
    const n = shape.reduce((x, y) => x * y, 1), d = new Float32Array(n);
    const base = name.replace(/^l\d+_/, '');
    if (base.endsWith('_g')) d.fill(1);
    else if (/^(b1|b2|.*_b)$/.test(base)) d.fill(0);
    else { const std = base === 'wo' || base === 'w2' ? res : 0.02; for (let i = 0; i < n; i++) d[i] = rnd() * std; }
    out[name] = { shape, data: d };
  }
  return out;
}

export async function loadTF() {
  const log = console.log, warn = console.warn;
  console.log = console.warn = (...x) => { if (!String(x[0]).includes('TensorFlow.js in Node.js')) log(...x); };
  let native = false;
  try { await import('@tensorflow/tfjs-node'); native = true; } catch { /* optional */ }
  let tf;
  try { tf = await import('@tensorflow/tfjs'); }
  catch { console.log = log; console.warn = warn; throw new Error('training needs TensorFlow.js: run `npm install` in this folder first'); }
  if (native && tf.findBackend('tensorflow')) await tf.setBackend('tensorflow');
  else {
    try {
      const wasm = await import('@tensorflow/tfjs-backend-wasm');
      const dist = path.join(path.dirname(createRequire(import.meta.url).resolve('@tensorflow/tfjs-backend-wasm/package.json')), 'dist') + path.sep;
      wasm.setWasmPaths(dist);
      await tf.setBackend('wasm');
    } catch (e) { console.log(`(wasm backend unavailable: ${String(e).slice(0, 80)}; using the very slow pure-JS backend)`); await tf.setBackend('cpu'); }
  }
  await tf.ready();
  console.log = log; console.warn = warn;
  return tf;
}

/** Builds the differentiable loss for a config. Parameters are tf variables named like the safetensors keys. */
export function buildGraph(tf, c, vars) {
  const { V, D, L, H, T } = c, dh = D / H, scale = 1 / Math.sqrt(dh);
  const ln = (x, g, b) => {
    const xc = tf.sub(x, tf.mean(x, -1, true));
    const v = tf.mean(tf.square(xc), -1, true);
    return tf.add(tf.mul(tf.mul(xc, tf.rsqrt(tf.add(v, 1e-5))), g), b);
  };
  const act = (x) => {
    const x3 = tf.mul(tf.mul(x, x), x);
    return tf.mul(tf.mul(0.5, x), tf.add(1, tf.tanh(tf.mul(0.7978845608028654, tf.add(x, tf.mul(0.044715, x3))))));
  };
  // gather + scatter-add gradient done by hand (the wasm backend has no UnsortedSegmentSum kernel)
  // The wasm backend only has a fast kernel for plain (non-transposed) matmuls, so every product here, and
  // both of its gradients, is expressed as a plain matmul on an explicitly transposed copy.
  const tr = (x) => tf.transpose(x, x.rank === 2 ? [1, 0] : [0, 1, 3, 2]);
  const mm = (a, b) => tf.customGrad((A, Bm, save) => {
    save([A, Bm]);
    return { value: tf.matMul(A, Bm), gradFunc: (dy, [A_, B_]) => [tf.matMul(dy, tr(B_)), tf.matMul(tr(A_), dy)] };
  })(a, b);
  // batched matmul as a loop of fast 2D matmuls (the wasm batched kernel is ~10x slower on small matrices)
  const tr3 = (x) => tf.transpose(x, [0, 2, 1]);
  const per = (a, b) => { const bs = tf.unstack(b); return tf.stack(tf.unstack(a).map((x, i) => tf.matMul(x, bs[i]))); };
  const bmm = (a, b) => tf.customGrad((A, Bm, save) => {
    save([A, Bm]);
    return { value: per(A, Bm), gradFunc: (dy, [A_, B_]) => [per(dy, tr3(B_)), per(tr3(A_), dy)] };
  })(a, b);
  const embed = (E, ids) => tf.customGrad((Ev) => ({
    value: tf.gather(Ev, tf.tensor1d(ids, 'int32')),
    gradFunc: (dy) => {
      const d = dy.dataSync(), g = new Float32Array(V * D);
      for (let n = 0; n < ids.length; n++) { const r = ids[n] * D, s = n * D; for (let j = 0; j < D; j++) g[r + j] += d[s + j]; }
      return [tf.tensor2d(g, [V, D])];
    },
  }))(E);
  // mean cross-entropy; the value and the gradient (softmax - onehot) are computed here to avoid huge one-hot tensors
  const xent = (logits, y) => tf.customGrad((lg, save) => {
    save([lg]);
    const N = y.length, lse = tf.logSumExp(lg, 1).dataSync(), ld = lg.dataSync();
    let sum = 0; for (let n = 0; n < N; n++) sum += lse[n] - ld[n * V + y[n]];
    return {
      value: tf.scalar(sum / N),
      gradFunc: (dy, saved) => {
        const g = Float32Array.from(tf.softmax(saved[0]).dataSync()), k = dy.dataSync()[0] / N;
        for (let n = 0; n < N; n++) g[n * V + y[n]] -= 1;
        for (let i = 0; i < g.length; i++) g[i] *= k;
        return [tf.tensor2d(g, [N, V])];
      },
    };
  })(logits);
  const masks = {};
  const maskFor = (Tn) => masks[Tn] || (masks[Tn] = tf.keep(tf.tensor2d(
    Float32Array.from({ length: Tn * Tn }, (_, i) => (i % Tn > Math.floor(i / Tn) ? -1e9 : 0)), [Tn, Tn])));
  const logitsFn = (ids, B, Tn) => {
    let h = embed(vars.wte, ids);
    const pos = Tn === T ? vars.wpe : tf.slice(vars.wpe, [0, 0], [Tn, D]);
    h = tf.reshape(tf.add(tf.reshape(h, [B, Tn, D]), pos), [B * Tn, D]);
    const mask = maskFor(Tn);
    for (let i = 0; i < L; i++) {
      const p = (n) => vars[`l${i}_${n}`];
      const a = ln(h, p('ln1_g'), p('ln1_b'));
      const [q, k, v] = tf.split(mm(a, p('wqkv')), 3, 1)
        .map((t) => tf.reshape(tf.transpose(tf.reshape(t, [B, Tn, H, dh]), [0, 2, 1, 3]), [B * H, Tn, dh]));
      const att = tf.softmax(tf.add(tf.mul(bmm(q, tr3(k)), scale), mask));
      const o = tf.reshape(tf.transpose(tf.reshape(bmm(att, v), [B, H, Tn, dh]), [0, 2, 1, 3]), [B * Tn, D]);
      h = tf.add(h, mm(o, p('wo')));
      const a2 = ln(h, p('ln2_g'), p('ln2_b'));
      h = tf.add(h, tf.add(mm(act(tf.add(mm(a2, p('w1')), p('b1'))), p('w2')), p('b2')));
    }
    return mm(ln(h, vars.lnf_g, vars.lnf_b), tr(vars.wte));
  };
  return { logitsFn, loss: (ids, y, B, Tn) => xent(logitsFn(ids, B, Tn), y) };
}

export function lrAt(count, S, peak) {
  const warm = Math.max(1, Math.min(100, Math.floor(S / 10))), end = peak * 0.1;
  if (count < warm) return (peak * count) / warm;
  const prog = Math.min(1, (count - warm) / Math.max(1, S - warm));
  return end + (peak - end) * 0.5 * (1 + Math.cos(Math.PI * prog));
}

function saveModel(state, cfg, out) {
  fs.mkdirSync(out, { recursive: true });
  const t = {};
  for (const [k, v] of Object.entries(state)) t[k] = { shape: v.shape, data: v.data };
  writeSafetensors(path.join(out, W), t);
  fs.writeFileSync(path.join(out, CFG), JSON.stringify(cfg, null, 2));
}

function writeCard(out, repo, cfg) {
  const r = repo || 'USER/NAME', c = JSON.stringify({ V: cfg.V, D: cfg.D, L: cfg.L, H: cfg.H, T: cfg.T });
  fs.writeFileSync(path.join(out, 'README.md'), `---
library_name: custom
tags: [small-language-model, from-scratch]
---
# ${r}

Small GPT-style model trained with \`slm.mjs\` / \`slm.py\` (${cfg.steps_done} steps, val loss ${cfg.val_loss}).
Config: \`${c}\`

Node: \`node slm.mjs gen --repo ${r} "your prompt"\`
Python: \`python slm.py gen --repo ${r} "your prompt"\`

Continue training any time: \`node slm.mjs train --repo ${r} --data more.txt --steps 1000\`
`);
}

async function cmdTrain(a) {
  const out = a.dir, token = hfToken();
  if (a.repo && !a.fresh && !fs.existsSync(path.join(out, W))) {
    try { await pull(a.repo, out, token); console.log('resuming from the Hub copy'); }
    catch (e) { console.log(`no existing model on the Hub (${e.name || 'Error'}: ${String(e.message).slice(0, 80)}); starting fresh`); }
  }
  fs.mkdirSync(out, { recursive: true });
  const text = await loadText(a);
  const resume = fs.existsSync(path.join(out, W)) && !a.fresh;
  const tokPath = path.join(out, TOK);
  let cfg, flat = null;
  if (resume) {
    cfg = JSON.parse(fs.readFileSync(path.join(out, CFG), 'utf8'));
    flat = readSafetensors(path.join(out, W));
    console.log(`resuming: ${cfg.steps_done} steps already done`);
  } else {
    console.log('training tokenizer ...');
    fs.writeFileSync(tokPath, JSON.stringify(trainBPE(text, a.vocab)));
  }
  const tok = Tokenizer.fromFile(tokPath);
  if (!cfg) cfg = { V: tok.size, D: a.D, L: a.L, H: a.H, T: a.T, steps_done: 0, tokens_seen: 0 };
  const { D, L, H, T } = cfg, B = a.B, S = a.steps;
  if (D % H) throw new Error('--D must be divisible by --H');
  const ids = Int32Array.from(tok.encode(text));
  const cut = Math.floor(ids.length * 0.95), tr = ids.subarray(0, cut), va = ids.subarray(cut);
  if (tr.length <= T + 2) throw new Error('not enough text: need more data or a smaller --T');

  const tf = await loadTF();
  const state = flat || initParams(cfg, cfg.steps_done);
  const shapes = paramShapes(cfg);
  const names = Object.keys(shapes);
  for (const n of names) {
    if (!state[n] || state[n].data.length !== shapes[n].reduce((x, y) => x * y, 1)) throw new Error(`saved model does not match config at ${n}`);
  }
  const vars = {}, mom = {};
  for (const n of names) {
    vars[n] = tf.variable(tf.tensor(Float32Array.from(state[n].data), shapes[n]), true, n);
    mom[n] = { m: new Float32Array(state[n].data.length), v: new Float32Array(state[n].data.length) };
  }
  const nPar = names.reduce((s, n) => s + state[n].data.length, 0);
  console.log(`model: ${(nPar / 1e6).toFixed(2)}M params, ${fmtMB(nPar * 4)} MB fp32 | data: ${tr.length.toLocaleString()} train tokens | backend: ${tf.getBackend()}`);
  const g = buildGraph(tf, cfg, vars);

  const nv = Math.max(1, Math.min(Math.floor((va.length - 1) / T), 32));
  const vx = new Int32Array(nv * T), vy = new Int32Array(nv * T);
  for (let i = 0; i < nv; i++) for (let j = 0; j < T; j++) { vx[i * T + j] = va[i * T + j]; vy[i * T + j] = va[i * T + j + 1]; }
  const micro = Math.max(1, Math.min(B, a.micro));
  const evalLoss = () => {
    let tot = 0;
    for (let i0 = 0; i0 < nv; i0 += micro) {
      const nb = Math.min(micro, nv - i0);
      tot += (nb / nv) * tf.tidy(() => g.loss(vx.subarray(i0 * T, (i0 + nb) * T), vy.subarray(i0 * T, (i0 + nb) * T), nb, T).dataSync()[0]);
    }
    return tot;
  };
  const gradAcc = Object.fromEntries(names.map((n) => [n, new Float32Array(state[n].data.length)]));

  const rng = mulberry32(cfg.steps_done + 1), t0 = Date.now(), acc = [];
  const b1 = 0.9, b2 = 0.95, eps = 1e-8, wd = 0.1, base = { ...cfg };
  const checkpoint = (s) => saveModel(state, { ...base, steps_done: base.steps_done + s, tokens_seen: base.tokens_seen + s * B * T }, out);
  let vl = NaN;
  for (let s = 1; s <= S; s++) {
    const x = new Int32Array(B * T), y = new Int32Array(B * T);
    for (let b = 0; b < B; b++) {
      const i = Math.floor(rng() * (tr.length - T - 1));
      for (let j = 0; j < T; j++) { x[b * T + j] = tr[i + j]; y[b * T + j] = tr[i + j + 1]; }
    }
    let loss = 0; const grads = gradAcc;
    for (const n of names) grads[n].fill(0);
    for (let b0 = 0; b0 < B; b0 += micro) {          // gradient accumulation: same maths, a fraction of the memory
      const nb = Math.min(micro, B - b0), wgt = nb / B;
      const xs = x.subarray(b0 * T, (b0 + nb) * T), ys = y.subarray(b0 * T, (b0 + nb) * T);
      tf.tidy(() => {
        const r = tf.variableGrads(() => g.loss(xs, ys, nb, T));
        loss += wgt * r.value.dataSync()[0];
        for (const n of names) { const d = r.grads[n].dataSync(), acc = grads[n]; for (let i = 0; i < d.length; i++) acc[i] += wgt * d[i]; }
      });
    }
    let gn = 0; for (const n of names) { const d = grads[n]; for (let i = 0; i < d.length; i++) gn += d[i] * d[i]; }
    gn = Math.sqrt(gn);
    const clip = gn > 1 ? 1 / gn : 1, lr = lrAt(s - 1, S, a.lr), c1 = 1 - b1 ** s, c2 = 1 - b2 ** s;
    for (const n of names) {
      const p = state[n].data, d = grads[n], { m, v } = mom[n], decay = shapes[n].length >= 2 ? wd : 0;
      for (let i = 0; i < p.length; i++) {
        const gi = d[i] * clip;
        m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= lr * ((m[i] / c1) / (Math.sqrt(v[i] / c2) + eps) + decay * p[i]);
      }
    }
    tf.tidy(() => { for (const n of names) vars[n].assign(tf.tensor(Float32Array.from(state[n].data), shapes[n])); });
    acc.push(loss);
    if (s % 25 === 0 || s === S || s <= 2) {
      console.log(`step ${s}/${S}  loss ${(acc.reduce((x, y2) => x + y2, 0) / acc.length).toFixed(3)}  ${Math.round((s * B * T) / ((Date.now() - t0) / 1000)).toLocaleString()} tok/s`);
      acc.length = 0;
    }
    if (s % a.eval_every === 0 || s === S) {
      vl = evalLoss();
      console.log(`  val loss ${vl.toFixed(3)}  ppl ${Math.exp(vl).toFixed(1)}`);
      checkpoint(s);
    }
  }
  cfg = { ...base, steps_done: base.steps_done + S, tokens_seen: base.tokens_seen + S * B * T, val_loss: Math.round(vl * 1e4) / 1e4 };
  saveModel(state, cfg, out);
  try { fs.copyFileSync(SELF, path.join(out, 'slm.mjs')); } catch { /* not fatal */ }
  writeCard(out, a.repo, cfg);
  console.log(`saved to ${out}/  (total steps ${cfg.steps_done})`);
  if (a.repo && !a.no_push) await push(a.repo, out, token, a.private, `train: ${cfg.steps_done} steps, val loss ${cfg.val_loss}`);
}

// ------------------------------------------------------------------ gen / serve / space
async function cmdGen(a) {
  if (a.repo && !fs.existsSync(path.join(a.dir, W))) await pull(a.repo, a.dir);
  const { model, tok } = loadModel(a.dir);
  const ids = tok.encode(a.prompt);
  if (!ids.length) ids.push(0);
  process.stdout.write(a.prompt);
  const t0 = Date.now();
  let buf = [], shown = ids.slice(), n = 0;
  for (const t of model.generate(ids.slice(), a.tokens, { temp: a.temp, topK: a.top_k, seed: a.seed })) {
    buf.push(t); n++;
    const s = tok.decode(shown.concat(buf)).slice(tok.decode(shown).length);
    if (s && !s.endsWith('�')) { process.stdout.write(s); shown = shown.concat(buf); buf = []; }
  }
  console.log(`\n[${Math.round(n / ((Date.now() - t0) / 1000))} tok/s]`);
}

const PAGE = (title) => `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title}</title><style>body{font:16px system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}textarea{width:100%;font:inherit;padding:.6rem;box-sizing:border-box}
button{font:inherit;padding:.5rem 1.2rem;margin:.6rem 0}pre{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:8px;min-height:6rem}label{margin-right:1rem}</style>
<h2>${title}</h2><textarea id=p rows=3>The </textarea><br>
<label>tokens <input id=n type=number value=100 min=10 max=500></label><label>temperature <input id=t type=number step=0.1 value=0.8 min=0 max=2></label><br>
<button id=b>Generate</button><pre id=o></pre>
<script>b.onclick=async()=>{b.disabled=true;o.textContent='...';try{const r=await fetch('generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt:p.value,tokens:+n.value,temp:+t.value})});o.textContent=(await r.json()).text}catch(e){o.textContent=String(e)}b.disabled=false}</script>`;

async function cmdServe(a) {
  if (a.repo && !fs.existsSync(path.join(a.dir, W))) await pull(a.repo, a.dir);
  const { model, tok } = loadModel(a.dir);
  const title = a.repo || 'my-slm';
  const server = http.createServer((req, res) => {
    if (req.method === 'POST' && req.url.endsWith('/generate')) {
      let body = '';
      req.on('data', (c) => { body += c; if (body.length > 1e5) req.destroy(); });
      req.on('end', () => {
        try {
          const j = JSON.parse(body || '{}');
          const ids = tok.encode(String(j.prompt ?? '')); if (!ids.length) ids.push(0);
          const gen = [...model.generate(ids.slice(), Math.min(Math.max(+j.tokens || 100, 1), 500), { temp: Math.min(Math.max(+j.temp || 0, 0), 2) })];
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ text: tok.decode(ids.concat(gen)) }));
        } catch (e) { res.writeHead(400, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: String(e) })); }
      });
    } else { res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); res.end(PAGE(title)); }
  });
  server.listen(a.port, () => console.log(`serving ${title} at http://localhost:${a.port}`));
}

async function cmdSpace(a) {
  const H = await hub(), token = hfToken(), r = { type: 'space', name: a.space };
  try { await H.createRepo({ repo: r, accessToken: token, sdk: 'docker' }); } catch (e) { if (!exists409(e)) throw e; }
  const pkg = path.join(path.dirname(SELF), 'package.json');
  const files = [
    ['README.md', `---\ntitle: ${a.space.split('/').pop()}\nsdk: docker\napp_port: 7860\n---\n`],
    ['Dockerfile', `FROM node:22-slim\nWORKDIR /app\nCOPY package.json ./\nRUN npm install --omit=optional\nCOPY slm.mjs ./\nENV MODEL_REPO=${a.repo}\nEXPOSE 7860\nCMD ["sh", "-c", "node slm.mjs serve --repo $MODEL_REPO --dir /tmp/model --port 7860"]\n`],
    ['slm.mjs', fs.readFileSync(SELF)],
    ['package.json', fs.existsSync(pkg) ? fs.readFileSync(pkg) : JSON.stringify({ name: 'slm-space', type: 'module', dependencies: { '@huggingface/hub': '^2.0.0' } })],
  ].map(([p, c]) => ({ path: p, content: new Blob([c]) }));
  await H.uploadFiles({ repo: r, accessToken: token, files, commitTitle: 'deploy demo' });
  console.log(`space building at https://huggingface.co/spaces/${a.space}`);
  console.log('if the model repo is private: add HF_TOKEN as a secret in the Space settings');
}

// ------------------------------------------------------------------ cli
const BOOL = new Set(['fresh', 'no_push', 'private', 'help']);
const NUM = new Set(['steps', 'B', 'T', 'D', 'L', 'H', 'vocab', 'lr', 'eval_every', 'micro', 'tokens', 'temp', 'top_k', 'seed', 'port']);
export function parseArgs(argv) {
  const a = { _: [], data: [] };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (!t.startsWith('--')) { a._.push(t); continue; }
    const k = t.slice(2).replace(/-/g, '_');
    if (BOOL.has(k)) a[k] = true;
    else if (k === 'data') { while (argv[i + 1] && !argv[i + 1].startsWith('--')) a.data.push(argv[++i]); }
    else { const v = argv[++i]; a[k] = NUM.has(k) ? Number(v) : v; }
  }
  return a;
}
const USAGE = `slm.mjs - train a small model, keep it on Hugging Face, use it anywhere
  train  --repo U/N --data f.txt --steps 1000 [--dir model --fresh --no-push --private --D 192 --L 5 --H 3 --T 256 --B 32 --vocab 4096 --lr 0.002 --eval-every 100 --micro 8]
  pull   --repo U/N [--dir model]
  gen    "prompt" [--repo U/N --dir model --tokens 100 --temp 0.8 --top_k 40 --seed 1]
  serve  [--repo U/N --dir model --port 7860]
  space  --repo U/N --space U/DEMO`;

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  const a = parseArgs(rest);
  const d = { dir: 'model', steps: 1000, D: 192, L: 5, H: 3, T: 256, B: 32, vocab: 4096, lr: 2e-3, eval_every: 100, micro: 8, tokens: 100, temp: 0.8, top_k: 40, port: 7860 };
  for (const [k, v] of Object.entries(d)) if (a[k] === undefined) a[k] = v;
  a.prompt = a._[0] ?? 'The ';
  if (cmd === 'train') await cmdTrain(a);
  else if (cmd === 'pull') { if (!a.repo) throw new Error('--repo is required'); await pull(a.repo, a.dir); }
  else if (cmd === 'gen') await cmdGen(a);
  else if (cmd === 'serve') await cmdServe(a);
  else if (cmd === 'space') { if (!a.repo || !a.space) throw new Error('--repo and --space are required'); await cmdSpace(a); }
  else { console.log(USAGE); process.exit(cmd && cmd !== '--help' ? 1 : 0); }
}

const isMain = (() => {
  try { return !!process.argv[1] && fs.realpathSync(process.argv[1]) === fs.realpathSync(SELF); } catch { return false; }
})();
if (isMain) {
  main().then(() => { if (process.argv[2] !== 'serve') process.exit(0); })
    .catch((e) => {
      const m = String(e.message);
      const hint = /fetch failed|ENOTFOUND|ECONNREFUSED|ETIMEDOUT/i.test(m) ? ' (could not reach huggingface.co: check your internet connection)'
        : /401|403|unauthorized|forbidden/i.test(m) ? ' (check HF_TOKEN: it must be a Write token)' : '';
      console.error(`error: ${m}${hint}`);
      process.exit(1);
    });
}
