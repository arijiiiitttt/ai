# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tokenizers"]
# ///
"""Run the tiny language model. Needs only: pip install numpy tokenizers
Usage: python run.py "The history of the Roman Empire" --tokens 120 --temp 0.8
"""
import argparse, math, time
import numpy as np
from tokenizers import Tokenizer

F = np.float32


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


class Model:
    def __init__(self, path):
        z = np.load(path)
        self.V, self.D, self.L, self.H, self.T = [int(t) for t in z["cfg"]]
        self.dh = self.D // self.H

        def get(k):
            if k + "__q" in z.files:
                return z[k + "__q"].astype(F) * z[k + "__s"].astype(F)
            return z[k].astype(F)

        self.wte, self.wpe = get("wte"), get("wpe")
        self.lnf_g, self.lnf_b = get("lnf_g"), get("lnf_b")
        names = ["ln1_g", "ln1_b", "wqkv", "wo", "ln2_g", "ln2_b", "w1", "b1", "w2", "b2"]
        self.layers = [{n: get(f"l{i}_{n}") for n in names} for i in range(self.L)]

    def new_cache(self):
        return [(np.zeros((self.H, self.T, self.dh), F), np.zeros((self.H, self.T, self.dh), F))
                for _ in range(self.L)]

    def forward(self, ids, cache, pos, all_logits=False):
        n = len(ids)
        h = self.wte[ids] + self.wpe[pos:pos + n]
        H, dh, D = self.H, self.dh, self.D
        mask = np.triu(np.full((n, pos + n), -1e9, F), k=pos + 1)
        for l, (K, Vc) in zip(self.layers, cache):
            a = ln(h, l["ln1_g"], l["ln1_b"])
            q, k, v = np.split(a @ l["wqkv"], 3, -1)
            q, k, v = [t.reshape(n, H, dh).transpose(1, 0, 2) for t in (q, k, v)]
            K[:, pos:pos + n], Vc[:, pos:pos + n] = k, v
            att = q @ K[:, :pos + n].transpose(0, 2, 1) / math.sqrt(dh) + mask
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            o = (att @ Vc[:, :pos + n]).transpose(1, 0, 2).reshape(n, D)
            h = h + o @ l["wo"]
            a = ln(h, l["ln2_g"], l["ln2_b"])
            h = h + gelu(a @ l["w1"] + l["b1"]) @ l["w2"] + l["b2"]
        h = ln(h if all_logits else h[-1:], self.lnf_g, self.lnf_b)
        return h @ self.wte.T

    def generate(self, ids, n_new, temp=0.8, top_k=40, top_p=0.95, rep_pen=1.1, seed=None):
        rng = np.random.default_rng(seed)
        ids = list(ids)[-(self.T - 1):]
        cache, pos = self.new_cache(), 0
        logits = self.forward(np.array(ids), cache, 0)[-1]
        pos = len(ids)
        out = []
        for _ in range(n_new):
            lg = logits.copy()
            for t in set(ids[-64:]):
                lg[t] = lg[t] / rep_pen if lg[t] > 0 else lg[t] * rep_pen
            if temp <= 0:
                nxt = int(lg.argmax())
            else:
                lg = lg / temp
                if top_k:
                    kth = np.partition(lg, -top_k)[-top_k]
                    lg[lg < kth] = -1e9
                p = np.exp(lg - lg.max()); p /= p.sum()
                if top_p < 1:
                    o = np.argsort(-p); cs = np.cumsum(p[o])
                    p[o[cs - p[o] > top_p]] = 0; p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            out.append(nxt); ids.append(nxt)
            if pos >= self.T:  # context full: restart from the most recent half
                keep = ids[-self.T // 2:]
                cache = self.new_cache()
                logits = self.forward(np.array(keep), cache, 0)[-1]
                pos = len(keep)
            else:
                logits = self.forward(np.array([nxt]), cache, pos)[-1]
                pos += 1
            yield nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=" = The history of ")
    ap.add_argument("--model", default="model_int8.npz")
    ap.add_argument("--tok", default="tokenizer.json")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    m, tok = Model(a.model), Tokenizer.from_file(a.tok)
    ids = tok.encode(a.prompt).ids
    print(a.prompt, end="", flush=True)
    t0, buf, n = time.time(), [], 0
    for t in m.generate(ids, a.tokens, a.temp, a.top_k, seed=a.seed):
        buf.append(t); n += 1
        s = tok.decode(ids + buf)[len(tok.decode(ids)):]
        if s and not s.endswith("�"):
            print(s, end="", flush=True); ids = ids + buf; buf = []
    print(f"\n\n[{n / (time.time() - t0):.0f} tokens/sec on CPU]")


if __name__ == "__main__":
    main()
