#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tokenizers", "safetensors", "huggingface_hub", "jax", "optax"]
# ///
"""slm.py - train a small GPT-style model whenever you want, keep it on Hugging Face, use it anywhere.

  python slm.py train --repo YOU/my-slm --data my.txt --steps 2000   # train (or resume), then push
  python slm.py pull  --repo YOU/my-slm                              # download into ./model
  python slm.py gen   --repo YOU/my-slm "Once upon a time"           # generate (pulls if needed)
  python slm.py space --repo YOU/my-slm --space YOU/my-slm-demo      # free online demo

Auth: set HF_TOKEN (a *write* token from huggingface.co/settings/tokens) or run `huggingface-cli login`.
Install: pip install numpy tokenizers safetensors huggingface_hub     (training also needs: jax optax)
"""
import argparse, json, math, os, re, shutil, sys, time
import numpy as np

CFG_NAME, W_NAME, TOK_NAME = "config.json", "model.safetensors", "tokenizer.json"
WIKI = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/"


# ---------------------------------------------------------------- weights io
def save_model(params, cfg, out):
    from safetensors.numpy import save_file
    os.makedirs(out, exist_ok=True)
    flat = {k: np.ascontiguousarray(np.asarray(v, np.float32)) for k, v in params.items() if k != "layers"}
    for i, l in enumerate(params["layers"]):
        for k, v in l.items():
            flat[f"l{i}_{k}"] = np.ascontiguousarray(np.asarray(v, np.float32))
    save_file(flat, os.path.join(out, W_NAME))
    json.dump(cfg, open(os.path.join(out, CFG_NAME), "w"), indent=2)


def load_flat(d):
    from safetensors.numpy import load_file
    return load_file(os.path.join(d, W_NAME)), json.load(open(os.path.join(d, CFG_NAME)))


def pull(repo, out, token=None):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=out, token=token or os.environ.get("HF_TOKEN"))
    print(f"pulled {repo} -> {out}")


# ---------------------------------------------------------------- numpy inference (no jax needed)
F = np.float32


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class Model:
    def __init__(self, d):
        w, c = load_flat(d)
        self.V, self.D, self.L, self.H, self.T = c["V"], c["D"], c["L"], c["H"], c["T"]
        self.dh = self.D // self.H
        self.w = w
        self.layers = [{k[len(f"l{i}_"):]: w[k] for k in w if k.startswith(f"l{i}_")} for i in range(self.L)]

    def new_cache(self):
        z = lambda: np.zeros((self.H, self.T, self.dh), F)
        return [(z(), z()) for _ in range(self.L)]

    def forward(self, ids, cache, pos):
        w, n, H, dh, D = self.w, len(ids), self.H, self.dh, self.D
        h = w["wte"][ids] + w["wpe"][pos:pos + n]
        mask = np.triu(np.full((n, pos + n), -1e9, F), k=pos + 1)
        for l, (K, Vc) in zip(self.layers, cache):
            a = _ln(h, l["ln1_g"], l["ln1_b"])
            q, k, v = [t.reshape(n, H, dh).transpose(1, 0, 2) for t in np.split(a @ l["wqkv"], 3, -1)]
            K[:, pos:pos + n], Vc[:, pos:pos + n] = k, v
            att = q @ K[:, :pos + n].transpose(0, 2, 1) / math.sqrt(dh) + mask
            att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
            h = h + (att @ Vc[:, :pos + n]).transpose(1, 0, 2).reshape(n, D) @ l["wo"]
            a = _ln(h, l["ln2_g"], l["ln2_b"])
            h = h + _gelu(a @ l["w1"] + l["b1"]) @ l["w2"] + l["b2"]
        return _ln(h[-1:], w["lnf_g"], w["lnf_b"]) @ w["wte"].T

    def generate(self, ids, n_new, temp=0.8, top_k=40, top_p=0.95, rep_pen=1.1, seed=None):
        rng = np.random.default_rng(seed)
        ids = list(ids)[-(self.T - 1):]
        cache = self.new_cache()
        logits, pos = self.forward(np.array(ids), cache, 0)[-1], len(ids)
        for _ in range(n_new):
            lg = logits.copy()
            for t in set(ids[-64:]):
                lg[t] = lg[t] / rep_pen if lg[t] > 0 else lg[t] * rep_pen
            if temp <= 0:
                nxt = int(lg.argmax())
            else:
                lg = lg / temp
                if top_k:
                    lg[lg < np.partition(lg, -top_k)[-top_k]] = -1e9
                p = np.exp(lg - lg.max()); p /= p.sum()
                o = np.argsort(-p); cs = np.cumsum(p[o]); p[o[cs - p[o] > top_p]] = 0; p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            ids.append(nxt)
            yield nxt
            if pos >= self.T:                      # context full: restart from the recent half
                keep = ids[-self.T // 2:]; cache = self.new_cache()
                logits, pos = self.forward(np.array(keep), cache, 0)[-1], len(keep)
            else:
                logits = self.forward(np.array([nxt]), cache, pos)[-1]; pos += 1


def load(d):
    from tokenizers import Tokenizer
    return Model(d), Tokenizer.from_file(os.path.join(d, TOK_NAME))


def complete(m, tok, prompt, n=100, temp=0.8, top_k=40, seed=None):
    ids = tok.encode(prompt).ids or [0]
    return tok.decode(ids + list(m.generate(ids, n, temp, top_k, seed=seed)))


def cmd_gen(a):
    d = a.dir
    if a.repo and not os.path.exists(os.path.join(d, W_NAME)):
        pull(a.repo, d)
    m, tok = load(d)
    ids = tok.encode(a.prompt).ids or [0]
    print(a.prompt, end="", flush=True)
    t0, buf, shown, n = time.time(), [], ids, 0
    for t in m.generate(ids, a.tokens, a.temp, a.top_k, seed=a.seed):
        buf.append(t); n += 1
        s = tok.decode(shown + buf)[len(tok.decode(shown)):]
        if s and not s.endswith("�"):
            print(s, end="", flush=True); shown = shown + buf; buf = []
    print(f"\n[{n / (time.time() - t0):.0f} tok/s]")


# ---------------------------------------------------------------- data
def clean_wikitext(s):
    s = s.replace("<unk>", "").replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?%)\]])", r"\1", s); s = re.sub(r"([(\[$]) ", r"\1", s)
    return re.sub(r"  +", " ", s.replace(" 's", "'s").replace(" n't", "n't"))


def load_text(a):
    if a.data:
        return "\n".join(open(f, encoding="utf8", errors="ignore").read() for f in a.data)
    import urllib.request
    print("no --data given: downloading WikiText-2 (demo corpus, ~12 MB)")
    return clean_wikitext("\n".join(urllib.request.urlopen(WIKI + f).read().decode() for f in ("train.txt", "valid.txt")))


def build_tokenizer(text, vocab, path):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(text.split("\n"), trainers.BpeTrainer(
        vocab_size=vocab, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(path)


# ---------------------------------------------------------------- training (jax)
def cmd_train(a):
    import jax, jax.numpy as jnp, optax
    from tokenizers import Tokenizer
    out = a.dir
    token = os.environ.get("HF_TOKEN")
    if a.repo and not a.fresh and not os.path.exists(os.path.join(out, W_NAME)):
        try:
            pull(a.repo, out, token); print("resuming from the Hub copy")
        except Exception as e:
            print(f"no existing model on the Hub ({type(e).__name__}); starting fresh")
    os.makedirs(out, exist_ok=True)
    text = load_text(a)
    resume = os.path.exists(os.path.join(out, W_NAME)) and not a.fresh
    tokp = os.path.join(out, TOK_NAME)
    if resume:
        flat, cfg = load_flat(out)
        tok = Tokenizer.from_file(tokp)
        print(f"resuming: {cfg['steps_done']} steps already done")
    else:
        build_tokenizer(text, a.vocab, tokp); tok = Tokenizer.from_file(tokp); cfg = None
    V = tok.get_vocab_size()
    if cfg is None:
        cfg = dict(V=V, D=a.D, L=a.L, H=a.H, T=a.T, steps_done=0, tokens_seen=0)
    D, L, H, T = cfg["D"], cfg["L"], cfg["H"], cfg["T"]
    ids = np.array(tok.encode(text).ids, dtype=np.int32)
    cut = int(len(ids) * 0.95); tr, va = ids[:cut], ids[cut:]
    assert len(tr) > T + 2, "not enough text: need more data or a smaller --T"

    def init(key):
        ks = iter(jax.random.split(key, 2 + 4 * L)); n = lambda s, sd: jax.random.normal(next(ks), s) * sd
        p = {"wte": n((V, D), .02), "wpe": n((T, D), .02), "lnf_g": jnp.ones(D), "lnf_b": jnp.zeros(D), "layers": []}
        for _ in range(L):
            p["layers"].append({"ln1_g": jnp.ones(D), "ln1_b": jnp.zeros(D), "wqkv": n((D, 3 * D), .02),
                                "wo": n((D, D), .02 / math.sqrt(2 * L)), "ln2_g": jnp.ones(D), "ln2_b": jnp.zeros(D),
                                "w1": n((D, 4 * D), .02), "b1": jnp.zeros(4 * D),
                                "w2": n((4 * D, D), .02 / math.sqrt(2 * L)), "b2": jnp.zeros(D)})
        return p

    def ln(x, g, b):
        m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) * jax.lax.rsqrt(v + 1e-5) * g + b

    def fwd(p, x):
        Bn, Tn = x.shape; dh = D // H
        h = p["wte"][x] + p["wpe"][:Tn]; mask = jnp.tril(jnp.ones((Tn, Tn), bool))
        sh = lambda t: t.reshape(Bn, Tn, H, dh).transpose(0, 2, 1, 3)
        for l in p["layers"]:
            q, k, v = [sh(t) for t in jnp.split(ln(h, l["ln1_g"], l["ln1_b"]) @ l["wqkv"], 3, -1)]
            att = jax.nn.softmax(jnp.where(mask, (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dh), -1e9), -1)
            h = h + (att @ v).transpose(0, 2, 1, 3).reshape(Bn, Tn, D) @ l["wo"]
            h = h + jax.nn.gelu(ln(h, l["ln2_g"], l["ln2_b"]) @ l["w1"] + l["b1"]) @ l["w2"] + l["b2"]
        return ln(h, p["lnf_g"], p["lnf_b"]) @ p["wte"].T

    loss_fn = lambda p, x, y: optax.softmax_cross_entropy_with_integer_labels(fwd(p, x), y).mean()

    p = init(jax.random.PRNGKey(cfg["steps_done"]))
    if resume:
        p["wte"], p["wpe"] = jnp.array(flat["wte"]), jnp.array(flat["wpe"])
        p["lnf_g"], p["lnf_b"] = jnp.array(flat["lnf_g"]), jnp.array(flat["lnf_b"])
        p["layers"] = [{k[len(f"l{i}_"):]: jnp.array(flat[k]) for k in flat if k.startswith(f"l{i}_")} for i in range(L)]
    n_par = sum(x.size for x in jax.tree_util.tree_leaves(p))
    print(f"model: {n_par/1e6:.2f}M params, {n_par*4/1e6:.1f} MB fp32 | data: {len(tr):,} train tokens | backend: {jax.devices()[0].platform}")

    S = a.steps
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, max(1, min(100, S // 10)), S, a.lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(
        sched, b1=.9, b2=.95, weight_decay=.1, mask=lambda t: jax.tree_util.tree_map(lambda x: x.ndim >= 2, t)))
    st = opt.init(p)

    @jax.jit
    def step(p, st, x, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, y)
        u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l

    ev = jax.jit(loss_fn)
    nv = max(1, min((len(va) - 1) // T, 32))
    vx = np.stack([va[i * T:(i + 1) * T] for i in range(nv)]); vy = np.stack([va[i * T + 1:(i + 1) * T + 1] for i in range(nv)])
    rng = np.random.default_rng(cfg["steps_done"] + 1); t0 = time.time(); acc = []

    def checkpoint():
        c = dict(cfg, steps_done=cfg["steps_done"] + s, tokens_seen=cfg["tokens_seen"] + s * a.B * T)
        save_model(p, c, out)

    for s in range(1, S + 1):
        ix = rng.integers(0, len(tr) - T - 1, a.B)
        x = np.stack([tr[i:i + T] for i in ix]); y = np.stack([tr[i + 1:i + T + 1] for i in ix])
        p, st, l = step(p, st, x, y); acc.append(float(l))
        if s % 25 == 0 or s == S:
            print(f"step {s}/{S}  loss {np.mean(acc):.3f}  {s*a.B*T/(time.time()-t0):,.0f} tok/s", flush=True); acc = []
        if s % a.eval_every == 0 or s == S:
            vl = float(ev(p, vx, vy)); print(f"  val loss {vl:.3f}  ppl {math.exp(vl):.1f}", flush=True); checkpoint()
    cfg.update(steps_done=cfg["steps_done"] + S, tokens_seen=cfg["tokens_seen"] + S * a.B * T, val_loss=round(vl, 4))
    save_model(p, cfg, out)
    shutil.copy(os.path.abspath(__file__), os.path.join(out, "slm.py"))
    write_card(out, a.repo, cfg)
    print(f"saved to {out}/  (total steps {cfg['steps_done']})")
    if a.repo and not a.no_push:
        push(a.repo, out, token, a.private, f"train: {cfg['steps_done']} steps, val loss {cfg['val_loss']}")


# ---------------------------------------------------------------- Hugging Face
def write_card(out, repo, cfg):
    card = f"""---
library_name: custom
tags: [small-language-model, from-scratch]
---
# {repo or 'my-slm'}

Small GPT-style model trained with `slm.py` ({cfg['steps_done']} steps, val loss {cfg.get('val_loss')}).
Config: `{json.dumps({k: cfg[k] for k in ('V', 'D', 'L', 'H', 'T')})}`

```
pip install numpy tokenizers safetensors huggingface_hub
python slm.py gen --repo {repo or 'USER/NAME'} "your prompt"
```
Continue training any time: `python slm.py train --repo {repo or 'USER/NAME'} --data more.txt --steps 1000`
"""
    open(os.path.join(out, "README.md"), "w").write(card)


def push(repo, folder, token=None, private=False, msg="update"):
    from huggingface_hub import HfApi
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo, exist_ok=True, private=private)
    api.upload_folder(folder_path=folder, repo_id=repo, commit_message=msg)
    print(f"pushed -> https://huggingface.co/{repo}")


SPACE_APP = '''import os, gradio as gr
from huggingface_hub import snapshot_download
import slm
d = snapshot_download(os.environ["MODEL_REPO"], token=os.environ.get("HF_TOKEN"))
m, tok = slm.load(d)
def go(prompt, n, temp):
    ids = tok.encode(prompt).ids or [0]
    return tok.decode(ids + list(m.generate(ids, int(n), temp)))
gr.Interface(go, [gr.Textbox(label="Prompt", lines=3), gr.Slider(10, 300, 100, label="New tokens"),
                  gr.Slider(0.1, 1.5, 0.8, label="Temperature")], gr.Textbox(label="Output", lines=12),
             title=os.environ["MODEL_REPO"]).launch()
'''


def cmd_space(a):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(a.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(a.space, "MODEL_REPO", a.repo)
    here = os.path.abspath(__file__)
    for name, data in [("app.py", SPACE_APP), ("requirements.txt", "numpy\ntokenizers\nsafetensors\nhuggingface_hub\n"),
                       ("README.md", f"---\ntitle: {a.space.split('/')[-1]}\nsdk: gradio\napp_file: app.py\n---\n")]:
        api.upload_file(path_or_fileobj=data.encode(), path_in_repo=name, repo_id=a.space, repo_type="space")
    api.upload_file(path_or_fileobj=here, path_in_repo="slm.py", repo_id=a.space, repo_type="space")
    print(f"space live soon at https://huggingface.co/spaces/{a.space}")
    print("if the model repo is private: add HF_TOKEN as a secret in the Space settings")


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--repo"); t.add_argument("--dir", default="model")
    t.add_argument("--data", nargs="*"); t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--fresh", action="store_true", help="ignore any existing model and start over")
    t.add_argument("--no-push", action="store_true"); t.add_argument("--private", action="store_true")
    for k, v in dict(D=192, L=5, H=3, T=256, B=32, vocab=4096).items(): t.add_argument(f"--{k}", type=int, default=v)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--eval-every", type=int, default=100)
    t.set_defaults(f=cmd_train)
    q = sp.add_parser("pull"); q.add_argument("--repo", required=True); q.add_argument("--dir", default="model")
    q.set_defaults(f=lambda a: pull(a.repo, a.dir))
    g = sp.add_parser("gen"); g.add_argument("prompt", nargs="?", default="The "); g.add_argument("--repo")
    g.add_argument("--dir", default="model"); g.add_argument("--tokens", type=int, default=100)
    g.add_argument("--temp", type=float, default=0.8); g.add_argument("--top_k", type=int, default=40)
    g.add_argument("--seed", type=int); g.set_defaults(f=cmd_gen)
    s = sp.add_parser("space"); s.add_argument("--repo", required=True); s.add_argument("--space", required=True)
    s.set_defaults(f=cmd_space)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
