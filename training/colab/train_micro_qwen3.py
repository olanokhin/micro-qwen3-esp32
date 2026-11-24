# %% [markdown]
# # Train micro-qwen3 from scratch
#
# This notebook trains the model that runs on the ESP32-S3 board and in the
# [TensorLupa](https://tensorlupa.com) studio: 1,056,656 parameters, the Qwen3 architecture,
# a vocabulary of 4096 tokens. It follows the same recipe as the published weights, in PyTorch:
#
# 1. **Data.** TinyStories V2, a byte-level BPE tokenizer trained on it.
# 2. **Base model.** From scratch on 500 MB of stories.
# 3. **Instruction tuning.** 900K "write a story with these words" dialogues, with
#    quantization-aware training, so the model learns with the 4-bit weights it will have on the chip.
# 4. **Export.** One `.bin` file in the W4R format, the one the C engine and the board read.
# 5. **Run.** The C engine is built here and writes a story from the file you just trained.
#
# **How to run:** Runtime > Change runtime type > T4 GPU, then Runtime > Run all.
#
# Code and details: [micro-qwen3-esp32](https://github.com/olanokhin/micro-qwen3-esp32).

# %% [markdown]
# ## Settings
#
# `QUICK = True` runs the whole chain on a small slice of the data in a few minutes.
# The stories will be poor, but every step runs, which is a good first check.

# %%
import os

QUICK = os.environ.get("MQ_QUICK") == "1"

BASE_DOCS = 30_000 if QUICK else 800_000      # TinyStories V2 stories
BASE_BYTES = 1e7 if QUICK else 5e8            # how much text the base model sees
SFT_DOCS = 20_000 if QUICK else 900_000       # TinyStoriesInstruct dialogues
SFT_BYTES = 5e6 if QUICK else 6e8             # how much text the instruction tuning sees
VAL_DOCS = 500 if QUICK else 2_000            # held out, never trained on
SEQ_LEN, BATCH = 256, 64
SEED = 0

# %%
import codecs, json, math, re, struct, subprocess, time, urllib.request
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
AMP = DEVICE == "cuda"     # half-precision forward and backward on the GPU; the weights stay float32
WORK = Path(os.environ.get("MQ_WORK", "mq"))
WORK.mkdir(parents=True, exist_ok=True)
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
if DEVICE != "cuda":
    print("No GPU: the full run will be very slow. Set the runtime type to T4 GPU.")

# %% [markdown]
# ## 1. Data and tokenizer
#
# The real Qwen3 vocabulary has 151,936 tokens. One embedding matrix of that size would be many
# times larger than this whole model, so we train our own tokenizer of the same kind
# (byte-level BPE with the Qwen pre-tokenization rule) with 4096 tokens.
#
# The corpus is tokenized once into a flat array of token ids. Training then reads random
# windows of 256 tokens from it.

# %%
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

# The pre-tokenization rule of the real Qwen3 tokenizer: every digit is its own piece,
# a word goes together with the space before it, punctuation goes in groups.
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
              r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def stream_txt(repo, filename, sep="<|endoftext|>"):
    """Documents from a large .txt on Hugging Face. Only the part that is read gets downloaded."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    buf, dec = "", codecs.getincrementaldecoder("utf-8")()
    with urllib.request.urlopen(url) as r:
        while chunk := r.read(1 << 20):
            buf += dec.decode(chunk)
            *docs, buf = buf.split(sep)
            for d in docs:
                yield d.strip()
    if buf.strip():
        yield buf.strip()


def download(repo, filename, num_docs, convert=None):
    docs = stream_txt(repo, filename)
    if convert:
        docs = (c for c in map(convert, docs) if c)
    texts = list(tqdm(islice(docs, num_docs), total=num_docs, desc="download"))
    return [t for t in texts if t and t.strip()]


def train_tokenizer(texts, vocab_size=4096):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2)
    tok.train_from_iterator(texts, trainer, length=len(texts))
    return tok


def encode_stories(tok, texts, eot, batch=2000):
    chunks = []
    for i in tqdm(range(0, len(texts), batch), desc="tokenize"):
        for enc in tok.encode_batch(texts[i: i + batch]):
            chunks.append(np.asarray(enc.ids + [eot], dtype=np.uint16))
    return np.concatenate(chunks)


def prepare_stories(out, num_docs):
    """TinyStories V2 -> tokenizer.json, train.bin, val.bin, meta.json."""
    out = Path(out)
    if (out / "meta.json").exists():
        return json.loads((out / "meta.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    texts = download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt", num_docs)
    val, train = texts[:VAL_DOCS], texts[VAL_DOCS:]
    tok = train_tokenizer(train[:50_000])
    tok.save(str(out / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    meta = {"vocab_size": tok.get_vocab_size()}
    for name, part in (("train", train), ("val", val)):
        ids = encode_stories(tok, part, eot)
        ids.tofile(out / f"{name}.bin")
        # Bytes of text per token: turns the loss into bits per byte, a number
        # that does not depend on the tokenizer.
        nbytes = sum(len(t.encode("utf-8")) + 1 for t in part)
        meta[f"{name}_tokens"], meta[f"{name}_bytes_per_token"] = len(ids), nbytes / len(ids)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


BASE_DATA = WORK / "data-stories"
base_meta = prepare_stories(BASE_DATA, BASE_DOCS)
print(json.dumps(base_meta, indent=2))

# %% [markdown]
# ## 2. The model
#
# The Qwen3 architecture, layer by layer:
#
# ```
# tokens -> embed_tokens -> [DecoderLayer] x 3 -> norm -> logits (the same matrix as embed_tokens)
#
# DecoderLayer:  h = x + Attention(RMSNorm(x));   y = h + MLP(RMSNorm(h))
# Attention:     4 query heads share 2 key/value heads (GQA), RMSNorm on q and k (QK-norm), RoPE
# MLP:           W_down( silu(W_gate x) * (W_up x) )
# ```
#
# Parameter names match Hugging Face, so the saved weights are interchangeable with the published ones.

# %%
@dataclass
class ModelConfig:
    vocab_size: int = 4096
    hidden_size: int = 112
    intermediate_size: int = 336
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 256
    tie_word_embeddings: bool = True
    model_type: str = "qwen3"


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight. No mean subtraction and no bias."""

    def __init__(self, dims, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return y.to(x.dtype)


class Matrix(nn.Module):
    """A weight matrix without bias. Every matrix of the model is one of these.

    `Matrix.quant` is a switch for the whole model: when it is set, the forward pass sees
    the weights through it (rounded to 4 bits), while the stored weights stay float32.
    """

    quant = None

    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * 0.02)

    def w(self):
        return self.weight if Matrix.quant is None else Matrix.quant(self.weight)

    def forward(self, x):
        return F.linear(x, self.w().to(x.dtype))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, hd = cfg.hidden_size, cfg.head_dim
        self.n_heads, self.n_kv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, hd
        self.q_proj = Matrix(d, self.n_heads * hd)
        self.k_proj = Matrix(d, self.n_kv * hd)
        self.v_proj = Matrix(d, self.n_kv * hd)
        self.o_proj = Matrix(self.n_heads * hd, d)
        self.q_norm = RMSNorm(hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, cfg.rms_norm_eps)
        # RoPE: coordinate pairs (i, i + hd/2) are rotated by the angle position * theta^(-2i/hd)
        freqs = cfg.rope_theta ** (-torch.arange(hd // 2) / (hd // 2))
        ang = torch.arange(cfg.max_position_embeddings)[:, None] * freqs[None]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def rope(self, x):
        L, half = x.shape[-2], self.hd // 2
        cos, sin = self.cos[:L].to(x.dtype), self.sin[:L].to(x.dtype)
        a, b = x[..., :half], x[..., half:]
        return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, L, self.n_heads, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, L, self.n_kv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        rep = self.n_heads // self.n_kv            # GQA: each key/value head serves `rep` query heads
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = Matrix(cfg.hidden_size, cfg.intermediate_size)
        self.up_proj = Matrix(cfg.hidden_size, cfg.intermediate_size)
        self.down_proj = Matrix(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = Matrix(cfg.hidden_size, cfg.vocab_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)

    def forward(self, ids):
        m = self.model
        E = m.embed_tokens.w()                     # one matrix: token embeddings and the output layer
        h = F.embedding(ids, E)
        if AMP:
            h = h.half()
        for layer in m.layers:
            h = layer(h)
        return F.linear(m.norm(h), E.to(h.dtype))


cfg = ModelConfig()
torch.manual_seed(SEED)
model = Qwen3ForCausalLM(cfg).to(DEVICE)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 3. Training
#
# One function runs both stages. The recipe:
#
# - AdamW, betas 0.9 and 0.95, weight decay 0.1, gradient clipping at 1.0
# - learning rate: linear warmup, then cosine decay to 10% of the peak
# - batches of 64 windows of 256 tokens
# - every 500 steps the loss is measured on the same held-out windows, and the best weights are kept
#
# The budget is set in bytes of text, not in steps, so runs with different tokenizers stay comparable.
#
# **Quantization-aware training (QAT).** With `qat=True` the forward pass rounds every matrix to
# 4-bit integers with one scale per row, exactly as the export does later. For each row, 25 clipping
# levels between 40% and 100% of the largest weight are tried, and the one with the smallest error wins.
# The backward pass ignores the rounding (straight-through), so the float32 weights keep learning.

# %%
CLIP_GRID = torch.linspace(0.4, 1.0, 25)


def int4_scales(w):
    """One scale per row: of 25 fractions of max|row|, the one with the least squared error."""
    amax = w.abs().amax(1, keepdim=True) + 1e-12
    s_all = amax[None] * CLIP_GRID.to(w.device)[:, None, None] / 7.0          # [25, out, 1]
    err = ((w[None] / s_all).round().clamp(-7, 7) * s_all - w[None]).pow(2).sum(2)
    return s_all[err.argmin(0), torch.arange(w.shape[0], device=w.device)]    # [out, 1]


def fake_int4(w):
    """Forward: w rounded to 15 levels per row. Backward: the gradient as if there were no rounding."""
    with torch.no_grad():
        s = int4_scales(w)
        wq = (w / s).round().clamp(-7, 7) * s
    return w + (wq - w).detach()


class TokenStream:
    """Random windows of seq_len + 1 tokens from a flat token array, kept on the GPU."""

    def __init__(self, path, seq_len, seed):
        path = Path(path)
        self.data = torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int16)).to(DEVICE)
        mask = path.with_suffix(".mask.bin")     # dialogues: 1 = an answer token, the only ones we train on
        self.mask = torch.from_numpy(np.fromfile(mask, dtype=np.uint8)).to(DEVICE) if mask.exists() else None
        self.seq_len = seq_len
        self.gen = torch.Generator().manual_seed(seed)
        self.ar = torch.arange(seq_len + 1, device=DEVICE)

    def pack(self, starts):
        idx = starts.to(DEVICE)[:, None] + self.ar[None]
        chunk = self.data[idx].long()
        w = self.mask[idx[:, 1:]].float() if self.mask is not None else torch.ones_like(chunk[:, 1:], dtype=torch.float32)
        return chunk[:, :-1], chunk[:, 1:], w

    def batch(self, n):
        return self.pack(torch.randint(0, len(self.data) - self.seq_len - 1, (n,), generator=self.gen))

    def fixed_batches(self, n_batches, n):
        """The same windows at every evaluation, spread evenly over the array."""
        starts = torch.from_numpy(np.linspace(0, len(self.data) - self.seq_len - 1, n_batches * n).astype(np.int64))
        return [self.pack(starts[i: i + n]) for i in range(0, len(starts), n)]


@torch.no_grad()
def evaluate(model, batches, quant=None):
    """Mean loss per token on the held-out windows, in float32. quant=fake_int4: with 4-bit weights."""
    global AMP
    amp, AMP, Matrix.quant = AMP, False, quant
    tot = cnt = 0.0
    for x, y, w in batches:
        ce = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none")
        tot += (ce * w.flatten()).sum().item()
        cnt += w.sum().item()
    AMP, Matrix.quant = amp, None
    return tot / max(cnt, 1.0)


def train(model, data, out, train_bytes, lr, warmup, qat=False, eval_every=500, eval_batches=40,
          min_lr_ratio=0.1, weight_decay=0.1, grad_clip=1.0):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "meta.json").read_text())
    bpt = meta.get("val_answer_bytes_per_token") or meta["val_bytes_per_token"]
    steps = math.ceil(train_bytes / (meta["train_bytes_per_token"] * BATCH * SEQ_LEN))
    stream = TokenStream(data / "train.bin", SEQ_LEN, SEED)
    val = TokenStream(data / "val.bin", SEQ_LEN, SEED + 1).fixed_batches(eval_batches, BATCH)
    print(f"{train_bytes / 1e6:.0f} MB of text -> {steps} steps, "
          f"{steps * BATCH * SEQ_LEN / len(stream.data):.2f} passes over {len(stream.data):,} tokens")

    def lr_at(it):
        if it < warmup:
            return 1e-7 + (lr - 1e-7) * it / warmup
        end = lr * min_lr_ratio
        return end + 0.5 * (lr - end) * (1 + math.cos(math.pi * min(it - warmup, steps - warmup) / (steps - warmup)))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=AMP)
    history, best, run_loss, t0 = [], math.inf, None, time.perf_counter()
    model.train()
    for it in tqdm(range(1, steps + 1), desc=out.name, mininterval=10):
        for g in opt.param_groups:
            g["lr"] = lr_at(it - 1)
        x, y, w = stream.batch(BATCH)
        Matrix.quant = fake_int4 if qat else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            logits = model(x)
        Matrix.quant = None
        ce = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")   # softmax in float32
        loss = (ce * w.flatten()).sum() / w.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()

        if it % eval_every == 0 or it == steps:
            vl = evaluate(model, val)
            rec = {"step": it, "train_loss": run_loss, "val_loss": vl, "val_bpb": vl / math.log(2) / bpt,
                   "tok_s": it * BATCH * SEQ_LEN / (time.perf_counter() - t0)}
            history.append(rec)
            tqdm.write(f"step {it:6d} | train loss {run_loss:.3f} | val loss {vl:.3f} | "
                       f"val bpb {rec['val_bpb']:.3f} | {rec['tok_s'] / 1e3:.0f}k tok/s")
            if vl < best:
                best = vl
                save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    model.load_state_dict(load_file(str(out / "model.safetensors"), device=DEVICE))   # back to the best weights
    model.eval()
    (out / "config.json").write_text(json.dumps(asdict(model.cfg), indent=2))
    (out / "tokenizer.json").write_text((data / "tokenizer.json").read_text())
    (out / "log.json").write_text(json.dumps(history, indent=2))
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min, best val bpb {best / math.log(2) / bpt:.3f}")
    return history


@torch.no_grad()
def sample(model, tok, prompt, max_tokens=150, temperature=0.6, top_k=40, seed=1):
    """A simple sampler for a look at the model. The board's own decoding loop runs in section 6."""
    ids = tok.encode(prompt).ids
    n0 = len(ids)
    stop = {tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|im_end|>")}
    gen = torch.Generator().manual_seed(seed)
    for _ in range(max_tokens):
        x = torch.tensor([ids[-model.cfg.max_position_embeddings:]], device=DEVICE)
        v, i = model(x)[0, -1].float().cpu().topk(top_k)
        nxt = i[torch.multinomial(F.softmax(v / temperature, -1), 1, generator=gen)].item()
        if nxt in stop:
            break
        ids.append(nxt)
    return tok.decode(ids[n0:])

# %% [markdown]
# ## 4. Base model
#
# From scratch on TinyStories V2: 500 MB of text, peak learning rate 3e-3.
# The reference run reached **0.669 bits per byte** on held-out stories.

# %%
BASE_RUN = WORK / "run-base"
base_log = train(model, BASE_DATA, BASE_RUN, train_bytes=BASE_BYTES, lr=3e-3, warmup=200)

# %%
tok = Tokenizer.from_file(str(BASE_RUN / "tokenizer.json"))
prompt = "Once upon a time, there was a little dog named"
print(prompt + sample(model, tok, prompt))

# %% [markdown]
# ## 5. Instruction tuning with 4-bit weights
#
# The base model continues a text. To make it write a story to order, we go on training it on
# [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct): each record
# becomes a dialogue in the Qwen chat format,
#
# ```
# <|im_start|>user
# Write a short story. Use the words: dragon, cookie, school.<|im_end|>
# <|im_start|>assistant
# Once upon a time ...<|im_end|>
# ```
#
# and the loss counts only the answer. The same tokenizer is used. QAT is on from the first step.

# %%
FIELD_RE = re.compile(r"^(Features|Words|Summary|Random sentence|Story):[ \t]*", re.M)
FEATURES = {"Dialogue": "dialogue", "BadEnding": "a bad ending", "MoralValue": "a moral",
            "Twist": "a plot twist", "Foreshadowing": "foreshadowing", "Conflict": "a conflict"}
ASSISTANT = "<|im_start|>assistant\n"


def to_chat(doc):
    """A TinyStoriesInstruct record -> a dialogue. Two tasks, as in the dataset: a summary that comes
    after the story means "summarize"; otherwise "write a story" with the listed conditions."""
    parts = FIELD_RE.split(doc)
    head, fields, order = parts[0].strip(), {}, []
    for name, val in zip(parts[1::2], parts[2::2]):
        fields[name], order = val.strip(), order + [name]
    story = fields.get("Story") or head
    if not story:
        return None
    story_pos = order.index("Story") if "Story" in order else -1
    if "Summary" in fields and order.index("Summary") > story_pos:
        user, answer = f"Summarize this story in one sentence.\n\n{story}", fields["Summary"]
    else:
        req = []
        if "Words" in fields:
            req.append(f"Use the words: {fields['Words']}.")
        if "Features" in fields:
            feats = [FEATURES.get(f.strip(), f.strip().lower()) for f in fields["Features"].split(",") if f.strip()]
            req.append("The story should have " + ", ".join(feats) + ".")
        if "Summary" in fields:
            req.append(f"Plot: {fields['Summary']}")
        if "Random sentence" in fields:
            req.append(f"Include the sentence: \"{fields['Random sentence']}\"")
        user, answer = " ".join(["Write a short story."] + req), story
    if not answer:
        return None
    return f"<|im_start|>user\n{user}<|im_end|>\n{ASSISTANT}{answer}<|im_end|>"


def encode_chat(tok, texts, eot, batch=2000):
    """The question and the answer are encoded separately, the same way as on the board.
    Mask: 1 for answer tokens (with <|im_end|>), 0 for the question and <|endoftext|>."""
    ids, masks, nbytes = [], [], 0
    for i in tqdm(range(0, len(texts), batch), desc="tokenize"):
        part = texts[i: i + batch]
        cut = [t.index(ASSISTANT) + len(ASSISTANT) for t in part]
        qs = tok.encode_batch([t[:c] for t, c in zip(part, cut)])
        ans = tok.encode_batch([t[c:] for t, c in zip(part, cut)])
        for t, c, q, a in zip(part, cut, qs, ans):
            ids.append(np.asarray(q.ids + a.ids + [eot], dtype=np.uint16))
            masks.append(np.asarray([0] * len(q.ids) + [1] * len(a.ids) + [0], dtype=np.uint8))
            nbytes += len(t[c:].replace("<|im_end|>", "").encode("utf-8")) + 1
    return np.concatenate(ids), np.concatenate(masks), nbytes


def prepare_dialogues(out, num_docs, tokenizer_json):
    out = Path(out)
    if (out / "meta.json").exists():
        return json.loads((out / "meta.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    texts = download("roneneldan/TinyStoriesInstruct", "TinyStories-Instruct-train.txt", num_docs, to_chat)
    tok = Tokenizer.from_file(str(tokenizer_json))
    tok.save(str(out / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    meta = {"vocab_size": tok.get_vocab_size()}
    for name, part in (("train", texts[VAL_DOCS:]), ("val", texts[:VAL_DOCS])):
        ids, mask, answer_bytes = encode_chat(tok, part, eot)
        ids.tofile(out / f"{name}.bin")
        mask.tofile(out / f"{name}.mask.bin")
        nbytes = sum(len(t.encode("utf-8")) + 1 for t in part)
        meta[f"{name}_tokens"], meta[f"{name}_bytes_per_token"] = len(ids), nbytes / len(ids)
        meta[f"{name}_answer_bytes_per_token"] = answer_bytes / int(mask.sum())
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


SFT_DATA = WORK / "data-dialogues"
sft_meta = prepare_dialogues(SFT_DATA, SFT_DOCS, BASE_RUN / "tokenizer.json")
print(json.dumps(sft_meta, indent=2))

# %% [markdown]
# 600 MB of text, peak learning rate 1e-3. The reference run reached **0.706 bits per byte** of the
# answer with the float32 weights.

# %%
SFT_RUN = WORK / "run-instruct"
sft_log = train(model, SFT_DATA, SFT_RUN, train_bytes=SFT_BYTES, lr=1e-3, warmup=100, qat=True, eval_every=1000)

# %% [markdown]
# What the 4 bits cost: the same held-out dialogues with float32 weights and with the weights
# rounded the way the export will round them.

# %%
sft_val = TokenStream(SFT_DATA / "val.bin", SEQ_LEN, SEED + 1).fixed_batches(40, BATCH)
to_bpb = 1 / math.log(2) / sft_meta["val_answer_bytes_per_token"]
bpb_fp, bpb_q4 = evaluate(model, sft_val) * to_bpb, evaluate(model, sft_val, fake_int4) * to_bpb
print(f"float32 weights: {bpb_fp:.3f} bpb | 4-bit weights: {bpb_q4:.3f} bpb | cost {bpb_q4 - bpb_fp:+.3f}")

# %%
order = "<|im_start|>user\nWrite a short story. Use the words: dragon, cookie, school.<|im_end|>\n" + ASSISTANT
Matrix.quant = fake_int4          # sample with the 4-bit weights
print(sample(model, tok, order))
Matrix.quant = None

# %% [markdown]
# ## 6. Export to W4R and run in the C engine
#
# One file holds everything the board needs: the header with the model shape, every matrix as
# 4-bit integers (two per byte) with a float32 scale per row, the norms, and the tokenizer.

# %%
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def q4_rows(w):
    """int4 with one scale per row: q in [-7, 7], stored as u = q + 8, 32 weights in 16 bytes
    (low nibbles = weights 0..15 of the group, high = 16..31), the row padded to a multiple of 32."""
    out, d = w.shape
    G = (d + 31) // 32
    amax = np.abs(w).max(axis=1) + 1e-12
    best_s = best_e = None
    for a in np.linspace(0.4, 1.0, 25):            # the same search as in training
        sc = amax * a / 7.0
        e = ((np.clip(np.round(w / sc[:, None]), -7, 7) * sc[:, None] - w) ** 2).sum(1)
        if best_s is None:
            best_s, best_e = sc.copy(), e
        else:
            m = e < best_e
            best_s[m], best_e[m] = sc[m], e[m]
    s = best_s.astype(np.float32)
    u = np.full((out, G * 32), 8, dtype=np.uint8)
    u[:, :d] = (np.clip(np.round(w / s[:, None]), -7, 7) + 8).astype(np.uint8)
    ug = u.reshape(out, G, 32)
    packed = (ug[..., :16] | (ug[..., 16:] << 4)).astype(np.uint8)
    sb = s.tobytes()
    return packed.tobytes() + sb + b"\0" * ((-len(sb)) % 16)


def export_w4r(run, out_path):
    run = Path(run)
    known = ModelConfig.__dataclass_fields__
    cfg = ModelConfig(**{k: v for k, v in json.loads((run / "config.json").read_text()).items() if k in known})
    sd = {k: v.float().numpy() for k, v in load_file(str(run / "model.safetensors")).items()}
    out = bytearray(b"MQW3")
    out += struct.pack("<9I2f", 3, cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers,
                       cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                       cfg.max_position_embeddings, cfg.rope_theta, cfg.rms_norm_eps)
    out += q4_rows(sd["model.embed_tokens.weight"])
    out += sd["model.norm.weight"].tobytes()
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            out += sd[p + n + ".weight"].tobytes()
        for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            out += q4_rows(sd[p + n + ".weight"])
    n_weights = len(out)
    # tokenizer: the raw bytes of every token, then the BPE merges in rank order
    tj = json.loads((run / "tokenizer.json").read_text())
    vocab = tj["model"]["vocab"]
    inv = {c: b for b, c in bytes_to_unicode().items()}
    special = {t["content"] for t in tj["added_tokens"]}
    id2raw = [b""] * len(vocab)
    for s, i in vocab.items():
        id2raw[i] = b"" if s in special else bytes(inv[c] for c in s)
    out += struct.pack("<I", len(id2raw))
    for r in id2raw:
        out += struct.pack("<B", len(r)) + r
    merges = [m.split(" ") if isinstance(m, str) else m for m in tj["model"]["merges"]]
    out += struct.pack("<I", len(merges))
    for a, b in merges:
        out += struct.pack("<3H", vocab[a], vocab[b], vocab[a + b])
    Path(out_path).write_bytes(bytes(out))
    print(f"{out_path}: {len(out) / 1e6:.2f} MB (weights {n_weights / 1e6:.2f} MB)")


BIN = WORK / "micro-qwen3-w4r.bin"
export_w4r(SFT_RUN, BIN)

# %% [markdown]
# The same C code that runs on the ESP32-S3, built for this machine. `order` writes a story with
# the board's decoding loop: the bonus for requested words, the spelling automaton, the repetition penalty.
# Arguments: the model file, the words, the seed.

# %%
REPO = Path(os.environ.get("MQ_REPO", "micro-qwen3-esp32"))
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/olanokhin/micro-qwen3-esp32", str(REPO)], check=True)
subprocess.run(["make", "-C", str(REPO / "host"), "order"], check=True)
print(subprocess.run([str(REPO / "host" / "order"), str(BIN), "dragon, cookie, school", "33"],
                     capture_output=True, text=True, check=True).stdout)

# %% [markdown]
# ## Take the model with you
#
# `micro-qwen3-w4r.bin` is the file the firmware reads from its model partition.
# The repository README shows how to flash it to a board.

# %%
try:
    from google.colab import files
    files.download(str(BIN))
except ImportError:
    print("the model is at", BIN.resolve())
