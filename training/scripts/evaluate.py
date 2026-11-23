"""Detailed evaluation of a checkpoint after training.

    uv run training/scripts/evaluate.py --ckpt runs/sweep-v4096
    uv run training/scripts/evaluate.py --ckpt runs/sweep-v4096 --data data/fw-v4096

Computed on ALL of val (non-overlapping windows of seq_len):
  1. bits-per-byte: the main metric, comparable across tokenizers;
  2. loss by position in the context: does the model use the context;
  3. bits-per-byte by token type (words / digits / punctuation / spaces):
     where the model struggles and whether another way of splitting helps;
  4. repetition in generation: the share of repeated word 4-grams.
Result: <ckpt>/eval.json (shown by the dashboard).
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

from micro_qwen3.config import ModelConfig
from micro_qwen3.model import Qwen3ForCausalLM
from micro_qwen3.sampling import generate

CLASSES = ["words", "digits", "punctuation", "spaces", "special", "mixed"]


def bytes_to_unicode() -> dict[int, str]:
    """Byte-level BPE table (GPT-2): byte -> printable character in the tokenizer vocabulary."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_table(tok: Tokenizer) -> tuple[np.ndarray, np.ndarray]:
    """For every id: the exact number of text bytes and the token class."""
    inv = {c: b for b, c in bytes_to_unicode().items()}
    special = {t.content for t in tok.get_added_tokens_decoder().values()}
    V = tok.get_vocab_size()
    n_bytes = np.zeros(V, dtype=np.float64)
    cls = np.zeros(V, dtype=np.int64)
    for i in range(V):
        piece = tok.id_to_token(i)
        if piece in special:
            n_bytes[i], cls[i] = 1, CLASSES.index("special")  # <|endoftext|> = separator, 1 byte
            continue
        raw = bytes(inv[c] for c in piece)
        n_bytes[i] = len(raw)
        s = raw.decode("utf-8", errors="replace").strip()
        if not s:
            c = "spaces"
        elif all(ch.isdigit() for ch in s):
            c = "digits"
        elif all(ch.isalpha() for ch in s):
            c = "words"
        elif not any(ch.isalnum() for ch in s):
            c = "punctuation"
        else:
            c = "mixed"
        cls[i] = CLASSES.index(c)
    return n_bytes, cls


def repetition(text: str, n: int = 4) -> float:
    words = text.split()
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, default=None, help="default: from the run's status.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples", type=int, default=16, help="how many generations for the repetition metric")
    p.add_argument("--gen-tokens", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    status_path = args.ckpt / "status.json"
    data = args.data or (Path(json.loads(status_path.read_text())["data"]) if status_path.exists() else None)
    if data is None:
        raise SystemExit("pass --data: the run has no status.json")

    cfg = ModelConfig.load(args.ckpt / "config.json")
    tok = Tokenizer.from_file(str(args.ckpt / "tokenizer.json"))
    model = Qwen3ForCausalLM(cfg)
    model.load_weights(str(args.ckpt / "model.safetensors"))
    model.eval()
    n_bytes, cls = token_table(tok)

    T = cfg.max_position_embeddings
    val = np.fromfile(data / "val.bin", dtype=np.uint16).astype(np.int32)
    n_win = (len(val) - 1) // T
    windows = np.stack([val[i * T : i * T + T + 1] for i in range(n_win)])

    t0 = time.perf_counter()
    pos_loss = np.zeros(T)
    cls_bits = np.zeros(len(CLASSES))
    cls_bytes = np.zeros(len(CLASSES))
    cls_count = np.zeros(len(CLASSES))
    for i in range(0, n_win, args.batch_size):
        w = windows[i : i + args.batch_size]
        x, y = mx.array(w[:, :-1]), w[:, 1:]
        logits = model(x).astype(mx.float32)
        loss = np.array(nn.losses.cross_entropy(logits, mx.array(y), reduction="none"))  # (B, T) nats
        pos_loss += loss.sum(0)
        bits, c = loss.ravel() / math.log(2), cls[y.ravel()]
        np.add.at(cls_bits, c, bits)
        np.add.at(cls_bytes, c, n_bytes[y.ravel()])
        np.add.at(cls_count, c, 1)
    pos_loss /= n_win
    total_bits, total_bytes = cls_bits.sum(), cls_bytes.sum()
    bpb = total_bits / total_bytes
    print(f"val: {n_win:,} windows × {T} = {n_win * T:,} tokens, {total_bytes / 1e6:.1f} MB of text "
          f"({time.perf_counter() - t0:.1f} s)")
    print(f"bits-per-byte: {bpb:.4f}   loss: {pos_loss.mean():.4f} nats/token   "
          f"ppl: {math.exp(pos_loss.mean()):.2f}")

    by_class = {}
    print(f"\n{'token type':<12}{'token share':>14}{'byte share':>11}{'bpb':>8}{'bit share':>10}")
    for k, name in enumerate(CLASSES):
        if cls_count[k] == 0:
            continue
        by_class[name] = {
            "token_share": cls_count[k] / cls_count.sum(), "byte_share": cls_bytes[k] / total_bytes,
            "bpb": cls_bits[k] / cls_bytes[k], "bit_share": cls_bits[k] / total_bits,
        }
        r = by_class[name]
        print(f"{name:<12}{r['token_share']:>14.1%}{r['byte_share']:>11.1%}{r['bpb']:>8.3f}{r['bit_share']:>10.1%}")

    marks = [0, 1, 3, 7, 15, 31, 63, 127, T - 1]
    print("\nloss by position (nats/token): " + "  ".join(f"[{m + 1}] {pos_loss[m]:.2f}" for m in marks))

    # Repetition: continue the beginnings of val documents
    mx.random.seed(args.seed)
    eot = tok.token_to_id("<|endoftext|>")
    doc_starts = [0] + [int(j) + 1 for j in np.where(val == eot)[0][: args.samples * 2]]
    prompts = [val[s : s + 12].tolist() for s in doc_starts if eot not in val[s : s + 12]][: args.samples]
    reps, samples = [], []
    for pr in prompts:
        out = list(generate(model, pr, args.gen_tokens, 0.8, 40, {eot}))
        text = tok.decode(pr + out)
        reps.append(repetition(tok.decode(out)))
        samples.append(text)
    rep = float(np.mean(reps))
    print(f"\nrepetition in generation (share of repeated word 4-grams): {rep:.3f}")
    print(f"example: {samples[0][:300]!r}")

    result = {
        "data": str(data), "val_tokens": n_win * T, "val_bytes": total_bytes,
        "bpb": bpb, "loss": float(pos_loss.mean()), "ppl": math.exp(pos_loss.mean()),
        "bytes_per_token": total_bytes / (n_win * T),
        "by_class": by_class, "pos_loss": pos_loss.tolist(),
        "repetition_4gram": rep, "samples": samples[:4], "evaluated_at": time.time(),
    }
    (args.ckpt / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\nsaved: {args.ckpt / 'eval.json'}")


if __name__ == "__main__":
    main()
