"""Export a micro-qwen3 model to a single binary file for C inference (Mac and ESP32-S3).

    uv run training/scripts/export_esp.py --ckpt runs/ts-base --out models/ts-base-w8.bin

W8 format (little-endian):
  header: "MQW3", u32 version=1, u32 vocab, hidden, inter, layers, heads, kv_heads, head_dim,
          max_pos, f32 rope_theta, f32 eps
  embeddings: i8[vocab][hidden] + f32[vocab]   (scale per row; they are also the output layer)
  f32[hidden] final norm
  per layer: f32[hidden] input_norm, f32[hidden] post_norm, f32[head_dim] q_norm, f32[head_dim] k_norm,
             then q, k, v, o, gate, up, down: i8[out][in] + f32[out]
  tokenizer: u32 n; n times (u8 length, token bytes); u32 m; m times (u16 a, u16 b, u16 merged): BPE merges by rank
W4 format (--bits 4, version=2): every matrix in groups of 32 weights along the input (input zero-padded
  to a multiple of 32): u8[out][G][16], low nibbles = weights 0..15 of the group, high = 16..31, stored u = q + 8,
  q ∈ [-7, 7]; then f16[out][G] scales and zero padding to 16 bytes. Norms as in W8.
W4R format (--bits 4r, version=3): int4 with ONE scale per row (like W8): u8[out][G][16], the same
  nibble packing, input padded to a multiple of 32 (u=8, i.e. zero), then f32[out] scales
  and padding to 16 bytes. The device pipeline is the same as for int8, plus unpacking in registers.
Version 4: like W4R, but after the final norm comes a separate output layer [vocab][hidden] (a model with a rotation
  baked into the weights: the output layer is untied from the embeddings).
Next to it, <out>.ref.json is written: an MLX reference (simulating the same weight format, int8 activations) for checking the C code.
"""

import argparse
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from micro_qwen3.config import ModelConfig
from micro_qwen3.model import Qwen3ForCausalLM


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def q8_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.abs(w).max(axis=1) / 127.0 + 1e-12
    q = np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s.astype(np.float32)


def q4_groups(w: np.ndarray) -> tuple[bytes, bytes]:
    """int4 with a scale per group of 32 (symmetric, q ∈ [-7, 7]); packed for unpacking with a mask and a shift."""
    out, d = w.shape
    G = (d + 31) // 32
    wp = np.zeros((out, G * 32), dtype=np.float32)
    wp[:, :d] = w
    wg = wp.reshape(out, G, 32)
    s = (np.abs(wg).max(-1) / 7.0).astype(np.float16)              # fp16 scale: the one the device will see
    s32 = s.astype(np.float32)
    q = np.where(s32[..., None] > 0, np.round(wg / np.maximum(s32[..., None], 1e-30)), 0)
    u = (np.clip(q, -7, 7) + 8).astype(np.uint8)                    # u ∈ [1, 15]
    packed = (u[..., :16] | (u[..., 16:] << 4)).astype(np.uint8)    # [out][G][16]
    sb = s.tobytes()
    sb += b"\0" * ((-len(sb)) % 16)                                  # align the next matrix to 16
    return packed.tobytes(), sb


CLIP = False   # --clip: row scale with clipped tails (as in QAT --qat int4row-clip)


def row_scales(w: np.ndarray) -> np.ndarray:
    amax = np.abs(w).max(axis=1) + 1e-12
    if not CLIP:
        return (amax / 7.0).astype(np.float32)
    best_s, best_e = None, None
    for a in np.linspace(0.4, 1.0, 25):            # the same fraction search as in train.py (fake_int4row_clip)
        sc = amax * a / 7.0
        e = ((np.clip(np.round(w / sc[:, None]), -7, 7) * sc[:, None] - w) ** 2).sum(1)
        if best_s is None:
            best_s, best_e = sc.copy(), e
        else:
            m = e < best_e
            best_s[m], best_e[m] = sc[m], e[m]
    return best_s.astype(np.float32)


def q4_rows(w: np.ndarray) -> tuple[bytes, bytes]:
    """int4 with one scale per row (q ∈ [-7, 7], u = q + 8), packed as in q4_groups."""
    out, d = w.shape
    G = (d + 31) // 32
    s = row_scales(w)
    q = np.clip(np.round(w / s[:, None]), -7, 7)
    u = np.full((out, G * 32), 8, dtype=np.uint8)
    u[:, :d] = (q + 8).astype(np.uint8)
    ug = u.reshape(out, G, 32)
    packed = (ug[..., :16] | (ug[..., 16:] << 4)).astype(np.uint8)
    sb = s.tobytes()
    sb += b"\0" * ((-len(sb)) % 16)
    return packed.tobytes(), sb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt", default="Once upon a time, there was a little dog named")
    p.add_argument("--greedy-tokens", type=int, default=60)
    p.add_argument("--bits", choices=["8", "4", "4r"], default="8")
    p.add_argument("--clip", action="store_true", help="W4R: row scale with clipped tails (for models trained with QAT --qat int4row-clip)")
    args = p.parse_args()
    global CLIP
    CLIP = args.clip

    cfg = ModelConfig.load(args.ckpt / "config.json")
    assert cfg.norm_type == "rms", "export supports only the Qwen3 architecture"
    untied = not cfg.tie_word_embeddings
    assert not untied or args.bits == "4r", "a separate output layer is supported only in W4R"
    model = Qwen3ForCausalLM(cfg)
    model.load_weights(str(args.ckpt / "model.safetensors"))
    W = lambda a: np.array(a.astype(mx.float32))  # noqa: E731
    tok = Tokenizer.from_file(str(args.ckpt / "tokenizer.json"))

    out = bytearray()
    out += b"MQW3" + struct.pack("<9I2f", 4 if untied else {"8": 1, "4": 2, "4r": 3}[args.bits], cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size,
                                 cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads,
                                 cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, cfg.rms_norm_eps)

    def put_q8(w):
        if args.bits in ("4", "4r"):
            packed, sb = (q4_groups if args.bits == "4" else q4_rows)(w)
            out.extend(packed)
            out.extend(sb)
            return
        q, s = q8_rows(w)
        out.extend(q.tobytes())
        out.extend(s.tobytes())

    put_q8(W(model.model.embed_tokens.weight))
    out.extend(W(model.model.norm.weight).astype(np.float32).tobytes())
    if untied:   # version 4: W4R + a separate output layer right after the final norm
        put_q8(W(model.lm_head.weight))
    for L in model.model.layers:
        for n in (L.input_layernorm, L.post_attention_layernorm, L.self_attn.q_norm, L.self_attn.k_norm):
            out.extend(W(n.weight).astype(np.float32).tobytes())
        a, m = L.self_attn, L.mlp
        for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj):
            put_q8(W(lin.weight))
    weights_bytes = len(out)

    # tokenizer: raw bytes of each token + BPE merges by rank
    tj = json.loads((args.ckpt / "tokenizer.json").read_text())
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(bytes(out))
    print(f"{args.out}: {len(out) / 1e6:.2f} MB (weights {weights_bytes / 1e6:.2f} MB), "
          f"vocab {len(id2raw)}, merges {len(merges)}")

    # reference for checking: an MLX simulation of the same weight format as in the file (W8: int8 per row; W4R: int4 per row,
    # clipped with --clip; W4: int4 in groups of 32), int8 activations per token
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from quant_device import build
    wfmt = {"8": "int8-row", "4": "int4-g32", "4r": "int4c-row" if args.clip else "int4-row"}[args.bits]
    qm, head = build(args.ckpt, wfmt, 8, False)
    ids = tok.encode(args.prompt).ids
    logits = head(qm.model(mx.array([ids])))[0, -1]
    gen = list(ids)
    for _ in range(args.greedy_tokens):  # greedy generation by full recomputation (simple and safe)
        nxt = int(mx.argmax(head(qm.model(mx.array([gen])))[0, -1]).item())
        gen.append(nxt)
        if nxt == tok.token_to_id("<|endoftext|>"):
            break
    ref = {"prompt": args.prompt, "prompt_ids": ids, "last_logits": np.array(logits).tolist(),
           "greedy_ids": gen[len(ids):], "greedy_text": tok.decode(gen[len(ids):])}
    Path(str(args.out) + ".ref.json").write_text(json.dumps(ref, ensure_ascii=False))
    print("reference:", repr(args.prompt + ref["greedy_text"][:200]))


if __name__ == "__main__":
    main()
