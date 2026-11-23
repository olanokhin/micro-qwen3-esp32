"""Device formats (ESP32-S3): quantization of weights AND activations, with and without a Hadamard rotation.

    uv run training/scripts/quant_device.py --ckpt runs/x1000-adamw --ckpt runs/x1000-rot-b16

Every linear layer (and the output layer = embeddings) computes y = Q(x) · Q(W)^T, where
  Q(W): int8/int4 with a scale per row or per group of g weights, or Lloyd-Max int4;
  Q(x): int8 per token (symmetric over max|x|), which is how the S3 vector unit computes; or no quantization.
With rotation: x -> x·H, W -> W·H (H is Hadamard, input zero-padded to 2^k). The product is the same,
but outliers are "spread" across coordinates BEFORE quantization. On the device the residual stream rotation
is folded into the weights at export (QuaRot), so it is almost free.
"""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from micro_qwen3.config import ModelConfig
from micro_qwen3.model import Qwen3ForCausalLM

from quant_capacity import hadamard, lloyd_max_gaussian

WFORMATS = ["int8-row", "int4-row", "int4c-row", "int4-g64", "int4-g32", "lm4-row", "lm4-g32"]
_LM4 = lloyd_max_gaussian(16)


def quant_w(w: np.ndarray, fmt: str) -> np.ndarray:
    kind, gran = fmt.split("-")
    out, d = w.shape
    g = d if gran == "row" else int(gran[1:])
    pad = (-d) % g
    wg = np.pad(w, ((0, 0), (0, pad))).reshape(out, -1, g)
    if kind == "int4c":   # int4, clipped scale: search a fraction of max|w|, minimum squared error per row
        qmax = 7
        amax = np.abs(wg).max(-1, keepdims=True) + 1e-12
        best_q, best_e = None, None
        for a in np.linspace(0.4, 1.0, 25):
            s = amax * a / qmax
            q = np.clip(np.round(wg / s), -qmax, qmax) * s
            e = ((q - wg) ** 2).sum(-1, keepdims=True)
            if best_q is None:
                best_q, best_e = q, e
            else:
                m = e < best_e
                best_q = np.where(m, q, best_q); best_e = np.where(m, e, best_e)
        q = best_q
    elif kind.startswith("int"):
        qmax = 2 ** (int(kind[3:]) - 1) - 1
        s = np.abs(wg).max(-1, keepdims=True) / qmax + 1e-12
        q = np.clip(np.round(wg / s), -qmax, qmax) * s
    else:  # lm4: Lloyd-Max codebook for N(0,1), the group normalized by its std
        s = wg.std(-1, keepdims=True) + 1e-12
        q = _LM4[np.abs((wg / s)[..., None] - _LM4).argmin(-1)] * s
    return q.reshape(out, -1)[:, :d]


def quant_x(x: mx.array, bits: int | None) -> mx.array:
    if not bits:
        return x
    qmax = 2 ** (bits - 1) - 1
    s = mx.max(mx.abs(x), axis=-1, keepdims=True) / qmax + 1e-12
    return mx.clip(mx.round(x / s), -qmax, qmax) * s


class QLinear(nn.Module):
    """Replacement for nn.Linear / the output layer: quantized weights (ahead of time) and activations (on the fly)."""

    def __init__(self, w: np.ndarray, wfmt: str, abits: int | None, rotate: bool):
        super().__init__()
        d = w.shape[1]
        self.rotate, self.abits, self.d = rotate, abits, d
        if rotate:
            n = 1 << (d - 1).bit_length()
            self.H = mx.array(hadamard(n).astype(np.float32))
            w = np.pad(w, ((0, 0), (0, n - d))) @ hadamard(n)
            self.n = n
        self.wq = mx.array(quant_w(w, wfmt).astype(np.float32)) if wfmt else mx.array(w.astype(np.float32))
        self.freeze()

    def __call__(self, x):
        x = x.astype(mx.float32)
        if self.rotate:
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, self.n - self.d)]) @ self.H
        return quant_x(x, self.abits) @ self.wq.T


def build(ck: Path, wfmt: str | None, abits: int | None, rotate: bool):
    cfg = ModelConfig.load(ck / "config.json")
    m = Qwen3ForCausalLM(cfg)
    m.load_weights(str(ck / "model.safetensors"))
    for layer in m.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.children().items()):
                if isinstance(child, nn.Linear):
                    w = np.array(child.weight.astype(mx.float32))
                    setattr(parent, name, QLinear(w, wfmt, abits, rotate))
    E = np.array(m.model.embed_tokens.weight.astype(mx.float32))
    # output layer: the (tied) embeddings or a separate lm_head (untied in the rotated model)
    H = E if cfg.tie_word_embeddings else np.array(m.lm_head.weight.astype(mx.float32))
    head = QLinear(H, wfmt, abits, rotate)
    # input embeddings: just a quantized table (the device reads one row)
    if wfmt:
        m.model.embed_tokens.weight = mx.array(quant_w(E, wfmt).astype(np.float32))
    if cfg.head_mix:   # matrix before the output layer (rotated model with tied embeddings): not quantized
        Mx = m.head_mix
        inner = head
        head = lambda h: inner(h @ Mx)  # noqa: E731
    return m, head


def bpb(m, head, windows, bpt, batch=32):
    tot = cnt = 0
    for i in range(0, len(windows), batch):
        w = windows[i : i + batch]
        logits = head(m.model(mx.array(w[:, :-1])))
        tot += nn.losses.cross_entropy(logits, mx.array(w[:, 1:]), reduction="sum").item()
        cnt += w[:, 1:].size
    return tot / cnt / math.log(2) / bpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, action="append", required=True)
    p.add_argument("--data", type=Path, default=Path("data/fw-v4096-qwen"))
    p.add_argument("--windows", type=int, default=600)
    p.add_argument("--wformats", nargs="+", default=WFORMATS)
    args = p.parse_args()

    meta = json.loads((args.data / "meta.json").read_text())
    val = np.fromfile(args.data / "val.bin", dtype=np.uint16).astype(np.int32)
    T = 256
    windows = np.stack([val[i * T : i * T + T + 1] for i in range(args.windows)])
    bpt = meta["val_bytes_per_token"]

    for ck in args.ckpt:
        m, head = build(ck, None, None, False)
        ref = bpb(m, head, windows, bpt)
        print(f"\n=== {ck.name}: float32 bpb {ref:.4f} ===")
        print(f"{'weights':10} | {'act. float':>12} {'act. int8':>12} | {'Hadamard + act. float':>20} {'Hadamard + act. int8':>19}")
        res = {}
        for wf in args.wformats:
            cells = []
            for rot in (False, True):
                for ab in (None, 8):
                    m, head = build(ck, wf, ab, rot)
                    r = bpb(m, head, windows, bpt)
                    res[f"{wf}|{'rot' if rot else 'plain'}|a{ab or 'fp'}"] = r
                    cells.append(f"{r - ref:+.3f}")
            print(f"{wf:10} | {cells[0]:>12} {cells[1]:>12} | {cells[2]:>20} {cells[3]:>19}")
        (ck / "quant_device.json").write_text(json.dumps({"ref": ref, "results": res}, indent=1))


if __name__ == "__main__":
    main()
