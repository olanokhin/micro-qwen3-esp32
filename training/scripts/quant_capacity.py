"""Weight capacity: how many bits per parameter the model really needs.

    uv run training/scripts/quant_capacity.py --ckpt runs/shapeX-3x112-attn4x64
    uv run training/scripts/quant_capacity.py --ckpt runs/stal-adamw --ckpt runs/stal-muon --bits 6 4 3

All matrices (embeddings included) are quantized per row to b bits in four ways; norms
stay in float32. Each gets a val bpb. "Bits needed" = the smallest b at which
bpb is worse than the original by at most --tol.

  uniform     : symmetric uniform grid over the row's max|w| (naive, suffers from outliers);
  lloyd       : Lloyd-Max codebook for N(0,1), the row normalized by its std;
  rot-uniform : Hadamard rotation along the input axis (pad to 2^k), then uniform, then rotate back;
  rot-lloyd   : Hadamard rotation + Lloyd-Max (bitplane-ann method: after rotation the coordinates
                are nearly Gaussian, so the codebook does not depend on the data).
"""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from micro_qwen3.config import ModelConfig
from micro_qwen3.model import Qwen3ForCausalLM

METHODS = ["uniform", "lloyd", "rot-uniform", "rot-lloyd"]


def lloyd_max_gaussian(levels: int, iters: int = 300) -> np.ndarray:
    """MSE-optimal levels for N(0,1): alternate "thresholds = midpoints" and "levels = conditional means"."""
    pdf = lambda x: np.exp(-x * x / 2) / math.sqrt(2 * math.pi)  # noqa: E731
    cdf = lambda x: 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))  # noqa: E731
    c = np.linspace(-2.5, 2.5, levels)
    for _ in range(iters):
        t = np.concatenate([[-np.inf], (c[1:] + c[:-1]) / 2, [np.inf]])
        mass = cdf(t[1:]) - cdf(t[:-1])
        c = (pdf(t[:-1]) - pdf(t[1:])) / np.maximum(mass, 1e-12)  # E[X | t_i < X < t_{i+1}]
    return c


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def quantize_rows(w: np.ndarray, bits: int, method: str, codebooks: dict) -> np.ndarray:
    if method.endswith("uniform"):
        qmax = 2 ** (bits - 1) - 1
        scale = np.abs(w).max(axis=1, keepdims=True) / max(qmax, 1) + 1e-12
        return np.clip(np.round(w / scale), -qmax - (bits == 1), qmax) * scale
    cb = codebooks[bits]
    s = w.std(axis=1, keepdims=True) + 1e-12
    z = w / s
    idx = np.abs(z[..., None] - cb).argmin(axis=-1)
    return cb[idx] * s


def quantize_matrix(w: np.ndarray, bits: int, method: str, codebooks: dict) -> np.ndarray:
    if not method.startswith("rot"):
        return quantize_rows(w, bits, method, codebooks)
    d = w.shape[1]
    n = 1 << (d - 1).bit_length()
    H = hadamard(n)
    wr = np.pad(w, ((0, 0), (0, n - d))) @ H        # rotation along the input axis
    return (quantize_rows(wr, bits, method, codebooks) @ H.T)[:, :d]


def bpb(model, windows: np.ndarray, bytes_per_token: float, batch: int = 32) -> float:
    total, count = 0.0, 0
    for i in range(0, len(windows), batch):
        w = windows[i : i + batch]
        logits = model(mx.array(w[:, :-1])).astype(mx.float32)
        total += nn.losses.cross_entropy(logits, mx.array(w[:, 1:]), reduction="sum").item()
        count += w[:, 1:].size
    return total / count / math.log(2) / bytes_per_token


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, action="append", required=True)
    p.add_argument("--data", type=Path, default=None, help="default: from the run's status.json")
    p.add_argument("--bits", type=int, nargs="+", default=[8, 6, 5, 4, 3, 2])
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--windows", type=int, default=2000, help="val windows of seq_len (2000 ≈ 0.5M tokens)")
    p.add_argument("--tol", type=float, default=0.01, help="allowed bpb loss for \"bits needed\"")
    args = p.parse_args()

    codebooks = {b: lloyd_max_gaussian(2**b) for b in args.bits}
    for ck in args.ckpt:
        status = json.loads((ck / "status.json").read_text()) if (ck / "status.json").exists() else {}
        data = args.data or Path(status["data"])
        meta = json.loads((data / "meta.json").read_text())
        cfg = ModelConfig.load(ck / "config.json")
        model = Qwen3ForCausalLM(cfg)
        model.load_weights(str(ck / "model.safetensors"))
        base = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())}
        T = cfg.max_position_embeddings
        val = np.fromfile(data / "val.bin", dtype=np.uint16).astype(np.int32)
        n = min(args.windows, (len(val) - 1) // T)
        windows = np.stack([val[i * T : i * T + T + 1] for i in range(n)])

        ref = bpb(model, windows, meta["val_bytes_per_token"])
        n_params = sum(v.size for v in base.values())
        print(f"\n=== {ck.name}: {n_params:,} parameters, float32 bpb {ref:.4f} ({n} windows) ===")
        print(f"{'bits':>4} | " + " | ".join(f"{m:>15}" for m in args.methods))
        table = {m: {} for m in args.methods}
        for b in sorted(args.bits, reverse=True):
            cells = []
            for m in args.methods:
                q = {k: (quantize_matrix(v, b, m, codebooks) if v.ndim == 2 else v) for k, v in base.items()}
                model.update(tree_unflatten([(k, mx.array(v)) for k, v in q.items()]))
                r = bpb(model, windows, meta["val_bytes_per_token"])
                table[m][b] = r
                cells.append(f"{r:.4f} ({r - ref:+.3f})")
            print(f"{b:>4} | " + " | ".join(f"{c:>15}" for c in cells))
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in base.items()]))

        need = {m: min((b for b, r in table[m].items() if r - ref <= args.tol), default=None)
                for m in args.methods}
        print("bits needed per parameter (loss ≤ %.3f bpb): " % args.tol
              + ", ".join(f"{m} {need[m]}" for m in args.methods))
        out = {"ref_bpb": ref, "n_params": n_params, "table": table, "bits_needed": need, "tol": args.tol}
        (ck / "quant.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
