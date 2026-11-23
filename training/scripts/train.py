"""Pretraining micro-Qwen3 on MLX.

    uv run training/scripts/train.py --recipe training/recipes/ts2-base.json --out runs/base
    uv run training/scripts/train.py --recipe training/recipes/ts2-base.json --out runs/exp --set num_hidden_layers=4
    uv run training/scripts/train.py --data data/fw-v4096-qwen --out runs/x --preset micro --train-bytes 5e8

A recipe (training/recipes/*.json) sets three levels of decisions: shape (model shape), interface
(data, tokenizer, context, RoPE) and training (training recipe). Command line arguments
and --set override the recipe. The final recipe is saved to <out>/recipe.json.

Logs train/val loss, perplexity and bits-per-byte to <out>/log.jsonl, the status to status.json.
"""

import argparse
import json
import math
import shutil
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import mlx.core as mx
import numpy as np
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_map_with_path
from tokenizers import Tokenizer

from micro_qwen3.config import PRESETS, ModelConfig
from micro_qwen3.data import TokenStream
from micro_qwen3.model import Qwen3ForCausalLM, init_ternary, init_weights
from micro_qwen3.optim import build_optimizer, norm_knobs, outlier_stats
from micro_qwen3.sampling import generate


def weighted_ce(logits, y, w):
    """Mean CE over tokens with weight w (the answer mask for dialogues, otherwise ones)."""
    ce = nn.losses.cross_entropy(logits, y, reduction="none")
    return mx.sum(ce * w) / mx.maximum(mx.sum(w), 1.0)


def evaluate(model, batches) -> float:
    """Mean loss on a fixed set of val windows (the same at every evaluation), weighted by tokens."""
    tot = cnt = 0.0
    for x, y, w in batches:
        ce = nn.losses.cross_entropy(model(x).astype(mx.float32), y, reduction="none")
        tot += mx.sum(ce * w).item()
        cnt += mx.sum(w).item()
    return tot / max(cnt, 1.0)


# Which recipe fields map to which command line arguments.
RECIPE_ARGS = {
    "interface": {"data": "data", "seq_len": "seq_len"},
    "training": {k: k for k in ("train_bytes", "batch_size", "dtype", "optimizer", "rot_block", "lr", "muon_lr",
                                "min_lr_ratio", "warmup",
                                "weight_decay", "grad_clip", "eval_every", "eval_batches", "seed")},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", type=Path, default=None,
                   help="JSON recipe with shape / interface / training sections (see training/recipes/ts2-base.json)")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preset", choices=PRESETS, default="micro")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--stop-at-step", type=int, default=None,
                   help="stop early WITHOUT compressing the lr schedule: to compare the start of training\n"
                        "with full runs at the same points")
    p.add_argument("--train-bytes", type=float, default=None,
                   help="budget in bytes of source text (e.g. 5e8); overrides --steps, "
                        "so that models with different tokenizers see the same amount of text")
    p.add_argument("--num-layers", type=int, default=None, help="override the preset's number of layers")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any ModelConfig field, e.g. --set hidden_size=48 --set head_dim=12")
    p.add_argument("--lr", type=float, default=3e-3, help="small models tolerate a large lr")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--sample-prompt", default="The")
    p.add_argument("--optimizer", choices=["adamw", "adamw-cautious", "adamw-rot", "muon"], default="adamw",
                   help="adamw-rot: AdamW in a rotated basis; muon: Muon for matrices + AdamW for the rest")
    p.add_argument("--muon-lr", type=float, default=0.02, help="peak lr for Muon (layer matrices)")
    p.add_argument("--rot-block", type=int, default=None,
                   help="adamw-rot: block Hadamard with block N (rotation dose); default: full rotation")
    p.add_argument("--init-from", type=Path, default=None,
                   help="start from this run's weights (fine-tuning) instead of random ones")
    p.add_argument("--freeze-norms", action="store_true",
                   help="do not train the RMSNorms before blocks and the final one (for a rotated model: they must stay at ones)")
    p.add_argument("--qat", choices=["none", "int4row", "int4row-clip", "int3row-clip", "int2row-clip"], default="none",
                   help="int4row: in the forward pass matrices are rounded to int4 with a scale per row (as in the W4R export), "
                        "the gradient is straight-through")
    p.add_argument("--ternary-init", choices=["twn", "thirds"], default="twn",
                   help="threshold for initializing ternary matrices (model with ternary=True): TWN 0.7·mean|w| or exactly a third of zeros")
    p.add_argument("--stage2", action="store_true",
                   help="deepgrove stage 2: ternary weights frozen, only row scales and norms are trained")
    p.add_argument("--teacher", type=Path, default=None,
                   help="distillation: a teacher run with the same vocabulary; its logits are computed on the fly and frozen")
    p.add_argument("--kd-alpha", type=float, default=0.5, help="share of plain CE in the loss; the rest is KL to the teacher")
    p.add_argument("--kd-temp", type=float, default=2.0, help="softening temperature of distributions in distillation")
    p.add_argument("--kd-topk", type=int, default=64,
                   help="KL over the teacher's top-k tokens + one \"other\" bucket (0: over the whole vocabulary)")
    p.add_argument("--save-every", type=int, default=0,
                   help="save a weight snapshot every N steps to <out>/ckpt/ (for measurements during training)")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16",
                   help="compute type for forward/backward; master weights and AdamW are always float32")
    p.add_argument("--seed", type=int, default=0)

    # The recipe sets defaults; explicit command line arguments override them.
    pre, _ = p.parse_known_args()
    recipe = json.loads(pre.recipe.read_text()) if pre.recipe else {}
    p.set_defaults(**{dest: recipe[sec][key] for sec, mapping in RECIPE_ARGS.items()
                      for key, dest in mapping.items() if key in recipe.get(sec, {})})
    args = p.parse_args()
    if args.data is None:
        p.error("need --data or a recipe with interface.data")

    mx.random.seed(args.seed)
    tok = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    meta = json.loads((args.data / "meta.json").read_text())
    # for dialogues (there is an answer mask) bpb is computed over answer bytes
    bytes_per_token = meta.get("val_answer_bytes_per_token") or meta.get("val_bytes_per_token")
    iface = recipe.get("interface", {})
    for key in ("vocab_size", "pretokenizer"):  # the recipe and the data must agree
        if key in iface and iface[key] != meta.get(key, "gpt2" if key == "pretokenizer" else None):
            p.error(f"recipe: {key}={iface[key]}, but {args.data}/meta.json has {meta.get(key)}")
    cfg = replace(PRESETS[args.preset], **recipe.get("shape", {}), vocab_size=tok.get_vocab_size(),
                  max_position_embeddings=args.seq_len)
    if "rope_theta" in iface:
        cfg = replace(cfg, rope_theta=float(iface["rope_theta"]))
    if args.num_layers:
        cfg = replace(cfg, num_hidden_layers=args.num_layers)
    for kv in args.set:
        key, value = kv.split("=", 1)
        field_type = type(getattr(cfg, key))  # AttributeError on a typo, which is good
        val = value.lower() in ("1", "true", "yes") if field_type is bool else field_type(value)  # bool("False") == True!
        cfg = replace(cfg, **{key: val})
    if args.train_bytes:
        train_bpt = meta["train_bytes_per_token"]
        args.steps = math.ceil(args.train_bytes / (train_bpt * args.batch_size * args.seq_len))
        print(f"budget {args.train_bytes / 1e6:.0f} MB of text at {train_bpt:.2f} bytes/token "
              f"-> {args.steps} steps")

    model = Qwen3ForCausalLM(cfg)
    init_weights(model)
    if args.init_from:
        src_cfg = json.loads((args.init_from / "config.json").read_text())
        # full-precision model -> ternary: the file has no scales and signs, init_ternary sets them
        tern_from_fp = cfg.ternary and not src_cfg.get("ternary", False)
        model.load_weights(str(args.init_from / "model.safetensors"), strict=not tern_from_fp)
        print(f"starting from weights {args.init_from}" + (" (full precision -> ternarization)" if tern_from_fp else ""))
    else:
        tern_from_fp = cfg.ternary
    if tern_from_fp:
        init_ternary(model, seed=args.seed, method=args.ternary_init)
        print(f"ternary weights: init {args.ternary_init}, Hadamard block {cfg.hadamard_block}")
    elif cfg.ternary:
        model.freeze(keys=["signs"])
    if args.stage2:
        model.freeze()
        model.unfreeze(keys=["scales", "weight"], strict=False)
        for _, m in model.named_modules():   # of "weight", unfreeze only the norms (1-D), not the matrices
            if hasattr(m, "weight") and m.weight.ndim == 2:
                m.freeze(keys=["weight"], recurse=False)
        print("stage 2: only scales and norms are trained")
    if args.freeze_norms:
        model.model.norm.freeze()
        for L in model.model.layers:
            L.input_layernorm.freeze()
            L.post_attention_layernorm.freeze()
        if cfg.head_mix:
            model.freeze(keys=["head_mix"], recurse=False)
        print("norms before blocks and the final norm are frozen" + (", head_mix too" if cfg.head_mix else ""))
    mx.eval(model.parameters())
    bd = cfg.param_breakdown()
    print(f"parameters: {model.num_params():,} "
          f"(embeddings {bd['embedding']:,}, layers {bd['layers_total']:,})")

    train = TokenStream(args.data / "train.bin", args.seq_len, seed=args.seed, weights=True)
    val = TokenStream(args.data / "val.bin", args.seq_len, seed=args.seed + 1, weights=True)
    if train.mask is not None:
        print(f"answer mask: training on {train.mask.mean():.0%} of tokens")
    val_batches = val.fixed_batches(args.eval_batches, args.batch_size)
    tokens_per_step = args.batch_size * args.seq_len
    print(f"train tokens: {len(train):,}; during training the model will see "
          f"{args.steps * tokens_per_step:,} ({args.steps * tokens_per_step / len(train):.2f} epochs)")

    schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, args.lr, args.warmup),
         optim.cosine_decay(args.lr, args.steps - args.warmup, args.lr * args.min_lr_ratio)],
        [args.warmup],
    )
    muon_schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, args.muon_lr, args.warmup),
         optim.cosine_decay(args.muon_lr, args.steps - args.warmup, args.muon_lr * args.min_lr_ratio)],
        [args.warmup],
    )
    optimizer = build_optimizer(args.optimizer, cfg.hidden_size, schedule, muon_schedule,
                                weight_decay=args.weight_decay, seed=args.seed, rot_block=args.rot_block)
    probe_x = val_batches[0][0][:16]  # a fixed batch for the "stalactite ruler"
    # Mixed precision: `model` holds the float32 master weights (AdamW updates them, we save them),
    # `compute_model` is their copy in compute_dtype that forward and backward go through.
    # With --dtype float32 the cast does nothing, and this is ordinary training.
    compute_dtype = getattr(mx, args.dtype)
    compute_model = Qwen3ForCausalLM(cfg)
    compute_model.update(model.parameters())   # frozen parameters (norms, head_mix) must match model

    def fake_int4row(p):
        # rounding as in the W4R export: q = clip(round(p / s), -7, 7), s = max|row| / 7;
        # p + stop_gradient(Q(p) - p): forward uses quantized weights, backward a gradient as if there were no rounding
        s = mx.max(mx.abs(p), axis=1, keepdims=True) / 7.0 + 1e-12
        return p + mx.stop_gradient(mx.clip(mx.round(p / s), -7, 7) * s - p)

    clip_grid = mx.array(np.linspace(0.4, 1.0, 25).astype(np.float32))

    def fake_int4row_clip(p):
        # scale with clipped tails: of 25 fractions of max|row| take the one with the least squared error
        # (the same search as in the --clip export); the scale choice has no gradient, rounding is straight-through
        amax = mx.max(mx.abs(p), axis=1, keepdims=True) + 1e-12
        s_all = mx.stop_gradient(amax[None] * clip_grid[:, None, None] / 7.0)          # [25, out, 1]
        q_all = mx.clip(mx.round(mx.stop_gradient(p)[None] / s_all), -7, 7) * s_all
        err = mx.sum(mx.square(q_all - mx.stop_gradient(p)[None]), axis=2)            # [25, out]
        best = mx.argmin(err, axis=0)                                                  # [out]
        s = mx.take_along_axis(s_all[:, :, 0], best[None], axis=0)[0][:, None]
        return p + mx.stop_gradient(mx.clip(mx.round(p / s), -7, 7) * s - p)

    clip_grid_lo = mx.array(np.linspace(0.15, 1.0, 35).astype(np.float32))

    def fake_midrise_clip(p, bits):
        # int2/int3 without zero: all 2^bits codes are levels (k + 0.5)·s, k ∈ [-2^(bits-1), 2^(bits-1) - 1]
        # (int2: ±0.5s, ±1.5s; int3: ±0.5s … ±3.5s). Step s: a search over fractions of max|w| per row, rounding is STE.
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        amax = mx.max(mx.abs(p), axis=1, keepdims=True) + 1e-12
        s_all = mx.stop_gradient(amax[None] * clip_grid_lo[:, None, None] / (hi + 0.5))
        pd = mx.stop_gradient(p)[None]
        q_all = (mx.clip(mx.floor(pd / s_all), lo, hi) + 0.5) * s_all
        best = mx.argmin(mx.sum(mx.square(q_all - pd), axis=2), axis=0)
        s = mx.take_along_axis(s_all[:, :, 0], best[None], axis=0)[0][:, None]
        return p + mx.stop_gradient((mx.clip(mx.floor(p / s), lo, hi) + 0.5) * s - p)

    def prep(path, p):
        # a ternary model quantizes its layer matrices itself; QAT int4 is only for full-precision matrices
        # (embeddings when ternary_embed=False)
        own = cfg.ternary and not (path.startswith("model.embed_tokens") and not cfg.ternary_embed)
        if args.qat == "int4row" and p.ndim == 2 and not own:
            p = fake_int4row(p)
        elif args.qat == "int4row-clip" and p.ndim == 2 and not own:
            p = fake_int4row_clip(p)
        elif args.qat in ("int3row-clip", "int2row-clip") and p.ndim == 2 and not own:
            # layers int3/int2, embeddings (the tied output layer) int4: it is the most sensitive
            bits = int(args.qat[3])
            p = fake_int4row_clip(p) if path.startswith("model.embed_tokens") else fake_midrise_clip(p, bits)
        return p.astype(compute_dtype)

    teacher = None
    if args.teacher:
        tcfg = ModelConfig.load(args.teacher / "config.json")
        assert tcfg.vocab_size == cfg.vocab_size, "the teacher and the student must share one vocabulary"
        teacher = Qwen3ForCausalLM(tcfg)
        teacher.load_weights(str(args.teacher / "model.safetensors"))
        teacher.update(tree_map(lambda a: a.astype(compute_dtype), teacher.parameters()))
        teacher.freeze()
        mx.eval(teacher.parameters())
        print(f"distillation: teacher {args.teacher} ({teacher.num_params():,} parameters), "
              f"alpha {args.kd_alpha}, T {args.kd_temp}, " + (f"top-{args.kd_topk} + tail" if args.kd_topk else "whole vocabulary"))

    def kd_loss(s_logits, t_logits, w):
        """KL(teacher || student) at temperature T, ×T² (gradient scale as for CE), only over tokens with weight w.
        top-k: the teacher distribution is the k most likely tokens + one bucket with all the remaining mass;
        the student's bucket gets its mass outside the same k tokens."""
        T = args.kd_temp
        lt = t_logits / T
        ls = nn.log_softmax(s_logits / T, axis=-1)
        if args.kd_topk:
            k = args.kd_topk
            idx = mx.stop_gradient(mx.argpartition(-lt, kth=k - 1, axis=-1)[..., :k])
            lpt = mx.take_along_axis(nn.log_softmax(lt, axis=-1), idx, axis=-1)          # teacher log p, top-k
            lqs = mx.take_along_axis(ls, idx, axis=-1)                                     # student log q at the same tokens
            pt_tail = mx.maximum(1 - mx.sum(mx.exp(lpt), axis=-1), 1e-6)
            qs_tail = mx.maximum(1 - mx.sum(mx.exp(lqs), axis=-1), 1e-6)
            kl = mx.sum(mx.exp(lpt) * (lpt - lqs), axis=-1) + pt_tail * (mx.log(pt_tail) - mx.log(qs_tail))
        else:
            lpt = nn.log_softmax(lt, axis=-1)
            kl = mx.sum(mx.exp(lpt) * (lpt - ls), axis=-1)
        return T * T * mx.sum(kl * w) / mx.maximum(mx.sum(w), 1.0)

    def train_loss(params, x, y, w):
        compute_model.update(tree_map_with_path(prep, params))
        logits = compute_model(x).astype(mx.float32)  # softmax and CE in fp32
        ce = weighted_ce(logits, y, w)
        if teacher is None:
            return ce
        t_logits = mx.stop_gradient(teacher(x).astype(mx.float32))
        return args.kd_alpha * ce + (1 - args.kd_alpha) * kd_loss(logits, t_logits, w)

    # The gradient through astype comes back in fp32, exactly the type of the master weights.
    loss_and_grad = mx.value_and_grad(train_loss)

    state = [model.state, compute_model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, w):
        loss, grads = loss_and_grad(model.trainable_parameters(), x, y, w)
        grads, gnorm = optim.clip_grad_norm(grads, args.grad_clip)
        optimizer.update(model, grads)
        return loss, gnorm

    args.out.mkdir(parents=True, exist_ok=True)
    cfg.save(args.out / "config.json")
    # The final recipe (with --set and arguments applied), so the run can be reproduced with one command.
    shape_keys = ("num_hidden_layers", "hidden_size", "intermediate_size",
                  "num_attention_heads", "num_key_value_heads", "head_dim")
    (args.out / "recipe.json").write_text(json.dumps({
        "shape": {k: getattr(cfg, k) for k in shape_keys},
        "interface": {"data": str(args.data), "vocab_size": cfg.vocab_size,
                      "pretokenizer": meta.get("pretokenizer", "gpt2"),
                      "seq_len": args.seq_len, "rope_theta": cfg.rope_theta},
        "training": {k: getattr(args, k) for k in RECIPE_ARGS["training"]},
    }, indent=2, ensure_ascii=False))
    shutil.copy(args.data / "tokenizer.json", args.out / "tokenizer.json")
    log = open(args.out / "log.jsonl", "a")
    eot = tok.token_to_id("<|endoftext|>")

    train_log = open(args.out / "train.jsonl", "a")
    run_info = {
        "data": str(args.data), "pretokenizer": meta.get("pretokenizer", "gpt2"),
        "vocab_size": cfg.vocab_size, "num_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size, "num_params": model.num_params(),
        "batch_size": args.batch_size, "seq_len": args.seq_len, "dtype": args.dtype,
        "peak_lr": args.lr, "total_steps": args.steps,
        "total_tokens": args.steps * tokens_per_step, "train_bytes": args.train_bytes,
        "started_at": time.time(),
    }
    last_sample = ""

    def write_status(it: int, state: str, extra: dict | None = None) -> None:
        # Atomic write: the dashboard never reads a half-written file.
        dt = time.perf_counter() - t0
        tok_s = tokens_seen / dt if dt > 0 else 0.0
        st = {**run_info, "state": state, "step": it, "tokens": tokens_seen,
              "elapsed_s": dt, "eta_s": (args.steps - it) * tokens_per_step / tok_s if tok_s else None,
              "tok_s": tok_s, "loss": run_loss, "lr": schedule(it).item(),
              "peak_mem_mb": mx.get_peak_memory() / 2**20,
              "best_val_bpb": (best_val / math.log(2) / bytes_per_token
                               if bytes_per_token and best_val < math.inf else None),
              "last_sample": last_sample, "updated_at": time.time(), **(extra or {})}
        tmp = args.out / "status.json.tmp"
        tmp.write_text(json.dumps(st, ensure_ascii=False))
        tmp.replace(args.out / "status.json")

    best_val = math.inf
    t0, tokens_seen, run_loss = time.perf_counter(), 0, None
    last_step = min(args.steps, args.stop_at_step or args.steps)
    for it in range(1, last_step + 1):
        x, y, w = train.batch(args.batch_size)
        loss, gnorm = step(x, y, w)
        mx.eval(state, loss, gnorm)
        tokens_seen += tokens_per_step
        l = loss.item()
        run_loss = l if run_loss is None else 0.98 * run_loss + 0.02 * l

        if it % 10 == 0:
            write_status(it, "running", {"gnorm": gnorm.item()})

        if it % 50 == 0:
            dt = time.perf_counter() - t0
            rec = {"step": it, "tokens": tokens_seen, "loss": run_loss, "gnorm": gnorm.item(),
                   "lr": schedule(it).item(), "tok_s": tokens_seen / dt}
            train_log.write(json.dumps(rec) + "\n"); train_log.flush()
            print(f"step {it:6d} | loss {run_loss:.3f} | ppl {math.exp(run_loss):8.1f} | "
                  f"gnorm {gnorm.item():.2f} | lr {schedule(it).item():.2e} | "
                  f"{tokens_seen / dt / 1e3:.0f}k tok/s | peak mem {mx.get_peak_memory() / 2**20:.0f} MB")

        if it % args.eval_every == 0 or it == last_step:
            vl = evaluate(model, val_batches)
            # bits-per-byte: loss in nats/token -> bits/token -> bits/byte of source text.
            # Unlike perplexity, it is comparable across tokenizers and vocabulary sizes.
            bpb = vl / math.log(2) / bytes_per_token if bytes_per_token else None
            ol = outlier_stats(model, probe_x)
            rec = {"step": it, "tokens": tokens_seen, "train_loss": run_loss,
                   "val_loss": vl, "val_ppl": math.exp(vl), "val_bpb": bpb,
                   "out_channel": ol["channel"], "out_peak": ol["peak"],
                   "w_peak": ol["w_peak"], "w_kurt": ol["w_kurt"],
                   "out_channel_by_layer": ol["channel_by_layer"], "out_peak_by_layer": ol["peak_by_layer"],
                   "knobs": norm_knobs(model)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"  >> val loss {vl:.3f} | val ppl {math.exp(vl):.1f}"
                  + (f" | val bpb {bpb:.3f}" if bpb else "")
                  + f" | stalactites: channel {ol['channel']:.1f}x, peak {ol['peak']:.0f}x, weight {ol['w_peak']:.1f}x")
            if vl < best_val:
                best_val = vl
                model.save_weights(str(args.out / "model.safetensors"))
        if args.save_every and it % args.save_every == 0:
            (args.out / "ckpt").mkdir(exist_ok=True)
            model.save_weights(str(args.out / "ckpt" / f"step{it:06d}.safetensors"))
            ids = tok.encode(args.sample_prompt).ids
            out = list(generate(model, ids, max_tokens=60, temperature=0.8, stop_ids={eot}))
            last_sample = args.sample_prompt + tok.decode(out)
            print("  >> " + repr(last_sample))

    write_status(last_step, "done" if last_step == args.steps else "stopped", {"gnorm": gnorm.item()})
    print(f"done. best val ppl: {math.exp(best_val):.1f}, checkpoint: {args.out}")


if __name__ == "__main__":
    main()
