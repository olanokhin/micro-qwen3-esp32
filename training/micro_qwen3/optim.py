"""Optimizers for the "stalactite" experiments: outliers in weights and activations.

Hypothesis: outliers grow from step to step because AdamW normalises the step per coordinate,
so the residual stream gets a "privileged basis" and individual channels run away.
Two ways to avoid it:

  adamw-rot : AdamW whose moments and per-coordinate normalisation live in a rotated basis
              of the residual stream (a fixed random orthogonal matrix Q, d x d).
              For SGD the rotation would change nothing; for Adam it does, because Adam is not
              rotation invariant. The model function and architecture stay the same.
  muon      : Muon (Newton-Schulz orthogonalisation of the step) for matrices inside layers,
              AdamW for embeddings and 1-D parameters (norms).
"""

import math

import mlx.core as mx
import mlx.optimizers as optim


class RotatedAdamW(optim.AdamW):
    """AdamW working in a rotated basis of the residual stream of size `hidden_size`.

    For a matrix W with an axis of size d (input: W @ x, or output: W^T) the gradient is rotated
    g' = g Q (input axis) and/or g' = Q^T g (output axis), Adam keeps its moments in these
    coordinates, and the step is rotated back. Weight decay commutes with the rotation.
    """

    def __init__(self, hidden_size: int, seed: int = 0, block: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if block:  # block Hadamard: only coordinates within a block mix (a "dose" of rotation)
            assert hidden_size % block == 0 and block & (block - 1) == 0, "block must be a power of two that divides hidden"
            h = mx.array([[1.0]])
            while h.shape[0] < block:
                h = mx.concatenate([mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0)
            h = h / math.sqrt(block)
            q = mx.zeros((hidden_size, hidden_size))
            for i in range(0, hidden_size, block):
                q[i : i + block, i : i + block] = h
        else:      # full rotation: a random orthogonal matrix
            a = mx.random.normal((hidden_size, hidden_size), key=mx.random.key(seed))
            q, _ = mx.linalg.qr(a, stream=mx.cpu)
        self.Q = q.astype(mx.float32)
        self.d = hidden_size

    def _rotate(self, x: mx.array, inverse: bool = False) -> mx.array:
        if x.ndim != 2:
            return x
        Q = self.Q.T if inverse else self.Q
        if x.shape[-1] == self.d:   # input axis: embeddings, q/k/v, gate/up
            x = x @ Q
        if x.shape[0] == self.d:    # output axis into the residual stream: o_proj, down_proj
            x = Q.T @ x
        return x

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        g = self._rotate(gradient)
        m = b1 * state["m"] + (1 - b1) * g
        v = b2 * state["v"] + (1 - b2) * mx.square(g)
        state["m"], state["v"] = m, v
        if self.bias_correction:
            step = self.step.astype(gradient.dtype)
            m, v = m / (1 - b1**step), v / (1 - b2**step)
        update = self._rotate(m / (mx.sqrt(v) + self.eps), inverse=True)
        return parameter * (1 - lr * self.weight_decay) - lr * update


class CautiousAdamW(optim.AdamW):
    """Cautious AdamW (Liang et al., 2025; used to train deepgrove Bonsai): the Adam step is applied only in
    coordinates where its sign matches the sign of the current gradient, and is renormalised by the share of such
    coordinates so the mean step size does not drop."""

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        m = b1 * state["m"] + (1 - b1) * gradient
        v = b2 * state["v"] + (1 - b2) * mx.square(gradient)
        state["m"], state["v"] = m, v
        step = self.step.astype(gradient.dtype)
        u = (m / (1 - b1**step)) / (mx.sqrt(v / (1 - b2**step)) + self.eps)
        mask = (u * gradient > 0).astype(u.dtype)
        u = u * mask / mx.maximum(mx.mean(mask), 1e-3)
        return parameter * (1 - lr * self.weight_decay) - lr * u


# compressor/limiter knobs and row scales of ternary matrices: no weight decay
KNOBS = ("log_t", "r_raw", "k_raw", "alpha", "scales")


def build_optimizer(name: str, hidden_size: int, schedule, muon_schedule=None,
                    weight_decay: float = 0.1, seed: int = 0, rot_block: int | None = None):
    main = _build_main(name, hidden_size, schedule, muon_schedule, weight_decay, seed, rot_block)
    knob_cls = CautiousAdamW if name == "adamw-cautious" else optim.AdamW
    knobs = knob_cls(learning_rate=schedule, betas=[0.9, 0.95], weight_decay=0.0)
    is_knob = lambda path, g: path.rsplit(".", 1)[-1] in KNOBS  # noqa: E731
    return optim.MultiOptimizer([knobs, main], [is_knob])


def _build_main(name, hidden_size, schedule, muon_schedule, weight_decay, seed, rot_block):
    betas = [0.9, 0.95]
    if name == "adamw":
        return optim.AdamW(learning_rate=schedule, betas=betas, weight_decay=weight_decay)
    if name == "adamw-cautious":
        return CautiousAdamW(learning_rate=schedule, betas=betas, weight_decay=weight_decay)
    if name == "adamw-rot":
        return RotatedAdamW(hidden_size, seed=seed, block=rot_block, learning_rate=schedule,
                            betas=betas, weight_decay=weight_decay)
    if name == "muon":
        muon = optim.Muon(learning_rate=muon_schedule, momentum=0.95, weight_decay=0.01)
        adamw = optim.AdamW(learning_rate=schedule, betas=betas, weight_decay=weight_decay)
        is_hidden_matrix = lambda path, g: g.ndim == 2 and "embed_tokens" not in path  # noqa: E731
        return optim.MultiOptimizer([muon, adamw], [is_hidden_matrix])
    raise ValueError(f"unknown optimizer: {name}")


def outlier_stats(model, x: mx.array) -> dict:
    """A "stalactite ruler" on a fixed batch.

    channel: loudest residual-stream channel / median channel (by channel RMS), max over layers;
    peak:    max |activation| / RMS of all activations, max over layers;
    w_peak:  max |weight| / weight RMS over all matrices; w_kurt: max weight kurtosis over matrices.
    """
    from mlx.utils import tree_flatten

    h = model.model.embed_tokens(x)
    channel, peak = [], []
    for layer in model.model.layers:
        h = layer(h, "causal")
        a = h.astype(mx.float32).reshape(-1, h.shape[-1])
        ch = mx.sqrt(mx.mean(a * a, axis=0))
        med = mx.sort(ch)[ch.shape[0] // 2]
        channel.append((mx.max(ch) / med).item())
        peak.append((mx.max(mx.abs(a)) / mx.sqrt(mx.mean(a * a))).item())
    w_peak, w_kurt = 0.0, 0.0
    for _, w in tree_flatten(model.parameters()):
        if w.ndim != 2:
            continue
        w = w.astype(mx.float32)
        r = mx.sqrt(mx.mean(w * w))
        z = (w - mx.mean(w)) / mx.std(w)
        w_peak = max(w_peak, (mx.max(mx.abs(w)) / r).item())
        w_kurt = max(w_kurt, (mx.mean(z**4) - 3).item())
    return {"channel": max(channel), "peak": max(peak), "channel_by_layer": channel,
            "peak_by_layer": peak, "w_peak": w_peak, "w_kurt": w_kurt}


def norm_knobs(model) -> list[dict]:
    """Learned knobs of the "chains" before blocks (T, R, K or alpha) per layer, plus the final norm."""
    out = []
    for i, layer in enumerate(model.model.layers):
        for name in ("input_layernorm", "post_attention_layernorm"):
            m = getattr(layer, name)
            if hasattr(m, "knobs"):
                out.append({"where": f"{i}.{name.split('_')[0]}", **m.knobs()})
    if hasattr(model.model.norm, "knobs"):
        out.append({"where": "final", **model.model.norm.knobs()})
    return out
