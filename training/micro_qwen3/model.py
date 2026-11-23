"""The Qwen3 architecture, rebuilt layer by layer in MLX.

Layout (top to bottom):

    tokens ─► embed_tokens ─► [DecoderLayer] x N ─► norm ─► lm_head (= embed_tokens^T) ─► logits

    DecoderLayer (pre-norm residual):
        h = x + Attention(RMSNorm(x))
        y = h + MLP(RMSNorm(h))

    Attention (GQA + QK-norm + RoPE):
        q = RMSNorm_per_head(Wq x), k = RMSNorm_per_head(Wk x), v = Wv x
        q, k = RoPE(q), RoPE(k)
        out = Wo · softmax(q k^T / sqrt(d_head) + causal_mask) v

    MLP (SwiGLU):
        y = W_down( silu(W_gate x) * (W_up x) )

Qwen3 vs Qwen2: QK-norm (RMSNorm on q and k per head) and no bias in q/k/v.
Vs classic GPT-2: RMSNorm instead of LayerNorm, RoPE instead of learned positions,
SwiGLU instead of a GELU MLP, GQA, no bias.

Module names match HF/mlx-lm, so weights are interchangeable.
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .config import ModelConfig


# ---------------------------------------------------------------- 1. RMSNorm
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight.

    Unlike LayerNorm it does not subtract the mean and has no bias.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Equivalent: x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + eps) * weight
        # mx.fast.rms_norm does the same in one fused Metal kernel.
        return mx.fast.rms_norm(x, self.weight, self.eps)


# ---------------------------------------------------------------- 1b. Experimental "chains" before a block
# Not part of Qwen3: testing the hypothesis "loud channels squash quiet ones during normalisation".
# Compressors work on the vector normalised by its RMS (u = x / rms(x)): the threshold T is in units of
# "how many times louder than average", like an auto-leveller. Then a regular RMSNorm.

def _rms_unit(x: mx.array, eps: float) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True) + eps).astype(x.dtype)


class DynTanh(nn.Module):
    """A "limiter" (Dynamic Tanh): gamma * tanh(alpha * x) + beta instead of normalisation."""

    def __init__(self, dims: int, alpha0: float = 0.5):
        super().__init__()
        self.alpha = mx.array([alpha0])
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))

    def __call__(self, x):
        return self.weight * mx.tanh(self.alpha * x) + self.bias

    def knobs(self) -> dict:
        return {"alpha": self.alpha.item()}


class CompAsinh(nn.Module):
    """A "mu-law compressor": T * asinh(u / T), then RMSNorm. Below T almost linear,
    above it logarithmic, with no ceiling. T is learned (through log, so it stays > 0)."""

    def __init__(self, dims: int, eps: float, t0: float = 3.0):
        super().__init__()
        self.log_t = mx.array([math.log(t0)])
        self.norm = RMSNorm(dims, eps)
        self.eps = eps

    def __call__(self, x):
        t = mx.exp(self.log_t).astype(x.dtype)
        return self.norm(t * mx.arcsinh(_rms_unit(x, self.eps) / t))

    def knobs(self) -> dict:
        return {"T": math.exp(self.log_t.item())}


class CompRatio(nn.Module):
    """A "compressor with ratio and soft knee", then RMSNorm. For |u| above threshold T the slope is 1/R,
    within a knee of width K a smooth quadratic join (the classic soft-knee formula).
    T, R, K are learned: T = exp(.), R = 1 + softplus(.), K = softplus(.)."""

    def __init__(self, dims: int, eps: float, t0: float = 3.0, r0: float = 4.0, k0: float = 1.0):
        super().__init__()
        inv_softplus = lambda y: math.log(math.expm1(y))  # noqa: E731
        self.log_t = mx.array([math.log(t0)])
        self.r_raw = mx.array([inv_softplus(r0 - 1)])
        self.k_raw = mx.array([inv_softplus(k0)])
        self.norm = RMSNorm(dims, eps)
        self.eps = eps

    def params(self):
        return mx.exp(self.log_t), 1 + nn.softplus(self.r_raw), nn.softplus(self.k_raw) + 1e-3

    def __call__(self, x):
        t, r, k = (p.astype(x.dtype) for p in self.params())
        u = _rms_unit(x, self.eps)
        a = mx.abs(u)
        slope = 1 / r - 1
        knee = a + slope * mx.square(a - t + k / 2) / (2 * k)      # inside the knee
        above = t + (a - t) / r                                   # above the knee
        y = mx.where(a <= t - k / 2, a, mx.where(a >= t + k / 2, above, knee))
        return self.norm(mx.sign(u) * y)

    def knobs(self) -> dict:
        t, r, k = self.params()
        return {"T": t.item(), "R": r.item(), "K": k.item()}


def make_block_norm(cfg: "ModelConfig") -> nn.Module:
    if cfg.norm_type == "rms":
        return RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    if cfg.norm_type == "dyt":
        return DynTanh(cfg.hidden_size)
    if cfg.norm_type == "comp-asinh":
        return CompAsinh(cfg.hidden_size, cfg.rms_norm_eps)
    if cfg.norm_type == "comp-ratio":
        return CompRatio(cfg.hidden_size, cfg.rms_norm_eps)
    raise ValueError(f"unknown norm_type: {cfg.norm_type}")


# ---------------------------------------------------------------- 1c. Ternary weights (experimental branch)
# Training recipe: deepgrove Bonsai, y = (v ⊙ rownorm(Q(T))) x, Q(T) = round(clamp(T, -1, 1)),
# gradient passes through the rounding unchanged (STE), v is a learned per-row scale.
# Format: PrismML Bonsai 2 (checked against their weights): everything is ternary, including tied embeddings,
# and each matrix input is rotated: R = H·S/√b, a block Walsh-Hadamard H and fixed signs S.
# The rotation cannot be baked into ternary weights (W·R is no longer ternary), so it is computed on the fly.

_HADAMARD: dict[int, mx.array] = {}


def _hadamard(b: int) -> mx.array:
    """Normalised Sylvester matrix b x b: symmetric, orthogonal, H·H = I."""
    if b not in _HADAMARD:
        h = mx.array([[1.0]])
        while h.shape[0] < b:
            h = mx.concatenate([mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0)
        _HADAMARD[b] = h / math.sqrt(b)
    return _HADAMARD[b]


def hadamard_rotate(x: mx.array, signs: mx.array, block: int) -> mx.array:
    """x -> R x for row vectors: signs, then Hadamard within each block."""
    if not block:
        return x
    shape = x.shape
    y = (x * signs.astype(x.dtype)).reshape(*shape[:-1], shape[-1] // block, block)
    return (y @ _hadamard(block).astype(x.dtype)).reshape(shape)


def hadamard_unrotate(y: mx.array, signs: mx.array, block: int) -> mx.array:
    """Inverse rotation: x = R^T y = S·H·y."""
    if not block:
        return y
    shape = y.shape
    x = y.reshape(*shape[:-1], shape[-1] // block, block) @ _hadamard(block).astype(y.dtype)
    return x.reshape(shape) * signs.astype(y.dtype)


def ternary_weight(t: mx.array, scales: mx.array) -> mx.array:
    """Effective matrix (in the rotated basis): v_i · q_i / ||q_i||, q = round(clamp(T, -1, 1))."""
    q = t + mx.stop_gradient(mx.round(mx.clip(t, -1, 1)) - t)
    # max before the root: an all-zero row has Σq² = 0, and the derivative of sqrt at zero is infinite, 0·∞ = NaN.
    # q are integers, so a non-zero row has Σq² >= 1 and max changes nothing.
    norm = mx.sqrt(mx.maximum(mx.sum(mx.square(q), axis=1, keepdims=True), 1.0))
    return q * (scales[:, None].astype(t.dtype) / norm)


class TernaryLinear(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, block: int):
        super().__init__()
        assert not block or in_dims % block == 0, f"input {in_dims} is not divisible by the Hadamard block {block}"
        self.weight = mx.zeros((out_dims, in_dims))   # latent weights T (in the rotated basis)
        self.scales = mx.ones((out_dims,))            # v
        self.signs = mx.ones((in_dims,))              # S, fixed; set in init_ternary
        self.block = block

    def __call__(self, x: mx.array) -> mx.array:
        return hadamard_rotate(x, self.signs, self.block) @ ternary_weight(self.weight, self.scales).T


class TernaryEmbedding(nn.Module):
    """Tied embeddings: row i = R^T·w_i on input, logits = W (R h) on output."""

    def __init__(self, num_embeddings: int, dims: int, block: int):
        super().__init__()
        assert not block or dims % block == 0, f"hidden {dims} is not divisible by the Hadamard block {block}"
        self.weight = mx.zeros((num_embeddings, dims))
        self.scales = mx.ones((num_embeddings,))
        self.signs = mx.ones((dims,))
        self.block = block

    def __call__(self, ids: mx.array) -> mx.array:
        w = ternary_weight(self.weight, self.scales)[ids]
        return hadamard_unrotate(w, self.signs, self.block)

    def as_linear(self, h: mx.array) -> mx.array:
        return hadamard_rotate(h, self.signs, self.block) @ ternary_weight(self.weight, self.scales).T


def make_linear(cfg: "ModelConfig", in_dims: int, out_dims: int) -> nn.Module:
    if cfg.ternary:
        return TernaryLinear(in_dims, out_dims, cfg.hadamard_block)
    return nn.Linear(in_dims, out_dims, bias=False)


# ---------------------------------------------------------------- 2. RoPE
def apply_rope(x: mx.array, base: float, offset: int = 0) -> mx.array:
    """Rotary Position Embedding. x: (B, n_heads, L, head_dim).

    Coordinate pairs (i, i + d/2) are rotated by the angle pos * base^(-2i/d).
    traditional=False is the "rotate_half" variant used by Qwen/Llama in HF.
    """
    return mx.fast.rope(x, x.shape[-1], traditional=False, base=base, scale=1.0, offset=offset)


# ---------------------------------------------------------------- KV-cache
class KVCache:
    """The simplest key/value cache for generation."""

    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        self.offset += k.shape[2]
        return self.keys, self.values


def causal_mask(L: int, offset: int = 0) -> mx.array:
    """Boolean mask (L, offset+L): position i sees all positions <= i."""
    q_pos = mx.arange(offset, offset + L)
    k_pos = mx.arange(offset + L)
    return q_pos[:, None] >= k_pos[None, :]


def attention_mask(L: int, offset: int = 0):
    """Which mask to pass to SDPA.

    - L == 1 (generating one token at a time): no mask needed, the token sees the whole cache.
    - offset == 0 (training, evaluation, first generation pass): the string "causal",
      the fused MLX kernel masks the future itself without building an L x L matrix in memory.
    - otherwise (several new tokens on top of the cache): an explicit boolean mask.
    """
    if L == 1:
        return None
    if offset == 0:
        return "causal"
    return causal_mask(L, offset)


# ---------------------------------------------------------------- 3. Attention
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, hd = cfg.hidden_size, cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        assert self.n_heads % self.n_kv_heads == 0, "GQA: n_heads must be divisible by n_kv_heads"
        self.scale = hd**-0.5
        self.rope_theta = cfg.rope_theta

        self.q_proj = make_linear(cfg, d, self.n_heads * hd)
        self.k_proj = make_linear(cfg, d, self.n_kv_heads * hd)
        self.v_proj = make_linear(cfg, d, self.n_kv_heads * hd)
        self.o_proj = make_linear(cfg, self.n_heads * hd, d)

        # QK-norm is the Qwen3 signature: it stabilises attention logits.
        self.q_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache: KVCache | None = None) -> mx.array:
        B, L, _ = x.shape

        # (B, L, H*hd) -> (B, L, H, hd) -> norm over hd -> (B, H, L, hd)
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = apply_rope(q, self.rope_theta, offset)
        k = apply_rope(k, self.rope_theta, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # SDPA in MLX "repeats" kv heads for GQA by itself (n_heads / n_kv_heads).
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------- 4. MLP (SwiGLU)
class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dim, hidden_dim = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = make_linear(cfg, dim, hidden_dim)
        self.up_proj = make_linear(cfg, dim, hidden_dim)
        self.down_proj = make_linear(cfg, hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------- 5. Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = make_block_norm(cfg)
        self.post_attention_layernorm = make_block_norm(cfg)

    def __call__(self, x: mx.array, mask=None, cache: KVCache | None = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


# ---------------------------------------------------------------- 6. The whole model
class Qwen3Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.ternary and cfg.ternary_embed:
            assert cfg.tie_word_embeddings or cfg.head_width, "ternary embeddings: tied or with a wide head"
            self.embed_tokens = TernaryEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.hadamard_block)
        else:
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = make_block_norm(cfg)

    def __call__(self, inputs: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        h = self.embed_tokens(inputs)
        L = h.shape[1]
        offset = cache[0].offset if cache is not None else 0
        mask = attention_mask(L, offset)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        if cfg.head_width:
            assert cfg.ternary and not cfg.tie_word_embeddings, "wide head is for a ternary model without tying"
            self.head_up = TernaryLinear(cfg.hidden_size, cfg.head_width, cfg.hadamard_block)
            self.lm_head = TernaryLinear(cfg.head_width, cfg.vocab_size, cfg.hadamard_block)
        elif not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.head_mix:
            self.head_mix = mx.eye(cfg.hidden_size)

    def __call__(self, inputs: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        h = self.model(inputs, cache)
        if self.cfg.head_mix:
            h = h @ self.head_mix
        if self.cfg.head_width:
            return self.lm_head(self.head_up(h))
        if self.cfg.tie_word_embeddings:
            # lm_head shares its matrix with embed_tokens: logits = h @ E^T
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def num_params(self) -> int:
        return sum(v.size for _, v in tree_flatten(self.parameters()))


def init_weights(model: nn.Module, std: float = 0.02) -> None:
    """Initialisation as in HF Qwen3 (_init_weights): N(0, 0.02) for matrices, 1 for norms."""
    new = [
        (k, v if v.ndim == 1 else mx.random.normal(v.shape) * std)
        for k, v in tree_flatten(model.parameters())
    ]
    model.update(tree_unflatten(new))


def init_ternary(model: nn.Module, seed: int = 0, method: str = "twn") -> None:
    """Start ternary matrices from already initialised W (random or trained), as in deepgrove:
    T = sign(w)·[|w| > threshold] ∈ {-1, 0, +1}, v_i = ||w_i||, rotation signs S are random ±1.

    method: twn    : threshold 0.7·mean|w_i| (Ternary Weight Networks; ~40% zeros for Gaussian weights),
            thirds : threshold = 1/3 quantile of |w_i|: exactly a third zeros, as in the Bonsai 2 weights.
    Trained W are first moved to the rotated basis (W·R^T) so the layer function does not change.
    """
    key = mx.random.key(seed)
    for _, m in model.named_modules():
        if not isinstance(m, (TernaryLinear, TernaryEmbedding)):
            continue
        key, sub = mx.random.split(key)
        m.signs = mx.where(mx.random.bernoulli(0.5, m.signs.shape, key=sub), 1.0, -1.0)
        w = hadamard_rotate(m.weight.astype(mx.float32), m.signs, m.block)   # W R^T: the same x -> W x
        a = mx.abs(w)
        if method == "twn":
            thr = 0.7 * mx.mean(a, axis=1, keepdims=True)
        elif method == "thirds":
            thr = mx.sort(a, axis=1)[:, a.shape[1] // 3][:, None]
        else:
            raise ValueError(f"unknown ternarisation method: {method}")
        m.weight = mx.sign(w) * (a > thr)
        m.scales = mx.sqrt(mx.sum(w * w, axis=1))
    model.freeze(keys=["signs"])
