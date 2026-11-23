"""micro-Qwen3 configuration.

Field names match config.json of the real Qwen3 (HF transformers),
so it is easy to compare with the original and to load into mlx-lm.

For comparison, Qwen3-0.6B:
    vocab_size=151936, hidden_size=1024, intermediate_size=3072,
    num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
    head_dim=128, rope_theta=1e6, tie_word_embeddings=True  -> ~0.6B parameters

The defaults are the final micro-qwen3 shape (run shapeX-3x112-attn4x64),
chosen by shape and attention sweeps; the reasoning is in the README.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    hidden_size: int = 112
    intermediate_size: int = 336  # 3x hidden, as in Qwen3-0.6B (3072 / 1024)
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    num_key_value_heads: int = 2  # GQA 2:1, as in Qwen3-0.6B (16 / 8)
    # Head size is not tied to hidden (as in Qwen3: 16 x 128 = 2048 with hidden 1024).
    # 64 is the quality ceiling for hidden ~100 and a size MLX has fast kernels for.
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    # Qwen3 uses 1e6 for long context (32k+). With a 256-token context most frequencies
    # would barely rotate, so we use the classic 1e4.
    rope_theta: float = 10000.0
    max_position_embeddings: int = 256
    tie_word_embeddings: bool = True
    model_type: str = "qwen3"
    # Experimental branch (NOT part of Qwen3): what sits before each block instead of RMSNorm.
    #   rms        : RMSNorm as in Qwen3 (default; the reference-match test covers it)
    #   dyt        : "limiter", gamma * tanh(alpha * x) + beta (Dynamic Tanh)
    #   comp-asinh : "mu-law compressor", T * asinh(u / T) -> RMSNorm, u = x / rms(x), T learned
    #   comp-ratio : "compressor with ratio and soft knee", T, R, K learned -> RMSNorm
    norm_type: str = "rms"
    # Experimental branch: a hidden x hidden matrix before the output layer (logits = (h·M)·E^T).
    # Needed by a model with the rotation baked into its weights, to keep embeddings tied: M absorbs
    # the final RMSNorm gains (M = Q^T·diag(g)·Q). Off by default: it is not part of Qwen3.
    head_mix: bool = False
    # Ternary weights {-1, 0, +1} (deepgrove Bonsai recipe + PrismML Bonsai 2 format), not part of Qwen3:
    # all linear layers and embeddings (tied to the output layer) keep latent weights T, the forward pass uses
    # round(clamp(T, -1, 1)) with a learned per-row scale; each matrix input is rotated by a block
    # Hadamard with random signs (hadamard_block = 0 means no rotation).
    ternary: bool = False
    ternary_embed: bool = True   # False: only layers are ternary, embeddings stay full precision (as in deepgrove)
    hadamard_block: int = 64
    # > 0: a "wide" ternary head instead of the tied one: h -> head_up (hidden -> head_width) -> lm_head
    # (vocab x head_width). The product of two ternary matrices is a nearly full-precision vocab x hidden matrix.
    head_width: int = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text()))

    def param_breakdown(self) -> dict[str, int]:
        d, h = self.hidden_size, self.head_dim
        attn = d * (self.num_attention_heads * h)          # q_proj
        attn += 2 * d * (self.num_key_value_heads * h)     # k_proj, v_proj
        attn += (self.num_attention_heads * h) * d         # o_proj
        attn += 2 * h                                      # q_norm, k_norm
        mlp = 3 * d * self.intermediate_size               # gate, up, down
        norms = 2 * d                                      # input/post_attention layernorm
        per_layer = attn + mlp + norms
        emb = self.vocab_size * d
        head = 0 if self.tie_word_embeddings else emb
        return {
            "embedding": emb,
            "per_layer": per_layer,
            "layers_total": per_layer * self.num_hidden_layers,
            "lm_head": head,
            "final_norm": d,
            "total": emb + per_layer * self.num_hidden_layers + head + d,
        }


PRESETS: dict[str, ModelConfig] = {
    # ~1.06M: final shape (shapeX): 3 layers x 112, attention 4 x 64. Best bpb at ~1M
    # and almost the speed of the original 0.6M model.
    "micro": ModelConfig(),
    # ~0.64M: best shape at the original 0.6M budget (shapeA): 3 x 88, attention 4 x 22.
    "micro-0.6m": ModelConfig(hidden_size=88, intermediate_size=264, head_dim=22),
    # ~0.61M: first version before the shape sweeps (7 x 64, attention 4 x 16), to reproduce
    # early runs (vocabulary sweep, context 512).
    "micro-v0": ModelConfig(hidden_size=64, intermediate_size=192, num_hidden_layers=7, head_dim=16),
    # ~2.23M ternary = 590 KB on the board, like X in W4R (552 KB): T2, equal size
    "micro-t2": ModelConfig(hidden_size=192, intermediate_size=576, ternary=True),
    # ~3.67M ternary = 0.95 MB: the "1 MB on the board" budget, hidden / layers = 32
    "micro-t1mb": ModelConfig(hidden_size=192, intermediate_size=576, num_hidden_layers=6, ternary=True),
    # ~2.07M in int4 = ~1.05 MB: counterpart to micro-t1mb at equal size
    "micro-1mb-int4": ModelConfig(hidden_size=160, intermediate_size=480, num_hidden_layers=4),
    # ~0.18M: for quick smoke tests
    "nano": ModelConfig(hidden_size=32, intermediate_size=96, num_hidden_layers=4, head_dim=8),
}
