from pathlib import Path

import mlx.core as mx
import numpy as np


class TokenStream:
    """Random windows of length seq_len+1 from a flat token array (uint16 memmap)."""

    def __init__(self, path: str | Path, seq_len: int, seed: int = 0, weights: bool = False):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        # weights=True: batches are triples (x, y, w), w is the weight of each target token. If
        # <name>.mask.bin sits next to it (dialogues), w is the "this is the answer" mask, otherwise ones.
        self.weights = weights
        mask = Path(path).with_suffix(".mask.bin")
        self.mask = np.memmap(mask, dtype=np.uint8, mode="r") if weights and mask.exists() else None

    def _pack(self, starts: np.ndarray):
        x = np.stack([self.data[s : s + self.seq_len + 1] for s in starts]).astype(np.int32)
        out = [mx.array(x[:, :-1]), mx.array(x[:, 1:])]
        if self.weights:
            w = (np.stack([self.mask[s + 1 : s + self.seq_len + 1] for s in starts]).astype(np.float32)
                 if self.mask is not None else np.ones((len(starts), self.seq_len), np.float32))
            out.append(mx.array(w))
        return tuple(out)

    def __len__(self) -> int:
        return len(self.data)

    def fixed_batches(self, n_batches: int, batch_size: int) -> list[tuple[mx.array, mx.array]]:
        """A deterministic set of windows spread evenly over the whole array.

        For validation: every evaluation runs on the same texts, so the curve
        reflects changes in the model, not which pieces happened to be drawn.
        """
        n = n_batches * batch_size
        starts = np.linspace(0, len(self.data) - self.seq_len - 1, n).astype(np.int64)
        return [self._pack(starts[i : i + batch_size]) for i in range(0, n, batch_size)]

    def batch(self, batch_size: int) -> tuple[mx.array, mx.array]:
        starts = self.rng.integers(0, len(self.data) - self.seq_len - 1, batch_size)
        return self._pack(starts)
