"""Loop penalty (a soft constraint like WordBias, only with a minus sign).

Mentioning the hero many times is fine ("the dragon ... the dragon"); looping is bad:
  - an immediate repeat inside a phrase: "hot hot", "brave and brave", "very tall and very tall"  -> -imm on the logit
    of the token that makes it;
  - a repeated sequence within a window (the DRY sampler idea): if the last k >= 2 tokens already occurred in the last
    window tokens and were then followed by token t, t gets -mult·base^(k-2) (the longer the match, the stronger).
Punctuation and line breaks end a match: "Tim. Tim loved" is not penalised.
"""

from tokenizers import Tokenizer

BREAK_CHARS = set('.,!?;:"\n')


class LoopPenalty:
    def __init__(self, tok: Tokenizer, imm: float = 0.0, mult: float = 0.0, base: float = 1.75, window: int = 32,
                 max_k: int = 8):
        self.imm, self.mult, self.base, self.window, self.max_k = imm, mult, base, window, max_k
        vocab = tok.get_vocab_size()
        texts = [tok.decode([i]) for i in range(vocab)]
        self.breaker = {i for i, t in enumerate(texts) if any(c in BREAK_CHARS for c in t)}
        self.joiner = {i for i, t in enumerate(texts) if t in (" and", " or")}
        self.word_start = {i for i, t in enumerate(texts) if t.startswith(" ")}

    def __call__(self, gen: list[int]) -> dict[int, float]:
        out: dict[int, float] = {}
        n = len(gen)
        if not n:
            return out
        if self.imm:
            # "X X": repeating the last token, if it starts a word (otherwise it is a word piece, "hot" + "ter")
            if gen[-1] in self.word_start and gen[-1] not in self.breaker:
                out[gen[-1]] = out.get(gen[-1], 0.0) - self.imm
            # "X and X"
            if n >= 2 and gen[-1] in self.joiner and gen[-2] not in self.breaker:
                out[gen[-2]] = out.get(gen[-2], 0.0) - self.imm
            # "X and W X": a workaround through an inserted word, "very tall and very tall"
            if n >= 3 and gen[-2] in self.joiner and gen[-3] in self.word_start and gen[-3] not in self.breaker \
                    and gen[-1] in self.word_start and gen[-1] not in self.breaker and gen[-1] != gen[-3]:
                out[gen[-3]] = out.get(gen[-3], 0.0) - self.imm
        if self.mult and n >= 3:
            seq: dict[int, float] = {}          # the strongest penalty per token (from the longest match)
            lo = max(0, n - self.window)
            for j in range(lo + 2, n):          # candidate gen[j]: does what precedes it match the end of the text
                t = gen[j]
                k = 0
                while k < self.max_k and j - 1 - k >= lo and gen[j - 1 - k] == gen[n - 1 - k] \
                        and gen[n - 1 - k] not in self.breaker and n - 1 - k > j - 1:
                    k += 1
                if k >= 2:
                    seq[t] = max(seq.get(t, 0.0), self.mult * self.base ** (k - 2))
            for t, pen in seq.items():
                out[t] = out.get(t, 0.0) - pen
        return out


def combine(bias_fn, penalty_fn=None):
    """Bonus for requested words + loop penalty. Tokens the bonus is pushing right now (the next piece of a word
    that has not appeared yet) are exempt from the penalty, otherwise the penalty could take the requested word away."""
    if penalty_fn is None:
        return bias_fn
    if bias_fn is None:
        return penalty_fn

    def f(gen):
        out = dict(bias_fn(gen))
        for k, v in penalty_fn(gen).items():
            if k not in out:
                out[k] = v
        return out
    return f
