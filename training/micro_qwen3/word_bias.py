"""A bonus for requested words when generating an order (logit bias until the word is first used).

The model misses a word when, in its "window" (where it fits by meaning and grammar), its probability is
only a few percent. A +b bonus on the logit lifts it in that window; once the word appears in the text
(in any form: cat -> cats), the bonus is removed so the word is not forced again.

A word the tokenizer splits into pieces ("Ġwar" + "d" + "robe"): with multi_token=True the bonus goes to its
first piece, and if the last tokens already match the start of the word, to the next piece.

mode: how the bonus changes once some words have appeared (n words in the order, u of them used):
  const     : every remaining word gets b;
  escalate  : b·(u + 1): after the first word the rest get 2b, after the second 3b (push the last one home);
  conserve  : b·n / (n - u): a total budget of b·n is shared by the remaining words.

ramp_tau > 0: the bonus also grows over time, x(1 + t / ramp_tau), where t is how many tokens have passed since
the last requested word appeared (or since the answer started). Proportions between words stay the same, the
lever itself grows: the longer a word is missing, the harder we pull; a new word resets t.

full_start: how much of a full order the start assumes; u starts at full_start·(max_words - n). 1 (True): the
cascade of a full order,
a single word gets the pressure of the last of three right away (escalate: 3b); 0.5 is the board rule u0 = (3 - n)/2:
one word 2b, two words 1.5b; an order of three is unchanged for any value.
u0: an explicit cascade start (for example 1: a single word gets 2b), but no more than max_words - n.
"""

import re

from tokenizers import Tokenizer


class WordBias:
    def __init__(self, tok: Tokenizer, words: list[str], bias: float, multi_token: bool = False, mode: str = "const",
                 ramp_tau: float = 0.0, full_start: float = 0.0, max_words: int = 3, u0: float | None = None):
        assert mode in ("const", "escalate", "conserve"), mode
        self.tok, self.bias, self.mode, self.n, self.tau = tok, bias, mode, len(words), ramp_tau
        self.u0 = float(full_start) * max(max_words - len(words), 0)
        if u0 is not None:                   # explicit cascade start (no more than a full order)
            self.u0 = min(u0, max(max_words - len(words), 0))
        self.last_used, self.since = -1, 0          # how many words were used and since which step
        self.seqs = {}
        for w in words:
            ids = tok.encode(" " + w).ids
            if len(ids) == 1 or multi_token:
                self.seqs[w] = ids
        self.pattern = {w: re.compile(rf"\b{re.escape(w)}", re.I) for w in self.seqs}

    def __call__(self, generated: list[int]) -> dict[int, float]:
        if not self.bias or not self.seqs:
            return {}
        text = self.tok.decode(generated) if generated else ""
        left = [w for w in self.seqs if not self.pattern[w].search(text)]   # used words get no bonus
        used = self.n - len(left) - (self.n - len(self.seqs))              # multi-piece words without multi_token do not count
        b = {"const": self.bias, "escalate": self.bias * (used + self.u0 + 1),
             "conserve": self.bias * len(self.seqs) / max(len(left), 1)}[self.mode]
        if used != self.last_used:                   # a new word appeared (or first call): restart the count
            self.last_used, self.since = used, len(generated)
        if self.tau:
            b *= 1 + (len(generated) - self.since) / self.tau
        out: dict[int, float] = {}
        for w in left:
            seq = self.seqs[w]
            j = 0                                          # how many leading pieces of the word already sit at the end of the text
            for k in range(min(len(seq) - 1, len(generated)), 0, -1):
                if generated[-k:] == seq[:k]:
                    j = k
                    break
            out[seq[j]] = out.get(seq[j], 0.0) + b
        return out
