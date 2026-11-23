"""A state machine for requested words: soft about WHEN to insert a word (WordBias), strict about HOW to spell it.

States of one requested word (seq is its BPE pieces with a leading space, " luxury" -> ` l` `u` `x` `u` `ry`):
  waiting     : the WordBias bonus is active (cascade + ramp);
  started     : the text ends with the word's first piece (k = 1): the next piece gets a soft start_bonus,
                because the model may have started another word (` l` -> "like"), so it is not forced;
  assembling  : k >= lock pieces match: only the next piece is allowed (+FORCE), the word is written out in full;
  word done   : the text ends with a normal form of the word (the word and its English forms, see forms): pieces that continue the
                word with letters are banned (-FORCE), except those leading to another normal form. This prevents
                "warnerer", "stayer", "examineineine..."; a space, punctuation or a new word are fine.

A used word no longer gets the bonus (WordBias handles that), but the "word done" rule applies every time
it appears.
"""

import re

from tokenizers import Tokenizer

from .word_bias import WordBias

FORCE = 1e9
NOUN_END = ("tion", "sion", "ment", "ness", "ity", "ship", "hood", "ism")
ADJ_END = ("ful", "ous", "ive", "less", "able", "ible", "ish")
IRREGULAR = set("""be begin bite blow break bring build buy catch choose come dig do draw drink drive eat fall feel fight
find fly forget freeze get give go grow have hear hide hold keep know leave lose make meet ride ring rise run say see
sell send shake shine sing sink sit sleep speak spin stand steal swim take teach tell think throw wake wear win
write""".split())


def forms(w: str) -> list[str]:
    """Normal forms of a word by English rules (as mq_word_forms in microqwen.c): the word, -s/-es/-ies, -ed/-d/-ied,
    -ing/-ying with a silent e dropped; the consonant doubles only in one-syllable words (run -> running, not dragon);
    nouns ending in -tion etc. get only -s, adjectives ending in -ful etc. get no forms, irregular verbs get no -ed."""
    out = [w]
    if len(w) < 2:
        return out
    v = lambda c: c in "aeiou"
    z, y, n = w[-1], w[-2], len(w)
    groups = sum(1 for k, c in enumerate(w) if v(c) and (k == 0 or not v(w[k - 1])))
    nounish = any(n > len(e) + 1 and w.endswith(e) for e in NOUN_END) or (n > 4 and groups >= 2 and w.endswith("est"))
    if any(n > len(e) + 1 and w.endswith(e) for e in ADJ_END):
        return out
    cvc = n >= 3 and groups == 1 and not v(z) and z not in "wxy" and v(y) and not v(w[-3])
    cons_y = z == "y" and not v(y)
    sib = z in "sxz" or (z == "h" and y in "cs")
    out.append(w[:-1] + "ies" if cons_y else w + ("es" if sib else "s"))
    if nounish:
        return out
    if w not in IRREGULAR:
        out.append(w + "d" if z == "e" else w[:-1] + "ied" if cons_y else w + z + "ed" if cvc else w + "ed")
    out.append(w[:-2] + "ying" if w.endswith("ie") else w[:-1] + "ing" if z == "e" and y not in "eoy"
               else w + z + "ing" if cvc else w + "ing")
    return out


class WordFSM:
    def __init__(self, tok: Tokenizer, words: list[str], bias: WordBias | None, lock: int = 2,
                 start_bonus: float = 4.0, suffix_rule: bool = True, assemble: bool = True):
        self.bias, self.lock, self.start_bonus = bias, lock, start_bonus
        self.suffix_rule, self.assemble = suffix_rule, assemble
        self.tok = tok
        self.seqs = {w: tok.encode(" " + w).ids for w in words}
        self.used_re = {w: re.compile(rf"\b{re.escape(w)}", re.I) for w in words}
        # normal forms of all requested words as piece sequences (with a leading space and capitalised)
        self.form_seqs: list[list[int]] = []
        for w in words:
            for f in forms(w.lower()):
                for v in (f, f.capitalize()):
                    self.form_seqs.append(tok.encode(" " + v).ids)
                    self.form_seqs.append(tok.encode(v).ids)          # at the start of a line, no space
        vocab = tok.get_vocab_size()
        texts = [tok.decode([i]) for i in range(vocab)]
        self.letter_cont = [i for i, t in enumerate(texts) if t[:1].isalpha()]   # continue a word with letters

    def __call__(self, gen: list[int]) -> dict[int, float]:
        n = len(gen)
        # 1) assembling a multi-piece word: the text ends with the first k pieces of an unfinished requested word
        if self.assemble:
            for w, seq in self.seqs.items():
                for k in range(min(len(seq) - 1, n), 0, -1):
                    if gen[n - k:] == seq[:k]:
                        if k >= self.lock:
                            return {seq[k]: FORCE}
                        # soft bonus for the second piece, also for an already used word: tested that without it
                        # repeated mentions come out mangled more often ("uniforn"), and repeats do not get rarer
                        out = dict(self.bias(gen)) if self.bias else {}
                        out[seq[k]] = out.get(seq[k], 0.0) + self.start_bonus
                        return out
        out = dict(self.bias(gen)) if self.bias else {}
        # 2) word done: the text ends with a normal form of a requested word -> do not continue it with letters
        if self.suffix_rule:
            for s in self.form_seqs:
                L = len(s)
                if n >= L and gen[n - L:] == s:
                    allowed = {s2[L] for s2 in self.form_seqs if len(s2) > L and s2[:L] == s}
                    for t in self.letter_cont:
                        if t not in allowed:
                            out[t] = out.get(t, 0.0) - FORCE
                    break
        return out


def plain_forms_regex(w: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(map(re.escape, forms(w.lower()))) + r")\b", re.I)
