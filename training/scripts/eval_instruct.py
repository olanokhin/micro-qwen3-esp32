"""Does the model follow the request: "write a story with the words ..." tasks from TinyStories-Instruct-valid.

    uv run training/scripts/eval_instruct.py --ckpt runs/tsi-sft-x --ckpt runs/ts2-qat-w4r

The request holds only the words and fixed options (as in tinystories-instruct-wf). Metrics over N tasks with a Words field:
  words    : share of requested words that appear in the answer (any ending: dog -> dogs);
  stop     : share of answers closed by <|im_end|> (the model knows where the answer ends);
  len      : mean answer length in tokens;
  dialogue : share of stories with quotes when dialogue was requested and when not (does it follow the option);
  nll      : mean −log p of answer tokens under the model itself without the bonus: how natural the text is for it
             (grows when the bonus forces words in);
  repeats  : share of word 4-grams already seen in this story (loops like "the warning was a warning").
For a base model that never saw the chat format, the same prompts give a baseline of "random" word hits.
"""

import argparse
import json
import re
import sys
from itertools import islice
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

from micro_qwen3.config import ModelConfig
from micro_qwen3.model import Qwen3ForCausalLM
from micro_qwen3.sampling import generate
from micro_qwen3.loop_penalty import LoopPenalty, combine
from micro_qwen3.word_bias import WordBias
from micro_qwen3.word_fsm import WordFSM

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import stream_txt, to_chat  # noqa: E402


def tasks(n: int, n_words: int = 3) -> list[tuple[str, list[str], bool]]:
    """Tasks "write a story with the words ..."; with n_words < 3 the request keeps the first n_words words of the triple."""
    out = []
    for doc in stream_txt("roneneldan/TinyStoriesInstruct", "TinyStories-Instruct-valid.txt"):
        chat = to_chat(doc, words_features_only=True)   # request: only the words and fixed options
        m = re.search(r"Use the words: ([^.]*)\.", chat or "")
        if not chat or not m or "Summarize" in chat:
            continue
        prompt = chat[: chat.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")]
        words = [w.strip().lower() for w in m.group(1).split(",") if w.strip()][:n_words]
        prompt = prompt.replace(m.group(0), f"Use the words: {', '.join(words)}.")
        out.append((prompt, words, "dialogue" in prompt))
        if len(out) == n:
            return out
    return out


LOOP_RE = re.compile(r"\b([a-z']+)(?: (?:and|or)(?: [a-z']+)?)? \1\b")   # "hot hot", "tall and tall", "tall and very tall"


def loop_stats(text: str, ws: list[str], window: int = 20) -> tuple[float, float]:
    """Looping, not just repetition: (share of words repeating the previous word within a phrase, directly or
    via and/or: "hot hot", "brave and brave"; punctuation between them is no longer a loop: "Tim. Tim loved";
    share of 3-grams already seen in the last window words)."""
    if len(ws) < 2:
        return 0.0, 0.0
    imm = sum(1 for _ in re.finditer(r"(?=" + LOOP_RE.pattern + ")", text))
    grams = [tuple(ws[i: i + 3]) for i in range(len(ws) - 2)]
    loc = sum(g in grams[max(0, i - window): i] for i, g in enumerate(grams))
    return imm / len(ws), loc / max(len(grams), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, action="append", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--word-bias", type=float, default=0.0, help="bonus to the logit of a requested word until it first appears")
    p.add_argument("--bias-mode", choices=["const", "escalate", "conserve"], default="const")
    p.add_argument("--triples", type=Path, default=None,
                   help="your own \"verb,noun,adjective\" triples, one per line; the request is only the words, no options")
    p.add_argument("--samples", type=int, default=1, help="how many stories per task (different seeds)")
    p.add_argument("--ramp-tau", type=float, default=0.0, help="the bonus grows ×(1 + t/tau) with tokens without a new word")
    p.add_argument("--n-words", type=int, default=3, help="how many words of the triple to keep in the request (1-3)")
    p.add_argument("--bias-full-start", type=float, nargs="?", const=1.0, default=0.0,
                   help="cascade start u0 = fraction·(3 − n): without a value 1 (full), 0.5 is the board rule")
    p.add_argument("--bias-u0", type=float, default=None, help="explicit cascade start u0 (at most 3 − n)")
    p.add_argument("--loop-imm", type=float, default=0.0, help="penalty for an immediate repeat \"X X\" / \"X and X\"")
    p.add_argument("--loop-mult", type=float, default=0.0, help="penalty for repeating a sequence (DRY): mult·base^(k−2)")
    p.add_argument("--loop-window", type=int, default=32)
    p.add_argument("--fsm", action="store_true", help="automaton: strict assembly of multi-token words + no junk suffixes")
    p.add_argument("--fsm-lock", type=int, default=2, help="after how many matched pieces the word is assembled strictly")
    p.add_argument("--fsm-parts", choices=["both", "assemble", "suffix"], default="both")
    p.add_argument("--dump", type=Path, default=None, help="dump stories (words, text) to jsonl")
    p.add_argument("--show-loops", action="store_true", help="print stories with loops")
    p.add_argument("--bias-multi", action="store_true", help="also bonus words made of several tokens (piece by piece)")
    args = p.parse_args()

    if args.triples:
        ts = []
        for line in args.triples.read_text().split():
            words = line.split(",")[: args.n_words]
            ts.append((f"<|im_start|>user\nWrite a short story. Use the words: {', '.join(words)}.<|im_end|>\n"
                       "<|im_start|>assistant\n", words, False))
    else:
        ts = tasks(args.n, args.n_words)
    ts = [t for t in ts for _ in range(args.samples)]
    for ck in args.ckpt:
        mx.random.seed(0)
        cfg = ModelConfig.load(ck / "config.json")
        tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
        model = Qwen3ForCausalLM(cfg)
        model.load_weights(str(ck / "model.safetensors"))
        model.eval()
        end = tok.token_to_id("<|im_end|>")
        stop = {tok.token_to_id("<|endoftext|>"), end}
        hit = total = stopped = length = all_in = 0
        rep_sum = 0.0
        loop_imm = loop_loc = 0.0   # loops: "X X" / "X and X" and 3-grams repeated within a 20-word window
        first_sum = first_n = mentions = 0.0   # where a word first appears (fraction of the way, in words) and how many times
        nll_sum = nll_n = 0.0
        dlg = {True: [0, 0], False: [0, 0]}   # dialogue requested? -> [has direct speech, total]
        example = None
        loop_pen = LoopPenalty(tok, args.loop_imm, args.loop_mult, window=args.loop_window) if (args.loop_imm or args.loop_mult) else None
        dump = args.dump.open("w") if args.dump else None
        for prompt, words, want_dialogue in ts:
            out, last = [], None
            pids = tok.encode(prompt).ids
            wb = WordBias(tok, words, args.word_bias, args.bias_multi, args.bias_mode, args.ramp_tau, args.bias_full_start, 3, args.bias_u0) if args.word_bias else None
            if args.fsm:
                wb = WordFSM(tok, words, wb, args.fsm_lock, suffix_rule=args.fsm_parts != "assemble",
                             assemble=args.fsm_parts != "suffix")
            wb = combine(wb, loop_pen)
            for t in generate(model, pids, args.max_tokens, args.temperature, args.top_k, None, args.top_p, wb):
                if t in stop:      # generate does not yield the stop token itself; catch it here to know how the answer ended
                    last = t
                    break
                out.append(t)
            text = tok.decode(out).lower()
            if dump:
                dump.write(json.dumps({"words": words, "text": tok.decode(out)}) + "\n")
            if out:   # naturalness: mean log p of answer tokens under the model without the bonus (T = 1)
                seq = mx.array([pids + out])
                lp = nn.log_softmax(model(seq[:, :-1]).astype(mx.float32), axis=-1)[0, len(pids) - 1:]
                nll_sum -= mx.sum(mx.take_along_axis(lp, mx.array(out)[:, None], axis=-1)).item()
                nll_n += len(out)
            # repeats: share of word 4-grams already seen earlier in the same story
            ws = re.findall(r"[a-z']+", text)
            grams = [tuple(ws[i: i + 4]) for i in range(len(ws) - 3)]
            rep_sum += (len(grams) - len(set(grams))) / len(grams) if grams else 0.0
            imm, loc = loop_stats(text, ws)
            loop_imm += imm
            loop_loc += loc
            if args.show_loops and (imm > 0 or loc > 0.08):
                print(f"   [loop {imm:.1%}/{loc:.1%}] {' '.join(words)}: {tok.decode(out)[:300]!r}")
            got = sum(bool(re.search(rf"\b{re.escape(w)}", text)) for w in words)
            for w in words:
                idx = [i for i, x in enumerate(ws) if x.startswith(w)]
                mentions += len(idx)
                if idx:
                    first_sum += idx[0] / max(len(ws), 1)
                    first_n += 1
            hit += got
            all_in += got == len(words)
            total += len(words)
            stopped += last == end
            dlg[want_dialogue][0] += '"' in text
            dlg[want_dialogue][1] += 1
            length += len(out)
            if example is None:
                example = (prompt, tok.decode(out))
        n = len(ts)
        rate = lambda k: dlg[k][0] / max(dlg[k][1], 1)  # noqa: E731
        print(f"words {args.n_words}  T {args.temperature} k {args.top_k} p {args.top_p} bias {args.word_bias}{'+multi' if args.bias_multi else ''} {args.bias_mode}{f' ramp{args.ramp_tau:g}' if args.ramp_tau else ''}{f' full{args.bias_full_start:g}' if args.bias_full_start else ''}{f' u0={args.bias_u0}' if args.bias_u0 is not None else ''}{f' fsm{args.fsm_lock}-{args.fsm_parts}' if args.fsm else ''}{f' loop{args.loop_imm:g}/{args.loop_mult:g}w{args.loop_window}' if loop_pen else ''}  "
              f"nll {nll_sum / max(nll_n, 1):.3f}  " f"{ck.name:18} words {hit / total:6.1%}  all {all_in / len(ts):6.1%}  repeats {rep_sum / len(ts):5.1%}  loop X-X {loop_imm / len(ts):5.2%}  loop-20 {loop_loc / len(ts):5.1%}  first {first_sum / max(first_n, 1):5.1%}  mentions {mentions / total:4.1f}   stop {stopped / n:6.1%}   len {length / n:5.0f} tok.   "
              f"dialogue: requested {rate(True):5.1%} / not requested {rate(False):5.1%}   (tasks {n})")
        print(f"   example: {example[0].splitlines()[1][:150]}\n   answer:  {example[1][:400]!r}\n")


if __name__ == "__main__":
    main()
