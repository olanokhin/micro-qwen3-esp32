"""Check the C inference (host/microqwen) against the MLX reference written by export_esp.py.

    python parity/check_c.py --bin models/tsim900-sft-x-w4r.bin --tokenizer models/tsim900-sft-x.tokenizer.json --exe host/microqwen

1. tokenizer: C encode against HF tokenizers on a set of strings;
2. logits of the last prompt position: max |difference| and whether argmax matches;
3. greedy generation: how many leading tokens match.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TEXTS = [
    "Once upon a time, there was a little dog named Max.",
    "Lily said, \"I'm happy!\" She didn't want to go.",
    "In 1999, Tom had 3 apples and 42 pears; he ate 7.",
    "Hello   world\n\nNew paragraph\twith tab.",
    "It's, they're, we've, I'll, he'd -- (test) [brackets] {braces}!",
    "  leading spaces and trailing   ",
    "URL: https://example.com/path?a=1&b=2 #tag @user",
    # Qwen chat format: special tokens are recognised whole, the text between them goes through normal BPE
    "<|im_start|>user\nWrite a short story. Use the words: dog, ball, rain. The story should have dialogue."
    "<|im_end|>\n<|im_start|>assistant\n",
    "a<|endoftext|>b <|im_end|><|im_start|> <| not special |>",
]


def run(exe, *args) -> str:
    return subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True, check=True).stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--exe", type=Path, default=Path("host/microqwen"))
    args = p.parse_args()
    ref = json.loads(Path(str(args.bin) + ".ref.json").read_text())
    tok = Tokenizer.from_file(str(args.tokenizer))

    print("1. tokenizer")
    ok = 0
    for t in TEXTS:
        c = [int(x) for x in run(args.exe, args.bin, "encode", t).strip().split(",") if x]
        h = tok.encode(t).ids
        ok += c == h
        if c != h:
            print(f"   ✗ {t!r}\n     C : {c}\n     HF: {h}")
    print(f"   matched {ok}/{len(TEXTS)}")

    print("2. logits of the last position")
    ids = ",".join(map(str, ref["prompt_ids"]))
    c = np.array(run(args.exe, args.bin, "logits", ids).split(), dtype=np.float32)
    r = np.array(ref["last_logits"], dtype=np.float32)
    print(f"   max |C − MLX| = {np.abs(c - r).max():.5f} (logit scale ~{np.abs(r).max():.1f}); "
          f"argmax: C {c.argmax()} / MLX {r.argmax()}; top-5 matches: {set(c.argsort()[-5:]) == set(r.argsort()[-5:])}")

    print("3. greedy generation")
    g = [int(x) for x in run(args.exe, args.bin, "greedy", ids, len(ref["greedy_ids"])).strip().split(",") if x]
    same = next((i for i, (a, b) in enumerate(zip(g, ref["greedy_ids"])) if a != b), min(len(g), len(ref["greedy_ids"])))
    print(f"   first {same} of {len(ref['greedy_ids'])} tokens match")
    print(f"   C  : {tok.decode(g)[:160]!r}")


if __name__ == "__main__":
    main()
