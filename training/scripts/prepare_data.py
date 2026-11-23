"""Downloads a dataset, trains a small byte-level BPE tokenizer and tokenizes the corpus.

Why our own tokenizer: the Qwen3 vocabulary has 151,936 tokens. Even with hidden=64, one
embedding matrix would take 9.7M parameters, 16 times the whole budget.
So we use the same kind of tokenizer (byte-level BPE), but with a 4096-token vocabulary.

Example:
    uv run training/scripts/prepare_data.py --dataset fineweb-edu --num-docs 200000
    uv run training/scripts/prepare_data.py --dataset tinystories --num-docs 500000
    uv run training/scripts/prepare_data.py --dataset tinystories-v2 --num-docs 800000 --out data/ts2-v4096-qwen
    uv run training/scripts/prepare_data.py --dataset tinystories-instruct --num-docs 300000 \
        --tokenizer data/ts2-v4096-qwen/tokenizer.json --out data/tsi-v4096-qwen
"""

import argparse
import codecs
import json
import re
import urllib.request
from itertools import islice
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tqdm import tqdm

DATASETS = {
    # (hf repo, config name, text field)
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "tinystories": ("roneneldan/TinyStories", None, "text"),
    # V2: GPT-4 stories only, one text file, stories separated by <|endoftext|>
    "tinystories-v2": ("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt", None),
    # instructions: fields Features / Words / Summary / Random sentence / Story -> Qwen chat format (see to_chat)
    "tinystories-instruct": ("roneneldan/TinyStoriesInstruct", "TinyStories-Instruct-train.txt", None),
    # the same, but one task: a story from a set of words and fixed options (no summary, plot, sentence)
    "tinystories-instruct-wf": ("roneneldan/TinyStoriesInstruct", "TinyStories-Instruct-train.txt", None),
}

# Qwen-style special tokens: the chat format is built in from the start for the SFT stage.
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


# Pre-tokenization rule from the real Qwen3 tokenizer.json (as in GPT-4/cl100k):
# - \p{N}: every digit separately: "1900" -> "1","9","0","0", no random "190" tokens
# - (?i:'s|'t|...): contractions regardless of case
# - [^\r\n\p{L}\p{N}]?\p{L}+: a word together with one preceding character (usually a space)
# - ?[^\s\p{L}\p{N}]+[\r\n]*: punctuation in groups: "):\n", "});" and so on
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
              r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def stream_txt(repo: str, filename: str, sep: str = "<|endoftext|>"):
    """Documents from a large .txt on Hugging Face, streamed: only the part of the file read so far is downloaded."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    buf, dec = "", codecs.getincrementaldecoder("utf-8")()   # a character split across blocks is not lost
    with urllib.request.urlopen(url) as r:
        while chunk := r.read(1 << 20):
            buf += dec.decode(chunk)
            *docs, buf = buf.split(sep)
            for d in docs:
                yield d.strip()
    if buf.strip():
        yield buf.strip()


FIELD_RE = re.compile(r"^(Features|Words|Summary|Random sentence|Story):[ \t]*", re.M)
FEATURES = {"Dialogue": "dialogue", "BadEnding": "a bad ending", "MoralValue": "a moral",
            "Twist": "a plot twist", "Foreshadowing": "foreshadowing", "Conflict": "a conflict"}


def to_chat(doc: str, words_features_only: bool = False) -> str | None:
    """A TinyStoriesInstruct record -> a dialogue in Qwen format (<|im_start|>role ... <|im_end|>).

    Two tasks, as in the dataset itself: if Summary comes after the story, "summarize"; otherwise
    "write a story" with conditions (words, features, summary, a required sentence).
    """
    parts = FIELD_RE.split(doc)
    head, fields, order = parts[0].strip(), {}, []
    for name, val in zip(parts[1::2], parts[2::2]):
        fields[name], order = val.strip(), order + [name]
    story = fields.get("Story") or head          # without a Story label, the story is the text before the first field
    if not story:
        return None
    story_pos = order.index("Story") if "Story" in order else -1
    if not words_features_only and "Summary" in fields and order.index("Summary") > story_pos:   # summary after the story
        user, answer = f"Summarize this story in one sentence.\n\n{story}", fields["Summary"]
    else:
        req = []
        if "Words" in fields:
            req.append(f"Use the words: {fields['Words']}.")
        if "Features" in fields:
            feats = [FEATURES.get(f.strip(), f.strip().lower()) for f in fields["Features"].split(",") if f.strip()]
            req.append("The story should have " + ", ".join(feats) + ".")
        if "Summary" in fields and not words_features_only:
            req.append(f"Plot: {fields['Summary']}")
        if "Random sentence" in fields and not words_features_only:
            req.append(f"Include the sentence: \"{fields['Random sentence']}\"")
        user, answer = " ".join(["Write a short story."] + req), story
    if not answer:
        return None
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"


def train_tokenizer(texts: list[str], vocab_size: int, pretokenizer: str = "gpt2") -> Tokenizer:
    tok = Tokenizer(models.BPE())
    if pretokenizer == "qwen":
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:  # gpt2: the built-in ByteLevel rule
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer, length=len(texts))
    return tok


def encode(tok: Tokenizer, texts: list[str], eot: int, batch: int = 2000) -> np.ndarray:
    chunks = []
    for i in tqdm(range(0, len(texts), batch), desc="tokenize"):
        for enc in tok.encode_batch(texts[i : i + batch]):
            chunks.append(np.asarray(enc.ids + [eot], dtype=np.uint16))
    return np.concatenate(chunks)


ASSISTANT = "<|im_start|>assistant\n"


def encode_chat(tok: Tokenizer, texts: list[str], eot: int, batch: int = 2000):
    """A dialogue is encoded in two pieces, the question (up to and including "assistant\\n") and the answer, the same
    way as on the device. Mask 1 = answer tokens (with <|im_end|>), 0 = question and <|endoftext|>. Also returns the
    number of answer bytes (for bits-per-byte over the answer)."""
    ids, masks, nbytes = [], [], 0
    for i in tqdm(range(0, len(texts), batch), desc="tokenize"):
        part = texts[i: i + batch]
        cut = [t.index(ASSISTANT) + len(ASSISTANT) for t in part]
        qs = tok.encode_batch([t[:c] for t, c in zip(part, cut)])
        ans = tok.encode_batch([t[c:] for t, c in zip(part, cut)])
        for t, c, q, a in zip(part, cut, qs, ans):
            ids.append(np.asarray(q.ids + a.ids + [eot], dtype=np.uint16))
            masks.append(np.asarray([0] * len(q.ids) + [1] * len(a.ids) + [0], dtype=np.uint8))
            nbytes += len(t[c:].replace("<|im_end|>", "").encode("utf-8")) + 1   # +1 for <|im_end|>
    return np.concatenate(ids), np.concatenate(masks), nbytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default="fineweb-edu")
    p.add_argument("--num-docs", type=int, default=200_000)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--tokenizer-docs", type=int, default=50_000, help="how many documents to train BPE on")
    p.add_argument("--val-docs", type=int, default=2_000)
    p.add_argument("--pretokenizer", choices=["gpt2", "qwen"], default="qwen",
                   help="pre-tokenization rule applied before BPE")
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="an existing tokenizer.json instead of training a new one (e.g. data/fw-v4096-qwen/tokenizer.json): "
                        "so that different datasets/sizes share the same tokens")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    assert args.vocab_size <= 65536, "tokens are stored as uint16"

    out = args.out or Path("data") / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    repo, name, field = DATASETS[args.dataset]
    # Raw text cache: to train tokenizers of different sizes on the same documents
    # without downloading again.
    raw_cache = Path("data") / "raw" / f"{args.dataset}_{args.num_docs}.jsonl"
    if raw_cache.exists():
        with raw_cache.open() as f:
            texts = [json.loads(line) for line in f]
        print(f"raw text from cache: {raw_cache}")
    else:
        if field is None:   # one large .txt
            ds = stream_txt(repo, name)
        else:
            ds = (ex[field] for ex in load_dataset(repo, name, split="train", streaming=True))
        if args.dataset.startswith("tinystories-instruct"):
            wf = args.dataset.endswith("-wf")
            ds = (c for c in (to_chat(d, wf) for d in ds) if c)
        texts = list(tqdm(islice(ds, args.num_docs), total=args.num_docs, desc="download"))
        texts = [t for t in texts if t and t.strip()]
        raw_cache.parent.mkdir(parents=True, exist_ok=True)
        with raw_cache.open("w") as f:
            f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in texts)
    print(f"documents: {len(texts):,}, characters: {sum(map(len, texts)):,}")

    val_texts, train_texts = texts[: args.val_docs], texts[args.val_docs :]

    if args.tokenizer:
        tok = Tokenizer.from_file(str(args.tokenizer))
        src_meta = args.tokenizer.parent / "meta.json"  # take the pre-tokenization from the tokenizer's source
        if src_meta.exists():
            args.pretokenizer = json.loads(src_meta.read_text()).get("pretokenizer", "gpt2")
        print(f"tokenizer: {args.tokenizer} (vocab {tok.get_vocab_size()}, pre-tokenizer {args.pretokenizer})")
    else:
        tok = train_tokenizer(train_texts[: args.tokenizer_docs], args.vocab_size, args.pretokenizer)
    tok.save(str(out / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")

    chat = args.dataset.startswith("tinystories-instruct")
    mask_bytes = {}
    for split, split_texts in (("train", train_texts), ("val", val_texts)):
        if chat:   # dialogues: next to the tokens, an "this is the answer" mask (we train only on it and on <|im_end|>)
            ids, mask, mask_bytes[split] = encode_chat(tok, split_texts, eot)
            mask.tofile(out / f"{split}.mask.bin")
        else:
            ids = encode(tok, split_texts, eot)
        ids.tofile(out / f"{split}.bin")
        print(f"{split}: {len(ids):,} tokens" + (f", of them answer: {int(mask.sum()):,}" if chat else ""))

    n_chars = sum(map(len, train_texts))
    n_train = (out / "train.bin").stat().st_size // 2
    n_val = (out / "val.bin").stat().st_size // 2
    # UTF-8 bytes (+1 for the document separator, which we encode as the <|endoftext|> token).
    # Needed for bits-per-byte, a metric that does not depend on the tokenizer.
    val_bytes = sum(len(t.encode("utf-8")) + 1 for t in val_texts)
    train_bytes = sum(len(t.encode("utf-8")) + 1 for t in train_texts)
    if chat:   # bits-per-byte over the answer: answer bytes / answer tokens
        for split in ("train", "val"):
            n_ans = int(np.fromfile(out / f"{split}.mask.bin", dtype=np.uint8).sum())
            mask_bytes[split] = mask_bytes[split] / n_ans
    meta = {
        "dataset": args.dataset,
        "vocab_size": tok.get_vocab_size(),
        "pretokenizer": args.pretokenizer,
        "tokenizer_from": str(args.tokenizer) if args.tokenizer else None,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "chars_per_token": round(n_chars / n_train, 3),
        "val_bytes_per_token": val_bytes / n_val,
        "train_bytes_per_token": train_bytes / n_train,
        **({"val_answer_bytes_per_token": mask_bytes["val"], "train_answer_bytes_per_token": mask_bytes["train"]}
           if chat else {}),
        "special_tokens": {t: tok.token_to_id(t) for t in SPECIAL_TOKENS},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
