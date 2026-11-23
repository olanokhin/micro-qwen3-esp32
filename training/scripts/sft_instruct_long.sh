#!/bin/bash
# Instruction tuning of ts2-base (the demo model): 900K TinyStoriesInstruct dialogues in the Qwen chat format,
# 600 MB of text, loss on the answer only, lr 1e-3, quantization-aware training for W4R from the start.
set -e
cd "$(dirname "$0")/../.."
[ -f data/tsim900-v4096-qwen/train.mask.bin ] || uv run training/scripts/prepare_data.py --dataset tinystories-instruct \
    --num-docs 900000 --tokenizer data/ts2-v4096-qwen/tokenizer.json --out data/tsim900-v4096-qwen
uv run training/scripts/train.py --recipe runs/ts2-base/recipe.json --data data/tsim900-v4096-qwen --init-from runs/ts2-base \
    --train-bytes 6e8 --lr 1e-3 --warmup 100 --qat int4row-clip --eval-every 1000 --eval-batches 40 \
    --out runs/tsim900-sft-x > runs/tsim900-sft-x.log 2>&1
uv run training/scripts/eval_instruct.py --ckpt runs/tsim900-sft-x --n 200
