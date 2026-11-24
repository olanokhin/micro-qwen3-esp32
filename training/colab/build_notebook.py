"""Builds train_micro_qwen3.ipynb from train_micro_qwen3.py (cells are separated by "# %%" lines).

    python training/colab/build_notebook.py

The .py file is the source: it also runs as a plain script, which is how it is tested.
"""

import json
from pathlib import Path

here = Path(__file__).parent
cells, kind, lines = [], None, []


def flush():
    if kind is None:
        return
    text = "\n".join(lines).strip("\n")
    if kind == "markdown":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n"))
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})


for line in (here / "train_micro_qwen3.py").read_text().split("\n"):
    if line.startswith("# %%"):
        flush()
        kind, lines = ("markdown" if "[markdown]" in line else "code"), []
    else:
        lines.append(line)
flush()

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}}
(here / "train_micro_qwen3.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"{len(cells)} cells")
