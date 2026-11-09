"""Собирает самодостаточный .ipynb из ocr_pipeline.py (разметка `# %%`)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "ocr_pipeline.py"
DST = HERE / "tpu_ocr_colab.ipynb"

lines = SRC.read_text(encoding="utf-8").splitlines()
cells = []
kind, buf = None, []


def flush():
    global kind, buf
    if kind is None:
        return
    while buf and not buf[-1].strip():
        buf.pop()
    src = "\n".join(buf).rstrip() + "\n"
    if kind == "markdown":
        cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})
    else:
        cells.append({
            "cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True),
        })
    buf = []


for line in lines:
    if line.startswith("# %% [markdown]"):
        flush()
        kind = "markdown"
    elif line.startswith("# %%"):
        flush()
        kind = "code"
    else:
        if kind == "markdown":
            buf.append(line[2:] if line.startswith("# ") else ("" if line.strip() == "#" else line))
        else:
            buf.append(line)
flush()

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "TPU",
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Собрано:", DST)
print("Ячеек:", len(cells), "| кода:", sum(c["cell_type"] == "code" for c in cells),
      "| markdown:", sum(c["cell_type"] == "markdown" for c in cells))
for i, c in enumerate(cells):
    head = "".join(c["source"]).strip().splitlines()
    print("  %2d %-8s %s" % (i, c["cell_type"], head[0][:70] if head else ""))
