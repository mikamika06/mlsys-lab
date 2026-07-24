"""Deterministic fixture for rws-hit-a-param-budget-after-coupled-slicing.

A small LLaMA-style transformer stack (12 layers + tied-free embedding/head +
final norm) described generically: every coupled tensor's shape at hidden
width `d` is `(coef0*d + const0, coef1*d + const1)`. Slicing the model to a
new width `d` scales every one of these tensors simultaneously ("coupled
slicing").

Run with:
    python3 tasks/rws-hit-a-param-budget-after-coupled-slicing/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

VOCAB = 32000
N_LAYERS = 12
BUDGET = 600_000_000


def _layer_templates():
    # (coef0, const0, coef1, const1): shape = (coef0*d+const0, coef1*d+const1)
    return [
        (3, 0, 1, 0),   # combined QKV proj: (3d, d)
        (1, 0, 1, 0),   # attention output proj: (d, d)
        (8, 0, 1, 0),   # SwiGLU combined gate+up proj: (8d, d)
        (1, 0, 4, 0),   # MLP down proj: (d, 4d)
        (1, 0, 0, 1),   # input layernorm weight: (d,) -> (d, 1)
        (1, 0, 0, 1),   # post-attention layernorm weight: (d,)
    ]


def build():
    templates = []
    for _ in range(N_LAYERS):
        templates.extend(_layer_templates())
    templates.append((0, VOCAB, 1, 0))  # token embedding: (vocab, d)
    templates.append((0, VOCAB, 1, 0))  # untied lm_head: (vocab, d)
    templates.append((1, 0, 0, 1))      # final norm weight: (d,)

    templates = np.array(templates, dtype=np.int64)
    coefs = templates[:, [0, 2]]
    consts = templates[:, [1, 3]]
    widths = np.arange(256, 4096 + 1, 64, dtype=np.int64)

    return coefs, consts, widths, np.int64(BUDGET)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    coefs, consts, widths, budget = build()
    np.save(OUT / "cs_coefs.npy", coefs)
    np.save(OUT / "cs_consts.npy", consts)
    np.save(OUT / "cs_widths.npy", widths)
    np.save(OUT / "cs_budget.npy", np.array(budget, dtype=np.int64))
    print("wrote coefs", coefs.shape, "consts", consts.shape, "widths", widths.shape, "budget", budget)
