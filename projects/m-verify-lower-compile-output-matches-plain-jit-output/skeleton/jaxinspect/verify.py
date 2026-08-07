import numpy as np


def verify_compile_vs_jit(aot_outputs: list[dict], jit_outputs: list[dict]) -> dict:
    """Verify numeric equivalence between lower().compile() and jit() recorded outputs."""
    raise NotImplementedError
