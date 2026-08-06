import json
import os

import numpy as np

FIX = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "_fixtures", "gguf"))


def clean_blob():
    with open(os.path.join(FIX, "slice.gguf"), "rb") as f:
        return f.read()


def corrupt_blob():
    with open(os.path.join(FIX, "slice_corrupt.gguf"), "rb") as f:
        return f.read()


def truth():
    with open(os.path.join(FIX, "container_truth.json"), encoding="utf-8") as f:
        return json.load(f)


def corruption():
    with open(os.path.join(FIX, "corruption_truth.json"), encoding="utf-8") as f:
        return json.load(f)


def weights():
    return np.load(os.path.join(FIX, "dequantized_truth.npz"))


def close(a, b, tol=1e-6):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return False, float("inf")
    if a.size == 0:
        return False, float("inf")
    err = float(np.abs(a - b).max())
    scale = float(np.abs(b).max()) or 1.0
    return err <= tol * scale, err / scale
