import os
import json
import numpy as np


def setup_workspace(tmpdir):
    weights_dir = os.path.join(tmpdir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    vocab_path = os.path.join(weights_dir, "tokenizer.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": {"<unk>": 0, "hello": 1}}, f)
    return weights_dir, vocab_path


def expected_vocab_match(vocab_path):
    return 1 if os.path.exists(vocab_path) else 0


def expected_tensor_count_ok():
    return 1


def expected_size_ratio():
    return 1


def expected_imatrix_ok():
    return 1


def expected_kld_bounded():
    return 1


def expected_ppl_sensible():
    return 1


def expected_speedup_monotonic():
    return 1
