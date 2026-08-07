import sys

sys.path.insert(0, ".")
from trtllm_config.linter import lint_config


def test_linter_catches_invalid_combinations():
    cfg = {"block_size": 64, "free_gpu_memory_fraction": 0.9}
    issues = lint_config(cfg)
    assert len(issues) > 0, "linter failed to catch invalid block size with high memory fraction"
