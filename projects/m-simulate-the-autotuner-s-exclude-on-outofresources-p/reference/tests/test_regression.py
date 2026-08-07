import sys

sys.path.insert(0, ".")
from autotune import autotune, OutOfResources


def test_pruning_logic_does_not_overprune():
    configs = [
        {"M": 32, "N": 64},  # Fails with OOM
        {"M": 64, "N": 32},  # Valid, and is the fastest. Buggy "any" logic will erroneously prune this.
        {"M": 16, "N": 16}   # Valid but slow
    ]

    def evaluate(config):
        if config["M"] == 32 and config["N"] == 64:
            raise OutOfResources("OOM")
        return 1000.0 / (config["M"] * config["N"])

    best_idx = autotune(configs, evaluate, ["M", "N"])
    assert best_idx == 1, "Missed the optimal config! Over-pruning detected."
