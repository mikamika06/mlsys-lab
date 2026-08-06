SCENARIOS = [
    {"base_tokens": 100000, "target_length": 8192, "base_length": 2048, "alpha": 1.5},
    {"base_tokens": 500000, "target_length": 16384, "base_length": 4096, "alpha": 1.2},
    {"base_tokens": 200000, "target_length": 32768, "base_length": 8192, "alpha": 1.8},
]

ROPE_SCENARIOS = [
    {"stages": 3, "initial_base": 10000.0, "target_base": 100000.0},
    {"stages": 5, "initial_base": 50000.0, "target_base": 500000.0},
    {"stages": 4, "initial_base": 10000.0, "target_base": 1000000.0},
]

EVAL_SCENARIOS = [
    {"abf_ppl": 4.5, "yarn_ppl": 4.8},
    {"abf_ppl": 6.1, "yarn_ppl": 5.9},
    {"abf_ppl": 3.2, "yarn_ppl": 3.2},
]


def estimate_cost(base_tokens: int, target_length: int, base_length: int, alpha: float) -> float:
    ratio = float(target_length) / float(base_length)
    return float(base_tokens) * (ratio ** alpha)


def simulate_rope_schedule(stages: int, initial_base: float, target_base: float) -> list[float]:
    if stages <= 1:
        return [float(target_base)]
    bases = []
    for i in range(stages):
        factor = i / (stages - 1)
        val = initial_base * ((target_base / initial_base) ** factor)
        bases.append(float(val))
    return bases


def compare_strategies(abf_ppl: float, yarn_ppl: float) -> dict:
    diff = float(yarn_ppl) - float(abf_ppl)
    better = "abf" if abf_ppl < yarn_ppl else "yarn" if yarn_ppl < abf_ppl else "tie"
    return {"perplexity_difference": diff, "preferred": better}
