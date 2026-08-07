import sys

sys.path.insert(0, ".")
from cascade.benchmark import compute_latency_ratio, run_cascade_benchmark


class MockStage1:
    def __init__(self, step_cost=0.002):
        self.step_cost = step_cost

    def generate_draft(self, inputs, draft_steps):
        return list(range(draft_steps)), self.step_cost * draft_steps


class MockStage2:
    def __init__(self, verify_cost=0.005, accept_rate=0.8):
        self.verify_cost = verify_cost
        self.accept_rate = accept_rate

    def verify_and_generate(self, inputs, draft_tokens):
        accepted_n = max(1, int(len(draft_tokens) * self.accept_rate))
        return draft_tokens[:accepted_n], self.verify_cost


class MockTarget:
    def __init__(self, token_cost=0.01):
        self.token_cost = token_cost

    def generate_baseline(self, inputs, total_tokens):
        cost = self.token_cost * total_tokens
        return list(range(total_tokens)), cost


def test_cascade_latency_monotonicity():
    cfg1 = {
        "stage1_model": MockStage1(step_cost=0.001),
        "stage2_model": MockStage2(verify_cost=0.004, accept_rate=1.0),
        "target_model": MockTarget(token_cost=0.008),
        "inputs": [1, 2, 3],
        "draft_steps": 2
    }
    cfg2 = {
        "stage1_model": MockStage1(step_cost=0.001),
        "stage2_model": MockStage2(verify_cost=0.004, accept_rate=1.0),
        "target_model": MockTarget(token_cost=0.008),
        "inputs": [1, 2, 3],
        "draft_steps": 5
    }
    res1 = run_cascade_benchmark(cfg1)
    res2 = run_cascade_benchmark(cfg2)
    assert res2["cascade_latency"] > res1["cascade_latency"]


def test_cascade_ratio_bounds():
    ratio = compute_latency_ratio(0.04, 0.10)
    assert 0.0 < ratio < 1.0
