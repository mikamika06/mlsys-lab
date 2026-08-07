import sys


class DummyDraftModel:
    def __init__(self, step_latency=0.002):
        self.step_latency = step_latency

    def generate_draft(self, inputs, draft_steps):
        dur = self.step_latency * draft_steps
        tokens = [inputs[-1] + i + 1 for i in range(draft_steps)]
        return tokens, dur


class DummyTargetModel:
    def __init__(self, verify_latency=0.005, accept_rate=0.8, per_token_latency=0.012):
        self.verify_latency = verify_latency
        self.accept_rate = accept_rate
        self.per_token_latency = per_token_latency

    def verify_and_generate(self, inputs, draft_tokens):
        num_accepted = max(1, int(len(draft_tokens) * self.accept_rate))
        return draft_tokens[:num_accepted], self.verify_latency

    def generate_baseline(self, inputs, total_tokens):
        dur = self.per_token_latency * total_tokens
        tokens = [inputs[-1] + i + 1 for i in range(total_tokens)]
        return tokens, dur


CONFIGS = [
    {
        "stage1_model": DummyDraftModel(0.001),
        "stage2_model": DummyTargetModel(0.003, 0.8, 0.010),
        "target_model": DummyTargetModel(0.003, 0.8, 0.010),
        "inputs": [10, 20],
        "draft_steps": 3
    },
    {
        "stage1_model": DummyDraftModel(0.002),
        "stage2_model": DummyTargetModel(0.004, 0.5, 0.015),
        "target_model": DummyTargetModel(0.004, 0.5, 0.015),
        "inputs": [10, 20],
        "draft_steps": 4
    },
    {
        "stage1_model": DummyDraftModel(0.001),
        "stage2_model": DummyTargetModel(0.002, 1.0, 0.012),
        "target_model": DummyTargetModel(0.002, 1.0, 0.012),
        "inputs": [5, 15],
        "draft_steps": 5
    },
    {
        "stage1_model": DummyDraftModel(0.0015),
        "stage2_model": DummyTargetModel(0.0035, 0.75, 0.014),
        "target_model": DummyTargetModel(0.0035, 0.75, 0.014),
        "inputs": [1],
        "draft_steps": 2
    }
]


def build_reference_results():
    sys.path.insert(0, "reference")
    from cascade.benchmark import run_cascade_benchmark
    results = []
    for cfg in CONFIGS:
        results.append(run_cascade_benchmark(cfg))
    return results
