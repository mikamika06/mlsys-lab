class ProfilerAnalyzer:
    def __init__(self, env):
        self.env = env

    def compare_modes(self, mode_a: str, mode_b: str) -> dict:
        cost_a = self.env.measure(mode_a)
        cost_b = self.env.measure(mode_b)
        return {"cost_a": cost_a, "cost_b": cost_b, "diff": abs(cost_a - cost_b)}

    def select_mode(self, question_type: str) -> str:
        if question_type == "micro":
            return "instrumentation"
        return "sampling"

    def verify_invariant(self, conclusion_func, modes: list) -> bool:
        base_val = conclusion_func(self.env.measure(modes[0]))
        for m in modes[1:]:
            if conclusion_func(self.env.measure(m)) != base_val:
                return False
        return True

    def check_discrepancy(self, mode: str, threshold: float) -> bool:
        clean_val = self.env.measure("clean")
        mode_val = self.env.measure(mode)
        return abs(mode_val - clean_val) <= threshold
