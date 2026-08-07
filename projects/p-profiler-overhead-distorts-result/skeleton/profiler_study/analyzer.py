class ProfilerAnalyzer:
    def __init__(self, env):
        raise NotImplementedError

    def compare_modes(self, mode_a: str, mode_b: str) -> dict:
        raise NotImplementedError

    def select_mode(self, question_type: str) -> str:
        raise NotImplementedError

    def verify_invariant(self, conclusion_func, modes: list) -> bool:
        raise NotImplementedError

    def check_discrepancy(self, mode: str, threshold: float) -> bool:
        raise NotImplementedError
