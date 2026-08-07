class MemoryAdvisor:
    def __init__(self, predictor=None):
        raise NotImplementedError

    def analyze(self, config: dict) -> dict:
        raise NotImplementedError

    def suggest_fixes(self, config: dict) -> list:
        raise NotImplementedError
