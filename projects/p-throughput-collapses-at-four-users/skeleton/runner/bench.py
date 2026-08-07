from runner.engine import Engine, Request, RequestMetrics

class LoadBench:
    def __init__(self, warmup_runs: int = 2):
        raise NotImplementedError

    def generate_workload(self, num_users: int, prompt_len: int = 32, output_len: int = 100) -> list[Request]:
        raise NotImplementedError

    def run_benchmark(self, engine: Engine, workload: list[Request]) -> dict:
        raise NotImplementedError
