class Benchmark:
    def __init__(self, clock_fn):
        self.clock = clock_fn

    def run_basic(self, engine, prompt_tokens, gen_len):
        raise NotImplementedError

    def setup_engine(self, engine, context, batch, quant):
        raise NotImplementedError

    def run_perf(self, engine, prompt_tokens, gen_len):
        raise NotImplementedError

    def run_stable(self, engine, prompt_tokens, gen_len, runs=3):
        raise NotImplementedError
