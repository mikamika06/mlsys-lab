"""Benchmark runner for simulated vLLM serving engine."""


def simulate_request(prompt_tokens, max_tokens, rate):
    raise NotImplementedError


def run_bench_serve(requests, rate, num_workers=1):
    raise NotImplementedError


def run_multi_rate_bench(requests, rates):
    raise NotImplementedError
