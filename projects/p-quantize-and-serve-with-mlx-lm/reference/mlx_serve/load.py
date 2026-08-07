import ref

def benchmark_load(concurrency: int, latencies: list) -> dict:
    return ref.run_load_test(concurrency, latencies)
