def benchmark_compiled_step(model, inputs, warmup=2, iters=5):
    raise NotImplementedError

def robust_summary(latencies):
    raise NotImplementedError

def paired_bootstrap_ci(a, b, n_boot=500, alpha=0.05, seed=42):
    raise NotImplementedError
