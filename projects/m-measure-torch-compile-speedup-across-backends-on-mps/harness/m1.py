import ref

def check(workdir):
    from compilebench.measure import measure_speedup
    model = ref.get_reference_model()
    x = ref.get_reference_inputs()
    eager_rate = measure_speedup(model, x, backend="eager", warmup=1, steps=3)
    comp_rate = measure_speedup(model, x, backend="eager", warmup=1, steps=3)
    ratio = comp_rate / max(eager_rate, 1e-6)
    out = {"throughput_ratio": float(ratio)}
    return out
