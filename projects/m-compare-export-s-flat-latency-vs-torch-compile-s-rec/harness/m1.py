import harness.ref as ref


def check(workdir):
    import exportbench.runner as runner

    out = {"latency_ratio": 0.0, "export_flatness": 1.0}

    batch_seq = ref.generate_batch_sequence(seed=42, num_requests=30)
    model = runner.SimulatedModule(hidden_dim=128, static_compile=True)

    try:
        res = runner.benchmark_runtimes(model, batch_seq)
    except Exception as e:
        out["_note"] = f"benchmark_runtimes raised Exception: {type(e).__name__}: {str(e)[:120]}"
        return out

    max_c = res.get("max_compile_spike", 0.0)
    max_e = res.get("max_export_spike", 1e-6)
    export_std = res.get("export_std", 1.0)

    if max_e > 0:
        out["latency_ratio"] = float(max_c / max_e)
    out["export_flatness"] = float(export_std)

    return out
