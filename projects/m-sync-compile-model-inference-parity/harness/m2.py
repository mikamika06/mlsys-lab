import ref


def check(workdir):
    from ovruntime.core import Core
    from ovruntime.infer import benchmark_pipeline

    out = {"sync_parity": 0.0, "async_parity": 0.0, "latency_ratio": 1.0}

    try:
        core = Core()
        cfg = ref.TEST_CONFIGS[0]
        compiled = core.compile_model(cfg)

        sync_outs, sync_ticks = benchmark_pipeline(
            compiled, ref.TEST_INPUTS, mode="sync"
        )
        async_outs, async_ticks = benchmark_pipeline(
            compiled, ref.TEST_INPUTS, mode="async"
        )

        sync_ok = True
        async_ok = True

        for inp, s_out, a_out in zip(ref.TEST_INPUTS, sync_outs, async_outs):
            want = ref.reference_infer(cfg, inp)
            if ref.compute_rel_err(s_out, want) > 1e-5:
                sync_ok = False
            if ref.compute_rel_err(a_out, want) > 1e-5:
                async_ok = False

        out["sync_parity"] = 1.0 if sync_ok else 0.0
        out["async_parity"] = 1.0 if async_ok else 0.0

        if sync_ticks > 0:
            out["latency_ratio"] = float(async_ticks) / float(sync_ticks)
        else:
            out["latency_ratio"] = 1.0
    except Exception as e:
        out["_note"] = f"m2 check failed: {type(e).__name__}: {str(e)[:120]}"

    return out
