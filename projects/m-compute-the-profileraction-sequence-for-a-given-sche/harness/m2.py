import tempfile
import ref

def check(workdir):
    from profiler_util.instrument import run_training_loop

    out = {"trace_count_match": 0.0}
    cfg = ref.CONFIGS[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            count = run_training_loop(
                cfg["total_steps"],
                {
                    "wait": cfg["wait"],
                    "warmup": cfg["warmup"],
                    "active": cfg["active"],
                    "repeat": cfg["repeat"]
                },
                tmpdir
            )
            want_actions = ref.compute_reference_actions(cfg)
            from torch.profiler import ProfilerAction
            want_count = sum(1 for a in want_actions if a == ProfilerAction.RECORD_AND_SAVE)
            if count == want_count:
                out["trace_count_match"] = 1.0
            else:
                out["_note"] = f"got trace count {count}, expected {want_count}"
        except Exception as e:
            out["_note"] = f"instrumentation raised exception: {type(e).__name__}: {str(e)}"
    return out
