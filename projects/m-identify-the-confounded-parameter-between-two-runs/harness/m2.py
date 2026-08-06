import ref


def check(workdir):
    from benchopt.tuning import tune_parameters

    out = {"throughput_beaten": 0.0}
    try:
        optimized = tune_parameters(ref.DEFAULT_CONFIG)
        if optimized.get("pp_throughput", 0.0) > ref.DEFAULT_CONFIG["pp_throughput"]:
            out["throughput_beaten"] = 1.0
        else:
            out["_note"] = f"Optimized throughput {optimized.get('pp_throughput')} did not beat default {ref.DEFAULT_CONFIG['pp_throughput']}"
    except Exception as e:
        out["_note"] = f"Error during tuning: {type(e).__name__}: {str(e)[:120]}"
    return out
