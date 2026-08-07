import ref


def check(workdir):
    from chkpt import profile

    out = {"memory_measured": 0.0, "time_measured": 0.0}
    try:
        mem_no, time_no = profile.measure(n_layers=10, checkpoint=False)
        mem_yes, time_yes = profile.measure(n_layers=10, checkpoint=True)
        ref_mem_no, ref_time_no = ref.simulate_profile(10, 0)
        ref_mem_yes, ref_time_yes = ref.simulate_profile(10, 2)

        if isinstance(mem_no, (int, float)) and isinstance(time_no, (int, float)):
            out["memory_measured"] = 1.0
        if isinstance(mem_yes, (int, float)) and isinstance(time_yes, (int, float)):
            out["time_measured"] = 1.0
    except Exception as e:
        out["_note"] = f"measure raised {type(e).__name__}: {str(e)[:120]}"
    return out
