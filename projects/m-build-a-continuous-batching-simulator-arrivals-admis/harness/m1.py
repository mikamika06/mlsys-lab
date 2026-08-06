import ref

def check(workdir):
    out = {"static_err": 1.0, "continuous_err": 1.0, "ratio_err": 1.0}
    
    try:
        import cbsim.engine as eng
    except ImportError:
        out["_note"] = "Could not import cbsim.engine"
        return out
        
    trace = ref.generate_trace(60, seed=123)
    
    try:
        r_stat_ticks, _ = ref.simulate_static(trace, 4)
        l_stat_ticks, _ = eng.simulate_static(trace, 4)
        if r_stat_ticks > 0:
            out["static_err"] = abs(r_stat_ticks - l_stat_ticks) / r_stat_ticks
    except Exception as e:
        out["_note"] = f"simulate_static failed: {e}"
        
    try:
        r_cont_ticks, _ = ref.simulate_continuous(trace, 4)
        l_cont_ticks, _ = eng.simulate_continuous(trace, 4)
        if r_cont_ticks > 0:
            out["continuous_err"] = abs(r_cont_ticks - l_cont_ticks) / r_cont_ticks
    except Exception as e:
        out["_note"] = out.get("_note", "") + f" | simulate_continuous failed: {e}"
        
    try:
        r_ratio = ref.compare_throughput(trace, 4)
        l_ratio = eng.compare_throughput(trace, 4)
        if r_ratio > 0:
            out["ratio_err"] = abs(r_ratio - l_ratio) / r_ratio
    except Exception as e:
        out["_note"] = out.get("_note", "") + f" | compare_throughput failed: {e}"
        
    return out
