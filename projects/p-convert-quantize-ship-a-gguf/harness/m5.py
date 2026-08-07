import ref


def check(workdir):
    m = {"speedup_monotonic": 0.0}
    import sys
    sys.path.insert(0, workdir)
    try:
        import gguf_pipe.eval as ev
        sp_fp16 = ev.measure_throughput("model_FP16.gguf")
        sp_q8 = ev.measure_throughput("model_Q8_0.gguf")
        sp_q4 = ev.measure_throughput("model_Q4_K_M.gguf")
        if sp_q4 > sp_q8 and sp_q8 > sp_fp16:
            m["speedup_monotonic"] = 1.0
    except Exception:
        pass
    return m
