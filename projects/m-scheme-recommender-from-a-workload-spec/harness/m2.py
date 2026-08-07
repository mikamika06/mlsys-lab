import ref


def check(workdir):
    try:
        import sys
        if workdir not in sys.path:
            sys.path.insert(0, workdir)
        from quantrec.crossover import crossover_batch_size
    except ImportError:
        return {"crossover_rel_err": 1.0}

    max_err = 0.0
    for w in ref.TEST_WORKLOADS:
        bw = w["bandwidth_gbps"]
        tf = w["tflops_w16"]
        want = ref.crossover_batch_size(bw, tf)
        try:
            got = float(crossover_batch_size(bw, tf))
        except Exception:
            return {"crossover_rel_err": 1.0}
        if want > 0:
            err = abs(got - want) / want
        else:
            err = abs(got - want)
        if err > max_err:
            max_err = err
    return {"crossover_rel_err": float(max_err)}
