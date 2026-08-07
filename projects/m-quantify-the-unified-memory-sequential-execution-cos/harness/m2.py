import ref


def check(workdir):
    from seqcost.cost import execution_cost_ratio

    max_err = 0.0
    for cfg in ref.CONFIGS:
        want = ref.execution_cost_ratio(cfg, 2048, 4, 150.0)
        got = execution_cost_ratio(cfg, 2048, 4, 150.0)
        if want > 0:
            err = abs(want - got) / want
            max_err = max(max_err, err)

    return {"rel_err": float(max_err)}
