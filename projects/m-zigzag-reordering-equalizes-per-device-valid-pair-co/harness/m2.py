import ref


def check(workdir):
    from ringbal import assign, metrics

    out = {"metrics_match": 0.0, "imbalance_ordered": 0.0}

    test_assignment = [[0, 1], [2, 3]]
    got = metrics.workload_imbalance(test_assignment)
    want = ref.workload_imbalance(test_assignment)

    for k in ["mean", "max", "rel_err"]:
        if abs(got.get(k, -1) - want[k]) > 1e-6:
            out["_note"] = f"metrics mismatch on {test_assignment}: got {got}, want {want}"
            return out

    out["metrics_match"] = 1.0

    ordered = True
    for N, D in ref.CONFIGS:
        naive_err = metrics.workload_imbalance(assign.naive_assignment(N, D))["rel_err"]
        striped_err = metrics.workload_imbalance(assign.striped_assignment(N, D))["rel_err"]
        zigzag_err = metrics.workload_imbalance(assign.zigzag_assignment(N, D))["rel_err"]

        if not (naive_err > striped_err > zigzag_err) or abs(zigzag_err) > 1e-7:
            ordered = False
            out["_note"] = f"order property failed N={N}, D={D}: naive={naive_err}, striped={striped_err}, zigzag={zigzag_err}"
            break

    if ordered:
        out["imbalance_ordered"] = 1.0

    return out
