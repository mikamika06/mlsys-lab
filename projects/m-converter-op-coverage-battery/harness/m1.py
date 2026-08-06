import ref


def check(workdir):
    from converter.coverage import audit_op_coverage

    out = {"graphs_audited": 0.0, "coverage_match": 0.0}

    ok = 0
    total = len(ref.TEST_GRAPHS)

    for i, graph in enumerate(ref.TEST_GRAPHS):
        runtime = ref.TEST_RUNTIMES[i % len(ref.TEST_RUNTIMES)]
        got = audit_op_coverage(graph, runtime)

        supported = set(runtime["supported_ops"])
        decomposable = set(runtime["decomposable_ops"].keys())

        expected_status = {}
        s_cnt, d_cnt, u_cnt = 0, 0, 0
        for n in graph["nodes"]:
            op = n["op_type"]
            nid = n["id"]
            if op in supported:
                expected_status[nid] = "NATIVE"
                s_cnt += 1
            elif op in decomposable:
                expected_status[nid] = "DECOMPOSABLE"
                d_cnt += 1
            else:
                expected_status[nid] = "UNSUPPORTED"
                u_cnt += 1

        tot_nodes = len(graph["nodes"])
        exp_ratio = (s_cnt + d_cnt) / tot_nodes if tot_nodes > 0 else 0.0

        if (got.get("node_status") == expected_status and
            got.get("supported_count") == s_cnt and
            got.get("decomposable_count") == d_cnt and
            got.get("unsupported_count") == u_cnt and
            abs(got.get("coverage_ratio", 0.0) - exp_ratio) < 1e-6):
            ok += 1
        else:
            out["_note"] = f"Mismatch on graph {i}: got {got}"

    if ok == total:
        out["graphs_audited"] = 1.0
        out["coverage_match"] = 1.0

    return out
