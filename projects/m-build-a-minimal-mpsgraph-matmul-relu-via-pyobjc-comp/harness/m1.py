import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from mpsgraph.graph import MPSGraphMatMulReLU

    out = {"max_abs_err": 100.0, "graph_built": 0.0}
    test_inputs = ref.generate_test_inputs()

    max_err = 0.0
    valid_graphs = 0

    for a, b in test_inputs:
        model = MPSGraphMatMulReLU(a.shape, b.shape)
        res = model.compare_with_numpy(a, b)
        err = res["max_abs_err"]
        if err > max_err:
            max_err = err

        nodes = getattr(model, "nodes", [])
        has_matmul = any("matrixMultiplication" in n for n in nodes)
        has_relu = any("reLU" in n or "relu" in n for n in nodes)
        if has_matmul and has_relu:
            valid_graphs += 1

    out["max_abs_err"] = float(max_err)
    out["graph_built"] = 1.0 if valid_graphs == len(test_inputs) else 0.0
    return out
