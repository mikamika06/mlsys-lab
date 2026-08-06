import ref


def check(workdir):
    from ortopt.optimizer import optimize_graph

    out = {"max_abs_err": 1.0}
    dummy_graph = "decoder_layer_0"
    baseline = [1.0, 2.0, 3.0]
    res_trans = optimize_graph(dummy_graph, "transformers")
    err = ref.evaluate_error(res_trans, baseline)
    out["max_abs_err"] = float(err)
    return out
