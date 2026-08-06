import ref
import copy


def check(workdir):
    from optimizer.subgraph import extract_subgraph

    model = ref.build_test_model()
    got = extract_subgraph(copy.deepcopy(model), ["tensor_a"], ["tensor_b"])

    out = {"subgraph_matched": 0.0}
    if got is not None and len(got.graph.node) == 1 and got.graph.node[0].op_type == "Relu":
        out["subgraph_matched"] = 1.0
    else:
        out["_note"] = "subgraph extraction failed or returned incorrect node count"
    return out
