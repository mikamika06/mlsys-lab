import torch._dynamo as dyn


def inventory(model, example):
    dyn.reset()
    exp = dyn.explain(model, example)
    return {"graph_count": exp.graph_count,
            "graph_break_count": exp.graph_break_count,
            "op_count": exp.op_count}
