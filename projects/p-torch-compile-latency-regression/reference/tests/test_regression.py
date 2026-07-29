import sys

import torch
import torch._dynamo as dyn

sys.path.insert(0, ".")


def _example(batch=4):
    import service.preprocess as pre
    return pre.normalise(torch.randn(batch, 64))


def _model():
    from service.model import Classifier
    torch.manual_seed(0)
    return Classifier().eval()


def test_service_has_no_graph_breaks():
    dyn.reset()
    exp = dyn.explain(_model(), _example())
    dyn.reset()
    assert exp.graph_break_count == 0, f"{exp.graph_break_count} graph break(s) are back"


def test_fullgraph_compiles():
    m = _model()
    x = _example()
    dyn.reset()
    with torch.no_grad():
        out = torch.compile(m, fullgraph=True)(x)
        ref = m(x)
    dyn.reset()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_batch_schedule_does_not_explode_graphs():
    import service.preprocess as pre
    m = _model()
    dyn.reset()
    dyn.utils.counters.clear()
    c = torch.compile(m, dynamic=True)
    with torch.no_grad():
        for b in [1, 2, 3, 4, 5, 6, 7, 8]:
            c(pre.normalise(torch.randn(b, 64)))
    n = dyn.utils.counters.get("stats", {}).get("unique_graphs", 99)
    dyn.reset()
    assert n <= 3, f"{n} unique graphs across the batch schedule"
