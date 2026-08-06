from autotune_metrics.analyzer import locate_cuda_graph_recapture


def test_locate_cuda_graph_recapture_valid():
    events = [
        {"name": "kernel_exec", "ph": "X"},
        {"name": "cuda_graph_recapture", "ph": "R"},
        {"name": "kernel_exec", "ph": "X"}
    ]
    res = locate_cuda_graph_recapture(events)
    assert len(res) == 1
    assert res[0] == 1


def test_locate_cuda_graph_recapture_empty():
    events = [{"name": "foo", "ph": "X"}]
    res = locate_cuda_graph_recapture(events)
    assert len(res) == 0
