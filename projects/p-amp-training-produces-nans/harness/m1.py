import ref
from amp_fix.detector import locate_first_nan


def check(workdir):
    m = {"located_correctly": 0.0}
    graph = ref.get_mock_graph()
    first = locate_first_nan(graph)
    if first == "attention":
        m["located_correctly"] = 1.0
    return m
