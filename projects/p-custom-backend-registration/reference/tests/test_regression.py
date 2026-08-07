from compiler.backend import CompilationGraph
from compiler.passes import CustomOptimizationPass, check_equivalence, apply_with_fallback

def test_optimization_equivalence():
    nodes = [{"id": 1, "type": "target_op"}, {"id": 2, "type": "other_op"}]
    g = CompilationGraph(nodes)
    p = CustomOptimizationPass()
    g_opt = p.run(g)
    assert check_equivalence(g, g_opt)

def test_fallback_behavior():
    nodes = [{"id": 1, "type": "target_op"}]
    g = CompilationGraph(nodes)
    class BrokenPass:
        def run(self, graph):
            raise RuntimeError("fail")
    res = apply_with_fallback(g, BrokenPass())
    assert res is not None
    assert len(res.get_nodes()) == 1
