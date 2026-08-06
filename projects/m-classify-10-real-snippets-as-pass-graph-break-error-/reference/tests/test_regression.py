from compiletracer.classifier import classify_snippets
from compiletracer.dynamic_shapes import track_shape_compilations


def test_classifier_fullgraph_error():
    snippets = [
        {
            "id": "test_snip_1",
            "has_data_dependent_branch": True,
            "has_unsupported_side_effect": False,
            "has_invalid_shape_or_type": False
        }
    ]
    res = classify_snippets(snippets)
    assert len(res) == 1
    assert res[0]["default"] == "graph_break"
    assert res[0]["fullgraph"] == "error"


def test_dynamic_shape_overhead_bound():
    shapes = [(1, 16), (1, 32), (1, 64), (1, 128)]
    res_static = track_shape_compilations(shapes, dynamic=False, base_compile_cost=10.0)
    res_dynamic = track_shape_compilations(shapes, dynamic=True, base_compile_cost=10.0, dynamic_compile_cost=25.0)
    assert res_static["total_recompilations"] == 4
    assert res_dynamic["total_recompilations"] == 3
    assert res_dynamic["is_generalized"] is True
