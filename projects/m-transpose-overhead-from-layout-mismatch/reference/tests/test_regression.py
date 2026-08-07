import numpy as np
from edgeio.pipeline import preprocess_app_side, preprocess_in_graph_node


def test_pipeline_equivalence_and_layout_invariant():
    np.random.seed(42)
    raw = np.random.randint(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    app_res = preprocess_app_side(raw, mean, std)
    graph_res = preprocess_in_graph_node(raw, mean, std)

    assert app_res.shape == (2, 3, 64, 64), f"Unexpected shape {app_res.shape}"
    assert app_res.flags["C_CONTIGUOUS"], "app_side output must be contiguous"
    assert graph_res.flags["C_CONTIGUOUS"], "in_graph output must be contiguous"

    max_err = np.max(np.abs(app_res - graph_res))
    assert max_err < 1e-5, f"Preprocessing outputs mismatch, max diff: {max_err}"
