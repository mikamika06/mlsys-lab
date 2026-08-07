import numpy as np
from edgeio.pipeline import preprocess_app_side, preprocess_in_graph_node


def check_roundtrip_equivalence(raw_frames: np.ndarray, mean: list, std: list, rtol: float = 1e-5, atol: float = 1e-5) -> dict:
    """Checks floating-point equivalence between app-side and in-graph preprocessing."""
    app_out = preprocess_app_side(raw_frames, mean, std)
    graph_out = preprocess_in_graph_node(raw_frames, mean, std)

    diff = np.abs(app_out - graph_out)
    max_diff = float(np.max(diff))
    is_equivalent = bool(np.allclose(app_out, graph_out, rtol=rtol, atol=atol))

    return {
        "equivalent": is_equivalent,
        "max_absolute_error": max_diff,
        "shapes_match": bool(app_out.shape == graph_out.shape),
    }
