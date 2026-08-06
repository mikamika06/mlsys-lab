import numpy as np
import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    from edgepipe.export import fold_normalize_into_graph
    from edgepipe.layout import diagnose_and_fix_layout

    out = {"layout_correct": 0.0, "folded_match": 0.0}

    np.random.seed(42)
    raw_img = np.random.randint(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)

    got_layout = diagnose_and_fix_layout(
        raw_img,
        src_format="NHWC",
        dst_format="NCHW",
        src_order="BGR",
        dst_order="RGB",
    )
    ref_layout = ref.diagnose_and_fix_layout(
        raw_img,
        src_format="NHWC",
        dst_format="NCHW",
        src_order="BGR",
        dst_order="RGB",
    )

    if np.array_equal(got_layout, ref_layout):
        out["layout_correct"] = 1.0

    weights = np.random.randn(16, 3, 3, 3).astype(np.float32)
    bias = np.random.randn(16).astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    got_w, got_b = fold_normalize_into_graph(weights, bias, mean, std)
    ref_w, ref_b = ref.fold_normalize_into_graph(weights, bias, mean, std)

    if np.allclose(got_w, ref_w, rtol=1e-5, atol=1e-5) and np.allclose(
        got_b, ref_b, rtol=1e-5, atol=1e-5
    ):
        out["folded_match"] = 1.0

    return out
