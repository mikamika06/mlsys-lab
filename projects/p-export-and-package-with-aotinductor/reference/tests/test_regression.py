import numpy as np
from exporter.custom_ops import CustomOpRegistry, custom_fused_op_impl
from exporter.export_pipeline import export_model_with_dynamic_shapes
from exporter.runtime_runner import StandaloneAOTRunner


def test_custom_op_shape_inference():
    registry = CustomOpRegistry()
    registry.register()
    out_shape = registry.meta_impl((2, 16, 32), (32, 64))
    assert out_shape == (2, 16, 64)


def test_export_pipeline_dynamic_bounds():
    dummy_model = object()
    sample_input = {"x": np.ones((4, 32, 16)), "weight": np.ones((16, 32))}
    dynamic_shapes = {"batch": (1, 64), "seq_len": (1, 128)}
    prog = export_model_with_dynamic_shapes(dummy_model, sample_input, dynamic_shapes)
    assert prog["status"] == "exported"


def test_runner_execution(tmp_path=None):
    import os
    import tempfile
    so_file = os.path.join(tempfile.gettempdir(), "test_model.so")
    with open(so_file, "wb") as f:
        f.write(b"\x7fELF_MOCK")

    runner = StandaloneAOTRunner(so_file)
    x = np.random.randn(2, 8, 16).astype(np.float32)
    w = np.random.randn(16, 32).astype(np.float32)
    out = runner.run({"x": x, "weight": w})

    expected = custom_fused_op_impl(x, w)
    assert np.allclose(out, expected, atol=1e-5)
