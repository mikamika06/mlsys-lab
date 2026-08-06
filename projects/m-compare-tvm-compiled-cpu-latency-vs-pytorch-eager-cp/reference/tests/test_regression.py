from tvm_bench.frontend import capture_frontend_error


class MockModel:

    def __init__(self, ops):
        self.ops = ops

    def __call__(self, x):
        return x


def test_unsupported_op_detection():
    """Verify that unsupported ops trigger UnsupportedOpError."""
    supported = {"conv2d", "relu", "add"}
    bad_model = MockModel(["conv2d", "custom_deform_conv"])
    captured, msg = capture_frontend_error(bad_model, [1.0], supported)
    assert captured, f"Failed to capture unsupported op: {msg}"
    assert "custom_deform_conv" in msg
