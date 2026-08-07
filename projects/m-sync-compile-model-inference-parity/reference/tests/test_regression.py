import numpy as np
from ovruntime.core import Core


def test_wrong_shape_infer_raises_value_error():
    core = Core()
    config = {
        "input_shape": (4, 16),
        "layers": [
            {
                "weights": np.ones((16, 8), dtype=np.float32),
                "bias": np.zeros(8, dtype=np.float32),
            }
        ],
    }
    compiled = core.compile_model(config)
    req = compiled.create_infer_request()
    wrong_input = np.ones((2, 16), dtype=np.float32)

    raised = False
    try:
        req.infer(wrong_input)
    except ValueError:
        raised = True

    assert raised, "Expected ValueError on wrong input shape for infer()"


def test_wrong_shape_async_raises_value_error():
    core = Core()
    config = {
        "input_shape": (4, 16),
        "layers": [
            {
                "weights": np.ones((16, 8), dtype=np.float32),
                "bias": np.zeros(8, dtype=np.float32),
            }
        ],
    }
    compiled = core.compile_model(config)
    req = compiled.create_infer_request()
    wrong_input = np.ones((4, 8), dtype=np.float32)

    raised = False
    try:
        req.start_async(wrong_input)
    except ValueError:
        raised = True

    assert raised, "Expected ValueError on wrong input shape for start_async()"


def test_sync_async_parity():
    core = Core()
    config = {
        "input_shape": (4, 16),
        "layers": [
            {
                "weights": np.ones((16, 8), dtype=np.float32),
                "bias": np.zeros(8, dtype=np.float32),
            }
        ],
    }
    compiled = core.compile_model(config)
    inp = np.ones((4, 16), dtype=np.float32)

    sync_out = compiled(inp)

    req = compiled.create_infer_request()
    req.start_async(inp)
    async_out = req.wait()

    np.testing.assert_allclose(sync_out, async_out)
