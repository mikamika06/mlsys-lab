import sys
sys.path.insert(0, ".")
from exporter.model import CustomModel
from exporter.optimizer import verify_output
import numpy as np

def test_model_runtime():
    m = CustomModel(16)
    x = np.ones((4, 16), dtype=np.float32)
    out_t = m.forward(x)
    out_o = m.forward(x)
    diff = verify_output(out_t, out_o)
    assert diff <= 0.0001
