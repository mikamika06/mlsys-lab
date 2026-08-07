import sys

sys.path.insert(0, ".")
from tflite_pipe.flex_opt import strip_flex_ops

def test_no_flex_ops():
    ops = ["ADD", "SELECT_TF_OPS:ResizeBilinear", "MUL"]
    cleaned = strip_flex_ops(ops)
    assert not any("SELECT_TF_OPS" in o for o in cleaned), "Flex/SELECT_TF_OPS dependency remains"
