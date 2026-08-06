import sys

sys.path.insert(0, ".")
from accounting.sizes import layer_bytes

def test_joint_int8_is_strictly_smaller_than_fp16_lut():
    shape = [32, 16, 3, 3]
    l4_fp16 = layer_bytes(shape, "lut4_channel_fp16")
    l4_joint = layer_bytes(shape, "lut4_joint_int8_channel")
    assert l4_joint < l4_fp16, "joint int8+LUT should have smaller LUT overhead than fp16"

    l8_fp16 = layer_bytes(shape, "lut8_channel_fp16")
    l8_joint = layer_bytes(shape, "lut8_joint_int8_channel")
    assert l8_joint < l8_fp16, "joint int8+LUT should have smaller LUT overhead than fp16"

def test_lut_is_worse_for_depthwise():
    shape = [64, 1, 1, 1]
    fp16 = layer_bytes(shape, "float16")
    l4_joint = layer_bytes(shape, "lut4_joint_int8_channel")
    assert l4_joint > fp16, "depthwise LUT should be worse than fp16"
