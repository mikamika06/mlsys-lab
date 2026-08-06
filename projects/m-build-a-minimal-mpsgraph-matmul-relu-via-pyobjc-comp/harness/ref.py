import numpy as np


def generate_test_inputs(seed=42):
    rng = np.random.default_rng(seed)
    shapes = [((16, 32), (32, 64)), ((64, 128), (128, 256)), ((4, 8), (8, 4))]
    inputs = []
    for sa, sb in shapes:
        a = rng.normal(size=sa).astype(np.float32)
        b = rng.normal(size=sb).astype(np.float32)
        inputs.append((a, b))
    return inputs


TEST_OP_SEQUENCES = [
    ["matmul", "relu"],
    ["linear", "relu"],
    ["conv2d", "add", "gelu"],
    ["linear", "layernorm", "softmax"],
]

APPLE_MPSGRAPH_MAPPINGS = {
    "matmul": ["matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:"],
    "add": ["additionWithPrimaryTensor:secondaryTensor:name:"],
    "sub": ["subtractionWithPrimaryTensor:secondaryTensor:name:"],
    "mul": ["multiplicationWithPrimaryTensor:secondaryTensor:name:"],
    "relu": ["reLUWithTensor:name:"],
    "gelu": ["gELUWithTensor:name:"],
    "conv2d": ["convolution2DWithSourceTensor:weightsTensor:descriptor:name:"],
    "softmax": ["softMaxWithTensor:axis:name:"],
    "linear": [
        "matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:",
        "additionWithPrimaryTensor:secondaryTensor:name:",
    ],
    "layernorm": [
        "meanOfTensor:axes:name:",
        "varianceOfTensor:meanTensor:axes:name:",
        "layerNormalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:epsilon:name:",
    ],
}


def ref_map_recorded_sequence(op_sequence):
    primitives = []
    for op in op_sequence:
        key = op.lower().strip()
        primitives.extend(APPLE_MPSGRAPH_MAPPINGS[key])
    return primitives
