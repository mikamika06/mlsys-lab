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


def map_op_to_primitives(op_name: str) -> list[str]:
    """Maps a recorded framework op to Apple MPSGraph primitive name(s)."""
    key = op_name.lower().strip()
    if key not in APPLE_MPSGRAPH_MAPPINGS:
        raise KeyError(f"Unsupported recorded op: {op_name}")
    return list(APPLE_MPSGRAPH_MAPPINGS[key])


def map_recorded_sequence(op_sequence: list[str]) -> list[str]:
    """Maps a sequence of recorded ops into a flattened list of MPSGraph primitives."""
    primitives = []
    for op in op_sequence:
        primitives.extend(map_op_to_primitives(op))
    return primitives
