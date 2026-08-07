import mlx.core as mx


def matmul_metal(a: mx.array, b: mx.array, threadgroup_shape: tuple = (16, 16)) -> mx.array:
    raise NotImplementedError
