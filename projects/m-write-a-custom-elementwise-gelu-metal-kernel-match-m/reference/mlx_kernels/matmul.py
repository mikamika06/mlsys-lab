import mlx.core as mx


def matmul_metal(a: mx.array, b: mx.array, threadgroup_shape: tuple = (16, 16)) -> mx.array:
    M, K = a.shape
    K_b, N = b.shape
    if K != K_b:
        raise ValueError("Inner dimensions must match")

    source = """
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += (float)a[row * K + k] * (float)b[k * N + col];
    }
    out[row * N + col] = (T)acc;
    """

    kernel = mx.fast.metal_kernel(
        name="tiled_matmul",
        input_names=["a", "b"],
        output_names=["out"],
        source=source,
        header="""
        #define M params[0]
        #define N params[1]
        #define K params[2]
        """,
    )

    grid_x = (N + threadgroup_shape[0] - 1) // threadgroup_shape[0] * threadgroup_shape[0]
    grid_y = (M + threadgroup_shape[1] - 1) // threadgroup_shape[1] * threadgroup_shape[1]

    out = kernel(
        inputs=[a, b],
        template=[("T", a.dtype)],
        grid=(grid_x, grid_y, 1),
        threadgroup=(threadgroup_shape[0], threadgroup_shape[1], 1),
        output_shapes=[(M, N)],
        output_dtypes=[a.dtype],
        init_value=0,
    )[0]
    return out
