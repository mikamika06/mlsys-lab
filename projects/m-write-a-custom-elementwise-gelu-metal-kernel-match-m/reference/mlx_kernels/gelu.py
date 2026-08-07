import mlx.core as mx


def gelu_metal(x: mx.array) -> mx.array:
    source = """
    uint elem = thread_position_in_grid.x;
    if (elem >= x_shape[0]) return;
    float val = (float)x[elem];
    float cube = val * val * val;
    float inner = 0.7978845608028654f * (val + 0.044715f * cube);
    float g = 0.5f * val * (1.0f + tanhf(inner));
    out[elem] = (T)g;
    """
    kernel = mx.fast.metal_kernel(
        name="gelu_kernel",
        input_names=["x"],
        output_names=["out"],
        source=source,
    )
    flat_x = mx.reshape(x, (-1,))
    out = kernel(
        inputs=[flat_x],
        template=[("T", x.dtype)],
        grid=(flat_x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[flat_x.shape],
        output_dtypes=[x.dtype],
    )[0]
    return mx.reshape(out, x.shape)
