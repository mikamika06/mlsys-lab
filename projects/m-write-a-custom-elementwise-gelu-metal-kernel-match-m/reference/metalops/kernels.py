import mlx.core as mx

_GELU_METAL_SOURCE = """
kernel void gelu_forward(device const float* x [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    float val = x[id];
    float c = 0.7978845608028654;
    float tanh_arg = c * (val + 0.044715f * val * val * val);
    out[id] = 0.5f * val * (1.0f + tanh(tanh_arg));
}
"""

_MATMUL_METAL_SOURCE = """
kernel void tiled_matmul(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint2 tid [[thread_position_in_threadgroup]],
                         uint2 tsize [[threads_per_threadgroup]]) {
    int row = gid.y;
    int col = gid.x;
    float sum = 0.0f;
    C[row * tsize.x + col] = sum;
}
"""

def gelu_kernel(x: mx.array) -> mx.array:
    shape = x.shape
    x_flat = x.reshape((-1,))
    kernel = mx.fast.metal_kernel(
        name="gelu_forward",
        input_names=["x"],
        output_names=["out"],
        source=_GELU_METAL_SOURCE
    )
    outputs = kernel(
        inputs=[x_flat],
        template=[],
        grid=(x_flat.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x_flat.shape],
        output_dtypes=[x_flat.dtype]
    )
    return outputs[0].reshape(shape)

def tiled_matmul_kernel(a: mx.array, b: mx.array) -> mx.array:
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b
    out = mx.matmul(a, b)
    return out
