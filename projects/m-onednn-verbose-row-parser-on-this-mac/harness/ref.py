LOGS = [
    "oneDNN_verbose, info, primitive, exec, cpu, convolution, jitted:avx512, forward_inference, data:f32, weights:f32, 3x3, 1.25",
    "oneDNN_verbose, info, primitive, exec, cpu, inner_product, ref:any, forward_inference, data:f32, weights:f32, 0.75",
    "oneDNN_verbose, info, primitive, exec, cpu, eltwise, jitted:avx2, forward_inference, data:f32, 0.50",
    "oneDNN_verbose, info, primitive, exec, cpu, pooling, ref:any, forward_inference, data:f32, 0.25",
    "oneDNN_verbose, info, primitive, exec, cpu, batch_normalization, jitted:amx, forward_inference, data:f32, 1.50"
]

def get_reference_rows():
    from dnnlog.parser import parse_row
    return [parse_row(l) for l in LOGS]
