import ref
import numpy as np

def check(workdir):
    from quant.blockwise import quantize_blockwise, dequantize_blockwise
    tensor = ref.generate_tensor()
    block_size = 64
    q_want, s_want, shape_want = quantize_blockwise(tensor, block_size)
    dq_want = dequantize_blockwise(q_want, s_want, block_size, shape_want)
    try:
        q_got, s_got, shape_got = quantize_blockwise(tensor, block_size)
        dq_got = dequantize_blockwise(q_got, s_got, block_size, shape_got)
    except Exception as e:
        return {"quant_match": 0.0, "_note": f"raised {type(e).__name__}"}
    match = np.allclose(dq_got, dq_want, atol=1e-5)
    return {"quant_match": 1.0 if match else 0.0}
