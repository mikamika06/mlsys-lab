import os
import tempfile
import ref


def check(workdir):
    m = {"size_ratio_valid": 0.0}
    import sys
    sys.path.insert(0, workdir)
    try:
        import gguf_pipe.quantize as quant
        s_fp16 = quant.get_quantized_size("model.gguf", "FP16")
        s_q4 = quant.get_quantized_size("model.gguf", "Q4_K_M")
        if s_q4 < s_fp16:
            m["size_ratio_valid"] = 1.0
    except Exception:
        pass
    return m
