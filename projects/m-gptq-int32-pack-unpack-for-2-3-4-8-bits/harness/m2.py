import ref
import numpy as np

def check(workdir):
    from quantpack.convert import convert_awq_to_gptq
    from quantpack.shapes import packed_shape_and_stride
    test_cases = ref.get_test_matrices()
    convert_ok = 0
    stride_ok = 0
    total = len(test_cases)
    for mat, bits in test_cases:
        packed_ref = ref.pack_weights(mat, bits)
        conv_ref = ref.convert_awq_to_gptq(packed_ref, bits, mat.shape)
        shape_ref, stride_ref = ref.packed_shape_and_stride(mat.shape, bits)
        try:
            conv_got = convert_awq_to_gptq(packed_ref, bits, mat.shape)
            if np.array_equal(conv_ref, conv_got):
                convert_ok += 1
        except Exception:
            pass
        try:
            shape_got, stride_got = packed_shape_and_stride(mat.shape, bits)
            if shape_ref == shape_got and stride_ref == stride_got:
                stride_ok += 1
        except Exception:
            pass
    c_match = 1.0 if convert_ok == total else 0.0
    s_match = 1.0 if stride_ok == total else 0.0
    return {"convert_match": c_match, "stride_match": s_match}
