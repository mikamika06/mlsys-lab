import numpy as np
import ref

def check(workdir):
    from quantpack import convert_awq_to_gptq
    from quantpack.layout import get_packed_shape, get_memory_strides

    awq_ok = 0
    layout_ok = 0
    total = len(ref.LAYOUT_CASES)

    for rows, cols, bits in ref.LAYOUT_CASES:
        try:
            dummy = np.zeros(rows * cols, dtype=np.int32)
            ref_conv = ref.convert_awq_to_gptq(dummy, bits, rows, cols)
            got_conv = convert_awq_to_gptq(dummy, bits, rows, cols)
            if np.array_equal(ref_conv, got_conv):
                awq_ok += 1
        except Exception:
            pass

        try:
            ref_shape = ref.get_packed_shape(rows, cols, bits)
            ref_strides = ref.get_memory_strides(ref_shape, 4)
            got_shape = get_packed_shape(rows, cols, bits)
            got_strides = get_memory_strides(got_shape, 4)
            if ref_shape == got_shape and ref_strides == got_strides:
                layout_ok += 1
        except Exception:
            pass

    awq_match = float(awq_ok) / float(total)
    layout_match = float(layout_ok) / float(total)

    return {
        "awq_conversion_match": awq_match,
        "layout_match": layout_match
    }
