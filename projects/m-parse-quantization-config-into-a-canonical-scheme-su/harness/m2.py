import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from quantlib.scheme import compute_packed_shape
    from quantlib.packer import pack_quantized_tensor

    out = {"shapes_matched": 0.0, "packing_matched": 0.0}

    shapes_ok = True
    test_cases = [
        ((128, 512), 4, 128, -1),
        ((64, 256), 8, 64, 0),
        ((100, 300), 4, 64, -1),
        ((128, 128), 2, -1, -1),
    ]

    for shape, bits, group_size, axis in test_cases:
        want = ref.compute_packed_shape(shape, bits, group_size, axis)
        got = compute_packed_shape(shape, bits, group_size, axis)
        if got != want:
            shapes_ok = False
            out["_note"] = f"Shape mismatch for {shape}: got {got}, want {want}"
            break

    if shapes_ok:
        out["shapes_matched"] = 1.0

    packing_ok = True
    np.random.seed(42)
    t1 = np.random.randint(0, 15, size=(16, 64), dtype=np.int32)
    want_packed = ref.pack_quantized_tensor(t1, num_bits=4, axis=-1)
    got_packed = pack_quantized_tensor(t1, num_bits=4, axis=-1)

    if not np.array_equal(want_packed, got_packed):
        packing_ok = False
        out["_note"] = "Packed tensor values did not match reference implementation"

    if packing_ok:
        out["packing_matched"] = 1.0

    return out
