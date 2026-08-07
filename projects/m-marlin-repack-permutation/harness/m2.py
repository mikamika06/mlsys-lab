import os
import sys
import numpy as np
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from repack.pack import pack_int4_standard, unpack_int4_standard, repack_gptq_to_marlin
    except Exception as e:
        return {"byte_exact_fraction": 0.0, "_note": f"Import error: {e}"}

    total = float(len(ref.CONFIGS))
    passed = 0.0

    for K, N in ref.CONFIGS:
        W = ref.generate_test_matrix(K, N)

        want_std_pack = ref.pack_int4_standard(W)
        try:
            got_std_pack = pack_int4_standard(W)
            if not np.array_equal(got_std_pack, want_std_pack):
                return {"byte_exact_fraction": passed / total, "_note": f"mismatch in pack_int4_standard for ({K}, {N})"}
        except Exception as e:
            return {"byte_exact_fraction": passed / total, "_note": f"pack_int4_standard failed: {e}"}

        try:
            got_unpacked = unpack_int4_standard(got_std_pack, K, N)
            if not np.array_equal(got_unpacked, W):
                return {"byte_exact_fraction": passed / total, "_note": f"unpack_int4_standard failed to restore original W for ({K}, {N})"}
        except Exception as e:
            return {"byte_exact_fraction": passed / total, "_note": f"unpack_int4_standard failed: {e}"}

        want_marlin = ref.repack_gptq_to_marlin(want_std_pack, K, N)
        try:
            got_marlin = repack_gptq_to_marlin(got_std_pack, K, N)
            if not np.array_equal(got_marlin, want_marlin):
                return {"byte_exact_fraction": passed / total, "_note": f"mismatch in repack_gptq_to_marlin for ({K}, {N})"}
        except Exception as e:
            return {"byte_exact_fraction": passed / total, "_note": f"repack_gptq_to_marlin failed: {e}"}

        passed += 1.0

    return {"byte_exact_fraction": passed / total}
