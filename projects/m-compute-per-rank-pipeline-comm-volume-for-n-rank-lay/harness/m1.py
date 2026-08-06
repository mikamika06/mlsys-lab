import os
import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from mlxdist.pipeline import compute_pipeline_comm_volume
    except ImportError:
        return {"volume_matches": 0.0, "_note": "Failed to import mlxdist.pipeline"}

    test_cases = [
        ([0, 0, 1, 1, 2, 3], [(1, 64, 512)] * 5, 2),
        ([0, 1, 2, 3, 0, 1], [(2, 128, 1024)] * 5, 4),
        ([0, 0, 0, 0], [(1, 32, 256)] * 3, 2),
    ]

    ok = True
    for assignments, shapes, dtype_b in test_cases:
        want = ref.compute_pipeline_comm_volume(assignments, shapes, dtype_b)
        try:
            got = compute_pipeline_comm_volume(assignments, shapes, dtype_b)
        except Exception as e:
            return {"volume_matches": 0.0, "_note": f"Execution error: {e}"}

        if got != want:
            ok = False
            break

    return {"volume_matches": 1.0 if ok else 0.0}
