import numpy as np
import ref


def check(workdir):
    from imatrix_analysis.search import allocate_bits

    out = {"search_valid": 0.0, "target_met": 0.0, "beats_uniform": 0.0}
    tensors_data = ref.generate_fixtures(seed=456)
    allowed_bpws = [2.0, 3.0, 4.0, 6.0, 8.0]

    target_wmse = 0.08

    try:
        res = allocate_bits(tensors_data, target_wmse, allowed_bpws)
        if not isinstance(res, dict) or "allocations" not in res or "achieved_wmse" not in res or "avg_bpw" not in res:
            out["_note"] = "Return dictionary missing required keys"
            return out

        allocations = res["allocations"]
        achieved_wmse = res["achieved_wmse"]
        avg_bpw = res["avg_bpw"]

        if len(allocations) != len(tensors_data):
            out["_note"] = f"Expected {len(tensors_data)} allocations, got {len(allocations)}"
            return out

        if not all(b in allowed_bpws for b in allocations):
            out["_note"] = "Allocated BPW values contain unallowed numbers"
            return out

        out["search_valid"] = 1.0

        if achieved_wmse <= target_wmse + 1e-6:
            out["target_met"] = 1.0
        else:
            out["_note"] = f"Achieved WMSE {achieved_wmse} exceeded target {target_wmse}"

        weights = np.array([t["weight"] for t in tensors_data], dtype=np.float64)
        weights /= np.sum(weights)

        uniform_bpw = None
        for b in allowed_bpws:
            u_wmse = 0.0
            for t in tensors_data:
                err = np.interp(b, t["bpws"], t["imatrix_errors"])
                u_wmse += err
            u_wmse /= len(tensors_data)
            if u_wmse <= target_wmse:
                uniform_bpw = b
                break

        if uniform_bpw is None:
            out["beats_uniform"] = 1.0
        elif avg_bpw < uniform_bpw - 1e-4:
            out["beats_uniform"] = 1.0
        else:
            out["_note"] = f"Search average BPW ({avg_bpw}) did not beat uniform BPW ({uniform_bpw})"

    except Exception as e:  # noqa: BLE001
        out["_note"] = f"Error executing allocate_bits: {type(e).__name__}: {e}"

    return out
