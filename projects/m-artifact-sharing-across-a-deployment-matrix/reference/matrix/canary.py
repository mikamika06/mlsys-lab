import numpy as np


def validate_canary_artifact(candidate_trace: dict, reference_trace: dict, tolerance: float = 1e-4) -> bool:
    if candidate_trace.get("shape") != reference_trace.get("shape"):
        return False
    if candidate_trace.get("dtype") != reference_trace.get("dtype"):
        return False
    if candidate_trace.get("layout") != reference_trace.get("layout"):
        return False

    cand_data = np.array(candidate_trace["data"], dtype=np.float32)
    ref_data = np.array(reference_trace["data"], dtype=np.float32)

    if cand_data.shape != ref_data.shape:
        return False

    diff = np.abs(cand_data - ref_data)
    max_diff = float(np.max(diff))
    return max_diff <= tolerance
