import numpy as np
import ref


def check(workdir):
    from batchremap.mapping import create_batch_mapping, remap_batch_indices

    out = {"mappings_matched": 0.0, "remaps_matched": 0.0}

    valid_cases = ref.get_mapping_test_cases()
    valid_ok = True
    for tc in valid_cases:
        got = create_batch_mapping(tc["active"], tc["map"], tc["max_batch"])
        if not isinstance(got, np.ndarray) or not np.array_equal(got, tc["expected"]):
            valid_ok = False
            out["_note"] = f"create_batch_mapping failed on active={tc['active']}: got {got}, want {tc['expected']}"
            break

    invalid_cases = ref.get_invalid_mapping_test_cases()
    invalid_ok = True
    for tc in invalid_cases:
        try:
            create_batch_mapping(tc["active"], tc["map"], tc["max_batch"])
            invalid_ok = False
            out["_note"] = f"create_batch_mapping should have raised ValueError for invalid case {tc}"
            break
        except ValueError:
            pass
        except Exception as e:
            invalid_ok = False
            out["_note"] = f"expected ValueError but got {type(e).__name__}"
            break

    if valid_ok and invalid_ok:
        out["mappings_matched"] = 1.0

    remap_cases = ref.get_remap_test_cases()
    remap_ok = True
    for tc in remap_cases:
        got = remap_batch_indices(tc["old_idx"], tc["mask"])
        if not isinstance(got, np.ndarray) or not np.array_equal(got, tc["expected"]):
            remap_ok = False
            out["_note"] = f"remap_batch_indices failed: got {got}, want {tc['expected']}"
            break

    if remap_ok:
        out["remaps_matched"] = 1.0

    return out
